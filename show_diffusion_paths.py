#!/usr/bin/env python3

import isaacgym

import os
import sys
import numpy as np
import torch
# import csv
import json
import einops
import pickle
cwd = os.getcwd()
sys.path.append(cwd)
import diffuser.utils as utils
from panda_env import env_wrapper, INIT_JOINT_CONFIG
from planner.planner_utils import get_free_motion_gen
from scripts.panda_primitives import BodyConf
from functools import partial

from embed_env import load_model as load_embed
from diffusion import load_model as load_diffusion_model
from diffuser.utils.training import cycle
from diffuser.utils.arrays import to_device, batch_to_device
# from key_config import load_key_config, key_config_classify
from key_config_v2 import (load_key_config,
                           key_config_classify,
                           vectorize_primitives)
import time

from external.pybullet_planning.pybullet_tools.utils import wait_if_gui, get_sample_fn, get_joint_positions, \
    check_initial_end, get_distance_fn, get_extend_fn, get_collision_fn, MAX_DISTANCE
from external.pybullet_planning.motion.motion_planners.meta import check_direct
import multiprocessing
from scripts.panda_primitives import BodyPose, BodyConf, get_grasp_gen, get_ik_fn

SAVE_PATH = "./save"
GOAL_DISTANCE = 0.01
DEVICE: str = 'cuda:0'

from scripts.parallel_test import infer_trajectory

import trimesh
import open3d as o3d
from yourdfpy import URDF

from pkm.env.robot.franka_util import franka_fk
from pkm.util.vis.win_o3d import AutoWindow


def scene_from_obs_info(obs_info):
    meshes = []
    for o in obs_info:
        if o['shape'] == 'box':
            # halfExtents -> extents
            dims = np.asarray([2 * x for x in o['dims']])
            m = o3d.geometry.TriangleMesh.create_box(*dims).translate(
                -0.5 * dims + o['base_pos'])
            # m = trimesh.creation.box([2 * x for x in o['dims']])
            # m = m.apply_translation(o['base_pos'])
        else:
            m = o3d.geometry.TriangleMesh.create_sphere(
                o['dims']).translate(
                o['base_pos'])
        meshes.append(m)
        meshes.append(o3d.geometry.LineSet.create_from_triangle_mesh(
            m))
    return meshes


def normalize(x, normalizer, key):
    maxs = normalizer.normalizers[key].maxs
    mins = normalizer.normalizers[key].mins
    return 2 * ((x - mins) / (maxs - mins)) - 1


def unnormalize(x, normalizer, key, device=DEVICE):
    maxs = normalizer.normalizers[key].maxs
    mins = normalizer.normalizers[key].mins
    maxs = torch.tensor(maxs, dtype=torch.float, device=device)
    mins = torch.tensor(mins, dtype=torch.float, device=device)
    return (maxs - mins) * ((x + 1) / 2.0) + mins


def benchmark_env(env_type):

    if env_type == 0:
        obs_info = [
            {'shape': 'box',
             'mass': 0,
             'base_pos': [-0.3, 0.0, 0.5],
             'base_quat': [0, 0, 0, 1],
             'dims': [0.1, 0.5, 0.5]},
            {'shape': 'box',
             'mass': 0,
             'base_pos': [0.0, -0.3, 0.5],
             'base_quat': [0, 0, 0, 1],
             'dims': [0.5, 0.1, 0.5]},
            {'shape': 'box',
             'mass': 0,
             'base_pos': [0.0, 0.0, 0.9],
             'base_quat': [0, 0, 0, 1],
             'dims': [0.5, 0.5, 0.1]},
        ]
    elif env_type == 1:
        obs_info = [
            {'shape': 'box',
             'mass': 0,
             'base_pos': [-0.3, 0.0, 0.5],
             'base_quat': [0, 0, 0, 1],
             'dims': [0.1, 0.5, 0.5]},
            {'shape': 'box',
             'mass': 0,
             'base_pos': [0.0, -0.3, 0.5],
             'base_quat': [0, 0, 0, 1],
             'dims': [0.5, 0.1, 0.5]},
            {'shape': 'box',
             'mass': 0,
             'base_pos': [0.0, 0.0, 0.9],
             'base_quat': [0, 0, 0, 1],
             'dims': [0.5, 0.5, 0.1]},
            {'shape': 'sphere',
             'mass': 0,
             'base_pos': [0.2, 0.4, 0.55],
             'base_quat': [0, 0, 0, 1],
             'dims': 0.15
             },
            {'shape': 'sphere',
             'mass': 0,
             'base_pos': [0.2, 0.4, 0.15],
             'base_quat': [0, 0, 0, 1],
             'dims': 0.1
             },
        ]
    elif env_type == 2:
        obs_info = [
            {'shape': 'box',
             'mass': 0,
             'base_pos': [-0.3, 0.0, 0.5],
             'base_quat': [0, 0, 0, 1],
             'dims': [0.1, 0.5, 0.5]},
            {'shape': 'box',
             'mass': 0,
             'base_pos': [0.0, -0.3, 0.5],
             'base_quat': [0, 0, 0, 1],
             'dims': [0.5, 0.1, 0.5]},
            {'shape': 'box',
             'mass': 0,
             'base_pos': [0.0, 0.0, 0.9],
             'base_quat': [0, 0, 0, 1],
             'dims': [0.5, 0.5, 0.1]},
            {'shape': 'sphere',
             'mass': 0,
             # 'base_pos': [0.3, 0.4, 0.45],
             # 'base_pos': [0.3, 0.35, 0.45],
             'base_pos': [0.3, 0.35, 0.5],
             'base_quat': [0, 0, 0, 1],
             'dims': 0.2
             },
        ]
    else:
        obs_info = []

    return obs_info


def find_collision_free_intervals(binary_list, goal_idx):

    intervals = []
    start_index = None

    for i, value in enumerate(binary_list):
        # If i is the goal index, stop looking for intervals
        if i > goal_idx:
            break
        # Check if the current value is a 0 (collision-free)
        if value == 0:
            # If this is the start of a new interval, set start_index
            if start_index is None:
                start_index = i
            # If this is the element before the goal index, add the interval to
            # the list
            if i == goal_idx:
                intervals.append((start_index, i))
        else:
            # If we have an ongoing interval, save it
            if start_index is not None:
                intervals.append((start_index, i - 1))
                start_index = None

    final_intervals = []
    for idx, interval in enumerate(intervals):
        if idx < len(intervals) - 1:
            final_intervals.append(
                (interval[0], interval[1], intervals[idx + 1][0]))
        else:
            if interval[1] > goal_idx:
                final_intervals.append((interval[0], interval[1], goal_idx))
            else:
                final_intervals.append((interval[0], interval[1], None))

    return final_intervals


def main(kwargs):
    record_path = kwargs['record_path']
    dataset_config = utils.load_config(
        kwargs['diffusion_path'],
        'dataset_config.pkl')
    dataset = dataset_config()

    if test_kwargs['key_config']:
        key_configs = load_key_config(
            os.path.join(
                kwargs['diffusion_path'],
                'key_config/config.json'),
            **kwargs)

    _, ema_model, _ = load_diffusion_model(
        kwargs['diffusion_path'],
        kwargs['diffusion_epoch'],
        add_cond_embed=False,
        # w=0.0,
        w=-1.0,
        **kwargs
    )
    model = ema_model

    embed_model = load_embed(
        load_path='./save/classifier-free/embed_env_dmax200_clip/',
        epoch='latest',
        device=DEVICE)
    embed_model.eval()

    os.makedirs(name=kwargs['save_path'], exist_ok=True)

    env_type = kwargs['env_type']
    batch_size = 32
    out_paths = []
    out_samples = []
    titles = []
    time_ra = []
    collision_ra = []
    final_paths = []
    collision_infos = {}

    MODEL_PATH = os.path.join(os.getcwd(), 'models')
    PANDA_URDF = os.path.join(
        MODEL_PATH,
        'franka_description/robots/panda_arm_kinematics.urdf')

    panda = URDF.load(PANDA_URDF,
                      build_collision_scene_graph=True,
                      load_meshes=False,
                      load_collision_meshes=False)

    for idx in range(1, 25):
        # == Generate problem ==
        if kwargs['testset_path'] is not None:
            testset_path = kwargs['testset_path']
            if not os.path.exists(os.path.join(
                    testset_path, 'test_{}.pkl'.format(idx))):
                continue
            with open(os.path.join(testset_path, 'test_{}.pkl'.format(idx)), 'rb') as f:
                test_info = pickle.load(f)
            obs_info = test_info['obs_info']
            goal_joint_config = test_info['goal_joint_config']
            init_joint_config = test_info['init_joint_config']
        else:
            raise ValueError('stop')

        init_time = time.time()

        # == Compute Env Class ==
        if test_kwargs['diffuser']:
            env_class = 0
        elif test_kwargs['key_config']:
            USE_EMBED = False
            USE_ANALYTIC = True
            if not USE_ANALYTIC:
                env_class = key_config_classify(obs_info, key_configs)
            if USE_EMBED:
                with torch.inference_mode():
                    nrm = partial(normalize,
                                  normalizer=dataset.normalizer,
                                  key='observations')
                    tot = partial(torch.as_tensor,
                                  dtype=torch.float,
                                  device=DEVICE)
                    env_class = embed_model(tot(nrm(INIT_JOINT_CONFIG)),
                                            tot(nrm(goal_joint_config)),
                                            {'class': tot(env_class)})
            if USE_ANALYTIC:
                env_class = vectorize_primitives(obs_info, max_len=45)
        else:
            env_class = env_type

        # == Infer Trajectory ==
        print("Class: {}, rollout: {}".format(env_class, idx))
        torch.manual_seed(idx)
        np.random.seed(idx)

        diffusion_chain = infer_trajectory(model,
                                           dataset=dataset,
                                           horizon=kwargs['horizon'],
                                           init_config=INIT_JOINT_CONFIG,
                                           goal_config=goal_joint_config,
                                           env_class=env_class,
                                           batch_size=batch_size)
        print(diffusion_chain.shape)
        ee_poses = (franka_fk(
            torch.as_tensor(diffusion_chain[..., :7]))
            [..., :3, 3]).detach().cpu().numpy()
        print('ee_poses', ee_poses.shape)

        # == Create Scene ==
        geoms = scene_from_obs_info(obs_info)

        win = AutoWindow()
        vis = win.vis

        for i_g, g in enumerate(geoms):
            vis.add_geometry(F'obs-{i_g:02d}', g,
                             color=(1, 1, 1, 1))

        colors = np.random.uniform(0, 1, size=(batch_size, 4))
        colors[:, 3] = 1

        # == Create Path ==
        for T in range(0, len(diffusion_chain), 8):
            paths = diffusion_chain[T]  # B H D
            # ee_paths = []
            # for path in paths:
            #    ee_path = []
            #    for q in path:
            #        panda.update_cfg(q[:7])
            #        T_l = panda.get_transform('panda_leftfinger')
            #        T_r = panda.get_transform('panda_rightfinger')
            #        ee_path.append(0.5 * (T_l[:3, 3] + T_r[:3, 3]))
            #    ee_paths.append(ee_path)
            # ee_paths = np.asarray(ee_paths)
            ee_paths = ee_poses[T]

            geom_paths = []
            for ee_path in ee_paths:
                n: int = len(ee_path)
                points = o3d.utility.Vector3dVector(ee_path)
                indices = o3d.utility.Vector2iVector(
                    np.stack([np.arange(0, n - 1),
                              np.arange(1, n)],
                             axis=-1).astype(np.int32))
                path = o3d.geometry.LineSet(points=points, lines=indices)
                geom_paths.append(path)
            # o3d.visualization.draw(geoms + geom_paths)

            for i_p, g in enumerate(geom_paths):
                vis.add_geometry(F'path-{i_p:02d}', g,
                                 color=colors[i_p])

            win.wait()


if __name__ == '__main__':
    torch.cuda.set_device(DEVICE)

    test_kwargs = {
        'render': 'utils.PandaRenderer',
        'horizon': 1000,
        'enc_type': 'conv',
        'num_classes': 3,
        'num_steps': 256,
        'save_path': './tests/classifier-free/test/main_4_plain',
        'enc_type': 'conv',

        # 'diffusion_path': './save/classifier-free/diffusion2',
        # 'diffusion_path': './save/classifier-free/diffusion',
        # 'diffusion_path': './save/classifier-free/diffusion-embed-v2',
        # 'diffusion_path': './save/classifier-free/diffusion-embed-v2',
        # 'diffusion_path': './save/classifier-free/diffusion2',
        # 'diffusion_path': './save/classifier-free/diffusion-analytic',
        # 'diffusion_path': './save/classifier-free/diffusion-analytic-catembed',
        'diffusion_path': './save/classifier-free/diffusion-analytic-catembed-w0',


        # vanilla
        # ./save/classifier-free/diffusion2
        # embedding version
        # ./save/classifier-free/diffusion-embed-v2
        # analytic version
        # ./save/classifier-free/diffusion-analytic

        'diffusion_epoch': 'latest',
        'device': DEVICE,
        'key_config': True,
        'diffuser': False,
        'local_planning': False,
        'local_algo': 'rrt_connect',
        'local_num_samples': 80,
        'local_max_time': 20,
        'env_type': 2,

        'testset_path': os.path.join(cwd, 'testsets/main_4'),
        'record_path': os.path.join(cwd, 'record/test/main_4_plain')
    }

    main(test_kwargs)
