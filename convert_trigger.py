##!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import yaml
import json
# import csv
import pickle
sys.path.append('/input/diffuser_pybullet_simulation')
os.chdir('/input/diffuser_pybullet_simulation')
cwd = os.getcwd()
import diffuser.utils as utils


SAVE_PATH = "./save"
GOAL_DISTANCE = 0.01
DEVICE: str = 'cuda:0'


# import open3d as o3d
import trimesh
from yourdfpy import URDF
from pathlib import Path

URDF_TEMPLATE = """
<?xml version="1.0"?>
<robot name="scene">
{table}
{objects}
</robot>
"""

TABLE = """
  <link name="table">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="./meshes/table.stl" scale="1.00 1.00 1.00"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="./meshes/table.stl" scale="1.00 1.00 1.00"/>
      </geometry>
    </collision>
  </link>
"""

OBJECT_TEMPLATE = """
  <link name="obj_{i}">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="./meshes/obj_{i}.stl" scale="1.00 1.00 1.00"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="./meshes/obj_{i}.stl" scale="1.00 1.00 1.00"/>
      </geometry>
    </collision>
  </link>
  <joint name="joint_{i}" type="fixed">
    <parent link="table" />
    <child link="obj_{i}" />
  </joint>
"""


def _D(x):
    return Path(x).mkdir(parents=True, exist_ok=True)


def _F(x):
    _D(Path(x).parent)
    return x


def yaml_from_obs_info(scene_id,
                       obs_info):
    obs_info = list(obs_info)

    # Add ground plane at index 0.
    obs_info.insert(0, {
        'shape': 'box',
        'base_pos': (0.0, 0.0, -0.05),
        'dims': (1.0, 1.0, 0.05)  # == halfExtents!
    })

    out = {}
    out['scene_id'] = scene_id
    out['setting_desc'] = 'RandomSampling'
    out['num_primitives'] = len(obs_info)

    for i, o in enumerate(obs_info):
        out_i = {}
        if o['shape'] == 'box':
            # halfExtents -> extents
            out_i['dimensions'] = [float(2 * x) for x in o['dims']]
        else:
            try:
                out_i['dimensions'] = [float(x) for x in o['dims']]
            except BaseException:
                out_i['dimensions'] = [float(o['dims'])]
        out_i['orientation'] = {
            'w': 1.0,
            'x': 0.0,
            'y': 0.0,
            'z': 0.0
        }
        out_i['position'] = {
            'x': o['base_pos'][0],
            'y': o['base_pos'][1],
            'z': o['base_pos'][2],
        }
        out_i['type'] = o['shape'].upper()
        out[i] = out_i
    return out


def scene_from_obs_info(obs_info):
    meshes = []
    for o in obs_info:
        if o['shape'] == 'box':
            # "halfExtents" -> extents
            dims = np.asarray([2 * x for x in o['dims']])
            m = trimesh.creation.box(dims).apply_translation(o['base_pos'])
            # m = o3d.geometry.TriangleMesh.create_box(*dims).cpu().translate(
            #     -0.5 * dims + o['base_pos'])
            # print(type(m))
            # print(m.normals)
        else:
            m = trimesh.creation.icosphere(radius=o['dims']).apply_translation(
                o['base_pos'])
            # m = o3d.geometry.TriangleMesh.create_sphere(
            #     o['dims']).cpu().translate(
            #     o['base_pos'])
        meshes.append(m)
        # meshes.append(o3d.geometry.LineSet.create_from_triangle_mesh(
        #     m))
    return meshes


def main(kwargs):
    kwargs['record_path']
    dataset_config = utils.load_config(
        kwargs['diffusion_path'],
        'dataset_config.pkl')
    dataset_config()

    os.makedirs(name=kwargs['save_path'], exist_ok=True)

    kwargs['env_type']

    MODEL_PATH = os.path.join(os.getcwd(), 'models')

    fine_dir = Path('/input/Scene-Diffuser/envs/assets/scene_description/')
    # dthvl15jruz9i2fok6bsy3qamp8c4nex000/fine_urdf/'
    out_dir = '/input/Scene-Diffuser/grasp_and_armmotion/datasets/FK2PlanDataset2/'
    prefix = 'trigger'

    desc = {
        "pre_code": prefix,
        "num_scenes": 0,

        # hmm...
        "tras_per_scene": 1,
        "frames_per_tra": 200,

        "scene_id": {
            # "0": "trigger000"
        },
        "scene_urdf_path": {
            # "trigger000": "SceneDescription/trigger000/SceneDescription.urdf"
        },
        "scene_mesh_path": {
            # "trigger000": "SceneDescription/trigger000/SceneMesh.stl"
        }
    }

    pcds = {}
    out_pkl = {
        'info': {'pre_code': prefix},
        'metadata': []
    }
    for idx in range(25):
        key = F'trigger{idx:03d}'
        print('key', key)

        # == Generate problem ==
        if kwargs['testset_path'] is not None:
            testset_path = kwargs['testset_path']
            print(testset_path)
            if not os.path.exists(os.path.join(
                    testset_path, 'test_{}.pkl'.format(idx))):
                continue
            with open(os.path.join(testset_path, 'test_{}.pkl'.format(idx)), 'rb') as f:
                test_info = pickle.load(f)
            print(test_info.keys())
            obs_info = test_info['obs_info']
            q1 = test_info['goal_joint_config']
            q0 = test_info['init_joint_config']
        else:
            print('stop')
            raise ValueError('stop')
        print(obs_info)

        # obs_info -> meshes
        meshes = scene_from_obs_info(obs_info)
        # plane_box = o3d.geometry.TriangleMesh.create_box(
        #     2.0, 2.0, 0.1).cpu().translate(np.asarray([-1.0, -1.0, -0.1]))
        plane_box = trimesh.creation.box(
            (2.0, 2.0, 0.1)).apply_translation(np.asarray([0.0, 0.0, -0.05]))
        meshes.insert(0, plane_box)

        out_mesh = trimesh.util.concatenate(meshes)

        # out_mesh = o3d.geometry.TriangleMesh()
        # for m in meshes:
        #     out_mesh += m

        obj_mesh = trimesh.util.concatenate(meshes[1:])
        points, face_indices = trimesh.sample.sample_surface(obj_mesh,
                                                             40960)
        normals = obj_mesh.face_normals[face_indices]
        pcd = np.concatenate([points, normals], axis=-1)
        pcds[key] = pcd

        if True:
            traj = test_info['final_path']

            while len(traj) > 200:
                traj = traj[::2]

        out_pkl['metadata'].append(
            (key,
             np.concatenate([q0, q1], axis=-1),  # start-goal config
             np.asarray(traj, dtype=np.float32)
             )
        )

        # formatted `obs`
        out_yaml = yaml_from_obs_info(idx, obs_info)

        # SceneDescription/trigger000'
        with open(_F(F'{out_dir}/SceneDescription/{key}/SceneConfig.yaml'), 'w') as fp:
            yaml.dump(out_yaml, fp)
        desc['num_scenes'] += 1
        desc['scene_id'][idx] = key

        out_urdf_file = _F(
            F'{out_dir}/SceneDescription/{key}/SceneDescription.urdf')
        desc['scene_urdf_path'][key] = out_urdf_file
        with open(out_urdf_file, 'w') as fp:
            fp.write('''
<?xml version="1.0"?>
<robot name="scene">
  <link name="scene">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="SceneMesh.stl" scale="1.00 1.00 1.00"/>
      </geometry>
    </visual>
  </link>
</robot>
''')
        with open(str(_F(fine_dir / key / 'fine_urdf' / 'fine_scene.urdf')), 'w') as fp:
            fp.write(URDF_TEMPLATE.format(
                table=TABLE,
                objects='\n'.join([OBJECT_TEMPLATE.format(
                    i=i) for i in range(len(meshes[1:]))])
            ))

        table_file = _F(fine_dir / key / 'fine_urdf' / 'meshes' / F'table.stl')
        meshes[0].export(table_file)
        for i, m in enumerate(meshes[1:]):
            mesh_file = _F(
                fine_dir /
                key /
                'fine_urdf' /
                'meshes' /
                F'obj_{i}.stl')
            m.export(mesh_file)

        out_mesh_file = _F(F'{out_dir}/SceneDescription/{key}/SceneMesh.stl')
        # o3d.io.write_triangle_mesh(out_mesh_file, out_mesh)
        out_mesh.export(out_mesh_file)
        desc['scene_mesh_path'][key] = out_mesh_file

    with open(_F(F'{out_dir}/scene_pcds_nors.pkl'), 'wb') as fp:
        pickle.dump(pcds, fp)

    with open(_F(F'{out_dir}/desc.json'), 'w') as fp:
        json.dump(desc, fp)

    with open(_F(F'{out_dir}/fk2plan_dataset.pkl'), 'wb') as fp:
        pickle.dump(out_pkl, fp)


if __name__ == '__main__':
    torch.cuda.set_device(DEVICE)

    test_kwargs = {
        'render': 'utils.PandaRenderer',
        'horizon': 1000,
        'enc_type': 'conv',
        'num_classes': 3,
        'num_steps': 256,
        'save_path': './tests/classifier-free/test/main_5_plain',
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

        'testset_path': os.path.join(cwd, 'testsets/main_5'),
        'record_path': os.path.join(cwd, 'record/test/main_5_plain')
    }

    main(test_kwargs)
