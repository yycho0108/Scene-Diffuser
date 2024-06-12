#!/usr/bin/env python3

import os
import sys
import numpy as np
import open3d as o3d
import torch
import yaml
import json
import pickle
import trimesh
from yourdfpy import URDF
from pathlib import Path

from etude.data.ext_util.spec import (Scene,
                                      str_from_cls)
from etude.data.ext_util.open3d_spec import (
    to_open3d
)
from etude.data.ext_util.curobo_spec import (
    from_curobo,
)
from etude.data.ext_util.mingyo_spec import (
    from_mingyo
)

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


def yaml_from_scene(scene_id, scene: Scene):
    # Add ground plane at index 0.
    out = {}
    out['scene_id'] = scene_id
    out['setting_desc'] = 'RandomSampling'
    out['num_primitives'] = len(scene.geom)

    # for i, o in enumerate(obs_info):
    for i, (k, v) in enumerate(scene.geom.items()):
        out_i = {}
        geom_type = str_from_cls(v).upper()
        if geom_type == 'CUBOID':
            geom_type = 'BOX'

        if geom_type == 'BOX':
            # halfExtents -> extents
            out_i['dimensions'] = [float(2 * x) for x in v.radius]
        elif geom_type == 'SPHERE':
            out_i['dimensions'] = [v.radius]
        elif geom_type == 'CYLINDER':
            out_i['dimensions'] = [v.radius, 2.0 * v.half_length]
        else:
            raise ValueError(F'unknown geom_type={geom_type}')

        pose = v.pose
        if pose is None:
            pose = (0, 0, 0,
                    0, 0, 0, 1)

        out_i['position'] = {
            'x': float(v.pose[0]),
            'y': float(v.pose[1]),
            'z': float(v.pose[2])
        }
        out_i['orientation'] = {
            'w': float(v.pose[3 + 3]),
            'x': float(v.pose[3 + 0]),
            'y': float(v.pose[3 + 1]),
            'z': float(v.pose[3 + 2])
        }
        out_i['type'] = geom_type
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
    os.makedirs(name=kwargs['save_path'], exist_ok=True)
    fine_dir = Path('/input/Scene-Diffuser/envs/assets/scene_description/')
    out_dir = '/input/Scene-Diffuser/grasp_and_armmotion/datasets/FK2PlanDataset4/'
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
    # for idx in range(25):
    # idx = 0
    if True:
        # idx += 1
        # print('idx', idx)
        # '/tmp/docker/env_info.1/3_1_6_1000/info.yaml'
        # Load dataset
        #f_src = '/input/ETUDE-dev/datasets/etude_cabinet/raw/0000.pkl'
        # f_src = '/input/ETUDE-dev/datasets/etude_cabinet/merge/all.pkl'#raw/0000.pkl'
        f_src = '/input/ETUDE-dev/datasets/etude_cabinet_eval/obj-0-0/merge/all.pkl'
        with open(f_src, 'rb') as fp:
            d_src = pickle.load(fp)
            # -> (ws, ss, qs)

        WS = list(d_src['ws'])# * 25
        SS = list(d_src['ss'])# * 25
        QS = list(d_src['qs'])# * 25
        #print(WS[0],
        #      SS[0],
        #      QS[0][0],
        #      QS[0][-1])

        for idx, (w, s, q) in enumerate(zip(WS, SS, QS)):
            print('s', s)
            #if idx >= 25:
            #    break
            s = str(s).replace('datasets/etude_cabinet_eval/env_info',
                               # '/input/ETUDE-dev/datasets/'
                               #'/tmp/docker/env_info.1/'
                               '/input/ETUDE-dev/datasets/etude_cabinet_eval/env_info'
                               )
            key = F'{prefix}{idx:03d}'
            # world data
            with open(s, 'r') as fp:
                scene = from_mingyo(yaml.safe_load(fp))
                if True:
                    for k, v in scene.geom.items():
                        v.pose[..., :3] -= scene.base_pose[..., :3]
                    scene.base_pose[..., :3] -= scene.base_pose[..., :3]
                scene_o3d = to_open3d(scene, wire=False)

            # init-goal joint configs
            q0 = q[..., 0, :]
            q1 = q[..., -1, :]

            table_meshes = []
            objct_meshes = []
            for k, v in scene.geom.items():
                # print(k)
                #if 'shelf' in k or 'table' in k:
                if 'table' in k:
                    table_meshes.append(scene_o3d[k])
                else:
                    objct_meshes.append(scene_o3d[k])

            # obs_info -> meshes
            # meshes = scene_from_obs_info(obs_info)
            # plane_box = o3d.geometry.TriangleMesh.create_box(
            #     2.0, 2.0, 0.1).cpu().translate(np.asarray([-1.0, -1.0, -0.1]))
            # plane_box = trimesh.creation.box(
            #     (2.0, 2.0, 0.1)).apply_translation(np.asarray([0.0, 0.0, -0.05]))
            # meshes.insert(0, plane_box)

            out_mesh = sum(table_meshes[1:] + objct_meshes,
                           start=table_meshes[0])
            out_mesh = trimesh.Trimesh(
                vertices=np.asarray(out_mesh.vertices),
                faces=np.asarray(out_mesh.triangles)
            )
            tbl_mesh = sum(table_meshes[1:],
                           start=table_meshes[0])
            obj_mesh = sum(objct_meshes[1:],
                           start=objct_meshes[0])
            meshes = [tbl_mesh, *objct_meshes]
            meshes = [trimesh.Trimesh(
                vertices=np.asarray(mm.vertices),
                faces=np.asarray(mm.triangles)
            ) for mm in meshes]

            obj_mesh = trimesh.Trimesh(
                vertices=np.asarray(obj_mesh.vertices),
                faces=np.asarray(obj_mesh.triangles)
            )

            points, face_indices = trimesh.sample.sample_surface(obj_mesh,
                                                                 40960)
            normals = obj_mesh.face_normals[face_indices]
            pcd = np.concatenate([points, normals], axis=-1)
            pcds[key] = pcd

            if True:
                traj = q
                while len(traj) > 200:
                    traj = traj[::2]

            out_pkl['metadata'].append(
                (key,
                 np.concatenate([q0, q1], axis=-1),  # start-goal config
                 np.asarray(traj, dtype=np.float32)
                 )
            )

            # formatted `obs`
            out_yaml = yaml_from_scene(idx, scene)

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

            table_file = _F(
                fine_dir /
                key /
                'fine_urdf' /
                'meshes' /
                F'table.stl')
            meshes[0].export(table_file)
            for i, m in enumerate(meshes[1:]):
                mesh_file = _F(
                    fine_dir /
                    key /
                    'fine_urdf' /
                    'meshes' /
                    F'obj_{i}.stl')
                m.export(mesh_file)

            out_mesh_file = _F(
                F'{out_dir}/SceneDescription/{key}/SceneMesh.stl')
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
    DEVICE = 'cuda:1'
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

        # 'testset_path': os.path.join(cwd, 'testsets/main_5'),
        # 'record_path': os.path.join(cwd, 'record/test/main_5_plain')
    }

    main(test_kwargs)
