#!/usr/bin/env python3

import pickle
import trimesh

with open('./grasp_and_armmotion/datasets/FK2PlanDataset3/scene_pcds_nors.pkl', 'rb') as fp:
    d = pickle.load(fp)
p = next(iter(d.values()))
trimesh.PointCloud(p[..., :3]).export('/tmp/docker/cld.ply')
