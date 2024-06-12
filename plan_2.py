from isaacgym import gymapi, gymutil, gymtorch

from collections import defaultdict
import os
import hydra
import torch
import random
import numpy as np
import json
from omegaconf import DictConfig, OmegaConf
# from loguru import logger
# import logging import 

from utils.misc import timestamp_str, compute_model_dim
from utils.io import mkdir_if_not_exists
from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model
from models.environment import create_enviroment

def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """ load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    saved_state_dict = torch.load(path)['model']
    model_state_dict = model.state_dict()

    for key in model_state_dict:
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
            print(f'Load parameter {key} for current model.')
        ## model is trained with ddm
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
            print(f'Load parameter {key} for current model [Traind from multi GPUs].')
    
    model.load_state_dict(model_state_dict)

def run(env,
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, save_dir: str
        ):
    """ Planning within the environment
    
    Args:
        model: diffusion model
        dataloader: test dataloader
        save_dir: save_directory of result
    """
    # save_record_path = os.path.join('/home/puhao/dev/SceneDiffuser', 'succ_record.json')
    model.eval()
    device = model.device

    angle_normalize = dataloader.dataset.angle_normalize
    angle_denormalize = dataloader.dataset.angle_denormalize
    replay_res = {}
    res = defaultdict(list)
    res['succ'] = []
    res['eval_cnt'] = 0
    res['scene_list'] = []
    for i, data in enumerate(dataloader):
        # ????
        # if i == 0:
        #     continue
        if i > 10:
            break

        # Send to device...
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device)

        i_scene_id = data['scene_id'][0]
        print(i_scene_id)
        i_target_qpos = np.array(data['target'].cpu())

        # ???
        # if data['scene_id'][0] in succ_scene_record.keys():
        #     continue
        if data['scene_id'][0] in res['scene_list']:
            continue

        succ_record_list = []
        ## de-normalize for env input
        if dataloader.dataset.normalize_x:
            data['x'] = angle_denormalize(data['x'].cpu()).cuda()
            data['target'] = angle_denormalize(data['target'].cpu()).cuda()
            data['start'] = angle_denormalize(data['start'].cpu()).cuda()

        #env = FK2PlanningEnvCore(
        #    data=copy.deepcopy(data),
        #    scene_id=data['scene_id'][0],
        #    arrive_threshold=self.arrive_threshold,
        #    sims_per_step=self.sims_per_step,
        #    max_trajectory_length=self.max_trajectory_length
        #)

        ## re-normalize for model input
        if dataloader.dataset.normalize_x:
            data['x'] = angle_normalize(data['x'].cpu()).cuda()
            data['target'] = angle_normalize(data['target'].cpu()).cuda()
            data['start'] = angle_normalize(data['start'].cpu()).cuda()

        outputs = model.sample(data, k=env.max_sample_each_step)
        # [9, 1, 31, 65, 7]
        print('outputs', outputs.shape)

        # pred_next_qpos = outputs[:, 0, -1, 10, :]

        if dataloader.dataset.normalize_x:
            traj = angle_denormalize(outputs.cpu()).cuda()
        res['traj'].append(traj.detach().cpu().numpy().tolist())

        #print()
        #i_trajectories = env.get_trajectories()
        #i_trajectories = np.array([np.stack(case) for case in i_trajectories])
        ## save replayer recorder
        #replay_res[i_scene_id] = {'sample_trajs': i_trajectories,
        #                            'target_qpos': i_target_qpos}

        #for j in range(env.batch):
        #    succ_record_list.append(bool(env.end[j].cpu()))
        #    res['scene_list'].append(data['scene_id'][0])
        #    res['succ'].append(bool(env.end[j].cpu()))
        #    res['length'].append(float(env.trajectory_length[j]))
        # print(f'[{i}/{200}]')
        # succ_scene_record[data['scene_id'][0]] = succ_record_list
        #res['eval_cnt'] += env.batch

        # json.dump(succ_scene_record, open(save_record_path, 'w'))

        # del env
        # with open(res_saver_path, 'w') as fp:
        #     json.dump(res, fp)

    # with open(replay_saver_path, 'wb') as fp:
    #     pickle.dump(replay_res, fp)
    # for key in ['succ', 'length']:
    #     res[key+'_average'] = sum(res[key]) / len(res[key])

    ## save quantitative results
    save_path = os.path.join(save_dir, 'metrics.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as fp:
        json.dump(res, fp)
    # with open(res_saver_path, 'w') as fp:
    #     json.dump(res, fp)


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ## compute modeling dimension according to task
    cfg.model.d_x = compute_model_dim(cfg.task)
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True # update slurm config
    
    ## set output dir
    eval_dir = os.path.join(cfg.exp_dir, 'eval')
    mkdir_if_not_exists(eval_dir)
    res_dir = os.path.join(eval_dir, 'plan', timestamp_str())
    
    # print(res_dir + '/plan.log') # set logger file
    # logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg)) # record configuration

    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    ## prepare dataset for evaluating on planning task
    ## only load scene
    datasets = {
        'test': create_dataset(cfg.task.dataset, 'test', cfg.slurm, case_only=True),
    }
    for subset, dataset in datasets.items():
        # logger.info(f'Load {subset} dataset size: {len(dataset)}')
        pass
    
    if cfg.model.scene_model.name == 'PointTransformer':
        collate_fn = collate_fn_squeeze_pcd_batch
    else:
        collate_fn = collate_fn_general
    
    dataloaders = {
        'test': datasets['test'].get_dataloader(
            batch_size=cfg.task.test.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.test.num_workers,
            pin_memory=True,
            shuffle=False,
        )
    }
    
    ## create model and diffuser, load ckpt, create and load optimizer and planner for diffuser
    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)
    ## if your models are seperately saved in each epoch, you need to change the model path manually
    if cfg.model.name != 'ActorL2':
        load_ckpt(model, path=os.path.join(cfg.ckpt_dir, 'model.pth'))
    
    ## create environment for planning task and run
    env = create_enviroment(cfg.task.env)
    #env.run(model, dataloaders['test'], res_dir)
    run(env, model, dataloaders['test'], res_dir)

if __name__ == '__main__':
    ## set random seed
    seed = 0
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
