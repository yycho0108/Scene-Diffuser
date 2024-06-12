CKPT=$1

CUDA_VISIBLE_DEVICES=0 python3 plan_2.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT} \
                diffuser=ddpm \
                diffuser.steps=30 \
                model=unet_fk2 \
                model.use_position_embedding=true \
                task=franka_planning \
                task.dataset.normalize_x=true \
                planner=greedy_fk2_planning \
                planner.scale=0.2
