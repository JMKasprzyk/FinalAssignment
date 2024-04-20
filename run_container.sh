export WANDB_API_KEY=cadc5cb0be3f2d907bb3a64de10fa2818f2b1a53
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

torchrun --nnodes=1 --nproc_per_node=1 train.py \
        --data_path "/gpfs/work5/0/jhstue005/JHS_data/CityScapes"\
        --epochs 150 --lr 0.01\
        --wandb_name "UNet-CE"\
        --checkpoint_folder "checkpoints/report/UNet-CE"\
        --architecture "UNet"