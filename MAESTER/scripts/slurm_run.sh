#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=MAESTER_betaSeg
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=127000M
#SBATCH --time=0-24:00

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

config="default"
config_name=$config.yaml
model_config_dir="../configs/"

log_dir="./checkpoints/betaSeg/"${config}
echo log_dir : `pwd`/$log_dir
mkdir -p `pwd`/$log_dir

echo "$SLURM_NODEID master: $MASTER_ADDR"
echo "$SLURM_NODEID Launching python script"
echo "$SLURM_NTASKS tasks running"

srun python train.py --init_method tcp://$MASTER_ADDR:56279 \
  --world_size $SLURM_NTASKS \
  --logdir $log_dir \
  --model_config_dir $model_config_dir \
  --model_config_name $config_name > $log_dir/output
