#!/bin/bash
#SBATCH --job-name=eval_dparallel_no_prophet_baselines_%j
#SBATCH --time=4:00:00
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/hice1/kkatariya3/scratch/dParallel/slurm/logs/eval_dparallel_no_prophet_%j.out
#SBATCH --error=/home/hice1/kkatariya3/scratch/dParallel/slurm/logs/eval_dparallel_no_prophet_%j.err
#SBATCH --gres=gpu:H100:2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kkatariya3@gatech.edu

cd /home/hice1/kkatariya3/scratch

nvidia-smi

module load anaconda3
conda activate conda_envs/dparallel
cd dParallel/LLaDA
chmod +x eval.sh
bash ./eval.sh