#!/bin/bash
#SBATCH --job-name=eval_all_baselines_%j
#SBATCH --time=4:00:00
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/hice1/rbansal66/scratch/dParallel/slurm/logs/eval_all_baselines_%j.out
#SBATCH --error=/home/hice1/rbansal66/scratch/dParallel/slurm/logs/eval_all_baselines_%j.err
#SBATCH --gres=gpu:H100:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rbansal66@gatech.edu

cd /home/hice1/rbansal66/scratch

nvidia-smi

module load anaconda3
conda activate dparallel
cd dParallel/LLaDA

chmod +x eval.sh
bash ./eval.sh