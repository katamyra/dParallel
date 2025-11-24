#!/bin/bash
#SBATCH --job-name=eval_all_baselines_%j
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel/slurm/logs/eval_all_baselines_%j.out
#SBATCH --error=/home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel/slurm/logs/eval_all_baselines_%j.err
#SBATCH --gres=gpu:H100:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jzhang3463@gatech.edu

cd /home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel

nvidia-smi

module load anaconda3
module load cuda
conda activate dparallel
cd LLaDA

bash ./eval.sh