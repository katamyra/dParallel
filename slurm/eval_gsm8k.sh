#!/bin/bash
#SBATCH --job-name=eval_gsm8k_%j
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=/home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel/slurm/logs/eval_gsm8k_%j.out
#SBATCH --error=/home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel/slurm/logs/eval_gsm8k_%j.err
#SBATCH --gres=gpu:H100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jzhang3463@gatech.edu

cd /home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel

nvidia-smi

module load anaconda3
module load cuda
conda activate dparallel
cd LLaDA

bash ./eval.sh gsm8k