#!/bin/bash
#SBATCH --job-name=finetuning_clip
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
module load cuda/11.6.2
conda activate /scratch/ez2545/llava_env
python ./testt.py


