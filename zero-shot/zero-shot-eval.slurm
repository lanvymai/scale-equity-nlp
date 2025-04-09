#!/bin/bash
#SBATCH --partition=gpu               # might need to change this
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=clip_eval
#SBATCH --output=clip_eval_%j.out
#SBATCH --error=clip_eval_%j.err

# load the module?
# module load cuda/xxx

# environment
source xxx
conda activate xxx  # conda env name here

cd SCRATCH/$USER/scale-equity-nlp  # cd to whereever you place this slurm script

python evaluate_clip.py \
  --model openai/clip-vit-base-patch32 \
  --dataset evaluation_dataset.json \
  --debug
