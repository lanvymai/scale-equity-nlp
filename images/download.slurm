#!/bin/bash
#SBATCH --job-name=image_download
#SBATCH --output=download_images.out
#SBATCH --error=download_images.err
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_misc_v100

module load cuda/11.6.2
python -m venv llava_env
source llava_env/bin/activate

python /scratch/ez2545/scale-equity-nlp/images/parse-parquet.py
