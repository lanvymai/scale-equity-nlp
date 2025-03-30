#!/bin/bash
#SBATCH --job-name=image_download
#SBATCH --output=download_images.out
#SBATCH --error=download_images.err
#SBATCH --time=4:00:00
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --partition=cs

module load cuda/11.6.2
source llava_env/bin/activate

python /scratch/ez2545/scale-equity-nlp/images/parse-parquet.py
