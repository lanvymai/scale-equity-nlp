#!/bin/bash
#SBATCH --job-name=laion_download
#SBATCH --output=laion_download_%j.out
#SBATCH --error=laion_download_%j.err
#SBATCH --partition=cs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ez2545@nyu.edu

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate /scratch/ez2545/llava_env



INPUT_DIR="/scratch/ez2545/scale-equity-nlp/Data"
OUTPUT_DIR="/scratch/ez2545/scale-equity-nlp/laion_output"
mkdir -p "$OUTPUT_DIR"

for i in {1..4}; do
    PARQUET_PATH="$INPUT_DIR/laion400m-$i.parquet"
    OUTDIR="$OUTPUT_DIR/shard_$i"
    mkdir -p "$OUTDIR"

    echo "Processing $PARQUET_PATH..."
    
    img2dataset \
      --url_list "$PARQUET_PATH" \
      --output_folder "$OUTDIR" \
      --processes_count 16 \
      --thread_count 64 \
      --image_size 256 \
      --resize_mode no \
      --output_format webdataset \
      --input_format "parquet" \
      --url_col "url" \
      --caption_col "caption" \
      --enable_wandb False
done

echo "Download job completed"
