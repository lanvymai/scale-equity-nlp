#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --gres=gpu:1                     # requests 1 GPU per server
#SBATCH --gres=gpu:rtx8000:1             # To request specific GPU (v100 or rtx8000 or a100 or h100)
#SBATCH --time=1:00:00                   # requests 1 hour of compute time
#SBATCH --mem=32GB                        # requests 32GB of RAM memory per server
#SBATCH --job-name=train-llm
#SBATCH --output=train-llm.out

module purge
module load python/intel/3.8.6
module load openmpi/intel/4.0.5           # NOTE: we are loading "openmpi" module here
module load anaconda3/2020.07;

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/nm4867/llm;
export PATH=/scratch/nm4867/llm/bin:$PATH;

export HF_TOKEN=your_token_here;
srun trl sft \
    --model_name_or_path gpt2 \
    --dataset_name bigscience-data/roots_vi_wikiquote \
    --max_length 1024 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --save_steps 100 \
    --save_total_limit 2 \
    --warmup_ratio 0.05 \
    --use_liger_kernel \
    --logging_steps 1 \
    --output_dir gpt2-finetuned \
    --fp16
    # --bf16 \
    # --eval_strategy steps \
    # --eval_steps 100
