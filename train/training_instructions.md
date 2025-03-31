# Training LLM workflow

## 1. Login to Greene Cluster
```
ssh <NetID>@greene.hpc.nyu.edu 
```
(if off campus, turn on VPN before log into Greene cluster.)

- If "host key verification failed", run:
  ```
  rm -rf /Users/<folder_name_failed>/.ssh/known_hosts
  ```
- Then log in again.

## 2. Use tmux (Terminal Multiplexer)
- To maintain code running in the background even if we close terminal / VS Code & help us monitor different logs and run different scripts.
- Commands
    - Open a session:
      ```
      tmux
      ```
    - Check the list of session:
      ```
      tmux ls
      ```
    - Get into (aka. re-attach) already existing session
      ```
      tmux a -t <SessionID found from tmux ls>
      ```
    - Get out of (aka. detach from) a session: (on the keyboard) Ctrl + B, then D
    - Switch to next window: Ctrl + B, then N
    - Switch to previous window: Ctrl + B, then P

## 3. Create Conda environment
Run these in terminal:

### 3.1. Download package
```
module purge
module load anaconda3/2020.07
```
```
mkdir /home/<NetID>/.conda
mkdir /scratch/<NetID>/conda_pkgs
ln -s /scratch/<NetID>/conda_pkgs /home/<NetID>/.conda/pkgs
```
### 3.2. Install packages
```
conda create -p /scratch/<NetID>/llm python=3.10
```
### 3.3. Activate conda
```
conda activate /scratch/<NetID>/llm
conda init
```
### 3.4. Instal libraries / dependencies
```
pip install accelerate peft bitsandbytes transformers==4.50.0 trl==0.16.0 scipy numpy==2.0.2 liger-kernel==0.5.5
```


## 4. Training Workflow
- After logging into Greene cluster
- ```ssh burst```
- Go into `scratch/<NetID>`: ```git clone git@github.com:lanvymai/scale-equity-nlp.git```
- ```cd train```
- Edit the jobs.sbatch file: ```vi jobs.sbatch```
    - `hf_token`: go into huggingface -> your profile -> access token -> create new -> copy that token and paste it here (will never show you again)
    - Edit all the `<NetID>` to your NetID
    - Should customize:
        - GPU: check for available GPU (v100 or rtx8000 or a100 or h100), usually rtx8000 has highest capacity
        - `time`: small job request less time, time request can scale linearly with the number of task. Remember SLURM job prioritize job with less time so if we request 1 day we might don't have a slot
        - `mem`: (RAM memory) usually 32GB or 64GB, also depending on the task, but does not scale linearly
        - `job-name` & `output`: to distinguish between different jobs
        - `model_name_or_path`: specify model name (as long as it's on hugging face)
        - `dataset_name`: (need to figure out how to get a small sample to run first)
        - `max_length`: depending on models, should get a common max_length for all models being train for equitable comparision across models
        - `num_train_epochs`: usually 1 or 2
        - `per_device_train_batch_size`: depending on the models, usually try out on Google Collab first to find the max batch size, then start with that 
        (small batch size will take a long time, while large batch size can exceed memory)
        - `output_dir`: name of the output directory, will be within the /scratch/<NetID>/llm
- Check for status of the job: ```squeue -u <NetID>```
- Check output file: ```cat <output>```

## Trouble-shooting:
1. ERROR "`Disk quota exceeded`"
- Likely that we've been using home to download and run stuff and we've used up the allocated storage.
- If this is the case, run:
  ```
  quota
  du -hs ./*
  du -hs ./.cache/*
  rm -rf ./.cache/*
  ```
- Then, download things and direct results to `/scratch/<NetID>`, instead of `/home/<NetID>`.
