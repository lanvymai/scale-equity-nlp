from datasets import load_dataset

dataset = load_dataset("laion/laion400m", cache_dir="/scratch/$USER/datasets/laion400m")

print("Dataset Downloaded Successfully!")
