from datasets import load_dataset

dataset = load_dataset("laion/laion400m", cache_dir="/scratch/ez2545/datasets/laion400m")

print("Dataset Downloaded Successfully!")