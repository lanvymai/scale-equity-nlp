from datasets import load_dataset

dataset = load_dataset("laion/laion400m", split='train')
dataset.save_to_disk("/scratch/ez2545/datasets/laion400m")

print("Dataset Downloaded Successfully!")
