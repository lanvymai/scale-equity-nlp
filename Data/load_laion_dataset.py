from datasets import load_dataset

ds = load_dataset("laion/laion400m", split="train", streaming=True)

print("Dataset Downloaded Successfully!")
