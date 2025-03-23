from datasets import load_dataset

ds = load_dataset("laion/laion400m", split="train[75%:]")

ds.to_parquet("/scratch/ez2545/scale-equity-nlp/Data/laion400m-4.parquet")

print("Dataset Saved as Parquet!")
