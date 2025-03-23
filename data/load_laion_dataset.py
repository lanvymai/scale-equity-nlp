from datasets import load_dataset

ds = load_dataset("laion/laion400m", split="train[25%:50%]")

ds.to_parquet("/scratch/ez2545/scale-equity-nlp/Data/laion400m-2.parquet")

print("Dataset Saved as Parquet!")
