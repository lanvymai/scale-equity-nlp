import pandas as pd

file_path = "/scratch/nm4867/scale-equity-nlp/Data/laion400m-1.parquet"
try:
    df = pd.read_parquet(file_path, engine="pyarrow")
    print(df.head())
except Exception as e:
    print(f"Error reading Parquet file: {e}")