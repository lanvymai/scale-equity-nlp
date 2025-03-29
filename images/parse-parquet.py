import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import os
import hashlib

image_dir = "/scratch/ez2545/scale-equity-nlp/images"
os.makedirs(image_dir, exist_ok=True)

parquet_path = "/scratch/ez2545/scale-equity-nlp/Data/laion400m.parquet"
df_iter = pd.read_parquet(parquet_path, engine="pyarrow", columns=["url", "caption"], iterator=True)

def clean_filename(text, length=50):
    return hashlib.md5(text.encode()).hexdigest()[:10] + ".jpg"

def download_image(url, caption):
    try:
        filename = clean_filename(caption)
        image_path = os.path.join(image_dir, filename)

        if os.path.exists(image_path):  # Skip if already downloaded
            return

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(image_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed: {url} -> {e}")

with ThreadPoolExecutor(max_workers=10) as executor:
    for df in df_iter:
        urls, captions = df["url"], df["caption"]
        executor.map(download_image, urls, captions)

print("Image download complete.")
