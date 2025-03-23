import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import os

image_dir = "/scratch/ez2545/scale-equity-nlp/images"
os.makedirs(image_dir, exist_ok=True)

df = pd.read_parquet('/scratch/ez2545/scale-equity-nlp/Data/laion400m.parquet')

image_urls = df['url']  
captions = df['caption']

# bruv
def download_image(url, caption, image_dir):
    try:
        response = requests.get(url)
        response.raise_for_status()

        image_name = f"{caption[:50]}.jpg"  # file name = caption
        image_path = os.path.join(image_dir, image_name)

        with open(image_path, 'wb') as f:
            f.write(response.content)
        print(f"downloaded: {image_name}")
    except Exception as e:
        print(f"failed to download {url}: {e}")

# disgusting pig
with ThreadPoolExecutor(max_workers=4) as executor:
    for url, caption in zip(image_urls, captions):
        executor.submit(download_image, url, caption, image_dir)

print("images downloaded successfully")

