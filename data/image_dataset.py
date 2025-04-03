from datasets import load_dataset, Dataset
import requests
from io import BytesIO
from PIL import Image
import pandas as pd
from huggingface_hub import login
import os

login(token="token")

config = {
    "dataset_name": "ezruby/scale-equity-nlp",
    "output_dataset": "ezruby/laion-images",
    "chunk_size": 1000,  # adjust if needed
    "timeout": 10,
    "max_retries": 3
}

def download_image(url):
    for _ in range(config['max_retries']):
        try:
            response = requests.get(url, timeout=config['timeout'])
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            continue
    return None

def process_chunk(chunk):
    processed = []
    for item in chunk:
        img = download_image(item['URL'])
        if img:
            processed.append({
                "image": img,
                "caption": item['TEXT'],
                "original_url": item['URL']
            })
    return processed

def main():
    dataset = load_dataset(config['dataset_name'], split="train")
    
    for i in range(0, len(dataset), config['chunk_size']):
        print(f"Processing chunk {i//config['chunk_size']}")
        chunk = dataset[i:i+config['chunk_size']]
        processed_data = process_chunk(chunk)
        
        if processed_data:
            # Create dataset from processed data
            chunk_dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
            
            # Push to Hub
            chunk_dataset.push_to_hub(
                f"{config['output_dataset']}-chunk-{i//config['chunk_size']}",
                private=True
            )

if __name__ == "__main__":
    main()
