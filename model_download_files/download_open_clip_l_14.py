# Run this script to download OpenCLIP ViT-L/14

from transformers import CLIPProcessor, CLIPModel

# REPLACE THIS WITH YOUR NYU ID
id = "eyw2010"
local_dir = f"scale-equity-nlp/models/CLIP-ViT-L-14"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=local_dir, trust_remote_code=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=local_dir)


print("Model Downloaded Successfully to", local_dir)

