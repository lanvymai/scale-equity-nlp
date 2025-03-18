# Run this script to download OpenCLIP ViT-L/14

from transformers import AutoTokenizer, AutoModelForVision2Seq

model_id = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

# REPLACE THIS WITH YOUR NYU ID
id = "eyw2010"
local_dir = f"/scratch/{id}/models/CLIP-ViT-L-14"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir, trust_remote_code=True, use_fast=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=local_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True)


print("Model Downloaded Successfully to", local_dir)
