from transformers import AutoTokenizer, AutoModelForVision2Seq

# REPLACE THIS WITH YOUR NYU ID
id = "eyw2010"
model_id = "llava-hf/llava-1.5-7b-hf"
local_dir = f"scale-equity-nlp/models/llava-v1.5-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir, trust_remote_code=True, use_fast=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=local_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True, offload_folder="save_folder")

print("Model Downloaded Successfully to", local_dir)
