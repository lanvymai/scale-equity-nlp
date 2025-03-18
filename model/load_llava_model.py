from transformers import AutoTokenizer, AutoModelForVision2Seq

model_id = "llava-hf/llava-v1.5-7b"
local_dir = "/scratch/eyw2010/models/llava-v1.5-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir, trust_remote_code=True, use_fast=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=local_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True)

print("Model Downloaded Successfully to", local_dir)
