from transformers import AutoTokenizer, AutoModelForVision2Seq

model_path = "/scratch/eyw2010"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)

print("Model Loaded Successfully")
