from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/scratch/ez2545/models/llava-v1.5-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)

print("Model Loaded Successfully")
