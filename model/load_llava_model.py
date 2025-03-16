import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = f"/scratch/{os.environ['USER']}/models/llava-v1.5-7b"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

print("Model Loaded Successfully")
