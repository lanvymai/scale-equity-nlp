from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "scratch/ez2545/models/llava-v1.5-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
print("Model Loaded Successfully")
