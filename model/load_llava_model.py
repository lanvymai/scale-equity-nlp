from transformers import AutoModelForCasualLM, AutoTokenizer

model_name = "path/to/llava_model" # update with actual path

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
print("Model Loaded Successfully")
