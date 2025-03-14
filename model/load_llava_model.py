from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "scratch/$USER/scale-equity-nlp/model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
print("Model Loaded Successfully")
