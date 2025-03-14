from transformers import AutoTokenizer
from llava.model import LlavaForConditionalGeneration

model_path = "/scratch/ez2545/models/llava-v1.5-7b"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

print("Model Loaded Successfully")
