import sys
import torch
from transformers import AutoTokenizer

# Ensure Python can find LLaVA
sys.path.append("/scratch/ez2545/.local")

# Import LLaVA
from llava.model import LlavaForConditionalGeneration

# Model path
model_path = "/scratch/ez2545/models/llava-v1.5-7b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model with automatic device mapping
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)

print("Model Loaded Successfully")
