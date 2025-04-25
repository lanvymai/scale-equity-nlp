# Run this script to download SmolVLM
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# REPLACE THIS WITH YOUR NYU ID
id = "eyw2010"
local_dir = f"scale-equity-nlp/models/SmolVLM"

model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", cache_dir=local_dir, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", cache_dir=local_dir)


print("Model Downloaded Successfully to", local_dir)