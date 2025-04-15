# Run this script to download MobileVLM 1.4B

import torch
from transformers import AutoTokenizer, LlamaForCausalLM #AutoModelForCausalLM #LlamaForCausalLM, MobileLlamaForCausalLM
# from mobilevlm.model.mobilellama import MobileLlamaForCausalLM

model_id = 'mtgv/MobileVLM_V2-1.7B'

# REPLACE THIS WITH YOUR NYU ID
id = "eyw2010"
local_dir = f"scale-equity-nlp/models/MobileVLM_V2-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_id, cache_dir=local_dir)

print("Model Downloaded Successfully to", local_dir)
