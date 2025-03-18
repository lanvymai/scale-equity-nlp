# Run this script to download MobileVLM 1.4B

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_id = 'mtgv/MobileLLaMA-1.4B-Chat'

# REPLACE THIS WITH YOUR NYU ID
id = "eyw2010"
local_dir = f"/scratch/{id}/nlu/models/MobileLLaMA-1.4B-Chat"

tokenizer = LlamaTokenizer.from_pretrained(model_id, cache_dir=local_dir, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(
    model_id, cache_dir=local_dir)

print("Model Downloaded Successfully to", local_dir)
