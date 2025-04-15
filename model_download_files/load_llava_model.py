from transformers import AutoTokenizer, AutoModelForVision2Seq
from transformers import AutoProcessor, LlavaForConditionalGeneration


# REPLACE THIS WITH YOUR NYU ID
id = "eyw2010"
model_id = "llava-hf/llava-1.5-7b-hf"
local_dir = f"scale-equity-nlp/models/llava-v1.5-7b/use_fast"

# tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir, trust_remote_code=True, use_fast=False)
# processor = LlavaProcessor.from_pretrained(model_id, cache_dir=local_dir, trust_remote_code=True, use_fast=False)
model = LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir=local_dir, torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, offload_folder="save_folder")
processor = AutoProcessor.from_pretrained(model_id, cache_dir=local_dir, trust_remote_code=True, use_fast=True)

print("Model Downloaded Successfully to", local_dir)
