import argparse
import json
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, pipeline

torch.cuda.empty_cache()
def preprocess_image(image_url):
    # Ensure the URL starts with "http://" or "https://"
    try:
        # Make the HTTP request
        headers = {'User-Agent': 'NYU-NLU-class-project/0.0 (https://docs.google.com/document/d/1qJJJCK7chA6uXlnENVmzdz8LIj0FSW28-XLyHTKKEAc/edit?usp=sharing; nm4867@nyu.edu)'}
        response = requests.get(image_url, stream=True, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP errors

        # Check if the response is an image
        if "image" not in response.headers.get("Content-Type", ""):
            raise ValueError(f"URL does not point to an image: {image_url}")

        # Open the image
        image = Image.open(response.raw)#.convert("RGB")
        return image
    
    except requests.exceptions.RequestException as e:
        print(f"HTTP error while fetching image: {image_url} - {e}")
        return None
    except ValueError as e:
        print(f"Value error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while processing image: {image_url} - {e}")
        return None
        
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Bas%C3%ADlica_de_Esquipulas.jpg/330px-Bas%C3%ADlica_de_Esquipulas.jpg"

image = preprocess_image(image_url) #Image.open(requests.get(image_url, stream=True).raw)
use_fast_path = "/teamspace/studios/this_studio/scale-equity-nlp/models/llava-v1.5-7b/use_fast/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369"
model_path = "/teamspace/studios/this_studio/scale-equity-nlp/models/llava-v1.5-7b/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369"
processor = AutoProcessor.from_pretrained(use_fast_path)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

pipe = pipeline(task="image-text-to-text", model=use_fast_path, 
                tokenizer=processor.tokenizer,
            image_processor=processor.image_processor,
            model_kwargs={"quantization_config": quantization_config})
print("pipeline done")

max_new_tokens = 200
prompt = 'USER: <image>\nWhich caption best matches this image?\nOPTIONS: ["Basilica of the Cristo Negro of Esquipulas in Guatemala", "Culturally modified trees (CMTs) are when resources from a tree are used in a way that does not kill the tree itself.","South Moluccan shaman in an exorcism ritual involving children, Buru, Indonesia (1920)","Bonda \'disari\' (shaman) Sukra Dhangdamajhi shares his shamanic practices in the Bonda language"]\nASSISTANT:'

outputs = pipe(image, text=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
# Get the full generated text from the first (or only) result
full_text = outputs[0]['generated_text']

# Split on the keyword 'ASSISTANT:' and take the part after the last occurrence
model_output = full_text.split("ASSISTANT:")[-1].strip()

print("Model output:", model_output)
