
import torch
import requests
from PIL import Image
from io import BytesIO

from transformers import AutoModel, AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.mm_utils import process_images, load_image_from_base64


sample = {
        "question": "Which caption best matches this image?",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Bas%C3%ADlica_de_Esquipulas.jpg/330px-Bas%C3%ADlica_de_Esquipulas.jpg",
        "options": [
            "Basilica of the Cristo Negro of Esquipulas in Guatemala",
            "Culturally modified trees (CMTs) are when resources from a tree are used in a way that does not kill the tree itself.",
            "South Moluccan shaman in an exorcism ritual involving children, Buru, Indonesia (1920)",
            "Bonda \"disari\" (shaman) Sukra Dhangdamajhi shares his shamanic practices in the Bonda language"
        ],
        "answer": "Basilica of the Cristo Negro of Esquipulas in Guatemala",
        "category": "Indigenous Culture"
    }

model_path = 'scale-equity-nlp/models/llava-v1.5-7b/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(model_path)#, torch_dtype=torch.float16).cuda().eval()
print('model done')


# Helper: Download image from URL
def download_image(url):
    if url.startswith("//"):
        url = "https:" + url
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# Format prompt
def make_prompt(sample):
    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample["options"])])
    prompt = (
        f"USER: <image>\n{sample['question']}\n"
        f"{options_text}\n"
        f"Please choose the best answer (A, B, C, or D).\nASSISTANT:"
    )
    return prompt

# Run inference on one sample
def evaluate_sample(sample):
    image = download_image(sample["image"])
    loaded_img = load_image_from_base64(image)
    image_tensor = process_images([loaded_img], model.config).to(model.device, dtype=torch.float16)

    prompt = make_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            images=image_tensor,
            max_new_tokens=50
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Inference
model_response = evaluate_sample(sample)
print("Model Output:", model_response)
