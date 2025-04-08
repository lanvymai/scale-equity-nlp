
import torch
import requests
from PIL import Image
# from io import BytesIO

from transformers import AutoModel, AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
# from llava.mm_utils import process_images, load_image_from_pil


sample_question = {
        "question": "Which label is most visually consistent with the image shown?",
        "image": "//upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Cristo_negro_de_Esquipulas.jpg/220px-Cristo_negro_de_Esquipulas.jpg",
        "options": [
            "The image in its glass case",
            "A Shinto rite; Shinto is often called an \"indigenous religion\", though the reasons for this classification have been debated among scholars.[10]",
            "Black Christ of Esquipulas at Saint Joseph Cathedral of Antigua Guatemala",
            "Buryat shaman on Olkhon Island, Siberia"
        ],
        "answer": "The image in its glass case",
        "category": "Indigenous Culture"
}

model_path = 'scale-equity-nlp/models/llava-v1.5-7b/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = LlavaLlamaForCausalLM.from_pretrained(model_path)#, torch_dtype=torch.float16).cuda().eval()
print('model done')
