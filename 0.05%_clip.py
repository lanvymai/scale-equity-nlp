import os
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from itertools import islice

Image.MAX_IMAGE_PIXELS = None

shard_path = "/scratch/ez2545/scale-equity-nlp/laion_output/shard_1"
model_path = "/scratch/ez2545/models/CLIP-ViT-L-14/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
subset_size = 450000
batch_size = 8
num_epochs = 3
lr = 5e-5

processor = CLIPProcessor.from_pretrained(model_path)
model = CLIPModel.from_pretrained(model_path)

def initialize_peft(model):
    lora_modules = [
        "q_proj", "k_proj", "v_proj", "out_proj",
        "fc1", "fc2",
        "visual_projection", "text_projection"
    ]
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=lora_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=None
    )
    model = get_peft_model(model, config)
    print("✅ LoRA injected, trainable parameters:")
    model.print_trainable_parameters()
    return model

model = initialize_peft(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tar_files = [os.path.join(shard_path, f) for f in os.listdir(shard_path) if f.endswith(".tar")]

dataset = (
    wds.WebDataset(tar_files)
    .shuffle(10000)  
    .decode("pil")  
    .to_tuple("jpg", "txt") 
    .map_tuple(
        lambda img: img.convert("RGB"), 
        lambda txt: txt.strip() 
    )
)

def random_subsample(dataset, subset_size):
    return list(islice(dataset, subset_size))

subset_dataset = random_subsample(dataset, subset_size)

print(f"✅ Expected subset size: {subset_size}")
actual_len = len(subset_dataset)
print(f"✅ valid image-text pairs: {actual_len}")

def collate_fn(batch):
    images, texts = zip(*batch)
    inputs = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True)
    return {k: v.to(device) for k, v in inputs.items()}

dataloader = DataLoader(subset_dataset, batch_size=batch_size, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

model.train()
for epoch in range(num_epochs):
    print(f"🔥 Starting epoch {epoch+1}")
    total_loss = 0

    for batch in tqdm(dataloader):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(logits_per_image.size(0), device=device)

        loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1} completed, avg loss: {total_loss/len(dataloader):.4f}")

model.save_pretrained("./finetuned_1clip_model")
print("✅ Model saved to ./finetuned_1clip_model")

