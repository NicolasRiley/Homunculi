from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from config.settings import DEVICE

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_style_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features /= features.norm(p=2, dim=-1, keepdim=True)
    return features[0]
