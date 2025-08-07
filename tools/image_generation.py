from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
from config.settings import DEVICE, DTYPE, LORA_PATH

# Text‑to‑image pipeline
txt2img = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
).to(DEVICE)
txt2img.safety_checker = lambda images, **kwargs: (images, False)

# Only load LoRA weights if a path has been resolved.  This allows the
# pipeline to function even when no LoRA file exists in the
# `lora_models` directory.
if LORA_PATH:
    txt2img.load_lora_weights(LORA_PATH)

# Image‑to‑image pipeline
img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
).to(DEVICE)
img2img.safety_checker = lambda images, **kwargs: (images, False)
if LORA_PATH:
    img2img.load_lora_weights(LORA_PATH)

def generate_image_from_prompt(prompt):
    result = txt2img(prompt)
    return result.images[0]

def stylize_image(image_path, prompt, strength=0.9, guidance_scale=8.5):
    base = Image.open(image_path).convert("RGB").resize((1024, 1024))
    result = img2img(prompt=prompt, image=base, strength=strength, guidance_scale=guidance_scale)
    return result.images[0]
