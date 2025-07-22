import torch

LORA_PATH = "LoRA Files/dreamlookai_sdxl-v1_lora_db_46elI84n_ckp_O1rjmUuq_step_1500_image_in_ukj_style.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32