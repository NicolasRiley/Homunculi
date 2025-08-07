"""
Configuration module for the Homunculi project.

This module centralises device and model‑specific settings and
automatically discovers the most recent LoRA weight file from the
`lora_models` directory. By default, the agent looks for files
ending with `.safetensors` inside `lora_models` and picks the one
with the latest modification time. If no LoRA file is found, the
pipelines fall back to using the base model without additional
weights.
"""

import os
import glob
import torch

# Determine compute device and tensor dtype
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Dynamically determine the latest LoRA file in the `lora_models` folder.
# The lora_models directory resides at the root of the project next to
# agents/, config/, tools/, etc.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_DIR = os.path.join(BASE_DIR, "lora_models")

# Collect all safetensor files and sort by modification time
lora_files = glob.glob(os.path.join(LORA_DIR, "*.safetensors"))
LORA_PATH = None
if lora_files:
    # Sort files by modification time; newest file is last in the list
    lora_files.sort(key=lambda p: os.path.getmtime(p))
    LORA_PATH = lora_files[-1]

__all__ = ["LORA_PATH", "DEVICE", "DTYPE"]