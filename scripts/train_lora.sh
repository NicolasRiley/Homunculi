#!/usr/bin/env bash
set -euo pipefail

ACC="/home/nriley72/Nico-Riley-Homunculi/Homunculi/venvs/homunculi/bin/accelerate"
SCRIPT="/home/nriley72/Nico-Riley-Homunculi/Homunculi/kohya/sd-scripts/train_network.py"
BASE="/home/nriley72/Nico-Riley-Homunculi/Homunculi/stable-diffusion/stable-diffusion-v1-5"
DATA="/home/nriley72/Nico-Riley-Homunculi/Homunculi/training_data/HC_Portraits_r1"
OUT="/home/nriley72/Nico-Riley-Homunculi/Homunculi/lora_models"

STEPS="${1:-200}"
LR="${2:-1e-4}"

mkdir -p "$OUT"

"$ACC" launch --mixed_precision=bf16 "$SCRIPT" \
  --pretrained_model_name_or_path="$BASE" \
  --train_data_dir="$DATA" \
  --output_dir="$OUT" \
  --resolution=512,512 \
  --network_module=networks.lora \
  --max_train_steps="$STEPS" \
  --learning_rate="$LR" \
  --cache_latents \
  --save_model_as=safetensors
