# Homunculi — Setup & Quickstart

This guide gets you from zero → trained LoRA → generated images → CLIP scores on the H200 server.

## (1) Prereqs

- Linux server with NVIDIA GPU (H200), CUDA 12.1 drivers
- Python 3.12 (we use a venv)
- A Hugging Face account + access token (for SD 1.5)

## (2) Clone + create venv

```bash
git clone https://github.com/NicolasRiley/Homunculi.git
cd Homunculi

python3 -m venv venvs/homunculi
source venvs/homunculi/bin/activate
pip install --upgrade pip
```

## (3) Install dependencies
Requirements listed in requirements.txt 

```bash
pip install -r requirements.txt
```

## (4) Get Stable-Diffusion 1.5 (diffusers format)

```bash
# login once; choose "yes" for token-only (no git credential)
huggingface-cli login

mkdir -p stable-diffusion
huggingface-cli download runwayml/stable-diffusion-v1-5 \
  --local-dir stable-diffusion/stable-diffusion-v1-5
```

## (5) Dataset Layout
Organize training data in a DreamBooth-style folder structure:

```arduino 
training_data/
    HC_Portraits_r1/
        10_my_style/
            nicolasvision_00.jpg
            nicolasvision_01.jpg
            ...
            # optional captions:
            # nicolasvision_00.txt -> "my_style high-contrast portrait, rim lighting
```

## (7) Dry Run (Sanity Check)

Create the lora_models/last.safetensors file

```bash
accelerate launch --mixed_precision=bf16 \
  kohya/sd-scripts/train_network.py \
  --pretrained_model_name_or_path="stable-diffusion/stable-diffusion-v1-5" \
  --train_data_dir="training_data/HC_Portraits_r1" \
  --output_dir="lora_models" \
  --resolution=512,512 \
  --network_module=networks.lora \
  --enable_bucket \
  --max_train_steps=1 \
  --learning_rate=1e-4 \
  --cache_latents \
  --save_model_as=safetensors
```

## (8) Full Training

This will create numbered checkpoints like lora_models/at-step00002000.safetensors
and also update lora_models/last.safetensors.

```bash
accelerate launch --mixed_precision=bf16 \
  kohya/sd-scripts/train_network.py \
  --pretrained_model_name_or_path="stable-diffusion/stable-diffusion-v1-5" \
  --train_data_dir="training_data/HC_Portraits_r1" \
  --output_dir="lora_models" \
  --resolution=512,512 \
  --network_module=networks.lora \
  --network_dim=16 --network_alpha=16 \
  --enable_bucket --min_bucket_reso=256 --max_bucket_reso=1024 --bucket_reso_steps=64 --bucket_no_upscale \
  --max_train_steps=2000 \
  --learning_rate=1e-4 \
  --cache_latents \
  --save_model_as=safetensors \
  --save_every_n_steps=500 --save_last_n_steps=1
```

## (9) Inference

Script: scripts/inference_sd15_lora.py (LoRA is optional)

```bash
python scripts/inference_sd15_lora.py \
  --base "stable-diffusion/stable-diffusion-v1-5" \
  --lora "lora_models/last.safetensors" \
  --prompt "dramatic high-contrast portrait, rim lighting, 35mm film look" \
  --out "outputs/test_lora.png" \
  --steps 30 --scale 7.5 --width 512 --height 512 --lora_scale 1.0
```

## (10) Batch Generation (10 Images)

Script: scripts/gen_10.sh
Produces 5 LoRA + 5 baseline into outputs/run_<timestamp>/.

```bash
chmod +x scripts/gen_10.sh
./scripts/gen_10.sh
```

## (11) CLIP Scoring

Run scoring on the latest batch:

```bash
RUN_DIR=$(ls -d outputs/run_* | tail -n1)
python tools/clip_score.py \
  --run_dir "$RUN_DIR" \
  --ref_dir "training_data/HC_Portraits_r1/10_my_style" \
  --prompt "dramatic high-contrast portrait, rim lighting, 35mm film look"
```

Outputs:
    outputs/run_.../clip_scores.csv — per-image clip_text + clip_style
    outputs/run_.../clip_summary.json — label means

## Troubleshooting

(1) accelerate: command not found
    Activate venv or pip install accelerate.

(2) “model is not found / path wrong”
    Ensure stable-diffusion/stable-diffusion-v1-5 exists.

(3) Dataset “No data found”
    Point --train_data_dir to the parent of 10_my_style.

(4) AssertionError: image too large …
    Add --enable_bucket (see training command).

(5) PEFT backend is required
    pip install peft. Our inference script only loads LoRA when --lora is passed.
    
(6) Same images across runs
    We used fixed seeds. Remove --seed or randomize for variety.