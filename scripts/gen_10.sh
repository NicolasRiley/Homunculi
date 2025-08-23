# scripts/gen_10.sh
#!/usr/bin/env bash
set -euo pipefail

BASE="stable-diffusion/stable-diffusion-v1-5"
LORA="lora_models/at-step00002000.safetensors"   # or lora_models/last.safetensors
OUTDIR="outputs/run_$(date +%Y%m%d_%H%M%S)"
PROMPT='dramatic high-contrast portrait, rim lighting, 35mm film look'
NEG='blurry, low quality, watermark'

mkdir -p "$OUTDIR"

# 5 with LoRA (fixed seeds for comparability), 5 baseline (random seeds for variety)
for i in $(seq 0 9); do
  if [ $i -lt 5 ]; then
    SEED=$((12345 + i))
    python scripts/inference_sd15_lora.py \
      --base "$BASE" --lora "$LORA" \
      --prompt "$PROMPT" --out "$OUTDIR/lora_${i}.png" \
      --steps 30 --scale 7.5 --width 512 --height 512 --lora_scale 1.0 --seed "$SEED"
  else
    SEED=$(python - <<'PY'
import random; print(random.randrange(1<<31))
PY
)
    python scripts/inference_sd15_lora.py \
      --base "$BASE" \
      --prompt "$PROMPT" --out "$OUTDIR/baseline_${i-5}.png" \
      --steps 30 --scale 7.5 --width 512 --height 512 --seed "$SEED"
  fi
done

# Write a small manifest for later scoring
cat > "$OUTDIR/manifest.json" <<EOF
{
  "base": "$BASE",
  "lora": "$LORA",
  "prompt": "$PROMPT",
  "negative": "$NEG",
  "count_lora": 5,
  "count_baseline": 5
}
EOF

echo "Wrote images + manifest to $OUTDIR"
