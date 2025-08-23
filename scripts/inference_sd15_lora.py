#!/usr/bin/env python3
import argparse, torch, random
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from pathlib import Path

p = argparse.ArgumentParser(description="SD 1.5 + optional LoRA inference")
p.add_argument("--base", required=True)
p.add_argument("--lora", default=None)           # optional now
p.add_argument("--prompt", required=True)
p.add_argument("--out", default="output.png")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--scale", type=float, default=7.5)
p.add_argument("--width", type=int, default=512)
p.add_argument("--height", type=int, default=512)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--lora_scale", type=float, default=1.0)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
if args.seed is None:
    args.seed = random.randint(0, 2**31 - 1)
gen = torch.Generator(device=device).manual_seed(args.seed)

pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=dtype)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass
pipe = pipe.to(device)

# Load LoRA only if provided
if args.lora:
    pipe.load_lora_weights(args.lora)
    try:
        pipe.fuse_lora(lora_scale=args.lora_scale)
    except Exception:
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["default"], adapter_weights=[args.lora_scale])

image = pipe(
    prompt=args.prompt,
    num_inference_steps=args.steps,
    guidance_scale=args.scale,
    width=args.width,
    height=args.height,
    generator=gen,
).images[0]

out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
image.save(out)
print(f"Saved {out} (seed={args.seed})  lora={'None' if not args.lora else args.lora}")
