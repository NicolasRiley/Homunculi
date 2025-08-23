#!/usr/bin/env python3
import argparse, os, json, torch, pandas as pd
from pathlib import Path
from PIL import Image
import open_clip

def load_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    return model, preprocess

def encode_images(model, preprocess, paths, device):
    ims = []
    with torch.no_grad():
        for p in paths:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            ims.append(model.encode_image(img))
    return torch.nn.functional.normalize(torch.cat(ims, dim=0), dim=-1)

def encode_text(model, texts, device):
    with torch.no_grad():
        tok = open_clip.tokenize(texts).to(device)
        feats = model.encode_text(tok)
    return torch.nn.functional.normalize(feats, dim=-1)

def main():
    ap = argparse.ArgumentParser(description="CLIP scoring for Homunculi runs")
    ap.add_argument("--run_dir", required=True, help="outputs/run_YYYYMMDD_HHMMSS")
    ap.add_argument("--ref_dir", required=True, help="training_data/.../10_my_style (or any style refs)")
    ap.add_argument("--prompt", required=True, help="prompt used to generate images")
    ap.add_argument("--pattern", default="*.png", help="glob for images in run_dir")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(device)

    run_dir = Path(args.run_dir)
    ref_dir = Path(args.ref_dir)
    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "clip_scores.csv"

    # Collect images
    imgs = sorted([p for p in run_dir.glob(args.pattern) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    refs = sorted([p for p in ref_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    assert imgs, f"No images found in {run_dir}"
    assert refs, f"No reference images found in {ref_dir}"

    # Encode
    img_feats = encode_images(model, preprocess, imgs, device)
    ref_feats = encode_images(model, preprocess, refs, device)
    text_feat = encode_text(model, [args.prompt], device)[0].unsqueeze(0)

    # Compute cosine sims
    with torch.no_grad():
        sim_text = (img_feats @ text_feat.T).squeeze(1).cpu().numpy()          # image vs prompt
        sim_style = (img_feats @ ref_feats.T).cpu()                             # image vs each ref
        sim_style_mean = sim_style.mean(dim=1).numpy()                          # average over refs

    # Tag baseline vs lora by filename prefix
    rows = []
    for i, p in enumerate(imgs):
        label = "lora" if p.name.startswith("lora_") else ("baseline" if p.name.startswith("baseline_") else "unknown")
        rows.append({"image": p.name, "label": label, "clip_text": float(sim_text[i]), "clip_style": float(sim_style_mean[i])})

    df = pd.DataFrame(rows).sort_values(["label", "image"])
    df.to_csv(out_csv, index=False)

    # Small JSON summary
    summary = {
        "n_images": len(imgs),
        "prompt": args.prompt,
        "means": df.groupby("label")[["clip_text", "clip_style"]].mean().round(4).to_dict(),
        "csv": str(out_csv),
    }
    (run_dir / "clip_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_csv}\nSummary: {run_dir/'clip_summary.json'}")
    print(summary)

if __name__ == "__main__":
    main()
