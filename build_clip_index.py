import os, json, argparse, glob, warnings
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


META_DIR   = Path("./YTComment_frame_metadata")
FRAME_DIR  = Path("./YTComment_frame")
FEAT_DIR   = Path("./YTComment_clip_features")
INDEX_FILE = Path("./clip_feature_index.json")

FEAT_DIR.mkdir(parents=True, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = None, None
try:
    from transformers import CLIPProcessor, CLIPModel
    model_name = "openai/clip-vit-base-patch32"
    processor   = CLIPProcessor.from_pretrained(model_name)
    model       = CLIPModel.from_pretrained(model_name).to(device).eval()
except Exception as e:
    raise RuntimeError("🤖  Cannot load CLIP; install `transformers` & `sentencepiece`") from e



def load_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        warnings.warn(f"⚠️  Bad image skipped: {path}")
        return None



def process_one_video(meta_path: Path, batch_size: int = 512) -> Dict:
    meta = json.load(meta_path.open())
    vid  = meta["videoID"]
    frames = meta["frames"]
    img_dir = FRAME_DIR / vid

    images, kept_frames = [], []

    for f in frames:
        fp = img_dir / f["frame_name"]
        if fp.exists():
            img = load_image(fp)
            if img:
                images.append(img)
                kept_frames.append(f["frame_name"])
        else:
            warnings.warn(f"Missing jpg ⇒ {fp}")

    if not images:
        warnings.warn(f"No frames for {vid}; skipping.")
        return {}

    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size),
                      desc=f"CLIP {vid}", leave=False):
            batch_imgs = images[i : i + batch_size]
            inputs = processor(images=batch_imgs,
                               return_tensors="pt", padding=True).to(device)
            feats  = model.get_image_features(**inputs)
            feats  = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())

    feats = np.concatenate(all_feats, axis=0)
    out_file = FEAT_DIR / f"{vid}.npy"
    np.save(out_file, feats)

    return {
        "videoID": vid,
        "feature_path": str(out_file),
        "frame_names": kept_frames,
        "feat_shape": feats.shape
    }



def main(args):
    meta_files = sorted(META_DIR.glob("*.json"))
    if args.limit:
        meta_files = meta_files[: args.limit]


    if INDEX_FILE.exists():
        with INDEX_FILE.open() as f:
            index_data = {d["videoID"]: d for d in json.load(f)}
    else:
        index_data = {}

    for mf in tqdm(meta_files, desc="Videos"):
        vid = mf.stem.replace("_frames", "")
        if vid in index_data and not args.overwrite:
            continue
        entry = process_one_video(mf, batch_size=args.batch)
        if entry:
            index_data[vid] = entry

    with INDEX_FILE.open("w") as f:
        json.dump(list(index_data.values()), f, indent=2)
    print(f"Index written to {INDEX_FILE}   ({len(index_data)} videos)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64,
                        help="mini-batch size for CLIP forward pass")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only first N videos (debug)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-extract even if feature file already exists")
    args = parser.parse_args()
    main(args)