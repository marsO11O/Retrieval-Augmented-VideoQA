import argparse, json, os
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract CLIP text features for all captions in JSON")
    ap.add_argument("--json_path", required=True,
                    help="metadata produced by frame extractor (*_frames.json)")
    ap.add_argument("--out_meta_root", default="./YTComment_caption_metadata",
                    help="where to save updated json with feature paths")
    ap.add_argument("--out_npy_root",  default="./YTComment_caption_features",
                    help="where to dump *.npy files (one per video)")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="mini-batch size for CLIP forward pass")
    return ap.parse_args()


@torch.no_grad()
def encode_text_batch(texts: List[str],
                      processor: CLIPProcessor,
                      model: CLIPModel,
                      device: torch.device,
                      batch: int = 256) -> np.ndarray:
    """Vectorize a list of strings → L2-normed CLIP text features (np.float32)."""
    out_vecs = []
    for i in range(0, len(texts), batch):
        inp = processor(text=texts[i:i + batch],
                        return_tensors="pt",
                        padding=True,
                        truncation=True).to(device)
        feat = model.get_text_features(**inp)              # (B,512)
        feat = torch.nn.functional.normalize(feat, dim=-1) # L2 norm
        out_vecs.append(feat.cpu())
    return torch.cat(out_vecs, dim=0).numpy().astype("float32")


def main() -> None:
    args = parse_args()

    Path(args.out_meta_root).mkdir(parents=True, exist_ok=True)
    Path(args.out_npy_root).mkdir(parents=True,  exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    clip      = CLIPModel.from_pretrained(model_id).eval().to(device)


    with open(args.json_path, encoding="utf-8") as f:
        meta = json.load(f)

    for item_key, item in tqdm(meta["data"].items(), desc="Videos"):
        vid      = item["videoID"]
        captions = item["caption"]
        if not captions:
            print(f"[WARN] {vid} has no captions, skip.")
            continue

        texts = [cap["text"] for cap in captions]
        feats = encode_text_batch(texts, processor, clip,
                                  device, args.batch_size)

        npy_path = Path(args.out_npy_root) / f"{vid}.npy"
        np.save(npy_path, feats)

        item["caption_feature_path"] = str(npy_path)
        img_feat = Path("./YTComment_clip_features") / f"{vid}.npy"
        if img_feat.exists():
            item["image_feature_path"] = str(img_feat)

    out_json = Path(args.out_meta_root) / \
               (Path(args.json_path).stem + "_with_clip.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Updated metadata -> {out_json}")


if __name__ == "__main__":
    main()