import argparse, glob, json, warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
import torch
from transformers import (CLIPProcessor, CLIPModel,
                          CLIPTokenizerFast)



DEFAULT_OUTDIR = Path("./YTComment_query_vectors")
DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)


def load_questions(json_path: Path) -> List[Tuple[Dict, str]]:

    with json_path.open() as f:
        data = json.load(f)

    items = []
    for k, v in data["data"].items():
        q = v.get("question", "").strip()
        if not q:
            warnings.warn(f"{json_path} - key {k} has empty question")
            continue

        meta = {
            "sample_id": f"{json_path.stem}##{k}", 
            "videoID":   v.get("videoID", ""),
            "q_idx":     int(k), 
            "question":  q
        }
        items.append((meta, q))
    return items


class CLIPTextEncoder:

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model     = CLIPModel.from_pretrained(model_name)\
                                  .to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encode-text", ncols=100):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt",
                                    padding=True, truncation=True)\
                                    .to(self.device)
            feats  = self.model.get_text_features(**inputs)
            feats  = torch.nn.functional.normalize(feats, dim=-1)
            out.append(feats.cpu().float())
        return torch.cat(out, dim=0).numpy()        # (N,512) float32




def main(args):
    json_files = sorted(glob.glob(args.input_glob))
    if not json_files:
        raise FileNotFoundError(f"No file matches: {args.input_glob}")

    meta_rows: List[Dict] = []
    all_questions: List[str] = []

    for jf in json_files:
        for meta, q in load_questions(Path(jf)):
            meta_rows.append(meta)
            all_questions.append(q)

    print(f"🗂️  Collected {len(all_questions)} questions "
          f"from {len(json_files)} json files")

    encoder = CLIPTextEncoder(device=args.device)
    vecs = encoder.encode(all_questions, batch_size=args.batch)  # (N,512)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_path = out_dir / "query_embeddings.npy"
    id_path  = out_dir / "id_map.json"

    np.save(vec_path, vecs)
    with id_path.open("w") as f:
        json.dump(meta_rows, f, ensure_ascii=False, indent=2)

    print(f"vectors -> {vec_path}   shape={vecs.shape}")
    print(f"id_map  -> {id_path}    rows={len(meta_rows)}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_glob", required=True,
                   help="dataset.json")
    p.add_argument("--out_dir", default=str(DEFAULT_OUTDIR),
                   help="output directory")
    p.add_argument("--batch", type=int, default=64,
                   help="batch size for CLIP encoding")
    p.add_argument("--device", type=str, default=None,
                   help="'cpu' or 'cuda'")
    args = p.parse_args()
    main(args)