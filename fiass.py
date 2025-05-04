import argparse, json, faiss, numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def build_from_json(json_path: Path,
                    out_dir: Path,
                    item_key: Optional[str] = None,
                    max_rows: Optional[int] = None):

    meta = json.loads(json_path.read_text())
    raw_items = meta["data"]
    if item_key is not None:
        raw_items = {item_key: raw_items[item_key]}

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (key, v) in enumerate(raw_items.items()):
        vid = v["videoID"]
        if "frames" not in v or "caption" not in v:
            print(f"[WARN] video {vid} missing 'frames' or 'caption' — skipping.")
            continue

        frames   = v["frames"][:max_rows] if max_rows else v["frames"]
        captions = v["caption"][:max_rows] if max_rows else v["caption"]

        print(f"\n▶ building {vid} (frames={len(frames)}  caps={len(captions)})")
        
        img_path = Path(v["image_feature_path"])
        txt_path = Path(v["caption_feature_path"])

        if not img_path.exists() or not txt_path.exists():
            print(f"[SKIP] Features not found for videoID={vid}")
            continue

        img_feat_all = np.load(v["image_feature_path"]).astype("float32")
        txt_feat_all = np.load(v["caption_feature_path"]).astype("float32")

        img_vecs, cap_vecs, meta_list = [], [], []

        for fr in tqdm(frames, desc="  img"):
            idx = fr["clip_idx"]
            feat = img_feat_all[idx:idx+1]
            img_vecs.append(feat)
            meta_list.append({
                "tag": "img", "videoID": vid,
                **{k: fr.get(k) for k in ("clip_idx", "timestamp", "frame_name")}
            })

        for cp in tqdm(captions, desc="  txt"):
            cid = cp["id"]
            feat = txt_feat_all[cid:cid+1]
            cap_vecs.append(feat)
            meta_list.append({
                "tag": "txt", "videoID": vid,
                **{k: cp.get(k) for k in ("id", "start", "end")}
            })

        img_vec = l2norm(np.vstack(img_vecs))
        cap_vec = l2norm(np.vstack(cap_vecs))

        all_vec = np.concatenate([img_vec, cap_vec], axis=0)
        index = faiss.IndexFlatIP(all_vec.shape[1])
        index.add(all_vec)

        faiss.write_index(index, str(out_dir / f"{vid}.index"))
        (out_dir / f"{vid}.json").write_text(json.dumps(meta_list, ensure_ascii=False, indent=2))

        print(f"✔︎ saved: {vid}.index / {vid}.json")
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",    required=True)
    ap.add_argument("--out_dir", default="joint_index_out")
    ap.add_argument("--item",    help="item key", default=None)
    ap.add_argument("--max_rows", type=int, help="N  frame / caption")
    args = ap.parse_args()

    build_from_json(Path(args.json), Path(args.out_dir),
                    args.item, args.max_rows)