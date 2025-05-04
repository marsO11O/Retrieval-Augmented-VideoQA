import json
import argparse
from pathlib import Path
from typing import List
import faiss
import numpy as np
from tqdm import tqdm
import re


TOPK_TXT = 15
TOPK_IMG = 15
T_WINDOW = 12.0
ALPHA = 0.98
BETA = 2.95
GAMMA = 0.045
TOPN_FINAL = 5

def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def time_of(m: dict) -> float:
    return float(m["timestamp"]) if m["tag"] == "img" else (m["start"] + m["end"]) / 2

class VideoIndexer:
    def __init__(self, index_path: Path, meta_path: Path):
        self.videoID = index_path.stem
        self.meta = json.loads(meta_path.read_text())
        index = faiss.read_index(str(index_path))
        vecs = np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype=np.float32)
        vecs = l2norm(vecs)

        self.txt_vecs, self.txt_meta = [], []
        self.img_vecs, self.img_meta = [], []
        for vec, m in zip(vecs, self.meta):
            if m["tag"] == "txt":
                self.txt_vecs.append(vec)
                self.txt_meta.append(m)
            elif m["tag"] == "img":
                self.img_vecs.append(vec)
                self.img_meta.append(m)

        self.txt_vecs = np.array(self.txt_vecs)
        self.img_vecs = np.array(self.img_vecs)

        self.txt_index = faiss.IndexFlatIP(self.txt_vecs.shape[1])
        self.txt_index.add(self.txt_vecs)
        self.img_index = faiss.IndexFlatIP(self.img_vecs.shape[1])
        self.img_index.add(self.img_vecs)

    def _search_txt_img(self, q: np.ndarray):
        txt_scores, txt_ids = self.txt_index.search(q, TOPK_TXT * 2)
        img_scores, img_ids = self.img_index.search(q, TOPK_IMG * 2)

        txt_hits = [{"score": float(sc), "meta": self.txt_meta[idx], "tag": "txt"}
                    for sc, idx in zip(txt_scores[0], txt_ids[0])][:TOPK_TXT]
        img_hits = [{"score": float(sc), "meta": self.img_meta[idx], "tag": "img"}
                    for sc, idx in zip(img_scores[0], img_ids[0])][:TOPK_IMG]
        return txt_hits, img_hits

    def rerank(self, qid: int, qvec: np.ndarray):
        q = qvec[qid:qid+1]
        txt_hits, img_hits = self._search_txt_img(q)
        for h in txt_hits:
            h["t"] = time_of(h["meta"])
        for h in img_hits:
            h["t"] = time_of(h["meta"])

        results = []
        for th in txt_hits:
            partners = [ih for ih in img_hits if abs(ih["t"] - th["t"]) <= T_WINDOW]
            if partners:
                ih = min(partners, key=lambda x: abs(x["t"] - th["t"]))
                dt = abs(ih["t"] - th["t"])
                time_bonus = (T_WINDOW - dt) / T_WINDOW
                #final = ALPHA * th["score"] + BETA * ih["score"] + GAMMA * time_bonus
                final = ALPHA * th["score"] + BETA * ih["score"] - GAMMA * (dt / T_WINDOW)
                img_s = ih["score"]
            else:
                final = ALPHA * th["score"]
                img_s, dt = None, None

            results.append({
                "cap_id": th["meta"].get("id", "-"),
                "final": final,
                "txt_score": th["score"],
                "img_score": img_s,
                "dt": dt,
                "meta": th["meta"]
            })

        results.sort(key=lambda x: x["final"], reverse=True)
        return results[:TOPN_FINAL]

def main(args):
    query_vec = l2norm(np.load(args.query_vec).astype("float32"))
    query_map = json.loads(Path(args.query_map).read_text())
    qid2text = {r["q_idx"]: r["question"] for r in query_map}


    for index_file in sorted(args.index_dir.glob("*.index")):
        vid = index_file.stem  # videoID
        meta_file = args.index_dir / f"{vid}.json"

        if not meta_file.exists():
            print(f"[WARNING] Can't find meta：{meta_file}")
            continue


        index = faiss.read_index(str(index_file))
        meta = json.loads(meta_file.read_text())
        vecs = [index.reconstruct(i) for i in range(index.ntotal)]
        vecs = l2norm(np.array(vecs, dtype=np.float32))

        txt_vecs, txt_meta, img_vecs, img_meta = [], [], [], []
        for v, m in zip(vecs, meta):
            if m["tag"] == "txt":
                txt_vecs.append(v)
                txt_meta.append(m)
            elif m["tag"] == "img":
                img_vecs.append(v)
                img_meta.append(m)

        if not txt_vecs or not img_vecs:
            continue

        txt_vecs = np.array(txt_vecs)
        img_vecs = np.array(img_vecs)

        txt_index = faiss.IndexFlatIP(txt_vecs.shape[1])
        img_index = faiss.IndexFlatIP(img_vecs.shape[1])
        txt_index.add(txt_vecs)
        img_index.add(img_vecs)

        qids = [r["q_idx"] for r in query_map if r["videoID"] == vid]
        if not qids:
            continue

        print(f"\n=== videoID={vid} ===\n")
        for qid in qids:
            print(f"[Q{qid}] {qid2text[qid]}")

            q = query_vec[qid:qid+1]
            txt_scores, txt_ids = txt_index.search(q, TOPK_TXT * 2)
            img_scores, img_ids = img_index.search(q, TOPK_IMG * 2)

            # Top-K TXT
            txt_hits = []
            for sc, idx in zip(txt_scores[0], txt_ids[0]):
                m = txt_meta[idx]
                txt_hits.append({"score": float(sc), "meta": m, "tag": "txt"})
                if len(txt_hits) >= TOPK_TXT:
                    break

            # Top-K IMG
            img_hits = []
            for sc, idx in zip(img_scores[0], img_ids[0]):
                m = img_meta[idx]
                img_hits.append({"score": float(sc), "meta": m, "tag": "img"})
                if len(img_hits) >= TOPK_IMG:
                    break

            #
            # print("── Top-K TXT ─────────────────────────")
            # for i, h in enumerate(txt_hits, start=1):
            #     t = time_of(h["meta"])
            #     print(f"{i:2d}. cap_id={h['meta']['id']:>4}  t={t:6.1f}s  score={h['score']:.3f}")

            # print("── Top-K IMG ─────────────────────────")
            # for i, h in enumerate(img_hits, start=1):
            #     t = time_of(h["meta"])
            #     print(f"{i:2d}. img_id={h['meta']['clip_idx']:>4} t={t:6.1f}s  score={h['score']:.3f}")

            # Rerank
            for h in txt_hits:
                h["t"] = time_of(h["meta"])
            for h in img_hits:
                h["t"] = time_of(h["meta"])

            results = []
            for th in txt_hits:
                partners = [ih for ih in img_hits if abs(ih["t"] - th["t"]) <= T_WINDOW]
                if partners:
                    ih = min(partners, key=lambda x: abs(x["t"] - th["t"]))
                    dt = abs(ih["t"] - th["t"])
                    time_bonus = (T_WINDOW - dt) / T_WINDOW
                    final = ALPHA * th["score"] + BETA * ih["score"] + GAMMA * time_bonus
                    img_s = ih["score"]
                else:
                    final = ALPHA * th["score"]
                    img_s, dt = None, None

                results.append({
                    "cap_id": th["meta"]["id"],
                    "final": final,
                    "txt_score": th["score"],
                    "img_score": img_s,
                    "dt": dt,
                    "meta": th["meta"]
                })

            results.sort(key=lambda x: x["final"], reverse=True)
            print("── Rerank (TXT+IMG) ──────────────────")
            for r in results[:TOPN_FINAL]:
                if r["img_score"] is not None:
                    print(f"→ cap_id={r['cap_id']:>4} final={r['final']:.3f} "
                          f"txt={r['txt_score']:.3f} img={r['img_score']:.3f} dt={r['dt']:.2f}s")
                else:
                    print(f"→ cap_id={r['cap_id']:>4} final={r['final']:.3f} "
                          f"txt={r['txt_score']:.3f} no_img")
            print()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, type=Path)
    ap.add_argument("--query_vec", required=True, type=Path)
    ap.add_argument("--query_map", required=True, type=Path)
    args = ap.parse_args()
    main(args)
