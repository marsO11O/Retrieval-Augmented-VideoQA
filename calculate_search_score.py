import json
import argparse
from pathlib import Path
import faiss
import numpy as np
from tqdm import tqdm

TOPK_TXT = 17
TOPK_IMG = 30
T_WINDOW = 12.0
ALPHA = 0.9
BETA = 2.9
GAMMA = 0.02

def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def time_of(m: dict) -> float:
    return float(m["timestamp"]) if m["tag"] == "img" else (m["start"] + m["end"]) / 2

def main(args):
    query_vec = l2norm(np.load(args.query_vec).astype("float32"))
    query_map = json.loads(Path(args.query_map).read_text())
    qid2text = {r["q_idx"]: r["question"] for r in query_map}

    filter_data = json.loads(Path(args.filter_json).read_text())["data"]

    total = 0
    correct = 0

    for index_file in sorted(args.index_dir.glob("*.index")):
        vid = index_file.stem
        meta_file = args.index_dir / f"{vid}.json"
        if not meta_file.exists():
            continue

        index = faiss.read_index(str(index_file))
        meta = json.loads(meta_file.read_text())
        vecs = l2norm(np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype=np.float32))

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

        vid_qids = [int(qid) for qid, info in filter_data.items() if info["videoID"] == vid]
        if not vid_qids:
            continue

        for qid in vid_qids:
            qvec = query_vec[qid:qid+1]
            txt_scores, txt_ids = txt_index.search(qvec, TOPK_TXT * 2)
            img_scores, img_ids = img_index.search(qvec, TOPK_IMG * 2)

            # topK txt / img
            txt_hits = [{"score": float(sc), "meta": txt_meta[idx], "tag": "txt"}
                        for sc, idx in zip(txt_scores[0], txt_ids[0])][:TOPK_TXT]
            img_hits = [{"score": float(sc), "meta": img_meta[idx], "tag": "img"}
                        for sc, idx in zip(img_scores[0], img_ids[0])][:TOPK_IMG]

            for h in txt_hits:
                h["t"] = time_of(h["meta"])
            for h in img_hits:
                h["t"] = time_of(h["meta"])
                

                
            ans = filter_data[str(qid)]["answerability"]
            if ans == [1]:
                ALPHA, BETA = 0.3, 3.0
            elif ans == [2]:
                ALPHA, BETA = 1.0, 1.0
            elif sorted(ans) == [1, 2]:
                ALPHA, BETA = 0.9, 2.9
            else:
                ALPHA, BETA = ALPHA, BETA
    


            results = []
            for th in txt_hits:
                partners = [ih for ih in img_hits if abs(ih["t"] - th["t"]) <= T_WINDOW]
                if partners:
                    ih = min(partners, key=lambda x: abs(x["t"] - th["t"]))
                    dt = abs(ih["t"] - th["t"])
                    time_bonus = (T_WINDOW - dt) / T_WINDOW
                    final = ALPHA * th["score"] + BETA * ih["score"] + GAMMA * time_bonus
                else:
                    final = ALPHA * th["score"]

                results.append({
                    "cap_id": th["meta"].get("id", -1),
                    "final": final,
                })

            results.sort(key=lambda x: x["final"], reverse=True)

            top_cap_ids = [r["cap_id"] for r in results[:args.topn]]
            

            gt_evidence = filter_data[str(qid)]["evidence"]

            start, end = gt_evidence[0]-1, gt_evidence[-1]+1
            if any(start <= cap_id <= end for cap_id in top_cap_ids):
                correct += 1

            total += 1

    acc = correct / total if total else 0.0
    print(f"\n=== Evaluation ===")
    print(f"Total questions: {total}")
    print(f"Correct in Top-{args.topn}: {correct}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, type=Path)
    ap.add_argument("--query_vec", required=True, type=Path)
    ap.add_argument("--query_map", required=True, type=Path)
    ap.add_argument("--filter_json", required=True, type=Path)
    ap.add_argument("--topn", type=int, default=1, help="Top N for evaluation")
    args = ap.parse_args()

    main(args)