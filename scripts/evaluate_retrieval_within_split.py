from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss

def recall_at_k(retrieved_class_ids: np.ndarray, true_class_ids: np.ndarray, k: int) -> float:
    hits = (retrieved_class_ids[:, :k] == true_class_ids[:, None]).any(axis=1)
    return float(hits.mean())

def main():
    repo_root = Path(__file__).resolve().parent.parent
    # art_dir = repo_root / "artifacts" / "sop_embeddings_finetuned"
    art_dir = repo_root / "artifacts" / "sop_embeddings"

    # model_name = "convnext_base_supcon_p256"
    model_name = "convnext_base"
    emb_path = art_dir / f"emb_{model_name}.npy"
    meta_path = art_dir / f"meta_{model_name}.parquet"

    X = np.load(emb_path).astype("float32")
    meta = pd.read_parquet(meta_path)
    assert len(X) == len(meta)

    # Choose split to evaluate (train or test)
    split = "test"   # change to "test" or "train" to evaluate within-test or within-train
    metric = "class_id" # can change to either "class_id" or "super_class_id"
    mask = (meta["split"].values == split)

    Xs = X[mask]
    metas = meta[mask].reset_index(drop=True)

    print(f"Evaluating within-{split}: {Xs.shape}")

    # Build FAISS index
    d = Xs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(Xs)

    class_ids = metas[metric].values.astype(np.int64)
    paths = metas["path"].values

    K = 10
    scores, idx = index.search(Xs, K + 1)  # includes self
    scores = scores[:, 1:]  # drop self
    idx = idx[:, 1:]

    retrieved_class_ids = class_ids[idx]
    true_class_ids = class_ids

    r1 = recall_at_k(retrieved_class_ids, true_class_ids, 1)
    r5 = recall_at_k(retrieved_class_ids, true_class_ids, 5)
    r10 = recall_at_k(retrieved_class_ids, true_class_ids, 10)

    print(f"\nReturning metric for {metric}:")
    print(f"\nWithin-{split} Recall@1 : {r1:.4f}")
    print(f"Within-{split} Recall@5 : {r5:.4f}")
    print(f"Within-{split} Recall@10: {r10:.4f}\n")

    # Save per-query results
    # out_dir = repo_root / "artifacts" / "retrieval_results_finetuned"
    out_dir = repo_root / "artifacts" / "retrieval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for q_i in range(len(metas)):
        row = {
            "query_idx": q_i,
            "query_path": paths[q_i],
            "query_class_id": int(true_class_ids[q_i]),
            "query_super_class_id": int(metas.loc[q_i, "super_class_id"]),
        }
        for k in range(K):
            j = idx[q_i, k]
            row[f"rank{k+1}_path"] = paths[j]
            row[f"rank{k+1}_class_id"] = int(class_ids[j])
            row[f"rank{k+1}_score"] = float(scores[q_i, k])
        rows.append(row)

    results = pd.DataFrame(rows)
    results_path = out_dir / f"results_{model_name}_{split}_K{K}.parquet"
    results.to_parquet(results_path, index=False)
    print("Saved retrieval results:", results_path)

    # Failures (no correct class in top5)
    top5_hit = (retrieved_class_ids[:, :5] == true_class_ids[:, None]).any(axis=1)
    failures = results.loc[~top5_hit].reset_index(drop=True)
    fail_path = out_dir / f"failures_{model_name}_{split}_top5.parquet"
    failures.to_parquet(fail_path, index=False)

    print("\nSaved failures:", fail_path)
    print("Num failures (top5):", len(failures))

if __name__ == "__main__":
    main()