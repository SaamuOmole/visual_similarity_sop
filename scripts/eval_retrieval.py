# scripts/eval_retrieval.py
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss

def recall_at_k(retrieved_class_ids: np.ndarray, true_class_ids: np.ndarray, k: int) -> float:
    """
    retrieved_class_ids: (Nq, Kmax) class ids of retrieved neighbors
    true_class_ids: (Nq,) true class id for each query
    """
    hits = (retrieved_class_ids[:, :k] == true_class_ids[:, None]).any(axis=1)
    return float(hits.mean())

def main():
    repo_root = Path(__file__).resolve().parents[1]
    art_dir = repo_root / "scripts" / "artifacts" / "sop_embeddings"

    model_name = "convnext_base"  # change if you used another
    emb_path = art_dir / f"emb_{model_name}.npy"
    meta_path = art_dir / f"meta_{model_name}.parquet"

    print("Loading:", emb_path)
    X = np.load(emb_path).astype("float32")  # (N, D)
    meta = pd.read_parquet(meta_path)

    assert len(X) == len(meta), "Embeddings and metadata length mismatch"

    # Split
    train_mask = meta["split"].values == "train"
    test_mask  = meta["split"].values == "test"

    X_train = X[train_mask]
    X_test  = X[test_mask]

    meta_train = meta[train_mask].reset_index(drop=True)
    meta_test  = meta[test_mask].reset_index(drop=True)

    print(f"Train embeddings: {X_train.shape}, Test embeddings: {X_test.shape}")

    # Build FAISS index (cosine similarity via inner product because embeddings were L2-normalized)
    d = X_train.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X_train)

    # Retrieve
    Kmax = 10
    # scores: (Nq, Kmax), idx: (Nq, Kmax) indices into train set
    scores, idx = index.search(X_test, Kmax)

    # Convert retrieved train indices -> class_ids
    train_class_ids = meta_train["class_id"].values.astype(np.int64)
    retrieved_class_ids = train_class_ids[idx]  # (Nq, Kmax)

    true_class_ids = meta_test["class_id"].values.astype(np.int64)

    r1  = recall_at_k(retrieved_class_ids, true_class_ids, 1)
    r5  = recall_at_k(retrieved_class_ids, true_class_ids, 5)
    r10 = recall_at_k(retrieved_class_ids, true_class_ids, 10)

    print(f"Recall@1 : {r1:.4f}")
    print(f"Recall@5 : {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")

    # Save results for debugging
    out_dir = repo_root / "artifacts" / "retrieval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # For each query, store topK paths + scores
    train_paths = meta_train["path"].values
    test_paths = meta_test["path"].values

    rows = []
    for q_i in range(len(meta_test)):
        row = {
            "query_path": test_paths[q_i],
            "query_class_id": int(true_class_ids[q_i]),
            "query_super_class_id": int(meta_test.loc[q_i, "super_class_id"]),
        }
        for k in range(Kmax):
            row[f"rank{k+1}_path"] = train_paths[idx[q_i, k]]
            row[f"rank{k+1}_class_id"] = int(train_class_ids[idx[q_i, k]])
            row[f"rank{k+1}_score"] = float(scores[q_i, k])
        rows.append(row)

    results = pd.DataFrame(rows)
    results_path = out_dir / f"results_{model_name}_K{Kmax}.parquet"
    results.to_parquet(results_path, index=False)
    print("Saved retrieval results:", results_path)

    # Save failure cases (where top5 has no correct match)
    top5_hit = (retrieved_class_ids[:, :5] == true_class_ids[:, None]).any(axis=1)
    failures = results.loc[~top5_hit].reset_index(drop=True)
    fail_path = out_dir / f"failures_{model_name}_top5.parquet"
    failures.to_parquet(fail_path, index=False)
    print("Saved failures:", fail_path)
    print("Num failures (top5):", len(failures))

if __name__ == "__main__":
    main()