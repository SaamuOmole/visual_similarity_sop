from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector

def main():
    repo_root = Path(__file__).resolve().parent.parent
    art_dir = repo_root / "artifacts" / "sop_embeddings_finetuned_updated"
    out_dir = repo_root / "artifacts" / "tensorboard_embeddings_finetuned_updated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # model_name = "convnext_base"  # change if needed
    model_name = "convnext_base_supcon_p256"
    emb_path = art_dir / f"emb_{model_name}.npy"
    meta_path = art_dir / f"meta_{model_name}.parquet"

    X = np.load(emb_path).astype("float32")        # (N, D)
    meta = pd.read_parquet(meta_path)              # must align with X
    assert len(X) == len(meta), "Embeddings and metadata length mismatch"

    # Subsample so projector stays responsive
    n = min(100000, len(X))
    if n == len(X):
        print(f"Selecting all {n} embeddings")
    else:
        print(f"Selecting {n} embeddings...")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n, replace=False)
    X = X[idx]
    meta = meta.iloc[idx].reset_index(drop=True)
    
    # 1) Create a TF variable containing embeddings
    embedding_var = tf.Variable(X, name="embeddings")

    # 2) Save embeddings as a checkpoint
    ckpt = tf.train.Checkpoint(embeddings=embedding_var)
    ckpt_prefix = str(out_dir / "embeddings.ckpt")
    ckpt.save(ckpt_prefix)

    # 3) Write metadata TSV
    meta_cols = [c for c in ["path", "class_id", "super_class_id", "split"] if c in meta.columns]
    meta_tsv = out_dir / "metadata.tsv"
    meta[meta_cols].to_csv(meta_tsv, sep="\t", index=False)

    # 4) Create projector config
    config = projector.ProjectorConfig()
    emb = config.embeddings.add()
    # IMPORTANT: variable name must match TF variable name exactly
    emb.tensor_name = "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    emb.metadata_path = meta_tsv.name  # relative to out_dir

    projector.visualize_embeddings(out_dir, config)

    # 5) Write at least one event file so TensorBoard sees "a run"
    # (This avoids “No dashboards are active” in many cases.)
    writer = tf.summary.create_file_writer(str(out_dir))
    with writer.as_default():
        tf.summary.scalar("dummy", 1.0, step=0)
        writer.flush()

    print("Exported TensorBoard projector files to:", out_dir)
    print("\nRun:")
    print(f"  tensorboard --logdir {out_dir}")
    print("\nThen open TensorBoard and go to:")
    print("  Projector")

if __name__ == "__main__":
    main()
