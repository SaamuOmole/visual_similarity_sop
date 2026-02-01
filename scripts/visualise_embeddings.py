# scripts/visualise_embeddings.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional (uncomment if installed)
from sklearn.manifold import TSNE
import umap

def plot_2d(Z, labels, title, s=6):
    """Simple 2D scatter colored by integer labels."""
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=s)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.colorbar(sc, label="label")
    plt.tight_layout()
    plt.show()

def main():
    repo_root = Path(__file__).resolve().parent.parent
    art_dir = repo_root / "artifacts" / "sop_embeddings_finetuned_updated"

    # model_name = "convnext_base"  # change if needed
    model_name = "convnext_base_supcon_p256"
    emb_path = art_dir / f"emb_{model_name}.npy"
    meta_path = art_dir / f"meta_{model_name}.parquet"

    print("Loading:", emb_path)
    X = np.load(emb_path).astype("float32")  # (N, D)
    meta = pd.read_parquet(meta_path)
    assert len(X) == len(meta), "Embeddings and metadata length mismatch"

    # --- Choose split to visualize (optional) ---
    split = "test"  # "train" or "test" or None for all
    if split is not None and "split" in meta.columns:
        mask = meta["split"].values == split
        X = X[mask]
        meta = meta[mask].reset_index(drop=True)

    print("Using:", X.shape,", split =", split)

    # --- Subsample for speed/readability ---
    # SOP can be big; plotting all points can be slow and unreadable.
    n_show = min(100000, len(X))  # adjust
    if n_show == len(X):
        print(f"Selecting all {n_show} embeddings")
    else:
        print(f"Selecting {n_show} embeddings...")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n_show, replace=False)
    Xs = X[idx]
    metas = meta.iloc[idx].reset_index(drop=True)

    # Labels to color by (super_class_id is usually easiest)
    y_super = metas["super_class_id"].values.astype(np.int64)

    # Optional: if you want class_id (can be many, plot gets noisy)
    # y_class = metas["class_id"].values.astype(np.int64)

    # --- Dimensionality reduction ---
    # If your embeddings are L2-normalized, PCA still works fine.
    # Standardizing sometimes helps PCA; you can toggle this.
    use_standardize = False
    if use_standardize:
        X_in = StandardScaler().fit_transform(Xs)
    else:
        X_in = Xs

    # 1) PCA to 2D (fast, good first look)
    # pca = PCA(n_components=2, random_state=42)
    # Z_pca = pca.fit_transform(X_in)
    # print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    # plot_2d(Z_pca, y_super, f"{model_name} embeddings (PCA 2D) colored by super_class_id")

    # 2) Optional: t-SNE (slower, can reveal local structure)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    Z_tsne = tsne.fit_transform(Xs)
    plot_2d(Z_tsne, y_super, f"{model_name} embeddings (t-SNE 2D) colored by super_class_id", s=8)

    # 3) Optional: UMAP (often best for embeddings, requires umap-learn)
    # reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    # Z_umap = reducer.fit_transform(Xs)
    # plot_2d(Z_umap, y_super, f"{model_name} embeddings (UMAP 2D) colored by super_class_id", s=8)

if __name__ == "__main__":
    main()
