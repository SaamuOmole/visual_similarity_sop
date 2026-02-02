from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import faiss


# -------------------------
# Model definition (must match training)
# -------------------------
class EmbedNet(nn.Module):
    def __init__(self, backbone_name="convnext_base", proj_dim=256):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        backbone_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, proj_dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.proj(h)
        return F.normalize(z, dim=1)


def show_query_and_results(query_path, result_paths, scores, topk=5):
    plt.figure(figsize=(2*(topk+1), 3))

    qimg = Image.open(query_path).convert("RGB")
    plt.subplot(1, topk+1, 1)
    plt.imshow(qimg)
    plt.title("QUERY")
    plt.axis("off")

    for i in range(topk):
        img = Image.open(result_paths[i]).convert("RGB")
        plt.subplot(1, topk+1, i+2)
        plt.imshow(img)
        plt.title(f"rank{i+1}\n{scores[i]:.3f}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    repo_root = Path(__file__).resolve().parent.parent

    # ---- Configure paths ----
    queries_dir = repo_root / "data" / "real_world_queries" / "lamp" # Change path to query images
    assert queries_dir.exists(), f"Missing: {queries_dir}"

    # Use your fine-tuned embeddings + metadata
    emb_dir = repo_root / "artifacts" / "sop_embeddings_finetuned_updated"
    emb_path = emb_dir / "emb_convnext_base_supcon_p256.npy"
    meta_path = emb_dir / "meta_convnext_base_supcon_p256.parquet"

    # Use your best checkpoint (early-stopped/best)
    ckpt_path = repo_root / "artifacts" / "checkpoints" / "sop_supcon_convnext_updated.pt"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    # ---- Device ----
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Device:", device)

    # ---- Load catalog embeddings + metadata ----
    X = np.load(emb_path).astype("float32")
    meta = pd.read_parquet(meta_path)
    assert len(X) == len(meta)

    # catalog = train split only
    train_mask = meta["split"].values == "train"
    X_cat = X[train_mask]
    meta_cat = meta[train_mask].reset_index(drop=True)

    print("Catalog embeddings:", X_cat.shape)

    # ---- Build FAISS index ----
    d = X_cat.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity via inner product (because normalized)
    index.add(X_cat)

    # ---- Load model ----
    model = EmbedNet("convnext_base", proj_dim=256).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)

    # support either {"model": state_dict} or raw state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    # ---- Preprocessing (must match extraction) ----
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    # ---- Query all images in folder ----
    query_paths = sorted([p for p in queries_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print("Num queries:", len(query_paths))
    if not query_paths:
        print(f"No images found in {queries_dir}")
        return

    topk = 5
    for qp in query_paths:
        # embed query
        img = Image.open(qp).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            z = model(x).cpu().numpy().astype("float32")  # (1, 256)

        scores, idx = index.search(z, topk)
        idx = idx[0]
        scores = scores[0]

        result_paths = meta_cat.iloc[idx]["path"].tolist()

        print(f"\nQuery: {qp.name}")
        for r, s in zip(result_paths, scores):
            print(f"  {s:.3f}  {r}")

        show_query_and_results(qp, result_paths, scores, topk=topk)

if __name__ == "__main__":
    main()
