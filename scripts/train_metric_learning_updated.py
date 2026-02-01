from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import timm
from tqdm import tqdm

from pytorch_metric_learning.losses import SupConLoss
import faiss


# -------------------------
# Dataset
# -------------------------
class SOPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        x = self.transform(img)
        y = int(row.class_id)
        return x, y


# -------------------------
# PK Sampler: P classes, K samples each
# -------------------------
class PKSampler(Sampler):
    def __init__(self, labels, P=16, K=2, steps_per_epoch=1000, seed=42):
        self.labels = np.array(labels)
        self.P = P
        self.K = K
        self.steps_per_epoch = steps_per_epoch
        self.rng = random.Random(seed)

        self.class_to_indices = {}
        for i, c in enumerate(self.labels):
            self.class_to_indices.setdefault(int(c), []).append(i)

        # keep only classes that have at least K examples
        self.classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) >= K]
        if len(self.classes) < P:
            raise ValueError(f"Not enough classes with >=K samples. Have {len(self.classes)}, need P={P}.")

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []
            chosen_classes = self.rng.sample(self.classes, self.P)
            for c in chosen_classes:
                idxs = self.class_to_indices[c]
                batch.extend(self.rng.sample(idxs, self.K))
            yield batch

    def __len__(self):
        return self.steps_per_epoch


# -------------------------
# Model: backbone + projection head
# -------------------------
class EmbedNet(nn.Module):
    def __init__(self, backbone_name="convnext_base", proj_dim=256, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
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


# -------------------------
# Helpers: split by class_id so no leakage
# -------------------------
def split_train_val_by_class(df: pd.DataFrame, val_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    classes = df["class_id"].unique()
    rng.shuffle(classes)
    n_val = int(len(classes) * val_frac)
    val_classes = set(classes[:n_val])
    train_df = df[~df["class_id"].isin(val_classes)].copy()
    val_df = df[df["class_id"].isin(val_classes)].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# -------------------------
# Validation: within-val Recall@1 using FAISS, excluding self
# -------------------------
@torch.no_grad()
def compute_within_split_recall_at_1(model, df_val, device, batch_size=64):
    model.eval()

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    ds = SOPDataset(df_val, eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    embs = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        z = model(x).cpu().numpy().astype("float32")
        embs.append(z)
        labels.append(y.numpy().astype(np.int64))

    X = np.vstack(embs)
    y = np.concatenate(labels)

    # FAISS exact cosine/inner-product search (embeddings are normalized)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    _, idx = index.search(X, 2)  # self + nearest neighbor
    nn_idx = idx[:, 1]                # exclude self
    y_nn = y[nn_idx]

    recall1 = float((y_nn == y).mean())
    model.train()
    return recall1


def main():
    # ----- paths -----
    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path("/Users/samuel.omole/OneDrive - Science and Technology Facilities Council/Stanford_Online_Products")
    train_txt = data_root / "Ebay_train.txt"

    # cols = ["image_id", "class_id", "super_class_id", "path"]
    df = pd.read_csv(train_txt, sep=" ")

    # absolute paths
    df["path"] = df["path"].apply(lambda p: str(data_root / p))

    # Keep only class_ids with >=2 images (needed for K=2)
    counts = df["class_id"].value_counts()
    df = df[df["class_id"].isin(counts[counts >= 2].index)].reset_index(drop=True)

    print("Total train images:", len(df))
    print("Total train classes:", df["class_id"].nunique())

    # ----- device -----
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Device:", device)

    # ----- train/val split by class_id -----
    train_df, val_df = split_train_val_by_class(df, val_frac=0.1, seed=42)
    print("Train images:", len(train_df), "Train classes:", train_df["class_id"].nunique())
    print("Val images:", len(val_df), "Val classes:", val_df["class_id"].nunique())

    # ----- training transforms -----
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    train_ds = SOPDataset(train_df, train_tf)

    # ----- PK sampling -----
    P, K = 16, 2
    steps_per_epoch = 2000
    batch_sampler = PKSampler(train_df["class_id"].values, P=P, K=K, steps_per_epoch=steps_per_epoch, seed=42)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    # ----- model -----
    model = EmbedNet(backbone_name="convnext_base", proj_dim=256, pretrained=True).to(device)

    # ----- loss: library SupCon -----
    criterion = SupConLoss(temperature=0.07)

    # ----- optimizer -----
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ----- training control -----
    max_epochs = 30
    patience = 5            # early stopping patience
    min_delta = 1e-4        # minimum improvement in val metric to count

    out_dir = repo_root / "artifacts" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "sop_supcon_convnext_updated.pt"

    best_val = -1.0
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"epoch {epoch}/{max_epochs}")

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            z = model(x)          # (B, D), already normalized
            loss = criterion(z, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(train_loss=np.mean(losses[-50:]))

        # ----- validate (Recall@1 within val set) -----
        val_r1 = compute_within_split_recall_at_1(model, val_df, device, batch_size=64 if device != "cpu" else 16)
        print(f"Epoch {epoch}: train_loss={np.mean(losses):.4f}, val_R@1={val_r1:.4f}")

        # ----- early stopping + save best -----
        if val_r1 > best_val + min_delta:
            best_val = val_r1
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict()},
                best_ckpt
            )
            print(f"New best model saved: {best_ckpt} (val_R@1={best_val:.4f})")
        else:
            bad_epochs += 1
            print(f"No improvement. bad_epochs={bad_epochs}/{patience}")

        if bad_epochs >= patience:
            print(f"⏹ Early stopping: val_R@1 did not improve for {patience} epochs.")
            break

    print(f"Training done. Best val_R@1={best_val:.4f}. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
