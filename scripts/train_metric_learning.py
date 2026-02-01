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
    def __init__(self, labels, P=16, K=2, steps_per_epoch=1000):
        self.labels = np.array(labels)
        self.P = P
        self.K = K
        self.steps_per_epoch = steps_per_epoch

        self.class_to_indices = {}
        for i, c in enumerate(self.labels):
            self.class_to_indices.setdefault(int(c), []).append(i)

        # keep only classes that have at least K examples
        self.classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) >= K]

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []
            chosen_classes = random.sample(self.classes, self.P)
            for c in chosen_classes:
                idxs = self.class_to_indices[c]
                batch.extend(random.sample(idxs, self.K))
            yield batch

    def __len__(self):
        return self.steps_per_epoch


# -------------------------
# Model: backbone + projection head
# -------------------------
class EmbedNet(nn.Module):
    def __init__(self, backbone_name="convnext_base", proj_dim=256):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)  # (B, D)
        backbone_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_dim, proj_dim),
        )

    def forward(self, x):
        h = self.backbone(x)              # (B, D)
        z = self.proj(h)                  # (B, proj_dim)
        z = F.normalize(z, dim=1)
        return z


# -------------------------
# Supervised Contrastive loss
# -------------------------
def supervised_contrastive_loss(z, y, temperature=0.07):
    """
    z: (B, D) normalized embeddings
    y: (B,) labels (class_id)
    """
    device = z.device
    y = y.view(-1, 1)
    B = z.size(0)

    # similarity matrix
    sim = (z @ z.T) / temperature  # (B, B)

    # mask self
    logits_mask = torch.ones((B, B), device=device) - torch.eye(B, device=device)

    # positives mask: same label, not self
    pos_mask = (y == y.T).float() * logits_mask

    # log-softmax over all others
    sim = sim - sim.max(dim=1, keepdim=True).values  # numerical stability
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # mean log-likelihood over positives
    pos_count = pos_mask.sum(dim=1)
    # avoid divide by zero (shouldn't happen with PK sampling)
    loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-12)
    return loss.mean()


def main():
    repo_root = Path(__file__).resolve().parent.parent

    data_root = Path("/Users/samuel.omole/OneDrive - Science and Technology Facilities Council/Stanford_Online_Products")
    train_txt = data_root / "Ebay_train.txt"

    cols = ["image_id", "class_id", "super_class_id", "path"]
    train_df = pd.read_csv(train_txt, sep=" ", header=None, names=cols)

    # make absolute paths
    train_df["path"] = train_df["path"].apply(lambda p: str(data_root / p))

    # Keep only class_ids with >= 2 images (K=2)
    counts = train_df["class_id"].value_counts()
    train_df = train_df[train_df["class_id"].isin(counts[counts >= 2].index)].reset_index(drop=True)

    print("Train images:", len(train_df))
    print("Train classes:", train_df["class_id"].nunique())

    # ---- Device ----
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Device:", device)

    # ---- Transforms (stronger than center crop) ----
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    ds = SOPDataset(train_df, transform)

    # ---- PK sampling config ----
    P, K = 16, 2
    steps_per_epoch = 2000  # reduce if slow
    batch_sampler = PKSampler(train_df["class_id"].values, P=P, K=K, steps_per_epoch=steps_per_epoch)

    loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,  # yields lists of indices
        num_workers=0,                # safest on mac; increase in a script if stable
        pin_memory=(device == "cuda"),
    )

    # ---- Model ----
    model = EmbedNet(backbone_name="convnext_base", proj_dim=256).to(device)

    # ---- Optimizer ----
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ---- Train ----
    epochs = 2  # start small
    temperature = 0.07
    model.train()

    out_dir = repo_root / "artifacts" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        losses = []
        pbar = tqdm(loader, total=steps_per_epoch, desc=f"epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            z = model(x)
            loss = supervised_contrastive_loss(z, y, temperature=temperature)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(losses[-50:]))

        ckpt = out_dir / f"sop_supcon_convnext_epoch{epoch}.pt"
        torch.save({"model": model.state_dict()}, ckpt)
        print("Saved:", ckpt)

    print("Training done.")

if __name__ == "__main__":
    main()