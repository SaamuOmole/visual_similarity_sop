from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


class SOPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        x = self.transform(img)
        return x, row.path, int(row.class_id), int(row.super_class_id), row.split


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


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path("/Users/samuel.omole/OneDrive - Science and Technology Facilities Council/Stanford_Online_Products")

    # Load SOP splits
    train_df = pd.read_csv(data_root / "Ebay_train.txt", sep=" ")
    test_df  = pd.read_csv(data_root / "Ebay_test.txt", sep=" ")

    train_df["split"] = "train"
    test_df["split"] = "test"

    df_all = pd.concat([train_df, test_df], ignore_index=True)
    df_all["path"] = df_all["path"].apply(lambda p: str(data_root / p))

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Device:", device)

    # Deterministic eval transforms (important!)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    ds = SOPDataset(df_all, transform)
    loader = DataLoader(
        ds,
        batch_size=64 if device != "cpu" else 16,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    # Load model + checkpoint
    model_name = "convnext_base"
    proj_dim = 256
    model = EmbedNet(model_name, proj_dim=proj_dim).to(device).eval()

    ckpt_path = repo_root / "artifacts" / "checkpoints" / "sop_supcon_convnext_updated.pt"  # change path to model
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    print("Loaded checkpoint:", ckpt_path)

    # Extract
    embs, meta_rows = [], []
    with torch.no_grad():
        for x, paths, class_ids, super_ids, splits in tqdm(loader):
            x = x.to(device)
            z = model(x).cpu().numpy().astype("float32")
            embs.append(z)
            for p, cid, sid, sp in zip(paths, class_ids, super_ids, splits):
                meta_rows.append((p, int(cid), int(sid), sp))

    embeddings = np.vstack(embs)
    meta = pd.DataFrame(meta_rows, columns=["path", "class_id", "super_class_id", "split"])

    out_dir = repo_root / "artifacts" / "sop_embeddings_finetuned_updated"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{model_name}_supcon_p{proj_dim}"
    np.save(out_dir / f"emb_{tag}.npy", embeddings)
    meta.to_parquet(out_dir / f"meta_{tag}.parquet", index=False)

    print("Saved:", out_dir / f"emb_{tag}.npy")
    print("Saved:", out_dir / f"meta_{tag}.parquet")
    print("Embeddings:", embeddings.shape)

if __name__ == "__main__":
    main()
