# extract_embeddings.py
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

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

def main():
    data_root = Path("/Users/samuel.omole/OneDrive - Science and Technology Facilities Council/Stanford_Online_Products")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(
        data_root / "Ebay_train.txt",
        sep=" ",
    )

    test_df = pd.read_csv(
        data_root / "Ebay_test.txt",
        sep=" ",
    )
    print(f"Train set with length {len(train_df)}: ")
    print(train_df.head())
    print(f"\nTest set with length {len(test_df)}: ")
    print(test_df.head())

    train_df = train_df.copy()
    train_df["split"] = "train"
    test_df  = test_df.copy()
    test_df["split"] = "test"
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    df_all["path"] = df_all["path"].apply(lambda p: str(data_root / p))

    model_name = "convnext_base"
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def forward_batch(x):
        z = model(x)
        return F.normalize(z, dim=1)

    batch_size = 64 if device == "cuda" else 16

    loader = DataLoader(
        SOPDataset(df_all, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device=="cuda"),
    )

    embs = []
    meta = []

    for x, paths, class_ids, super_ids, splits in tqdm(loader):
        x = x.to(device, non_blocking=True)
        z = forward_batch(x).cpu().numpy().astype("float32")
        embs.append(z)

        for p, cid, sid, sp in zip(paths, class_ids, super_ids, splits):
            meta.append((p, int(cid), int(sid), sp))

    embeddings = np.vstack(embs)
    meta_df = pd.DataFrame(meta, columns=["path", "class_id", "super_class_id", "split"])

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "artifacts" / "sop_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"emb_{model_name}.npy", embeddings)
    meta_df.to_parquet(out_dir / f"meta_{model_name}.parquet", index=False)

if __name__ == "__main__":
    main()