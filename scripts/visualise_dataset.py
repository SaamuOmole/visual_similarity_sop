import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

data_root = Path("/Users/samuel.omole/OneDrive - Science and Technology Facilities Council/Stanford_Online_Products")

train_df = pd.read_csv(
    data_root / "Ebay_info.txt",
    sep=" ",
)

# Images per class
class_counts = train_df["class_id"].value_counts()

# Summary statistics
stats = {
    "min": class_counts.min(),
    "max": class_counts.max(),
    "mean": class_counts.mean(),
    "median": class_counts.median(),
    "std": class_counts.std(),
}

plt.figure(figsize=(12, 4))

# Bar plot
plt.bar(
    class_counts.index,
    class_counts.values,
    alpha=0.7
)

# Overlay horizontal reference lines
plt.axhline(stats["mean"], linestyle="--", color="red", linewidth=1, label=f"Mean = {stats['mean']:.1f}")
plt.axhline(stats["median"], linestyle=":", color="blue", linewidth=1, label=f"Median = {stats['median']:.1f}")
plt.axhline(stats["min"], linestyle="-.", color="green", linewidth=1, label=f"Min = {stats['min']}")
plt.axhline(stats["max"], linestyle="-", color="black", linewidth=1, label=f"Max = {stats['max']}")

plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.title("Images per Class")
plt.legend(fontsize=9, ncol=4)
plt.tight_layout()
plt.show()

# Optional: print numeric summary below
print("Class size statistics:")
for k, v in stats.items():
    print(f"{k:>6}: {v:.2f}" if isinstance(v, float) else f"{k:>6}: {v}")



