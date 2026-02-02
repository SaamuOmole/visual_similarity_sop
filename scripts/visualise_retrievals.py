from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def show_row(row, topk=5):
    plt.figure(figsize=(2*(topk+1), 3))

    q = Image.open(row["query_path"]).convert("RGB")
    plt.subplot(1, topk+1, 1)
    plt.imshow(q)
    plt.title(f"QUERY\nclass={row['query_class_id']}")
    plt.axis("off")

    for i in range(topk):
        p = row[f"rank{i+1}_path"]
        cid = row[f"rank{i+1}_class_id"]
        score = row[f"rank{i+1}_score"]
        img = Image.open(p).convert("RGB")
        plt.subplot(1, topk+1, i+2)
        plt.imshow(img)
        plt.title(f"r{i+1}, class={cid}\nscore={score:.3f}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    repo_root = Path(__file__).resolve().parent.parent
    # in_dir = repo_root / "artifacts" / "retrieval_results"
    in_dir = repo_root / "artifacts" / "retrieval_results_finetuned_updated"

    # model_name = "convnext_base"
    model_name = "convnext_base_supcon_p256"
    split = "test"

    results_path = in_dir / f"results_{model_name}_{split}_K10.parquet"
    failures_path = in_dir / f"failures_{model_name}_{split}_top5.parquet"

    results = pd.read_parquet(results_path)
    failures = pd.read_parquet(failures_path)

    print("Total queries:", len(results))
    print("Failures:", len(failures))

    # Show a few successes (top1 correct)
    success = results[results["rank1_class_id"] == results["query_class_id"]]
    print("Top1 successes:", len(success))

    # print("\nShowing 5 successes...")
    # for i in range(min(5, len(success))):
    #     show_row(success.iloc[i], topk=5)

    # print("\nShowing 5 failures...")
    # for i in range(min(5, len(failures))):
    #     show_row(failures.iloc[i], topk=5)

    print("\nShowing 5 failures...")
    for _, row in failures.sample(n=min(5, len(failures))).iterrows():
        show_row(row, topk=5)
        
    print("\nShowing 5 successes...")
    for _, row in success.sample(n=min(5, len(success))).iterrows():
        show_row(row, topk=5)

if __name__ == "__main__":
    main()