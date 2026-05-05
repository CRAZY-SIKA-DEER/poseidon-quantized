# SAPQ/plot_batch0_1_grads.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")

CSV_PATH = (
    REPO_ROOT
    / "SAPQ/debug_batch0_1_out/NS-SVS-L/NS-SVS/batch0_1_layer_summary.csv"
)

OUT_DIR = (
    REPO_ROOT
    / "SAPQ/debug_batch0_1_out/NS-SVS-L/NS-SVS/plots"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_grad(df, batch_idx: int, grad_col: str, title: str, save_name: str):
    d = df[df["batch_idx"] == batch_idx].copy()

    # Keep model traversal order from the CSV
    d = d.reset_index(drop=True)
    x = range(len(d))

    plt.figure(figsize=(18, 5))
    plt.plot(x, d[grad_col].values, linewidth=1.0)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)

    plt.xlabel("Layer index in model traversal order")
    plt.ylabel(grad_col)
    plt.title(title)

    plt.tight_layout()
    save_path = OUT_DIR / save_name
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved -> {save_path}")


def main():
    df = pd.read_csv(CSV_PATH)

    plot_grad(
        df,
        batch_idx=0,
        grad_col="like_grad_mean",
        title="Batch 0: Likelihood Gradient by Layer",
        save_name="batch0_likelihood_grad_by_layer.png",
    )

    plot_grad(
        df,
        batch_idx=0,
        grad_col="prior_contrib_grad_mean",
        title="Batch 0: Prior Gradient by Layer",
        save_name="batch0_prior_grad_by_layer.png",
    )

    plot_grad(
        df,
        batch_idx=1,
        grad_col="like_grad_mean",
        title="Batch 1: Likelihood Gradient by Layer",
        save_name="batch1_likelihood_grad_by_layer.png",
    )

    plot_grad(
        df,
        batch_idx=1,
        grad_col="prior_contrib_grad_mean",
        title="Batch 1: Prior Gradient by Layer",
        save_name="batch1_prior_grad_by_layer.png",
    )


if __name__ == "__main__":
    main()