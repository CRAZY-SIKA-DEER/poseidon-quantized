# SAPQ/plot_like_prior_grad_by_layer.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")
CSV_PATH = REPO / "SAPQ/debug_grad_outputs/NS-SVS-L/like_prior_grad_by_layer.csv"
OUT_DIR = REPO / "SAPQ/debug_grad_outputs/NS-SVS-L/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_IDX = 4         # change to 1,2,3,4 if needed
VALUE_COL = "sum_g_over_s"
# VALUE_COL = "grad_mean"
# VALUE_COL = "grad_abs_mean"


def plot_one(df, grad_type: str):
    sub = df[
        (df["batch_idx"] == BATCH_IDX) &
        (df["grad_type"] == grad_type)
    ].copy()

    # keep original network order from CSV
    sub = sub.reset_index(drop=True)
    x = range(len(sub))
    y = sub[VALUE_COL].values

    plt.figure(figsize=(22, 5))
    plt.plot(x, y, linewidth=0.8)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)

    plt.xlabel("Layer index in network order")
    plt.ylabel(VALUE_COL)
    plt.title(f"{grad_type.capitalize()} gradient by layer | batch {BATCH_IDX}")

    plt.tight_layout()

    out_path = OUT_DIR / f"{grad_type}_{VALUE_COL}_batch{BATCH_IDX}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SAVED] {out_path}")


def main():
    df = pd.read_csv(CSV_PATH)

    print(f"[LOAD] {CSV_PATH}")
    print(f"[INFO] rows = {len(df)}")
    print(f"[INFO] columns = {list(df.columns)}")

    plot_one(df, "likelihood")
    plot_one(df, "prior")


if __name__ == "__main__":
    main()