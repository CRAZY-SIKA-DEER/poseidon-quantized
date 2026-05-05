# SAPQ/plot_prior_grad_by_layer.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")

# Use the signed gradient CSV you already have
CSV_PATH = REPO / "SAPQ/debug_grad_outputs/NS-SVS-L/signed_like_prior_grad_by_layer.csv"

OUT_DIR = REPO / "SAPQ/debug_grad_outputs/NS-SVS-L/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# plot first batch only
BATCH_LIST = [4]

# signed gradient: positive => pushes bitwidth up, negative => pushes bitwidth down
VALUE_COL = "grad_mean"
# VALUE_COL = "grad_sum"
# VALUE_COL = "grad_abs_mean"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"[LOAD] {CSV_PATH}")
    print(f"[INFO] rows = {len(df)}")
    print(f"[INFO] columns = {list(df.columns)}")

    if VALUE_COL not in df.columns:
        raise KeyError(
            f"Column '{VALUE_COL}' not found. Available columns: {list(df.columns)}"
        )

    plt.figure(figsize=(22, 5))

    plotted = False

    for batch_idx in BATCH_LIST:
        sub = df[
            (df["batch_idx"] == batch_idx)
            & (df["grad_type"] == "prior")
        ].copy()

        if len(sub) == 0:
            print(f"[WARN] no prior rows found for batch {batch_idx}")
            continue

        # keep original network order from CSV
        sub = sub.reset_index(drop=True)

        x = range(len(sub))
        y = sub[VALUE_COL].values

        plt.plot(x, y, linewidth=1.0, label=f"batch {batch_idx}")
        plotted = True

    if not plotted:
        raise RuntimeError("No data was plotted. Check batch_idx and grad_type.")

    plt.axhline(0.0, linestyle="--", linewidth=1.0)

    plt.xlabel("Layer index in network order")
    plt.ylabel(VALUE_COL)
    plt.title(f"Prior gradient by layer | {VALUE_COL}")

    plt.legend()
    plt.tight_layout()

    batch_name = "_".join(str(b) for b in BATCH_LIST)
    out_path = OUT_DIR / f"prior_{VALUE_COL}_batch{batch_name}.png"

    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()