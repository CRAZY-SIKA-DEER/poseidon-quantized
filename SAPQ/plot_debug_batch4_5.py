# SAPQ/plot_debug_batch4_5.py

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")

BASE_DIR = (
    REPO_ROOT
    / "SAPQ/debug_batch4_5_out/NS-SVS-L/NS-SVS"
)

GLOBAL_JSON = BASE_DIR / "batch0_to_5_global_summary.json"
LAYER_CSV = BASE_DIR / "batch4_5_layer_grad_summary.csv"

OUT_DIR = BASE_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_global_bit_stats():
    with open(GLOBAL_JSON, "r") as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    x = df["batch_idx"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["avg_bits_after"], marker="o")
    plt.axhline(8.0, linestyle="--", linewidth=1)
    plt.axhline(4.0, linestyle="--", linewidth=1)
    plt.xlabel("Batch index")
    plt.ylabel("Average bitwidth after update")
    plt.title("Average Bitwidth Evolution: Batch 0 to 5")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "avg_bits_batch0_to_5.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["after_num_layers_above_12"], marker="o", label="layers > 12 bits")
    plt.plot(x, df["after_num_layers_above_10"], marker="o", label="layers > 10 bits")
    plt.plot(x, df["after_num_layers_above_8"], marker="o", label="layers > 8 bits")
    plt.xlabel("Batch index")
    plt.ylabel("Number of layers")
    plt.title("Number of High-Bit Layers After Each Update")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "high_bit_layer_counts_batch0_to_5.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["after_max_layer_bits"], marker="o", label="max layer bits")
    plt.plot(x, df["after_top10_mean_layer_bits"], marker="o", label="top-10 mean layer bits")
    plt.axhline(8.0, linestyle="--", linewidth=1)
    plt.xlabel("Batch index")
    plt.ylabel("Layer average bitwidth")
    plt.title("Extreme Layer Bitwidth After Each Update")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "extreme_layer_bits_batch0_to_5.png", dpi=300)
    plt.close()

    print("Saved global plots.")


def plot_layer_grads(batch_idx: int):
    df = pd.read_csv(LAYER_CSV)
    d = df[df["batch_idx"] == batch_idx].reset_index(drop=True)
    x = range(len(d))

    # 1. likelihood gradient
    plt.figure(figsize=(18, 5))
    plt.plot(x, d["like_grad_mean"], linewidth=1)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Layer index in model traversal order")
    plt.ylabel("mean likelihood gradient")
    plt.title(f"Batch {batch_idx}: Likelihood Gradient by Layer")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"batch{batch_idx}_likelihood_grad_by_layer.png", dpi=300)
    plt.close()

    # 2. effective prior gradient
    plt.figure(figsize=(18, 5))
    plt.plot(x, d["prior_contrib_grad_mean"], linewidth=1)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Layer index in model traversal order")
    plt.ylabel("mean effective prior gradient")
    plt.title(f"Batch {batch_idx}: Effective Prior Gradient by Layer")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"batch{batch_idx}_effective_prior_grad_by_layer.png", dpi=300)
    plt.close()

    # 3. delta bits
    plt.figure(figsize=(18, 5))
    plt.plot(x, d["delta_bits_mean"], linewidth=1)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Layer index in model traversal order")
    plt.ylabel("mean delta bits")
    plt.title(f"Batch {batch_idx}: Bitwidth Change by Layer")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"batch{batch_idx}_delta_bits_by_layer.png", dpi=300)
    plt.close()

    # 4. agreement comparison
    plt.figure(figsize=(18, 5))
    plt.plot(x, d["agree_like_with_delta"], linewidth=1, label="likelihood agreement")
    plt.plot(x, d["agree_prior_with_delta"], linewidth=1, label="prior agreement")
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Layer index in model traversal order")
    plt.ylabel("agreement with actual update")
    plt.title(f"Batch {batch_idx}: Gradient Agreement with Actual ΔS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"batch{batch_idx}_agreement_by_layer.png", dpi=300)
    plt.close()

    print(f"Saved batch {batch_idx} layer plots.")


def main():
    plot_global_bit_stats()
    plot_layer_grads(batch_idx=4)
    plot_layer_grads(batch_idx=5)
    print(f"All plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()