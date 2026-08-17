from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="rollout_figures")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    gt = data["gt"]          # [N, T, C, H, W]
    times = data["times"]    # [T]

    method_order = [
        ("Uniform-w4", "Uniform\n4-bit"),
        ("BRECQ-w4", "BRECQ\n4-bit"),
        ("PPQ", "PPQ\n7-bit"),
        ("Uniform-w8", "Uniform\n8-bit"),
        ("BRECQ-w8", "BRECQ\n8-bit"),
        ("SAPQ", "Ours\n4-bit"),
    ]

    if "incompressible" in args.dataset:
        fields = {"u": 1, "v": 2}
    elif "compressible" in args.dataset:
        fields = {"rho": 0, "u": 1, "v": 2, "p": 3}
    elif "wave" in args.dataset.lower():
        fields = {"u": 0, "c": 1}
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")

    for field_name, ch in fields.items():
        n_rows = len(times)
        n_cols = len(method_order) + 1  # + GT

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.6 * n_cols, 2.5 * n_rows),
        )

        if n_rows == 1:
            axes = axes[np.newaxis, :]

        vmin = np.percentile(gt[args.sample_idx, :, ch], 10)
        vmax = np.percentile(gt[args.sample_idx, :, ch], 90)

        for r, t in enumerate(times):
            for c, (method_key, method_title) in enumerate(method_order):
                ax = axes[r, c]

                pred_key = f"pred_{method_key}"
                if pred_key not in data:
                    ax.axis("off")
                    ax.set_title(f"{method_title}\nmissing", fontsize=11)
                    continue

                img = data[pred_key][args.sample_idx, r, ch]

                im = ax.imshow(
                    img.T,
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                )

                if r == 0:
                    ax.set_title(method_title, fontsize=13)

                if c == 0:
                    ax.set_ylabel(f"t={int(t)}", fontsize=13)

                ax.set_xticks([])
                ax.set_yticks([])

            # GT column
            ax = axes[r, -1]
            img = gt[args.sample_idx, r, ch]

            im = ax.imshow(
                img.T,
                origin="lower",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )

            if r == 0:
                ax.set_title("GT", fontsize=13)

            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Rollout field: {field_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.96, 0.96])

        cbar_ax = fig.add_axes([0.965, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        save_path = save_dir / f"rollout_{field_name}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved -> {save_path}")


if __name__ == "__main__":
    main()