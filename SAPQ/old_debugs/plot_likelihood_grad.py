import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# path to your CSV
csv_path = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized/SAPQ/debug_grad_outputs/NS-SVS-L/likelihood_grad_by_layer.csv")

df = pd.read_csv(csv_path)

# ---- choose one batch (e.g. batch 0) ----
batch_idx = 0
df_batch = df[df["batch_idx"] == batch_idx].copy()

# sort by sensitivity (or grad magnitude)
df_batch = df_batch.sort_values(by="sens_mean", ascending=False)

# x = layer index after sorting
x = range(len(df_batch))

# ---- plot ----
plt.figure(figsize=(12, 5))

plt.plot(x, df_batch["grad_mean"], label="grad_mean")
plt.axhline(0, linestyle="--")

plt.title(f"Likelihood Gradient per Layer (batch {batch_idx})")
plt.xlabel("Layer index (sorted by sensitivity)")
plt.ylabel("Gradient (dJ/dS)")

plt.legend()
plt.tight_layout()

save_path = csv_path.parent / f"likelihood_grad_plot_batch{batch_idx}.png"
plt.savefig(save_path, dpi=200)
print(f"Saved plot to {save_path}")

plt.show()