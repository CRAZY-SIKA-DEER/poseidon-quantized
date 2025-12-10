import json
import os
import matplotlib.pyplot as plt

# Adjust path if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(
    PROJECT_ROOT,
    "ppq_artifacts",
    "NS-PwC-T",
    "PwC-T-results-1-7.json"
)

# Load results
with open(results_path, "r") as f:
    results = json.load(f)

# Containers
epochs = []
ppq_l1, dyn16_l1 = [], []
ppq_rel, dyn16_rel = [], []

# Extract data
for ep_str, rec in results.items():
    ep = int(ep_str)
    epochs.append(ep)

    ppq_l1.append(rec["ppq"]["l1"])
    dyn16_l1.append(rec["dyn16"]["l1"])

    ppq_rel.append(rec["ppq"]["rel_l1"])
    dyn16_rel.append(rec["dyn16"]["rel_l1"])

# Sort chronologically
idx = sorted(range(len(epochs)), key=lambda i: epochs[i])
epochs      = [epochs[i]      for i in idx]
ppq_l1      = [ppq_l1[i]      for i in idx]
dyn16_l1    = [dyn16_l1[i]    for i in idx]
ppq_rel     = [ppq_rel[i]     for i in idx]
dyn16_rel   = [dyn16_rel[i]   for i in idx]

# Save directory
save_dir = os.path.join(PROJECT_ROOT, "ppq_artifacts", "NS-PwC-T")
os.makedirs(save_dir, exist_ok=True)

# =========================================
#   Plot 1: L1 error (PPQ vs 16-bit)
# =========================================
plt.figure(figsize=(8,5))
plt.plot(epochs, ppq_l1,   label="PPQ (weight-only)", linewidth=2)
plt.plot(epochs, dyn16_l1, label="Dynamic 16-bit", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("L1 error")
plt.title("L1 error vs Epoch (PPQ(8.49453bit) vs Dynamic 16-bit)")
plt.grid(True)
plt.legend()
plt.tight_layout()

out1 = os.path.join(save_dir, "PwC-T_L1_vs_epoch_PPQ_vs_16bit-1-7.png")
plt.savefig(out1, dpi=250)
plt.close()
print(f"Saved: {out1}")

# =========================================
#   Plot 2: Relative L1 (PPQ vs 16-bit)
# =========================================
plt.figure(figsize=(8,5))
plt.plot(epochs, ppq_rel,   label="PPQ (weight-only)", linewidth=2)
plt.plot(epochs, dyn16_rel, label="Dynamic 16-bit", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Relative L1 (%)")
plt.title("Relative L1 vs Epoch (PPQ vs Dynamic 16-bit)")
plt.grid(True)
plt.legend()
plt.tight_layout()

out2 = os.path.join(save_dir, "PwC-T_relL1_vs_epoch_PPQ_vs_16bit-1-7.png")
plt.savefig(out2, dpi=250)
plt.close()
print(f"Saved: {out2}")
