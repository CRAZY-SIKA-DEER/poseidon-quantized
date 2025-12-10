import json
import os
import matplotlib.pyplot as plt

# Adjust path if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(
    PROJECT_ROOT,
    "ppq_artifacts",
    "NS-PwC-T",
    "PwC-T-results-1-5-conv.json"
)

# Load results
with open(results_path, "r") as f:
    results = json.load(f)

# Containers
epochs = []
ppq_l1 = []
ppq_rel = []

# dynamic-8 only exists in FINAL evaluation
dyn8_l1 = None
dyn8_rel = None

# Extract epoch-wise PPQ results
for ep_str, rec in results.items():
    ep = int(ep_str)
    if "ppq" in rec:          # only intermediate epochs
        epochs.append(ep)
        ppq_l1.append(rec["ppq"]["l1"])
        ppq_rel.append(rec["ppq"]["rel_l1"])

    # capture dynamic-8 only once (final epoch)
    if "dyn8" in rec:
        dyn8_l1 = rec["dyn8"]["l1"]
        dyn8_rel = rec["dyn8"]["rel_l1"]

# Sort chronologically
idx = sorted(range(len(epochs)), key=lambda i: epochs[i])
epochs  = [epochs[i] for i in idx]
ppq_l1  = [ppq_l1[i] for i in idx]
ppq_rel = [ppq_rel[i] for i in idx]

# Save directory
save_dir = os.path.join(PROJECT_ROOT, "ppq_artifacts", "NS-PwC-T")
os.makedirs(save_dir, exist_ok=True)

# =========================================
#   Plot 1: L1 error (PPQ vs dynamic-8 final)
# =========================================
plt.figure(figsize=(8,5))
plt.plot(epochs, ppq_l1, label="PPQ (Conv+Linear)", linewidth=2)

if dyn8_l1 is not None:
    plt.axhline(y=dyn8_l1, color="red", linestyle="--", linewidth=2,
                label=f"Dynamic 8-bit (final): {dyn8_l1:.2f}")

plt.xlabel("Epoch")
plt.ylabel("L1 error")
plt.title("L1 Error vs Epoch (PPQ vs Dynamic 8-bit)")
plt.grid(True)
plt.legend()
plt.tight_layout()

out1 = os.path.join(save_dir, "PwC-T_L1_vs_epoch_PPQ_vs_dyn8.png")
plt.savefig(out1, dpi=250)
plt.close()
print(f"Saved: {out1}")

# =========================================
#   Plot 2: Relative L1 (%) (PPQ vs dynamic-8)
# =========================================
plt.figure(figsize=(8,5))
plt.plot(epochs, ppq_rel, label="PPQ (Conv+Linear)", linewidth=2)

if dyn8_rel is not None:
    plt.axhline(y=dyn8_rel, color="red", linestyle="--", linewidth=2,
                label=f"Dynamic 8-bit (final): {dyn8_rel:.3f}")

plt.xlabel("Epoch")
plt.ylabel("Relative L1 (%)")
plt.title("Relative L1 vs Epoch (PPQ vs Dynamic 8-bit)")
plt.grid(True)
plt.legend()
plt.tight_layout()

out2 = os.path.join(save_dir, "PwC-T_relL1_conv2d_vs_epoch_PPQ_vs_dyn8.png")
plt.savefig(out2, dpi=250)
plt.close()
print(f"Saved: {out2}")
