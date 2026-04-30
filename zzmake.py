import torch
from pathlib import Path
import matplotlib.pyplot as plt

repo_root = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized")
model_name = "NS-PwC-L"

sens_path = (
    repo_root
    / "SAPQ"
    / "prior_sensitivity_sobo"
    / model_name
    / "prior_sensitivity.pt"
)

obj = torch.load(sens_path, map_location="cpu")

# use RAW layer sensitivity
layer_sens = obj["layer_sensitivity_raw"]

layer_values = [sens.mean().item() for sens in layer_sens.values()]

plt.figure(figsize=(14, 5))
plt.bar(range(1, len(layer_values) + 1), layer_values)

# show tick every 10 layers
max_idx = len(layer_values)

ticks = [1] + list(range(50, max_idx + 1, 50))
plt.xticks(ticks, [str(t) for t in ticks])

plt.xlabel("Layer index")
plt.ylabel("Average importance")
plt.title("Layer importance")

plt.tight_layout()

save_path = sens_path.parent / "layer_importance_bar.png"
plt.savefig(save_path, dpi=200)
plt.close()

print(f"Saved plot -> {save_path}")