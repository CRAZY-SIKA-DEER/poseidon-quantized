import torch
from pathlib import Path
import json

# -------------------------
# Change this path if needed
# -------------------------
root = Path("ppq_artifacts/NS-PwC-L_layerio")

meta_path = root / "meta.json"
layer_dir = root / "layer_io"

print("=== META ===")
with open(meta_path, "r") as f:
    meta = json.load(f)

print("num_layers:", meta["num_layers"])
print("num_batches:", meta["num_batches"])
print("save_dtype:", meta["save_dtype"])
print("first 3 layers:", meta["layers"][:3])

# -------------------------
# Inspect one layer file
# -------------------------
layer_file = layer_dir / "layer_00000.pt"   # change index if needed
print("\n=== LOAD:", layer_file, "===")

obj = torch.load(layer_file, map_location="cpu")

print("keys:", obj.keys())
print("layer_index:", obj["layer_index"])
print("layer_name:", obj["layer_name"])
print("num_batches:", obj["num_batches"])
print("save_dtype:", obj["save_dtype"])

batches = obj["batches"]
print("\n=== FIRST BATCH ===")

b0 = batches[0]

print("keys in batch:", b0.keys())
print("x_pre shape:", b0["x_pre"].shape, b0["x_pre"].dtype)
print("y_post shape:", b0["y_post"].shape, b0["y_post"].dtype)

# -------------------------
# sanity: check few batches
# -------------------------
print("\n=== CHECK MULTIPLE BATCHES ===")
for i in [0, 1, 2]:
    if i >= len(batches):
        break
    xb = batches[i]["x_pre"]
    yb = batches[i]["y_post"]
    print(f"batch {i}: x={tuple(xb.shape)}, y={tuple(yb.shape)}")