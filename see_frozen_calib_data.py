import torch

path = "/home/u6ey/yiheng.u6ey/poseidon-quantized/ppq_artifacts/frozen_calibration_batches.pt"
obj = torch.load(path, map_location="cpu")

print("type(obj):", type(obj))

if isinstance(obj, list):
    print("len(obj):", len(obj))
    if len(obj) > 0:
        print("type(obj[0]):", type(obj[0]))
        if isinstance(obj[0], dict):
            print("keys:", obj[0].keys())
            for k, v in obj[0].items():
                if torch.is_tensor(v):
                    print(k, v.shape, v.dtype)
                else:
                    print(k, type(v))
elif isinstance(obj, dict):
    print("keys:", obj.keys())
    for k, v in obj.items():
        print(k, type(v))

print("\n=== time values in first few batches ===")
all_t = []
for bi in range(min(5, len(obj))):
    t = obj[bi]["time"]
    print(f"batch {bi}: time shape={t.shape}, values={t.flatten().tolist()}")
    all_t.append(t.flatten())

all_t = torch.cat(all_t)
print("\nunique time values:", torch.unique(all_t))