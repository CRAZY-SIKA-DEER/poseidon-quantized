import torch
from pathlib import Path

path = Path("/home/u6ey/yiheng.u6ey/poseidon-quantized/ppq_artifacts/NS-PwC-L_layerio/layer_io/layer_00000.pt")

obj = torch.load(path, map_location="cpu")

print("type(obj):", type(obj))

if isinstance(obj, dict):
    print("keys:", obj.keys())

    for k, v in obj.items():
        print("\nKEY:", k)
        print("  type:", type(v))

        if torch.is_tensor(v):
            print("  shape:", tuple(v.shape), "dtype:", v.dtype)

        elif isinstance(v, list):
            print("  len:", len(v))
            if len(v) > 0:
                print("  first type:", type(v[0]))
                if torch.is_tensor(v[0]):
                    print("  first shape:", tuple(v[0].shape), "dtype:", v[0].dtype)

        elif isinstance(v, tuple):
            print("  len:", len(v))
            for i, item in enumerate(v):
                print("   tuple item", i, "type:", type(item))
                if torch.is_tensor(item):
                    print("   shape:", tuple(item.shape), "dtype:", item.dtype)
else:
    print(obj)


b0 = obj["batches"][0]
print("batch keys:", b0.keys())

for k, v in b0.items():
    print("\nKEY:", k)
    print(" type:", type(v))
    if torch.is_tensor(v):
        print(" shape:", tuple(v.shape), "dtype:", v.dtype)