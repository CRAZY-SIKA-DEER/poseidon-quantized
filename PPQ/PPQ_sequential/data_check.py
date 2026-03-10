
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# NOW import everything else
import torch
from scOT.problems.base import get_dataset
from torch.utils.data import DataLoader

# ==============================================
# 1. INVESTIGATE SINGLE SAMPLE FROM DATASET
# ==============================================
print("=" * 60)
print("INVESTIGATING DATASET STRUCTURE")
print("=" * 60)

dataset_name = "fluids.incompressible.PiecewiseConstants"
data_path = "dataset/NS-PwC"

# Create a small dataset
train_ds = get_dataset(
    dataset_name, 
    which="train",
    num_trajectories=10,  # Just 10 samples for testing
    data_path=data_path
)

print(f"\n1. Dataset length: {len(train_ds)}")
print(f"   Dataset type: {type(train_ds)}")

# Get a single sample
sample = train_ds[0]
print(f"\n2. Sample type: {type(sample)}")

if isinstance(sample, dict):
    print("\n3. Dictionary keys:")
    for key in sample.keys():
        print(f"   - {key}")
    
    print("\n4. Each key's content:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            # Skip statistics for boolean tensors
            if value.dtype != torch.bool:
                print(f"          min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}")
            else:
                print(f"          (boolean mask, skipping statistics)")
        else:
            print(f"   {key}: type={type(value)}, value={value}")


# ==============================================
# 2. INVESTIGATE DATALOADER BATCH
# ==============================================
print("\n" + "=" * 60)
print("INVESTIGATING DATALOADER BATCHES")
print("=" * 60)

loader = DataLoader(train_ds, batch_size=4, shuffle=False)

# Get first batch
batch = next(iter(loader))
print(f"\n1. Batch type: {type(batch)}")

if isinstance(batch, dict):
    print("\n2. Batch dictionary keys:")
    for key in batch.keys():
        print(f"   - {key}")
    
    print("\n3. Each key's batched content:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"          min={value.min().item():.4f}, max={value.max().item():.4f}")
        elif isinstance(value, list):
            print(f"   {key}: list of length {len(value)}, first item: {value[0]}")
        else:
            print(f"   {key}: type={type(value)}, value={value}")
elif isinstance(batch, (tuple, list)):
    print(f"\n2. Batch is a tuple/list with {len(batch)} elements:")
    for i, item in enumerate(batch):
        if isinstance(item, torch.Tensor):
            print(f"   Element {i}: shape={item.shape}, dtype={item.dtype}")
        else:
            print(f"   Element {i}: type={type(item)}")

# ==============================================
# 3. TEST UNPACKING (for PPQ compatibility)
# ==============================================
print("\n" + "=" * 60)
print("TESTING PPQ-STYLE UNPACKING")
print("=" * 60)

try:
    inputs, targets = batch
    print("✅ SUCCESS: Batch can be unpacked as (inputs, targets)")
    print(f"   inputs shape: {inputs.shape}")
    print(f"   targets shape: {targets.shape}")
except Exception as e:
    print(f"❌ FAILED: Cannot unpack batch as (inputs, targets)")
    print(f"   Error: {e}")
    print("\n   Need to create adapter function!")
    
    # Suggest possible keys
    if isinstance(batch, dict):
        print(f"\n   Available keys: {list(batch.keys())}")
        print("   Likely candidates for inputs/targets:")
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor) and batch[key].ndim >= 2:
                print(f"      - {key}: shape {batch[key].shape}")

# ==============================================
# 4. CHECK MULTIPLE BATCHES
# ==============================================
print("\n" + "=" * 60)
print("CHECKING MULTIPLE BATCHES")
print("=" * 60)

for i, batch in enumerate(loader):
    if i >= 3:  # Check first 3 batches
        break
    print(f"\nBatch {i+1}:")
    if isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}")
            else:
                print(f"   {key}: {type(value)}")

print("\n" + "=" * 60)
print("INVESTIGATION COMPLETE")
print("=" * 60)
print(f"Channel description: {train_ds.printable_channel_description}")
sample = train_ds[0]
print(sample['pixel_values'].shape) # (channels, H, W)
print(sample['labels'].shape)      # (channels, H, W)
print(sample['time'])              # scalar float (lead time)
