"""
This file is for basic component testing (model loading, data loading, inference, evaluation).
Not for quantization yet - just for code verification.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from scOT.model import ScOT
from scOT.problems.base import get_dataset
from scOT.metrics import relative_lp_error, lp_error
from scOT.problems.fluids.normalization_constants import CONSTANTS
from torch.utils.data import DataLoader
from tqdm import tqdm
import json


# =============================================================================
# Configuration
# =============================================================================
model_path   = "models/NS-PwC"
data_path    = "dataset/NS-PwC"
dataset_name = "fluids.incompressible.PiecewiseConstants"

print("=" * 80)
print("POSEIDON QUANTIZATION - INITIALIZATION")
print("=" * 80)


# =============================================================================
# 1. Load Poseidon Model
# =============================================================================
def load_poseidon_model(model_path: str, device: str = "cuda"):
    """Load the Poseidon ScOT model for quantization."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n[1/5] Loading Poseidon model from: {model_path}")
    
    model = ScOT.from_pretrained(model_path).to(device).eval()
    torch.set_float32_matmul_precision("high")
    
    print(f"  ✓ Model loaded successfully")
    print(f"  ✓ Device: {device}")
    print(f"  ✓ Model type: {type(model).__name__}")
    print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, device


# =============================================================================
# 2. Build Data Loaders
# =============================================================================
def build_poseidon_loaders(
    dataset_name: str,
    data_path: str,
    calib_batchsize: int = 8,
    calib_steps: int = 8,
    val_batchsize: int = 16,
    val_steps: int = 50
):
    """Build calibration and validation data loaders."""
    print(f"\n[2/5] Building data loaders")
    print(f"  Dataset: {dataset_name}")
    print(f"  Data path: {data_path}")
    
    # Load train split for calibration
    print(f"  Loading training data (calibration)...")
    train_ds = get_dataset(dataset_name, which="train",
                           num_trajectories=2048, data_path=data_path)
    print(f"    ✓ Train dataset length: {len(train_ds)} samples")
    
    # Load validation split
    print(f"  Loading validation data...")
    try:
        val_ds = get_dataset(dataset_name, which="val",
                             num_trajectories=256, data_path=data_path)
        print(f"    ✓ Val dataset length: {len(val_ds)} samples")
    except Exception as e:
        print(f"    ! Val split not found, using test split instead")
        val_ds = get_dataset(dataset_name, which="test",
                             num_trajectories=256, data_path=data_path)
        print(f"    ✓ Test dataset length: {len(val_ds)} samples")

    # Create data loaders
    calib_loader = DataLoader(
        train_ds, batch_size=calib_batchsize, shuffle=True,
        num_workers=min(os.cpu_count() or 0, 16), pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batchsize, shuffle=False,
        num_workers=min(os.cpu_count() or 0, 16), pin_memory=True
    )
    
    print(f"  ✓ Calibration loader: {len(calib_loader)} batches (batch_size={calib_batchsize})")
    print(f"  ✓ Validation loader: {len(val_loader)} batches (batch_size={val_batchsize})")
    print(f"  ✓ Will use first {calib_steps} calibration batches, {val_steps} validation batches")

    # Optional: restrict the number of batches
    def take(loader, steps):
        for i, b in enumerate(loader):
            if i >= steps: break
            yield b

    calib_iter = lambda: take(calib_loader, calib_steps)
    val_iter   = lambda: take(val_loader,   val_steps)

    return calib_loader, val_loader, calib_iter, val_iter


# =============================================================================
# 3. Forward Pass Function
# =============================================================================
def poseidon_forward(model, batch, device):
    """Run inference with Poseidon model."""
    x  = batch["pixel_values"].to(device)
    t  = batch.get("time")
    pm = batch.get("pixel_mask")
    y  = batch.get("labels")
    
    out = model(
        pixel_values=x,
        time=(t.to(device) if t is not None else None),
        pixel_mask=(pm.to(device) if pm is not None else None),
        labels=(y.to(device) if y is not None else None),
    )
    return out.output  # predictions tensor


# =============================================================================
# 4. Test Inference
# =============================================================================
def test_inference(model, loader, device, num_batches=2):
    """Test inference on a few batches to verify everything works."""
    print(f"\n[3/5] Testing inference on {num_batches} batches")
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            
            print(f"\n  Batch {i+1}:")
            print(f"    pixel_values: {batch['pixel_values'].shape} {batch['pixel_values'].dtype}")
            print(f"    labels: {batch['labels'].shape} {batch['labels'].dtype}")
            print(f"    time: {batch['time'].shape if hasattr(batch['time'], 'shape') else type(batch['time'])}")
            if 'pixel_mask' in batch:
                print(f"    pixel_mask: {batch['pixel_mask'].shape} {batch['pixel_mask'].dtype}")
            
            # Run inference
            output = poseidon_forward(model, batch, device)
            print(f"    → Output: {output.shape} {output.dtype}")
            print(f"    → Output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"    ✓ Forward pass successful")
    
    print(f"\n  ✓ All inference tests passed!")


# =============================================================================
# 5. Divergence Computation
# =============================================================================
def compute_divergence_numpy(fields: np.ndarray):
    """
    Compute divergence of velocity field from numpy arrays.
    
    Args:
        fields: np.ndarray of shape (N, C, H, W) where C includes [rho, u, v, p]
    
    Returns:
        div: np.ndarray of shape (N, W-2, H-2) - divergence at interior points
    """
    # Extract and denormalize u & v (channels 1, 2)
    means = np.array(CONSTANTS["mean"][1:3]).reshape(1, 2, 1, 1)
    stds = np.array(CONSTANTS["std"][1:3]).reshape(1, 2, 1, 1)
    
    uv_norm = fields[:, 1:3, :, :]  # (N, 2, H, W)
    uv = uv_norm * stds + means      # denormalize
    
    # Transpose to match original code: (N, H, W) -> (N, W, H)
    u = uv[:, 0].transpose(0, 2, 1)  # (N, W, H)
    v = uv[:, 1].transpose(0, 2, 1)  # (N, W, H)
    
    # Grid spacing (assuming domain [0,1] x [0,1])
    N, W, H = u.shape
    dx = 1.0 / (W - 1)
    dy = 1.0 / (H - 1)
    
    # Central differences (matching your original code exactly)
    # du/dx: derivative in x-direction (last axis H, axis=2)
    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)  # (N, W, H-2)
    
    # dv/dy: derivative in y-direction (middle axis W, axis=1)
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)  # (N, W-2, H)
    
    # Crop to common interior (W-2, H-2)
    du_dx = du_dx[:, 1:-1, :]  # (N, W-2, H-2)
    dv_dy = dv_dy[:, :, 1:-1]  # (N, W-2, H-2)
    
    # Compute divergence
    div = du_dx + dv_dy  # (N, W-2, H-2)
    
    return div



def compute_divergence_stats(div: np.ndarray, name: str = "Field"):
    """
    Compute divergence statistics.
    
    Args:
        div: np.ndarray of shape (N, H-2, W-2) - divergence field
        name: Name for the field (e.g., "Predictions", "Labels")
    
    Returns:
        stats: Dictionary of divergence statistics
    """
    abs_div = np.abs(div)
    
    # Per-sample statistics
    per_sample_mean = abs_div.reshape(div.shape[0], -1).mean(axis=1)
    per_sample_max = abs_div.reshape(div.shape[0], -1).max(axis=1)
    
    # Global statistics
    stats = {
        f'{name.lower()}_mean_abs_div': float(abs_div.mean()),
        f'{name.lower()}_median_abs_div': float(np.median(abs_div)),
        f'{name.lower()}_max_abs_div': float(abs_div.max()),
        f'{name.lower()}_std_abs_div': float(abs_div.std()),
    }
    
    return stats


# =============================================================================
# 6. Evaluate Model (with divergence)
# =============================================================================
def evaluate_model(model, loader_iter, device, num_batches=50, description="Model"):
    """
    Evaluate model using ScOT's metrics (relative L1/L2, absolute L1/L2) and divergence.
    
    Args:
        model: PyTorch model
        loader_iter: Iterator or callable that yields batches
        device: torch device
        num_batches: Number of batches to evaluate
        description: Description for progress bar
    
    Returns:
        metrics: Dictionary containing error metrics and divergence metrics
    """
    print(f"\n[4/5] Evaluating {description}")
    print(f"  Using {num_batches} batches for evaluation")
    
    model.eval()
    
    # Collect all predictions and labels
    all_predictions = []
    all_labels = []
    
    # If loader_iter is a callable (lambda), call it to get the iterator
    if callable(loader_iter):
        loader_iter = loader_iter()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader_iter, total=num_batches, desc="Evaluating")):
            if i >= num_batches:
                break
            
            # Run inference
            output = poseidon_forward(model, batch, device)
            labels = batch["labels"].to(device)
            
            # Move to CPU and convert to numpy
            all_predictions.append(output.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"\n  Collected {predictions.shape[0]} samples")
    print(f"  Shape: {predictions.shape}")
    
    # =============================================================================
    # Compute Error Metrics
    # =============================================================================
    print(f"\n  Computing error metrics...")
    
    # L1 errors
    relative_l1_errors = relative_lp_error(predictions, labels, p=1, return_percent=True)
    l1_errors = lp_error(predictions, labels, p=1)
    
    # L2 errors
    relative_l2_errors = relative_lp_error(predictions, labels, p=2, return_percent=True)
    l2_errors = lp_error(predictions, labels, p=2)
    
    # Compute statistics
    metrics = {
        # Relative L1 (percentage)
        'relative_l1_mean': np.mean(relative_l1_errors),
        'relative_l1_median': np.median(relative_l1_errors),
        'relative_l1_std': np.std(relative_l1_errors),
        'relative_l1_min': np.min(relative_l1_errors),
        'relative_l1_max': np.max(relative_l1_errors),
        
        # Absolute L1
        'l1_mean': np.mean(l1_errors),
        'l1_median': np.median(l1_errors),
        'l1_std': np.std(l1_errors),
        'l1_min': np.min(l1_errors),
        'l1_max': np.max(l1_errors),
        
        # Relative L2 (percentage)
        'relative_l2_mean': np.mean(relative_l2_errors),
        'relative_l2_median': np.median(relative_l2_errors),
        'relative_l2_std': np.std(relative_l2_errors),
        'relative_l2_min': np.min(relative_l2_errors),
        'relative_l2_max': np.max(relative_l2_errors),
        
        # Absolute L2
        'l2_mean': np.mean(l2_errors),
        'l2_median': np.median(l2_errors),
        'l2_std': np.std(l2_errors),
        'l2_min': np.min(l2_errors),
        'l2_max': np.max(l2_errors),
    }
    
    # =============================================================================
    # Compute Divergence Metrics (reusing same predictions/labels)
    # =============================================================================
    print(f"  Computing divergence metrics...")
    
    # Compute divergence for predictions
    div_pred = compute_divergence_numpy(predictions)
    div_pred_stats = compute_divergence_stats(div_pred, name="predictions")
    metrics.update(div_pred_stats)
    
    # Compute divergence for labels
    div_labels = compute_divergence_numpy(labels)
    div_labels_stats = compute_divergence_stats(div_labels, name="labels")
    metrics.update(div_labels_stats)
    
    # Divergence ratio
    div_ratio = metrics['predictions_mean_abs_div'] / (metrics['labels_mean_abs_div'] + 1e-10)
    metrics['divergence_ratio'] = div_ratio
    
    # =============================================================================
    # Print Results
    # =============================================================================
    print(f"\n  {'='*60}")
    print(f"  EVALUATION RESULTS - {description}")
    print(f"  {'='*60}")
    
    # Error metrics
    print(f"\n  Relative L1 Error (%):")
    print(f"    Mean:   {metrics['relative_l1_mean']:.4f}%")
    print(f"    Median: {metrics['relative_l1_median']:.4f}%")
    print(f"    Std:    {metrics['relative_l1_std']:.4f}%")
    print(f"    Range:  [{metrics['relative_l1_min']:.4f}%, {metrics['relative_l1_max']:.4f}%]")
    
    print(f"\n  Absolute L1 Error:")
    print(f"    Mean:   {metrics['l1_mean']:.6f}")
    print(f"    Median: {metrics['l1_median']:.6f}")
    
    print(f"\n  Relative L2 Error (%):")
    print(f"    Mean:   {metrics['relative_l2_mean']:.4f}%")
    print(f"    Median: {metrics['relative_l2_median']:.4f}%")
    
    print(f"\n  Absolute L2 Error:")
    print(f"    Mean:   {metrics['l2_mean']:.6f}")
    print(f"    Median: {metrics['l2_median']:.6f}")
    
    # Divergence metrics
    print(f"\n  {'─'*60}")
    print(f"  DIVERGENCE-FREE CONSTRAINT")
    print(f"  {'─'*60}")
    print(f"\n  Predictions:")
    print(f"    Mean |∇·u|:   {metrics['predictions_mean_abs_div']:.6e}")
    print(f"    Median |∇·u|: {metrics['predictions_median_abs_div']:.6e}")
    print(f"    Max |∇·u|:    {metrics['predictions_max_abs_div']:.6e}")
    
    print(f"\n  Labels (Ground Truth):")
    print(f"    Mean |∇·u|:   {metrics['labels_mean_abs_div']:.6e}")
    print(f"    Median |∇·u|: {metrics['labels_median_abs_div']:.6e}")
    print(f"    Max |∇·u|:    {metrics['labels_max_abs_div']:.6e}")
    
    print(f"\n  Divergence Ratio (pred/label):")
    print(f"    {div_ratio:.2f}x", end="")
    if div_ratio < 1.5:
        print(f"  ✓ Excellent divergence-free preservation!")
    elif div_ratio < 3.0:
        print(f"  ⚠ Moderate divergence increase")
    else:
        print(f"  ✗ Significant divergence violation")
    
    print(f"  {'='*60}\n")
    
    return metrics


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Load model
    model, device = load_poseidon_model(model_path)
    
    # Build loaders
    calib_loader, val_loader, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        calib_batchsize=8,
        calib_steps=8,
        val_batchsize=16,
        val_steps=50
    )
    
    # Test inference
    test_inference(model, calib_loader, device, num_batches=2)
    
    # Evaluate model (baseline) - now includes divergence
    baseline_metrics = evaluate_model(
        model=model,
        loader_iter=val_iter,
        device=device,
        num_batches=50,
        description="Full Precision Baseline"
    )
    
    print("\n" + "=" * 80)
    print("[5/5] INITIALIZATION COMPLETE - Ready for quantization!")
    print("=" * 80)
    print("\nBaseline metrics (errors + divergence) saved for comparison.")
    print("\nNext steps:")
    print("  - Apply PPQ calibration using calib_iter()")
    print("  - Compute data ranges for quantization")
    print("  - Optimize step sizes with MC gradient descent")
    print("  - Evaluate quantized model and compare with baseline")
    print("=" * 80)
    
    # Convert numpy types to Python native types for JSON serialization
    baseline_metrics_json = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in baseline_metrics.items()
    }
    
    # Save baseline metrics
    with open('baseline_metrics.json', 'w') as f:
        json.dump(baseline_metrics_json, f, indent=2)
    print("\n✓ Baseline metrics saved to 'baseline_metrics.json'")
