"""
This file is for basic component testing (model loading, data loading, inference, evaluation),
and now also includes a fake-quant evaluation path that uses PPQ-learned step sizes
to quantize selected Linear layers deterministically.

- We DO NOT change poseidon_forward() or evaluate_model().
- We only create a quantized copy of the model and then call evaluate_model() on it.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import json
from typing import Dict, Tuple, Iterable, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from scOT.model import ScOT
from scOT.problems.base import get_dataset
from scOT.metrics import relative_lp_error, lp_error
from scOT.problems.fluids.normalization_constants import CONSTANTS

from DynamicQ.dynamic_weight import make_dynamic_weight_quantized_copy
from DynamicQ.dynamic_weight_activation import make_dynamic_weight_activation_quantized_copy

# =============================================================================
# Configuration
# =============================================================================
model_path   = "models/NS-PwC-B"
data_path    = "dataset/NS-PwC"
dataset_name = "fluids.incompressible.PiecewiseConstants"

# model_path   = "models/NS-SVS-B"
# data_path    = "dataset/NS-SVS"
# dataset_name = "fluids.incompressible.VortexSheet"

# model_path   = "models/NS-BB-B"
# data_path    = "dataset/NS-BB"
# dataset_name = "fluids.incompressible.BrownianBridge"



# Path helpers (project root = parent of this PPQ file)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))          # e.g. main/PPQ
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))     # e.g. main
INSPECT_DIR  = os.path.join(PROJECT_ROOT, "inspect_layers")        # main/inspect_layers

# Per-model artifacts directory, matching the training script
MODEL_NAME = os.path.basename(model_path.rstrip("/"))              # e.g. "NS-PwC-B"
PPQ_DIR    = os.path.join(PROJECT_ROOT, "ppq_artifacts", MODEL_NAME)

# PPQ step sizes (learned) for this model
PPQ_STEPS_PATH   = os.path.join(PPQ_DIR, "ppq_step_sizes.pt")      # preferred
PPQ_STEPS_JSON   = os.path.join(PPQ_DIR, "ppq_step_sizes.json")    # optional fallback

# Layer list from inspection
QUANT_LAYERS_PATH = os.path.join(INSPECT_DIR, "B_quantize_layers.pt")

QMIN, QMAX       = -127, 127                                       # symmetric int8 emu (zero-point = 0)


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

    print(f"  Loading training data (calibration)...")
    train_ds = get_dataset(dataset_name, which="train",
                           num_trajectories=2048, data_path=data_path)
    print(f"    ✓ Train dataset length: {len(train_ds)} samples")

    print(f"  Loading validation data...")
    try:
        val_ds = get_dataset(dataset_name, which="val",
                             num_trajectories=256, data_path=data_path)
        print(f"    ✓ Val dataset length: {len(val_ds)} samples")
    except Exception:
        print(f"    ! Val split not found, using test split instead")
        val_ds = get_dataset(dataset_name, which="test",
                             num_trajectories=256, data_path=data_path)
        print(f"    ✓ Test dataset length: {len(val_ds)} samples")

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

    def take(loader, steps):
        for i, b in enumerate(loader):
            if i >= steps: break
            yield b

    calib_iter = lambda: take(calib_loader, calib_steps)
    val_iter   = lambda: take(val_loader,   val_steps)
    return calib_loader, val_loader, calib_iter, val_iter


# =============================================================================
# 3. Forward Pass Function (kept unchanged)
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
# 4. Test Inference (kept unchanged)
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

            output = poseidon_forward(model, batch, device)
            print(f"    → Output: {output.shape} {output.dtype}")
            print(f"    → Output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"    ✓ Forward pass successful")

    print(f"\n  ✓ All inference tests passed!")


# =============================================================================
# 5. Divergence helpers (kept unchanged)
# =============================================================================
def compute_divergence_numpy(fields: np.ndarray):
    means = np.array(CONSTANTS["mean"][1:3]).reshape(1, 2, 1, 1)
    stds = np.array(CONSTANTS["std"][1:3]).reshape(1, 2, 1, 1)

    uv_norm = fields[:, 1:3, :, :]  # (N, 2, H, W)
    uv = uv_norm * stds + means

    u = uv[:, 0].transpose(0, 2, 1)  # (N, W, H)
    v = uv[:, 1].transpose(0, 2, 1)  # (N, W, H)

    N, W, H = u.shape
    dx = 1.0 / (W - 1)
    dy = 1.0 / (H - 1)

    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)  # (N, W, H-2)
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)  # (N, W-2, H)

    du_dx = du_dx[:, 1:-1, :]  # (N, W-2, H-2)
    dv_dy = dv_dy[:, :, 1:-1]  # (N, W-2, H-2)
    div = du_dx + dv_dy
    return div


def compute_divergence_stats(div: np.ndarray, name: str = "Field"):
    abs_div = np.abs(div)
    stats = {
        f'{name.lower()}_mean_abs_div': float(abs_div.mean()),
        f'{name.lower()}_median_abs_div': float(np.median(abs_div)),
        f'{name.lower()}_max_abs_div': float(abs_div.max()),
        f'{name.lower()}_std_abs_div': float(abs_div.std()),
    }
    return stats


# =============================================================================
# 6. Evaluate Model (kept unchanged)
# =============================================================================
def evaluate_model(model, loader_iter, device, num_batches=50, description="Model"):
    print(f"\n[4/5] Evaluating {description}")
    print(f"  Using {num_batches} batches for evaluation")

    model.eval()
    all_predictions, all_labels = [], []

    if callable(loader_iter):
        loader_iter = loader_iter()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader_iter, total=num_batches, desc="Evaluating")):
            if i >= num_batches: break
            output = poseidon_forward(model, batch, device)
            labels = batch["labels"].to(device)
            all_predictions.append(output.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"\n  Collected {predictions.shape[0]} samples")
    print(f"  Shape: {predictions.shape}")

    print(f"\n  Computing error metrics...")
    relative_l1_errors = relative_lp_error(predictions, labels, p=1, return_percent=True)
    l1_errors         = lp_error(predictions, labels, p=1)
    relative_l2_errors = relative_lp_error(predictions, labels, p=2, return_percent=True)
    l2_errors         = lp_error(predictions, labels, p=2)

    metrics = {
        'relative_l1_mean': np.mean(relative_l1_errors),
        'relative_l1_median': np.median(relative_l1_errors),
        'relative_l1_std': np.std(relative_l1_errors),
        'relative_l1_min': np.min(relative_l1_errors),
        'relative_l1_max': np.max(relative_l1_errors),

        'l1_mean': np.mean(l1_errors),
        'l1_median': np.median(l1_errors),
        'l1_std': np.std(l1_errors),
        'l1_min': np.min(l1_errors),
        'l1_max': np.max(l1_errors),

        'relative_l2_mean': np.mean(relative_l2_errors),
        'relative_l2_median': np.median(relative_l2_errors),
        'relative_l2_std': np.std(relative_l2_errors),
        'relative_l2_min': np.min(relative_l2_errors),
        'relative_l2_max': np.max(relative_l2_errors),

        'l2_mean': np.mean(l2_errors),
        'l2_median': np.median(l2_errors),
        'l2_std': np.std(l2_errors),
        'l2_min': np.min(l2_errors),
        'l2_max': np.max(l2_errors),
    }

    print(f"  Computing divergence metrics...")
    div_pred = compute_divergence_numpy(predictions)
    div_labels = compute_divergence_numpy(labels)
    metrics.update(compute_divergence_stats(div_pred, name="predictions"))
    metrics.update(compute_divergence_stats(div_labels, name="labels"))
    div_ratio = metrics['predictions_mean_abs_div'] / (metrics['labels_mean_abs_div'] + 1e-10)
    metrics['divergence_ratio'] = div_ratio

    print(f"\n  {'='*60}")
    print(f"  EVALUATION RESULTS - {description}")
    print(f"  {'='*60}")

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
# 7. PPQ Fake-Quant helpers (NEW)
# =============================================================================
def _as_tensor_1d(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32).contiguous()
    return torch.tensor(x, dtype=torch.float32, device=device).contiguous()


def load_ppq_step_sizes(
    path_pt: str = PPQ_STEPS_PATH,
    path_json: str = PPQ_STEPS_JSON,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns a dict: {layer_name: (w_step[out_features], a_step[in_features])}
    Prefers the .pt file; falls back to .json if needed.
    """
    device = torch.device(device)
    if os.path.isfile(path_pt):
        obj = torch.load(path_pt, map_location="cpu")
        step_dict = obj["step_sizes"]
        out = {}
        for name, (w_step, a_step) in step_dict.items():
            out[name] = (_as_tensor_1d(w_step, device), _as_tensor_1d(a_step, device))
        print(f"  ✓ Loaded step sizes from {path_pt} ({len(out)} layers)")
        return out

    # fallback JSON
    if os.path.isfile(path_json):
        with open(path_json, "r") as f:
            obj = json.load(f)
        step_dict = obj["step_sizes"]
        out = {}
        for name, (w_list, a_list) in step_dict.items():
            out[name] = (_as_tensor_1d(w_list, device), _as_tensor_1d(a_list, device))
        print(f"  ✓ Loaded step sizes from {path_json} ({len(out)} layers)")
        return out

    raise FileNotFoundError("Could not find PPQ step size files.")


def load_quantize_layers(path: str = QUANT_LAYERS_PATH) -> Set[str]:
    """
    Load the set of layer names to quantize.

    Expected formats:
      - list / set / tuple of names
      - dict with key 'quantize_layers' or 'layers'
      - other dict: we *don't* treat dict keys as layer names here
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"quantize_layers file not found: {path}")

    obj = torch.load(path, map_location="cpu")

    # Case 1: plain container of names
    if isinstance(obj, (list, set, tuple)):
        names = set(obj)

    # Case 2: dict with explicit list under a key
    elif isinstance(obj, dict):
        if "quantize_layers" in obj:
            names = set(obj["quantize_layers"])
        elif "layers" in obj:
            names = set(obj["layers"])
        else:
            raise ValueError(
                f"quantize_layers.pt is a dict but has no 'quantize_layers' or 'layers' key. "
                f"Keys found: {list(obj.keys())}"
            )
    else:
        # Last-resort fallback
        names = set(obj)

    print(f"  ✓ Loaded quantize layer names from {path} ({len(names)} layers)")
    # Optional: print a few examples to sanity check
    example_names = list(names)[:5]
    print(f"    e.g. {example_names}")
    return names



class QuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that applies symmetric per-channel
    quantize-dequantize on the input (per in-feature) and uses dequantized
    int8-clamped weights (per out-feature). Bias stays in fp32.
    """
    def __init__(self, base_linear: nn.Linear,
                 a_step: torch.Tensor,  # [in_features]
                 w_step: torch.Tensor,  # [out_features]
                 qmin: int = QMIN, qmax: int = QMAX):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        in_features  = base_linear.in_features
        out_features = base_linear.out_features
        assert a_step.numel() == in_features,  f"a_step length {a_step.numel()} != in_features {in_features}"
        assert w_step.numel() == out_features, f"w_step length {w_step.numel()} != out_features {out_features}"

        # register buffers so they move with .to(device)
        self.register_buffer("a_step", a_step.view(1, *([1] * 0), in_features))  # will broadcast on last dim
        self.register_buffer("w_step", w_step.view(out_features, 1))             # broadcast across columns

        # Precompute quantized & dequantized weights
        # W: [out_features, in_features]
        W = base_linear.weight.detach()
        W_int = torch.round(W / self.w_step).clamp(qmin, qmax)
        W_deq = W_int * self.w_step

        self.register_buffer("W_deq", W_deq)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [..., in_features] ; a_step shape: [1, in_features] (broadcast on last dim)
        x_int = torch.round(x / self.a_step).clamp(QMIN, QMAX)
        x_qdq = x_int * self.a_step
        return F.linear(x_qdq, self.W_deq, self.bias)


def _set_module_by_name(root: nn.Module, dotted: str, new_module: nn.Module):
    """
    Replace a submodule given its dotted name (e.g., 'encoder.blocks.0.attn.proj').
    """
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def make_quantized_copy(
    model: nn.Module,
    step_sizes: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    quantize_names: Iterable[str],
    device: Union[str, torch.device] = "cpu"
) -> nn.Module:
    """
    Returns a deep-copied model where selected nn.Linear layers are replaced
    with QuantLinear using the provided per-channel steps.
    """
    device = torch.device(device)
    model_q = copy.deepcopy(model).to(device).eval()

    # build a quick map of name->module
    name_to_module = dict(model_q.named_modules())
    n_replaced, n_skipped = 0, 0

    for name in quantize_names:
        mod = name_to_module.get(name, None)
        steps = step_sizes.get(name, None)
        if mod is None:
            print(f"  ! Layer '{name}' not found in model; skipping")
            n_skipped += 1
            continue
        if not isinstance(mod, nn.Linear):
            print(f"  ! Layer '{name}' exists but is not nn.Linear; skipping")
            n_skipped += 1
            continue
        if steps is None:
            print(f"  ! No step sizes for layer '{name}'; leaving FP")
            n_skipped += 1
            continue

        w_step, a_step = steps
        # ensure device/dtype
        w_step = w_step.to(device=device, dtype=torch.float32)
        a_step = a_step.to(device=device, dtype=torch.float32)

        qlin = QuantLinear(mod, a_step=a_step, w_step=w_step)
        _set_module_by_name(model_q, name, qlin)
        n_replaced += 1

    print(f"  ✓ Quantized copy: replaced {n_replaced} layers; skipped {n_skipped}")
    return model_q


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

    # Test inference (FP)
    test_inference(model, calib_loader, device, num_batches=2)

    # Evaluate model (baseline FP)
    baseline_metrics = evaluate_model(
        model=model,
        loader_iter=val_iter,
        device=device,
        num_batches=50,
        description="Full Precision Baseline"
    )

    # Save baseline metrics
    baseline_metrics_json = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in baseline_metrics.items()
    }
    os.makedirs(PPQ_DIR, exist_ok=True)
    baseline_path = os.path.join(PPQ_DIR, "baseline_metrics.json")
    with open(baseline_path, 'w') as f:
        json.dump(baseline_metrics_json, f, indent=2)
    print(f"\n✓ Baseline metrics saved to '{baseline_path}'")


    print("\n" + "=" * 80)
    print("[5/5] INITIALIZATION COMPLETE - Building PPQ Fake-Quant model…")
    print("=" * 80)

    # ---- Build quantized copy using saved step sizes + quantize_layers list
    # Load steps (prefer .pt -> tensors; fallback .json)
    try:
        step_sizes = load_ppq_step_sizes(device=device)
    except FileNotFoundError:
        # If .pt not present, try JSON-only workflow
        step_sizes = load_ppq_step_sizes(path_pt="__missing__", path_json=PPQ_STEPS_JSON, device=device)

    # Load which Linear layers to quantize
    quantize_names = load_quantize_layers(QUANT_LAYERS_PATH)

    # Create quantized copy
    model_q = make_quantized_copy(model, step_sizes, quantize_names, device=device)

    # Evaluate quantized model (same eval pipeline, unchanged)
    quant_metrics = evaluate_model(
        model=model_q,
        loader_iter=val_iter,
        device=device,
        num_batches=50,
        description="PPQ Fake-Quant (int8 emu)"
    )


    # ============================================================
    # Build & evaluate DynamicQ (weight-only) quantized model
    # ============================================================
    print("\n" + "=" * 80)
    print("[6/6] Building DynamicQ weight-only model…")
    print("=" * 80)

    model_dyn = make_dynamic_weight_quantized_copy(
        model=model,
        quantize_names=quantize_names,
        bitwidth=8,
        device=device,
    )

    dyn_metrics = evaluate_model(
        model=model_dyn,
        loader_iter=val_iter,
        device=device,
        num_batches=50,
        description="DynamicQ Weight-Only (int8 emu)"
    )


    # ============================================================
    # Build & evaluate DynamicQ (weight + activation) quantized model
    # ============================================================
    print("\n" + "=" * 80)
    print("[7/7] Building DynamicQ weight+activation model…")
    print("=" * 80)

    model_dyn_wa = make_dynamic_weight_activation_quantized_copy(
        model=model,
        quantize_names=quantize_names,
        bitwidth=8,
        device=device,
    )

    dyn_wa_metrics = evaluate_model(
        model=model_dyn_wa,
        loader_iter=val_iter,
        device=device,
        num_batches=50,
        description="DynamicQ Weight+Activation (int8 emu)"
    )



    # -----------------------------
    # Compare all 4: baseline vs PPQ vs DynamicQ (w) vs DynamicQ (w+act)
    # -----------------------------
    keys_to_compare = [
        "l1_mean",
        "relative_l1_mean",
        "l2_mean",
        "relative_l2_mean",
        "divergence_ratio",
    ]

    print("\n" + "=" * 80)
    print("COMPARISON: FP vs PPQ vs DynamicQ(w) vs DynamicQ(w+act)")
    print("=" * 80)

    comparison = {}
    for k in keys_to_compare:
        base_val   = float(baseline_metrics[k])
        ppq_val    = float(quant_metrics[k])      # PPQ metrics
        dyn_w_val  = float(dyn_metrics[k])        # Dynamic weight-only
        dyn_wa_val = float(dyn_wa_metrics[k])     # Dynamic weight + activation

        ppq_diff   = ppq_val   - base_val
        dyn_w_diff = dyn_w_val - base_val
        dyn_wa_diff= dyn_wa_val- base_val

        ppq_ratio   = ppq_val   / (base_val + 1e-12)
        dyn_w_ratio = dyn_w_val / (base_val + 1e-12)
        dyn_wa_ratio= dyn_wa_val/ (base_val + 1e-12)

        comparison[k] = {
            "baseline": base_val,
            "ppq": ppq_val,
            "dynamic_weight": dyn_w_val,
            "dynamic_weight_activation": dyn_wa_val,
            "ppq_diff": ppq_diff,
            "ppq_ratio": ppq_ratio,
            "dynamic_weight_diff": dyn_w_diff,
            "dynamic_weight_ratio": dyn_w_ratio,
            "dynamic_weight_activation_diff": dyn_wa_diff,
            "dynamic_weight_activation_ratio": dyn_wa_ratio,
        }

        print(f"\nMetric: {k}")
        print(f"  Baseline              : {base_val:.6f}")
        print(f"  PPQ                   : {ppq_val:.6f}  "
            f"(diff {ppq_diff:+.6f}, ratio {ppq_ratio:.4f}x)")
        print(f"  DynamicQ (weight-only): {dyn_w_val:.6f}  "
            f"(diff {dyn_w_diff:+.6f}, ratio {dyn_w_ratio:.4f}x)")
        print(f"  DynamicQ (w+act)      : {dyn_wa_val:.6f}  "
            f"(diff {dyn_wa_diff:+.6f}, ratio {dyn_wa_ratio:.4f}x)")

    print("\n" + "=" * 80)
    print("SUMMARY (ratio vs baseline):")
    for k in keys_to_compare:
        r_ppq   = comparison[k]["ppq_ratio"]
        r_dw    = comparison[k]["dynamic_weight_ratio"]
        r_dwa   = comparison[k]["dynamic_weight_activation_ratio"]
        print(f"  {k:23s}: PPQ={r_ppq:.4f}x, DynW={r_dw:.4f}x, DynW+Act={r_dwa:.4f}x")
    print("=" * 80 + "\n")



    # --------------------------------------------------------
    # Save all metrics (+ comparison) in one JSON
    # --------------------------------------------------------
    ppq_metrics_json = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in quant_metrics.items()
    }
    dyn_w_metrics_json = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in dyn_metrics.items()
    }
    dyn_wa_metrics_json = {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in dyn_wa_metrics.items()
    }

    out_json = {
        "ppq_metrics": ppq_metrics_json,
        "dynamic_weight_metrics": dyn_w_metrics_json,
        "dynamic_weight_activation_metrics": dyn_wa_metrics_json,
        "comparison_vs_baseline": comparison,
    }

    quant_path = os.path.join(PPQ_DIR, "quant_metrics.json")
    with open(quant_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"\n✓ Quantized metrics (PPQ + DynamicW + DynamicW+Act) + comparison saved to '{quant_path}'")

    print("\nComparison ready ✔")

