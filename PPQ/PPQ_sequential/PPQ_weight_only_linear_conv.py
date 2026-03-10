"This file is for applying the final PPQ to our poseidon model"

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scOT.model import ScOT
import torch
from scOT.problems.base import get_dataset
from torch.utils.data import DataLoader
import torch.nn as nn  
import torch.optim as optim
import json
from scOT.metrics import relative_lp_error, lp_error
import numpy as np


# helper alias for the quantizable layers
QMODULES = (nn.Linear, nn.Conv2d)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # main/PPQ
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # main
INSPECT_DIR = os.path.join(PROJECT_ROOT, "inspect_layers")       # main/inspect_layers



model_path   = "models/NS-PwC"
data_path    = "dataset/NS-PwC"
dataset_name = "fluids.incompressible.PiecewiseConstants"

# ==================================================================================================================================
# ==================================================================================================================================
# ==================================================================================================================================
# PPQ Functions
# ==================================================================================================================================
# ==================================================================================================================================
# ==================================================================================================================================


def load_poseidon_model(model_path: str, device: str = "cuda"):
    """
    Load the Poseidon ScOT model for quantization.
    
    Args:
        model_path: Path to pretrained model
        device: Target device ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model in eval mode
        device: Torch device object
    """
    # Ensure device is available
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ScOT.from_pretrained(model_path).to(device)
    model.eval()  # Set to eval mode for quantization
    
    # Optional: Set matmul precision for stability
    torch.set_float32_matmul_precision("high")
    
    print(f"Model loaded on device: {device}")
    print(f"Model type: {type(model)}")
    
    return model, device



def build_poseidon_loaders(
    dataset_name: str,
    data_path: str,
    calib_batchsize: int = 8,
    calib_steps: int = 8,
    val_batchsize: int = 16,
    val_steps: int = 50
):
    # Load train split (all time pairs)
    train_ds = get_dataset(dataset_name, which="train",
                           num_trajectories=2048, data_path=data_path)
    # Load validation split
    try:
        val_ds = get_dataset(dataset_name, which="val",
                             num_trajectories=256, data_path=data_path)
    except Exception:
        val_ds = get_dataset(dataset_name, which="test",
                             num_trajectories=256, data_path=data_path)

    calib_loader = DataLoader(
        train_ds, batch_size=calib_batchsize, shuffle=True,
        num_workers=min(os.cpu_count() or 0, 16), pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batchsize, shuffle=False,
        num_workers=min(os.cpu_count() or 0, 16), pin_memory=True
    )

    # Optional: restrict the number of batches for fast calibration/eval
    def take(loader, steps):                             
        for i, b in enumerate(loader):
            if i >= steps: break
            yield b

    calib_iter = lambda: take(calib_loader, calib_steps)
    val_iter   = lambda: take(val_loader,   val_steps)

    return calib_loader, val_loader, calib_iter, val_iter



def poseidon_forward(model, batch, device):
    x = batch["pixel_values"].to(device)
    t = batch["time"].to(device)  # Changed
    pm = batch["pixel_mask"].to(device)  # Changed
    y = batch.get("labels")  # labels is optional (only during training)
    
    out = model(
        pixel_values=x,
        time=t,  # Changed
        pixel_mask=pm,  # Changed
        labels=(y.to(device) if y is not None else None),  # labels can still be None
    )
    return out.output


'''
The following is the implementation of PPQ on poseidon
'''

# ========================
# Add Quantization Noise (Adapted for Poseidon)
# ========================

def add_quantization_noise(tensor, step_sizes, channel_axis):
    """
    Add uniform quantization noise to tensor for Monte Carlo simulation.
    
    Each unique step size corresponds to one output channel.
    For Linear layers: all elements in a channel share the same step size.
    
    Args:
        tensor: Input tensor (weights or activations)
                - Weights: shape [out_features, in_features], channel_axis=0
                - Activations: shape [batch, seq, features], channel_axis=-1 (last dim)
        step_sizes: Per-channel step sizes, shape [num_channels]
        channel_axis: Which dimension represents channels
                      - For weights: 0 (output channels)
                      - For activations: -1 (features)
    
    Returns:
        Noisy tensor: tensor + uniform noise scaled by step_sizes
    
    Example:
        tensor shape: [16, 256, 192] (batch, seq, features)
        step_sizes shape: [192] (one per feature channel)
        channel_axis: -1
        
        → noise shape: [1, 1, 192] broadcasted over all batch/seq positions
        → result: [16, 256, 192] with noise scaled by step_sizes per feature
    """

    assert step_sizes.numel() == tensor.shape[channel_axis], f"step_sizes ({step_sizes.numel()}) does not match tensor shape at channel_axis ({tensor.shape[channel_axis]})"
    # Create broadcast shape: put 1s everywhere except channel axis
    shape = [1] * tensor.dim()
    shape[channel_axis] = tensor.size(channel_axis)

    # print(f"[DEBUG1] tensor.shape = {tensor.shape}, step_sizes.shape = {step_sizes.shape}, channel_axis = {channel_axis}")
    # print(f"[DEBUG2] intended shape for broadcast = {shape}")

    
    # Reshape step_sizes to match broadcast shape
    step_sizes_broadcast = step_sizes.view(shape)
    
    # Sample uniform noise in [-0.5, 0.5] LSB
    # This represents ±0.5 quantization error from rounding
    # This step can be improved later by setting the value of dynamic quantization
    noise = (torch.rand_like(tensor) - 0.5) * step_sizes_broadcast
    
    return tensor + noise


# ========================
# Adapted get_clean_output Functions for Poseidon
# ========================
def get_clean_outputs_poseidon(model, dataloader, device, layer_names):
    """
    Run the clean model and store both input (pre-op) and output (post-op) activations
    for each target layer (Linear or Conv2d).

    Returns:
        clean_inputs:  {layer_name: [X_batch_0, X_batch_1, ...], ...}
        clean_outputs: {layer_name: [Y_batch_0, Y_batch_1, ...], ...}

    For:
      - Linear: 
          X_pre shape ~ [B, seq, in_features] or [B, in_features]
          Y_post shape ~ [B, seq, out_features] or [B, out_features]
      - Conv2d:
          X_pre shape ~ [B, C_in, H, W]
          Y_post shape ~ [B, C_out, H, W]
    """
    model.eval()

    clean_inputs  = {name: [] for name in layer_names}
    clean_outputs = {name: [] for name in layer_names}

    # allow callable iterator (e.g., calib_iter)
    if callable(dataloader):
        dataloader = dataloader()

    name2mod = dict(model.named_modules())

    with torch.inference_mode():
        any_batch = False
        for batch_idx, batch in enumerate(dataloader):
            any_batch = True
            x  = batch["pixel_values"].to(device)
            t  = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y  = batch.get("labels", None)

            if t  is not None: t  = t.to(device)
            if pm is not None: pm = pm.to(device)
            if y  is not None: y  = y.to(device)

            layer_io = {}  # layer_name -> (X_pre, Y_post)

            def get_hook(name):
                def hook(mod, inp, out):
                    X_pre = inp[0]
                    Y_post = out

                    # Sanity checks depending on module type
                    if isinstance(mod, nn.Linear):
                        if hasattr(mod, "in_features"):
                            if X_pre.shape[-1] != mod.in_features:
                                print(
                                    f"[WARN] {name}: X_pre last dim {X_pre.shape[-1]} "
                                    f"!= in_features {mod.in_features}"
                                )
                        if hasattr(mod, "out_features"):
                            if Y_post.shape[-1] != mod.out_features:
                                print(
                                    f"[WARN] {name}: Y_post last dim {Y_post.shape[-1]} "
                                    f"!= out_features {mod.out_features}"
                                )

                    elif isinstance(mod, nn.Conv2d):
                        # Expect X: [B, C_in, H, W], Y: [B, C_out, H, W]
                        if X_pre.dim() == 4:
                            if X_pre.shape[1] != mod.in_channels:
                                print(
                                    f"[WARN] {name}: X_pre C={X_pre.shape[1]} "
                                    f"!= in_channels {mod.in_channels}"
                                )
                        if Y_post.dim() == 4:
                            if Y_post.shape[1] != mod.out_channels:
                                print(
                                    f"[WARN] {name}: Y_post C={Y_post.shape[1]} "
                                    f"!= out_channels {mod.out_channels}"
                                )

                    layer_io[name] = (
                        X_pre.detach().cpu(),
                        Y_post.detach().cpu(),
                    )
                return hook

            handles = []
            for name, mod in model.named_modules():
                if name in layer_names and isinstance(mod, QMODULES):
                    handles.append(mod.register_forward_hook(get_hook(name)))

            _ = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )

            for name in layer_names:
                if name in layer_io:
                    X_pre, Y_post = layer_io[name]
                    clean_inputs[name].append(X_pre)
                    clean_outputs[name].append(Y_post)
                else:
                    clean_inputs[name].append(None)
                    clean_outputs[name].append(None)

            for h in handles:
                h.remove()

    if any_batch:
        print(f"\n✓ Collected clean inputs/outputs for {len(layer_names)} layers from {batch_idx+1} batches")
        active_layers = sum(
            1 for outs in clean_outputs.values() if outs and outs[0] is not None
        )
        print(f"✓ {active_layers}/{len(layer_names)} layers produced outputs")
    else:
        print("\n[WARN] get_clean_outputs_poseidon: dataloader yielded no batches.")

    return clean_inputs, clean_outputs







# ========================
# Adapted: Compute Data Ranges for Poseidon
# ========================

def compute_data_ranges_poseidon(model, dataloader, device, layer_names, percentile_prob=1e-4):
    """
    Compute the data range [R_l]_k for each channel using percentile-based clipping.
    Adapted for Poseidon with dictionary batches and flexible activation shapes.
    
    Args:
        model: Poseidon/ScOT model
        dataloader: DataLoader or callable that yields dictionary batches
        device: torch device
        layer_names: List of layer names to compute ranges for
        percentile_prob: Probability for percentile clipping (default 1e-4)
    
    Returns:
        ranges_dict: Dictionary mapping layer names to weight/activation ranges
    """
    model.eval()
    ranges_dict = {}
    
    # Pre-compute the erf_inv constant
    z = torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
        torch.tensor(1.0 - 2.0 * percentile_prob, device=device)
    )
    erf_inv_value = float(z)
    
    with torch.no_grad():
        # ==============================
        # 1. Weight Ranges (Percentile-based)
        # ==============================
        print(f"Computing weight ranges...")
        for name, module in model.named_modules():
            if name not in layer_names:
                continue

            # ---- Linear weights: [out_features, in_features] ----
            if isinstance(module, nn.Linear):
                weight = module.weight.data.to(device)              # [out_f, in_f]
                w_flat = weight.view(weight.size(0), -1)            # [out_f, *]

            # ---- Conv2d weights: [out_channels, in_channels, kH, kW] ----
            elif isinstance(module, nn.Conv2d):
                weight = module.weight.data.to(device)              # [out_c, in_c, kH, kW]
                w_flat = weight.view(weight.size(0), -1)            # [out_c, in_c*kH*kW]

            else:
                # Not a quantizable module (we only care about Linear / Conv2d here)
                continue

            # Compute μ and σ per output channel
            w_mean = w_flat.mean(dim=1)             # [out_channels]
            w_std  = w_flat.std(dim=1, unbiased=False)

            # Percentile threshold τ
            tau  = w_mean + w_std * erf_inv_value

            # Symmetric clipping thresholds
            beta = tau
            alpha = 2 * w_mean - beta

            # Range = beta - alpha
            weight_ranges = (beta - alpha).clamp(min=1e-8)

            # Store per-layer info
            ranges_dict[name] = {
                "weight_ranges": weight_ranges.to(device),
                "act_stats": []
            }

        print(f"✓ Computed weight ranges for {len(ranges_dict)} layers")

        
        # ==============================
        # 2. Activation Ranges (Percentile-based)
        # ==============================
        print(f"Computing activation ranges...")
        
        # Get iterator if callable
        if callable(dataloader):
            dataloader = dataloader()
        
        for batch_idx, batch in enumerate(dataloader):
            # Dictionary-based batch extraction
            x = batch["pixel_values"].to(device)
            t = batch["time"].to(device)
            pm = batch["pixel_mask"].to(device)
            y = batch["labels"].to(device)
            
            def get_hook(name, module_ref):
                def hook(module, input, output):
                    x = input[0]  # Pre-op activation

                    # Handle different activation shapes
                    # We want shape [channels, N] for stats, where N is "all other dims"
                    if x.dim() == 2:
                        # [B, C] -> [C, B]
                        x_flat = x.transpose(0, 1)  # [C, B]

                    elif x.dim() == 3:
                        # [B, seq, C] -> [C, B*seq]
                        x_flat = x.permute(2, 0, 1).reshape(x.size(-1), -1)

                    elif x.dim() == 4:
                        # Two main cases:
                        #   - Conv2d (channels-first): [B, C, H, W]
                        #   - Generic last-channel layout: [B, H, W, C]
                        if isinstance(module_ref, nn.Conv2d):
                            # Conv-style [B, C, H, W] -> [C, B*H*W]
                            x_flat = x.permute(1, 0, 2, 3).reshape(x.size(1), -1)
                        else:
                            # Fallback: [B, H, W, C] -> [C, B*H*W]
                            x_flat = x.permute(3, 0, 1, 2).reshape(x.size(-1), -1)

                    else:
                        # Fallback: assume last dim is channels
                        # [*, C] -> [C, *]
                        x_flat = x.reshape(-1, x.size(-1)).transpose(0, 1)

                    # Compute μ and σ per channel
                    x_mean = x_flat.mean(dim=1)  # [channels]
                    x_std  = x_flat.std(dim=1, unbiased=False)

                    # Store statistics for later aggregation
                    ranges_dict[name]['act_stats'].append({
                        'mean': x_mean.cpu(),
                        'std':  x_std.cpu(),
                    })
                return hook

            
            # Register hooks for both Linear and Conv2d
            handles = []
            for name, module in model.named_modules():
                if name in layer_names and isinstance(module, QMODULES):
                    handles.append(module.register_forward_hook(get_hook(name, module)))
            
            # Poseidon forward pass
            _ = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            
            # Clean up hooks
            for h in handles:
                h.remove()
        
        print(f"✓ Collected activation stats from {batch_idx+1} batches")
        
        # ==============================
        # 3. Aggregate Activation Stats and Compute Ranges
        # ==============================
        print(f"Aggregating activation statistics...")
        for name in ranges_dict:
            if not ranges_dict[name]['act_stats']:
                print(f"  Warning: No activation stats for {name}")
                continue
            
            # Average μ and σ across batches
            all_means = torch.stack([s['mean'] for s in ranges_dict[name]['act_stats']])
            all_stds = torch.stack([s['std'] for s in ranges_dict[name]['act_stats']])
            
            avg_mean = all_means.mean(dim=0).to(device)
            avg_std = all_stds.mean(dim=0).to(device)
            
            # Compute percentile threshold τ
            tau = avg_mean + avg_std * erf_inv_value
            
            # Clipping thresholds
            beta = tau
            alpha = 2 * avg_mean - tau
            
            # Range = beta - alpha
            activation_ranges = (beta - alpha).clamp(min=1e-8)
            
            # Store final ranges
            ranges_dict[name]['activation_ranges'] = activation_ranges
            del ranges_dict[name]['act_stats']  # Clean up
        
        print(f"✓ Computed activation ranges for {len(ranges_dict)} layers")
    
    return ranges_dict


# ========================
# Adapted: Evaluate Quantized Model for Poseidon (for finding the best percentile)
# ========================

def evaluate_quantized_model_poseidon(model, dataloader, ranges, device, layer_names, num_bits=8):
    """
    Evaluate with fake quantization on both weights and activations.
    Adapted for Poseidon with detailed debugging.
    
    Returns:
        loss: Average prediction error (MSE-based)
    """
    model.eval()
    total_loss = 0.0
    num_levels = 2 ** num_bits - 1
    count = 0
    
    # DEBUG: Track quantization application
    quantization_applied = False
    hook_fire_count = 0
    
    # Get iterator if callable
    if callable(dataloader):
        dataloader = dataloader()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch["pixel_values"].to(device)
            t = batch["time"].to(device)
            pm = batch["pixel_mask"].to(device)
            y = batch["labels"].to(device)
            
            # Apply fake quantization via hooks
            handles = []
            
            def get_quantize_hook(name, w_ranges, a_ranges):
                def hook(mod, inp, out):
                    nonlocal quantization_applied, hook_fire_count
                    quantization_applied = True
                    hook_fire_count += 1
                    
                    x = inp[0]
                    
                    # DEBUG: Print first hook info
                    if hook_fire_count == 1:
                        print(f"\n  [DEBUG] First hook fired: {name}")
                        print(f"    Input shape: {x.shape}")
                        print(f"    Weight shape: {mod.weight.shape}")
                        print(f"    Activation ranges shape: {a_ranges.shape}")
                        print(f"    Weight ranges shape: {w_ranges.shape}")
                    
                    # Quantize input activations (per-channel)
                    if x.dim() == 2:
                        a_step = a_ranges.view(1, -1) / num_levels
                        x_quantized = torch.round(x / a_step) * a_step
                    elif x.dim() == 3:
                        a_step = a_ranges.view(1, 1, -1) / num_levels
                        x_quantized = torch.round(x / a_step) * a_step
                    elif x.dim() == 4:
                        a_step = a_ranges.view(1, 1, 1, -1) / num_levels
                        x_quantized = torch.round(x / a_step) * a_step
                    else:
                        x_quantized = x
                    
                    # DEBUG: Check quantization effect
                    if hook_fire_count == 1:
                        print(f"    Max activation change: {(x - x_quantized).abs().max():.6f}")
                    
                    # Quantize weights (per output channel)
                    w_flat = mod.weight.view(mod.weight.size(0), -1)
                    w_step = w_ranges.view(-1, 1) / num_levels
                    w_quantized = torch.round(w_flat / w_step) * w_step
                    w_quantized = w_quantized.view_as(mod.weight)
                    
                    # DEBUG: Check weight quantization
                    if hook_fire_count == 1:
                        print(f"    Max weight change: {(mod.weight - w_quantized).abs().max():.6f}")
                    
                    # Compute output with quantized inputs and weights
                    y = torch.nn.functional.linear(x_quantized, w_quantized, mod.bias)
                    
                    return y
                return hook
            
            # Register hooks
            for name, module in model.named_modules():
                if name in layer_names and isinstance(module, nn.Linear) and name in ranges:
                    w_ranges = ranges[name]['weight_ranges']
                    '''
                    a_ranges is:

                    Computed per channel (dimension: [num_channels])

                    Aggregated across all calibration batches by averaging the statistics

                    One fixed value per channel used for all evaluation
                    '''
                    a_ranges = ranges[name]['activation_ranges']
                    handles.append(module.register_forward_hook(
                        get_quantize_hook(name, w_ranges, a_ranges)
                    ))
            
            # DEBUG: Print hook count
            if batch_idx == 0:
                print(f"  [DEBUG] Registered {len(handles)} hooks")
            
            # Forward pass
            outputs = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            
            # Compute loss
            pred = outputs.output
            loss = torch.nn.functional.mse_loss(pred, y)
            
            # DEBUG: Print first batch details
            if batch_idx == 0:
                print(f"  [DEBUG] Prediction shape: {pred.shape}")
                print(f"  [DEBUG] Label shape: {y.shape}")
                print(f"  [DEBUG] MSE Loss: {loss.item():.8f}")
                print(f"  [DEBUG] Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
                print(f"  [DEBUG] Label range: [{y.min():.4f}, {y.max():.4f}]")
                print(f"  [DEBUG] Max diff: {(pred - y).abs().max():.6f}")
                print(f"  [DEBUG] Quantization applied: {quantization_applied}")
                print(f"  [DEBUG] Total hook fires: {hook_fire_count}")
            
            total_loss += loss.item()
            count += 1
            
            for h in handles:
                h.remove()
    
    avg_loss = total_loss / count if count > 0 else float('inf')
    print(f"  [DEBUG] Final average loss: {avg_loss:.8f}")
    
    return avg_loss



# ========================
# Adapted: Find Best Percentile for Poseidon
# ========================

def find_best_percentile_poseidon(model, dataloader, device, layer_names,
                                   percentile_candidates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
    """
    Grid search to find the best percentile probability for Poseidon.
    
    Returns:
        best_percentile: The P value that gives lowest prediction error
    """
    best_loss = float('inf')
    best_percentile = percentile_candidates[0]
    
    print("\n" + "="*80)
    print("SEARCHING FOR BEST PERCENTILE")
    print("="*80)
    
    for P in percentile_candidates:
        print(f"\nTrying percentile P={P:.1e}...")
        
        # Compute ranges with this P
        ranges = compute_data_ranges_poseidon(model, dataloader, device, layer_names, percentile_prob=P)
        
        # Evaluate quantized model
        loss = evaluate_quantized_model_poseidon(model, dataloader, ranges, device, layer_names)
        
        print(f"  → P={P:.1e}: Loss={loss:.6f}")
        
        if loss < best_loss:
            best_loss = loss
            best_percentile = P
    
    print(f"\n✓ Best percentile: {best_percentile:.1e} (Loss: {best_loss:.6f})")
    print("="*80 + "\n")
    
    return best_percentile



# ========================
# Adapted: Compute the MDL prior
# ========================
def compute_mdl_prior(step_sizes_dict, ranges_dict, gamma=0.001, eps=1e-8):
    """
    MDL prior: gamma * sum_k log2(R_k / S_k), **only over weight step sizes**.

    We intentionally DO NOT regularize activation step sizes here, because MDL
    is about model description length (stored weights), not temporary activations.

    - step_sizes_dict: {layer_name: (weight_step_sizes, activation_step_sizes)}
    - ranges_dict: same as from compute_data_ranges_poseidon
    """
    # Find a device from any parameter
    try:
        some_param = next(p for pair in step_sizes_dict.values() for p in pair if p is not None)
        device = some_param.device
    except StopIteration:
        device = torch.device('cpu')

    prior_loss = torch.zeros((), device=device)

    for name, (weight_step_sizes, activation_step_sizes) in step_sizes_dict.items():
        rec = ranges_dict.get(name)
        if rec is None:
            continue

        # --- MDL PRIOR ON WEIGHTS ONLY ---
        w_ranges = rec.get('weight_ranges', None)
        if w_ranges is None:
            continue

        w_ranges = w_ranges.to(device)
        assert w_ranges.numel() == weight_step_sizes.numel(), \
            f"[{name}] weight: {w_ranges.shape} vs step: {weight_step_sizes.shape}"

        w_term = torch.log2(
            torch.clamp(w_ranges, min=eps) /
            torch.clamp(weight_step_sizes, min=eps)
        )
        prior_loss = prior_loss + gamma * torch.sum(w_term)

        # NOTE: we do NOT add any prior on activation_step_sizes here.

    return prior_loss



# ========================
# Adapted: Compute the mc loss(likelihood)
# ========================

def compute_mc_loss_single_batch(model,
                                 step_sizes_dict,
                                 clean_inputs,
                                 clean_outputs,
                                 batch_idx: int,
                                 num_mc_samples: int = 10,
                                 eta: float = 1e-4,
                                 device: str = 'cpu'):
    """
    Monte Carlo likelihood loss (PPQ) for a SINGLE calibration batch.

    - Uses cached clean_inputs / clean_outputs at index `batch_idx`
    - Averages over all target layers that are compatible
    - Supports:
        * nn.Linear   (X: [..., in_features])
        * nn.Conv2d   (X: [B, C_in, H, W])
    """
    if model is not None:
        model.eval()

    # Build a stable layer list that exists in both dicts
    target_layers = [
        name for name in step_sizes_dict.keys()
        if name in clean_inputs and name in clean_outputs
    ]

    if not target_layers:
        raise ValueError("No overlapping layers between step_sizes_dict and clean IO caches.")

    name2module = dict(model.named_modules())

    batch_loss = None
    layer_contrib = 0

    for name in target_layers:
        module = name2module.get(name)
        if module is None:
            continue

        Xb = clean_inputs[name][batch_idx]
        Yb = clean_outputs[name][batch_idx]
        if (Xb is None) or (Yb is None):
            # layer didn't fire for this batch
            continue

        X_clean = Xb.to(device)
        Y_clean = Yb.to(device)

        # shared: step sizes
        w_step, a_step = step_sizes_dict[name]
        w_step = w_step.to(device)
        # a_step is still unused (weights-only noise)

        mc_losses = []

        # ------------------------------------------------------------------
        # Case 1: Linear layer
        # ------------------------------------------------------------------
        if isinstance(module, torch.nn.Linear):
            W_clean = module.weight.to(device)

            # Shape checks
            assert X_clean.shape[-1] == W_clean.shape[1], \
                f"{name}: X in_features {X_clean.shape[-1]} vs W in_features {W_clean.shape[1]}"
            assert w_step.numel() == W_clean.shape[0], \
                f"{name}: w_step {w_step.numel()} vs out_features {W_clean.shape[0]}"

            for _ in range(num_mc_samples):
                X_noisy = X_clean  # no activation noise (yet)

                # Per-output-channel weight noise (channel_axis=0)
                W_noisy = add_quantization_noise(W_clean, w_step, channel_axis=0)

                # Forward through quantized Linear
                Y_noisy = torch.nn.functional.linear(X_noisy, W_noisy, module.bias)

                loss_elem = torch.mean((Y_noisy - Y_clean) ** 2) / (2 * eta)
                mc_losses.append(loss_elem)

        # ------------------------------------------------------------------
        # Case 2: Conv2d layer
        # ------------------------------------------------------------------
        elif isinstance(module, torch.nn.Conv2d):
            W_clean = module.weight.to(device)

            # Shape checks: w_step matches out_channels
            assert w_step.numel() == W_clean.shape[0], \
                f"{name}: w_step {w_step.numel()} vs out_channels {W_clean.shape[0]}"

            for _ in range(num_mc_samples):
                X_noisy = X_clean  # no activation noise

                # Per-output-channel weight noise: [out_c, in_c, kH, kW], channel_axis=0
                W_noisy = add_quantization_noise(W_clean, w_step, channel_axis=0)

                # Forward through quantized Conv2d
                Y_noisy = torch.nn.functional.conv2d(
                    X_noisy,
                    W_noisy,
                    bias=module.bias,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                )

                # Sanity: shapes should match cached clean output
                if Y_noisy.shape != Y_clean.shape:
                    raise RuntimeError(
                        f"{name}: Y_noisy.shape={Y_noisy.shape} "
                        f"!= Y_clean.shape={Y_clean.shape}"
                    )

                loss_elem = torch.mean((Y_noisy - Y_clean) ** 2) / (2 * eta)
                mc_losses.append(loss_elem)

        # ------------------------------------------------------------------
        # Other layer types: skip
        # ------------------------------------------------------------------
        else:
            continue

        if not mc_losses:
            # something went wrong for this layer
            continue

        layer_loss = torch.stack(mc_losses).mean()  # avg over MC samples

        batch_loss = layer_loss if batch_loss is None else (batch_loss + layer_loss)
        layer_contrib += 1

    if layer_contrib == 0:
        # no valid layers for this batch
        return torch.zeros((), device=device)

    # average over layers for this batch
    batch_loss = batch_loss / layer_contrib
    return batch_loss






def compute_mc_loss_with_prior(
    model,
    step_sizes_dict,
    clean_inputs,
    clean_outputs,
    ranges_dict,
    batch_idx: int,
    num_mc_samples: int = 10,
    eta: float = 1e-4,
    gamma: float = 0.005,
    device: str = "cpu",
):
    """
    Monte Carlo likelihood loss (PPQ) + MDL prior **for a single batch**.

    Args:
        model:           Poseidon model
        step_sizes_dict: {layer_name: (w_step, a_step)} learnable step sizes
        clean_inputs:    {layer_name: [X_batch_0, X_batch_1, ...]}
        clean_outputs:   {layer_name: [Y_batch_0, Y_batch_1, ...]}
        ranges_dict:     output of compute_data_ranges_poseidon
        batch_idx:       which cached calibration batch to use
        num_mc_samples:  MC samples per layer
        eta:             noise variance scaling
        gamma:           MDL prior weight
        device:          torch device

    Returns:
        total_loss:      likelihood + prior (tensor with grad)
        likelihood_loss: MC likelihood term (tensor with grad)
        prior_loss:      MDL prior term (tensor with grad)
    """
    # Likelihood: single-batch MC loss using cached clean IO
    likelihood_loss = compute_mc_loss_single_batch(
        model=model,
        step_sizes_dict=step_sizes_dict,
        clean_inputs=clean_inputs,
        clean_outputs=clean_outputs,
        batch_idx=batch_idx,
        num_mc_samples=num_mc_samples,
        eta=eta,
        device=device,
    )

    # MDL prior on step sizes (uses ranges_dict, independent of batch_idx)
    prior_loss = compute_mdl_prior(
        step_sizes_dict=step_sizes_dict,
        ranges_dict=ranges_dict,
        gamma=gamma,
    )

    total_loss = likelihood_loss + prior_loss
    return total_loss, likelihood_loss, prior_loss





def build_channel_param_weights(model: nn.Module, layer_names):
    """
    For each quantized layer, build a 1D tensor of length = #output channels,
    where each element is the number of weights controlled by that channel.

    - Linear: n_{l,k} = in_features
    - Conv2d: n_{l,k} = in_channels * kernel_height * kernel_width
    """
    name2mod = dict(model.named_modules())
    channel_weights = {}

    for name in layer_names:
        mod = name2mod.get(name, None)

        if isinstance(mod, nn.Linear):
            in_f  = mod.in_features
            out_f = mod.out_features
            w = torch.full((out_f,), float(in_f))

        elif isinstance(mod, nn.Conv2d):
            in_c  = mod.in_channels
            out_c = mod.out_channels
            k_h, k_w = mod.kernel_size
            per_channel_params = in_c * k_h * k_w
            w = torch.full((out_c,), float(per_channel_params))

        else:
            continue

        channel_weights[name] = w

    return channel_weights




def compute_avg_bits(
    step_sizes_dict,
    ranges_dict,
    channel_weights=None,
    eps: float = 1e-8,
) -> float:
    """
    Parameter-weighted average effective bit-width over all *weight* channels.

    bits_{l,k} = log2(R_{l,k} / S_{l,k})

    If channel_weights is provided:
        weight per channel = channel_weights[name][k]
    Else:
        weight per channel = 1 (reduces to simple mean over channels).
    """
    total_bits_weighted = 0.0
    total_weight = 0.0

    for name, wa in step_sizes_dict.items():
        if name not in ranges_dict:
            continue

        w_step, a_step = wa
        rec = ranges_dict[name]

        # weights only
        if "weight_ranges" not in rec or w_step is None:
            continue

        w_range = rec["weight_ranges"].to(w_step.device)  # [out_features]
        # protect against zeros
        bits = torch.log2(
            (w_range + eps) / (w_step + eps)
        )  # [out_features]

        if channel_weights is not None and name in channel_weights:
            w = channel_weights[name].to(bits.device)  # [out_features]
            # safety: broadcast scalar if someone passed a scalar
            if w.numel() == 1:
                w = w.expand_as(bits)
        else:
            w = torch.ones_like(bits)

        total_bits_weighted += float((bits * w).sum().item())
        total_weight        += float(w.sum().item())

    if total_weight == 0.0:
        return float("nan")

    return total_bits_weighted / total_weight












# ========================
# Adapted: Optimization over step size/ scaler
# ========================

def optimize_step_sizes(
    model,
    dataloader,             # calib_iter or a DataLoader; used only to FREEZE batches
    ranges_dict=None,
    num_epochs=50,
    num_mc_samples=10,
    lr=1e-2,
    eta=1e-4,
    gamma=0.0,              # MDL (weights only)
    device="cuda",
    percentile_prob=1e-4,
    layer_names=None,
    init_bits=8,
    bmax_bits=16,
    log_every=10,
    updates_per_batch=1,
    return_ranges=False,    
    eval_every=None,        # ← NEW
    eval_callback=None,     # ← NEW
):
    """
    Updated optimize_step_sizes with:
      - return_ranges=True → returns (step_sizes_dict, ranges_dict)
      - MDL *only* affects weights (activation prior removed)
      - AvgBits is a parameter-weighted bitwidth
    """

    # ---- 0) Setup ----
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # ---- 1) Freeze calibration batches ----
    if callable(dataloader):
        frozen_batches = list(dataloader())
        def frozen_iter():
            for b in frozen_batches:
                yield b
        cal_iter = frozen_iter
    else:
        frozen_batches = list(iter(dataloader))
        def frozen_iter():
            for b in frozen_batches:
                yield b
        cal_iter = frozen_iter

    # ---- 2) Candidate layers ----
    name2mod = dict(model.named_modules())
    if layer_names is None:
        cand_layers = [n for n, m in name2mod.items() if isinstance(m, QMODULES)]
    else:
        cand_layers = [n for n in layer_names if isinstance(name2mod.get(n, None), QMODULES)]

    print(f"Candidate Linear & Conv2d layers: {len(cand_layers)}")

    # ---- 3) Compute ranges ----
    if ranges_dict is None:
        print(f"Computing ranges with percentile_prob={percentile_prob} ...")
        ranges_dict = compute_data_ranges_poseidon(
            model=model,
            dataloader=cal_iter,
            device=device,
            layer_names=cand_layers,
            percentile_prob=percentile_prob,
        )
    else:
        print("Using provided ranges_dict...")

    # ---- 4) Cache clean IO ----
    print("Caching clean inputs/outputs...")
    clean_inputs, clean_outputs = get_clean_outputs_poseidon(
        model=model,
        dataloader=cal_iter,
        device=device,
        layer_names=cand_layers,
    )

    # ---- 5) Find compatible layers ----
    target_layers = []
    for name in cand_layers:
        if name not in ranges_dict:
            continue
        if clean_inputs[name][0] is None:
            continue
        if clean_outputs[name][0] is None:
            continue

        mod = name2mod.get(name, None)
        if mod is None:
            continue

        x0 = clean_inputs[name][0]
        y0 = clean_outputs[name][0]

        a_ranges = ranges_dict[name].get("activation_ranges", None)
        w_ranges = ranges_dict[name].get("weight_ranges", None)

        if a_ranges is None or w_ranges is None:
            continue

        # For Linear: channel = last dim
        if isinstance(mod, nn.Linear):
            in_ch  = x0.shape[-1]
            out_ch = y0.shape[-1]

        # For Conv2d: channel = dim 1  (N, C, H, W)
        elif isinstance(mod, nn.Conv2d):
            if x0.dim() < 2 or y0.dim() < 2:
                continue
            in_ch  = x0.shape[1]
            out_ch = y0.shape[1]

        else:
            continue

        if a_ranges.numel() == in_ch and w_ranges.numel() == out_ch:
            target_layers.append(name)

    print(f"Optimizing {len(target_layers)} compatible layers.")


    if len(target_layers) == 0:
        raise ValueError("No compatible layers found.")

    # ---- 5.5) Precompute parameter counts per channel (n_k) ----
    channel_weights = build_channel_param_weights(model, target_layers)

    # ---- 6) Initialize step sizes ----
    step_sizes_dict = {}
    params = []

    for name in target_layers:
        mod = name2mod[name]

        w_range = ranges_dict[name]["weight_ranges"].to(device)
        a_range = ranges_dict[name]["activation_ranges"].to(device)

        # Init steps to R / 2^bits
        w_step_init = w_range / (2 ** init_bits)
        a_step_init = a_range / (2 ** init_bits)

        # Min allowed (bmax_bits)
        w_step_min = w_range / (2 ** bmax_bits)
        a_step_min = a_range / (2 ** bmax_bits)

        EPS = 1e-8
        w_step_min = torch.maximum(w_step_min, torch.full_like(w_step_min, EPS))
        a_step_min = torch.maximum(a_step_min, torch.full_like(a_step_min, EPS))

        # Clamp inside valid range
        w_step_init = torch.clamp(w_step_init, min=w_step_min, max=w_range)
        a_step_init = torch.clamp(a_step_init, min=a_step_min, max=a_range)

        w_step = nn.Parameter(w_step_init.clone().detach())
        a_step = nn.Parameter(a_step_init.clone().detach())

        step_sizes_dict[name] = (w_step, a_step)
        params.extend([w_step, a_step])

    # ---- Initial log of bitwidth (parameter-weighted) ----
    with torch.no_grad():
        avg_bits = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)
        print(f"[Init] AvgBits≈{avg_bits:.2f} (target={init_bits})")

    # ---- Optimizer ----
    optimizer = optim.Adam(params, lr=lr)

    # ---- Number of calibration batches ----
    num_batches = min(len(clean_inputs[name]) for name in target_layers)
    print(f"Number of cached calibration batches: {num_batches}")

    print(f"\nStarting optimization: epochs={num_epochs}, mc_samples={num_mc_samples}, "
          f"eta={eta}, gamma={gamma}, lr={lr}, updates_per_batch={updates_per_batch}")

    # ---- Main training loop ----
    for epoch in range(1, num_epochs + 1):
        for batch_idx in range(num_batches):
            for _ in range(updates_per_batch):

                optimizer.zero_grad()

                # Compute likelihood + weight-prior (activation prior removed)
                total_loss, like_loss, prior_loss = compute_mc_loss_with_prior(
                    model=model,
                    step_sizes_dict=step_sizes_dict,
                    clean_inputs=clean_inputs,
                    clean_outputs=clean_outputs,
                    ranges_dict=ranges_dict,
                    batch_idx=batch_idx,
                    num_mc_samples=num_mc_samples,
                    eta=eta,
                    gamma=gamma,     # now only affects weights
                    device=device,
                )

                total_loss.backward()
                optimizer.step()

                # ---- Clamp to valid range ----
                with torch.no_grad():
                    for lname, (w_step, a_step) in step_sizes_dict.items():
                        w_range = ranges_dict[lname]["weight_ranges"].to(device)
                        a_range = ranges_dict[lname]["activation_ranges"].to(device)

                        w_min = w_range / (2 ** bmax_bits)
                        a_min = a_range / (2 ** bmax_bits)

                        EPS = 1e-8
                        w_min = torch.maximum(w_min, torch.full_like(w_min, EPS))
                        a_min = torch.maximum(a_min, torch.full_like(a_min, EPS))

                        w_step.clamp_(min=w_min, max=w_range)
                        a_step.clamp_(min=a_min, max=a_range)

        # ---- Logging ----
        if epoch % log_every == 0 or epoch == 1 or epoch == num_epochs:
            with torch.no_grad():
                avg_bits = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)

            print(f"[Epoch {epoch:4d}] "
                  f"Total={total_loss.item():.6f} | "
                  f"Like={like_loss.item():.6f} | "
                  f"Prior={prior_loss.item():.6f} | "
                  f"AvgBits={avg_bits:.2f}")

        # ---- Evaluation callback (NEW) ----
        if eval_every is not None and eval_callback is not None:
            if epoch % eval_every == 0:
                eval_callback(epoch, step_sizes_dict, ranges_dict)

    # ------------------------------
    # Return updated step_sizes_dict
    # ------------------------------
    if return_ranges:
        return step_sizes_dict, ranges_dict
    else:
        return step_sizes_dict



def compute_dynamic_stepsizes(
    model: nn.Module,
    layer_names,
    num_bits: int = 8,
    device: str = "cuda",
):
    """
    Compute per-channel *dynamic* weight step sizes for given Linear layers.

    For each Linear layer and each output channel k:
        w_k      = weight[k, :]             # [in_features]
        max_abs  = max(|w_k|)
        step_k   = 2 * max_abs / (2^bits - 1)

    Args:
        model:       Poseidon / ScOT model (already loaded)
        layer_names: iterable of layer names to process
        num_bits:    target quantization bit-width (e.g. 4, 8, 16)
        device:      device string ("cuda" or "cpu")

    Returns:
        dynamic_steps: dict { layer_name: 1D tensor [out_features] }
                       (stored on CPU for convenience)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    name2mod = dict(model.named_modules())
    dynamic_steps = {}

    denom = (2 ** num_bits) - 1
    if denom <= 0:
        raise ValueError(f"Invalid num_bits={num_bits}, must be >= 1.")

    print(f"[compute_dynamic_stepsizes] num_bits={num_bits}, denom={denom}")

    with torch.no_grad():
        for name in layer_names:
            mod = name2mod.get(name, None)
            if not isinstance(mod, nn.Linear):
                # Only handle Linear layers here
                continue

            # Weight: [out_features, in_features]
            w = mod.weight.data.to(device)
            out_features = w.size(0)

            # Flatten per-output-channel
            w_flat = w.view(out_features, -1)              # [out_features, in_features]
            max_abs = w_flat.abs().max(dim=1).values       # [out_features]

            # Symmetric dynamic range: [-max_abs, +max_abs]
            # Step size per channel: 2*max_abs / (2^bits - 1)
            step = (2.0 * max_abs) / float(denom)          # [out_features]

            dynamic_steps[name] = step.cpu()

            print(f"  - {name}: out_features={out_features}, step.shape={tuple(step.shape)}")

    print(f"[compute_dynamic_stepsizes] Collected dynamic steps for {len(dynamic_steps)} layers.")
    return dynamic_steps



def evaluate_with_stepsizes(
    model: nn.Module,
    val_loader,
    weight_steps,
    act_steps,              # kept for API compatibility; not used (we only quantize weights)
    layer_names,
    device: str = "cuda",
):
    """
    Evaluate the model with *weight-only* fake quantization using given step sizes.

    - model:       Poseidon / ScOT model
    - val_loader:  DataLoader OR callable that returns an iterator over validation batches
                   (same pattern as calib_iter / val_iter in your code)
    - weight_steps:
        PPQ case:   { layer_name: (w_step_param, a_step_tensor) }
        Dynamic case: { layer_name: 1D tensor OR list of step sizes [out_features] }
    - act_steps:   ignored (placeholder for future activation quantization)
    - layer_names: list of layer names to quantize
    - device:      "cuda" or "cpu"

    Returns:
        metrics: {
            "l1":     average absolute L1 error (using lp_error, p=1),
            "rel_l1": average relative L1 error (using relative_lp_error, p=1),
        }
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Map layer name -> module
    name2mod = dict(model.named_modules())

    # Prepare per-layer weight step tensors on device
    w_steps_per_layer = {}
    for name in layer_names:
        if name not in name2mod:
            continue
        if not isinstance(name2mod[name], QMODULES):
            continue
        if name not in weight_steps:
            continue

        w_info = weight_steps[name]
        # PPQ case: (w_step_param, a_step)
        if isinstance(w_info, (tuple, list)) and len(w_info) >= 1:
            w_step = w_info[0]
        else:
            # Dynamic case: directly a tensor or list
            w_step = w_info

        if isinstance(w_step, torch.nn.Parameter):
            w_step = w_step.detach()

        if not isinstance(w_step, torch.Tensor):
            w_step = torch.tensor(w_step)

        w_steps_per_layer[name] = w_step.to(device)

    # Build quantization hooks (weights only)
    def make_weight_quant_hook(layer_name, w_step_tensor):
        def hook(mod, inp, out):
            x = inp[0]  # input activation (we do NOT quantize it here)
            w = mod.weight

            # Per-output-channel step: [out_channels]
            step = w_step_tensor
            if isinstance(step, torch.nn.Parameter):
                step = step.detach()
            step = step.to(w.device)

            # Flatten per-output-channel
            w_flat = w.view(w.size(0), -1)              # [out_channels, *]
            step_flat = step.view(-1, 1)                # [out_channels, 1]

            w_quant_flat = torch.round(w_flat / step_flat) * step_flat
            w_quant = w_quant_flat.view_as(w)

            if isinstance(mod, nn.Linear):
                y = torch.nn.functional.linear(x, w_quant, mod.bias)
            elif isinstance(mod, nn.Conv2d):
                y = torch.nn.functional.conv2d(
                    x,
                    w_quant,
                    bias=mod.bias,
                    stride=mod.stride,
                    padding=mod.padding,
                    dilation=mod.dilation,
                    groups=mod.groups,
                )
            else:
                # Should not happen, but just in case
                y = out

            return y
        return hook


    # Register hooks
    handles = []
    for name, mod in name2mod.items():
        if name in w_steps_per_layer and isinstance(mod, nn.Linear):
            h = mod.register_forward_hook(
                make_weight_quant_hook(name, w_steps_per_layer[name])
            )
            handles.append(h)

    # Prepare val iterator (callable or DataLoader)
    if callable(val_loader):
        loader = val_loader()
    else:
        loader = val_loader

    # Accumulate metrics over val set
    rel_l1_list = []
    abs_l1_list = []

    with torch.no_grad():
        for batch in loader:
            x  = batch["pixel_values"].to(device)
            t  = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y  = batch.get("labels", None)

            if t is not None:
                t = t.to(device)
            if pm is not None:
                pm = pm.to(device)
            if y is not None:
                y = y.to(device)
            else:
                # for evaluation we expect labels
                continue

            outputs = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            pred = outputs.output

            # Poseidon's own metrics
            # p=1 → L1 and relative L1
            # convert to numpy (scOT metric functions expect numpy)
            pred_np = pred.detach().cpu().numpy()
            y_np    = y.detach().cpu().numpy()

            # relative_lp_error / lp_error return per-sample arrays → take batch mean
            batch_rel = relative_lp_error(pred_np, y_np, p=1, return_percent=True)
            batch_abs = lp_error(pred_np, y_np, p=1)

            rel_l1 = float(np.mean(batch_rel))
            abs_l1 = float(np.mean(batch_abs))

            rel_l1_list.append(rel_l1)
            abs_l1_list.append(abs_l1)

    # Remove hooks
    for h in handles:
        h.remove()

    # Average over all validation batches
    if len(rel_l1_list) == 0:
        metrics = {"l1": float("nan"), "rel_l1": float("nan")}
    else:
        metrics = {
            "l1":     float(sum(abs_l1_list) / len(abs_l1_list)),
            "rel_l1": float(sum(rel_l1_list) / len(rel_l1_list)),
        }

    print(f"[EVAL] L1={metrics['l1']:.6e} | RelL1={metrics['rel_l1']:.6e}")
    return metrics




def main():

    # --- Paths / config ---
    model_path   = "models/NS-PwC-T"
    data_path    = "dataset/NS-PwC"
    dataset_name = "fluids.incompressible.PiecewiseConstants"

    # PPQ hyperparams
    num_epochs      = 800
    num_mc_samples  = 5
    lr              = 1e-4
    eta             = 1e-6
    gamma           = 1e-5               # small MDL prior on weights
    percentile_prob = 1e-4
    init_bits       = 4                  # match dynamic-4 init
    bmax_bits       = 20
    device          = "cuda"

    eval_every      = 4                  # evaluate PPQ every 4 epochs

    # --- 1) Load Poseidon model ---
    model, device = load_poseidon_model(model_path, device=device)

    # --- 2) Build calibration + validation data ---
    calib_loader, val_loader, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        calib_batchsize=2,
        calib_steps=4,         # can increase to 64 later
        val_batchsize=4,
        val_steps=4            # can increase to 40 later
    )

    # --- 3) Load quantizable layers (Linear + Conv2d) ---
    b_quant_path = os.path.join(INSPECT_DIR, "T_quantize_layers.pt")
    print(f"[INFO] Loading quantize layer list from: {b_quant_path}")
    layer_data = torch.load(b_quant_path)
    name2mod = dict(model.named_modules())

    # Quantize both Linear and Conv2d layers
    cand_layers = [
        n for n in layer_data["quantize_layers"]
        if isinstance(name2mod.get(n, None), QMODULES)
    ]
    print(f"[INFO] {len(cand_layers)} candidate layers (Linear + Conv2d)")

    # ================================================================================
    # (A) Load precomputed 4-bit dynamic step sizes (weights-only)
    # ================================================================================
    dyn_stats_dir = os.path.join(PROJECT_ROOT, "dynamic_stats")
    dyn4_file = os.path.join(dyn_stats_dir, "NS-PwC-T-dynamic-stepsizes-4.json")

    with open(dyn4_file, "r") as f:
        dyn4_raw = json.load(f)["step_sizes"]    # {layer_name: [S_out_channels]}

    # convert lists → tensors on same device as model
    dyn4_steps = {
        name: torch.tensor(s, dtype=torch.float32, device=device)
        for name, s in dyn4_raw.items()
    }
    print(f"[INFO] Loaded 4-bit dynamic steps from: {dyn4_file}")

    # ================================================================================
    # (B) Compute dynamic-8 and dynamic-16 step sizes on the fly
    # ================================================================================
    print("[INFO] Computing 8-bit and 16-bit dynamic step sizes...")
    dyn8_steps  = compute_dynamic_stepsizes(model, cand_layers, num_bits=8,  device=device)
    dyn16_steps = compute_dynamic_stepsizes(model, cand_layers, num_bits=16, device=device)

    # ================================================================================
    # (C) Prepare results storage (for per-epoch eval)
    # ================================================================================
    model_name     = os.path.basename(model_path.rstrip("/"))
    artifacts_root = os.path.join(PROJECT_ROOT, "ppq_artifacts")
    artifacts_dir  = os.path.join(artifacts_root, model_name)
    os.makedirs(artifacts_dir, exist_ok=True)

    results_path = os.path.join(artifacts_dir, "PwC-T-results-1-5-conv.json")
    results = {}

    # --- precompute parameter counts per channel for all cand_layers ---
    channel_weights = build_channel_param_weights(model, cand_layers)

    # ================================================================================
    # (D) Define evaluation callback (called from inside optimize_step_sizes)
    #      - Only evaluates PPQ every eval_every epochs
    #      - Dynamic baselines are evaluated once at the end (see section E.2)
    # ================================================================================
    def eval_callback(
        epoch: int,
        step_sizes_dict: dict[str, tuple[torch.nn.Parameter, torch.Tensor]],
        ranges_dict: dict[str, dict[str, torch.Tensor]],
    ):

        print(f"\n================ EVALUATION @ epoch {epoch} ================")

        # 1) Average bits for current PPQ state (parameter-weighted)
        avg_bits = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)
        print(f"[INFO] AvgBits (PPQ) @ epoch {epoch}: {avg_bits:.3f}")

        # 2) Evaluate PPQ (current learned steps) — weights only
        ppq_metrics = evaluate_with_stepsizes(
            model=model,
            val_loader=val_iter,      # callable, evaluate_with_stepsizes handles this
            weight_steps=step_sizes_dict,
            act_steps=None,           # weights-only quantization
            layer_names=cand_layers,
            device=device,
        )

        # 3) Store only PPQ-related metrics for this epoch
        results[epoch] = {
            "avg_bits": float(avg_bits),
            "ppq":      ppq_metrics,
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[INFO] Saved evaluation results @ epoch {epoch} → {results_path}")

    # ================================================================================
    # (E) Run PPQ optimization WITH eval callback
    # ================================================================================
    step_sizes_dict, ranges_dict = optimize_step_sizes(
        model=model,
        dataloader=calib_iter,
        ranges_dict=None,
        num_epochs=num_epochs,
        num_mc_samples=num_mc_samples,
        lr=lr,
        eta=eta,
        gamma=gamma,
        device=device,
        percentile_prob=percentile_prob,
        layer_names=cand_layers,
        init_bits=init_bits,
        bmax_bits=bmax_bits,
        log_every=1,
        updates_per_batch=1,
        return_ranges=True,
        eval_every=eval_every,    # PPQ evaluated every `eval_every` epochs
        eval_callback=eval_callback,
    )

    # ================================================================================
    # (E.2) FINAL evaluation: PPQ + dynamic baselines (only once at the end)
    # ================================================================================
    print("\n================ FINAL EVALUATION (PPQ + dynamic) ================\n")

    # Final avg bits with learned step sizes
    final_avg_bits = compute_avg_bits(step_sizes_dict, ranges_dict, channel_weights)
    print(f"[FINAL] AvgBits (PPQ) ≈ {final_avg_bits:.3f}")

    # Final PPQ evaluation
    final_ppq_metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=step_sizes_dict,
        act_steps=None,
        layer_names=cand_layers,
        device=device,
    )

    # Dynamic 4 / 8 / 16 baselines (only now)
    final_dyn4_metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=dyn4_steps,
        act_steps=None,
        layer_names=cand_layers,
        device=device,
    )

    final_dyn8_metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=dyn8_steps,
        act_steps=None,
        layer_names=cand_layers,
        device=device,
    )

    final_dyn16_metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=dyn16_steps,
        act_steps=None,
        layer_names=cand_layers,
        device=device,
    )

    # Merge final metrics into the results dict under the last epoch key
    final_epoch_key = num_epochs
    if final_epoch_key not in results:
        results[final_epoch_key] = {}

    results[final_epoch_key].update({
        "final_avg_bits": float(final_avg_bits),
        "final_ppq":      final_ppq_metrics,
        "dyn4":           final_dyn4_metrics,
        "dyn8":           final_dyn8_metrics,
        "dyn16":          final_dyn16_metrics,
    })

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Saved FINAL evaluation results → {results_path}")

    # ================================================================================
    # (F) Save final learned PPQ step sizes
    # ================================================================================
    save_obj = {
        "step_sizes": {
            k: (v[0].detach().cpu(), v[1].detach().cpu())
            for k, v in step_sizes_dict.items()
        },
        "meta": {
            "model_path":       model_path,
            "dataset_name":     dataset_name,
            "eta":              eta,
            "gamma":            gamma,
            "percentile_prob":  percentile_prob,
            "num_epochs":       num_epochs,
            "num_mc_samples":   num_mc_samples,
            "init_bits":        init_bits,
            "bmax_bits":        bmax_bits,
        },
    }

    step_pt_path   = os.path.join(artifacts_dir, "ppq_step_sizes-1-5-conv.pt")
    step_json_path = os.path.join(artifacts_dir, "ppq_step_sizes-1-5-conv.json")

    torch.save(save_obj, step_pt_path)
    with open(step_json_path, "w") as f:
        json.dump(
            {
                "step_sizes": {
                    k: (v[0].tolist(), v[1].tolist())
                    for k, v in save_obj["step_sizes"].items()
                },
                "meta": save_obj["meta"],
            },
            f,
            indent=2,
        )

    print(f"\n✓ Saved learned PPQ step sizes → {step_pt_path}")
    print(f"✓ Also saved JSON version     → {step_json_path}\n")




# ==================================================================================================================================
# ==================================================================================================================================
# ==================================================================================================================================
# Test Functions
# ==================================================================================================================================
# ==================================================================================================================================
# ==================================================================================================================================

def debug_test_compute_data_ranges(model, calib_loader, device, layer_names):
    """
    Quick sanity check that compute_data_ranges_poseidon works
    for both Linear and Conv2d layers.

    Prints each layer name, module type, and the shapes of
    weight_ranges and activation_ranges.
    """
    model = model.to(device).eval()

    ranges = compute_data_ranges_poseidon(
        model=model,
        dataloader=calib_loader,   # can be DataLoader or calib_iter()
        device=device,
        layer_names=layer_names,
        percentile_prob=1e-4,
    )

    name2mod = dict(model.named_modules())

    print("\n====== DEBUG: RANGES PER LAYER ======")
    for name in sorted(ranges.keys()):
        mod = name2mod.get(name, None)
        w_range = ranges[name].get("weight_ranges", None)
        a_range = ranges[name].get("activation_ranges", None)

        print(f"\nLayer: {name}")
        print(f"  Type: {type(mod).__name__ if mod is not None else 'UNKNOWN'}")
        if w_range is not None:
            print(f"  weight_ranges.shape     = {tuple(w_range.shape)}")
        else:
            print("  weight_ranges: None")

        if a_range is not None:
            print(f"  activation_ranges.shape = {tuple(a_range.shape)}")
        else:
            print("  activation_ranges: None")



def debug_test_get_clean_outputs(model, calib_loader, device, layer_names):
    """
    Debug helper to verify that get_clean_outputs_poseidon correctly
    captures inputs/outputs for both Linear and Conv2d layers.
    """
    name2mod = dict(model.named_modules())

    print("\n" + "=" * 80)
    print("DEBUG: Testing get_clean_outputs_poseidon (Linear + Conv2d)")
    print("=" * 80)

    clean_inputs, clean_outputs = get_clean_outputs_poseidon(
        model=model,
        dataloader=calib_loader,
        device=device,
        layer_names=layer_names,
    )

    for name in layer_names:
        mod = name2mod.get(name, None)
        if mod is None:
            print(f"\nLayer: {name}")
            print("  [WARN] Module not found in model.named_modules()")
            continue

        Xin_list = clean_inputs.get(name, [])
        Yout_list = clean_outputs.get(name, [])

        Xin0 = Xin_list[0] if Xin_list and Xin_list[0] is not None else None
        Yout0 = Yout_list[0] if Yout_list and Yout_list[0] is not None else None

        print(f"\nLayer: {name}")
        print(f"  Type: {mod.__class__.__name__}")

        if Xin0 is None or Yout0 is None:
            print("  [WARN] No activations captured for first batch.")
            continue

        print(f"  X_pre shape: {tuple(Xin0.shape)}")
        print(f"  Y_post shape: {tuple(Yout0.shape)}")

        if isinstance(mod, nn.Linear):
            print(f"  in_features:  {mod.in_features}")
            print(f"  out_features: {mod.out_features}")
        elif isinstance(mod, nn.Conv2d):
            print(f"  in_channels:  {mod.in_channels}")
            print(f"  out_channels: {mod.out_channels}")
            print(f"  kernel_size:  {mod.kernel_size}")
            print(f"  stride:       {mod.stride}")
            print(f"  padding:      {mod.padding}")

    print("\n" + "=" * 80)
    print("DEBUG: get_clean_outputs_poseidon test complete")
    print("=" * 80 + "\n")



def debug_test_mc_loss_conv2d(model,
                              calib_loader,
                              device,
                              layer_names,
                              percentile_prob=1e-4):
    """
    Quick sanity check that Conv2d layers work inside compute_mc_loss_single_batch.
    We:
      1) Pick the first Conv2d layer from `layer_names`
      2) Compute ranges + clean IO ONLY for that layer
      3) Build a dummy step_sizes_dict from its ranges
      4) Run compute_mc_loss_single_batch on batch_idx=0
    """
    model.eval()
    name2mod = dict(model.named_modules())

    # 1) pick first Conv2d layer from the provided list
    conv_layers = [n for n in layer_names if isinstance(name2mod.get(n, None), nn.Conv2d)]
    if not conv_layers:
        print("[DEBUG Conv2d] No Conv2d layers found in layer_names.")
        return

    target = conv_layers[0]
    print(f"[DEBUG Conv2d] Testing MC loss on layer: {target}")

    # 2) Compute ranges & clean IO for this single layer
    def one_layer_loader():
        # make sure we only take a few batches to keep it light
        if callable(calib_loader):
            loader = calib_loader()
        else:
            loader = calib_loader
        for i, b in enumerate(loader):
            if i >= 2:
                break
            yield b

    ranges_dict = compute_data_ranges_poseidon(
        model=model,
        dataloader=one_layer_loader,
        device=device,
        layer_names=[target],
        percentile_prob=percentile_prob,
    )

    clean_inputs, clean_outputs = get_clean_outputs_poseidon(
        model=model,
        dataloader=one_layer_loader,
        device=device,
        layer_names=[target],
    )

    # 3) Build a simple step_sizes_dict: w_step from ranges / 2^4
    rec = ranges_dict[target]
    w_range = rec["weight_ranges"].to(device)
    a_range = rec["activation_ranges"].to(device)

    w_step_init = w_range / (2 ** 4)
    a_step_init = a_range / (2 ** 4)

    w_step = nn.Parameter(w_step_init.clone())
    a_step = nn.Parameter(a_step_init.clone())

    step_sizes_dict = {target: (w_step, a_step)}

    # 4) Run MC loss on the first cached batch (batch_idx=0)
    loss = compute_mc_loss_single_batch(
        model=model,
        step_sizes_dict=step_sizes_dict,
        clean_inputs=clean_inputs,
        clean_outputs=clean_outputs,
        batch_idx=0,
        num_mc_samples=3,
        eta=1e-4,
        device=device,
    )

    print(f"[DEBUG Conv2d] MC loss for {target} on batch 0: {loss.item():.6e}")


if __name__ == "__main__":
    
    # debug_test_compute_data_ranges(
    #     model=model,
    #     calib_loader=calib_iter,   # or calib_loader
    #     device=device,
    #     layer_names=cand_layers,
    # )

    # debug_test_get_clean_outputs(
    #     model=model,
    #     calib_loader=calib_iter,   # or calib_loader
    #     device=device,
    #     layer_names=cand_layers,
    # )


    # debug_test_mc_loss_conv2d(
    #     model=model,
    #     calib_loader=calib_iter,   # or calib_loader
    #     device=device,
    #     layer_names=cand_layers,
    #     percentile_prob=percentile_prob,
    # )


    main()


    #empty the gpu
    #import torch
    #torch.cuda.empty_cache()
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()



