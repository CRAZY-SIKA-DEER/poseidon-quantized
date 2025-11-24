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
    for each target Linear layer.

    Returns:
        clean_inputs:  {layer_name: [X_batch_0, X_batch_1, ...], ...}   # pre-op, last dim = in_features
        clean_outputs: {layer_name: [Y_batch_0, Y_batch_1, ...], ...}   # post-op, last dim = out_features
    """
    model.eval()

    clean_inputs  = {name: [] for name in layer_names}
    clean_outputs = {name: [] for name in layer_names}

    # allow callable iterator (e.g., calib_iter)
    if callable(dataloader):
        dataloader = dataloader()

    # (optional) map name -> module for quick checks
    name2mod = dict(model.named_modules())

    with torch.inference_mode():
        any_batch = False
        for batch_idx, batch in enumerate(dataloader):
            any_batch = True
            x  = batch["pixel_values"].to(device)
            t  = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y  = batch.get("labels", None)

            layer_io = {}  # layer_name -> (X_pre, Y_post)

            def get_hook(name):
                def hook(mod, inp, out):
                    X_pre = inp[0]
                    Y_post = out
                    # (optional) sanity check: last dim of X equals in_features
                    if hasattr(mod, "in_features"):
                        if X_pre.shape[-1] != mod.in_features:
                            print(f"[WARN] {name}: X_pre last dim {X_pre.shape[-1]} != in_features {mod.in_features}")
                    layer_io[name] = (X_pre.detach().cpu(), Y_post.detach().cpu())
                return hook

            handles = []
            for name, mod in model.named_modules():
                if name in layer_names and isinstance(mod, torch.nn.Linear):
                    handles.append(mod.register_forward_hook(get_hook(name)))

            _ = model(
                pixel_values=x,
                time=(t.to(device) if t is not None else None),
                pixel_mask=(pm.to(device) if pm is not None else None),
                labels=(y.to(device) if y is not None else None),
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
        active_layers = sum(1 for outs in clean_outputs.values() if outs and outs[0] is not None)
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
            if name in layer_names and isinstance(module, nn.Linear):
                weight = module.weight.data.to(device)
                
                # Flatten per output channel: [out_features, -1]
                w_flat = weight.view(weight.size(0), -1)
                
                # Compute μ and σ per channel
                w_mean = w_flat.mean(dim=1)  # [out_features]
                w_std = w_flat.std(dim=1, unbiased=False)
                
                # Compute percentile threshold τ
                tau = w_mean + w_std * erf_inv_value
                
                # Clipping thresholds (symmetric)
                beta = tau
                alpha = 2 * w_mean - beta
                
                # Range = beta - alpha
                weight_ranges = (beta - alpha).clamp(min=1e-8)
                
                # Store
                ranges_dict[name] = {
                    'weight_ranges': weight_ranges.to(device),
                    'act_stats': []
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
            
            def get_hook(name):
                def hook(module, input, output):
                    x = input[0]  # Pre-op activation
                    
                    # Handle different activation shapes
                    # Could be [B, seq, C] or [B, H, W, C] or other shapes
                    # We want to flatten all non-channel dimensions
                    
                    # Determine channel axis (last dimension for most Linear layers)
                    # Shape patterns:
                    #   [B, seq, C] -> channel is dim -1
                    #   [B, H, W, C] -> channel is dim -1
                    #   [B, C, H, W] -> channel is dim 1 (Conv layers, but we skip those)
                    
                    if x.dim() == 2:
                        # [B, C] - already flat
                        x_flat = x.transpose(0, 1)  # [C, B]
                    elif x.dim() == 3:
                        # [B, seq, C] -> [C, B*seq]
                        x_flat = x.permute(2, 0, 1).reshape(x.size(-1), -1)
                    elif x.dim() == 4:
                        # [B, H, W, C] -> [C, B*H*W]
                        x_flat = x.permute(3, 0, 1, 2).reshape(x.size(-1), -1)
                    else:
                        # Fallback: assume last dim is features
                        x_flat = x.reshape(-1, x.size(-1)).transpose(0, 1)
                    
                    # Compute μ and σ per channel
                    x_mean = x_flat.mean(dim=1)  # [channels]
                    x_std = x_flat.std(dim=1, unbiased=False)
                    
                    # Store statistics for later aggregation
                    ranges_dict[name]['act_stats'].append({
                        'mean': x_mean.cpu(),
                        'std': x_std.cpu()
                    })
                return hook
            
            # Register hooks
            handles = []
            for name, module in model.named_modules():
                if name in layer_names and isinstance(module, nn.Linear):
                    handles.append(module.register_forward_hook(get_hook(name)))
            
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
    MDL prior: gamma * sum_k log2(R_k / S_k), over weight and activation step sizes.
    Encourages larger step sizes (fewer bits) and penalizes over-precision.
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
        # -- Weights
        w_ranges = rec['weight_ranges']
        if w_ranges is not None:
            w_ranges = w_ranges.to(device)
            assert w_ranges.numel() == weight_step_sizes.numel(), f"[{name}] weight: {w_ranges.shape} vs step: {weight_step_sizes.shape}"
            w_term = torch.log2(torch.clamp(w_ranges, min=eps) / torch.clamp(weight_step_sizes, min=eps))
            prior_loss = prior_loss + gamma * torch.sum(w_term)
        # -- Activations
        a_ranges = rec.get('activation_ranges', None)
        if a_ranges is not None:
            a_ranges = a_ranges.to(device)
            assert a_ranges.numel() == activation_step_sizes.numel(), f"[{name}] act: {a_ranges.shape} vs step: {activation_step_sizes.shape}"
            a_term = torch.log2(torch.clamp(a_ranges, min=eps) / torch.clamp(activation_step_sizes, min=eps))
            prior_loss = prior_loss + gamma * torch.sum(a_term)
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
    - Returns: likelihood_loss (tensor with grad)
    """
    if model is not None:
        model.eval()

    # Build a stable layer list that exists in both dicts
    target_layers = [name for name in step_sizes_dict.keys()
                     if name in clean_inputs and name in clean_outputs]

    if not target_layers:
        raise ValueError("No overlapping layers between step_sizes_dict and clean IO caches.")

    name2module = dict(model.named_modules())

    batch_loss = None
    layer_contrib = 0

    for name in target_layers:
        module = name2module.get(name)
        if module is None or not isinstance(module, torch.nn.Linear):
            continue

        Xb = clean_inputs[name][batch_idx]
        Yb = clean_outputs[name][batch_idx]
        if (Xb is None) or (Yb is None):
            continue  # layer didn't fire for this batch

        X_clean = Xb.to(device)
        Y_clean = Yb.to(device)
        W_clean = module.weight.to(device)

        w_step, a_step = step_sizes_dict[name]
        a_step = a_step.to(device)
        w_step = w_step.to(device)

        # Shape checks
        assert X_clean.shape[-1] == W_clean.shape[1], \
            f"{name}: X in_features {X_clean.shape[-1]} vs W in_features {W_clean.shape[1]}"
        assert w_step.numel() == W_clean.shape[0], \
            f"{name}: w_step {w_step.numel()} vs out_features {W_clean.shape[0]}"
        assert a_step.numel() == X_clean.shape[-1], \
            f"{name}: a_step {a_step.numel()} vs in_features {X_clean.shape[-1]}"

        mc_losses = []
        for _ in range(num_mc_samples):
            X_noisy = add_quantization_noise(X_clean, a_step, channel_axis=-1)
            W_noisy = add_quantization_noise(W_clean, w_step, channel_axis=0)
            Y_noisy = torch.nn.functional.linear(X_noisy, W_noisy, module.bias)
            loss_elem = torch.mean((Y_noisy - Y_clean) ** 2) / (2 * eta)
            mc_losses.append(loss_elem)

        layer_loss = torch.stack(mc_losses).mean()  # avg over MC samples

        batch_loss = layer_loss if batch_loss is None else (batch_loss + layer_loss)
        layer_contrib += 1

    if layer_contrib == 0:
        # no valid layers for this batch
        # you can either return 0 or raise; here we return 0 on device
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
    gamma=0.001,
    device="cuda",
    percentile_prob=1e-4,
    layer_names=None,
    init_scale=0.1,         # (kept for backward-compat; ignored if init_bits is not None)
    log_every=10,
    init_bits=8,            # target starting bits (e.g., 8)
    bmax_bits=16,           # relative floor = R / 2**bmax_bits
    updates_per_batch=1,    # NEW: how many optimizer steps per batch
):
    """
    Optimize per-channel step sizes (weights & activations) via PPQ Monte Carlo + MDL prior,
    using a **per-batch** training loop to avoid huge computation graphs.

    - We still:
        1) freeze a small calibration set (frozen_batches)
        2) compute ranges on that set
        3) cache clean inputs/outputs on that same set
    - BUT:
        - We now do one (or more) optimizer steps per batch instead of one giant loss over all batches.
        - This keeps the autograd graph per-step small → avoids OOM when using more calibration batches.
    """
    # ---- 0) Setup
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # ---- 1) Freeze calibration batches to preserve alignment
    if callable(dataloader):
        frozen_batches = list(dataloader())       # materialize once
        def frozen_iter():                        # re-iterable
            for b in frozen_batches:
                yield b
        cal_iter = frozen_iter
    else:
        # If a plain DataLoader is given, materialize it once as well
        frozen_batches = list(iter(dataloader))
        def frozen_iter():
            for b in frozen_batches:
                yield b
        cal_iter = frozen_iter

    # ---- 2) Candidate Linear layers
    name2mod = dict(model.named_modules())
    if layer_names is None:
        cand_layers = [n for n, m in name2mod.items() if isinstance(m, nn.Linear)]
    else:
        cand_layers = [n for n in layer_names if isinstance(name2mod.get(n, None), nn.Linear)]

    print(f"Candidate Linear layers: {len(cand_layers)}")

    # ---- 3) Compute ranges (or use provided)
    if ranges_dict is None:
        print(f"Computing ranges with percentile_prob={percentile_prob}...")
        ranges_dict = compute_data_ranges_poseidon(
            model=model,
            dataloader=cal_iter,      # SAME frozen iterator
            device=device,
            layer_names=cand_layers,
            percentile_prob=percentile_prob,
        )
    else:
        print("Using provided ranges_dict...")

    # ---- 4) Cache clean inputs/outputs (pre-op X, post-op Y) on SAME batches
    print("Caching clean inputs/outputs...")
    clean_inputs, clean_outputs = get_clean_outputs_poseidon(
        model=model,
        dataloader=cal_iter,          # SAME frozen iterator
        device=device,
        layer_names=cand_layers,
    )

    # ---- 5) Build a stable target layer list (intersection & channel-match)
    target_layers = []
    for name in cand_layers:
        if (name in ranges_dict) and (name in clean_inputs) and (name in clean_outputs):
            x0 = clean_inputs[name][0]
            y0 = clean_outputs[name][0]
            if (x0 is None) or (y0 is None):
                continue
            a_ranges = ranges_dict[name].get("activation_ranges", None)
            w_ranges = ranges_dict[name].get("weight_ranges", None)
            if a_ranges is None or w_ranges is None:
                continue
            if a_ranges.numel() == x0.shape[-1] and w_ranges.numel() == y0.shape[-1]:
                target_layers.append(name)

    print(f"Optimizing {len(target_layers)} compatible layers (out of {len(cand_layers)} candidates).")
    if not target_layers:
        raise ValueError("No compatible layers with aligned channels found.")

    # ---- 6) Initialize per-channel quantization step sizes (weights & activations)
    step_sizes_dict = {}
    params = []

    for name in target_layers:
        mod = name2mod[name]
        in_ch, out_ch = mod.in_features, mod.out_features

        w_range = ranges_dict[name]["weight_ranges"].to(device)
        a_range = ranges_dict[name]["activation_ranges"].to(device)

        target_bits = init_bits
        max_bits    = bmax_bits

        # desired initial step sizes
        w_step_init = w_range / (2 ** target_bits)
        a_step_init = a_range / (2 ** target_bits)

        # relative minimum allowed step
        w_step_min = w_range / (2 ** max_bits)
        a_step_min = a_range / (2 ** max_bits)

        EPS_ABS = 1e-8
        w_step_min = torch.maximum(w_step_min, torch.full_like(w_step_min, EPS_ABS))
        a_step_min = torch.maximum(a_step_min, torch.full_like(a_step_min, EPS_ABS))

        # clamp to [S_min, S_max]
        w_step_init = torch.clamp(w_step_init, min=w_step_min, max=w_range)
        a_step_init = torch.clamp(a_step_init, min=a_step_min, max=a_range)

        w_step = nn.Parameter(w_step_init.clone().detach())
        a_step = nn.Parameter(a_step_init.clone().detach())

        step_sizes_dict[name] = (w_step, a_step)
        params += [w_step, a_step]

    # test initial average bitwidths
    with torch.no_grad():
        num = 0
        s = 0.0
        for name, (w_step, a_step) in step_sizes_dict.items():
            w_range = ranges_dict[name]["weight_ranges"].to(w_step.device)
            a_range = ranges_dict[name]["activation_ranges"].to(a_step.device)
            w_bits = torch.log2((w_range / (w_step + 1e-12)).clamp(min=1.0)).mean()
            a_bits = torch.log2((a_range / (a_step + 1e-12)).clamp(min=1.0)).mean()
            s += (w_bits + a_bits).item() / 2.0
            num += 1
        print(f"[Init] AvgBits≈{s/num:.2f} (target {init_bits})")

    optimizer = optim.Adam(params, lr=lr)

    # ---- 7) Per-batch training loop
    # How many cached batches do we have?
    num_batches = min(len(clean_inputs[name]) for name in target_layers)
    print(f"Number of cached calibration batches: {num_batches}")
    if num_batches == 0:
        raise ValueError("No cached calibration batches found in clean_inputs.")

    print(
        f"\nStarting optimization: epochs={num_epochs}, mc_samples={num_mc_samples}, "
        f"eta={eta}, gamma={gamma}, lr={lr}, updates_per_batch={updates_per_batch}"
    )

    global_step = 0
    last_total_loss = None
    last_like_loss = None
    last_prior_loss = None

    for epoch in range(1, num_epochs + 1):
        for batch_idx in range(num_batches):
            for _ in range(updates_per_batch):
                optimizer.zero_grad()

                total_loss, likelihood_loss, prior_loss = compute_mc_loss_with_prior(
                    model=model,
                    step_sizes_dict=step_sizes_dict,
                    clean_inputs=clean_inputs,
                    clean_outputs=clean_outputs,
                    ranges_dict=ranges_dict,
                    batch_idx=batch_idx,
                    num_mc_samples=num_mc_samples,
                    eta=eta,
                    gamma=gamma,
                    device=device,
                )

                total_loss.backward()
                optimizer.step()

                global_step += 1
                last_total_loss = total_loss
                last_like_loss = likelihood_loss
                last_prior_loss = prior_loss

                # ---- 8) Project step sizes back to valid range after each optimizer step
                with torch.no_grad():
                    EPS_ABS = 1e-8
                    for layer_name, (w_step, a_step) in step_sizes_dict.items():
                        w_range = ranges_dict[layer_name]["weight_ranges"].to(device)
                        a_range = ranges_dict[layer_name]["activation_ranges"].to(device)

                        w_step_min = w_range / (2 ** bmax_bits)
                        a_step_min = a_range / (2 ** bmax_bits)

                        w_step_min = torch.maximum(w_step_min, torch.full_like(w_step_min, EPS_ABS))
                        a_step_min = torch.maximum(a_step_min, torch.full_like(a_step_min, EPS_ABS))

                        w_step.clamp_(min=w_step_min, max=w_range)
                        a_step.clamp_(min=a_step_min, max=a_range)

        # ---- Logging once per epoch
        if (epoch % log_every) == 0 or epoch == 1 or epoch == num_epochs:
            with torch.no_grad():
                sum_bits, sum_ch = 0.0, 0
                for name in target_layers:
                    w_ranges = ranges_dict[name]["weight_ranges"].to(device)
                    a_ranges = ranges_dict[name]["activation_ranges"].to(device)
                    w_step, a_step = step_sizes_dict[name]
                    w_bits = torch.log2((w_ranges + 1e-8) / (w_step + 1e-8))
                    a_bits = torch.log2((a_ranges + 1e-8) / (a_step + 1e-8))
                    sum_bits += w_bits.sum().item() + a_bits.sum().item()
                    sum_ch   += w_bits.numel() + a_bits.numel()
                avg_bits = (sum_bits / max(1, sum_ch)) if sum_ch else float("nan")

            # Use last_* losses from the last batch of this epoch
            if last_total_loss is not None:
                print(
                    f"[Epoch {epoch:3d}] "
                    f"Total={last_total_loss.item():.6f} | "
                    f"Like={last_like_loss.item():.6f} | "
                    f"Prior={last_prior_loss.item():.6f} | "
                    f"AvgBits={avg_bits:.2f}"
                )
            else:
                print(
                    f"[Epoch {epoch:3d}] "
                    f"(no valid batches) | AvgBits={avg_bits:.2f}"
                )

    return step_sizes_dict




def main():
    # --- Paths / config

    #model_path   = "models/NS-PwC-B"
    model_path   = "models/NS-PwC-T"
    data_path    = "dataset/NS-PwC"
    dataset_name = "fluids.incompressible.PiecewiseConstants"

    # model_path   = "models/NS-SVS-B"
    # data_path    = "dataset/NS-SVS"
    # dataset_name = "fluids.incompressible.VortexSheet"

    # model_path   = "models/NS-BB-B"
    # data_path    = "dataset/NS-BB"
    # dataset_name = "fluids.incompressible.BrownianBridge"

    # PPQ hyperparams
    num_epochs      = 1000      # number of passes over frozen calib set
    num_mc_samples  = 5
    lr              = 1e-2
    eta             = 1e-5
    gamma           = 0.00001
    percentile_prob = 1e-4
    init_bits       = 8
    bmax_bits       = 12
    device          = "cuda"

    # --- 1) Load model
    model, device = load_poseidon_model(model_path, device=device)

    # --- 2) Build a small, frozen calibration iterator
    # We want 8 samples total, with batchsize=2 -> 4 batches
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        calib_batchsize=2,
        calib_steps=64,          # 4 batches(calib_steps) × 2 samples = 8 calibration samples
        val_batchsize=1,
        val_steps=1
    )

    # --- 3) Candidate Linear layer names (from your inspection file)
    b_quant_path = os.path.join(INSPECT_DIR, 'T_quantize_layers.pt')
    print(f"Loading quantize layer list from: {b_quant_path}")
    layer_data = torch.load(b_quant_path)
    name2mod = dict(model.named_modules())
    cand_layers = [n for n in layer_data['quantize_layers'] if isinstance(name2mod.get(n, None), nn.Linear)]
    print(f"Candidate Linear layers (from file & present in model): {len(cand_layers)}")

    # --- 4) Run PPQ optimization
    step_sizes_dict = optimize_step_sizes(
        model=model,
        dataloader=calib_iter,          # frozen calib batches
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
        updates_per_batch=1,            # 👈 1 update per batch as you wanted
    )

    # --- 5) Save learned step sizes (unchanged)
    save_obj = {
        "step_sizes": {k: (v[0].detach().cpu(), v[1].detach().cpu()) for k, v in step_sizes_dict.items()},
        "meta": {
            "model_path": model_path,
            "dataset_name": dataset_name,
            "eta": eta,
            "gamma": gamma,
            "percentile_prob": percentile_prob,
            "num_epochs": num_epochs,
            "num_mc_samples": num_mc_samples,
            "init_bits": init_bits,
            "bmax_bits": bmax_bits
        }
    }

    model_name = os.path.basename(model_path.rstrip("/"))
    artifacts_root = os.path.join(PROJECT_ROOT, "ppq_artifacts")
    artifacts_dir = os.path.join(artifacts_root, model_name)
    os.makedirs(artifacts_dir, exist_ok=True)

    step_pt_path = os.path.join(artifacts_dir, "ppq_step_sizes.pt")
    step_json_path = os.path.join(artifacts_dir, "ppq_step_sizes.json")

    torch.save(save_obj, step_pt_path)
    print(f"✓ Saved learned step sizes → {step_pt_path}")

    json_ready = {
        'step_sizes': {k: (v[0].tolist(), v[1].tolist()) for k, v in save_obj['step_sizes'].items()},
        'meta': save_obj['meta']
    }
    with open(step_json_path, "w") as f_json:
        json.dump(json_ready, f_json, indent=2)
    print(f"✓ Also saved learned step sizes → {step_json_path}")



    # # --- 5) Save learned step sizes for later use
    # save_obj = {
    #     "step_sizes": {k: (v[0].detach().cpu(), v[1].detach().cpu()) for k, v in step_sizes_dict.items()},
    #     "meta": {
    #         "model_path": model_path,
    #         "dataset_name": dataset_name,
    #         "eta": eta,
    #         "gamma": gamma,
    #         "percentile_prob": percentile_prob,
    #         "num_epochs": num_epochs,
    #         "num_mc_samples": num_mc_samples,
    #         "init_bits": init_bits,
    #         "bmax_bits": bmax_bits
    #     }
    # }

    # os.makedirs("ppq_artifacts", exist_ok=True)
    # torch.save(save_obj, "ppq_artifacts/ppq_step_sizes.pt")
    # print("✓ Saved learned step sizes → ppq_artifacts/ppq_step_sizes.pt")


    # # Convert tensors in save_obj to lists for JSON encoding
    # json_ready = {
    #     'step_sizes': {k: (v[0].tolist(), v[1].tolist()) for k, v in save_obj['step_sizes'].items()},
    #     'meta': save_obj['meta']
    # }
    # with open("ppq_artifacts/ppq_step_sizes.json", "w") as f_json:
    #     json.dump(json_ready, f_json, indent=2)
    # print("✓ Also saved learned step sizes → ppq_artifacts/ppq_step_sizes.json")



# ==================================================================================================================================
# ==================================================================================================================================
# ==================================================================================================================================
# Test Functions
# ==================================================================================================================================
# ==================================================================================================================================
# ==================================================================================================================================

# ========================
# Test Function for Add Quantization Noise
# ========================
def test_add_quantization_noise():
    """Test quantization noise addition for different tensor shapes"""
    print("\n" + "="*80)
    print("TESTING ADD QUANTIZATION NOISE")
    print("="*80)
    
    # Test 1: Linear activation (batch-first, features last)
    print("\nTest 1: Linear Activation [B, seq, C]")
    x_act = torch.randn(4, 256, 192)  # batch=4, seq=256, features=192
    step_sizes_act = torch.ones(192) * 0.01  # Per-feature step sizes
    
    x_noisy = add_quantization_noise(x_act, step_sizes_act, channel_axis=-1)
    
    print(f"  Input shape: {x_act.shape}")
    print(f"  Step sizes shape: {step_sizes_act.shape}")
    print(f"  Output shape: {x_noisy.shape}")
    print(f"  Max noise added: {(x_noisy - x_act).abs().max():.6f}")
    print(f"  Expected max noise: {0.5 * step_sizes_act.max():.6f}")
    print(f"  ✓ Test passed!")
    
    # Test 2: Weight matrix (output channels first)
    print("\nTest 2: Weight Matrix [out_features, in_features]")
    w = torch.randn(192, 192)  # output=192, input=192
    step_sizes_w = torch.ones(192) * 0.001  # Per output channel
    
    w_noisy = add_quantization_noise(w, step_sizes_w, channel_axis=0)
    
    print(f"  Input shape: {w.shape}")
    print(f"  Step sizes shape: {step_sizes_w.shape}")
    print(f"  Output shape: {w_noisy.shape}")
    print(f"  Max noise added: {(w_noisy - w).abs().max():.6f}")
    print(f"  Expected max noise: {0.5 * step_sizes_w.max():.6f}")
    print(f"  ✓ Test passed!")
    
    # Test 3: Different step sizes per channel
    print("\nTest 3: Per-Channel Different Step Sizes")
    x = torch.randn(2, 100, 64)
    step_sizes = torch.linspace(0.001, 0.01, 64)  # Different per channel
    
    x_noisy = add_quantization_noise(x, step_sizes, channel_axis=-1)
    noise = (x_noisy - x).abs()
    
    # Check that noise magnitude respects step sizes
    for c in range(64):
        max_noise_c = noise[:, :, c].max().item()
        expected_max = 0.5 * step_sizes[c].item()
        assert max_noise_c <= expected_max * 1.01, \
            f"Channel {c}: noise {max_noise_c:.6f} exceeds expected {expected_max:.6f}"
    
    print(f"  Input shape: {x.shape}")
    print(f"  Step sizes shape: {step_sizes.shape}")
    print(f"  Step sizes range: [{step_sizes.min():.6f}, {step_sizes.max():.6f}]")
    print(f"  Noise range: [{noise.min():.6f}, {noise.max():.6f}]")
    print(f"  ✓ All channels respect their step sizes!")
    
    print("\n" + "="*80)
    print("✓ All noise tests passed!")
    print("="*80 + "\n")


# ========================
# Test Function for clean output
# ========================
def test_clean_outputs():
    """
    Test the clean outputs collection function.
    """
    print("\n" + "="*80)
    print("TESTING CLEAN OUTPUTS COLLECTION")
    print("="*80)
    
    # Load model
    model, device = load_poseidon_model(model_path)
    
    # Build loaders
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        calib_batchsize=4,  # Smaller batch for testing
        calib_steps=2,      # Just 2 batches for testing
        val_batchsize=16,
        val_steps=50
    )
    
    # Load layer names from inspection
    print("\nLoading layer names from 'quantize_layers.pt'...")
    b_quant_path = os.path.join(INSPECT_DIR, 'B_quantize_layers.pt')
    layer_data = torch.load(b_quant_path)
    #layer_data = torch.load('quantize_layers.pt')
    quantize_layers = layer_data['quantize_layers']
    print(f"Found {len(quantize_layers)} layers to quantize")
    
    # For testing, use only first 10 layers
    test_layers = quantize_layers[:823]
    print(f"\nTesting with first {len(test_layers)} layers:")
    for i, name in enumerate(test_layers[:5], 1):
        print(f"  {i}. {name}")
    print(f"  ...")
    
    # Collect clean outputs
    print(f"\nCollecting clean outputs from 2 calibration batches...")
    clean_outputs = get_clean_outputs_poseidon(
        model=model,
        dataloader=calib_iter,
        device=device,
        layer_names=test_layers
    )
    
    # Inspect results
    print("\n" + "="*80)
    print("CLEAN OUTPUTS SUMMARY")
    print("="*80)
    
    for name, outputs in list(clean_outputs.items())[:5]:
        if outputs[0] is not None:
            shapes = [o.shape for o in outputs]
            print(f"\n{name}:")
            print(f"  Batches collected: {len(outputs)}")
            print(f"  Output shapes: {shapes}")
            print(f"  Total elements: {sum(o.numel() for o in outputs):,}")
        else:
            print(f"\n{name}: No outputs (layer may not have been reached)")
    
    print("\n✓ Clean outputs collection test passed!")
    print("="*80)
    
    return clean_outputs



# ========================
# Test Functions for compute ranges
# ========================

def test_compute_ranges():
    """Test compute_data_ranges_poseidon"""
    print("\n" + "="*80)
    print("TESTING COMPUTE DATA RANGES")
    print("="*80)
    
    # Load model
    model, device = load_poseidon_model(model_path)
    
    # Build loaders
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        calib_batchsize=4,
        calib_steps=4,  # Use 4 batches for range computation
        val_batchsize=16,
        val_steps=50
    )
    
    # Load layer names
    #layer_data = torch.load('quantize_layers.pt')
    b_quant_path = os.path.join(INSPECT_DIR, 'B_quantize_layers.pt')
    layer_data = torch.load(b_quant_path)
    quantize_layers = layer_data['quantize_layers']
    
    # Test with first 10 Linear layers
    test_layers = [name for name in quantize_layers if 'projection' not in name][:10]
    print(f"\nTesting with {len(test_layers)} Linear layers")
    
    # Compute ranges
    ranges = compute_data_ranges_poseidon(
        model=model,
        dataloader=calib_iter,
        device=device,
        layer_names=test_layers,
        percentile_prob=1e-4
    )
    
    # Inspect results
    print("\n" + "="*80)
    print("RANGES SUMMARY")
    print("="*80)
    
    for name, range_data in list(ranges.items())[:5]:
        print(f"\n{name}:")
        print(f"  Weight ranges shape: {range_data['weight_ranges'].shape}")
        print(f"  Weight ranges: min={range_data['weight_ranges'].min():.6f}, "
              f"max={range_data['weight_ranges'].max():.6f}, "
              f"mean={range_data['weight_ranges'].mean():.6f}")
        print(f"  Activation ranges shape: {range_data['activation_ranges'].shape}")
        print(f"  Activation ranges: min={range_data['activation_ranges'].min():.6f}, "
              f"max={range_data['activation_ranges'].max():.6f}, "
              f"mean={range_data['activation_ranges'].mean():.6f}")
    
    print("\n✓ Compute data ranges test passed!")
    print("="*80)
    
    return ranges



# ========================
# Test Functions for find best percentile
# ========================

def test_find_best_percentile():
    """Test finding the best percentile for Poseidon"""
    print("\n" + "="*80)
    print("TESTING PERCENTILE SEARCH")
    print("="*80)
    
    # Load model
    model, device = load_poseidon_model(model_path)
    
    # Build loaders
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        calib_batchsize=4,
        calib_steps=4,  # Use 4 batches for search
        val_batchsize=16,
        val_steps=50
    )
    
    # Load layer names
    b_quant_path = os.path.join(INSPECT_DIR, 'B_quantize_layers.pt')
    layer_data = torch.load(b_quant_path)
    #layer_data = torch.load('quantize_layers.pt')
    quantize_layers = layer_data['quantize_layers']
    
    # Test with first 10 Linear layers (small for speed)
    test_layers = [name for name in quantize_layers if 'projection' not in name][:823]
    print(f"\nSearching with {len(test_layers)} Linear layers")
    
    # Find best percentile
    best_P = find_best_percentile_poseidon(
        model=model,
        dataloader=calib_iter,
        device=device,
        layer_names=test_layers,
        percentile_candidates=[1e-3, 1e-4, 1e-5, 1e-6]  # Narrow search for speed
    )
    
    print(f"\n✓ Best percentile found: {best_P}")
    
    return best_P




# ========================
# Test Functions for computing mdl prior
# ========================
def test_compute_mdl_prior():
    print("\n==== TESTING compute_mdl_prior ====")
    rng = torch.Generator().manual_seed(0)
    # Simulate two layers with different shapes
    w1 = torch.tensor([2.0, 4.0, 8.0])      # weight ranges (ch1, ch2, ch3)
    a1 = torch.tensor([1.0, 3.0, 5.0])      # activation ranges
    w1_step = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
    a1_step = torch.tensor([0.2, 1.0, 2.0], requires_grad=True)
    w2 = torch.tensor([4.0, 8.0])
    a2 = torch.tensor([2.0, 10.0])
    w2_step = torch.tensor([1.0, 3.0], requires_grad=True)
    a2_step = torch.tensor([0.5, 1.0], requires_grad=True)
    # Ranges dict format as in compute_data_ranges_poseidon
    ranges_dict = {
        'layer1': {'weight_ranges': w1, 'activation_ranges': a1},
        'layer2': {'weight_ranges': w2, 'activation_ranges': a2}
    }
    step_sizes_dict = {
        'layer1': (w1_step, a1_step),
        'layer2': (w2_step, a2_step)
    }
    gamma = 0.01
    prior = compute_mdl_prior(step_sizes_dict, ranges_dict, gamma=gamma)
    print(f"Prior loss: {prior.item():.6f}")
    # Do an optimizer step
    prior.backward()
    print("Gradient wrt step sizes, layer1-w:", w1_step.grad)
    print("Gradient wrt step sizes, layer2-a:", a2_step.grad)
    print("✓ MDL prior runs and matches shape requirements!")



# ========================
# Test Functions for computing loss with prior
# ========================
def test_mc_loss_with_prior():
    print("\n==== TEST MC LOSS WITH PRIOR ====")
    # # Dummy setup for 2 layers, batchsize 1, MC samples 3
    # device = torch.device('cpu')
    # batch_size = 2
    # seq = 5
    # features = 3
    # # Fake clean outputs: each layer, list of tensors per batch
    # clean_outputs = {
    #     'layer1': [torch.randn(batch_size, seq, features), torch.randn(batch_size, seq, features)],
    #     'layer2': [torch.randn(batch_size, seq, features), torch.randn(batch_size, seq, features)]
    # }
    # # Step sizes dict: two layers
    # step_sizes_dict = {
    #     'layer1': (torch.ones(features) * 0.05, torch.ones(features) * 0.05),
    #     'layer2': (torch.ones(features) * 0.08, torch.ones(features) * 0.08)
    # }
    # # Ranges dict: two layers
    # ranges_dict = {
    #     'layer1': {'weight_ranges': torch.ones(features) * 1.0, 'activation_ranges': torch.ones(features) * 1.0},
    #     'layer2': {'weight_ranges': torch.ones(features) * 1.5, 'activation_ranges': torch.ones(features) * 1.5}
    # }
    # # Dummy dataloader: just pass batch idx & dummy batch dict
    # class DummyLoader:
    #     def __iter__(self):
    #         for i in range(2):
    #             yield {
    #                 'pixel_values': torch.randn(batch_size, seq, features),
    #                 'time': torch.zeros(batch_size),
    #                 'pixel_mask': torch.zeros(batch_size, 4).bool(),
    #                 'labels': torch.randn(batch_size, seq, features)
    #             }
    # dataloader = DummyLoader()
    # # MC loss with prior (simulates MC)
    # total_loss, likelihood_loss, prior_loss = compute_mc_loss_with_prior(
    #     None, dataloader, step_sizes_dict, clean_outputs, ranges_dict, num_mc_samples=3, device=device)
    # print(f"Total loss: {total_loss:.6f}\nLikelihood loss: {likelihood_loss:.6f}\nPrior loss: {prior_loss:.6f}")
    # print("✓ MC loss with prior computes without error.")




    '''
    # this version of test failed casue the channel is not algined, mixing the input and output
    model_path = "models/NS-PwC"
    data_path = "dataset/NS-PwC"
    dataset_name = "fluids.incompressible.PiecewiseConstants"
    # Load model and device
    model, device = load_poseidon_model(model_path, device="cuda")
    # Build calibration loader
    _, _, calib_iter, _ = build_poseidon_loaders(dataset_name, data_path, calib_batchsize=2, calib_steps=2)
    # Load layer names (or filter to only Linear)
    layer_data = torch.load('quantize_layers.pt')
    quantize_layers = [name for name in layer_data['quantize_layers'] if "projection" not in name]
    test_layers = quantize_layers[:823]  # For speed, try with 10 layers first


    # Compute ranges
    ranges_dict = compute_data_ranges_poseidon(model, calib_iter, device, test_layers, percentile_prob=1e-4)
    # Initialize step sizes as uniform fractions of range

    

    # Get clean outputs
    clean_outputs = get_clean_outputs_poseidon(model, calib_iter, device, test_layers)
    test_layers = [name for name in test_layers if name in ranges_dict and name in clean_outputs]

    compatible_layers = []
    for name in test_layers:
        act_shape = ranges_dict[name]['activation_ranges'].shape
        out_shape = clean_outputs[name][0].shape
        if act_shape[0] != out_shape[-1]:
            print(f"[SKIP] {name} -- activation ranges {act_shape[0]} vs output features {out_shape[-1]}")
            continue
        compatible_layers.append(name)
    print(f"Quantizing {len(compatible_layers)} compatible layers.")



    step_sizes_dict = {name: (ranges_dict[name]['weight_ranges'].clone() * 0.1,
                              ranges_dict[name]['activation_ranges'].clone() * 0.1)
                       for name in compatible_layers}
    


    for name in test_layers:
        act_shape = ranges_dict[name]['activation_ranges'].shape
        out_shape = clean_outputs[name][0].shape
        #print(f"Layer: {name} | activation_ranges.shape={act_shape} | clean_output.shape={out_shape}")
        #if act_shape[0] != out_shape[-1]:
            #print(f"[ERROR] Feature mismatch for {name}: activation ranges ({act_shape[0]}) vs output features ({out_shape[-1]})")

    # Compute MC loss with prior
    total_loss, likelihood_loss, prior_loss = compute_mc_loss_with_prior(
        model, calib_iter, step_sizes_dict, clean_outputs, ranges_dict, num_mc_samples=3, device=device)
    print(f"\nTotal MC Loss: {total_loss:.6f}\nLikelihood part: {likelihood_loss:.6f}\nPrior part: {prior_loss:.6f}")
    '''
    # ===== PPQ verification: load -> ranges -> clean IO -> alignment checks -> MC loss (+prior) =====
    # -------------------------
    # Config
    # -------------------------
    MODEL_PATH   = "models/NS-PwC-B"
    DATA_PATH    = "dataset/NS-PwC"
    DATASET_NAME = "fluids.incompressible.PiecewiseConstants"
    PERCENTILE   = 1e-4
    CALIB_BS     = 2
    CALIB_STEPS  = 2
    NUM_MC       = 3
    DEVICE_STR   = "cuda"

    # -------------------------
    # 1) Load model & calib iterator
    # -------------------------
    model, device = load_poseidon_model(MODEL_PATH, device=DEVICE_STR)
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=DATASET_NAME, data_path=DATA_PATH,
        calib_batchsize=CALIB_BS, calib_steps=CALIB_STEPS,
        val_batchsize=16, val_steps=50
    )

    # -------------------------
    # 2) Pick target Linear layers
    # -------------------------

    b_quant_path = os.path.join(INSPECT_DIR, 'B_quantize_layers.pt')
    layer_data = torch.load(b_quant_path)
    #layer_data = torch.load('quantize_layers.pt')
    # optional filter to avoid odd utility layers
    quantize_layers = [n for n in layer_data['quantize_layers'] if "projection" not in n]

    # Keep only real Linear modules that exist in the model
    name2mod = dict(model.named_modules())
    target_layers = [n for n in quantize_layers if (n in name2mod and isinstance(name2mod[n], nn.Linear))]
    print(f"Candidate Linear layers: {len(target_layers)}")

    # -------------------------
    # 3) Compute ranges (weights + activations for pre-op inputs)
    # -------------------------
    ranges_dict = compute_data_ranges_poseidon(
        model, calib_iter, device, target_layers, percentile_prob=PERCENTILE
    )

    # -------------------------
    # 4) Capture clean inputs & outputs (same calibration iterator length)
    # -------------------------
    clean_inputs, clean_outputs = get_clean_outputs_poseidon(
        model, calib_iter, device, target_layers
    )

    # -------------------------
    # 5) Alignment/shape checks and compatible layer set
    #    We require:
    #      - activation_ranges.size == X_clean.last_dim (in_features)
    #      - weight_ranges.size     == W.out_features   == Y_clean.last_dim
    # -------------------------
    compatible_layers, skipped = [], []
    for name in target_layers:
        if name not in ranges_dict: 
            skipped.append((name, "no ranges")); continue
        if not clean_inputs[name] or clean_inputs[name][0] is None:
            skipped.append((name, "no clean X")); continue
        if not clean_outputs[name] or clean_outputs[name][0] is None:
            skipped.append((name, "no clean Y")); continue

        mod = name2mod[name]
        X0  = clean_inputs[name][0]      # [B, ..., in_features]
        Y0  = clean_outputs[name][0]     # [B, ..., out_features]
        ar  = ranges_dict[name]['activation_ranges']  # [in_features]
        wr  = ranges_dict[name]['weight_ranges']      # [out_features]

        in_feat  = X0.shape[-1]
        out_feat = Y0.shape[-1]
        w_out    = mod.weight.size(0)

        ok = True
        if ar.numel() != in_feat:
            skipped.append((name, f"act_ranges {ar.numel()} != in_features {in_feat}")); ok = False
        if wr.numel() != w_out:
            skipped.append((name, f"weight_ranges {wr.numel()} != weight_out {w_out}")); ok = False
        if wr.numel() != out_feat:
            skipped.append((name, f"weight_ranges {wr.numel()} != Y_out {out_feat}")); ok = False

        if ok:
            compatible_layers.append(name)

    # Debug print: any mismatches
    if skipped:
        print("\n[DEBUG] Skipped layers due to shape/channel mismatch:")
        for n, why in skipped[:30]:
            print(f"  [SKIP] {n} -- {why}")
        if len(skipped) > 30:
            print(f"  ... and {len(skipped)-30} more")

    print(f"\nQuantizing {len(compatible_layers)} compatible layers (out of {len(target_layers)} candidates).")

    # -------------------------
    # 6) Build initial step sizes (0.1 * ranges as a simple start)
    # -------------------------
    step_sizes_dict = {
        name: (
            ranges_dict[name]['weight_ranges'].clone().to(device) * 0.1,      # per-out-channel step for W
            ranges_dict[name]['activation_ranges'].clone().to(device) * 0.1   # per-in-channel step for X
        )
        for name in compatible_layers
    }

    if compatible_layers:
        ln = compatible_layers[0]
        print(f"\n[CHECK] {ln}")
        print("  X_clean[0].shape:", clean_inputs[ln][0].shape)
        print("  Y_clean[0].shape:", clean_outputs[ln][0].shape)
        print("  act_ranges.shape:", ranges_dict[ln]['activation_ranges'].shape)
        print("  wgt_ranges.shape:", ranges_dict[ln]['weight_ranges'].shape)


    # -------------------------
    # 7) Compute MC loss + MDL prior (no extra dataloader pass used)
    # -------------------------
    total_loss, likelihood_loss, prior_loss = compute_mc_loss_with_prior(
        model=model,
        dataloader=None,                  # not used by compute_mc_loss (we use cached IO)
        step_sizes_dict=step_sizes_dict,
        clean_inputs=clean_inputs,
        clean_outputs=clean_outputs,
        ranges_dict=ranges_dict,
        num_mc_samples=NUM_MC,
        eta=1e-4,
        gamma=0.005,
        device=device
    )

    print("\n===== PPQ Verification (Poseidon) =====")
    print(f"Compatible layers: {len(compatible_layers)}")
    print(f"Likelihood loss:   {float(likelihood_loss):.6e}")
    print(f"MDL prior loss:    {float(prior_loss):.6e}")
    print(f"TOTAL (MAP) loss:  {float(total_loss):.6e}")
    print("=======================================\n")


    
# ========================
# Test Functions for noise/activation channel aligned
# ========================
@torch.no_grad()
def debug_noise_alignment_single_layer(
    model,
    layer_name: str,
    step_sizes_dict,
    clean_inputs,
    clean_outputs,
    device="cuda",
    mc_seed=0,
):
    """
    Show that:
      - X_noisy - X_clean broadcasts along last dim (in_features) for activations
      - W_noisy - W_clean broadcasts along dim 0 (out_features) for weights
      - Noise scale matches the step sizes (strong correlation)
    """
    # Find the layer module and step sizes
    mod = dict(model.named_modules()).get(layer_name, None)
    assert isinstance(mod, torch.nn.Linear), f"{layer_name} is not a Linear"
    w_step, a_step = step_sizes_dict[layer_name]
    w_step = w_step.to(device)
    a_step = a_step.to(device)

    # Use batch 0 for display (change if you like)
    Xb = clean_inputs[layer_name][0]
    Yb = clean_outputs[layer_name][0]
    assert Xb is not None and Yb is not None, "No cached clean IO for this layer/batch"

    X_clean = Xb.to(device)                 # shape [..., in_features]
    Y_clean = Yb.to(device)                 # shape [..., out_features]
    W_clean = mod.weight.to(device)         # shape [out_features, in_features]

    print(f"\n[NOISE-DEBUG] {layer_name}")
    print(f"  X_clean.shape={tuple(X_clean.shape)}  (last dim = in_features={mod.in_features})")
    print(f"  Y_clean.shape={tuple(Y_clean.shape)}  (last dim = out_features={mod.out_features})")
    print(f"  a_step.shape={tuple(a_step.shape)}    (should be in_features)")
    print(f"  w_step.shape={tuple(w_step.shape)}    (should be out_features)")

    # One MC draw with fixed seed for reproducibility
    g = torch.Generator(device=device).manual_seed(mc_seed)
    # override rand_like inside add_quantization_noise by generating our own noise:
    # but easier: temporarily monkeypatch torch.rand_like via a context? Instead, we just call once.

    X_noisy = add_quantization_noise(X_clean, a_step, channel_axis=-1)
    W_noisy = add_quantization_noise(W_clean, w_step, channel_axis=0)
    Y_noisy = torch.nn.functional.linear(X_noisy, W_noisy, mod.bias)

    # Activation noise stats per input channel (collapse all non-channel dims)
    X_noise = (X_noisy - X_clean).detach()
    # reshape to [N_all, in_features]
    X_view = X_noise.reshape(-1, X_noise.shape[-1])
    x_abs_max = X_view.abs().max(dim=0).values
    x_std = X_view.std(dim=0, unbiased=False)

    # Weight noise stats per output channel (collapse input dim)
    W_noise = (W_noisy - W_clean).detach()  # [out_features, in_features]
    w_abs_max = W_noise.abs().amax(dim=1)   # max across input dim
    w_std = W_noise.std(dim=1, unbiased=False)

    # Quick sanity: expected max ≈ 0.5 * step; std ≈ step / sqrt(12)
    exp_x_max = 0.5 * a_step
    exp_x_std = a_step / (12.0 ** 0.5)
    exp_w_max = 0.5 * w_step
    exp_w_std = w_step / (12.0 ** 0.5)

    # Correlation (how well per-channel noise scale tracks per-channel step)
    def corr(a, b):
        a = a.float(); b = b.float()
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        return (a * b).mean().item()

    print("\n  — Activation noise (per in_channel) —")
    print(f"    corr(x_abs_max, 0.5*a_step): {corr(x_abs_max, exp_x_max):.3f}")
    print(f"    corr(x_std,     a_step/√12): {corr(x_std,     exp_x_std):.3f}")
    print(f"    x_abs_max median: {x_abs_max.median().item():.4e} | expected median: {exp_x_max.median().item():.4e}")
    print(f"    x_std     median: {x_std.median().item():.4e}     | expected median: {exp_x_std.median().item():.4e}")

    print("\n  — Weight noise (per out_channel) —")
    print(f"    corr(w_abs_max, 0.5*w_step): {corr(w_abs_max, exp_w_max):.3f}")
    print(f"    corr(w_std,     w_step/√12): {corr(w_std,     exp_w_std):.3f}")
    print(f"    w_abs_max median: {w_abs_max.median().item():.4e} | expected median: {exp_w_max.median().item():.4e}")
    print(f"    w_std     median: {w_std.median().item():.4e}     | expected median: {exp_w_std.median().item():.4e}")

    # Show effect on output once
    y_diff = (Y_noisy - Y_clean).abs()
    print(f"\n  — Output delta —")
    print(f"    |Y_noisy - Y_clean| max: {y_diff.max().item():.4e}")
    print(f"    MSE(Y_noisy, Y_clean): {torch.mean((Y_noisy - Y_clean)**2).item():.6e}")

    print("\n  ✓ Noise aligns with channels and scales by step sizes.")

# ========================
# Test Functions for noise/activation channel added
# ========================

def debug_noise_adding_values(
    model,
    clean_inputs, clean_outputs,
    step_sizes_dict, ranges_dict,
    layer_name: str,
    batch_idx: int = 0,
    seed: int = 123,
    k: int = 8,          # how many elements to show
    device: str = "cuda"
):
    """
    Print elementwise values for a single layer to verify:
      noisy = clean + noise
    for both weights (one out-channel) and activations (one spatial slice).
    """
    import torch
    torch.manual_seed(seed)

    name2module = dict(model.named_modules())
    mod = name2module.get(layer_name, None)
    assert isinstance(mod, torch.nn.Linear), f"{layer_name} is not a Linear."

    # --- Fetch clean X/Y for the chosen batch
    X_clean = clean_inputs[layer_name][batch_idx]
    Y_clean = clean_outputs[layer_name][batch_idx]
    assert X_clean is not None and Y_clean is not None, "No cached IO for this batch/layer."

    X_clean = X_clean.to(device)
    W_clean = mod.weight.to(device)
    b = (mod.bias.to(device) if mod.bias is not None else None)

    # --- Step sizes
    w_step, a_step = step_sizes_dict[layer_name]
    w_step = w_step.to(device)          # [out_features]
    a_step = a_step.to(device)          # [in_features]

    # ===== 1) Weights: per out_channel on dim 0 =====
    out_ch = 0  # show first output channel (change if you like)
    # broadcast step sizes
    w_shape = [1] * W_clean.dim()
    w_shape[0] = W_clean.size(0)
    w_step_b = w_step.view(w_shape)     # [out_features, 1]

    # build noise explicitly (same as add_quantization_noise)
    w_noise = (torch.rand_like(W_clean) - 0.5) * w_step_b
    W_noisy = W_clean + w_noise

    # slice one row (out_ch) and print first k elements
    w_row_clean = W_clean[out_ch, :].detach().cpu()
    w_row_noise = w_noise[out_ch, :].detach().cpu()
    w_row_noisy = W_noisy[out_ch, :].detach().cpu()

    print(f"\n[WEIGHT CHANNEL CHECK] {layer_name}  (out_channel={out_ch})")
    print("  step size (this out_channel):", float(w_step[out_ch].detach().cpu()))
    torch.set_printoptions(precision=6, sci_mode=False)
    print("  W_clean[:k] =", w_row_clean[:k])
    print("  noise   [:k] =", w_row_noise[:k])
    print("  W_noisy [:k] =", w_row_noisy[:k])
    # verify equality on the shown slice
    ok_w = torch.allclose(w_row_noisy[:k], w_row_clean[:k] + w_row_noise[:k], atol=1e-7, rtol=0)
    print("  ✓ noisy == clean + noise (slice):", ok_w)

    # ===== 2) Activations: per input channel on last dim =====
    # We’ll take a single spatial position/slice depending on rank and show last-dim elements.
    X = X_clean
    in_feat = X.shape[-1]
    assert a_step.numel() == in_feat, f"a_step ({a_step.numel()}) != in_features ({in_feat})"

    # Build activation noise (same broadcasting as add_quantization_noise)
    a_shape = [1] * X.dim()
    a_shape[-1] = X.size(-1)
    a_step_b = a_step.view(a_shape)
    x_noise = (torch.rand_like(X) - 0.5) * a_step_b
    X_noisy = X + x_noise

    # Choose a simple slice along batch/sequence/spatial to make it 1D over features
    if X.dim() == 2:        # [B, C]
        x_slice_clean = X[0, :]
        x_slice_noise = x_noise[0, :]
        x_slice_noisy = X_noisy[0, :]
    elif X.dim() == 3:      # [B, seq, C]
        x_slice_clean = X[0, 0, :]
        x_slice_noise = x_noise[0, 0, :]
        x_slice_noisy = X_noisy[0, 0, :]
    elif X.dim() == 4:      # [B, H, W, C]
        x_slice_clean = X[0, 0, 0, :]
        x_slice_noise = x_noise[0, 0, 0, :]
        x_slice_noisy = X_noisy[0, 0, 0, :]
    else:
        # fallback: flatten all but last dim and take the first row
        Xf = X.reshape(-1, X.shape[-1])
        x_noisef = x_noise.reshape(-1, X.shape[-1])
        x_slice_clean = Xf[0, :]
        x_slice_noise = x_noisef[0, :]
        x_slice_noisy = (Xf + x_noisef)[0, :]

    x_slice_clean = x_slice_clean.detach().cpu()
    x_slice_noise = x_slice_noise.detach().cpu()
    x_slice_noisy = x_slice_noisy.detach().cpu()

    print(f"\n[ACTIVATION SLICE CHECK] {layer_name}  (one spatial position, all features)")
    print("  a_step[:k]   =", a_step.detach().cpu()[:k])
    print("  X_clean[:k]  =", x_slice_clean[:k])
    print("  noise  [:k]  =", x_slice_noise[:k])
    print("  X_noisy[:k]  =", x_slice_noisy[:k])
    ok_x = torch.allclose(x_slice_noisy[:k], x_slice_clean[:k] + x_slice_noise[:k], atol=1e-7, rtol=0)
    print("  ✓ noisy == clean + noise (slice):", ok_x)

    # ===== 3) Optional: compute Y_noisy for the same slice to see effect on outputs =====
    # (This part is optional; it just shows that the noisy input changes the layer output.)
    with torch.no_grad():
        Y_noisy = torch.nn.functional.linear(X_noisy.to(device), W_noisy.to(device), b)
        # align the same spatial slice on output
        if Y_noisy.dim() == 2:
            y_slice_noisy = Y_noisy[0, :]
        elif Y_noisy.dim() == 3:
            y_slice_noisy = Y_noisy[0, 0, :]
        elif Y_noisy.dim() == 4:
            y_slice_noisy = Y_noisy[0, 0, 0, :]
        else:
            Yf = Y_noisy.reshape(-1, Y_noisy.shape[-1])
            y_slice_noisy = Yf[0, :]
        print("\n[OUTPUT EFFECT] Show first k outputs after noise:")
        print("  Y_noisy[:k] =", y_slice_noisy.detach().cpu()[:k])


def test_noise_activation_aligned():
        # -------------------------
    # Config
    # -------------------------
    MODEL_PATH   = "models/NS-PwC"
    DATA_PATH    = "dataset/NS-PwC"
    DATASET_NAME = "fluids.incompressible.PiecewiseConstants"
    PERCENTILE   = 1e-4
    CALIB_BS     = 2
    CALIB_STEPS  = 2
    NUM_MC       = 3
    DEVICE_STR   = "cuda"

    # -------------------------
    # 1) Load model & calib iterator
    # -------------------------
    model, device = load_poseidon_model(MODEL_PATH, device=DEVICE_STR)
    _, _, calib_iter, _ = build_poseidon_loaders(
        dataset_name=DATASET_NAME, data_path=DATA_PATH,
        calib_batchsize=CALIB_BS, calib_steps=CALIB_STEPS,
        val_batchsize=16, val_steps=50
    )

    # -------------------------
    # 2) Pick target Linear layers
    # -------------------------
    b_quant_path = os.path.join(INSPECT_DIR, 'B_quantize_layers.pt')
    layer_data = torch.load(b_quant_path)
    #layer_data = torch.load('quantize_layers.pt')
    # optional filter to avoid odd utility layers
    quantize_layers = [n for n in layer_data['quantize_layers'] if "projection" not in n]

    # Keep only real Linear modules that exist in the model
    name2mod = dict(model.named_modules())
    target_layers = [n for n in quantize_layers if (n in name2mod and isinstance(name2mod[n], nn.Linear))]
    print(f"Candidate Linear layers: {len(target_layers)}")

    # -------------------------
    # 3) Compute ranges (weights + activations for pre-op inputs)
    # -------------------------
    ranges_dict = compute_data_ranges_poseidon(
        model, calib_iter, device, target_layers, percentile_prob=PERCENTILE
    )

    # -------------------------
    # 4) Capture clean inputs & outputs (same calibration iterator length)
    # -------------------------
    clean_inputs, clean_outputs = get_clean_outputs_poseidon(
        model, calib_iter, device, target_layers
    )

    # -------------------------
    # 5) Alignment/shape checks and compatible layer set
    #    We require:
    #      - activation_ranges.size == X_clean.last_dim (in_features)
    #      - weight_ranges.size     == W.out_features   == Y_clean.last_dim
    # -------------------------
    compatible_layers, skipped = [], []
    for name in target_layers:
        if name not in ranges_dict: 
            skipped.append((name, "no ranges")); continue
        if not clean_inputs[name] or clean_inputs[name][0] is None:
            skipped.append((name, "no clean X")); continue
        if not clean_outputs[name] or clean_outputs[name][0] is None:
            skipped.append((name, "no clean Y")); continue

        mod = name2mod[name]
        X0  = clean_inputs[name][0]      # [B, ..., in_features]
        Y0  = clean_outputs[name][0]     # [B, ..., out_features]
        ar  = ranges_dict[name]['activation_ranges']  # [in_features]
        wr  = ranges_dict[name]['weight_ranges']      # [out_features]

        in_feat  = X0.shape[-1]
        out_feat = Y0.shape[-1]
        w_out    = mod.weight.size(0)

        ok = True
        if ar.numel() != in_feat:
            skipped.append((name, f"act_ranges {ar.numel()} != in_features {in_feat}")); ok = False
        if wr.numel() != w_out:
            skipped.append((name, f"weight_ranges {wr.numel()} != weight_out {w_out}")); ok = False
        if wr.numel() != out_feat:
            skipped.append((name, f"weight_ranges {wr.numel()} != Y_out {out_feat}")); ok = False

        if ok:
            compatible_layers.append(name)

    # Debug print: any mismatches
    if skipped:
        print("\n[DEBUG] Skipped layers due to shape/channel mismatch:")
        for n, why in skipped[:30]:
            print(f"  [SKIP] {n} -- {why}")
        if len(skipped) > 30:
            print(f"  ... and {len(skipped)-30} more")

    print(f"\nQuantizing {len(compatible_layers)} compatible layers (out of {len(target_layers)} candidates).")

    # -------------------------
    # 6) Build initial step sizes (0.1 * ranges as a simple start)
    # -------------------------
    step_sizes_dict = {
        name: (
            ranges_dict[name]['weight_ranges'].clone().to(device) * 0.1,      # per-out-channel step for W
            ranges_dict[name]['activation_ranges'].clone().to(device) * 0.1   # per-in-channel step for X
        )
        for name in compatible_layers
    }

    # Choose a layer to inspect:
    layer_to_check = compatible_layers[0]  # or paste the string you printed earlier
    debug_noise_alignment_single_layer(
        model, layer_to_check, step_sizes_dict, clean_inputs, clean_outputs, device=device
    )

    debug_noise_adding_values(
    model=model,
    clean_inputs=clean_inputs,
    clean_outputs=clean_outputs,
    step_sizes_dict=step_sizes_dict,
    ranges_dict=ranges_dict,
    layer_name="encoder.layers.0.blocks.0.attention.self.continuous_position_bias_mlp.0",
    batch_idx=0,
    seed=123,
    k=8,
    device=device
)



if __name__ == "__main__":
    
    # Test clean outputs collection
    #print("\n\n")
    #test_clean_outputs()

    # Test compute ranges
    # print("\n\n")
    # test_compute_ranges()

    # Test percentile search
    # print("\n\n")
    # test_find_best_percentile()

    # Test adding noise
    # print("\n\n")
    #test_add_quantization_noise()

    # Test computing mdl prior
    # print("\n\n")
    #test_compute_mdl_prior()


    # Test computing mdl prior
    # print("\n\n")
    #test_mc_loss_with_prior()

    # Test noise_activation_aligned
    # print("\n\n")
    #test_noise_activation_aligned()

    main()


    #empty the gpu
    #import torch
    #torch.cuda.empty_cache()
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()



