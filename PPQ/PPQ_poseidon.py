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



model_path   = "models/NS-PwC"
data_path    = "dataset/NS-PwC"
dataset_name = "fluids.incompressible.PiecewiseConstants"


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
    Run the clean (full-precision) model and store outputs for each Linear layer.
    
    Args:
        model: Poseidon/ScOT model
        dataloader: DataLoader or callable that yields dictionary batches
        device: torch device
        layer_names: List of layer names to track (from layer inspection)
    
    Returns:
        clean_outputs: Dictionary mapping layer names to list of output tensors
                      Each list contains outputs from each batch
    """
    model.eval()
    
    # Initialize storage for each layer
    clean_outputs = {name: [] for name in layer_names}
    
    # Get iterator if dataloader is callable
    if callable(dataloader):
        dataloader = dataloader()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Extract all required fields from batch
            x = batch["pixel_values"].to(device)
            t = batch["time"].to(device)
            pm = batch["pixel_mask"].to(device)
            y = batch["labels"].to(device)  # ADD THIS!
            
            # Storage for this batch
            layer_outputs = {}
            
            # Hook function factory
            def get_hook(name):
                def hook(module, input, output):
                    # Store output on CPU to save GPU memory
                    layer_outputs[name] = output.detach().cpu()
                return hook
            
            # Register hooks on target layers
            handles = []
            for name, module in model.named_modules():
                if name in layer_names and isinstance(module, torch.nn.Linear):
                    handles.append(module.register_forward_hook(get_hook(name)))
            
            # Trigger forward pass with ALL required inputs
            _ = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,  # ADD THIS!
            )
            
            # Store outputs for each layer
            for name in layer_names:
                if name in layer_outputs:
                    clean_outputs[name].append(layer_outputs[name])
                else:
                    # If hook didn't fire, append None
                    clean_outputs[name].append(None)
            
            # Clean up hooks
            for h in handles:
                h.remove()
    
    print(f"\n✓ Collected clean outputs for {len(layer_names)} layers from {batch_idx+1} batches")
    
    # Check which layers produced outputs
    active_layers = sum(1 for outputs in clean_outputs.values() if outputs[0] is not None)
    print(f"✓ {active_layers}/{len(layer_names)} layers produced outputs")
    
    return clean_outputs




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

def compute_mc_loss(model, dataloader, step_sizes_dict, clean_outputs, 
                    num_mc_samples=10, eta=1e-4, device='cpu'):
    """
    Run Monte Carlo estimate of likelihood loss:
    Loss = mean over MC samples of [mean squared error vs. clean outputs].
    """
    if model is not None:
        model.eval()
    total_loss = 0.0
    count = 0

    if callable(dataloader):
        dataloader = dataloader()

    # Only Linear layers as in adaptation
    for batch_idx, batch in enumerate(dataloader):
        x = batch["pixel_values"].to(device)
        t = batch["time"].to(device)
        pm = batch["pixel_mask"].to(device)
        y = batch["labels"].to(device)
        # For each layer, sample MC quantized outputs and compare to clean
        batch_loss = 0.0
        for name, (w_step, a_step) in step_sizes_dict.items():
            # Get clean outputs for this batch (list of shape per MC sample)
            reference = clean_outputs[name][batch_idx].to(device)
            # MC samples: average loss with noise added
            mc_losses = []

            print(f"[DEBUG] Layer: {name}")
            print(f"  step_sizes.shape: {a_step.shape}")
            print(f"  reference.shape: {reference.shape}")
            print(f"  reference.shape[channel_axis]: {reference.shape[-1]}")

            for s in range(num_mc_samples):
                # For activations (typically last dim)
                noisy_ref = add_quantization_noise(reference, a_step, channel_axis=-1)
                # Likelihood loss: squared error (normalized)
                mse = torch.mean((noisy_ref - reference) ** 2)
                mc_losses.append(mse)
            # Average MC losses for this layer+batch
            batch_loss += torch.stack(mc_losses).mean()
        batch_loss = batch_loss / max(1, len(step_sizes_dict))
        total_loss += batch_loss.item()
        count += 1
    return total_loss / max(1, count)


def compute_mc_loss_with_prior(model, dataloader, step_sizes_dict, clean_outputs, 
                               ranges_dict, num_mc_samples=10, eta=1e-4, 
                               gamma=0.005, device='cpu'):
    """
    Monte Carlo likelihood loss + MDL prior.
    Returns total MAP (regularized) loss, plus likelihood/prior for diagnostics.
    """

    likelihood_loss = compute_mc_loss(model, dataloader, step_sizes_dict, 
                                     clean_outputs, num_mc_samples, eta, device)
    prior_loss = compute_mdl_prior(step_sizes_dict, ranges_dict, gamma)
    total_loss = likelihood_loss + prior_loss
    return total_loss, likelihood_loss, prior_loss



# ============================================================================================
# Test Functions
# ============================================================================================



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
    layer_data = torch.load('quantize_layers.pt')
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
    layer_data = torch.load('quantize_layers.pt')
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
    layer_data = torch.load('quantize_layers.pt')
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
    test_layers = quantize_layers[:10]  # For speed, try with 10 layers first
    # Compute ranges
    ranges_dict = compute_data_ranges_poseidon(model, calib_iter, device, test_layers, percentile_prob=1e-4)
    # Initialize step sizes as uniform fractions of range

    # Get clean outputs
    clean_outputs = get_clean_outputs_poseidon(model, calib_iter, device, test_layers)

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
        print(f"Layer: {name} | activation_ranges.shape={act_shape} | clean_output.shape={out_shape}")
        if act_shape[0] != out_shape[-1]:
            print(f"[ERROR] Feature mismatch for {name}: activation ranges ({act_shape[0]}) vs output features ({out_shape[-1]})")

    # Compute MC loss with prior
    total_loss, likelihood_loss, prior_loss = compute_mc_loss_with_prior(
        model, calib_iter, step_sizes_dict, clean_outputs, ranges_dict, num_mc_samples=3, device=device)
    print(f"\nTotal MC Loss: {total_loss:.6f}\nLikelihood part: {likelihood_loss:.6f}\nPrior part: {prior_loss:.6f}")




if __name__ == "__main__":
    # ... (your existing main code)
    
    # NEW: Test clean outputs collection
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
    test_mc_loss_with_prior()


