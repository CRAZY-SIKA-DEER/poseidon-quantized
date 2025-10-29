import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import copy
import math

# ========================
# 1. Simple Model Example
# ========================
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ========================
# 2. Helper: Add Noise to Weights/Activations
# ========================
def add_quantization_noise(tensor, step_sizes, channel_axis):
    """
    Add uniform quantization noise to tensor. 
    Each unique step size is coresponding to the output channel, where means each filter shares the same step size
    
    Args:
        tensor: Input tensor (weights or activations)
        step_sizes: Per-channel step sizes (1D tensor)
        channel_axis: Which dimension represents channels
    
    Returns:
        Noisy tensor
    """
    # Broadcast step_sizes to match tensor shape along channel_axis
    shape = [1] * tensor.dim()
    shape[channel_axis] = tensor.size(channel_axis)
    step_sizes_broadcast = step_sizes.view(shape)
    
    # Sample uniform noise in [-0.5, 0.5] and scale by step size
    noise = (torch.rand_like(tensor) - 0.5) * step_sizes_broadcast
    return tensor + noise

# ========================
# 3. Store Clean Outputs
# ========================
def get_clean_outputs(model, dataloader, device):
    """
    Run the clean (full-precision) model and store outputs for each layer.

    Say model:
    Sequential(
    conv1: Conv2d(3,16,3),
    conv2: Conv2d(16,32,3),
    fc: Linear(128,10)
    )

    and dataloader has 2 batches of size 4 (total 8 samples). Then:
    clean_outputs = {
    'conv1': tensor(8,16,H1,W1),
    'conv2': tensor(8,32,H2,W2),
    'fc': tensor(8,10)
    }

    Those are the clean reference outputs.

    notion, storignt eh output of each channel may be too big for GPU
    """
    model.eval()
    clean_outputs = {name: [] for name, _ in model.named_modules() if isinstance(_, (nn.Conv2d, nn.Linear))}
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            # Forward with hooks to capture intermediate outputs, insert teh hook between layers
            handles = []
            layer_outputs = {}
            
            def get_hook(name):
                def hook(module, input, output):
                    layer_outputs[name] = output.detach().cpu()  # ← store on CPU
                return hook
            
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    handles.append(module.register_forward_hook(get_hook(name)))
            
            # trigger the forward hooks
            _ = model(inputs)
            
            # Append per-layer result (or None if the hook never fired)
            for name in clean_outputs.keys():
                clean_outputs[name].append(layer_outputs.get(name, None))
            
            for h in handles:
                h.remove()
    
    # Concatenate all batches
    # for name in clean_outputs:
    #     clean_outputs[name] = torch.cat(clean_outputs[name], dim=0)
    

    # inside get_clean_outputs (don’t cat at the end)
    # clean_outputs[name] stays a list: [batch0_tensor, batch1_tensor, ...]
    return clean_outputs

# ========================
# 4. Monte Carlo Loss Calculation
# ========================
def compute_mc_loss(model, dataloader, step_sizes_dict, clean_outputs, 
                    num_mc_samples=10, eta=1e-4, device='cpu'):
    model.eval()
    total_loss = torch.zeros((), device=device)
    batch_idx = 0
    
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        batch_size = inputs.size(0)
        mc_loss = torch.zeros((), device=device)
        
        for _mc in range(num_mc_samples):
            layer_outputs = {}
            handles = []

            # Build a hook that quantizes X (input) and W, then runs the op in-graph
            def get_hook(name, weight_step_sizes, activation_step_sizes, module):
                def hook(_module, hook_input, hook_output):
                    x = hook_input[0]  # input to this layer

                    # Activation quantization on input tensor (pre-op)
                    if activation_step_sizes is not None:
                        # Conv: NCHW -> axis=1 ; Linear: [N,D] -> axis=1
                        x_noisy = add_quantization_noise(x, activation_step_sizes, channel_axis=1)
                    else:
                        x_noisy = x

                    # Weight quantization (per-output-channel)
                    if weight_step_sizes is not None:
                        w_noisy = add_quantization_noise(module.weight, weight_step_sizes, channel_axis=0)
                    else:
                        w_noisy = module.weight

                    # Compute layer output with (X', W')
                    if isinstance(module, nn.Conv2d):
                        y = F.conv2d(
                            x_noisy, w_noisy, module.bias,
                            stride=module.stride, padding=module.padding,
                            dilation=module.dilation, groups=module.groups
                        )
                    elif isinstance(module, nn.Linear):
                        y = F.linear(x_noisy, w_noisy, module.bias)
                    else:
                        y = hook_output  # not expected here

                    layer_outputs[name] = y
                    return y  # replace module's output
                return hook

            # Register hooks on all target layers
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and name in step_sizes_dict:
                    weight_step_sizes, activation_step_sizes = step_sizes_dict[name]
                    handles.append(
                        module.register_forward_hook(
                            get_hook(name, weight_step_sizes, activation_step_sizes, module)
                        )
                    )

            # Forward pass: this triggers hooks layer-by-layer
            _ = model(inputs)

            # Clean up hooks
            for h in handles:
                h.remove()

            # Gaussian NLL surrogate across layers
            layer_loss = torch.zeros((), device=device)
            for name in step_sizes_dict.keys():
                # skip if the layer didn't fire in this forward
                if name not in layer_outputs:
                    continue
                # skip if the clean pass also didn't produce this layer for this batch
                clean_out_batch = clean_outputs[name][batch_idx]
                if clean_out_batch is None:
                    continue

                clean_out = clean_outputs[name][batch_idx].to(device, non_blocking=True)  # exact matching batch
                noisy_out = layer_outputs[name]
                layer_loss += torch.sum((clean_out - noisy_out) ** 2) / (2 * eta)

            mc_loss += layer_loss
            

        mc_loss /= num_mc_samples
        total_loss += mc_loss
        batch_idx += 1
 
    
    return total_loss / len(dataloader)



# ========================
# 5. For MDL prior
# ========================



# ========================
# Helper: Inverse Error Function
# ========================
def erf_inv(x):
    """
    Inverse error function approximation.
    PyTorch has torch.erfinv() built-in.
    """
    return torch.erfinv(x)

# ========================
# Compute Data Ranges with Percentile Clipping
# ========================
def compute_data_ranges(model, dataloader, device, percentile_prob=1e-4):
    """
    Compute the data range [R_l]_k for each channel using percentile-based clipping.
    """
    model.eval()
    ranges_dict = {}
    
    # Pre-compute the erf_inv constant (scalar)
    erf_inv_value = math.sqrt(2) * float(torch.erfinv(torch.tensor(1 - 2 * percentile_prob)))
    
    with torch.no_grad():
        # ==============================
        # 1. Weight Ranges (Percentile-based)
        # ==============================
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data.to(device)  # ← Ensure on correct device
                
                # Flatten per output channel: [out_channels, -1]
                w_flat = weight.view(weight.size(0), -1)
                
                # Compute μ and σ per channel
                w_mean = w_flat.mean(dim=1)  # [out_channels]
                w_std = w_flat.std(dim=1)    # [out_channels]
                
                # Compute percentile threshold τ
                # τ = μ + σ * sqrt(2) * erf_inv(1 - 2*P)
                tau = w_mean + w_std * erf_inv_value  # All on same device now
                
                # Clipping thresholds (symmetric)
                beta = tau                    # Upper threshold
                alpha = 2 * w_mean - beta     # Lower threshold
                
                # Range = beta - alpha
                weight_ranges = (beta - alpha).clamp(min=1e-8)
                
                # Store
                ranges_dict[name] = {
                    'weight_ranges': weight_ranges.to(device),
                    'act_stats': []
                }
        
        # ==============================
        # 2. Activation Ranges (Percentile-based)
        # ==============================
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            def get_hook(name):
                def hook(module, input, output):
                    x = input[0]  # Pre-op activation
                    
                    # Flatten per channel: [channels, -1]
                    x_flat = x.transpose(0, 1).reshape(x.size(1), -1)
                    
                    # Compute μ and σ per channel
                    x_mean = x_flat.mean(dim=1)
                    x_std = x_flat.std(dim=1)
                    
                    # Store statistics for later aggregation
                    ranges_dict[name]['act_stats'].append({
                        'mean': x_mean.cpu(),
                        'std': x_std.cpu()
                    })
                return hook
            
            handles = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    handles.append(module.register_forward_hook(get_hook(name)))
            
            _ = model(inputs)
            
            for h in handles:
                h.remove()
        
        # ==============================
        # 3. Aggregate Activation Stats and Compute Ranges
        # ==============================
        for name in ranges_dict:
            if not ranges_dict[name]['act_stats']:
                continue
            
            # Average μ and σ across batches
            all_means = torch.stack([s['mean'] for s in ranges_dict[name]['act_stats']])
            all_stds = torch.stack([s['std'] for s in ranges_dict[name]['act_stats']])
            
            avg_mean = all_means.mean(dim=0).to(device)
            avg_std = all_stds.mean(dim=0).to(device)
            
            # Compute percentile threshold τ
            tau = avg_mean + avg_std * erf_inv_value  # Use pre-computed constant
            
            # Clipping thresholds
            beta = tau
            alpha = 2 * avg_mean - tau
            
            # Range = beta - alpha
            activation_ranges = (beta - alpha).clamp(min=1e-8)
            
            # Store final ranges
            ranges_dict[name]['activation_ranges'] = activation_ranges
            del ranges_dict[name]['act_stats']  # Clean up
    
    return ranges_dict





# ========================
# OPTIONAL: Grid Search for Best Percentile
# ========================
def find_best_percentile(model, dataloader, device, 
                         percentile_candidates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
    """
    Grid search to find the best percentile probability.
    
    Returns:
        best_percentile: The P value that gives lowest CE loss
    """
    best_loss = float('inf')
    best_percentile = percentile_candidates[0]
    
    print("Searching for best percentile...")
    for P in percentile_candidates:
        # Compute ranges with this P
        ranges = compute_data_ranges(model, dataloader, device, percentile_prob=P)
        
        # Simple quantization test (8-bit uniform)
        # Forward pass and compute loss
        loss = evaluate_quantized_model(model, dataloader, ranges, device)
        
        print(f"  P={P:.1e}: Loss={loss:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            best_percentile = P
    
    print(f"Best percentile: {best_percentile:.1e}")
    return best_percentile


def evaluate_quantized_model(model, dataloader, ranges, device):
    """
    Helper: Evaluate model with simple 8-bit quantization.
    Returns cross-entropy loss.
    """
    # Simplified evaluation (you can expand this)
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Simple forward (in practice, apply quantization here)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)







# ========================
# NEW: Compute MDL Prior Loss
# ========================
def compute_mdl_prior(step_sizes_dict, ranges_dict, gamma=0.001, eps=1e-8):
    """
    MDL prior: gamma * sum_k log2(R_k / S_k) over weights and activations.
    Encourages larger step sizes S (i.e., fewer effective bits).
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

        # ---- Weights: ranges per OUT channel/feature
        w_ranges = rec['weight_ranges']
        if w_ranges is not None:
            w_ranges = w_ranges.to(device)
            # Optional sanity check
            assert w_ranges.numel() == weight_step_sizes.numel(), \
                f"[{name}] weight ranges len={w_ranges.numel()} vs step sizes len={weight_step_sizes.numel()}"
            w_term = torch.log2(torch.clamp(w_ranges, min=eps) / torch.clamp(weight_step_sizes, min=eps))
            prior_loss = prior_loss + gamma * torch.sum(w_term)

        # ---- Activations: ranges per IN channel/feature (pre-op design)
        a_ranges = rec.get('activation_ranges', None)
        if a_ranges is not None:
            a_ranges = a_ranges.to(device)
            assert a_ranges.numel() == activation_step_sizes.numel(), \
                f"[{name}] activation ranges len={a_ranges.numel()} vs step sizes len={activation_step_sizes.numel()}"
            a_term = torch.log2(torch.clamp(a_ranges, min=eps) / torch.clamp(activation_step_sizes, min=eps))
            prior_loss = prior_loss + gamma * torch.sum(a_term)

    return prior_loss


# ========================
# UPDATED: Compute MC Loss with Prior
# ========================
def compute_mc_loss_with_prior(model, dataloader, step_sizes_dict, clean_outputs, 
                                ranges_dict, num_mc_samples=10, eta=1e-4, 
                                gamma=0.005, device='cpu'):
    """
    Compute Monte Carlo loss + MDL prior.
    
    Total loss = Likelihood loss (negative log-likelihood) + Prior loss
    """
    # Likelihood term (existing)
    likelihood_loss = compute_mc_loss(model, dataloader, step_sizes_dict, 
                                     clean_outputs, num_mc_samples, eta, device)
    
    # Prior term (new)
    prior_loss = compute_mdl_prior(step_sizes_dict, ranges_dict, gamma)
    
    # Total MAP objective: maximize log p(Y|S) + log p(S)
    # = minimize -log p(Y|S) - log p(S)
    total_loss = likelihood_loss + prior_loss
    
    return total_loss, likelihood_loss, prior_loss


# ========================
# UPDATED: Optimize Step Sizes with MDL
# ========================
def optimize_step_sizes(model, dataloader, num_epochs=50, num_mc_samples=10, 
                       lr=1e-2, eta=1e-4, gamma=0.001, device='cpu', percentile_prob = 1e-4):
    """
    Optimize per-channel step sizes using Monte Carlo + MDL prior.
    
    Args:
        gamma: MDL penalty strength (0.001 to 0.019 typical range)
        percentile_prob: Probability for percentile clipping (default 1e-4)
    """
    model = model.to(device)
    model.eval()
    
    # Get clean outputs
    print("Computing clean outputs...")
    clean_outputs = get_clean_outputs(model, dataloader, device)
    
    # Compute data ranges for MDL prior (using the passed percentile_prob)
    print(f"Computing data ranges with percentile_prob={percentile_prob}...")
    ranges_dict = compute_data_ranges(model, dataloader, device, percentile_prob=percentile_prob)
    
    # Initialize step sizes for each layer
    step_sizes_dict = {}
    optimizable_params = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Weight step sizes
            weight_step_sizes = nn.Parameter(
                torch.full((module.out_channels if isinstance(module, nn.Conv2d) else module.out_features,), 
                          0.1, device=device)
            )
            
            # Activation step sizes
            activation_step_sizes = nn.Parameter(
                torch.full((module.in_channels if isinstance(module, nn.Conv2d) else module.in_features,), 
                          0.1, device=device)
            )
            
            step_sizes_dict[name] = (weight_step_sizes, activation_step_sizes)
            optimizable_params.extend([weight_step_sizes, activation_step_sizes])
    
    optimizer = optim.Adam(optimizable_params, lr=lr)
    
    print(f"Optimizing step sizes with gamma={gamma}...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute MC loss + MDL prior
        total_loss, likelihood_loss, prior_loss = compute_mc_loss_with_prior(
            model, dataloader, step_sizes_dict, clean_outputs, ranges_dict,
            num_mc_samples, eta, gamma, device
        )
        
        # Backpropagate
        total_loss.backward()
        optimizer.step()
        
        # Clamp step sizes to positive values
        with torch.no_grad():
            for name, (w_step, a_step) in step_sizes_dict.items():
                w_step.clamp_(min=1e-5)
                a_step.clamp_(min=1e-5)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss.item():.4f}, "
                  f"Likelihood: {likelihood_loss.item():.4f}, Prior: {prior_loss.item():.4f}")
            
            # Compute average bit-width for monitoring
            avg_bits = 0.0
            total_channels = 0
            with torch.no_grad():
                for name, (w_step, a_step) in step_sizes_dict.items():
                    if name in ranges_dict:
                        w_ranges = ranges_dict[name]['weight_ranges']
                        a_ranges = ranges_dict[name]['activation_ranges']
                        w_bits = torch.log2((w_ranges + 1e-8) / (w_step + 1e-8))
                        a_bits = torch.log2((a_ranges + 1e-8) / (a_step + 1e-8))
                        avg_bits += w_bits.sum().item() + a_bits.sum().item()
                        total_channels += len(w_step) + len(a_step)
            avg_bits /= total_channels
            print(f"  Average bit-width: {avg_bits:.2f}")

    return step_sizes_dict





# ========================
# 7. Fake Quantized Evaluation
# ========================
import torch.nn.functional as F
import copy

def quantize_per_channel_sym(x, step_sizes, channel_axis):
    """Fake symmetric per-channel quantization."""
    shape = [1] * x.dim()
    shape[channel_axis] = step_sizes.numel()
    s = step_sizes.view(shape)
    q = torch.clamp(torch.round(x / s), -65536, 65536)  # int8 grid, symmetric
    return q * s  # dequantized (fake quantized) tensor


def make_fake_quantized_model(model, step_sizes_dict):
    """
    Wraps model with forward hooks that inject fake quantization
    on inputs and weights using learned step sizes.
    """
    qmodel = copy.deepcopy(model)
    qmodel.eval()

    handles = []
    for name, module in qmodel.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name in step_sizes_dict:
            weight_step_sizes, activation_step_sizes = step_sizes_dict[name]

            def hook_factory(_module, _w_step, _a_step):
                def hook(mod, inp, out):
                    x = inp[0]
                    # Activation quant (pre-op): NCHW/ND → channel_axis=1
                    xq = quantize_per_channel_sym(
                        x, _a_step.to(x.device), channel_axis=1
                    )
                    # Weight quant (per-output-channel): channel_axis=0
                    wq = quantize_per_channel_sym(
                        mod.weight, _w_step.to(x.device), channel_axis=0
                    )
                    if isinstance(mod, nn.Conv2d):
                        y = F.conv2d(
                            xq, wq, mod.bias,
                            stride=mod.stride, padding=mod.padding,
                            dilation=mod.dilation, groups=mod.groups
                        )
                    else:
                        y = F.linear(xq, wq, mod.bias)
                    return y
                return hook

            h = module.register_forward_hook(
                hook_factory(module, weight_step_sizes, activation_step_sizes)
            )
            handles.append(h)

    return qmodel, handles


@torch.no_grad()
def compare_models(model_fp32, step_sizes_dict, dataloader, device):
    """
    Compare FP32 and fake-quantized model outputs.
    Returns MSE between them.
    """
    qmodel, handles = make_fake_quantized_model(model_fp32, step_sizes_dict)
    model_fp32.to(device).eval()
    qmodel.to(device).eval()

    mse_total, count = 0.0, 0
    for x, _ in dataloader:
        x = x.to(device)
        y_fp = model_fp32(x)
        y_q = qmodel(x)
        mse_total += torch.sum((y_fp - y_q) ** 2).item()
        count += y_fp.numel()

    mse = mse_total / count
    for h in handles:
        h.remove()

    print(f"\n[Comparison] Average MSE (Quantized vs FP32): {mse:.6e}")
    return mse


# ========================
# 8. Main Execution
# ========================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup
    num_samples = 100
    inputs = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    model = SimpleConvNet()
    model = model.to(device)  #move model to GPU
    
    # Search for best percentile ONCE (outside the gamma loop if you want)
    print("Searching for best percentile...")
    best_P = find_best_percentile(
        model, dataloader, device,
        percentile_candidates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    )
    print(f"Best percentile: {best_P}")
    
    # Now optimize with different gamma values using the same best_P
    for gamma in [0.001, 0.004, 0.007]:
        print(f"\n{'='*60}")
        print(f"Training with gamma = {gamma}")
        print(f"{'='*60}")
        
        step_sizes = optimize_step_sizes(
            model, dataloader, 
            num_epochs=50, 
            num_mc_samples=10, 
            lr=1e-2,
            gamma=gamma,
            device=device,
            percentile_prob=best_P  # ← Pass the found best_P
        )
        
        # Save the learned scales
        payload = {
            "step_sizes": {k: (w.detach().cpu(), a.detach().cpu())
                           for k, (w, a) in step_sizes.items()},
            "gamma": gamma,
            "percentile_prob": best_P
        }
        torch.save(payload, f"learned_step_sizes_gamma{gamma}.pt")
        
        # Compare fake-quant vs FP32
        mse = compare_models(model, step_sizes, dataloader, device)
        print(f"Gamma={gamma:.3f} → MSE between FP32 and Quantized: {mse:.6f}")


