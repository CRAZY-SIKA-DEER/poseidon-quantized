import torch
import torch.nn as nn

# ========================
# NEW: Compute Data Ranges
# ========================
def compute_data_ranges(model, dataloader, device):
    """
    Compute the data range [R_l]_k for each channel in each layer.
    R = max - min across all values in that channel.
    """
    model.eval()
    ranges_dict = {}
    
    with torch.no_grad():
        # Track min/max for weights
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Weight ranges: per output channel (axis=0)
                weight = module.weight.data
                w_flat = weight.view(weight.size(0), -1)  # [out_channels, -1]
                w_max = w_flat.max(dim=1)[0] # w_max = tensor([1.2, 3.4, 5.6,....]) = the number of filters
                w_min = w_flat.min(dim=1)[0]
                weight_ranges = w_max - w_min
                
                # Activation ranges: need to run forward pass
                ranges_dict[name] = {
                    'weight_ranges': weight_ranges.to(device),
                    'act_min': None,
                    'act_max': None
                }
        
        # Track min/max for activations
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            def get_hook(name):
                def hook(module, input, output):
                    x = input[0]  # Pre-op activation
                    # Per channel: axis=1
                    x_flat = x.transpose(0, 1).reshape(x.size(1), -1)  # [channels, -1]
                    x_max = x_flat.max(dim=1)[0]
                    x_min = x_flat.min(dim=1)[0]
                    
                    if ranges_dict[name]['act_min'] is None:
                        ranges_dict[name]['act_min'] = x_min
                        ranges_dict[name]['act_max'] = x_max
                    else:
                        ranges_dict[name]['act_min'] = torch.min(ranges_dict[name]['act_min'], x_min)
                        ranges_dict[name]['act_max'] = torch.max(ranges_dict[name]['act_max'], x_max)
                return hook
            
            handles = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    handles.append(module.register_forward_hook(get_hook(name)))
            
            _ = model(inputs)
            
            for h in handles:
                h.remove()
    
    # Compute activation ranges
    for name in ranges_dict:
        act_ranges = ranges_dict[name]['act_max'] - ranges_dict[name]['act_min']
        ranges_dict[name]['activation_ranges'] = act_ranges
        # Clean up temporary values
        del ranges_dict[name]['act_min']
        del ranges_dict[name]['act_max']
    
    return ranges_dict


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
                       lr=1e-2, eta=1e-4, gamma=0.001, device='cpu'):
    """
    Optimize per-channel step sizes using Monte Carlo + MDL prior.
    
    Args:
        gamma: MDL penalty strength (0.001 to 0.019 typical range)
    """
    model = model.to(device)
    model.eval()
    
    # Get clean outputs
    print("Computing clean outputs...")
    clean_outputs = get_clean_outputs(model, dataloader, device)
    
    # Compute data ranges for MDL prior
    print("Computing data ranges for MDL prior...")
    ranges_dict = compute_data_ranges(model, dataloader, device)
    
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
# Usage Example
# ========================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup (same as before)
    num_samples = 100
    inputs = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    model = SimpleConvNet()
    
    # Optimize with different gamma values
    for gamma in [0.001, 0.005, 0.010]:
        print(f"\n{'='*60}")
        print(f"Training with gamma = {gamma}")
        print(f"{'='*60}")
        
        step_sizes = optimize_step_sizes(
            model, dataloader, 
            num_epochs=50, 
            num_mc_samples=10, 
            lr=1e-2,
            gamma=gamma,  # ← MDL penalty strength
            device=device
        )
