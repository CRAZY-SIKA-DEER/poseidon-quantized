import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------
# 1) Tiny MLP (2-layer)
# y = W2 * (W1 * x)
# -------------------------
class TinyMLP(nn.Module):
    def __init__(self, d_in=1, d_hidden=1, d_out=1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        h = self.fc1(x)
        y = self.fc2(h)
        return y


# -------------------------
# 2) Weight-only uniform noise (like your PPQ MC)
# -------------------------
def add_weight_noise(W, step_sizes, channel_axis=0):
    # step_sizes is per-output-channel
    shape = [1] * W.dim()
    shape[channel_axis] = W.size(channel_axis)
    s = step_sizes.view(shape)
    noise = (torch.rand_like(W) - 0.5) * s
    return W + noise


# -------------------------
# 3) Cache clean layer I/O (pre + post)
# -------------------------
@torch.no_grad()
def cache_clean_io(model, x_batches, layer_names):
    model.eval()
    clean = {n: {"X": [], "Y": []} for n in layer_names}
    name2mod = dict(model.named_modules())

    for xb in x_batches:
        layer_io = {}

        def make_hook(name):
            def hook(mod, inp, out):
                layer_io[name] = (inp[0].detach().clone(), out.detach().clone())
            return hook

        handles = []
        for n in layer_names:
            handles.append(name2mod[n].register_forward_hook(make_hook(n)))

        _ = model(xb)

        for h in handles:
            h.remove()

        for n in layer_names:
            X, Y = layer_io[n]
            clean[n]["X"].append(X)
            clean[n]["Y"].append(Y)

    return clean


# -------------------------
# 4) PPQ MC loss (WEIGHT ONLY)
# -------------------------
def mc_loss_weight_only(model, clean_io, step_sizes, eta=1e-4, num_mc=8):
    """
    step_sizes: dict name -> per-out-channel step tensor (learnable)
    clean_io: cached X/Y per batch per layer
    """
    model.eval()
    name2mod = dict(model.named_modules())
    layer_names = list(step_sizes.keys())

    total = torch.zeros((), device=next(model.parameters()).device)

    # iterate over cached batches
    num_batches = len(clean_io[layer_names[0]]["X"])
    for b in range(num_batches):
        batch_loss = 0.0
        layer_count = 0

        for name in layer_names:
            mod = name2mod[name]
            X = clean_io[name]["X"][b]
            Y_clean = clean_io[name]["Y"][b]

            W = mod.weight
            s = step_sizes[name]  # per out channel

            # MC over weight noise
            mc = []
            for _ in range(num_mc):
                W_noisy = add_weight_noise(W, s, channel_axis=0)
                Y_noisy = F.linear(X, W_noisy, None)
                mc.append(torch.mean((Y_noisy - Y_clean) ** 2) / (2 * eta))

            layer_loss = torch.stack(mc).mean()
            batch_loss = batch_loss + layer_loss
            layer_count += 1

        total = total + batch_loss / max(layer_count, 1)

    return total / num_batches


# -------------------------
# 5) Simple symmetric rounding quantization (per out-channel step)
# -------------------------
def quantize_weight_round(W, step_sizes, channel_axis=0):
    shape = [1] * W.dim()
    shape[channel_axis] = W.size(channel_axis)
    s = step_sizes.view(shape)
    return torch.round(W / s) * s


@torch.no_grad()
def eval_with_weight_rounding(model, x, step_sizes):
    """
    Apply weight rounding quantization to fc1, fc2 using given step sizes
    and run a forward.
    """
    m = model
    name2mod = dict(m.named_modules())

    # save originals
    W_backup = {}
    for name, s in step_sizes.items():
        W_backup[name] = name2mod[name].weight.detach().clone()
        name2mod[name].weight.copy_(quantize_weight_round(name2mod[name].weight, s, channel_axis=0))

    y = m(x)

    # restore
    for name in step_sizes:
        name2mod[name].weight.copy_(W_backup[name])

    return y


# -------------------------
# 6) Demo experiment
# -------------------------
def main(device="cpu"):
    torch.manual_seed(0)
    device = torch.device(device)

    # Build model
    model = TinyMLP(d_in=1, d_hidden=1, d_out=1).to(device)

    # Set weights to create amplification: W2 large
    with torch.no_grad():
        model.fc1.weight[:] = torch.tensor([[1.30]], device=device)   # W1
        model.fc2.weight[:] = torch.tensor([[100.0]], device=device)  # W2 (amplify)

    # Toy data: constant x=1
    x = torch.ones(64, 1, device=device)
    # make batches
    x_batches = [x[i:i+16] for i in range(0, x.size(0), 16)]

    # Cache clean IO for both layers
    layer_names = ["fc1", "fc2"]
    clean_io = cache_clean_io(model, x_batches, layer_names)

    # --- Dynamic baseline: fixed 8-bit steps based on max-abs per out-channel ---
    def dynamic_steps(mod, num_bits=8):
        W = mod.weight.detach()
        out = W.size(0)
        Wflat = W.view(out, -1)
        max_abs = Wflat.abs().amax(dim=1)
        denom = (2**(num_bits-1) - 1)  # symmetric signed levels for int8-ish
        return (max_abs / denom).clamp(min=1e-8)

    dyn_steps = {
        "fc1": dynamic_steps(model.fc1, 8).to(device),
        "fc2": dynamic_steps(model.fc2, 8).to(device),
    }

    y_fp = model(x)
    y_dyn = eval_with_weight_rounding(model, x, dyn_steps)
    dyn_mse = torch.mean((y_dyn - y_fp) ** 2).item()
    print(f"[Dynamic-8] MSE vs FP: {dyn_mse:.6e}")

    # --- PPQ: learn per-layer step sizes (weight-only) ---
    # init from "4-bit" like you do: step = range / 2^bits
    # here range ~ 2*max_abs (simple proxy)
    with torch.no_grad():
        w1_range = 2.0 * model.fc1.weight.detach().abs().view(-1)
        w2_range = 2.0 * model.fc2.weight.detach().abs().view(-1)

    init_bits = 4
    step_sizes = {
        "fc1": nn.Parameter((w1_range / (2**init_bits)).to(device)),
        "fc2": nn.Parameter((w2_range / (2**init_bits)).to(device)),
    }

    opt = optim.Adam(list(step_sizes.values()), lr=1e-2)

    for epoch in range(200):
        opt.zero_grad()
        loss = mc_loss_weight_only(model, clean_io, step_sizes, eta=1e-4, num_mc=16)
        loss.backward()
        opt.step()

        # clamp positive
        with torch.no_grad():
            for k in step_sizes:
                step_sizes[k].clamp_(min=1e-8)

        if (epoch+1) % 50 == 0:
            print(f"[PPQ train] epoch={epoch+1:3d} loss={loss.item():.6e} "
                  f"s1={step_sizes['fc1'].item():.4e} s2={step_sizes['fc2'].item():.4e}")

    # Evaluate PPQ with rounding
    y_ppq = eval_with_weight_rounding(model, x, {k: step_sizes[k].detach() for k in step_sizes})
    ppq_mse = torch.mean((y_ppq - y_fp) ** 2).item()
    print(f"[PPQ] MSE vs FP: {ppq_mse:.6e}")

    # Also print the implied "bits" proxy: log2(range/step)
    with torch.no_grad():
        bits1 = torch.log2((w1_range + 1e-8) / (step_sizes["fc1"].detach() + 1e-8)).item()
        bits2 = torch.log2((w2_range + 1e-8) / (step_sizes["fc2"].detach() + 1e-8)).item()
        print(f"[PPQ] bits proxy: fc1={bits1:.2f}, fc2={bits2:.2f}, avg={(bits1+bits2)/2:.2f}")


if __name__ == "__main__":
    main("cuda" if torch.cuda.is_available() else "cpu")
