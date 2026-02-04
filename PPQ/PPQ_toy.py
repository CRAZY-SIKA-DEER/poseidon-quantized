# PPQ_toy.py
# Run: python -m PPQ.PPQ_toy
#
# Toy PPQ (weight-only) on a 2-layer MLP:
#   d_in=1, d_hidden=8, d_out=1
# We compare:
#   - Dynamic rounding int8 (max-abs / (2^(b-1)-1))
#   - Dynamic rounding int4 using your requested: (max-min) / 2^(b-1)
#   - PPQ learned step sizes (weight-only) with MDL prior
# And we print per-layer outputs FP32 vs PPQ-rounded.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#for penalise new prior

def _weights_per_out_channel(mod: nn.Module) -> int:
    if isinstance(mod, nn.Linear):
        return mod.in_features
    if isinstance(mod, nn.Conv2d):
        return (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
    raise TypeError(f"Unsupported module type: {type(mod)}")

def compute_bits_proxy_weight_only(step_sizes, ranges_dict, model, layer_names, eps=1e-8):
    """
    Returns:
      bits_k_dict[name] = bits per out-channel (tensor shape [out_features])
      bits_avg_weighted = scalar weighted by number of weights
    """
    device = next(iter(step_sizes.values())).device
    name2mod = dict(model.named_modules())

    total_bits_times_w = torch.zeros((), device=device)
    total_w = torch.zeros((), device=device)
    bits_k_dict = {}

    for name in layer_names:
        s = step_sizes[name]
        R = ranges_dict[name]["weight_ranges"].to(device)

        bits_k = torch.log2((R + eps) / (s + eps))  # [out_features]
        bits_k_dict[name] = bits_k

        w_per_ch = _weights_per_out_channel(name2mod[name])
        total_bits_times_w += bits_k.sum() * w_per_ch
        total_w += bits_k.numel() * w_per_ch

    bits_avg_weighted = total_bits_times_w / (total_w + eps)
    return bits_k_dict, bits_avg_weighted


def prior_weighted_avg_bits_cap(
    step_sizes,
    ranges_dict,
    model,
    layer_names,
    target_bits=4.0,
    lam=1.0,
    alpha=10.0,
    eps=1e-8,
):
    """
    Penalize ONLY if weighted-average bits > target_bits.
    Smooth hinge using softplus. Differentiable.
    """
    _, bits_avg = compute_bits_proxy_weight_only(step_sizes, ranges_dict, model, layer_names, eps=eps)
    excess = bits_avg - target_bits
    penalty = lam * F.softplus(alpha * excess) ** 2
    return penalty

# -------------------------
# 1) Model
# -------------------------
class TinyMLP(nn.Module):
    def __init__(self, d_in=1, d_hidden=8, d_out=1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# -------------------------
# 2) Noise / Quant helpers (weight-only)
# -------------------------
def add_weight_noise(W, step_sizes, channel_axis=0):
    # step_sizes: per-output-channel => for Linear/Conv, output channel axis = 0
    shape = [1] * W.dim()
    shape[channel_axis] = W.size(channel_axis)
    s = step_sizes.view(shape)
    noise = (torch.rand_like(W) - 0.5) * s
    return W + noise


def quantize_weight_round(W, step_sizes, channel_axis=0, eps=1e-8):
    shape = [1] * W.dim()
    shape[channel_axis] = W.size(channel_axis)
    s = step_sizes.view(shape).clamp(min=eps)
    return torch.round(W / s) * s


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
# 4) MC likelihood (weight-only) using cached clean IO
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

    num_batches = len(clean_io[layer_names[0]]["X"])
    for b in range(num_batches):
        batch_loss = 0.0
        layer_count = 0

        for name in layer_names:
            mod = name2mod[name]
            X = clean_io[name]["X"][b]        # pre-op clean input to this layer
            Y_clean = clean_io[name]["Y"][b]  # post-op clean output of this layer

            W = mod.weight
            s = step_sizes[name]  # per out channel

            mc = []
            for _ in range(num_mc):
                W_noisy = add_weight_noise(W, s, channel_axis=0)
                # weight-only: use clean X with noisy W
                Y_noisy = F.linear(X, W_noisy, None)
                mc.append(torch.mean((Y_noisy - Y_clean) ** 2) / (2 * eta))

            layer_loss = torch.stack(mc).mean()
            batch_loss = batch_loss + layer_loss
            layer_count += 1

        total = total + batch_loss / max(layer_count, 1)

    return total / num_batches


# -------------------------
# 5) Ranges for MDL prior (weight-only)
# -------------------------
@torch.no_grad()
def compute_weight_ranges_per_out_channel(model, layer_names, method="2maxabs", eps=1e-8):
    """
    ranges_dict[name]["weight_ranges"] has shape [out_features] (per output channel).
    method:
      - "2maxabs": R_k = 2*max_abs(row_k)  (simple proxy, stable)
      - "maxmin":  R_k = max(row_k) - min(row_k)  (can be small)
    """
    name2mod = dict(model.named_modules())
    ranges_dict = {}

    for name in layer_names:
        mod = name2mod[name]
        W = mod.weight.detach()
        out = W.size(0)
        Wflat = W.view(out, -1)

        if method == "2maxabs":
            R = 2.0 * Wflat.abs().amax(dim=1)
        elif method == "maxmin":
            R = (Wflat.amax(dim=1) - Wflat.amin(dim=1))
        else:
            raise ValueError(f"Unknown method: {method}")

        ranges_dict[name] = {"weight_ranges": R.clamp(min=eps)}

    return ranges_dict


def compute_mdl_prior_weight_only(step_sizes, ranges_dict, gamma=1e-3, eps=1e-8):
    """
    step_sizes: dict name -> learnable step tensor, shape [out_features]
    ranges_dict: dict name -> {"weight_ranges": tensor [out_features]}
    """
    device = next(iter(step_sizes.values())).device
    prior = torch.zeros((), device=device)

    for name, s in step_sizes.items():
        if name not in ranges_dict:
            continue
        R = ranges_dict[name]["weight_ranges"].to(device)
        assert R.numel() == s.numel(), f"{name}: R {R.shape} vs s {s.shape}"
        prior = prior + gamma * torch.sum(torch.log2(torch.clamp(R, min=eps) / torch.clamp(s, min=eps)))

    return prior


# -------------------------
# 6) Dynamic step sizes (baselines)
# -------------------------
@torch.no_grad()
def dynamic_steps_maxabs_per_out_channel(mod: nn.Linear, num_bits: int = 8, eps: float = 1e-8):
    """
    symmetric signed quant grid proxy:
      s_k = max_abs(W_k) / (2^(b-1)-1)
    """
    W = mod.weight.detach()
    out = W.size(0)
    Wflat = W.view(out, -1)
    max_abs = Wflat.abs().amax(dim=1)
    denom = float(2 ** (num_bits - 1) - 1)
    return (max_abs / denom).clamp(min=eps)


@torch.no_grad()
def dynamic_steps_range_per_out_channel(mod: nn.Linear, num_bits: int = 4, eps: float = 1e-8):
    """
    Your requested rule:
      s_k = (max(W_k) - min(W_k)) / 2^(b-1)
    """
    W = mod.weight.detach()
    out = W.size(0)
    Wflat = W.view(out, -1)
    w_max = Wflat.amax(dim=1)
    w_min = Wflat.amin(dim=1)
    denom = float(2 ** (num_bits - 1))
    return ((w_max - w_min) / denom).clamp(min=eps)


# -------------------------
# 7) Eval helpers + printing per-layer outputs
# -------------------------
@torch.no_grad()
def eval_with_weight_rounding(model, x, step_sizes):
    """
    Apply rounding quant to specified layers' weights, run forward, restore weights.
    step_sizes: dict name -> step tensor [out_features]
    """
    m = model
    name2mod = dict(m.named_modules())

    W_backup = {}
    for name, s in step_sizes.items():
        W_backup[name] = name2mod[name].weight.detach().clone()
        name2mod[name].weight.copy_(quantize_weight_round(name2mod[name].weight, s, channel_axis=0))

    y = m(x)

    for name in step_sizes:
        name2mod[name].weight.copy_(W_backup[name])

    return y


@torch.no_grad()
def capture_layer_io(model, x, layer_names):
    model.eval()
    name2mod = dict(model.named_modules())
    out = {}

    handles = []
    def make_hook(name):
        def hook(mod, inp, y):
            out[name] = {"X": inp[0].detach().clone(), "Y": y.detach().clone()}
        return hook

    for n in layer_names:
        handles.append(name2mod[n].register_forward_hook(make_hook(n)))

    _ = model(x)

    for h in handles:
        h.remove()

    return out


@torch.no_grad()
def run_with_weight_rounding_and_capture(model, x, step_sizes_dict, layer_names):
    model.eval()
    name2mod = dict(model.named_modules())
    backup = {}

    for name, s in step_sizes_dict.items():
        mod = name2mod[name]
        backup[name] = mod.weight.detach().clone()
        mod.weight.copy_(quantize_weight_round(mod.weight, s, channel_axis=0))

    q_io = capture_layer_io(model, x, layer_names)

    for name, W0 in backup.items():
        name2mod[name].weight.copy_(W0)

    return q_io


@torch.no_grad()
def print_fp_vs_ppq_layer_outputs(model, x, ppq_steps, layer_names=("fc1", "fc2"), max_print=8):
    fp_io = capture_layer_io(model, x, layer_names)
    ppq_io = run_with_weight_rounding_and_capture(model, x, ppq_steps, layer_names)

    print("\n=== Layer-wise outputs: FP32 vs PPQ-rounded ===")
    for name in layer_names:
        Y_fp = fp_io[name]["Y"]
        Y_q  = ppq_io[name]["Y"]
        mse_Y = torch.mean((Y_q - Y_fp) ** 2).item()

        yf = Y_fp.flatten()[:max_print].cpu()
        yq = Y_q.flatten()[:max_print].cpu()

        print(f"\n[{name}]")
        print(f"  Y_fp[:{max_print}]  = {yf}")
        print(f"  Y_ppq[:{max_print}] = {yq}")
        print(f"  layer Y MSE         = {mse_Y:.6e}")



def init_steps_minmax(mod, num_bits=4, eps=1e-8):
    W = mod.weight.detach()
    out = W.size(0)
    Wflat = W.view(out, -1)
    w_max = Wflat.max(dim=1).values
    w_min = Wflat.min(dim=1).values
    s = (w_max - w_min) / float(2 ** (num_bits - 1))   # your rule
    return s.clamp(min=eps)


@torch.no_grad()
def weighted_bits_per_weight(model, step_sizes, ranges_dict, layer_names, eps=1e-8):
    """
    Returns:
      bits_layer: dict layer -> (unweighted_mean_bits, weighted_bits_per_weight)
      bits_model_weighted: single scalar = weighted avg bits per weight across all layers
    Weighting rule (Linear):
      each output channel k has n_in weights => weight = in_features
    """
    name2mod = dict(model.named_modules())
    device = next(model.parameters()).device

    total_bits_times_w = torch.zeros((), device=device)
    total_w = torch.zeros((), device=device)

    bits_layer = {}

    for name in layer_names:
        mod = name2mod[name]
        s = step_sizes[name].detach()
        R = ranges_dict[name]["weight_ranges"].to(device)

        # per-channel bits proxy
        bits_k = torch.log2((R + eps) / (s + eps))  # shape [out_features]

        # how many weights per output channel?
        if isinstance(mod, nn.Linear):
            w_per_channel = mod.in_features
        elif isinstance(mod, nn.Conv2d):
            # (optional) if you later use convs:
            w_per_channel = (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        else:
            raise TypeError(f"Unsupported module type for weighted bits: {type(mod)}")

        # layer totals
        layer_weighted = (bits_k.sum() * w_per_channel) / (bits_k.numel() * w_per_channel)
        # note: for Linear/Conv, w_per_channel constant across k, so this equals bits_k.mean()
        # BUT the important part is model-level weighting across layers.

        bits_layer[name] = (bits_k.mean().item(), layer_weighted.item())

        # accumulate model-level weighted-by-#weights
        total_bits_times_w += bits_k.sum() * w_per_channel
        total_w += bits_k.numel() * w_per_channel

    bits_model_weighted = (total_bits_times_w / (total_w + eps)).item()
    return bits_layer, bits_model_weighted

@torch.no_grad()
def print_weight_diff_stats(W_fp: torch.Tensor, W_qdq: torch.Tensor, title: str):
    """
    Prints L1 distance and Frobenius norm between two weight tensors.
    L1 = sum |Δ|
    Fro = sqrt(sum Δ^2)
    Also prints mean |Δ| and RMSE for convenience.
    """
    diff = (W_qdq - W_fp).detach()
    l1_sum = diff.abs().sum().item()
    l1_mean = diff.abs().mean().item()

    fro = torch.linalg.norm(diff).item()  # Frobenius for 2D, general for ND
    rmse = torch.sqrt((diff ** 2).mean()).item()

    print(f"\n--- {title} weight diff vs FP32 ---")
    print(f"  L1 sum|Δ|        = {l1_sum:.6e}")
    print(f"  mean|Δ|          = {l1_mean:.6e}")
    print(f"  Frobenius ||Δ||F = {fro:.6e}")
    print(f"  RMSE             = {rmse:.6e}")


@torch.no_grad()
def print_l1_error(name, Y_fp, Y_q):
    diff = Y_q - Y_fp
    mean_l1 = diff.abs().mean().item()
    sum_l1  = diff.abs().sum().item()
    print(f"{name}: mean|Δ| = {mean_l1:.6e}   sum|Δ| = {sum_l1:.6e}")


# -------------------------
# 8) Main
# -------------------------
def main(device="cpu"):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(0)
    device = torch.device(device)

    # Build model
    model = TinyMLP(d_in=3, d_hidden=8, d_out=4).to(device)

    # Set weights (NOTE: your snippet had a bug: you overwrote fc1 twice; here we set fc2 correctly)
    with torch.no_grad():
        model.fc1.weight[:] = torch.tensor([
            [ 1.30,  0.20, -0.10],
            [ 0.90, -0.30,  0.15],
            [ 1.10,  0.10,  0.05],
            [ 1.50, -0.20,  0.25],
            [ 0.70,  0.40, -0.05],
            [ 1.00,  0.05,  0.10],
            [ 1.20, -0.10,  0.30],
            [ 0.80,  0.25, -0.20],
        ], device=device)

        model.fc2.weight[:] = torch.tensor([
            [ 8., -6.,  5.,  7., -4.,  3.,  6., -5.],
            [-7.,  9., -4.,  6.,  5., -3.,  4.,  8.],
            [ 5.,  4.,  8., -6.,  7., -5.,  9., -4.],
            [-6.,  7.,  3.,  8., -5.,  9., -4.,  6.],
        ], device=device)

    # Toy data: constant x=1 (as requested)
    x = torch.ones(64, 3, device=device)
    x_batches = [x[i:i+16] for i in range(0, x.size(0), 16)]
    layer_names = ["fc1", "fc2"]

    # Cache clean IO for PPQ likelihood
    clean_io = cache_clean_io(model, x_batches, layer_names)

    # FP32 outputs
    with torch.no_grad():
        y_fp = model(x)

    # ------------------------------------------------------------------
    # Dynamic-4: step size from your minmax rule, then rounding inference
    # ------------------------------------------------------------------
    dyn_bits = 4
    dyn4_steps = {
        "fc1": init_steps_minmax(model.fc1, dyn_bits).to(device),
        "fc2": init_steps_minmax(model.fc2, dyn_bits).to(device),
    }

    with torch.no_grad():
        y_dyn4 = eval_with_weight_rounding(model, x, dyn4_steps)

    # Also compute the dyn4 quant-dequant weights (for inspection prints)
    W1_dyn4_qdq = quantize_weight_round(model.fc1.weight.detach(), dyn4_steps["fc1"], channel_axis=0)
    W2_dyn4_qdq = quantize_weight_round(model.fc2.weight.detach(), dyn4_steps["fc2"], channel_axis=0)

    # ------------------------------------------------------------------
    # PPQ: learn step sizes (weight-only) with MDL prior
    # ------------------------------------------------------------------
    # prior range R (keep your stable method)
    ranges_dict = compute_weight_ranges_per_out_channel(
        model, layer_names, method="2maxabs", eps=1e-8
    )

    # init from dynamic-4 (minmax rule) as you requested
    step_sizes = {
        "fc1": nn.Parameter(init_steps_minmax(model.fc1, dyn_bits).to(device)),
        "fc2": nn.Parameter(init_steps_minmax(model.fc2, dyn_bits).to(device)),
    }
    init_step_sizes = {k: v.detach().clone() for k, v in step_sizes.items()}

    opt = optim.Adam(list(step_sizes.values()), lr=1e-2)

    eta = 1e-4
    gamma = 1e-3
    num_mc = 16
    epochs = 150

    for epoch in range(epochs):
        opt.zero_grad()

        like = mc_loss_weight_only(model, clean_io, step_sizes, eta=eta, num_mc=num_mc)
        #prior = compute_mdl_prior_weight_only(step_sizes, ranges_dict, gamma=gamma)
        # new prior assign big weight
        prior = prior_weighted_avg_bits_cap(
        step_sizes, ranges_dict, model, layer_names,
        target_bits=4.0, lam=1.0, alpha=1000.0
)

        total = like + prior

        total.backward()
        opt.step()

        # clamp s > 0 (and optionally cap by R)
        with torch.no_grad():
            for name in layer_names:
                s = step_sizes[name]
                s.clamp_(min=1e-8)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            bits_layer, bits_model_w = weighted_bits_per_weight(
                model=model,
                step_sizes=step_sizes,
                ranges_dict=ranges_dict,
                layer_names=layer_names,
            )

            b1_mean, _ = bits_layer["fc1"]
            b2_mean, _ = bits_layer["fc2"]

            print(
                f"[PPQ train] epoch={epoch+1:3d} total={total.item():.6e} "
                f"like={like.item():.6e} prior={prior.item():.6e} | "
                f"s1_mean={step_sizes['fc1'].mean().item():.4e} s2_mean={step_sizes['fc2'].mean().item():.4e} | "
                f"bits(mean): fc1={b1_mean:.2f}, fc2={b2_mean:.2f} | "
                f"bits(model weighted-by-#weights)={bits_model_w:.2f}"
            )

    # PPQ eval (rounding with learned step)
    ppq_steps_detached = {k: step_sizes[k].detach() for k in step_sizes}
    with torch.no_grad():
        y_ppq = eval_with_weight_rounding(model, x, ppq_steps_detached)

    W1_ppq_qdq = quantize_weight_round(model.fc1.weight.detach(), ppq_steps_detached["fc1"], channel_axis=0)
    W2_ppq_qdq = quantize_weight_round(model.fc2.weight.detach(), ppq_steps_detached["fc2"], channel_axis=0)

    # ------------------------------------------------------------------
    # INSPECTION: print everything (FP32 vs Dyn4 vs PPQ)
    # ------------------------------------------------------------------
    x_inspect = x[:8]  # small view

    def forward_dump_with_explicit_weights(xb, W1_qdq, W2_qdq):
        """
        Explicit forward using provided qdq weights (no hooks),
        so we can dump intermediate activations cleanly.
        """
        with torch.no_grad():
            Y1 = xb @ W1_qdq.t()                      # [B, 8]
            Y1_act = torch.relu(Y1) if hasattr(model, "act") else Y1  # if no relu in TinyMLP, keep Y1
            # If your TinyMLP uses relu, replace above with your actual activation.
            # Most of your outputs indicate it's linear, so leaving as identity is ok.
            Y2 = Y1_act @ W2_qdq.t()                  # [B, 4]
        return Y1, Y1_act, Y2

    # FP reference intermediates
    with torch.no_grad():
        # compute FP intermediates using true weights
        W1_fp = model.fc1.weight.detach()
        W2_fp = model.fc2.weight.detach()
        Y1_fp = x_inspect @ W1_fp.t()
        Y1_fp_act = Y1_fp  # adjust if you have activation
        Y2_fp = Y1_fp_act @ W2_fp.t()

    # dyn4 intermediates (using dyn4 qdq weights)
    Y1_dyn4, Y1_dyn4_act, Y2_dyn4 = forward_dump_with_explicit_weights(
        x_inspect, W1_dyn4_qdq, W2_dyn4_qdq
    )

    # ppq intermediates (using ppq qdq weights)
    Y1_ppq, Y1_ppq_act, Y2_ppq = forward_dump_with_explicit_weights(
        x_inspect, W1_ppq_qdq, W2_ppq_qdq
    )

    print("\n" + "="*80)
    print("INSPECTION DUMP (FP32 vs DYN4 quant-dequant vs PPQ quant-dequant)")
    print("="*80)

    print("\n--- input x (first 8) ---")
    print(x_inspect.detach().cpu())

    # Weights / steps
    print("\n--- W1 fp32 (fc1.weight) ---")
    print(W1_fp.cpu())
    print("\n--- W1 step size dyn4 s1_dyn4 (per out-channel) ---")
    print(dyn4_steps["fc1"].detach().cpu())
    print("\n--- W1 step size learned s1_ppq (per out-channel) ---")
    print(ppq_steps_detached["fc1"].detach().cpu())
    print("\n--- W1 qdq dyn4 ---")
    print(W1_dyn4_qdq.cpu())
    print("\n--- W1 qdq ppq ---")
    print(W1_ppq_qdq.cpu())
    print_weight_diff_stats(W1_fp, W1_dyn4_qdq, "W1 (fc1) DYN4")
    print_weight_diff_stats(W1_fp, W1_ppq_qdq,  "W1 (fc1) PPQ")

    print("\n--- W2 fp32 (fc2.weight) ---")
    print(W2_fp.cpu())
    print("\n--- W2 step size dyn4 s2_dyn4 (per out-channel) ---")
    print(dyn4_steps["fc2"].detach().cpu())
    print("\n--- W2 step size learned s2_ppq (per out-channel) ---")
    print(ppq_steps_detached["fc2"].detach().cpu())
    print("\n--- W2 qdq dyn4 ---")
    print(W2_dyn4_qdq.cpu())
    print("\n--- W2 qdq ppq ---")
    print(W2_ppq_qdq.cpu())
    print_weight_diff_stats(W2_fp, W2_dyn4_qdq, "W2 (fc2) DYN4")
    print_weight_diff_stats(W2_fp, W2_ppq_qdq,  "W2 (fc2) PPQ")

    # Layer outputs
    print("\n--- output after layer1 (fc1) FP32 pre-activation ---")
    print(Y1_fp.cpu())
    print("\n--- output after layer1 (fc1) DYN4 pre-activation ---")
    print(Y1_dyn4.cpu())
    print("\n--- output after layer1 (fc1) PPQ pre-activation ---")
    print(Y1_ppq.cpu())

    # If you really have an activation, print it too (currently identity)
    print("\n--- output after layer1 activation FP32 ---")
    print(Y1_fp_act.cpu())
    print("\n--- output after layer1 activation DYN4 ---")
    print(Y1_dyn4_act.cpu())
    print("\n--- output after layer1 activation PPQ ---")
    print(Y1_ppq_act.cpu())

    # --------------------------------------------------
    # L1 error vs FP32 (layer1 output)
    # --------------------------------------------------
    print("\n--- L1 error vs FP32 (layer1 output) ---")
    print_l1_error("DYN4", Y1_fp, Y1_dyn4)
    print_l1_error("PPQ ", Y1_fp, Y1_ppq)

    print("\n--- output after layer2 / final prediction FP32 ---")
    print(Y2_fp.cpu())
    print("\n--- output after layer2 / final prediction DYN4 ---")
    print(Y2_dyn4.cpu())
    print("\n--- output after layer2 / final prediction PPQ ---")
    print(Y2_ppq.cpu())

    # -------------------------
    # L1 error summary (Dyn4 vs PPQ)
    # -------------------------
    with torch.no_grad():
        # mean absolute error over all samples + all output dims
        l1_dyn4_mean = (Y2_dyn4 - Y2_fp).abs().mean().item()
        l1_ppq_mean  = (Y2_ppq  - Y2_fp).abs().mean().item()

        # sum absolute error over all samples + all output dims (optional)
        l1_dyn4_sum = (Y2_dyn4 - Y2_fp).abs().sum().item()
        l1_ppq_sum  = (Y2_ppq  - Y2_fp).abs().sum().item()

    print("\n--- L1 error vs FP32 (final output) ---")
    print(f"DYN4: mean|Δ| = {l1_dyn4_mean:.6e}   sum|Δ| = {l1_dyn4_sum:.6e}")
    print(f"PPQ : mean|Δ| = {l1_ppq_mean:.6e}   sum|Δ| = {l1_ppq_sum:.6e}")


    print("\n" + "="*80)
    print("DONE")
    print("="*80)



if __name__ == "__main__":
    main("cuda" if torch.cuda.is_available() else "cpu")









    # # Toy "real-ish" calibration data (diverse x)
    # x, x_batches = make_calib_x_batches_smooth_fields(
    #     num_samples=256,
    #     batch_size=16,
    #     d_in=1,
    #     length=128,
    #     device=device,
    #     seed=0,
    # )


    # x, x_batches = make_calib_x_batches_line(
    # num_samples=256,
    # batch_size=16,
    # device=device,
    # seed=0,
    # )
