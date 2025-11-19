# DynamicQ/dynamic_weight_activation.py

import copy
from typing import Iterable, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Signed 8-bit symmetric range (we use the "2^(b-1)-1" convention)
QMIN, QMAX = -127, 127


class DynamicWeightActLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that does *dynamic* per-channel
    quantization of BOTH:
      - weights (per out-feature)
      - activations (per in-feature)

    Both are quantized to int8 and then dequantized back to fp32
    on every forward pass (for fair comparison with PPQ fake-quant).

    Weight quantization:
      - per out-channel (dim=1 on [out_features, in_features])
    Activation quantization:
      - per in-channel (dim=0 on flattened [..., in_features])
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        #bitwidth: int = 8,
        bitwidth: int = 4,
        qmin: int = QMIN,
        qmax: int = QMAX,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.bitwidth = bitwidth
        self.qmin = qmin
        self.qmax = qmax

        # Store original weights & bias as Parameters (frozen)
        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- 1) Dynamic weight quantization (per out-channel) -----
        # W: [out_features, in_features]
        W = self.weight

        # per-out-channel max abs -> [out_features, 1]
        max_abs_w = W.abs().amax(dim=1, keepdim=True)
        eps = 1e-8
        max_abs_w = max_abs_w + eps

        # integer max for signed symmetric b-bit: 2^(b-1)-1
        int_max = float(2 ** (self.bitwidth - 1) - 1)

        # step size per output channel -> [out_features, 1]
        w_step = max_abs_w / int_max

        # quantize & dequantize weights
        W_int = torch.round(W / w_step).clamp(self.qmin, self.qmax)
        W_deq = W_int * w_step  # [out_features, in_features]

        # ----- 2) Dynamic activation quantization (per in-channel) -----
        # x has shape [..., in_features]
        last_dim = x.shape[-1]
        x_reshaped = x.reshape(-1, last_dim)  # [N_flat, in_features]

        # per-in-channel max abs over batch/other dims -> [1, in_features]
        max_abs_a = x_reshaped.abs().amax(dim=0, keepdim=True) + eps

        # same int_max, same bitwidth
        a_step = max_abs_a / int_max  # [1, in_features]

        # quantize & dequantize activations
        x_int = torch.round(x / a_step).clamp(self.qmin, self.qmax)
        x_qdq = x_int * a_step  # same shape as x

        # ----- 3) Linear with quantized-dequantized W and x -----
        return F.linear(x_qdq, W_deq, self.bias)


def _set_module_by_name(root: nn.Module, dotted: str, new_module: nn.Module):
    """
    Replace a submodule given its dotted name (e.g., 'encoder.blocks.0.attn.proj').
    """
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def make_dynamic_weight_activation_quantized_copy(
    model: nn.Module,
    quantize_names: Iterable[str],
    bitwidth: int = 8,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """
    Returns a deep-copied model where selected nn.Linear layers are replaced
    with DynamicWeightActLinear (dynamic per-channel quant of weights + activations).

    Args:
        model:          original fp32 model
        quantize_names: iterable of dotted module names to quantize
        bitwidth:       number of bits to emulate (default 8)
        device:         target device
    """
    device = torch.device(device)
    model_q = copy.deepcopy(model).to(device).eval()

    name_to_module: Dict[str, nn.Module] = dict(model_q.named_modules())
    n_replaced, n_skipped = 0, 0

    for name in quantize_names:
        mod = name_to_module.get(name, None)
        if mod is None:
            print(f"[DynamicQ-Act] ! Layer '{name}' not found in model; skipping")
            n_skipped += 1
            continue
        if not isinstance(mod, nn.Linear):
            print(f"[DynamicQ-Act] ! Layer '{name}' exists but is not nn.Linear; skipping")
            n_skipped += 1
            continue

        dyn_lin = DynamicWeightActLinear(mod, bitwidth=bitwidth)
        _set_module_by_name(model_q, name, dyn_lin)
        n_replaced += 1

    print(f"[DynamicQ-Act] ✓ Dynamic weight+activation quantized copy: "
          f"replaced {n_replaced} layers; skipped {n_skipped}")
    return model_q
