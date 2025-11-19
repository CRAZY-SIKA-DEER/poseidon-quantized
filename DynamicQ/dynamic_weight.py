# DynamicQ/dynamic_weight.py

import copy
from typing import Iterable, Union, Set, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

QMIN, QMAX = -127, 127  # symmetric int8 for 8-bit (−2^(b−1)+1, 2^(b−1)−1)


class DynamicWeightLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that does *dynamic* per-out-channel
    symmetric quantization of the weights on every forward.

    - We keep the original fp32 weight in a Parameter.
    - At each forward:
        * compute per-output-channel max abs
        * derive per-channel step size
        * quantize W -> int8
        * dequantize back to fp32
    - Activations are left in fp32 (for now).
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
        # W: [out_features, in_features]
        W = self.weight

        # per-out-channel max abs -> [out_features, 1]
        max_abs = W.abs().amax(dim=1, keepdim=True)  # (O, 1)
        eps = 1e-8
        max_abs = max_abs + eps

        # integer range for signed symmetric b-bit:
        # values in [−(2^(b−1)−1), + (2^(b−1)−1)]
        int_max = float(2 ** (self.bitwidth - 1) - 1)

        # step size per output channel -> [out_features, 1]
        step = max_abs / int_max

        # quantize & dequantize
        W_int = torch.round(W / step).clamp(self.qmin, self.qmax)
        W_deq = W_int * step  # [out_features, in_features]

        return F.linear(x, W_deq, self.bias)


def _set_module_by_name(root: nn.Module, dotted: str, new_module: nn.Module):
    """
    Replace a submodule given its dotted name (e.g., 'encoder.blocks.0.attn.proj').
    """
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def make_dynamic_weight_quantized_copy(
    model: nn.Module,
    quantize_names: Iterable[str],
    bitwidth: int = 8,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """
    Returns a deep-copied model where selected nn.Linear layers are replaced
    with DynamicWeightLinear (dynamic per-out-channel weight quant).

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
            print(f"[DynamicQ] ! Layer '{name}' not found in model; skipping")
            n_skipped += 1
            continue
        if not isinstance(mod, nn.Linear):
            print(f"[DynamicQ] ! Layer '{name}' exists but is not nn.Linear; skipping")
            n_skipped += 1
            continue

        dyn_lin = DynamicWeightLinear(mod, bitwidth=bitwidth)
        _set_module_by_name(model_q, name, dyn_lin)
        n_replaced += 1

    print(f"[DynamicQ] ✓ Dynamic weight quantized copy: replaced {n_replaced} layers; skipped {n_skipped}")
    return model_q
