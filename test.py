# transformer_block_woq.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def per_channel_symmetric_int8_fake_quant_weight(w: torch.Tensor):
    """
    w: (out_features, in_features)
    Returns dequantized weight, and per-channel scale.
    """
    # per-output-channel (row-wise) scale
    # scale_i = max(|w_i|)/127; zero_point = 0
    w_view = w.detach()
    max_abs = w_view.abs().amax(dim=1, keepdim=True)  # (out, 1)
    eps = torch.finfo(torch.float32).eps
    scale = torch.clamp(max_abs / 127.0, min=eps)     # (out, 1)
    w_q = torch.round(w_view / scale).clamp_(-127, 127)  # int8 range (symmetric)
    w_dq = w_q * scale  # dequantized back to float
    return w_dq.to(w.dtype), scale.squeeze(1)  # (out,in), (out,)


class WeightOnlyLinear(nn.Module):
    """
    Fake weight-only int8 quantization for nn.Linear.
    Quantizes weights per-output-channel each forward; bias kept fp32.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        # buffers to inspect scales (last forward)
        self.register_buffer("scale_per_out", torch.ones(out_features), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # quantize-dequantize the weight on the fly (no calibration/data needed)
        w_dq, scale = per_channel_symmetric_int8_fake_quant_weight(self.lin.weight)
        self.scale_per_out = scale  # store last-used scales (non-persistent)
        return F.linear(x, w_dq, self.lin.bias)


class SelfAttentionWOQ(nn.Module):
    """
    Self-attention with explicit Q/K/V linear layers so we can quantize their weights.
    """
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = WeightOnlyLinear(dim, dim, bias=True)
        self.k = WeightOnlyLinear(dim, dim, bias=True)
        self.v = WeightOnlyLinear(dim, dim, bias=True)
        self.proj = WeightOnlyLinear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        B, S, C = x.shape

        q = self.q(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,S,D)
        k = self.k(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,S,D)
        v = self.v(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,S,D)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,S,S)

        if attn_mask is not None:
            attn = attn + attn_mask  # broadcast as appropriate

        if key_padding_mask is not None:
            # key_padding_mask: (B, S) == True where padding
            mask = key_padding_mask[:, None, None, :].to(attn.dtype)  # (B,1,1,S)
            attn = attn.masked_fill(mask.bool(), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,H,S,D)
        out = out.transpose(1, 2).reshape(B, S, C)  # (B,S,C)
        out = self.proj_drop(self.proj(out))        # weight-only quantized proj
        return out


class MLPWOQ(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = WeightOnlyLinear(dim, hidden_dim, bias=True)
        self.fc2 = WeightOnlyLinear(hidden_dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)   # keep activations in fp32; you can switch to ReLU if desired
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlockWOQ(nn.Module):
    """
    Pre-LN Transformer block with *weight-only* int8 fake quant on all linears.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, attn_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttentionWOQ(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPWOQ(dim, int(dim * mlp_ratio), dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = x + self.drop1(self.attn(self.norm1(x), attn_mask, key_padding_mask))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)  # fix both inputs and weights
    B, S, C = 2, 16, 64
    x = torch.randn(B, S, C)

    # FP32 reference block (from your earlier code)
    class MLP(nn.Module):
        def __init__(self, dim, hidden_dim, dropout=0.0):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dim)
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            x = F.gelu(self.fc1(x))
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class TransformerBlockFP(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_dropout=0.0, dropout=0.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
            self.drop1 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
            self.drop2 = nn.Dropout(dropout)
        def forward(self, x, attn_mask=None, key_padding_mask=None):
            res = x
            x_norm = self.norm1(x)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            x = res + self.drop1(attn_out)
            res = x
            x = res + self.drop2(self.mlp(self.norm2(x)))
            return x

    fp = TransformerBlockFP(dim=C, num_heads=8)
    woq = TransformerBlockWOQ(dim=C, num_heads=8)

    y_fp = fp(x)
    y_woq = woq(x)

    print("y_fp shape:", y_fp.shape)
    print("y_woq shape:", y_woq.shape)

    # Print outputs
    print("\nOriginal FP32 output (y_fp):")
    print(y_fp)

    print("\nQuantized (WOQ) output (y_woq):")
    print(y_woq)

    # L2 norms
    global_l2 = torch.norm(y_fp - y_woq).item()
    per_batch_l2 = torch.norm(y_fp - y_woq, dim=(1, 2))  # L2 per sample in batch

    print(f"\nGlobal L2 diff: {global_l2:.6f}")
    print(f"Per-batch L2 diff: {per_batch_l2}")