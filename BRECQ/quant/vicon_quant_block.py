import torch
import torch.nn as nn
import torch.nn.functional as F

from BRECQ.quant.quant_block import BaseQuantBlock
from BRECQ.quant.quant_layer import QuantModule


class QuantMultiheadAttention(nn.Module):
    """
    Quantized wrapper for nn.MultiheadAttention.

    Quantized weights:
      - in_proj_weight: [3 * embed_dim, embed_dim]
      - out_proj.weight: [embed_dim, embed_dim]

    Kept in FP:
      - in_proj_bias
      - out_proj.bias
      - attention softmax/dropout
    """

    def __init__(
        self,
        attn: nn.MultiheadAttention,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        super().__init__()

        assert attn.batch_first is True, "VICON uses batch_first=True attention"
        assert attn._qkv_same_embed_dim is True, "Only packed qkv attention is supported now"

        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.dropout = attn.dropout
        self.batch_first = attn.batch_first
        self.head_dim = attn.head_dim

        self.kdim = attn.kdim
        self.vdim = attn.vdim
        self._qkv_same_embed_dim = attn._qkv_same_embed_dim

        self.in_proj_weight = attn.in_proj_weight
        self.org_in_proj_weight = attn.in_proj_weight.data.clone()
        self.in_proj_bias = attn.in_proj_bias

        self.out_proj_weight = attn.out_proj.weight
        self.org_out_proj_weight = attn.out_proj.weight.data.clone()
        self.out_proj_bias = attn.out_proj.bias

        self.in_proj_weight_quantizer = QuantModule(
            nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=attn.in_proj_bias is not None),
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
        ).weight_quantizer

        self.out_proj_weight_quantizer = QuantModule(
            attn.out_proj,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
        ).weight_quantizer

        self.use_weight_quant = False
        self.use_act_quant = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if self.use_weight_quant:
            in_proj_weight = self.in_proj_weight_quantizer(self.in_proj_weight)
            out_proj_weight = self.out_proj_weight_quantizer(self.out_proj_weight)
        else:
            in_proj_weight = self.org_in_proj_weight
            out_proj_weight = self.org_out_proj_weight

        # F.multi_head_attention_forward expects [seq_len, batch, embed_dim],
        # but VICON/PyTorch TransformerEncoderLayer uses batch_first=True:
        # [batch, seq_len, embed_dim].
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        attn_out, attn_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first:
            attn_out = attn_out.transpose(0, 1)

        return attn_out, attn_weights


class QuantTransformerEncoderLayer(BaseQuantBlock):
    """
    Quantized TransformerEncoderLayer for VICON.

    - Quantize:
        - self_attn.in_proj_weight
        - self_attn.out_proj.weight
        - linear1
        - linear2
    - Keep:
        - norm1, norm2
        - dropout
        - residual
    """

    def __init__(
        self,
        layer: nn.TransformerEncoderLayer,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        super().__init__(act_quant_params)

        self.embed_dim = layer.self_attn.embed_dim
        self.num_heads = layer.self_attn.num_heads

        self.self_attn = QuantMultiheadAttention(
            layer.self_attn,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
        )

        self.linear1 = QuantModule(
            layer.linear1,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
        )
        self.linear2 = QuantModule(
            layer.linear2,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
            disable_act_quant=True,
        )

        self.norm1 = layer.norm1
        self.norm2 = layer.norm2

        self.dropout = layer.dropout
        self.dropout1 = layer.dropout1
        self.dropout2 = layer.dropout2

        self.activation = layer.activation

    def forward(self, x, src_mask=None, is_causal=False, **kwargs):
        residual = x

        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            need_weights=False,
            is_causal=is_causal,
        )

        x = residual + self.dropout1(attn_out)
        x = self.norm1(x)

        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        x = residual + self.dropout2(x)
        x = self.norm2(x)

        if self.use_act_quant:
            x = self.act_quantizer(x)

        return x