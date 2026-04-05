'''
poseidon_quant_model.py
This file does model traversal and replacement.
It answers:
where to walk in the whole model
which child to replace
when to replace nn.Linear / nn.Conv2d
when to replace a whole ScOTLayer / ConvNeXtBlock / ResNetBlock
So it is the dispatcher / replacer.
poseidon_quant_block.py
This file defines what the replaced block actually is.
It answers:
if I replace ScOTLayer, what module should replace it?
what forward logic should that quantized block use?
which sublayers inside the block are quantized?
where is the residual add / norm / activation handled?
So it is the block wrapper definition.
Without this file, poseidon_quant_model.py would know that it should replace ScOTLayer, but not with what.

'''



import math
import collections
import torch
import torch.nn as nn

from BRECQ.quant.quant_block import BaseQuantBlock
from BRECQ.quant.quant_layer import QuantModule

# Import your Poseidon/ScOT classes from the real path in your repo.
# You may need to change this line depending on your repo structure.
from scOT.model import ScOTLayer, ConvNeXtBlock, ResNetBlock


class QuantConvNeXtBlock(BaseQuantBlock):
    """
    Quantized version of Poseidon ConvNeXtBlock.

    Notes:
    - We keep the original forward structure.
    - We quantize dwconv / pwconv1 / pwconv2.
    - We keep norm, GELU, DropPath, and residual add in float.
    - For weight-only BRECQ, act quant will normally stay off anyway.
    """
    def __init__(self, block: ConvNeXtBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.dwconv = QuantModule(
            block.dwconv,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
        )
        self.norm = block.norm
        self.pwconv1 = QuantModule(
            block.pwconv1,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
        )
        self.act = block.act
        self.pwconv2 = QuantModule(
            block.pwconv2,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
            disable_act_quant=True,
        )
        self.weight = block.weight
        self.drop_path = block.drop_path

    def forward(self, x, time):
        batch_size, sequence_length, hidden_size = x.shape
        input_dim = math.floor(sequence_length ** 0.5)

        residual = x
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)
        x = x.permute(0, 3, 1, 2)

        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x, time)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.weight is not None:
            x = self.weight * x

        x = x.reshape(batch_size, sequence_length, hidden_size)
        x = residual + self.drop_path(x)

        if self.use_act_quant:
            x = self.act_quantizer(x)
        return x


class QuantResNetBlock(BaseQuantBlock):
    """
    Quantized version of Poseidon ResNetBlock.

    Notes:
    - BN stays as original modules here.
    - If you want true inference-style BN folding later, that should be handled separately.
    - For now this keeps logic simple and structure faithful.
    """
    def __init__(self, block: ResNetBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.conv1 = QuantModule(
            block.conv1,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
        )
        self.conv2 = QuantModule(
            block.conv2,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
            disable_act_quant=True,
        )
        self.bn1 = block.bn1
        self.bn2 = block.bn2

    def forward(self, x, time):
        batch_size, sequence_length, hidden_size = x.shape
        input_dim = math.floor(sequence_length ** 0.5)

        residual = x
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, sequence_length, hidden_size)

        x = x + residual

        if self.use_act_quant:
            x = self.act_quantizer(x)
        return x


class QuantScOTLayer(BaseQuantBlock):
    """
    Quantized version of Poseidon ScOTLayer.

    Important:
    - This keeps the original block logic as intact as possible.
    - It does NOT quantize HuggingFace Swinv2Attention internals yet.
    - It DOES quantize the MLP part if those submodules are nn.Linear and are wrapped elsewhere,
      or if they are explicitly wrapped here.
    - Since Swinv2Attention / Swinv2Intermediate / Swinv2Output are HF modules, whether their inner
      Linear layers are quantized depends on whether your model-level recursive refactor goes inside them.

    So this class is mainly used as the BRECQ block boundary for block reconstruction.
    """
    def __init__(self, layer: ScOTLayer, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.chunk_size_feed_forward = layer.chunk_size_feed_forward
        self.shift_size = layer.shift_size
        self.window_size = layer.window_size
        self.input_resolution = layer.input_resolution

        # Keep the original HF submodules
        self.attention = layer.attention
        self.layernorm_before = layer.layernorm_before
        self.drop_path = layer.drop_path
        self.intermediate = layer.intermediate
        self.output = layer.output
        self.layernorm_after = layer.layernorm_after

        # Keep caches / helper state
        self.attn_mask_cache = layer.attn_mask_cache
        self.pad_cache = layer.pad_cache

    def set_shift_and_window_size(self, input_resolution):
        target_window_size = (
            self.window_size
            if isinstance(self.window_size, collections.abc.Iterable)
            else (self.window_size, self.window_size)
        )
        target_shift_size = (
            self.shift_size
            if isinstance(self.shift_size, collections.abc.Iterable)
            else (self.shift_size, self.shift_size)
        )
        window_dim = (
            input_resolution[0].item()
            if torch.is_tensor(input_resolution[0])
            else input_resolution[0]
        )
        self.window_size = (
            window_dim if window_dim <= target_window_size[0] else target_window_size[0]
        )
        self.shift_size = (
            0
            if input_resolution
            <= (
                self.window_size
                if isinstance(self.window_size, collections.abc.Iterable)
                else (self.window_size, self.window_size)
            )
            else target_shift_size[0]
        )

    def get_attn_mask(self, height, width, dtype):
        cache_key = (height, width, self.shift_size, self.window_size, dtype)
        if cache_key in self.attn_mask_cache:
            return self.attn_mask_cache[cache_key]

        from transformers.models.swinv2.modeling_swinv2 import window_partition

        if self.shift_size > 0:
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        self.attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        cache_key = (height, width, self.window_size)
        if cache_key in self.pad_cache:
            pad_values = self.pad_cache[cache_key]
            if pad_values[3] > 0 or pad_values[5] > 0:
                hidden_states = nn.functional.pad(hidden_states, pad_values)
            return hidden_states, pad_values

        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        self.pad_cache[cache_key] = pad_values

        if pad_right > 0 or pad_bottom > 0:
            hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions,
        time: torch.Tensor,
        head_mask=None,
        output_attentions: bool = False,
        always_partition: bool = False,
    ):
        from transformers.models.swinv2.modeling_swinv2 import window_partition, window_reverse

        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)

        height, width = input_dimensions
        batch_size, seq_len, channels = hidden_states.size()

        shortcut = hidden_states

        hidden_states = hidden_states.view(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape

        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(
                hidden_states,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_hidden_states = hidden_states

        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(
            -1, self.window_size * self.window_size, channels
        )

        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        attention_outputs = self.attention(
            hidden_states_windows,
            attn_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(
            -1, self.window_size, self.window_size, channels
        )
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )

        if self.shift_size > 0:
            attention_windows = torch.roll(
                shifted_windows,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(
            self.layernorm_before(attention_windows, time)
        )

        residual = hidden_states
        layer_output = self.output(self.intermediate(hidden_states))
        layer_output = residual + self.drop_path(
            self.layernorm_after(layer_output, time)
        )

        if self.use_act_quant:
            layer_output = self.act_quantizer(layer_output)

        if output_attentions:
            return (layer_output, attention_outputs[1])
        return (layer_output,)


specials = {
    ScOTLayer: QuantScOTLayer,
    ConvNeXtBlock: QuantConvNeXtBlock,
    ResNetBlock: QuantResNetBlock,
}