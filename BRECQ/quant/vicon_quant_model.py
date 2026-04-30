import torch.nn as nn

from BRECQ.quant.quant_layer import QuantModule, StraightThrough
from BRECQ.quant.quant_block import BaseQuantBlock

from BRECQ.quant.vicon_quant_block import QuantTransformerEncoderLayer, QuantMultiheadAttention


class VICONQuantModel(nn.Module):
    """
    Quant model for VICON.
    """

    def __init__(self, model: nn.Module,
                 weight_quant_params=None,
                 act_quant_params=None):
        super().__init__()

        weight_quant_params = {} if weight_quant_params is None else weight_quant_params
        act_quant_params = {} if act_quant_params is None else act_quant_params

        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module, wq_params, aq_params):
        prev_quant = None

        for name, child in module.named_children():

            # ---- Replace Transformer block ----
            if isinstance(child, nn.TransformerEncoderLayer):
                setattr(
                    module,
                    name,
                    QuantTransformerEncoderLayer(child, wq_params, aq_params)
                )
                prev_quant = None

            # ---- Replace Linear ----
            elif isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    QuantModule(child, wq_params, aq_params)
                )
                prev_quant = getattr(module, name)

            # ---- Activation absorb ----
            elif isinstance(child, (nn.ReLU, nn.GELU)):
                if prev_quant is not None:
                    prev_quant.activation_function = child
                    setattr(module, name, StraightThrough())
                else:
                    prev_quant = None

            # ---- Skip ----
            elif isinstance(child, (nn.Dropout, nn.LayerNorm)):
                prev_quant = None
                continue

            # ---- Recurse ----
            else:
                self.quant_module_refactor(child, wq_params, aq_params)
                prev_quant = None

    def set_quant_state(self, weight_quant=False, act_quant=False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock, QuantMultiheadAttention)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)