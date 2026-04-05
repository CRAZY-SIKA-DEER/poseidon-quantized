import torch.nn as nn

from BRECQ.quant.quant_layer import QuantModule, StraightThrough
from BRECQ.quant.quant_block import BaseQuantBlock

# change these imports to your real paths if needed
from poseidon_quant_block import specials
from scOT.model import LayerNorm, ConditionalLayerNorm


class PoseidonQuantModel(nn.Module):
    """
    Quant-model wrapper for Poseidon/ScOT.

    Main idea:
    - recursively traverse the model
    - if a module type is in `specials`, replace it with the corresponding quant block
    - otherwise replace plain Conv2d / Linear with QuantModule
    - keep norm / activation / other non-weight modules as they are
    """

    def __init__(self, model: nn.Module, weight_quant_params=None, act_quant_params=None):
        super().__init__()
        weight_quant_params = {} if weight_quant_params is None else weight_quant_params
        act_quant_params = {} if act_quant_params is None else act_quant_params

        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(
        self,
        module: nn.Module,
        weight_quant_params = None,
        act_quant_params = None,
    ):
        weight_quant_params = {} if weight_quant_params is None else weight_quant_params
        act_quant_params = {} if act_quant_params is None else act_quant_params

        prev_quantmodule = None

        for name, child_module in module.named_children():

            # 1) replace Poseidon special blocks first
            if type(child_module) in specials:
                quant_block = specials[type(child_module)](
                    child_module,
                    weight_quant_params,
                    act_quant_params,
                )
                setattr(module, name, quant_block)

                # very important:
                # after replacing outer block, still recurse into it,
                # so inner HF attention / MLP Linear layers can also be wrapped
                self.quant_module_refactor(
                    getattr(module, name),
                    weight_quant_params,
                    act_quant_params,
                )
                prev_quantmodule = None

            # 2) replace plain Conv / Linear
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(
                    module,
                    name,
                    QuantModule(
                        child_module,
                        weight_quant_params,
                        act_quant_params,
                    ),
                )
                prev_quantmodule = getattr(module, name)

            # 3) absorb simple activations into previous QuantModule when possible
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    prev_quantmodule = None
                    continue

            # 4) keep these modules unchanged
            elif isinstance(
                child_module,
                (
                    StraightThrough,
                    nn.Identity,
                    nn.Dropout,
                    nn.BatchNorm2d,
                    nn.LayerNorm,
                    LayerNorm,
                    ConditionalLayerNorm,
                ),
            ):
                prev_quantmodule = None
                continue

            # 5) recurse into everything else
            else:
                self.quant_module_refactor(
                    child_module,
                    weight_quant_params,
                    act_quant_params,
                )
                prev_quantmodule = None

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)