import copy
import torch
from typing import Any, Dict, List, Tuple, Union

from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.quant_block import BaseQuantBlock


class StopForwardException(Exception):
    """
    Used to stop the full forward once the target block has been reached.
    """
    pass


def move_to_device(obj: Any, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        out = [move_to_device(v, device) for v in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    else:
        return obj


def detach_to_cpu(obj: Any):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: detach_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_to_cpu(v) for v in obj)
    else:
        return obj


def clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def slice_batch(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """
    Slice a collated Poseidon batch dict along batch dimension.
    Non-tensor entries are copied as-is.
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[start:end]
        else:
            out[k] = copy.deepcopy(v)
    return out


def expand_cali_data(cali_data: Union[List[Dict[str, Any]], Dict[str, Any]], batch_size: int):
    """
    Normalize calibration data into a list of mini-batch dicts.

    Supported:
    1) cali_data is already a list of batch dicts
    2) cali_data is one big collated batch dict, which we slice by batch_size
    """
    if isinstance(cali_data, list):
        return cali_data

    if isinstance(cali_data, dict):
        first_tensor = None
        for v in cali_data.values():
            if torch.is_tensor(v):
                first_tensor = v
                break
        if first_tensor is None:
            raise ValueError("cali_data dict must contain at least one tensor.")
        n = first_tensor.shape[0]
        batches = []
        for i in range(0, n, batch_size):
            batches.append(slice_batch(cali_data, i, min(i + batch_size, n)))
        return batches

    raise TypeError("cali_data must be either a list of batch dicts or one collated batch dict.")


class DataSaverHook:
    """
    Forward hook that stores block input args and block output.
    For Poseidon ScOTLayer, the hook input is a tuple like:
        (hidden_states, input_dimensions, time, head_mask, output_attentions, always_partition)
    """
    def __init__(self, store_input: bool = False, store_output: bool = False, stop_forward: bool = False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch

        if self.store_output:
            # ScOTLayer returns (layer_output,) or (layer_output, attn)
            if isinstance(output_batch, tuple):
                self.output_store = output_batch[0]
            else:
                self.output_store = output_batch

        if self.stop_forward:
            raise StopForwardException


class PoseidonGetLayerInpOut:
    """
    Poseidon-specific version of GetLayerInpOut.

    Main difference from original BRECQ:
    - model input is a batch dict, not a single tensor
    - block input is a tuple of multiple arguments, not a single tensor
    """
    def __init__(
        self,
        model,
        layer: Union[QuantModule, BaseQuantBlock],
        device: torch.device,
        asym: bool = False,
        act_quant: bool = False,
    ):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, model_input: Dict[str, Any]):
        self.model.eval()
        self.model.set_quant_state(False, False)

        batch_on_device = move_to_device(model_input, self.device)

        handle = self.layer.register_forward_hook(self.data_saver)

        with torch.no_grad():
            try:
                _ = self.model(**batch_on_device)
            except StopForwardException:
                pass

            if self.asym:
                # recompute block input using quantized prefix, but still use FP target output
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    _ = self.model(**batch_on_device)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        # input_store is the full argument tuple passed to the block
        # output_store is the tensor target for reconstruction
        block_args = detach_to_cpu(self.data_saver.input_store)
        block_out = detach_to_cpu(self.data_saver.output_store)

        return block_args, block_out


def save_inp_oup_data(
    model,
    layer: Union[QuantModule, BaseQuantBlock],
    cali_data: Union[List[Dict[str, Any]], Dict[str, Any]],
    asym: bool = False,
    act_quant: bool = False,
    batch_size: int = 32,
):
    """
    Poseidon-specific cached IO saver.

    Returns:
        cached_inputs: list of block input tuples
        cached_outputs: list of output tensors

    This is intentionally different from original BRECQ, because Poseidon blocks
    need multiple forward arguments, not just one tensor.
    """
    device = next(model.parameters()).device
    get_inp_out = PoseidonGetLayerInpOut(
        model=model,
        layer=layer,
        device=device,
        asym=asym,
        act_quant=act_quant,
    )

    cached_inputs = []
    cached_outputs = []

    torch.cuda.empty_cache()
    cali_batches = expand_cali_data(cali_data, batch_size=batch_size)

    for batch in cali_batches:
        cur_inp, cur_out = get_inp_out(batch)
        cached_inputs.append(cur_inp)
        cached_outputs.append(cur_out)

    torch.cuda.empty_cache()
    return cached_inputs, cached_outputs


def move_block_args_to_device(block_args: Tuple[Any, ...], device: torch.device):
    """
    Move cached block input tuple back to GPU before reconstruction forward.
    """
    return tuple(move_to_device(x, device) for x in block_args)


def get_reconstruction_output(block_output):
    """
    Normalize block forward output to the main tensor output.
    """
    if isinstance(block_output, tuple):
        return block_output[0]
    return block_output