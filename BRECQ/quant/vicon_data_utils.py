import torch
import torch.nn.functional as F

from BRECQ.quant.quant_layer import QuantModule, Union
from BRECQ.quant.quant_block import BaseQuantBlock


class StopForwardException(Exception):
    pass


class DataSaverHook:
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


def forward_model(model, model_input, device):
    if isinstance(model_input, (list, tuple)):
        model_input = [x.to(device) for x in model_input]
        return model(model_input)
    else:
        return model(model_input.to(device))


class GetLayerInpOut:
    def __init__(self, model, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, model_input):
        self.model.eval()
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)

        with torch.no_grad():
            try:
                _ = forward_model(self.model, model_input, self.device)
            except StopForwardException:
                pass

            if self.asym:
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)

                try:
                    _ = forward_model(self.model, model_input, self.device)
                except StopForwardException:
                    pass

                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        return (
            self.data_saver.input_store[0].detach(),
            self.data_saver.output_store.detach()
        )


class GradSaverHook:
    def __init__(self):
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        self.grad_out = grad_output[0]


class GetLayerGrad:
    def __init__(self, model, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook()

    def __call__(self, model_input):
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)

        with torch.enable_grad():
            try:
                self.model.zero_grad()

                inputs = model_input
                out_fp = forward_model(self.model, inputs, self.device)

                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = forward_model(self.model, inputs, self.device)

                loss = F.mse_loss(out_q, out_fp.detach())
                loss.backward()

            except StopForwardException:
                pass

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        return self.data_saver.grad_out.data


def save_inp_oup_data(model, layer, cali_data,
                      asym=False, act_quant=False,
                      batch_size=32, keep_gpu=True):
    device = next(model.parameters()).device

    get_inp_out = GetLayerInpOut(
        model, layer,
        device=device,
        asym=asym,
        act_quant=act_quant
    )

    cached_batches = []

    for i in range(len(cali_data)):
        cur_inp, cur_out = get_inp_out(cali_data[i])
        cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])

    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)

    return cached_inps, cached_outs


def save_grad_data(model, layer, cali_data,
                   act_quant=False, batch_size=32,
                   keep_gpu=True):
    device = next(model.parameters()).device

    get_grad = GetLayerGrad(
        model, layer,
        device=device,
        act_quant=act_quant
    )

    cached_batches = []

    for i in range(len(cali_data)):
        cur_grad = get_grad(cali_data[i])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0

    if keep_gpu:
        cached_grads = cached_grads.to(device)

    return cached_grads


def quantize_model_till(model, layer, act_quant=False):
    model.set_quant_state(False, False)

    for module in model.modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)

        if module == layer:
            break