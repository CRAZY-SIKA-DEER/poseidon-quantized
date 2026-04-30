"""
Utilities for loading and interacting with the Poseidon (ScOT) model.

This module provides helper functions used by the PPQ pipeline to run
Poseidon inference and collect calibration data. The main functionalities include:

1. Model loading
   - `load_poseidon_model`: loads a pretrained Poseidon (ScOT) model and
     places it on the correct device for inference.

2. Dataset and dataloader construction
   - `build_poseidon_loaders`: builds training/validation datasets and
     PyTorch DataLoaders, and provides limited-step iterators used for
     calibration and evaluation.

3. Forward inference
   - `poseidon_forward`: performs a single forward pass of the Poseidon
     model using a batch from the dataset.

4. Layer-level clean output caching
   - `get_clean_outputs_poseidon`: runs the model on calibration data and
     caches clean inputs and outputs for selected Linear layers using
     forward hooks. These cached tensors are used for layer-wise
     quantization optimization.

5. Network-level clean output caching
   - `get_clean_network_outputs_poseidon`: runs the model on a fixed list
     of batches and stores the final network outputs. This is used for
     evaluating the full-model error after quantization updates.

These utilities are designed to support probabilistic post-training
quantization (PPQ) experiments on the Poseidon foundation model.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scOT.model import ScOT
from scOT.problems.base import get_dataset





def load_poseidon_model(model_path: str, device: str = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = ScOT.from_pretrained(model_path).to(device)
    model.eval()

    torch.set_float32_matmul_precision("high")

    print(f"Model loaded on device: {device}")
    #print(f"Model type: {type(model)}")

    return model, device


def build_poseidon_loaders(
    dataset_name: str,
    data_path: str,
    calib_batchsize: int = 8,
    calib_steps: int = 8,
    val_batchsize: int = 16,
    val_steps: int = 50,
):
    train_ds = get_dataset(
        dataset_name,
        which="train",
        num_trajectories=2048,
        data_path=data_path,
    )

    # try:
    #     val_ds = get_dataset(
    #         dataset_name,
    #         which="val",
    #         num_trajectories=256,
    #         data_path=data_path,
    #     )
    # except Exception:
    val_ds = get_dataset(
        dataset_name,
        which="test",
        num_trajectories=256,
        data_path=data_path,
    )

    calib_loader = DataLoader(
        train_ds,
        batch_size=calib_batchsize,
        shuffle=True,
        num_workers=min(os.cpu_count() or 0, 16),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batchsize,
        #shuffle=False,
        shuffle = True,
        num_workers=min(os.cpu_count() or 0, 16),
        pin_memory=True,
    )

    def take(loader, steps):
        for i, batch in enumerate(loader):
            if i >= steps:
                break
            yield batch

    calib_iter = lambda: take(calib_loader, calib_steps)
    val_iter = lambda: take(val_loader, val_steps)

    return calib_loader, val_loader, calib_iter, val_iter


def poseidon_forward(model, batch, device):
    x = batch["pixel_values"].to(device)
    t = batch["time"].to(device)
    pm = batch["pixel_mask"].to(device)
    y = batch.get("labels")

    out = model(
        pixel_values=x,
        time=t,
        pixel_mask=pm,
        labels=(y.to(device) if y is not None else None),
    )
    return out.output



def get_clean_outputs_poseidon(model, dataloader, device, layer_names):
    """
    Cache clean per-layer inputs and outputs for target Linear layers.

    Returns:
        clean_inputs:  dict[layer_name] -> list of X_pre tensors (CPU) or None
        clean_outputs: dict[layer_name] -> list of Y_post tensors (CPU) or None
    """
    model.eval()

    clean_inputs = {name: [] for name in layer_names}
    clean_outputs = {name: [] for name in layer_names}

    if callable(dataloader):
        dataloader = dataloader()

    with torch.inference_mode():
        for batch in dataloader:
            x = batch["pixel_values"].to(device)
            t = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y = batch.get("labels", None)

            if t is not None:
                t = t.to(device)
            if pm is not None:
                pm = pm.to(device)
            if y is not None:
                y = y.to(device)

            layer_io = {}

            def make_hook(name):
                def hook(mod, inp, out):
                    x_pre = inp[0].detach().cpu()
                    y_post = out.detach().cpu()
                    layer_io[name] = (x_pre, y_post)
                return hook

            handles = []
            for name, mod in model.named_modules():
                if name in layer_names and isinstance(mod, nn.Linear):
                    handles.append(mod.register_forward_hook(make_hook(name)))

            _ = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )

            for name in layer_names:
                if name in layer_io:
                    x_pre, y_post = layer_io[name]
                    clean_inputs[name].append(x_pre)
                    clean_outputs[name].append(y_post)
                else:
                    clean_inputs[name].append(None)
                    clean_outputs[name].append(None)

            for h in handles:
                h.remove()

    return clean_inputs, clean_outputs


def get_clean_network_outputs_poseidon(model, frozen_batches, device):
    """
    Cache clean final network outputs for a fixed list of batches.

    Returns:
        clean_net_outputs: list of final output tensors on CPU
    """
    model.eval()
    clean_net_outputs = []

    with torch.inference_mode():
        for batch in frozen_batches:
            x = batch["pixel_values"].to(device)
            t = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y = batch.get("labels", None)

            if t is not None:
                t = t.to(device)
            if pm is not None:
                pm = pm.to(device)
            if y is not None:
                y = y.to(device)

            outputs = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            clean_net_outputs.append(outputs.output.detach().cpu())

    return clean_net_outputs