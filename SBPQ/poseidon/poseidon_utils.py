
"""
Poseidon-specific utilities for SBPQ.

This module handles:
1. Loading the pretrained Poseidon model.
2. Building calibration and validation dataloaders.
3. Running Poseidon forward passes.
4. Caching clean layer-level outputs.
5. Caching clean network-level outputs.
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scOT.model import ScOT
from scOT.problems.base import get_dataset


def load_poseidon_model(
    model_path: str,
    device: str = "cuda",
):
    """
    Load a pretrained Poseidon model and place it on the selected device.

    Returns:
        model: loaded Poseidon model
        device: actual torch device being used
    """
    device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )

    model = ScOT.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    torch.set_float32_matmul_precision("high")

    print(f"[INFO] Poseidon model loaded on: {device}")

    return model, device


def build_poseidon_loaders(
    dataset_name: str,
    data_path: str,
    calib_batch_size: int = 8,
    calib_steps: int = 8,
    val_batch_size: int = 16,
    val_steps: int = 50,
    num_workers: int | None = None,
):
    """
    Build calibration and validation dataloaders.

    Also returns callable iterators that only yield the requested
    number of batches.
    """
    train_dataset = get_dataset(
        dataset_name,
        which="train",
        num_trajectories=2048,
        data_path=data_path,
    )

    try:
        val_dataset = get_dataset(
            dataset_name,
            which="val",
            num_trajectories=256,
            data_path=data_path,
        )
    except Exception:
        val_dataset = get_dataset(
            dataset_name,
            which="test",
            num_trajectories=256,
            data_path=data_path,
        )

    if num_workers is None:
        num_workers = min(os.cpu_count() or 0, 16)

    calib_loader = DataLoader(
        train_dataset,
        batch_size=calib_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    def take(loader, steps):
        for batch_index, batch in enumerate(loader):
            if batch_index >= steps:
                break
            yield batch

    def calib_iterator():
        return take(calib_loader, calib_steps)

    def val_iterator():
        return take(val_loader, val_steps)

    return (
        calib_loader,
        val_loader,
        calib_iterator,
        val_iterator,
    )


def move_poseidon_batch_to_device(batch, device):
    """
    Move the Poseidon batch fields to the selected device.

    pixel_values is required.
    time, pixel_mask, and labels are passed to the model when available.
    """
    pixel_values = batch["pixel_values"].to(device)

    time = batch.get("time", None)
    pixel_mask = batch.get("pixel_mask", None)
    labels = batch.get("labels", None)

    if time is not None:
        time = time.to(device)

    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)

    if labels is not None:
        labels = labels.to(device)

    return pixel_values, time, pixel_mask, labels


def poseidon_forward(model, batch, device):
    """
    Run one Poseidon forward pass and return the predicted field.
    """
    pixel_values, time, pixel_mask, labels = (
        move_poseidon_batch_to_device(batch, device)
    )

    outputs = model(
        pixel_values=pixel_values,
        time=time,
        pixel_mask=pixel_mask,
        labels=labels,
    )

    return outputs.output


def get_clean_outputs_poseidon(
    model,
    dataloader,
    device,
    layer_names,
):
    """
    Cache clean inputs and outputs for selected Linear layers.

    Returns:
        clean_inputs:
            dict[layer_name] -> list of clean input tensors

        clean_outputs:
            dict[layer_name] -> list of clean output tensors
    """
    model.eval()

    clean_inputs = {
        name: []
        for name in layer_names
    }

    clean_outputs = {
        name: []
        for name in layer_names
    }

    loader = dataloader() if callable(dataloader) else dataloader

    with torch.inference_mode():
        for batch in loader:
            pixel_values, time, pixel_mask, labels = (
                move_poseidon_batch_to_device(batch, device)
            )

            layer_io = {}

            def make_hook(layer_name):
                def hook(module, inputs, output):
                    clean_input = inputs[0].detach().cpu()
                    clean_output = output.detach().cpu()

                    layer_io[layer_name] = (
                        clean_input,
                        clean_output,
                    )

                return hook

            handles = []

            for name, module in model.named_modules():
                if (
                    name in layer_names
                    and isinstance(module, nn.Linear)
                ):
                    handle = module.register_forward_hook(
                        make_hook(name)
                    )
                    handles.append(handle)

            model(
                pixel_values=pixel_values,
                time=time,
                pixel_mask=pixel_mask,
                labels=labels,
            )

            for name in layer_names:
                if name in layer_io:
                    layer_input, layer_output = layer_io[name]

                    clean_inputs[name].append(layer_input)
                    clean_outputs[name].append(layer_output)
                else:
                    clean_inputs[name].append(None)
                    clean_outputs[name].append(None)

            for handle in handles:
                handle.remove()

    return clean_inputs, clean_outputs


def get_clean_network_outputs_poseidon(
    model,
    frozen_batches,
    device,
):
    """
    Cache the clean final network output for every frozen batch.

    These outputs will later serve as the targets of the
    network-wise Monte Carlo likelihood.
    """
    model.eval()
    clean_network_outputs = []

    with torch.inference_mode():
        for batch in frozen_batches:
            pixel_values, time, pixel_mask, labels = (
                move_poseidon_batch_to_device(batch, device)
            )

            outputs = model(
                pixel_values=pixel_values,
                time=time,
                pixel_mask=pixel_mask,
                labels=labels,
            )

            clean_network_outputs.append(
                outputs.output.detach().cpu()
            )

    return clean_network_outputs
