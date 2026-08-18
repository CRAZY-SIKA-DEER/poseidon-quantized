"""
Poseidon-specific quantization-range utilities for SBPQ.

This module:

1. Computes per-output-channel weight ranges for target Linear layers.
2. Computes per-input-channel activation ranges from calibration data.
3. Saves and loads precomputed ranges.
4. Optionally evaluates fixed-bit fake quantization.
5. Optionally searches for the best clipping percentile.

The range calculation is model-specific because it depends on:

    - Poseidon's forward arguments
    - Poseidon's Linear-layer inputs
    - Poseidon's calibration dataloader
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from SBPQ.poseidon.poseidon_utils import (
    move_poseidon_batch_to_device,
)


def _prepare_loader(
    dataloader,
) -> Iterable:
    """
    Return an iterable over calibration batches.

    The supplied object may be:

        - a normal DataLoader
        - a list of frozen batches
        - a callable iterator such as calib_iterator
    """
    if callable(dataloader):
        return dataloader()

    return dataloader


def gaussian_tail_multiplier(
    percentile_prob: float,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Calculate the Gaussian clipping multiplier.

    For a selected tail probability p:

        z = sqrt(2) * erfinv(1 - 2p)

    The estimated clipping interval is then based on:

        mean ± z * standard_deviation
    """
    if not 0.0 < percentile_prob < 0.5:
        raise ValueError(
            "percentile_prob must lie strictly between 0 and 0.5."
        )

    device = torch.device(device)

    probability = torch.tensor(
        percentile_prob,
        device=device,
        dtype=dtype,
    )

    return (
        torch.sqrt(
            torch.tensor(
                2.0,
                device=device,
                dtype=dtype,
            )
        )
        * torch.special.erfinv(1.0 - 2.0 * probability)
    )


def gaussian_range_from_statistics(
    mean: torch.Tensor,
    std: torch.Tensor,
    multiplier: torch.Tensor,
    minimum_range: float = 1e-12,
) -> torch.Tensor:
    """
    Calculate a full quantization range from Gaussian statistics.

    Upper clipping value:

        beta = mean + z * std

    Lower clipping value:

        alpha = mean - z * std

    Full range:

        R = beta - alpha = 2 * z * std

    The mean cancels from the final full-range value.
    """
    upper_bound = mean + std * multiplier
    lower_bound = mean - std * multiplier

    return (
        upper_bound - lower_bound
    ).clamp_min(float(minimum_range))


def percentile_range_from_samples(
    samples_by_channel: torch.Tensor,
    percentile_prob: float,
    minimum_range: float = 1e-12,
) -> torch.Tensor:
    """
    Calculate per-channel percentile clipping ranges.

    samples_by_channel is shaped [channels, samples]. For each channel:

        lower = quantile(samples, percentile_prob)
        upper = quantile(samples, 1 - percentile_prob)
        range = upper - lower
    """
    if samples_by_channel.ndim != 2:
        raise ValueError(
            "samples_by_channel must have shape [channels, samples]."
        )

    if not 0.0 < percentile_prob < 0.5:
        raise ValueError(
            "percentile_prob must lie strictly between 0 and 0.5."
        )

    lower = torch.quantile(
        samples_by_channel,
        q=float(percentile_prob),
        dim=1,
    )
    upper = torch.quantile(
        samples_by_channel,
        q=1.0 - float(percentile_prob),
        dim=1,
    )

    return (upper - lower).clamp_min(float(minimum_range))


def compute_weight_ranges_poseidon(
    model: nn.Module,
    layer_names: list[str] | tuple[str, ...] | set[str],
    device: str | torch.device,
    percentile_prob: float = 1e-4,
    minimum_range: float = 1e-12,
    range_method: str = "percentile",
) -> OrderedDict[str, torch.Tensor]:
    """
    Compute one weight range per output channel for each target Linear layer.

    For a Linear weight tensor shaped:

        [out_features, in_features]

    the returned range has shape:

        [out_features]
    """
    device = torch.device(device)
    target_layer_names = set(layer_names)

    if range_method not in {"percentile", "gaussian"}:
        raise ValueError(
            f"Unsupported range_method '{range_method}'."
        )

    multiplier = None
    if range_method == "gaussian":
        multiplier = gaussian_tail_multiplier(
            percentile_prob=percentile_prob,
            device=device,
        )

    weight_ranges = OrderedDict()

    with torch.no_grad():
        for layer_name, module in model.named_modules():
            if (
                layer_name not in target_layer_names
                or not isinstance(module, nn.Linear)
            ):
                continue

            weight = module.weight.detach().to(device)

            weight_flat = weight.reshape(
                weight.shape[0],
                -1,
            )

            if range_method == "percentile":
                layer_range = percentile_range_from_samples(
                    samples_by_channel=weight_flat,
                    percentile_prob=percentile_prob,
                    minimum_range=minimum_range,
                )
            else:
                channel_mean = weight_flat.mean(dim=1)

                channel_std = weight_flat.std(
                    dim=1,
                    unbiased=False,
                )

                layer_range = gaussian_range_from_statistics(
                    mean=channel_mean,
                    std=channel_std,
                    multiplier=multiplier,
                    minimum_range=minimum_range,
                )

            weight_ranges[layer_name] = layer_range

    missing_layers = (
        target_layer_names - set(weight_ranges.keys())
    )

    if missing_layers:
        raise ValueError(
            "Some requested Linear layers were not found:\n"
            + "\n".join(sorted(missing_layers))
        )

    return weight_ranges


def _flatten_linear_input_by_channel(
    activation: torch.Tensor,
) -> torch.Tensor:
    """
    Rearrange a Linear-layer input into:

        [input_features, number_of_samples]

    A Linear layer always treats the final tensor dimension as its
    feature dimension.

    Examples:

        [B, C]
            -> [C, B]

        [B, sequence, C]
            -> [C, B * sequence]

        [B, H, W, C]
            -> [C, B * H * W]
    """
    if activation.ndim < 2:
        raise ValueError(
            "A Linear-layer input must contain at least two dimensions, "
            f"but received shape {activation.shape}."
        )

    input_features = activation.shape[-1]

    return (
        activation
        .reshape(-1, input_features)
        .transpose(0, 1)
        .contiguous()
    )


def compute_activation_ranges_poseidon(
    model: nn.Module,
    dataloader,
    layer_names: list[str] | tuple[str, ...] | set[str],
    device: str | torch.device,
    percentile_prob: float = 1e-4,
    minimum_range: float = 1e-12,
    range_method: str = "percentile",
) -> OrderedDict[str, torch.Tensor]:
    """
    Compute one activation range per input feature for target Linear layers.

    The statistics are collected from each layer's input during Poseidon
    forward passes.

    Returns:

        {
            layer_name: tensor [in_features]
        }
    """
    device = torch.device(device)
    target_layer_names = set(layer_names)

    if range_method not in {"percentile", "gaussian"}:
        raise ValueError(
            f"Unsupported range_method '{range_method}'."
        )

    multiplier = None
    if range_method == "gaussian":
        multiplier = gaussian_tail_multiplier(
            percentile_prob=percentile_prob,
            device=device,
        )

    activation_statistics = OrderedDict(
        (
            layer_name,
            {
                "means": [],
                "stds": [],
                "lowers": [],
                "uppers": [],
            },
        )
        for layer_name in layer_names
    )

    handles = []

    def make_activation_hook(layer_name: str):
        def hook(
            module: nn.Module,
            inputs,
            output,
        ) -> None:
            if len(inputs) == 0:
                raise RuntimeError(
                    f"Layer '{layer_name}' received no input tensor."
                )

            activation = inputs[0].detach()

            flattened = _flatten_linear_input_by_channel(
                activation
            )

            if range_method == "percentile":
                lower = torch.quantile(
                    flattened,
                    q=float(percentile_prob),
                    dim=1,
                )
                upper = torch.quantile(
                    flattened,
                    q=1.0 - float(percentile_prob),
                    dim=1,
                )

                activation_statistics[layer_name]["lowers"].append(
                    lower.cpu()
                )
                activation_statistics[layer_name]["uppers"].append(
                    upper.cpu()
                )
            else:
                channel_mean = flattened.mean(dim=1)

                channel_std = flattened.std(
                    dim=1,
                    unbiased=False,
                )

                activation_statistics[layer_name]["means"].append(
                    channel_mean.cpu()
                )

                activation_statistics[layer_name]["stds"].append(
                    channel_std.cpu()
                )

        return hook

    for layer_name, module in model.named_modules():
        if (
            layer_name in target_layer_names
            and isinstance(module, nn.Linear)
        ):
            handles.append(
                module.register_forward_hook(
                    make_activation_hook(layer_name)
                )
            )

    if len(handles) == 0:
        raise RuntimeError(
            "No target Linear layers received activation hooks."
        )

    model.eval()

    try:
        with torch.no_grad():
            for batch in _prepare_loader(dataloader):
                (
                    pixel_values,
                    time,
                    pixel_mask,
                    labels,
                ) = move_poseidon_batch_to_device(
                    batch=batch,
                    device=device,
                )

                model(
                    pixel_values=pixel_values,
                    time=time,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )

    finally:
        for handle in handles:
            handle.remove()

    activation_ranges = OrderedDict()

    for layer_name, statistics in activation_statistics.items():
        if range_method == "percentile":
            lowers = statistics["lowers"]
            uppers = statistics["uppers"]

            if len(lowers) == 0:
                raise RuntimeError(
                    f"No activation statistics were captured for "
                    f"layer '{layer_name}'."
                )

            average_lower = torch.stack(
                lowers,
                dim=0,
            ).mean(dim=0).to(device)
            average_upper = torch.stack(
                uppers,
                dim=0,
            ).mean(dim=0).to(device)

            activation_ranges[layer_name] = (
                average_upper - average_lower
            ).clamp_min(float(minimum_range))
        else:
            means = statistics["means"]
            stds = statistics["stds"]

            if len(means) == 0:
                raise RuntimeError(
                    f"No activation statistics were captured for "
                    f"layer '{layer_name}'."
                )

            average_mean = torch.stack(
                means,
                dim=0,
            ).mean(dim=0).to(device)

            average_std = torch.stack(
                stds,
                dim=0,
            ).mean(dim=0).to(device)

            activation_ranges[layer_name] = (
                gaussian_range_from_statistics(
                    mean=average_mean,
                    std=average_std,
                    multiplier=multiplier,
                    minimum_range=minimum_range,
                )
            )

    return activation_ranges


def compute_data_ranges_poseidon(
    model: nn.Module,
    dataloader,
    device: str | torch.device,
    layer_names: list[str] | tuple[str, ...] | set[str],
    percentile_prob: float = 1e-4,
    minimum_range: float = 1e-12,
    range_method: str = "percentile",
) -> OrderedDict[str, dict[str, torch.Tensor]]:
    """
    Compute both weight and activation ranges for target Linear layers.

    Returns:

        {
            layer_name: {
                "weight_ranges": tensor [out_features],
                "activation_ranges": tensor [in_features],
            }
        }
    """
    model.eval()

    weight_ranges = compute_weight_ranges_poseidon(
        model=model,
        layer_names=layer_names,
        device=device,
        percentile_prob=percentile_prob,
        minimum_range=minimum_range,
        range_method=range_method,
    )

    activation_ranges = compute_activation_ranges_poseidon(
        model=model,
        dataloader=dataloader,
        layer_names=layer_names,
        device=device,
        percentile_prob=percentile_prob,
        minimum_range=minimum_range,
        range_method=range_method,
    )

    ranges = OrderedDict()

    for layer_name in layer_names:
        if layer_name not in weight_ranges:
            raise KeyError(
                f"Weight range missing for layer '{layer_name}'."
            )

        if layer_name not in activation_ranges:
            raise KeyError(
                f"Activation range missing for layer '{layer_name}'."
            )

        ranges[layer_name] = {
            "weight_ranges": weight_ranges[layer_name],
            "activation_ranges": activation_ranges[layer_name],
        }

    return ranges


def save_poseidon_ranges(
    ranges: dict[str, dict[str, torch.Tensor]],
    save_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """
    Save precomputed Poseidon ranges to disk.
    """
    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_ranges = OrderedDict()

    for layer_name, layer_ranges in ranges.items():
        if "weight_ranges" not in layer_ranges:
            raise KeyError(
                f"Layer '{layer_name}' has no weight_ranges."
            )

        if "activation_ranges" not in layer_ranges:
            raise KeyError(
                f"Layer '{layer_name}' has no activation_ranges."
            )

        saved_ranges[layer_name] = {
            "weight_ranges": (
                layer_ranges["weight_ranges"]
                .detach()
                .cpu()
            ),
            "activation_ranges": (
                layer_ranges["activation_ranges"]
                .detach()
                .cpu()
            ),
        }

    torch.save(
        {
            "ranges_dict": saved_ranges,
            "metadata": metadata or {},
        },
        save_path,
    )

    print(
        f"[INFO] Saved Poseidon ranges for "
        f"{len(saved_ranges)} layers to: {save_path}"
    )


def load_poseidon_ranges(
    ranges_path: str | Path,
    device: str | torch.device = "cpu",
) -> OrderedDict[str, dict[str, torch.Tensor]]:
    """
    Load a saved Poseidon range file.
    """
    ranges_path = Path(ranges_path)

    if not ranges_path.exists():
        raise FileNotFoundError(
            f"Poseidon range file was not found: {ranges_path}"
        )

    saved_object = torch.load(
        ranges_path,
        map_location="cpu",
    )

    if "ranges_dict" not in saved_object:
        raise KeyError(
            f"Range file '{ranges_path}' does not contain "
            "'ranges_dict'."
        )

    device = torch.device(device)
    loaded_ranges = OrderedDict()

    for layer_name, layer_ranges in saved_object[
        "ranges_dict"
    ].items():
        loaded_ranges[layer_name] = {
            "weight_ranges": (
                layer_ranges["weight_ranges"].to(device)
            ),
            "activation_ranges": (
                layer_ranges["activation_ranges"].to(device)
            ),
        }

    print(
        f"[INFO] Loaded Poseidon ranges for "
        f"{len(loaded_ranges)} layers from: {ranges_path}"
    )

    return loaded_ranges


def get_poseidon_range_cache_path(
    repo_root: str | Path,
    model_path: str | Path,
    dataset_name: str,
    percentile_prob: float,
    range_method: str = "percentile",
) -> Path:
    """
    Construct a model- and dataset-specific range-cache path.

    Different fine-tuned models and datasets therefore receive different
    saved range files.
    """
    repo_root = Path(repo_root)
    model_name = Path(model_path).name

    safe_dataset_name = (
        dataset_name
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
    )

    percentile_tag = f"p{float(percentile_prob):.0e}"

    return (
        repo_root
        / "SBPQ"
        / "artifacts"
        / "poseidon"
        / model_name
        / safe_dataset_name
        / "ranges"
        / str(range_method)
        / percentile_tag
        / "ranges.pt"
    )


def load_precalculated_ranges_if_exists(
    repo_root: str | Path,
    model_path: str | Path,
    dataset_name: str,
    percentile_prob: float,
    range_method: str = "percentile",
    device: str | torch.device = "cpu",
) -> OrderedDict[str, dict[str, torch.Tensor]] | None:
    """
    Load model- and dataset-specific cached ranges when available.

    Returns None when the cache file does not exist.
    """
    ranges_path = get_poseidon_range_cache_path(
        repo_root=repo_root,
        model_path=model_path,
        dataset_name=dataset_name,
        percentile_prob=percentile_prob,
        range_method=range_method,
    )

    if not ranges_path.exists():
        print(
            f"[INFO] Precalculated Poseidon ranges not found: "
            f"{ranges_path}"
        )
        return None

    return load_poseidon_ranges(
        ranges_path=ranges_path,
        device=device,
    )


def fake_quantize_tensor(
    tensor: torch.Tensor,
    step_size: torch.Tensor,
) -> torch.Tensor:
    """
    Apply simple symmetric fake quantization.

    The straight-through estimator is not needed here because this function
    is only used for evaluation.
    """
    safe_step_size = step_size.clamp_min(1e-12)

    return (
        torch.round(tensor / safe_step_size)
        * safe_step_size
    )


def evaluate_quantized_model_poseidon(
    model: nn.Module,
    dataloader,
    ranges: dict[str, dict[str, torch.Tensor]],
    device: str | torch.device,
    layer_names: list[str] | tuple[str, ...] | set[str],
    num_bits: int = 8,
) -> float:
    """
    Evaluate fixed-bit fake quantization using the precomputed ranges.

    This is only a diagnostic utility for percentile selection.
    It is not the final SBPQ likelihood.
    """
    if num_bits < 1:
        raise ValueError(
            "num_bits must be positive."
        )

    device = torch.device(device)
    target_layer_names = set(layer_names)

    number_of_levels = float(2**num_bits - 1)

    total_loss = 0.0
    number_of_batches = 0

    model.eval()

    with torch.no_grad():
        for batch in _prepare_loader(dataloader):
            (
                pixel_values,
                time,
                pixel_mask,
                labels,
            ) = move_poseidon_batch_to_device(
                batch=batch,
                device=device,
            )

            if labels is None:
                raise ValueError(
                    "Fake-quant evaluation requires batch labels."
                )

            handles = []

            def make_quantization_hook(
                weight_ranges: torch.Tensor,
                activation_ranges: torch.Tensor,
            ):
                def hook(
                    module: nn.Linear,
                    inputs,
                    output,
                ):
                    activation = inputs[0]

                    activation_ranges_device = activation_ranges.to(
                        device=activation.device,
                        dtype=activation.dtype,
                    )

                    activation_shape = (
                        [1] * (activation.ndim - 1)
                        + [activation.shape[-1]]
                    )

                    activation_step = (
                        activation_ranges_device.reshape(
                            activation_shape
                        )
                        / number_of_levels
                    )

                    quantized_activation = fake_quantize_tensor(
                        tensor=activation,
                        step_size=activation_step,
                    )

                    weight_ranges_device = weight_ranges.to(
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )

                    weight_step = (
                        weight_ranges_device.reshape(-1, 1)
                        / number_of_levels
                    )

                    quantized_weight = fake_quantize_tensor(
                        tensor=module.weight,
                        step_size=weight_step,
                    )

                    return F.linear(
                        quantized_activation,
                        quantized_weight,
                        module.bias,
                    )

                return hook

            for layer_name, module in model.named_modules():
                if (
                    layer_name in target_layer_names
                    and isinstance(module, nn.Linear)
                    and layer_name in ranges
                ):
                    handles.append(
                        module.register_forward_hook(
                            make_quantization_hook(
                                weight_ranges=ranges[layer_name][
                                    "weight_ranges"
                                ],
                                activation_ranges=ranges[layer_name][
                                    "activation_ranges"
                                ],
                            )
                        )
                    )

            try:
                outputs = model(
                    pixel_values=pixel_values,
                    time=time,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )

                batch_loss = F.mse_loss(
                    outputs.output,
                    labels,
                )

                total_loss += float(batch_loss.item())
                number_of_batches += 1

            finally:
                for handle in handles:
                    handle.remove()

    if number_of_batches == 0:
        return float("inf")

    return total_loss / number_of_batches


def find_best_percentile_poseidon(
    model: nn.Module,
    dataloader_factory,
    device: str | torch.device,
    layer_names: list[str] | tuple[str, ...] | set[str],
    percentile_candidates: tuple[float, ...] = (
        1e-2,
        1e-3,
        1e-4,
        1e-5,
        1e-6,
    ),
    num_bits: int = 8,
) -> tuple[float, float]:
    """
    Search for the percentile producing the lowest fake-quantization loss.

    dataloader_factory should be callable because the calibration data must
    be iterated once for range calculation and again for evaluation.

    Returns:

        best_percentile, best_loss
    """
    if not callable(dataloader_factory):
        raise TypeError(
            "dataloader_factory must be callable so it can create "
            "a fresh iterator for each percentile candidate."
        )

    best_percentile = None
    best_loss = float("inf")

    for percentile in percentile_candidates:
        print(
            f"[INFO] Testing clipping percentile: {percentile:.0e}"
        )

        ranges = compute_data_ranges_poseidon(
            model=model,
            dataloader=dataloader_factory,
            device=device,
            layer_names=layer_names,
            percentile_prob=percentile,
        )

        evaluation_loss = evaluate_quantized_model_poseidon(
            model=model,
            dataloader=dataloader_factory,
            ranges=ranges,
            device=device,
            layer_names=layer_names,
            num_bits=num_bits,
        )

        print(
            f"    fake-quantization loss = "
            f"{evaluation_loss:.6e}"
        )

        if evaluation_loss < best_loss:
            best_loss = evaluation_loss
            best_percentile = percentile

    if best_percentile is None:
        raise RuntimeError(
            "No percentile candidate was evaluated."
        )

    return best_percentile, best_loss
