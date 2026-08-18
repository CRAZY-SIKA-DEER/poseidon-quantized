"""
Generic block-wise Beta prior for SBPQ.

This module does not calculate sensitivity or build Beta parameters.
Those parameters are calculated beforehand and saved by:

    SBPQ/poseidon/beta_parameter_builder.py

During optimization, this module:

    learnable step size S
        ↓
    effective bitwidth B = log2(R / S)
        ↓
    normalized bitwidth u in (0, 1)
        ↓
    identify the block containing each quantized layer
        ↓
    load that block's alpha and beta
        ↓
    calculate the Beta negative log-prior
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn


def load_beta_parameter_file(
    parameter_path: str | Path,
) -> dict:
    """
    Load precomputed block-wise Beta parameters.

    Expected file structure:

        {
            "block_beta_parameters": {
                block_name: {
                    "alpha": ...,
                    "beta": ...,
                    ...
                }
            },
            "metadata": {...}
        }
    """
    parameter_path = Path(parameter_path)

    if not parameter_path.exists():
        raise FileNotFoundError(
            f"Beta-parameter file was not found: {parameter_path}"
        )

    saved_object = torch.load(
        parameter_path,
        map_location="cpu",
    )

    if "block_beta_parameters" not in saved_object:
        raise KeyError(
            "The Beta-parameter file does not contain "
            "'block_beta_parameters'."
        )

    block_parameters = saved_object["block_beta_parameters"]

    if not isinstance(block_parameters, Mapping):
        raise TypeError(
            "'block_beta_parameters' must be a mapping."
        )

    if len(block_parameters) == 0:
        raise ValueError(
            "The loaded Beta-parameter dictionary is empty."
        )

    return saved_object


def extract_block_shapes(
    saved_beta_object: Mapping,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> OrderedDict[str, dict[str, torch.Tensor]]:
    """
    Extract alpha and beta for every block and move them to a device.

    Returns:

        {
            block_name: {
                "alpha": scalar tensor,
                "beta": scalar tensor,
            }
        }
    """
    device = torch.device(device)

    if "block_beta_parameters" not in saved_beta_object:
        raise KeyError(
            "saved_beta_object does not contain "
            "'block_beta_parameters'."
        )

    block_shapes = OrderedDict()

    for block_name, parameters in saved_beta_object[
        "block_beta_parameters"
    ].items():
        if "alpha" not in parameters or "beta" not in parameters:
            raise KeyError(
                f"Block '{block_name}' does not contain both "
                "'alpha' and 'beta'."
            )

        alpha = torch.as_tensor(
            parameters["alpha"],
            device=device,
            dtype=dtype,
        ).reshape(())

        beta = torch.as_tensor(
            parameters["beta"],
            device=device,
            dtype=dtype,
        ).reshape(())

        if not torch.isfinite(alpha) or not torch.isfinite(beta):
            raise ValueError(
                f"Block '{block_name}' contains non-finite "
                "Beta parameters."
            )

        if alpha <= 0 or beta <= 0:
            raise ValueError(
                f"Block '{block_name}' has invalid Beta parameters: "
                f"alpha={alpha.item()}, beta={beta.item()}."
            )

        block_shapes[block_name] = {
            "alpha": alpha,
            "beta": beta,
        }

    return block_shapes


def effective_bitwidth(
    quantization_range: torch.Tensor,
    step_size: torch.Tensor,
    minimum_step_size: float = 1e-12,
) -> torch.Tensor:
    """
    Convert a quantization step size into continuous effective bitwidth.

    Formula:

        B = log2(R / S)

    Args:
        quantization_range:
            Quantization range R. It may be one scalar or one value
            per channel.

        step_size:
            Learnable quantization step size S.

        minimum_step_size:
            Lower numerical bound used to prevent division by zero.

    Returns:
        Effective bitwidth tensor with the broadcasted shape of R and S.
    """
    quantization_range = torch.as_tensor(
        quantization_range,
        device=step_size.device,
        dtype=step_size.dtype,
    )

    if torch.any(quantization_range <= 0):
        raise ValueError(
            "All quantization-range values must be positive."
        )

    safe_step_size = step_size.clamp_min(
        float(minimum_step_size)
    )

    return torch.log2(
        quantization_range / safe_step_size
    )


def normalize_bitwidth(
    bitwidth: torch.Tensor,
    minimum_bits: float,
    maximum_bits: float,
    boundary_epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Map effective bitwidth B into normalized Beta space u.

    Formula:

        u = (B - B_min) / (B_max - B_min)

    The result is clamped into:

        [boundary_epsilon, 1 - boundary_epsilon]

    because log(0) is undefined in the Beta log-density.
    """
    if minimum_bits >= maximum_bits:
        raise ValueError(
            "minimum_bits must be smaller than maximum_bits."
        )

    if not 0.0 < boundary_epsilon < 0.5:
        raise ValueError(
            "boundary_epsilon must lie between 0 and 0.5."
        )

    normalized_bitwidth = (
        bitwidth - float(minimum_bits)
    ) / float(maximum_bits - minimum_bits)

    return normalized_bitwidth.clamp(
        min=float(boundary_epsilon),
        max=1.0 - float(boundary_epsilon),
    )


def beta_negative_log_density(
    normalized_bitwidth: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    include_normalization_constant: bool = True,
) -> torch.Tensor:
    """
    Calculate element-wise negative Beta log-density.

    For u in (0, 1):

        -log p(u)
        =
        -(alpha - 1) log(u)
        -(beta - 1) log(1-u)
        + log Beta(alpha, beta)

    Returns a tensor with the same shape as normalized_bitwidth.
    """
    if torch.any(normalized_bitwidth <= 0):
        raise ValueError(
            "normalized_bitwidth must be strictly larger than zero."
        )

    if torch.any(normalized_bitwidth >= 1):
        raise ValueError(
            "normalized_bitwidth must be strictly smaller than one."
        )

    alpha = torch.as_tensor(
        alpha,
        device=normalized_bitwidth.device,
        dtype=normalized_bitwidth.dtype,
    )

    beta = torch.as_tensor(
        beta,
        device=normalized_bitwidth.device,
        dtype=normalized_bitwidth.dtype,
    )

    negative_log_density = (
        -(alpha - 1.0) * torch.log(normalized_bitwidth)
        -(beta - 1.0) * torch.log1p(-normalized_bitwidth)
    )

    if include_normalization_constant:
        log_beta_function = (
            torch.lgamma(alpha)
            + torch.lgamma(beta)
            - torch.lgamma(alpha + beta)
        )

        negative_log_density = (
            negative_log_density + log_beta_function
        )

    return negative_log_density


def reduce_prior_values(
    prior_values: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """
    Reduce element-wise prior values into one scalar.

    Supported reductions:

        "sum":
            Sum all channel-wise prior values.

        "mean":
            Average all channel-wise prior values.

        "none":
            Return values without reduction.
    """
    if reduction == "sum":
        return prior_values.sum()

    if reduction == "mean":
        return prior_values.mean()

    if reduction == "none":
        return prior_values

    raise ValueError(
        f"Unsupported reduction '{reduction}'. "
        "Use 'sum', 'mean', or 'none'."
    )


def compute_layer_beta_prior(
    step_size: torch.Tensor,
    quantization_range: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    minimum_bits: float,
    maximum_bits: float,
    boundary_epsilon: float = 1e-6,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Compute the Beta prior for one quantized layer.

    The layer may contain one step size per output channel.

    Pipeline:

        S
        -> B = log2(R / S)
        -> u = (B - B_min) / (B_max - B_min)
        -> -log Beta(u | alpha, beta)
    """
    bitwidth = effective_bitwidth(
        quantization_range=quantization_range,
        step_size=step_size,
    )

    normalized_bitwidth = normalize_bitwidth(
        bitwidth=bitwidth,
        minimum_bits=minimum_bits,
        maximum_bits=maximum_bits,
        boundary_epsilon=boundary_epsilon,
    )

    prior_values = beta_negative_log_density(
        normalized_bitwidth=normalized_bitwidth,
        alpha=alpha,
        beta=beta,
    )

    return reduce_prior_values(
        prior_values=prior_values,
        reduction=reduction,
    )


def compute_network_beta_prior(
    step_sizes: Mapping[str, torch.Tensor],
    quantization_ranges: Mapping[str, torch.Tensor],
    layer_to_block: Mapping[str, str],
    block_shapes: Mapping[
        str,
        Mapping[str, torch.Tensor],
    ],
    minimum_bits: float,
    maximum_bits: float,
    prior_scale: float = 1.0,
    boundary_epsilon: float = 1e-6,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Compute the complete block-wise Beta prior over all quantized layers.

    Args:
        step_sizes:
            Mapping from layer name to learnable step-size tensor:

                {
                    layer_name: step_size
                }

        quantization_ranges:
            Mapping from layer name to quantization range:

                {
                    layer_name: range
                }

        layer_to_block:
            Mapping generated by Poseidon blocks.py:

                {
                    layer_name: block_name
                }

        block_shapes:
            Precomputed block-wise Beta shapes:

                {
                    block_name: {
                        "alpha": alpha_b,
                        "beta": beta_b,
                    }
                }

        minimum_bits:
            Lowest allowed effective bitwidth.

        maximum_bits:
            Highest allowed effective bitwidth.

        prior_scale:
            Global coefficient multiplying the complete Beta prior.

        reduction:
            "sum" counts every step-size/channel contribution.
            "mean" averages contributions across all step sizes.

    Returns:
        One scalar Beta-prior loss.
    """
    if len(step_sizes) == 0:
        raise ValueError(
            "The step-size mapping is empty."
        )

    if prior_scale < 0:
        raise ValueError(
            "prior_scale must be non-negative."
        )

    all_prior_values = []

    for layer_name, step_size in step_sizes.items():
        if layer_name not in quantization_ranges:
            raise KeyError(
                f"No quantization range was found for "
                f"layer '{layer_name}'."
            )

        if layer_name not in layer_to_block:
            # Some layers may be outside the detected structural blocks.
            # Those layers are not given a block-wise Beta prior.
            continue

        block_name = layer_to_block[layer_name]

        if block_name not in block_shapes:
            raise KeyError(
                f"No Beta parameters were found for block "
                f"'{block_name}', used by layer '{layer_name}'."
            )

        alpha = block_shapes[block_name]["alpha"]
        beta = block_shapes[block_name]["beta"]

        bitwidth = effective_bitwidth(
            quantization_range=quantization_ranges[layer_name],
            step_size=step_size,
        )

        normalized_bitwidth = normalize_bitwidth(
            bitwidth=bitwidth,
            minimum_bits=minimum_bits,
            maximum_bits=maximum_bits,
            boundary_epsilon=boundary_epsilon,
        )

        layer_prior_values = beta_negative_log_density(
            normalized_bitwidth=normalized_bitwidth,
            alpha=alpha,
            beta=beta,
        )

        all_prior_values.append(
            layer_prior_values.reshape(-1)
        )

    if len(all_prior_values) == 0:
        raise RuntimeError(
            "No layers received a block-wise Beta prior. "
            "Check layer_to_block and the layer names."
        )

    complete_prior_values = torch.cat(
        all_prior_values,
        dim=0,
    )

    reduced_prior = reduce_prior_values(
        prior_values=complete_prior_values,
        reduction=reduction,
    )

    return float(prior_scale) * reduced_prior


class BlockwiseBetaPrior(nn.Module):
    """
    Reusable PyTorch module for the SBPQ block-wise Beta prior.

    The Beta parameters are fixed. Gradients flow only through the
    learnable step sizes.
    """

    def __init__(
        self,
        beta_parameter_path: str | Path,
        layer_to_block: Mapping[str, str],
        minimum_bits: float,
        maximum_bits: float,
        prior_scale: float = 1.0,
        boundary_epsilon: float = 1e-6,
        reduction: str = "sum",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        if reduction not in {"sum", "mean"}:
            raise ValueError(
                "BlockwiseBetaPrior reduction must be "
                "'sum' or 'mean'."
            )

        saved_object = load_beta_parameter_file(
            parameter_path=beta_parameter_path,
        )

        block_shapes = extract_block_shapes(
            saved_beta_object=saved_object,
            device=device,
        )

        self.layer_to_block = dict(layer_to_block)

        self.minimum_bits = float(minimum_bits)
        self.maximum_bits = float(maximum_bits)
        self.prior_scale = float(prior_scale)
        self.boundary_epsilon = float(boundary_epsilon)
        self.reduction = reduction

        # Register alpha and beta as buffers.
        # They are saved with the module but are not optimized.
        self._block_name_to_key = {}

        for index, (block_name, shapes) in enumerate(
            block_shapes.items()
        ):
            safe_key = f"block_{index}"

            self.register_buffer(
                f"{safe_key}_alpha",
                shapes["alpha"].detach().clone(),
            )

            self.register_buffer(
                f"{safe_key}_beta",
                shapes["beta"].detach().clone(),
            )

            self._block_name_to_key[block_name] = safe_key

    def get_block_shapes(
        self,
    ) -> OrderedDict[str, dict[str, torch.Tensor]]:
        """
        Reconstruct the block-name to alpha/beta mapping.
        """
        block_shapes = OrderedDict()

        for block_name, safe_key in self._block_name_to_key.items():
            block_shapes[block_name] = {
                "alpha": getattr(
                    self,
                    f"{safe_key}_alpha",
                ),
                "beta": getattr(
                    self,
                    f"{safe_key}_beta",
                ),
            }

        return block_shapes

    def forward(
        self,
        step_sizes: Mapping[str, torch.Tensor],
        quantization_ranges: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculate the complete SBPQ Beta-prior loss.
        """
        return compute_network_beta_prior(
            step_sizes=step_sizes,
            quantization_ranges=quantization_ranges,
            layer_to_block=self.layer_to_block,
            block_shapes=self.get_block_shapes(),
            minimum_bits=self.minimum_bits,
            maximum_bits=self.maximum_bits,
            prior_scale=self.prior_scale,
            boundary_epsilon=self.boundary_epsilon,
            reduction=self.reduction,
        )