from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scOT.metrics import lp_error, relative_lp_error

from SBPQ.poseidon.evaluation import compute_dynamic_weight_step_sizes
from SBPQ.poseidon.poseidon_utils import (
    build_poseidon_loaders,
    load_poseidon_model,
    move_poseidon_batch_to_device,
)
from SBPQ.poseidon.sobolev import (
    compute_spatial_sobolev_loss,
    select_physical_channels,
)


DEFAULT_RUN_DIR = Path(
    "SBPQ/artifacts/poseidon/NS-PwC-L/runs/"
    "network_global_B8_d2_k10_ps1_mc10_eta1em06_lr3em05_init8_sob0_"
    "sw1_cal512_val2_steps20_sens-sob0_sw1_snl1_tw1_sow1_cal512_sensb512"
)


def mean_dict(records: list[dict[str, float]]) -> dict[str, float]:
    values = defaultdict(list)
    for record in records:
        for key, value in record.items():
            values[key].append(float(value))

    return {
        key: float(np.mean(item_values))
        for key, item_values in values.items()
    }


def denormalize_tensor(
    tensor: torch.Tensor,
    constants: dict,
    dataset_name: str,
) -> torch.Tensor:
    """
    Convert Poseidon-normalized fields back to physical units.
    """
    dataset_name = dataset_name.lower()

    if "wave" in dataset_name:
        output = tensor.clone()

        mean_u = torch.as_tensor(
            constants["mean"],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        std_u = torch.as_tensor(
            constants["std"],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        mean_c = torch.as_tensor(
            constants["mean_c"],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        std_c = torch.as_tensor(
            constants["std_c"],
            dtype=tensor.dtype,
            device=tensor.device,
        )

        output[:, 0] = tensor[:, 0] * std_u + mean_u
        if tensor.shape[1] >= 2:
            output[:, 1] = tensor[:, 1] * std_c + mean_c
        return output

    mean = torch.as_tensor(
        constants["mean"],
        dtype=tensor.dtype,
        device=tensor.device,
    ).flatten()
    std = torch.as_tensor(
        constants["std"],
        dtype=tensor.dtype,
        device=tensor.device,
    ).flatten()

    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    return tensor * std + mean


def spatial_grads_np(
    field: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Central finite differences.

    Args:
        field: Shape [B, H, W].

    Returns:
        dy_field, dx_field with the same shape as field.
    """
    dy_field = np.zeros_like(field)
    dx_field = np.zeros_like(field)

    dy_field[..., 1:-1, :] = (
        field[..., 2:, :] - field[..., :-2, :]
    ) / (2.0 * dy)
    dx_field[..., :, 1:-1] = (
        field[..., :, 2:] - field[..., :, :-2]
    ) / (2.0 * dx)

    dy_field[..., 0, :] = (
        field[..., 1, :] - field[..., 0, :]
    ) / dy
    dy_field[..., -1, :] = (
        field[..., -1, :] - field[..., -2, :]
    ) / dy
    dx_field[..., :, 0] = (
        field[..., :, 1] - field[..., :, 0]
    ) / dx
    dx_field[..., :, -1] = (
        field[..., :, -1] - field[..., :, -2]
    ) / dx

    return dy_field, dx_field


def ns_divergence_and_vorticity_metrics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    constants: dict,
    dataset_name: str,
    transpose_spatial_axes: bool,
) -> dict[str, float]:
    """
    Compute NS physical metrics on denormalized velocity.

    divergence is the mean absolute divergence of the prediction.
    vorticity is the L1 error between predicted and reference vorticity.
    """
    prediction = denormalize_tensor(
        prediction,
        constants=constants,
        dataset_name=dataset_name,
    )
    reference = denormalize_tensor(
        reference,
        constants=constants,
        dataset_name=dataset_name,
    )

    prediction_np = prediction.detach().cpu().numpy()
    reference_np = reference.detach().cpu().numpy()

    if transpose_spatial_axes:
        prediction_np = np.swapaxes(prediction_np, -2, -1)
        reference_np = np.swapaxes(reference_np, -2, -1)

    if prediction_np.shape[1] == 3:
        u_index, v_index = 0, 1
    else:
        u_index, v_index = 1, 2

    u_prediction = prediction_np[:, u_index]
    v_prediction = prediction_np[:, v_index]
    u_reference = reference_np[:, u_index]
    v_reference = reference_np[:, v_index]

    height = u_prediction.shape[-2]
    width = u_prediction.shape[-1]
    dx = 1.0 / max(width - 1, 1)
    dy = 1.0 / max(height - 1, 1)

    du_prediction_dy, du_prediction_dx = spatial_grads_np(
        u_prediction,
        dx=dx,
        dy=dy,
    )
    dv_prediction_dy, dv_prediction_dx = spatial_grads_np(
        v_prediction,
        dx=dx,
        dy=dy,
    )
    du_reference_dy, du_reference_dx = spatial_grads_np(
        u_reference,
        dx=dx,
        dy=dy,
    )
    dv_reference_dy, dv_reference_dx = spatial_grads_np(
        v_reference,
        dx=dx,
        dy=dy,
    )

    prediction_divergence = du_prediction_dx + dv_prediction_dy
    reference_divergence = du_reference_dx + dv_reference_dy

    prediction_vorticity = dv_prediction_dx - du_prediction_dy
    reference_vorticity = dv_reference_dx - du_reference_dy

    divergence = float(np.mean(np.abs(prediction_divergence)))
    divergence_error = float(
        np.mean(np.abs(prediction_divergence - reference_divergence))
    )
    vorticity_error = float(
        np.mean(np.abs(prediction_vorticity - reference_vorticity))
    )

    return {
        "divergence": divergence,
        "divergence_error": divergence_error,
        "vorticity": vorticity_error,
    }


def poseidon_sobolev_metric(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    constants: dict,
    dataset_name: str,
    max_order: int,
    transpose_spatial_axes: bool,
) -> torch.Tensor:
    prediction = denormalize_tensor(
        prediction,
        constants=constants,
        dataset_name=dataset_name,
    )
    reference = denormalize_tensor(
        reference,
        constants=constants,
        dataset_name=dataset_name,
    )

    if transpose_spatial_axes:
        prediction = prediction.transpose(-2, -1)
        reference = reference.transpose(-2, -1)

    prediction = select_physical_channels(
        field=prediction,
        dataset_name=dataset_name,
    )
    reference = select_physical_channels(
        field=reference,
        dataset_name=dataset_name,
    )

    return compute_spatial_sobolev_loss(
        prediction=prediction,
        reference=reference,
        max_order=max_order,
        order_weights=1.0,
        norm="l1",
    )


def make_weight_quantization_hook(step_size: torch.Tensor):
    def hook(module: nn.Linear, inputs, output):
        weight = module.weight
        step = step_size.to(
            device=weight.device,
            dtype=weight.dtype,
        ).reshape(-1, 1)
        step = step.clamp_min(1e-12)
        quantized_weight = torch.round(weight / step) * step

        return F.linear(
            inputs[0],
            quantized_weight,
            module.bias,
        )

    return hook


def evaluate_model_with_steps(
    model: nn.Module,
    batches: list[dict],
    dataset,
    dataset_name: str,
    device: torch.device,
    layer_names: list[str],
    weight_step_sizes: dict[str, torch.Tensor],
    sobolev_order: int,
    transpose_spatial_axes: bool,
) -> dict[str, float]:
    model = model.to(device).eval()
    name_to_module = dict(model.named_modules())
    handles = []

    for layer_name in layer_names:
        if layer_name not in weight_step_sizes:
            continue

        module = name_to_module.get(layer_name)
        if not isinstance(module, nn.Linear):
            continue

        step_size = weight_step_sizes[layer_name]
        if isinstance(step_size, (tuple, list)):
            step_size = step_size[0]
        if isinstance(step_size, nn.Parameter):
            step_size = step_size.detach()

        handles.append(
            module.register_forward_hook(
                make_weight_quantization_hook(step_size)
            )
        )

    records = []
    constants = dataset.constants

    try:
        with torch.no_grad():
            for batch in batches:
                pixel_values, time, pixel_mask, labels = (
                    move_poseidon_batch_to_device(
                        batch=batch,
                        device=device,
                    )
                )
                if labels is None:
                    continue

                output = model(
                    pixel_values=pixel_values,
                    time=time,
                    pixel_mask=pixel_mask,
                    labels=labels,
                ).output

                output_np = output.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()

                sobolev_key = f"sobolev_order{sobolev_order}"

                record = {
                    "l1": float(
                        lp_error(output_np, labels_np, p=1).mean()
                    ),
                    "rel_l1": float(
                        relative_lp_error(
                            output_np,
                            labels_np,
                            p=1,
                            return_percent=True,
                        ).mean()
                    ),
                    sobolev_key: float(
                        poseidon_sobolev_metric(
                            prediction=output,
                            reference=labels,
                            constants=constants,
                            dataset_name=dataset_name,
                            max_order=sobolev_order,
                            transpose_spatial_axes=transpose_spatial_axes,
                        ).detach().cpu().item()
                    ),
                }

                if "incompressible" in dataset_name.lower():
                    record.update(
                        ns_divergence_and_vorticity_metrics(
                            prediction=output,
                            reference=labels,
                            constants=constants,
                            dataset_name=dataset_name,
                            transpose_spatial_axes=transpose_spatial_axes,
                        )
                    )

                records.append(record)
    finally:
        for handle in handles:
            handle.remove()

    if len(records) == 0:
        raise RuntimeError("No labeled validation batches were evaluated.")

    return mean_dict(records)


def load_run_config(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r") as handle:
            metrics = json.load(handle)
        return metrics.get("run_config", {})
    return {}


def load_sbpq_step_sizes(run_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    checkpoint_path = run_dir / "sbpq_trainer_state.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing SBPQ checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "optimized_step_sizes" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} does not contain optimized_step_sizes."
        )

    step_sizes = {
        layer_name: torch.as_tensor(step_size).detach().cpu()
        for layer_name, step_size in checkpoint[
            "optimized_step_sizes"
        ].items()
    }

    return step_sizes, checkpoint


def compute_increase_vs_fp(
    results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    fp_metrics = results["fp"]
    increases = {}

    for method, metrics in results.items():
        if method == "fp":
            continue

        increases[method] = {
            metric_name: float(metric_value - fp_metrics[metric_name])
            for metric_name, metric_value in metrics.items()
            if metric_name in fp_metrics
        }

    return increases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved SBPQ Poseidon run with L1 and physical "
            "metrics using saved learned step sizes."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Saved SBPQ run directory containing sbpq_trainer_state.pt.",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sobolev-order",
        type=int,
        default=2,
        help="Sobolev order used only for this metric evaluation.",
    )
    parser.add_argument(
        "--no-transpose-spatial-axes",
        action="store_true",
        help=(
            "Disable the Fisher-style spatial-axis transpose before "
            "physical derivative metrics."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults inside the run directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_config = load_run_config(run_dir)

    model_path = args.model_path or run_config.get(
        "model_path",
        "models/NS-PwC-L",
    )
    data_path = args.data_path or run_config.get(
        "data_path",
        "dataset/NS-PwC",
    )
    dataset_name = args.dataset_name or run_config.get(
        "dataset_name",
        "fluids.incompressible.PiecewiseConstants",
    )
    val_batch_size = args.val_batch_size or int(
        run_config.get("val_batch_size", 128)
    )
    val_steps = args.val_steps or int(
        run_config.get("val_steps", 2)
    )

    step_sizes, checkpoint = load_sbpq_step_sizes(run_dir)
    layer_names = list(step_sizes.keys())

    model, device = load_poseidon_model(
        model_path=model_path,
        device=args.device,
    )
    device = torch.device(device)

    _, val_loader, _, val_iterator = build_poseidon_loaders(
        dataset_name=dataset_name,
        data_path=data_path,
        calib_batch_size=1,
        calib_steps=1,
        val_batch_size=val_batch_size,
        val_steps=val_steps,
        num_workers=args.num_workers,
    )
    frozen_val_batches = list(val_iterator())

    transpose_spatial_axes = not args.no_transpose_spatial_axes

    fixed_8bit_steps = compute_dynamic_weight_step_sizes(
        model=model,
        layer_names=layer_names,
        num_bits=8,
        device=device,
    )
    fixed_4bit_steps = compute_dynamic_weight_step_sizes(
        model=model,
        layer_names=layer_names,
        num_bits=4,
        device=device,
    )

    results = {
        "fp": evaluate_model_with_steps(
            model=model,
            batches=frozen_val_batches,
            dataset=val_loader.dataset,
            dataset_name=dataset_name,
            device=device,
            layer_names=layer_names,
            weight_step_sizes={},
            sobolev_order=args.sobolev_order,
            transpose_spatial_axes=transpose_spatial_axes,
        ),
        "sbpq_saved": evaluate_model_with_steps(
            model=model,
            batches=frozen_val_batches,
            dataset=val_loader.dataset,
            dataset_name=dataset_name,
            device=device,
            layer_names=layer_names,
            weight_step_sizes=step_sizes,
            sobolev_order=args.sobolev_order,
            transpose_spatial_axes=transpose_spatial_axes,
        ),
        "fixed_8bit": evaluate_model_with_steps(
            model=model,
            batches=frozen_val_batches,
            dataset=val_loader.dataset,
            dataset_name=dataset_name,
            device=device,
            layer_names=layer_names,
            weight_step_sizes=fixed_8bit_steps,
            sobolev_order=args.sobolev_order,
            transpose_spatial_axes=transpose_spatial_axes,
        ),
        "fixed_4bit": evaluate_model_with_steps(
            model=model,
            batches=frozen_val_batches,
            dataset=val_loader.dataset,
            dataset_name=dataset_name,
            device=device,
            layer_names=layer_names,
            weight_step_sizes=fixed_4bit_steps,
            sobolev_order=args.sobolev_order,
            transpose_spatial_axes=transpose_spatial_axes,
        ),
    }

    output = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "sbpq_trainer_state.pt"),
        "model_path": model_path,
        "data_path": data_path,
        "dataset_name": dataset_name,
        "val_batch_size": val_batch_size,
        "val_steps": val_steps,
        "num_validation_batches": len(frozen_val_batches),
        "num_layers": len(layer_names),
        "sobolev_order_metric": args.sobolev_order,
        "sobolev_norm": "l1",
        "physical_metrics_use_denormalized_fields": True,
        "transpose_spatial_axes_for_physical_metrics": (
            transpose_spatial_axes
        ),
        "saved_average_bits": {
            "parameter_weighted": checkpoint.get(
                "parameter_weighted_average_bits"
            ),
            "unweighted": checkpoint.get("unweighted_average_bits"),
        },
        "metrics": results,
        "increase_vs_fp": compute_increase_vs_fp(results),
    }

    output_path = args.output
    if output_path is None:
        output_path = (
            run_dir / f"physics_metrics_order{args.sobolev_order}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(output, indent=2))
    print(f"[INFO] Saved physical metrics to: {output_path}")


if __name__ == "__main__":
    main()
