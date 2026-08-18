"""
End-to-end Poseidon runner for SBPQ.

Workflow:
    load model/data
    detect structural blocks
    compute/load ranges
    compute/load Sobolev-aware block sensitivity
    build parameter-weighted Beta prior parameters
    optimize step sizes with network-wise MC likelihood + Beta prior
    evaluate and save results
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from SBPQ.trainer import SBPQTrainer
from SBPQ.poseidon.config import SBPQConfig
from SBPQ.poseidon.poseidon_utils import (
    build_poseidon_loaders,
    get_clean_network_outputs_poseidon,
    load_poseidon_model,
)
from SBPQ.poseidon.blocks import (
    build_poseidon_block_mapping,
    compute_block_parameter_counts,
    print_poseidon_block_summary,
)
from SBPQ.poseidon.sobolev import compute_poseidon_sobolev_loss
from SBPQ.poseidon.sensitivity import (
    compute_poseidon_block_sensitivity,
    load_block_sensitivity,
    normalize_block_sensitivity,
    save_block_sensitivity,
)
from SBPQ.poseidon.beta_parameter_builder import (
    build_and_save_from_sensitivity_file,
)
from SBPQ.poseidon.ranges import (
    compute_data_ranges_poseidon,
    get_poseidon_range_cache_path,
    load_precalculated_ranges_if_exists,
    save_poseidon_ranges,
)
from SBPQ.poseidon.evaluation import (
    build_channel_parameter_weights,
    compute_dynamic_weight_step_sizes,
    evaluate_poseidon_with_weight_steps,
)


def freeze_batches(iterator) -> list[dict]:
    """
    Materialize calibration batches so likelihood targets stay fixed.
    """
    return list(iterator())


@contextmanager
def cache_writer_lock(
    target_path: Path,
    poll_seconds: float = 10.0,
):
    """
    Use an atomic directory lock to prevent concurrent cache writes.
    """
    lock_path = Path(f"{target_path}.lock")
    lock_acquired = False

    while not lock_acquired:
        try:
            lock_path.mkdir(parents=True)
            lock_acquired = True
        except FileExistsError:
            if target_path.exists():
                yield False
                return

            print(
                f"[INFO] Waiting for cache lock: {lock_path}"
            )
            time.sleep(poll_seconds)

    try:
        yield True
    finally:
        if lock_acquired:
            try:
                lock_path.rmdir()
            except OSError:
                pass


def format_value_for_path(value: Any) -> str:
    """
    Convert a hyperparameter value into a compact path-safe string.
    """
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)

    return (
        text
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
        .replace(",", "-")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
    )


def format_weights_for_path(weights) -> str:
    """
    Format Sobolev order weights for cache/run names.
    """
    if isinstance(weights, (int, float)):
        return format_value_for_path(float(weights))

    return "-".join(
        format_value_for_path(float(weight))
        for weight in weights
    )


def build_sensitivity_tag(
    cfg: SBPQConfig,
    number_of_calibration_batches: int,
) -> str:
    """
    Sensitivity depends on Sobolev settings and calibration data volume.
    """
    parts = [
        f"sob{cfg.sobolev_order}",
        f"sw{format_weights_for_path(cfg.sobolev_order_weights)}",
        f"sn{cfg.sobolev_norm}",
        f"tw{format_value_for_path(cfg.task_loss_weight)}",
        f"sow{format_value_for_path(cfg.sobolev_loss_weight)}",
        f"cal{number_of_calibration_batches}",
    ]

    if cfg.sensitivity_batches is not None:
        parts.append(f"sensb{cfg.sensitivity_batches}")

    if cfg.max_blocks is not None:
        parts.append(f"blocks{cfg.max_blocks}")

    return "_".join(parts)


def build_beta_tag(
    cfg: SBPQConfig,
    sensitivity_tag: str,
) -> str:
    """
    Beta parameters depend on sensitivity plus Beta-prior settings.
    """
    parts = [
        sensitivity_tag,
        f"B{format_value_for_path(cfg.reference_bits)}",
        f"d{format_value_for_path(cfg.delta_bits)}",
        f"k{format_value_for_path(cfg.beta_kappa)}",
        f"bmin{format_value_for_path(cfg.minimum_bits)}",
        f"bmax{format_value_for_path(cfg.maximum_bits)}",
        f"beps{format_value_for_path(cfg.beta_epsilon)}",
        f"releps{format_value_for_path(cfg.beta_relative_epsilon)}",
    ]

    return "_".join(parts)


def build_run_tag(
    cfg: SBPQConfig,
    sensitivity_tag: str,
) -> str:
    """
    Full optimization/evaluation run tag.
    """
    parts = [
        "network_global",
    ]

    if cfg.run_group:
        parts.append(
            f"group{format_value_for_path(cfg.run_group)}"
        )

    parts.extend([
        f"B{format_value_for_path(cfg.reference_bits)}",
        f"d{format_value_for_path(cfg.delta_bits)}",
        f"k{format_value_for_path(cfg.beta_kappa)}",
        f"ps{format_value_for_path(cfg.beta_prior_scale)}",
        f"mc{cfg.num_mc_samples}",
        f"eta{format_value_for_path(cfg.eta)}",
        f"lr{format_value_for_path(cfg.learning_rate)}",
        f"init{format_value_for_path(cfg.init_bits)}",
        f"sob{cfg.sobolev_order}",
        f"sw{format_weights_for_path(cfg.sobolev_order_weights)}",
        f"cal{cfg.calib_steps}",
        f"val{cfg.val_steps}",
    ])

    if cfg.num_optimization_steps is not None:
        parts.append(f"steps{cfg.num_optimization_steps}")

    if cfg.max_blocks is not None:
        parts.append(f"blocks{cfg.max_blocks}")

    parts.append(f"sens-{sensitivity_tag}")

    return "_".join(parts)


def load_candidate_layers(
    model: nn.Module,
    quant_layer_path: Path,
    fallback_layers,
) -> list[str]:
    """
    Load candidate quantization layers, falling back to detected block layers.
    """
    if not quant_layer_path.exists():
        print(
            f"[INFO] Quant layer file not found: {quant_layer_path}. "
            "Using all Linear layers inside detected blocks."
        )
        return list(fallback_layers)

    if quant_layer_path.suffix == ".pt":
        saved = torch.load(
            quant_layer_path,
            map_location="cpu",
        )
        if isinstance(saved, dict) and "quantize_layers" in saved:
            layer_names = saved["quantize_layers"]
        else:
            raise ValueError(
                f"Unsupported quant layer file format: {quant_layer_path}"
            )
    elif quant_layer_path.suffix == ".json":
        with quant_layer_path.open("r") as handle:
            saved = json.load(handle)
        if isinstance(saved, list):
            layer_names = saved
        elif isinstance(saved, dict) and "quantize_layers" in saved:
            layer_names = saved["quantize_layers"]
        else:
            raise ValueError(
                f"Unsupported quant layer file format: {quant_layer_path}"
            )
    else:
        raise ValueError(
            f"Unsupported quant layer file type: {quant_layer_path}"
        )

    name_to_module = dict(model.named_modules())
    candidate_layers = [
        layer_name
        for layer_name in layer_names
        if isinstance(name_to_module.get(layer_name), nn.Linear)
        and layer_name in fallback_layers
    ]

    if len(candidate_layers) == 0:
        raise RuntimeError(
            "No candidate Linear layers are inside detected blocks."
        )

    return candidate_layers


def build_sensitivity_loss_function(
    cfg: SBPQConfig,
    dataset,
):
    """
    Create L_task + L_sob for sensitivity computation.
    """

    def loss_function(
        prediction: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        task_loss = F.mse_loss(
            prediction,
            reference,
        )
        sobolev_loss = compute_poseidon_sobolev_loss(
            prediction=prediction,
            reference=reference,
            dataset=dataset,
            dataset_name=cfg.dataset_name,
            max_order=cfg.sobolev_order,
            order_weights=cfg.sobolev_order_weights,
            norm=cfg.sobolev_norm,
            transpose_spatial_axes=cfg.sobolev_transpose,
        )

        return (
            float(cfg.task_loss_weight) * task_loss
            + float(cfg.sobolev_loss_weight) * sobolev_loss
        )

    return loss_function


def tensor_dict_to_float_dict(values: dict) -> dict:
    return {
        key: float(torch.as_tensor(value).detach().cpu().item())
        for key, value in values.items()
    }


def main() -> None:
    cfg = SBPQConfig()
    cfg.create_directories()

    torch.manual_seed(cfg.random_seed)

    print("\n========== SBPQ POSEIDON CONFIG ==========")
    print(f"model_path:        {cfg.model_path}")
    print(f"data_path:         {cfg.data_path}")
    print(f"dataset_name:      {cfg.dataset_name}")
    print(f"reference_bits:    {cfg.reference_bits}")
    print(f"delta_bits:        {cfg.delta_bits}")
    print(f"beta_kappa:        {cfg.beta_kappa}")
    print(f"beta_prior_scale:  {cfg.beta_prior_scale}")
    print(f"num_mc_samples:    {cfg.num_mc_samples}")
    print(f"eta:               {cfg.eta}")
    print(f"learning_rate:     {cfg.learning_rate}")
    print(f"sobolev_order:     {cfg.sobolev_order}")
    print(f"sobolev_norm:      {cfg.sobolev_norm}")
    print("==========================================\n")

    model, device = load_poseidon_model(
        model_path=cfg.model_path,
        device=cfg.device,
    )

    (
        calib_loader,
        val_loader,
        calib_iterator,
        val_iterator,
    ) = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batch_size=cfg.calib_batch_size,
        calib_steps=cfg.calib_steps,
        val_batch_size=cfg.val_batch_size,
        val_steps=cfg.val_steps,
        num_workers=cfg.num_workers,
    )

    (
        blocks,
        block_to_layers,
        layer_to_block_all,
        unassigned_layers,
    ) = build_poseidon_block_mapping(model)

    if cfg.max_blocks is not None:
        selected_block_names = list(blocks.keys())[: cfg.max_blocks]
        blocks = type(blocks)(
            (
                block_name,
                blocks[block_name],
            )
            for block_name in selected_block_names
        )
        block_to_layers = {
            block_name: block_to_layers[block_name]
            for block_name in selected_block_names
        }
        layer_to_block_all = {
            layer_name: block_name
            for block_name, layer_names in block_to_layers.items()
            for layer_name in layer_names
        }
        print(
            f"[INFO] Limiting run to first {cfg.max_blocks} blocks "
            "for smoke/debug execution."
        )

    print_poseidon_block_summary(
        blocks=blocks,
        block_to_layers=block_to_layers,
        unassigned_layers=unassigned_layers,
    )

    fallback_layers = [
        layer_name
        for layers in block_to_layers.values()
        for layer_name in layers
    ]

    candidate_layers = load_candidate_layers(
        model=model,
        quant_layer_path=cfg.quant_layer_path,
        fallback_layers=fallback_layers,
    )

    layer_to_block = {
        layer_name: layer_to_block_all[layer_name]
        for layer_name in candidate_layers
    }

    block_parameter_counts = compute_block_parameter_counts(
        model=model,
        block_to_layers=block_to_layers,
        layer_names=candidate_layers,
    )

    frozen_batches = freeze_batches(
        calib_iterator
    )

    if len(frozen_batches) == 0:
        raise RuntimeError(
            "No calibration batches were frozen."
        )

    frozen_val_batches = freeze_batches(
        val_iterator
    )

    if len(frozen_val_batches) == 0:
        raise RuntimeError(
            "No validation batches were frozen."
        )

    sensitivity_tag = build_sensitivity_tag(
        cfg=cfg,
        number_of_calibration_batches=len(frozen_batches),
    )
    beta_tag = build_beta_tag(
        cfg=cfg,
        sensitivity_tag=sensitivity_tag,
    )
    run_name = build_run_tag(
        cfg=cfg,
        sensitivity_tag=sensitivity_tag,
    )

    print("\n========== SBPQ ARTIFACT PATHS ==========")
    print(f"sensitivity_tag: {sensitivity_tag}")
    print(f"beta_tag:        {beta_tag}")
    print(f"run_name:        {run_name}")
    print("=========================================\n")

    ranges = load_precalculated_ranges_if_exists(
        repo_root=cfg.repo_root,
        model_path=cfg.model_path,
        dataset_name=cfg.dataset_name,
        percentile_prob=cfg.percentile_prob,
        range_method=cfg.range_method,
        device=device,
    )

    if ranges is None:
        ranges = compute_data_ranges_poseidon(
            model=model,
            dataloader=frozen_batches,
            device=device,
            layer_names=candidate_layers,
            percentile_prob=cfg.percentile_prob,
            range_method=cfg.range_method,
        )

        save_poseidon_ranges(
            ranges=ranges,
            save_path=get_poseidon_range_cache_path(
                repo_root=cfg.repo_root,
                model_path=cfg.model_path,
                dataset_name=cfg.dataset_name,
                percentile_prob=cfg.percentile_prob,
                range_method=cfg.range_method,
            ),
            metadata={
                "model_path": cfg.model_path,
                "dataset_name": cfg.dataset_name,
                "percentile_prob": cfg.percentile_prob,
                "range_method": cfg.range_method,
                "layer_names": candidate_layers,
            },
        )

    sensitivity_path = (
        cfg.poseidon_artifact_dir
        / "sensitivity"
        / f"sobolev_block_sensitivity_{sensitivity_tag}.pt"
    )

    if not sensitivity_path.exists():
        with cache_writer_lock(sensitivity_path) as should_write:
            if should_write and not sensitivity_path.exists():
                loss_function = build_sensitivity_loss_function(
                    cfg=cfg,
                    dataset=calib_loader.dataset,
                )
                raw_sensitivity = compute_poseidon_block_sensitivity(
                    model=model,
                    dataloader=lambda: iter(frozen_batches),
                    device=device,
                    loss_function=loss_function,
                    max_batches=cfg.sensitivity_batches,
                    block_names=list(blocks.keys()),
                )
                normalized_sensitivity = normalize_block_sensitivity(
                    sensitivity=raw_sensitivity,
                    epsilon=cfg.sensitivity_epsilon,
                )
                save_block_sensitivity(
                    raw_sensitivity=raw_sensitivity,
                    normalized_sensitivity=normalized_sensitivity,
                    save_path=sensitivity_path,
                    metadata={
                        "model_path": cfg.model_path,
                        "dataset_name": cfg.dataset_name,
                        "sobolev_order": cfg.sobolev_order,
                        "sobolev_order_weights": cfg.sobolev_order_weights,
                        "sobolev_norm": cfg.sobolev_norm,
                        "task_loss_weight": cfg.task_loss_weight,
                        "sobolev_loss_weight": cfg.sobolev_loss_weight,
                        "calibration_batches": len(frozen_batches),
                    },
                )

    print(f"[INFO] Loading block sensitivity: {sensitivity_path}")
    sensitivity_object = load_block_sensitivity(
        sensitivity_path=sensitivity_path,
    )
    raw_sensitivity = sensitivity_object["raw_sensitivity"]

    beta_parameter_path = (
        cfg.poseidon_artifact_dir
        / "beta_parameters"
        / f"weighted_beta_parameters_{beta_tag}.pt"
    )

    if not beta_parameter_path.exists():
        with cache_writer_lock(beta_parameter_path) as should_write:
            if should_write and not beta_parameter_path.exists():
                build_and_save_from_sensitivity_file(
                    sensitivity_path=sensitivity_path,
                    save_path=beta_parameter_path,
                    minimum_bits=cfg.minimum_bits,
                    maximum_bits=cfg.maximum_bits,
                    reference_bits=cfg.reference_bits,
                    delta_bits=cfg.delta_bits,
                    beta_kappa=cfg.beta_kappa,
                    block_parameter_counts=block_parameter_counts,
                    mean_epsilon=cfg.beta_epsilon,
                    relative_epsilon=cfg.beta_relative_epsilon,
                    metadata={
                        "model_path": cfg.model_path,
                        "dataset_name": cfg.dataset_name,
                        "parameter_weighted_centering": True,
                        "sensitivity_tag": sensitivity_tag,
                        "beta_tag": beta_tag,
                    },
                )

    print(f"[INFO] Using Beta parameters: {beta_parameter_path}")

    clean_outputs = get_clean_network_outputs_poseidon(
        model=model,
        frozen_batches=frozen_batches,
        device=device,
    )

    channel_weights = build_channel_parameter_weights(
        model=model,
        layer_names=candidate_layers,
    )

    trainer = SBPQTrainer(
        model=model,
        frozen_batches=frozen_batches,
        clean_network_outputs=clean_outputs,
        ranges_dict=ranges,
        layer_to_block=layer_to_block,
        beta_parameter_path=beta_parameter_path,
        initial_bits=cfg.init_bits,
        minimum_bits=cfg.minimum_bits,
        maximum_bits=cfg.maximum_bits,
        learning_rate=cfg.learning_rate,
        num_mc_samples=cfg.num_mc_samples,
        eta=cfg.eta,
        prior_scale=cfg.beta_prior_scale,
        likelihood_scale=cfg.likelihood_scale,
        beta_boundary_epsilon=cfg.beta_epsilon,
        weight_decay=cfg.weight_decay,
        gradient_clip_norm=cfg.gradient_clip_norm,
        channel_weights=channel_weights,
        device=device,
    )

    number_of_steps = cfg.num_optimization_steps
    if number_of_steps is None:
        number_of_steps = (
            cfg.num_epochs
            * len(frozen_batches)
            * cfg.updates_per_batch
        )

    history = trainer.optimize(
        number_of_steps=number_of_steps,
        print_every=cfg.log_every,
    )

    learned_steps = trainer.get_step_sizes()

    fp_metrics = evaluate_poseidon_with_weight_steps(
        model=model,
        dataloader=frozen_val_batches,
        weight_step_sizes={},
        layer_names=candidate_layers,
        device=device,
    )
    sbpq_metrics = evaluate_poseidon_with_weight_steps(
        model=model,
        dataloader=frozen_val_batches,
        weight_step_sizes=learned_steps,
        layer_names=candidate_layers,
        device=device,
    )

    baseline_metrics = {}
    for bits in (4, 8, 16):
        dynamic_steps = compute_dynamic_weight_step_sizes(
            model=model,
            layer_names=candidate_layers,
            num_bits=bits,
            device=device,
        )
        baseline_metrics[f"fixed_{bits}bit"] = (
            evaluate_poseidon_with_weight_steps(
                model=model,
                dataloader=frozen_val_batches,
                weight_step_sizes=dynamic_steps,
                layer_names=candidate_layers,
                device=device,
            )
        )

    run_dir = cfg.poseidon_artifact_dir / "runs" / run_name
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save(
        save_path=run_dir / "sbpq_trainer_state.pt",
        metadata={
            "model_path": cfg.model_path,
            "dataset_name": cfg.dataset_name,
            "num_candidate_layers": len(candidate_layers),
            "num_frozen_batches": len(frozen_batches),
            "num_frozen_validation_batches": len(frozen_val_batches),
            "beta_parameter_path": str(beta_parameter_path),
            "sensitivity_path": str(sensitivity_path),
            "run_name": run_name,
            "sensitivity_tag": sensitivity_tag,
            "beta_tag": beta_tag,
        },
    )

    metrics = {
        "artifact_paths": {
            "run_dir": str(run_dir),
            "sensitivity_path": str(sensitivity_path),
            "beta_parameter_path": str(beta_parameter_path),
        },
        "run_config": {
            "model_path": cfg.model_path,
            "data_path": cfg.data_path,
            "dataset_name": cfg.dataset_name,
            "run_group": cfg.run_group,
            "reference_bits": cfg.reference_bits,
            "delta_bits": cfg.delta_bits,
            "beta_kappa": cfg.beta_kappa,
            "beta_prior_scale": cfg.beta_prior_scale,
            "minimum_bits": cfg.minimum_bits,
            "maximum_bits": cfg.maximum_bits,
            "initial_bits": cfg.init_bits,
            "num_mc_samples": cfg.num_mc_samples,
            "eta": cfg.eta,
            "likelihood_scale": cfg.likelihood_scale,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "gradient_clip_norm": cfg.gradient_clip_norm,
            "sobolev_order": cfg.sobolev_order,
            "sobolev_order_weights": cfg.sobolev_order_weights,
            "sobolev_norm": cfg.sobolev_norm,
            "task_loss_weight": cfg.task_loss_weight,
            "sobolev_loss_weight": cfg.sobolev_loss_weight,
            "calib_batch_size": cfg.calib_batch_size,
            "calib_steps": cfg.calib_steps,
            "val_batch_size": cfg.val_batch_size,
            "val_steps": cfg.val_steps,
            "num_frozen_calibration_batches": len(frozen_batches),
            "num_frozen_validation_batches": len(frozen_val_batches),
        },
        "fp": fp_metrics,
        "sbpq": sbpq_metrics,
        **baseline_metrics,
        "average_bits": {
            "parameter_weighted": trainer.calculate_average_bitwidth(
                parameter_weighted=True,
            ),
            "unweighted": trainer.calculate_average_bitwidth(
                parameter_weighted=False,
            ),
        },
        "block_sensitivity": tensor_dict_to_float_dict(
            raw_sensitivity,
        ),
    }

    with (run_dir / "history.json").open("w") as handle:
        json.dump(
            history,
            handle,
            indent=2,
        )

    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
        )

    print("\n========== SBPQ FINAL METRICS ==========")
    print(json.dumps(metrics, indent=2))
    print("========================================\n")


if __name__ == "__main__":
    main()
