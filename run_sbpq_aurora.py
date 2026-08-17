"""Stage-one SBPQ runner for Microsoft Aurora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from SBPQ.aurora.beta_parameter_builder import (
    build_aurora_beta_parameters,
    save_aurora_beta_parameters,
)
from SBPQ.aurora.blocks import (
    build_aurora_block_mapping,
    build_layer_to_block,
    compute_block_parameter_counts,
    select_quant_layers,
)
from SBPQ.aurora.config import AuroraSBPQConfig, default_checkpoint_for_model
from SBPQ.aurora.data_utils import (
    attach_static_vars,
    detach_batch_to_cpu,
    load_aurora_pickle_batch,
    load_static_vars,
    move_batch_to_device,
    spatial_crop_batch,
)
from SBPQ.aurora.evaluation import evaluate_fixed_bits, evaluate_learned_steps, evaluate_model
from SBPQ.aurora.era5_loader import build_era5_windows
from SBPQ.aurora.model_utils import load_aurora_model
from SBPQ.aurora.ranges import compute_weight_ranges_aurora
from SBPQ.aurora.sensitivity import (
    compute_aurora_block_sensitivity,
    proxy_weight_sensitivity,
    save_block_sensitivity,
)
from SBPQ.aurora.trainer import AuroraSBPQTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="small", choices=["small", "pretrained", "full"])
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument("--max-quant-layers", type=int, default=12)
    parser.add_argument("--all-quant-layers", action="store_true")
    parser.add_argument("--sensitivity-mode", choices=["gradient", "proxy"], default="gradient")
    parser.add_argument("--run-name", default="stage1_small")
    parser.add_argument("--data-source", choices=["pickle", "era5"], default="pickle")
    parser.add_argument("--era5-raw-dir", default="dataset/aurora/era5_025/raw")
    parser.add_argument("--era5-days", nargs="+", default=["2023-01-01", "2023-01-02"])
    parser.add_argument("--calib-samples", type=int, default=1)
    parser.add_argument("--val-samples", type=int, default=1)
    parser.add_argument("--crop-height", type=int, default=None)
    parser.add_argument("--crop-width", type=int, default=None)
    parser.add_argument("--autocast-dtype", choices=["none", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--likelihood-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AuroraSBPQConfig(
        model_name=args.model_name,
        checkpoint_name=args.checkpoint_name or default_checkpoint_for_model(args.model_name),
        num_optimization_steps=args.steps,
        num_mc_samples=args.mc_samples,
        max_quant_layers=None if args.all_quant_layers else args.max_quant_layers,
        sensitivity_mode=args.sensitivity_mode,
        run_name=args.run_name,
        device=args.device,
        data_source=args.data_source,
        era5_raw_dir=Path(args.era5_raw_dir),
        era5_days=tuple(args.era5_days),
        calib_samples=args.calib_samples,
        val_samples=args.val_samples,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        autocast_dtype=None if args.autocast_dtype == "none" else args.autocast_dtype,
        likelihood_scale=args.likelihood_scale,
    )
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = cfg.artifacts_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.data_source == "era5":
        windows = build_era5_windows(
            cfg.era5_raw_dir,
            list(cfg.era5_days),
            max_windows=cfg.calib_samples + cfg.val_samples,
            crop_height=cfg.crop_height,
            crop_width=cfg.crop_width,
        )
        if len(windows) < cfg.calib_samples + cfg.val_samples:
            raise RuntimeError(
                f"Requested {cfg.calib_samples + cfg.val_samples} ERA5 windows, "
                f"but only found {len(windows)}."
            )
        frozen_batches = [window.input_batch for window in windows[: cfg.calib_samples]]
        val_batch = windows[cfg.calib_samples].input_batch
        val_target = windows[cfg.calib_samples].target_batch
        sensitivity_batch = frozen_batches[0]
        sensitivity_target = windows[0].target_batch
        print("[INFO] ERA5 windows:")
        for index, window in enumerate(windows):
            split = "calib" if index < cfg.calib_samples else "val"
            print(f"[INFO]   {index}: {split} {window.input_times} -> {window.target_time}")
    else:
        input_path = cfg.hf_root / cfg.input_pickle
        target_path = cfg.hf_root / cfg.target_pickle
        batch = load_aurora_pickle_batch(input_path)
        target = load_aurora_pickle_batch(target_path)
        if not batch.static_vars:
            static_vars = load_static_vars(cfg.hf_root / cfg.static_pickle)
            batch = attach_static_vars(batch, static_vars)
        if not target.static_vars:
            static_vars = load_static_vars(cfg.hf_root / cfg.static_pickle)
            target = attach_static_vars(target, static_vars)
        batch = spatial_crop_batch(batch, cfg.crop_height, cfg.crop_width)
        target = spatial_crop_batch(target, cfg.crop_height, cfg.crop_width)
        frozen_batches = [batch]
        val_batch = batch
        val_target = target
        sensitivity_batch = batch
        sensitivity_target = target

    print(f"[INFO] Loading Aurora model={cfg.model_name} checkpoint={cfg.checkpoint_name}")
    model = load_aurora_model(cfg.model_name, cfg.checkpoint_name, device=device)
    block_mapping = build_aurora_block_mapping(model)
    quant_layers = select_quant_layers(block_mapping, cfg.max_quant_layers)
    filtered_block_mapping = {
        block: [layer for layer in layers if layer in set(quant_layers)]
        for block, layers in block_mapping.items()
    }
    filtered_block_mapping = {
        block: layers for block, layers in filtered_block_mapping.items() if layers
    }
    layer_to_block = build_layer_to_block(filtered_block_mapping)
    block_counts = compute_block_parameter_counts(model, filtered_block_mapping)

    print(f"[INFO] Aurora blocks: {len(filtered_block_mapping)}")
    for block, layers in filtered_block_mapping.items():
        print(f"[INFO]   {block}: {len(layers)} Linear layers")

    ranges = compute_weight_ranges_aurora(
        model,
        quant_layers,
        percentile_prob=cfg.range_percentile,
    )
    torch.save(ranges, run_dir / "ranges.pt")

    if cfg.sensitivity_mode == "gradient":
        try:
            sensitivity = compute_aurora_block_sensitivity(
                model,
                sensitivity_batch,
                sensitivity_target,
                filtered_block_mapping,
                device=device,
            )
        except RuntimeError as error:
            print(f"[WARN] Gradient sensitivity failed: {error}")
            print("[WARN] Falling back to proxy weight sensitivity for stage-one smoke.")
            sensitivity = proxy_weight_sensitivity(model, filtered_block_mapping)
    else:
        sensitivity = proxy_weight_sensitivity(model, filtered_block_mapping)

    save_block_sensitivity(
        sensitivity,
        run_dir / "block_sensitivity.pt",
        metadata={"mode": cfg.sensitivity_mode, "model_name": cfg.model_name},
    )

    beta_parameters = build_aurora_beta_parameters(
        sensitivity=sensitivity,
        minimum_bits=cfg.minimum_bits,
        maximum_bits=cfg.maximum_bits,
        reference_bits=cfg.reference_bits,
        delta_bits=cfg.delta_bits,
        beta_kappa=cfg.beta_kappa,
        block_parameter_counts=block_counts,
        mean_epsilon=cfg.beta_epsilon,
        relative_epsilon=cfg.beta_relative_epsilon,
    )
    beta_path = run_dir / "beta_parameters.pt"
    save_aurora_beta_parameters(
        beta_parameters,
        beta_path,
        metadata={"model_name": cfg.model_name, "block_counts": block_counts},
    )

    with torch.no_grad():
        clean_outputs = [
            detach_batch_to_cpu(model(move_batch_to_device(batch_item, device)))
            for batch_item in frozen_batches
        ]

    trainer = AuroraSBPQTrainer(
        model=model,
        frozen_batches=frozen_batches,
        clean_network_outputs=clean_outputs,
        ranges_dict=ranges,
        layer_to_block=layer_to_block,
        beta_parameter_path=beta_path,
        initial_bits=cfg.init_bits,
        minimum_bits=cfg.minimum_bits,
        maximum_bits=cfg.maximum_bits,
        learning_rate=cfg.learning_rate,
        num_mc_samples=cfg.num_mc_samples,
        eta=cfg.eta,
        likelihood_scale=cfg.likelihood_scale,
        prior_scale=cfg.beta_prior_scale,
        weight_decay=cfg.weight_decay,
        gradient_clip_norm=cfg.gradient_clip_norm,
        autocast_dtype={
            None: None,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg.autocast_dtype],
        device=device,
    )

    for step in range(cfg.num_optimization_steps):
        record = trainer.train_step(step % len(frozen_batches))
        print(f"[TRAIN] step={step + 1} {record}")

    fp_metrics = evaluate_model(model, val_batch, val_target, device)
    fixed8_metrics = evaluate_fixed_bits(model, val_batch, val_target, ranges, 8.0, device)
    sbpq_metrics = evaluate_learned_steps(model, val_batch, val_target, trainer.get_step_sizes(), device)

    result = {
        "config": cfg.__dict__ | {
            "artifacts_root": str(cfg.artifacts_root),
            "hf_root": str(cfg.hf_root),
            "era5_raw_dir": str(cfg.era5_raw_dir),
            "era5_days": list(cfg.era5_days),
        },
        "run_dir": str(run_dir),
        "fp_metrics": fp_metrics,
        "fixed8_metrics": fixed8_metrics,
        "sbpq_metrics": sbpq_metrics,
        "history": trainer.history,
        "quant_layers": quant_layers,
        "block_mapping": filtered_block_mapping,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    torch.save(
        {
            "optimized_step_sizes": {
                key: value.detach().cpu()
                for key, value in trainer.get_step_sizes().items()
            },
            "history": trainer.history,
        },
        run_dir / "sbpq_trainer_state.pt",
    )
    print(f"[DONE] Saved Aurora SBPQ stage-one outputs to {run_dir}")
    print(json.dumps({"fp": fp_metrics, "fixed8": fixed8_metrics, "sbpq": sbpq_metrics}, indent=2))


if __name__ == "__main__":
    main()
