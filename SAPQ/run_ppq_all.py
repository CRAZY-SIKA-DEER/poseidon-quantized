from __future__ import annotations

import json
from pathlib import Path
import copy
import os

import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import (
    load_poseidon_model,
    build_poseidon_loaders,
)
from PPQ.ranges import (
    load_precalculated_ranges_if_exists,
    compute_data_ranges_poseidon,
)
from PPQ.metrics import (
    evaluate_with_stepsizes,
    compute_dynamic_stepsizes,
)

from SAPQ.sapq_layerwise_trainer import SAPQLayerwiseTrainer


def load_candidate_layers(model, quant_layer_path: Path):
    """
    Load candidate quantization layer names and keep only raw-model nn.Linear layers.
    Supports:
      - .pt with {"quantize_layers": [...]}
      - .json as list
      - .json with {"quantize_layers": [...]}
    """
    if not quant_layer_path.exists():
        raise FileNotFoundError(f"Quant layer file not found: {quant_layer_path}")

    if quant_layer_path.suffix == ".pt":
        obj = torch.load(quant_layer_path, map_location="cpu")
        if isinstance(obj, dict) and "quantize_layers" in obj:
            layer_names = obj["quantize_layers"]
        else:
            raise ValueError(f"Unsupported PT quant layer format: {quant_layer_path}")
    elif quant_layer_path.suffix == ".json":
        with open(quant_layer_path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            layer_names = obj
        elif isinstance(obj, dict) and "quantize_layers" in obj:
            layer_names = obj["quantize_layers"]
        else:
            raise ValueError(f"Unsupported JSON quant layer format: {quant_layer_path}")
    else:
        raise ValueError(f"Unsupported quant layer file type: {quant_layer_path}")

    name2mod = dict(model.named_modules())
    candidate_layers = [
        name for name in layer_names
        if isinstance(name2mod.get(name, None), nn.Linear)
    ]

    print(f"[INFO] Loaded {len(candidate_layers)} candidate Linear layers.")
    return candidate_layers


def load_sapq_sensitivity(cfg: PPQConfig, device: torch.device):
    """
    Load precomputed SAPQ sensitivity from:
      SAPQ/prior_sensitivity/<model_name>/prior_sensitivity.pt

    Uses:
      save_obj["layer_sensitivity_norm"]
    which is channel-wise sensitivity in ORIGINAL PPQ namespace.
    """
    model_name = Path(cfg.model_path).name
    sens_path = (
        Path(cfg.repo_root)
        / "SAPQ"
        / "prior_sensitivity"
        / model_name
        / "prior_sensitivity.pt"
    )

    if not sens_path.exists():
        raise FileNotFoundError(
            f"SAPQ sensitivity file not found: {sens_path}\n"
            f"Please run SAPQ/sapq_sensitivity.py first."
        )

    print(f"[INFO] Loading SAPQ sensitivity from: {sens_path}")
    obj = torch.load(sens_path, map_location="cpu")

    if "layer_sensitivity_norm" not in obj:
        raise ValueError(f"'layer_sensitivity_norm' not found in: {sens_path}")

    sens_dict = {
        name: tensor.to(device)
        for name, tensor in obj["layer_sensitivity_norm"].items()
    }

    print(f"[INFO] Loaded sensitivity for {len(sens_dict)} layers.")
    return sens_dict


def maybe_load_or_compute_ranges(cfg: PPQConfig, model, frozen_iter, candidate_layers, device):
    model_name = Path(cfg.model_path).name

    if model_name == "NS-PwC-L":
        ranges_path = (
            Path(cfg.repo_root)
            / "precalculated_ranges"
            / model_name
            / "p1e-04"
            / "ranges.pt"
        )
    else:
        ranges_path = (
            Path(cfg.repo_root)
            / "precalculated_ranges"
            / model_name
            / "best"
            / "ranges.pt"
        )

    if ranges_path.exists():
        print(f"[INFO] Loading ranges from: {ranges_path}")
        obj = torch.load(ranges_path, map_location="cpu")
        ranges_dict = obj["ranges_dict"]

        out = {}
        for name, value in ranges_dict.items():
            out[name] = {
                "weight_ranges": value["weight_ranges"].to(device),
                "activation_ranges": value["activation_ranges"].to(device),
            }
        return out, ranges_path

    raise FileNotFoundError(f"Ranges file not found: {ranges_path}")


def load_frozen_calibration_batches(cfg, device: torch.device):
    """
    Load pre-frozen calibration batches from:
        <repo_root>/ppq_artifacts/frozen_calibration_batches.pt

    Expected format:
        list[dict] with keys:
            pixel_values, labels, time, pixel_mask
    """
    dataset_tag = Path(cfg.data_path).name
    frozen_path = (
        Path(cfg.repo_root)
        / "ppq_artifacts"
        / f"{dataset_tag}-calib"
        / "frozen_calibration_batches.pt"
    )

    if not frozen_path.exists():
        raise FileNotFoundError(
            f"Frozen calibration file not found: {frozen_path}"
        )

    print(f"[INFO] Loading frozen calibration batches from: {frozen_path}")
    frozen_batches = torch.load(frozen_path, map_location="cpu")

    if not isinstance(frozen_batches, list):
        raise ValueError(
            f"Expected frozen_batches to be a list, got {type(frozen_batches)}"
        )
    if len(frozen_batches) == 0:
        raise ValueError("Frozen calibration batch list is empty.")

    first_batch = frozen_batches[0]

    dataset_tag = Path(cfg.data_path).name

    if dataset_tag in {"Wave-Layer", "Wave-Gauss"}:
        required_keys = {"pixel_values", "labels", "time"}
    else:
        required_keys = {"pixel_values", "labels", "time", "pixel_mask"}

    missing = required_keys - set(first_batch.keys())
    if missing:
        raise ValueError(
            f"Frozen calibration batch missing keys: {missing}"
        )
    

    if not isinstance(first_batch, dict):
        raise ValueError(
            f"Expected each frozen batch to be a dict, got {type(first_batch)}"
        )

    missing = required_keys - set(first_batch.keys())
    if missing:
        raise ValueError(
            f"Frozen calibration batch missing keys: {missing}"
        )

    print(f"[INFO] Loaded {len(frozen_batches)} frozen calibration batches.")
    return frozen_batches


def main():
    cfg = PPQConfig()
    cfg.prior_mode = os.environ.get("SAPQ_PRIOR_MODE", "ppq")
    cfg.exp_name = cfg.prior_mode
    cfg.eval_every = None

    # --------------------------------------------------
    # Set ablation choice here
    # --------------------------------------------------
    #cfg.prior_mode = "ppq"         # choose from: "ppq", "block_no_sens", "block_sens"
    # cfg.exp_name = "layer_ppq"     # example: "layer_ppq", "layer_block_no_sens", "layer_block_sens"

    # cfg.prior_mode = "block_no_sens"
    # cfg.exp_name = "layer_block_no_sens"

    #cfg.prior_mode = "block_sens"
    #cfg.exp_name = "layer_block_sens"

    #cfg.prior_mode = os.environ.get("SAPQ_PRIOR_MODE", cfg.prior_mode)

    #exp_subdir = cfg.prior_mode
    cfg.eval_every = None

    print(f"Mode: {cfg.prior_mode}")
    print(f"Experiment: {cfg.exp_name}")

    print("Loading Poseidon model...")
    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    print("Building Poseidon loaders (validation loader + optional calib loader)...")
    calib_loader, val_loader, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=cfg.val_batchsize,
        val_steps=cfg.val_steps,
    )

    print(f"[INFO] quant_layer_path = {cfg.quant_layer_path}")
    candidate_layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

    # --------------------------------------------------
    # Load fixed frozen calibration dataset from disk
    # --------------------------------------------------
    frozen_batches = load_frozen_calibration_batches(cfg, device=device)

    def frozen_iter():
        for batch in frozen_batches:
            yield batch

    print(f"[INFO] Frozen calibration batches: {len(frozen_batches)}")

    ranges_dict, ranges_path = maybe_load_or_compute_ranges(
        cfg=cfg,
        model=model,
        frozen_iter=frozen_iter,
        candidate_layers=candidate_layers,
        device=device,
    )

    if cfg.prior_mode == "block_sens":
        sens_dict = load_sapq_sensitivity(cfg, device=device)
    else:
        sens_dict = None
        print(f"[INFO] prior_mode={cfg.prior_mode}; skip loading sensitivity.")

    # --------------------------------------------------
    # Optional dynamic baselines for comparison
    # --------------------------------------------------
    print("[INFO] Computing dynamic baselines...")
    dyn4_steps = compute_dynamic_stepsizes(
        model=model,
        layer_names=candidate_layers,
        num_bits=4,
        device=device,
    )
    dyn8_steps = compute_dynamic_stepsizes(
        model=model,
        layer_names=candidate_layers,
        num_bits=8,
        device=device,
    )
    dyn16_steps = compute_dynamic_stepsizes(
        model=model,
        layer_names=candidate_layers,
        num_bits=16,
        device=device,
    )

    print("Creating SAPQLayerwiseTrainer...")
    trainer = SAPQLayerwiseTrainer(
        model=copy.deepcopy(model),
        config=cfg,
        layer_names=candidate_layers,
        device=str(device),
    )

    def eval_callback(epoch, step_sizes_dict, ranges_dict_cb):
        print(f"\n================ SAPQ LAYERWISE EVAL @ epoch {epoch} ================")

        sapq_metrics = evaluate_with_stepsizes(
            model=model,
            val_loader=val_iter,
            weight_steps=step_sizes_dict,
            act_steps=None,
            layer_names=candidate_layers,
            device=device,
        )

        print(
            f"[SAPQ-LAYERWISE-EPOCH-{epoch}] "
            f"L1={sapq_metrics['l1']:.6e} | RelL1={sapq_metrics['rel_l1']:.6e}"
        )

    print("Starting SAPQ layerwise training...")
    step_sizes_dict, ranges_dict, history = trainer.train(
        dataloader=frozen_iter,
        ranges_dict=ranges_dict,
        sens_dict=sens_dict,
        eval_callback=eval_callback if cfg.eval_every is not None else None,
    )

    print("\nEvaluating FP / SAPQ-layerwise / Dyn4 / Dyn8 / Dyn16 on validation iterator...")

    fp_metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps={},
        act_steps=None,
        layer_names=candidate_layers,
        device=device,
    )

    sapq_metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=step_sizes_dict,
        act_steps=None,
        layer_names=candidate_layers,
        device=device,
    )

    dyn4_metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=dyn4_steps,
        act_steps=None,
        layer_names=candidate_layers,
        device=device,
    )
    dyn8_metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=dyn8_steps,
        act_steps=None,
        layer_names=candidate_layers,
        device=device,
    )
    dyn16_metrics = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=dyn16_steps,
        act_steps=None,
        layer_names=candidate_layers,
        device=device,
    )

    print("number of epochs:", cfg.num_epochs)
    print("\n========== FINAL RESULTS ==========")
    print(f"FP              | L1={fp_metrics['l1']:.6e} | RelL1={fp_metrics['rel_l1']:.6e}")
    print(f"SAPQ-Layerwise  | L1={sapq_metrics['l1']:.6e} | RelL1={sapq_metrics['rel_l1']:.6e}")
    print(f"Dyn4            | L1={dyn4_metrics['l1']:.6e} | RelL1={dyn4_metrics['rel_l1']:.6e}")
    print(f"Dyn8            | L1={dyn8_metrics['l1']:.6e} | RelL1={dyn8_metrics['rel_l1']:.6e}")
    print(f"Dyn16           | L1={dyn16_metrics['l1']:.6e} | RelL1={dyn16_metrics['rel_l1']:.6e}")

    # --------------------------------------------------
    # Save outputs (separate per experiment)
    # --------------------------------------------------
    model_tag = Path(cfg.model_path).name
    dataset_tag = Path(cfg.data_path).name

    out_dir = (
        Path(cfg.repo_root)
        / "ppq_experiments"
        / model_tag
        / dataset_tag
        / "layerwise_ppq"
    )

    print("[DEBUG] save out_dir =", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[DEBUG] exists =", out_dir.exists())

    steps_path = out_dir / "sapq_layerwise_step_sizes.pt"
    history_path = out_dir / "sapq_layerwise_history.json"
    results_path = out_dir / "sapq_layerwise_results.json"

    torch.save(
        {
            "step_sizes_dict": {
                name: (
                    pair[0].detach().cpu(),
                    pair[1].detach().cpu() if torch.is_tensor(pair[1]) else pair[1],
                )
                for name, pair in step_sizes_dict.items()
            },
            "meta": {
                "model_path": cfg.model_path,
                "dataset_name": cfg.dataset_name,
                "percentile_prob": float(cfg.percentile_prob),
                "init_bits": int(cfg.init_bits),
                "bmax_bits": int(cfg.bmax_bits),
                "target_bits": float(getattr(cfg, "target_bits", cfg.init_bits)),
                "sigma0": float(getattr(cfg, "sigma0", 0.5)),
                "alpha": float(getattr(cfg, "alpha", 1.0)),
                "prior_scale": float(getattr(cfg, "prior_scale", 1.0)),
                "num_mc_samples": int(cfg.num_mc_samples),
                "num_epochs": int(cfg.num_epochs),
                "updates_per_batch": int(cfg.updates_per_batch),
                "eta": float(cfg.eta),
                "prior_mode": str(getattr(cfg, "prior_mode", "block_sens")),
                "exp_name": cfg.prior_mode,
                "likelihood_mode": "layerwise",
                "model_name": Path(cfg.model_path).name,
                "dataset_tag": Path(cfg.data_path).name,
                "ranges_source": str(ranges_path) if "ranges_path" in locals() else "loaded_in_function",
                "frozen_calib_source": str(Path(cfg.repo_root) / "ppq_artifacts" / f"{Path(cfg.data_path).name}-calib" / "frozen_calibration_batches.pt"),
            },
        },
        steps_path,
    )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    with open(results_path, "w") as f:
        json.dump(
            {
                "FP": {"L1": fp_metrics["l1"], "RelL1": fp_metrics["rel_l1"]},
                "SAPQ-Layerwise": {
                    "L1": sapq_metrics["l1"],
                    "RelL1": sapq_metrics["rel_l1"],
                },
                "Dyn4": {"L1": dyn4_metrics["l1"], "RelL1": dyn4_metrics["rel_l1"]},
                "Dyn8": {"L1": dyn8_metrics["l1"], "RelL1": dyn8_metrics["rel_l1"]},
                "Dyn16": {"L1": dyn16_metrics["l1"], "RelL1": dyn16_metrics["rel_l1"]},
                "meta": {
                    "model_path": cfg.model_path,
                    "dataset_name": cfg.dataset_name,
                    "quant_layer_path": str(cfg.quant_layer_path),
                    "num_candidate_layers": len(candidate_layers),
                    "num_frozen_batches": len(frozen_batches),
                    "prior_mode": str(getattr(cfg, "prior_mode", "block_sens")),
                    "exp_name": cfg.prior_mode,
                    "likelihood_mode": "layerwise",
                },
            },
            f,
            indent=2,
        )

    print(f"\n[INFO] Saved step sizes -> {steps_path}")
    print(f"[INFO] Saved history    -> {history_path}")
    print(f"[INFO] Saved results    -> {results_path}")


if __name__ == "__main__":
    main()