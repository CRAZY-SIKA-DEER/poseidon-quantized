from __future__ import annotations
import numpy as np
import sys

import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from scOT.metrics import relative_lp_error, lp_error

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from PPQ.trainer import PPQTrainer
from PPQ.metrics import (
    evaluate_with_stepsizes,
    compute_avg_bits,
    compute_dynamic_stepsizes,
)

def load_candidate_linear_layers(model, quant_layer_path: Path):
    """
    Load candidate quantization layers from file, then keep only real Linear layers.
    """
    print(f"[INFO] Loading quantize layer list from: {quant_layer_path}")
    layer_data = torch.load(quant_layer_path)

    name2mod = dict(model.named_modules())
    candidate_layers = [
        name for name in layer_data["quantize_layers"]
        if isinstance(name2mod.get(name, None), nn.Linear)
    ]

    print(f"[INFO] {len(candidate_layers)} candidate Linear layers")
    return candidate_layers


def save_run_outputs(
    cfg: PPQConfig,
    gamma: float,
    history,
    ppq_epoch_evals,
    ppq_metrics,
    step_sizes_dict,
    final_avg_bits: float,
):
    """
    Save training history, eval results, and learned step sizes.
    """
    cfg.lr_dir.mkdir(parents=True, exist_ok=True)

    results_obj = {
        "gamma": float(gamma),
        "meta": {
            "model_path": cfg.model_path,
            "dataset_name": cfg.dataset_name,
            "eta": cfg.eta,
            "percentile_prob": cfg.percentile_prob,
            "num_epochs": cfg.num_epochs,
            "num_mc_samples": cfg.num_mc_samples,
            "init_bits": cfg.init_bits,
            "bmax_bits": cfg.bmax_bits,
            "base_lr": cfg.base_lr,
            "final_avg_bits": float(final_avg_bits),
        },
        "train_history": history,
        "ppq_epoch_evals": ppq_epoch_evals,
        "final_eval": {
            "ppq": ppq_metrics,
        },
    }

    result_file = cfg.lr_dir / f"PwC-T-gamma-{gamma:.0e}.json"
    with open(result_file, "w") as f:
        json.dump(results_obj, f, indent=2)
    print(f"[INFO] Saved results -> {result_file}")

    step_save_obj = {
        "step_sizes": {
            name: (w_step.detach().cpu(), a_step.detach().cpu())
            for name, (w_step, a_step) in step_sizes_dict.items()
        },
        "meta": {
            "model_path": cfg.model_path,
            "dataset_name": cfg.dataset_name,
            "gamma": float(gamma),
            "eta": cfg.eta,
            "percentile_prob": cfg.percentile_prob,
            "num_epochs": cfg.num_epochs,
            "num_mc_samples": cfg.num_mc_samples,
            "init_bits": cfg.init_bits,
            "bmax_bits": cfg.bmax_bits,
            "base_lr": cfg.base_lr,
            "final_avg_bits": float(final_avg_bits),
        },
    }

    step_pt_path = cfg.lr_dir / f"ppq_step_sizes-gamma-{gamma:.0e}.pt"
    torch.save(step_save_obj, step_pt_path)

    step_json_path = cfg.lr_dir / f"ppq_step_sizes-gamma-{gamma:.0e}.json"
    with open(step_json_path, "w") as f:
        json.dump(
            {
                "step_sizes": {
                    name: (w.cpu().tolist(), a.cpu().tolist())
                    for name, (w, a) in step_save_obj["step_sizes"].items()
                },
                "meta": step_save_obj["meta"],
            },
            f,
            indent=2,
        )

    print(f"[INFO] Saved step sizes -> {step_pt_path}")
    print(f"[INFO] Saved step sizes (JSON) -> {step_json_path}")



def load_dynamic_4_steps(dynamic_4_json_path: Path, device):
    """
    Load precomputed dynamic-4 step sizes from JSON.
    """
    with open(dynamic_4_json_path, "r") as f:
        dyn4_raw = json.load(f)["step_sizes"]

    dyn4_steps = {
        name: torch.tensor(step_list, dtype=torch.float32, device=device)
        for name, step_list in dyn4_raw.items()
    }
    return dyn4_steps



def evaluate_full_precision(model, val_loader, device):
    """
    Evaluate original full-precision model on the given validation iterator.
    Uses the same batches as the quantized evaluations.
    """
    model = model.to(device).eval()
    loader = val_loader() if callable(val_loader) else val_loader

    rel_l1_list = []
    abs_l1_list = []

    with torch.no_grad():
        for batch in loader:
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
            else:
                continue

            outputs = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            pred = outputs.output

            pred_np = pred.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            batch_rel = relative_lp_error(pred_np, y_np, p=1, return_percent=True)
            batch_abs = lp_error(pred_np, y_np, p=1)

            rel_l1_list.append(float(np.mean(batch_rel)))
            abs_l1_list.append(float(np.mean(batch_abs)))

    if len(rel_l1_list) == 0:
        return {"l1": float("nan"), "rel_l1": float("nan")}

    return {
        "l1": float(sum(abs_l1_list) / len(abs_l1_list)),
        "rel_l1": float(sum(rel_l1_list) / len(rel_l1_list)),
    }


def main():
    cfg = PPQConfig()

    # --------------------------------------------------
    # 1) Load model
    # --------------------------------------------------
    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    # --------------------------------------------------
    # 2) Build data loaders
    # --------------------------------------------------
    _calib_loader, _val_loader, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=cfg.val_batchsize,
        val_steps=cfg.val_steps,
    )

    # --------------------------------------------------
    # 3) Load candidate Linear layers
    # --------------------------------------------------
    candidate_layers = load_candidate_linear_layers(
        model=model,
        quant_layer_path=cfg.quant_layer_path,
    )


    # --------------------------------------------------
    # 3.5) Prepare dynamic baselines
    # --------------------------------------------------
    # dyn4_steps = load_dynamic_4_steps(cfg.dyn4_path, device=device)
    # dyn8_steps = compute_dynamic_stepsizes(
    #     model=model,
    #     layer_names=candidate_layers,
    #     num_bits=cfg.dyn8_bits,
    #     device=device,
    # )
    # dyn16_steps = compute_dynamic_stepsizes(
    #     model=model,
    #     layer_names=candidate_layers,
    #     num_bits=cfg.dyn16_bits,
    #     device=device,
    # )
    print("[INFO] Computing 8-bit and 16-bit dynamic step sizes...")
    dyn4_steps  = compute_dynamic_stepsizes(model, candidate_layers, num_bits=4,  device=device)
    dyn8_steps  = compute_dynamic_stepsizes(model, candidate_layers, num_bits=8,  device=device)
    dyn16_steps = compute_dynamic_stepsizes(model, candidate_layers, num_bits=16, device=device)

    # --------------------------------------------------
    # 4) Create trainer
    # --------------------------------------------------
    trainer = PPQTrainer(
        model=model,
        config=cfg,
        layer_names=candidate_layers,
        device=device,
    )

    # --------------------------------------------------
    # 5) Gamma sweep
    # --------------------------------------------------
    for gamma in cfg.gamma_list:
        print("\n" + "=" * 80)
        print(f"   STARTING PPQ TRAINING FOR gamma={gamma:.0e}")
        print("=" * 80)

        ppq_epoch_evals = {}

        def eval_callback(epoch, step_sizes_dict, ranges_dict):
            print(f"\n================ PPQ EVAL @ epoch {epoch} ================")

            ppq_metrics = evaluate_with_stepsizes(
                model=model,
                val_loader=val_iter,
                weight_steps=step_sizes_dict,
                act_steps=None,
                layer_names=candidate_layers,
                device=device,
            )

            ppq_epoch_evals[epoch] = ppq_metrics

            print(
                f"[PPQ-EPOCH-{epoch}] "
                f"L1={ppq_metrics['l1']:.6e} | RelL1={ppq_metrics['rel_l1']:.6e}"
            )

        step_sizes_dict, ranges_dict, history = trainer.train(
            dataloader=calib_iter,
            gamma=gamma,
            ranges_dict=None,
            eval_callback=eval_callback,
        )

        # --------------------------------------------------
        # 6) Final PPQ evaluation
        # --------------------------------------------------
                # --------------------------------------------------
        # 6) Final evaluations on the SAME val batches
        # --------------------------------------------------
        print("\n================ FINAL EVALUATION =================")

        fp_metrics = evaluate_full_precision(
            model=model,
            val_loader=val_iter,
            device=device,
        )

        ppq_metrics = evaluate_with_stepsizes(
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

        print(
            f"[FINAL] FP    | L1={fp_metrics['l1']:.6e} | RelL1={fp_metrics['rel_l1']:.6e}"
        )
        print(
            f"[FINAL] PPQ   | L1={ppq_metrics['l1']:.6e} | RelL1={ppq_metrics['rel_l1']:.6e}"
        )
        print(
            f"[FINAL] Dyn4  | L1={dyn4_metrics['l1']:.6e} | RelL1={dyn4_metrics['rel_l1']:.6e}"
        )
        print(
            f"[FINAL] Dyn8  | L1={dyn8_metrics['l1']:.6e} | RelL1={dyn8_metrics['rel_l1']:.6e}"
        )
        print(
            f"[FINAL] Dyn16 | L1={dyn16_metrics['l1']:.6e} | RelL1={dyn16_metrics['rel_l1']:.6e}"
        )

        with torch.no_grad():
            final_avg_bits = compute_avg_bits(
                step_sizes_dict=step_sizes_dict,
                ranges_dict=ranges_dict,
                channel_weights=trainer.channel_weights,
            )

        # NOTE:
        # The line above works, but it is a bit ugly / redundant.
        # It is only kept to avoid changing more files right now.
        # Later we should directly import compute_avg_bits at the top.

        save_run_outputs(
            cfg=cfg,
            gamma=gamma,
            history=history,
            ppq_epoch_evals=ppq_epoch_evals,
            ppq_metrics=ppq_metrics,
            step_sizes_dict=step_sizes_dict,
            final_avg_bits=final_avg_bits,
        )


if __name__ == "__main__":
    main()