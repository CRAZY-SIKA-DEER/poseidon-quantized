from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from SAPQ.run_sapq_network_global import load_candidate_layers

# 🔥 use your friend's code (DO NOT rewrite metrics)
from PDE_calc import IncompressibleLoss
from PPQ.metrics import evaluate_with_stepsizes

# =============================
# utils
# =============================
def mean_dict(lst):
    acc = defaultdict(list)
    for d in lst:
        for k, v in d.items():
            acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# =============================
# evaluation (same as before, only metrics changed)
# =============================
def eval_model(model, val_iter, layer_names, steps, constants, device):
    model = model.to(device).eval()
    name2mod = dict(model.named_modules())
    handles = []

    def make_hook(step):
        def hook(mod, inp, out):
            w = mod.weight
            w_flat = w.view(w.size(0), -1)
            s = step.view(-1, 1).to(w.device)
            wq = torch.round(w_flat / s) * s
            wq = wq.view_as(w)
            return torch.nn.functional.linear(inp[0], wq, mod.bias)
        return hook

    # register quant hooks (same as old code)
    for n in layer_names:
        if n in steps and isinstance(name2mod.get(n), nn.Linear):
            s = steps[n][0] if isinstance(steps[n], (list, tuple)) else steps[n]
            handles.append(name2mod[n].register_forward_hook(make_hook(s)))

    all_metrics = []

    with torch.no_grad():
        for batch in val_iter():
            x = batch["pixel_values"].to(device)
            t = batch["time"].to(device)
            pm = batch["pixel_mask"].to(device)
            y = batch["labels"].to(device)

            out = model(pixel_values=x, time=t, pixel_mask=pm, labels=y).output

            # 🔥 use friend's implementation
            loss_obj = IncompressibleLoss(
                preds=out,
                labels=y,
                constants=constants,
                transpose=False,
                denorm=True,
            )

            metrics = loss_obj.compute()

            # 🔥 ONLY keep required metrics
            all_metrics.append({
                "continuity": metrics["continuity"],
                "rel_continuity": metrics["rel_continuity"],
                "vorticity": metrics["vorticity_err"],
                "rel_vorticity": metrics["rel_vorticity_err"],
                "sobo_s01": metrics["sobolev_s01"],
                "sobo_s012": metrics["sobolev_s012"],
            })

    for h in handles:
        h.remove()

    return mean_dict(all_metrics)


# =============================
# MAIN
# =============================
def main():
    cfg = PPQConfig()

    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    _, val_loader, _, val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        val_batchsize=cfg.val_batchsize,
        val_steps=cfg.val_steps,
    )

    constants = val_loader.dataset.constants
    layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

    root = Path(cfg.repo_root)

    # ----------------------------
    # step paths (FIXED)
    # ----------------------------
    step_paths = {
        # global
        "network_block_sens_sobo":
            root / "SAPQ/artifacts_global/NS-PwC-L/network_block_sens_sobo/sapq_global_step_sizes.pt",
        "network_block_sens":
            root / "SAPQ/artifacts_global/NS-PwC-L/network_block_sens/sapq_global_step_sizes.pt",
        "network_block_no_sens":
            root / "SAPQ/artifacts_global/NS-PwC-L/network_block_no_sens/sapq_global_step_sizes.pt",

        # block
        "block_ppq":
            root / "SAPQ/artifacts_block/NS-PwC-L/ppq/sapq_step_sizes.pt",
        "block_sens":
            root / "SAPQ/artifacts_block/NS-PwC-L/block_sens/sapq_step_sizes.pt",
        "block_no_sens":
            root / "SAPQ/artifacts_block/NS-PwC-L/block_no_sens/sapq_step_sizes.pt",

        # layer
        "layer_ppq":
            root / "SAPQ/artifacts_layerwise/NS-PwC-L/ppq/sapq_layerwise_step_sizes.pt",
    }

    results = {}

    # ============================
    # FP
    # ============================
    print("Evaluating FP...")

    fp_l1 = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps={},
        act_steps=None,
        layer_names=layers,
        device=device,
    )

    fp_phys = eval_model(
        model, val_iter, layers,
        steps={},
        constants=constants,
        device=device
    )

    results["FP"] = {
        "L1": fp_l1["l1"],
        "RelL1": fp_l1["rel_l1"],
        **fp_phys,
    }

    # ============================
    # Quantized
    # ============================
    for name, path in step_paths.items():

        if not path.exists():
            print(f"[WARN] skip {name}, no file")
            continue

        print(f"Evaluating {name}...")

        obj = torch.load(path, map_location="cpu")

        # unify format
        if "step_sizes_dict" in obj:
            steps = obj["step_sizes_dict"]
        else:
            steps = obj

        # ---- L1 ----
        q_l1 = evaluate_with_stepsizes(
            model=model,
            val_loader=val_iter,
            weight_steps=steps,
            act_steps=None,
            layer_names=layers,
            device=device,
        )

        # ---- physical ----
        q_phys = eval_model(
            model, val_iter, layers,
            steps=steps,
            constants=constants,
            device=device
        )

        results[name] = {
            "L1": q_l1["l1"],
            "RelL1": q_l1["rel_l1"],
            **q_phys,
        }

    # ============================
    # save
    # ============================
    save_path = root / "SAPQ/eval_compare_results.json"

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    # ============================
    # print
    # ============================
    print("\n========== RESULTS ==========")
    for k, v in results.items():
        print(f"\n[{k}]")
        for kk, vv in v.items():
            print(f"{kk}: {vv:.6e}")

    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()