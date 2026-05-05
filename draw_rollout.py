from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from SAPQ.run_sapq_network_global import load_candidate_layers

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer


DATASETS = [
    ("NS-PwC-L", "NS-PwC", "fluids.incompressible.PiecewiseConstants"),
    ("NS-SVS-L", "NS-SVS", "fluids.incompressible.VortexSheet"),
    ("NS-BB-L",  "NS-BB",  "fluids.incompressible.BrownianBridge"),
]

TARGET_STEPS = [1, 5, 10, 15, 20]
BRECQ_ITERS = 1000


# =========================
# Utils
# =========================
def load_saved_steps(path):
    obj = torch.load(path, map_location="cpu")
    return obj["step_sizes_dict"] if "step_sizes_dict" in obj else obj


def build_brecq_model(fp_model, adaround_path, device, bits):
    wq_params = dict(n_bits=bits, channel_wise=True, scale_method="max")
    aq_params = dict(n_bits=8, channel_wise=False, scale_method="max")

    qnn = PoseidonQuantModel(fp_model, wq_params, aq_params).to(device)
    qnn.eval()

    state = torch.load(adaround_path, map_location="cpu")

    for name, m in qnn.model.named_modules():
        if not isinstance(m, QuantModule):
            continue
        if name not in state:
            continue

        item = state[name]
        q = m.weight_quantizer
        q.delta = item["delta"].to(device)
        q.zero_point = item["zero_point"].to(device)
        q.inited = True

        if "alpha" in item:
            ada = AdaRoundQuantizer(
                uaq=q,
                round_mode="learned_hard_sigmoid",
                weight_tensor=m.org_weight.data,
            )
            ada.alpha.data.copy_(item["alpha"].to(device))
            ada.soft_targets = False
            m.weight_quantizer = ada

    qnn.set_quant_state(True, False)
    return qnn


def register_hooks(model, layers, steps):
    name2mod = dict(model.named_modules())
    handles = []

    def hook(step):
        def f(mod, inp, out):
            w = mod.weight
            w_flat = w.view(w.size(0), -1)
            s = step.view(-1, 1).to(w.device)
            wq = torch.round(w_flat / s) * s
            return torch.nn.functional.linear(inp[0], wq.view_as(w), mod.bias)
        return f

    for name in layers:
        if name not in steps:
            continue
        mod = name2mod[name]
        step = steps[name][0] if isinstance(steps[name], (tuple, list)) else steps[name]
        handles.append(mod.register_forward_hook(hook(step)))

    return handles


def denorm(x, constants):
    mean = torch.tensor(constants["mean"]).view(1, 1, -1, 1, 1)
    std = torch.tensor(constants["std"]).view(1, 1, -1, 1, 1)
    return x * std + mean


# =========================
# Rollout
# =========================
@torch.no_grad()
def rollout(model, batch, device, T):
    x = batch["pixel_values"].to(device)
    y = batch["labels"].to(device)
    t = batch.get("time", None)
    pm = batch.get("pixel_mask", None)

    if t is not None: t = t.to(device)
    if pm is not None: pm = pm.to(device)

    preds = []

    for _ in range(T):
        out = model(
            pixel_values=x,
            time=t,
            pixel_mask=pm,
            labels=y,
            return_dict=True,
        ).output

        preds.append(out.cpu())
        x = out.detach()
        y = out.detach()

    return torch.stack(preds, dim=1)


# =========================
# Plot
# =========================
def plot(all_preds, save_dir, tag):
    methods = ["BRECQ-w4", "BRECQ-w8", "PPQ", "SAPQ", "FP", "GT"]
    gt = all_preds["GT"]

    v = np.percentile(np.abs(gt[0]), 99)

    fig, axes = plt.subplots(len(TARGET_STEPS), len(methods),
                             figsize=(3*len(methods), 3*len(TARGET_STEPS)))

    for r, step in enumerate(TARGET_STEPS):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            data = all_preds[m][0, step-1, 1]

            im = ax.imshow(data.T, cmap="RdBu_r", vmin=-v, vmax=v)

            if r == 0:
                fw = "bold" if m == "SAPQ" else "normal"
                ax.set_title(m, fontweight=fw)

            if c == 0:
                ax.set_ylabel(f"t={step}")

            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"{tag}.png", dpi=200)
    plt.close()


# =========================
# Main
# =========================
def main():
    cfg = PPQConfig()
    root = Path(cfg.repo_root)

    for model_tag, dataset_tag, dataset_name in DATASETS:
        print(f"\n==== {dataset_tag} ====")

        cfg.model_path = str(root / "models" / model_tag)
        cfg.data_path = str(root / "dataset" / dataset_tag)
        cfg.dataset_name = dataset_name

        save_dir = root / "zzzz_plot" / model_tag / dataset_tag
        save_dir.mkdir(parents=True, exist_ok=True)

        model, device = load_poseidon_model(cfg.model_path, cfg.device)

        _, val_loader, _, val_iter = build_poseidon_loaders(
            dataset_name=dataset_name,
            data_path=cfg.data_path,
            val_batchsize=1,
            val_steps=1,
            calib_batchsize=1,
            calib_steps=1,
        )

        batch = next(val_iter())
        constants = val_loader.dataset.constants
        layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

        sapq_steps = load_saved_steps(
            root / "sapq_experiments" / model_tag / dataset_tag /
            "network_block_sens_sobo/raw/sapq_global_step_sizes.pt"
        )

        ppq_steps = load_saved_steps(
            root / "ppq_experiments" / model_tag / dataset_tag /
            "layerwise_ppq/sapq_layerwise_step_sizes.pt"
        )

        max_T = max(TARGET_STEPS)
        all_preds = {}

        # FP
        all_preds["FP"] = denorm(rollout(model, batch, device, max_T), constants).numpy()

        # SAPQ
        h = register_hooks(model, layers, sapq_steps)
        all_preds["SAPQ"] = denorm(rollout(model, batch, device, max_T), constants).numpy()
        [x.remove() for x in h]

        # PPQ
        h = register_hooks(model, layers, ppq_steps)
        all_preds["PPQ"] = denorm(rollout(model, batch, device, max_T), constants).numpy()
        [x.remove() for x in h]

        # BRECQ w4
        p = root / "brecq_artifacts" / model_tag / "recon/w4/iters1000/adaround_state.pt"
        m,_ = load_poseidon_model(cfg.model_path, cfg.device)
        all_preds["BRECQ-w4"] = denorm(rollout(build_brecq_model(m, p, device, 4), batch, device, max_T), constants).numpy()

        # BRECQ w8
        p = root / "brecq_artifacts" / model_tag / "recon/w8/iters1000/adaround_state.pt"
        m,_ = load_poseidon_model(cfg.model_path, cfg.device)
        all_preds["BRECQ-w8"] = denorm(rollout(build_brecq_model(m, p, device, 8), batch, device, max_T), constants).numpy()

        # GT
        y = batch["labels"].cpu()
        all_preds["GT"] = denorm(y.unsqueeze(1).repeat(1, max_T, 1, 1, 1), constants).numpy()

        np.savez(save_dir / "preds.npz", **all_preds)
        plot(all_preds, save_dir, dataset_tag)

        # ✅ SAVE (MUST be here, inside loop)
        np.savez(save_dir / "preds.npz", **all_preds)

        # ✅ PLOT (MUST be here, SAME indentation as np.savez)
        plot(all_preds, save_dir, dataset_tag)


if __name__ == "__main__":
    main()