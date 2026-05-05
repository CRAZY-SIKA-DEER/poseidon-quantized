# zzzz_plot_rollout_poseidon.py
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
BRECQ_ITERS = 10000


def load_saved_steps(path: Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "step_sizes_dict" in obj:
        return obj["step_sizes_dict"]
    return obj


def register_hooks(model, layers, steps):
    name2mod = dict(model.named_modules())
    handles = []

    def make_hook(step):
        def hook(mod, inp, out):
            w = mod.weight
            w_flat = w.view(w.size(0), -1)
            s = step.view(-1, 1).to(w.device)
            wq = torch.round(w_flat / s) * s
            return torch.nn.functional.linear(inp[0], wq.view_as(w), mod.bias)
        return hook

    for name in layers:
        if name not in steps:
            continue
        mod = name2mod.get(name, None)
        if not isinstance(mod, nn.Linear):
            continue

        item = steps[name]
        step = item[0] if isinstance(item, (tuple, list)) else item
        handles.append(mod.register_forward_hook(make_hook(step)))

    return handles


def build_brecq_model(fp_model, adaround_path: Path, device, bits: int):
    wq_params = dict(n_bits=bits, channel_wise=True, scale_method="max")
    aq_params = dict(n_bits=8, channel_wise=False, scale_method="max", leaf_param=False)

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


def denorm_ns(x: torch.Tensor, constants):
    """
    x: [B,T,C,H,W] or [B,C,H,W]
    """
    mean = torch.as_tensor(constants["mean"], dtype=x.dtype, device=x.device).flatten()
    std = torch.as_tensor(constants["std"], dtype=x.dtype, device=x.device).flatten()

    if x.ndim == 5:
        mean = mean.view(1, 1, -1, 1, 1)
        std = std.view(1, 1, -1, 1, 1)
    elif x.ndim == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

    return x * std + mean


def collect_future_batches(val_iter_fn, T: int, device):
    """
    Collect consecutive one-step samples:
        batch 0 label = GT at t+1
        batch 1 label = GT at t+2
        ...
    """
    batches = []
    it = val_iter_fn()

    for _ in range(T):
        b = next(it)
        clean = {}
        for k, v in b.items():
            clean[k] = v.to(device) if torch.is_tensor(v) else v
        batches.append(clean)

    return batches


@torch.no_grad()
def rollout_autoregressive(model, future_batches, T: int, device):
    """
    Autoregressive rollout:
        x_0 = first pixel_values
        pred_1 = model(x_0)
        x_1 = pred_1
        pred_2 = model(x_1)
        ...

    GT:
        y_k = labels from future_batches[k]
    """
    model.eval()

    x = future_batches[0]["pixel_values"].to(device)
    preds = []
    gts = []

    for k in range(T):
        b = future_batches[k]

        t = b.get("time", None)
        pm = b.get("pixel_mask", None)
        y_gt = b["labels"].to(device)

        if t is not None:
            t = t.to(device)
        if pm is not None:
            pm = pm.to(device)

        # Important:
        # ScOT.forward uses labels inside pixel_mask replacement:
        # prediction[pixel_mask] = labels[pixel_mask]
        labels_for_forward = y_gt if pm is not None else None

        out = model(
            pixel_values=x,
            time=t,
            pixel_mask=pm,
            labels=labels_for_forward,
            return_dict=True,
        ).output

        preds.append(out.detach().cpu())
        gts.append(y_gt.detach().cpu())

        # autoregressive input for next step
        x = out.detach()

    preds = torch.stack(preds, dim=1)  # [B,T,C,H,W]
    gts = torch.stack(gts, dim=1)      # [B,T,C,H,W]
    return preds, gts


def plot_rollout(all_preds, gt, save_dir: Path, tag: str, channel_idx: int = 1):
    methods = ["BRECQ-w4", "BRECQ-w8", "PPQ", "SAPQ", "FP", "GT"]

    v = np.percentile(np.abs(gt[0, :, channel_idx]), 99)

    fig, axes = plt.subplots(
        len(TARGET_STEPS),
        len(methods),
        figsize=(3 * len(methods), 3 * len(TARGET_STEPS)),
        squeeze=False,
    )

    for r, step in enumerate(TARGET_STEPS):
        for c, m in enumerate(methods):
            ax = axes[r, c]

            if m == "GT":
                data = gt[0, step - 1, channel_idx]
            else:
                data = all_preds[m][0, step - 1, channel_idx]

            ax.imshow(data.T, cmap="RdBu_r", vmin=-v, vmax=v)

            if r == 0:
                ax.set_title(m, fontweight=("bold" if m == "SAPQ" else "normal"))
            if c == 0:
                ax.set_ylabel(f"step={step}")

            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"{tag}_rollout.png", dpi=200)
    plt.close()


def main():
    cfg = PPQConfig()
    root = Path(cfg.repo_root)
    max_T = max(TARGET_STEPS)

    for model_tag, dataset_tag, dataset_name in DATASETS:
        print(f"\n==== {dataset_tag} ====")

        cfg.model_path = str(root / "models" / model_tag)
        cfg.data_path = str(root / "dataset" / dataset_tag)
        cfg.dataset_name = dataset_name

        save_dir = root / "zzzz_plot" / model_tag / dataset_tag / "rollout"
        save_dir.mkdir(parents=True, exist_ok=True)

        model, device = load_poseidon_model(cfg.model_path, cfg.device)

        _, val_loader, _, val_iter = build_poseidon_loaders(
            dataset_name=dataset_name,
            data_path=cfg.data_path,
            val_batchsize=1,
            val_steps=max_T,
            calib_batchsize=1,
            calib_steps=1,
        )

        constants = val_loader.dataset.constants
        layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

        future_batches = collect_future_batches(val_iter, max_T, device)

        print("First batch keys:", future_batches[0].keys())
        print("pixel_values:", future_batches[0]["pixel_values"].shape)
        print("labels:", future_batches[0]["labels"].shape)
        if "time" in future_batches[0]:
            print("first time:", future_batches[0]["time"])

        sapq_steps = load_saved_steps(
            root / "sapq_experiments" / model_tag / dataset_tag /
            "network_block_sens_sobo" / "raw" / "sapq_global_step_sizes.pt"
        )

        ppq_steps = load_saved_steps(
            root / "ppq_experiments" / model_tag / dataset_tag /
            "layerwise_ppq" / "sapq_layerwise_step_sizes.pt"
        )

        all_preds = {}

        # FP
        pred, gt = rollout_autoregressive(model, future_batches, max_T, device)
        all_preds["FP"] = denorm_ns(pred, constants).numpy()
        gt_denorm = denorm_ns(gt, constants).numpy()

        # SAPQ
        h = register_hooks(model, layers, sapq_steps)
        pred, _ = rollout_autoregressive(model, future_batches, max_T, device)
        all_preds["SAPQ"] = denorm_ns(pred, constants).numpy()
        for hh in h:
            hh.remove()

        # PPQ
        h = register_hooks(model, layers, ppq_steps)
        pred, _ = rollout_autoregressive(model, future_batches, max_T, device)
        all_preds["PPQ"] = denorm_ns(pred, constants).numpy()
        for hh in h:
            hh.remove()

        # BRECQ-w4
        p = root / "brecq_artifacts" / model_tag / "recon" / "w4" / f"iters{BRECQ_ITERS}" / "adaround_state.pt"
        m, _ = load_poseidon_model(cfg.model_path, cfg.device)
        brecq = build_brecq_model(m, p, device, bits=4)
        pred, _ = rollout_autoregressive(brecq, future_batches, max_T, device)
        all_preds["BRECQ-w4"] = denorm_ns(pred, constants).numpy()

        # BRECQ-w8
        p = root / "brecq_artifacts" / model_tag / "recon" / "w8" / f"iters{BRECQ_ITERS}" / "adaround_state.pt"
        m, _ = load_poseidon_model(cfg.model_path, cfg.device)
        brecq = build_brecq_model(m, p, device, bits=8)
        pred, _ = rollout_autoregressive(brecq, future_batches, max_T, device)
        all_preds["BRECQ-w8"] = denorm_ns(pred, constants).numpy()

        np.savez(save_dir / "rollout_preds.npz", GT=gt_denorm, **all_preds)

        plot_rollout(all_preds, gt_denorm, save_dir, f"{dataset_tag}_u", channel_idx=1)
        plot_rollout(all_preds, gt_denorm, save_dir, f"{dataset_tag}_v", channel_idx=2)

        print(f"Saved -> {save_dir}")


if __name__ == "__main__":
    main()