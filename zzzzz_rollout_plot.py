from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model
from SAPQ.run_sapq_network_global import load_candidate_layers

from scOT.problems.base import get_dataset
from scOT.inference import get_trajectories

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer


DATASETS = [
    ("NS-PwC-L", "NS-PwC", "fluids.incompressible.PiecewiseConstants"),
    ("NS-SVS-L", "NS-SVS", "fluids.incompressible.VortexSheet"),
    ("NS-BB-L",  "NS-BB",  "fluids.incompressible.BrownianBridge"),
]

INITIAL_TIME = 0
FINAL_TIME = 20
AR_STEPS = 4          # 0->5->10->15->20
BRECQ_ITERS = 1000
TARGET_STEPS = [5, 10, 15, 20]


def load_saved_steps(path: Path):
    obj = torch.load(path, map_location="cpu")
    return obj["step_sizes_dict"] if isinstance(obj, dict) and "step_sizes_dict" in obj else obj


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

    qnn = PoseidonQuantModel(fp_model, wq_params, aq_params).to(device).eval()
    state = torch.load(adaround_path, map_location="cpu")

    for name, m in qnn.model.named_modules():
        if not isinstance(m, QuantModule) or name not in state:
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


def denorm(x, constants):
    mean = torch.as_tensor(constants["mean"], dtype=x.dtype, device=x.device).flatten()
    std = torch.as_tensor(constants["std"], dtype=x.dtype, device=x.device).flatten()

    if x.ndim == 5:
        mean = mean.view(1, 1, -1, 1, 1)
        std = std.view(1, 1, -1, 1, 1)
    elif x.ndim == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

    return x * std + mean


def make_rollout_dataset(dataset_name, data_path, delta_t):
    return get_dataset(
        dataset=dataset_name,
        which="test",
        num_trajectories=1,
        data_path=data_path,
        fix_input_to_time_step=INITIAL_TIME,
        time_step_size=delta_t,
        max_num_time_steps=1,
    )


@torch.no_grad()
def autoregressive_rollout(model, rollout_ds, gt_seq, device):
    """
    gt_seq: [N, AR_STEPS, C, H, W], normalized
    output: [N, AR_STEPS, C, H, W], normalized
    """
    model.eval()

    sample = rollout_ds[0]
    x = sample["pixel_values"].unsqueeze(0).to(device)
    time = torch.as_tensor(sample["time"]).view(1).to(device)
    pixel_mask = sample["pixel_mask"].unsqueeze(0).to(device)

    preds = []

    for step in range(AR_STEPS):
        label_for_mask = gt_seq[:, step].to(device)

        out = model(
            pixel_values=x,
            time=time,
            pixel_mask=pixel_mask,
            labels=label_for_mask,
            return_dict=True,
        ).output

        preds.append(out.detach().cpu())
        x = out.detach()

    return torch.stack(preds, dim=1)


def plot_field(all_preds, gt, save_dir, dataset_tag, field_name, ch):
    methods = ["BRECQ-w4", "BRECQ-w8", "PPQ", "SAPQ", "FP", "GT"]

    vmin = np.percentile(gt[0, :, ch], 1)
    vmax = np.percentile(gt[0, :, ch], 99)

    fig, axes = plt.subplots(
        len(TARGET_STEPS),
        len(methods),
        figsize=(3.2 * len(methods), 3.2 * len(TARGET_STEPS)),
        squeeze=False,
    )

    for r, t in enumerate(TARGET_STEPS):
        for c, m in enumerate(methods):
            ax = axes[r, c]

            if m == "GT":
                data = gt[0, r, ch]
            else:
                data = all_preds[m][0, r, ch]

            im = ax.imshow(data.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)

            if r == 0:
                ax.set_title(m, fontweight=("bold" if m == "SAPQ" else "normal"))
            if c == 0:
                ax.set_ylabel(f"t={t}")

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{dataset_tag} - {field_name}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 0.94, 0.96])

    cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    out_path = save_dir / f"{dataset_tag}_{field_name}_rollout.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = PPQConfig()
    root = Path(cfg.repo_root)

    delta_t = (FINAL_TIME - INITIAL_TIME) // AR_STEPS

    for model_tag, dataset_tag, dataset_name in DATASETS:
        print(f"\n==== {dataset_tag} ====")

        model_path = root / "models" / model_tag
        data_path = root / "dataset" / dataset_tag
        save_dir = root / "zzzzz_plot" / dataset_tag
        save_dir.mkdir(parents=True, exist_ok=True)

        rollout_ds = make_rollout_dataset(dataset_name, str(data_path), delta_t)
        constants = rollout_ds.constants

        gt_seq = get_trajectories(
            dataset=dataset_name,
            data_path=str(data_path),
            ar_steps=AR_STEPS,
            initial_time=INITIAL_TIME,
            final_time=FINAL_TIME,
            dataset_kwargs={},
        )[:1]  # [1, AR_STEPS, C, H, W]

        model, device = load_poseidon_model(str(model_path), cfg.device)
        layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

        all_preds = {}

        pred = autoregressive_rollout(model, rollout_ds, gt_seq, device)
        all_preds["FP"] = denorm(pred, constants).numpy()

        sapq_steps = load_saved_steps(
            root / "sapq_experiments" / model_tag / dataset_tag /
            "network_block_sens_sobo" / "raw" / "sapq_global_step_sizes.pt"
        )
        h = register_hooks(model, layers, sapq_steps)
        pred = autoregressive_rollout(model, rollout_ds, gt_seq, device)
        all_preds["SAPQ"] = denorm(pred, constants).numpy()
        for hh in h:
            hh.remove()

        ppq_steps = load_saved_steps(
            root / "ppq_experiments" / model_tag / dataset_tag /
            "layerwise_ppq" / "sapq_layerwise_step_sizes.pt"
        )
        h = register_hooks(model, layers, ppq_steps)
        pred = autoregressive_rollout(model, rollout_ds, gt_seq, device)
        all_preds["PPQ"] = denorm(pred, constants).numpy()
        for hh in h:
            hh.remove()

        p = root / "brecq_artifacts" / model_tag / "recon" / "w4" / f"iters{BRECQ_ITERS}" / "adaround_state.pt"
        m, _ = load_poseidon_model(str(model_path), cfg.device)
        brecq = build_brecq_model(m, p, device, 4)
        pred = autoregressive_rollout(brecq, rollout_ds, gt_seq, device)
        all_preds["BRECQ-w4"] = denorm(pred, constants).numpy()

        p = root / "brecq_artifacts" / model_tag / "recon" / "w8" / f"iters{BRECQ_ITERS}" / "adaround_state.pt"
        m, _ = load_poseidon_model(str(model_path), cfg.device)
        brecq = build_brecq_model(m, p, device, 8)
        pred = autoregressive_rollout(brecq, rollout_ds, gt_seq, device)
        all_preds["BRECQ-w8"] = denorm(pred, constants).numpy()

        gt_denorm = denorm(gt_seq, constants).numpy()

        np.savez(save_dir / "rollout_preds.npz", GT=gt_denorm, **all_preds)

        # NS channels: [rho, u, v, p]
        plot_field(all_preds, gt_denorm, save_dir, dataset_tag, "u", ch=1)
        plot_field(all_preds, gt_denorm, save_dir, dataset_tag, "v", ch=2)

        print(f"Saved -> {save_dir}")


if __name__ == "__main__":
    main()