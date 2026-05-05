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

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer


DATASETS = [
    ("NS-PwC-L", "NS-PwC", "fluids.incompressible.PiecewiseConstants"),
    ("NS-SVS-L", "NS-SVS", "fluids.incompressible.VortexSheet"),
    ("NS-BB-L",  "NS-BB",  "fluids.incompressible.BrownianBridge"),
]

TARGET_STEPS = [1, 5, 10]
BRECQ_ITERS = 10000
TRAJ_IDX = 0


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
            x = inp[0]
            w = mod.weight
            w_flat = w.view(w.size(0), -1)

            s = step.detach().to(w.device).view(-1, 1)
            wq = torch.round(w_flat / s) * s
            wq = wq.view_as(w)

            return torch.nn.functional.linear(x, wq, mod.bias)
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

    print(f"[INFO] registered quant hooks: {len(handles)}")
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


def norm_frame(x: torch.Tensor, constants):
    mean = torch.as_tensor(constants["mean"], dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std = torch.as_tensor(constants["std"], dtype=x.dtype, device=x.device).view(-1, 1, 1)
    return (x - mean) / std


def denorm_batch(x: torch.Tensor, constants):
    mean = torch.as_tensor(constants["mean"], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std = torch.as_tensor(constants["std"], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return x * std + mean


def read_ns_frame(ds, traj_idx: int, time_idx: int, device):
    """
    Return one normalized NS frame: [1, C, H, W]
    C = [rho, u, v, p]
    """
    real_i = ds.start + traj_idx

    vel = (
        torch.from_numpy(ds.reader["velocity"][real_i, time_idx, 0:2])
        .float()
        .reshape(2, ds.resolution, ds.resolution)
    )

    if getattr(ds, "transpose", False):
        vel = vel.transpose(-2, -1)

    density = torch.ones(1, ds.resolution, ds.resolution)
    pressure = torch.zeros(1, ds.resolution, ds.resolution)

    x = torch.cat([density, vel, pressure], dim=0)

    if getattr(ds, "res", None) is not None:
        x = ds._downsample(x, ds.res)

    x = norm_frame(x, ds.constants)
    return x.unsqueeze(0).to(device)


def build_gt_sequence(ds, traj_idx: int, max_steps: int, device):
    """
    GT sequence:
        step 0: raw time 0
        step 1: raw time 2
        step 2: raw time 4
        ...
    """
    frames = []
    dt = ds.time_step_size

    for k in range(max_steps + 1):
        frames.append(read_ns_frame(ds, traj_idx, k * dt, device))

    return torch.cat(frames, dim=0)  # [T+1, C, H, W]


@torch.no_grad()
def rollout_autoregressive(model, ds, traj_idx: int, max_steps: int, device):
    """
    Clean autoregressive rollout.

    No GT labels are passed into model.forward().
    Therefore no pixel_mask GT leakage happens.
    """
    model.eval()

    dt = ds.time_step_size
    time_value = dt / ds.constants["time"]
    time = torch.tensor([time_value], dtype=torch.float32, device=device)

    x = read_ns_frame(ds, traj_idx, 0, device)

    preds = []

    mask = ds.pixel_mask.bool().to(device)

    for _ in range(max_steps):
        out = model(
            pixel_values=x,
            time=time,
            pixel_mask=None,
            labels=None,
            return_dict=True,
        ).output

        # Keep known static/masked channels stable, e.g. pressure channel.
        # This avoids GT leakage but prevents feeding nonsense pressure back in.
        if mask.numel() == out.shape[1] and mask.any():
            out[:, mask, :, :] = x[:, mask, :, :]

        preds.append(out.detach().cpu())
        x = out.detach()

    preds = torch.cat(preds, dim=0)  # [T, C, H, W]
    return preds


def plot_rollout(all_preds, gt, save_dir: Path, tag: str, channel_idx: int):
    methods = ["BRECQ-w4", "BRECQ-w8", "PPQ", "SAPQ", "FP", "GT"]

    v = np.percentile(np.abs(gt[:, channel_idx]), 99)

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
                data = gt[step, channel_idx]
            else:
                data = all_preds[m][step - 1, channel_idx]

            ax.imshow(data.T, cmap="RdBu_r", vmin=-v, vmax=v)

            if r == 0:
                ax.set_title(m, fontweight=("bold" if m == "SAPQ" else "normal"))
            if c == 0:
                ax.set_ylabel(f"rollout step={step}")

            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"{tag}_rollout.png", dpi=200)
    plt.close()


def compute_rollout_errors(all_preds, gt, save_dir: Path, tag: str):
    """
    all_preds[m]: [T, C, H, W]
    gt:           [T+1, C, H, W]
    Compare pred step k with gt[k+1].
    """
    lines = []
    lines.append("method,step,l1,rel_l1_percent\n")

    print("\n========== Rollout L1 / RelL1 ==========")

    for method, pred in all_preds.items():
        print(f"\n[{method}]")

        for step in range(1, pred.shape[0] + 1):
            p = pred[step - 1]
            y = gt[step]

            l1 = np.mean(np.abs(p - y))
            rel_l1 = np.sum(np.abs(p - y)) / (np.sum(np.abs(y)) + 1e-12) * 100.0

            print(f"step={step:02d} | L1={l1:.6e} | RelL1={rel_l1:.4f}%")
            lines.append(f"{method},{step},{l1:.8e},{rel_l1:.6f}\n")

    out_csv = save_dir / f"{tag}_rollout_errors.csv"
    with open(out_csv, "w") as f:
        f.writelines(lines)

    print(f"\n[SAVED ERROR CSV] {out_csv}")

def plot_error_rollout(all_preds, gt, save_dir: Path, tag: str, channel_idx: int):
    methods = ["BRECQ-w4", "BRECQ-w8", "PPQ", "SAPQ", "FP"]

    # collect all errors for shared color scale
    all_errs = []
    for m in methods:
        for step in TARGET_STEPS:
            err = np.abs(all_preds[m][step - 1, channel_idx] - gt[step, channel_idx])
            all_errs.append(err)

    vmax = np.percentile(np.stack(all_errs), 99)

    fig, axes = plt.subplots(
        len(TARGET_STEPS),
        len(methods),
        figsize=(3 * len(methods), 3 * len(TARGET_STEPS)),
        squeeze=False,
    )

    for r, step in enumerate(TARGET_STEPS):
        for c, m in enumerate(methods):
            ax = axes[r, c]

            err = np.abs(all_preds[m][step - 1, channel_idx] - gt[step, channel_idx])

            im = ax.imshow(err.T, cmap="magma", vmin=0.0, vmax=vmax)

            if r == 0:
                ax.set_title(m, fontweight=("bold" if m == "SAPQ" else "normal"))
            if c == 0:
                ax.set_ylabel(f"rollout step={step}")

            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"{tag}_error_rollout.png", dpi=200)
    plt.close()

    print(f"[SAVED ERROR PLOT] {save_dir / f'{tag}_error_rollout.png'}")


def main():
    cfg = PPQConfig()
    root = Path(cfg.repo_root)
    max_steps = max(TARGET_STEPS)

    for model_tag, dataset_tag, dataset_name in DATASETS:
        print(f"\n========== {dataset_tag} ==========")

        cfg.model_path = str(root / "models" / model_tag)
        cfg.data_path = str(root / "dataset" / dataset_tag)
        cfg.dataset_name = dataset_name

        save_dir = root / "zzzz_plot" / model_tag / dataset_tag / "rollout_correct"
        save_dir.mkdir(parents=True, exist_ok=True)

        ds = get_dataset(
            dataset_name,
            which="val",
            num_trajectories=256,
            data_path=cfg.data_path,
            max_num_time_steps=10,
            time_step_size=2,
        )

        if max_steps > ds.max_num_time_steps:
            raise ValueError(
                f"TARGET_STEPS max={max_steps}, but dataset only supports "
                f"{ds.max_num_time_steps} rollout steps."
            )

        model, device = load_poseidon_model(cfg.model_path, cfg.device)
        layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

        gt_norm = build_gt_sequence(ds, TRAJ_IDX, max_steps, device)
        gt_denorm = denorm_batch(gt_norm, ds.constants).cpu().numpy()

        all_preds = {}

        # FP
        pred = rollout_autoregressive(model, ds, TRAJ_IDX, max_steps, device)
        all_preds["FP"] = denorm_batch(pred, ds.constants).numpy()

        # SAPQ
        sapq_path = (
            root / "sapq_experiments" / model_tag / dataset_tag
            / "network_block_sens_sobo" / "raw" / "sapq_global_step_sizes.pt"
        )
        sapq_steps = load_saved_steps(sapq_path)

        h = register_hooks(model, layers, sapq_steps)
        pred = rollout_autoregressive(model, ds, TRAJ_IDX, max_steps, device)
        all_preds["SAPQ"] = denorm_batch(pred, ds.constants).numpy()
        for hh in h:
            hh.remove()

        # PPQ
        ppq_path = (
            root / "ppq_experiments" / model_tag / dataset_tag
            / "layerwise_ppq" / "sapq_layerwise_step_sizes.pt"
        )
        ppq_steps = load_saved_steps(ppq_path)

        h = register_hooks(model, layers, ppq_steps)
        pred = rollout_autoregressive(model, ds, TRAJ_IDX, max_steps, device)
        all_preds["PPQ"] = denorm_batch(pred, ds.constants).numpy()
        for hh in h:
            hh.remove()

        # BRECQ-w4
        p = (
            root / "brecq_artifacts" / model_tag / "recon" / "w4"
            / f"iters{BRECQ_ITERS}" / "adaround_state.pt"
        )
        m, _ = load_poseidon_model(cfg.model_path, cfg.device)
        brecq = build_brecq_model(m, p, device, bits=4)
        pred = rollout_autoregressive(brecq, ds, TRAJ_IDX, max_steps, device)
        all_preds["BRECQ-w4"] = denorm_batch(pred, ds.constants).numpy()

        # BRECQ-w8
        p = (
            root / "brecq_artifacts" / model_tag / "recon" / "w8"
            / f"iters{BRECQ_ITERS}" / "adaround_state.pt"
        )
        m, _ = load_poseidon_model(cfg.model_path, cfg.device)
        brecq = build_brecq_model(m, p, device, bits=8)
        pred = rollout_autoregressive(brecq, ds, TRAJ_IDX, max_steps, device)
        all_preds["BRECQ-w8"] = denorm_batch(pred, ds.constants).numpy()

        np.savez(save_dir / "rollout_preds_correct.npz", GT=gt_denorm, **all_preds)

        plot_rollout(all_preds, gt_denorm, save_dir, f"{dataset_tag}_u", channel_idx=1)
        plot_rollout(all_preds, gt_denorm, save_dir, f"{dataset_tag}_v", channel_idx=2)

        compute_rollout_errors(all_preds, gt_denorm, save_dir, dataset_tag)

        plot_error_rollout(all_preds, gt_denorm, save_dir, f"{dataset_tag}_u", channel_idx=1)
        plot_error_rollout(all_preds, gt_denorm, save_dir, f"{dataset_tag}_v", channel_idx=2)

        print(f"[SAVED] {save_dir}")


if __name__ == "__main__":
    main()