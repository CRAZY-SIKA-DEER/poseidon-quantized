from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model
from SAPQ.run_sapq_network_global import load_candidate_layers

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer

from scOT.inference import get_trainer, rollout, get_test_set
from scOT.metrics import relative_lp_error, lp_error


# ============================================================
# CONFIG
# ============================================================

ROLLOUT_STEPS = [4]

DEFAULT_SBPQ_RUN_DIR = Path(
    "SBPQ/artifacts/poseidon/NS-PwC-L/runs/"
    "network_global_groupall_datasets_near_best_lks_B8_d2_k150_ps1_mc10_"
    "eta1em06_lr3p75em05_init8_sob2_sw1_cal512_val2_steps800_"
    "sens-sob2_sw1_snl1_tw1_sow1_cal512_sensb512"
)

BRECQ_BITS_LIST = [4]
BRECQ_ITERS_LIST = [10000]

UNIFORM_BITS_LIST = [8, 4]


# ============================================================
# HELPERS
# ============================================================

def load_saved_steps(path: Path):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "step_sizes_dict" in obj:
        return obj["step_sizes_dict"], obj.get("meta", {})

    if isinstance(obj, dict):
        return obj, {}

    raise ValueError(f"Unsupported step-size file format: {path}")


def load_sbpq_step_sizes(run_dir: Path):
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

    steps = {
        layer_name: torch.as_tensor(step_size).detach().cpu()
        for layer_name, step_size in checkpoint[
            "optimized_step_sizes"
        ].items()
    }
    meta = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "parameter_weighted_average_bits": checkpoint.get(
            "parameter_weighted_average_bits"
        ),
        "unweighted_average_bits": checkpoint.get(
            "unweighted_average_bits"
        ),
    }
    return steps, meta


def mean_dict(records):
    acc = defaultdict(list)

    for rec in records:
        for k, v in rec.items():
            acc[k].append(v)

    return {k: float(np.mean(v)) for k, v in acc.items()}


# ============================================================
# QUANT HOOKS
# ============================================================

def register_quant_hooks(model, layer_names, steps):
    name2mod = dict(model.named_modules())
    handles = []
    debug_counter = {"n": 0}

    def make_hook(item):
        def hook(mod, inp, out):
            if debug_counter["n"] < 5:
                print("[DEBUG] QUANT HOOK FIRED:", mod.__class__.__name__)
                debug_counter["n"] += 1

            x = inp[0]

            w = mod.weight
            w_flat = w.view(w.size(0), -1)

            # affine format
            if isinstance(item, dict):
                step = item["step"].to(w.device).view(-1, 1)

                zero = item["zero"].to(w.device).view(-1, 1)

                qmin = item["qmin"]
                qmax = item["qmax"]

                q = torch.round((w_flat - zero) / step)
                q = torch.clamp(q, qmin, qmax)

                wq = q * step + zero

            else:
                step = item[0] if isinstance(item, (tuple, list)) else item
                step = step.to(w.device).view(-1, 1)

                q = torch.round(w_flat / step)
                wq = q * step

            wq = wq.view_as(w)

            return torch.nn.functional.linear(x, wq, mod.bias)

        return hook

    for name in layer_names:
        if name not in steps:
            continue

        mod = name2mod.get(name, None)

        if not isinstance(mod, nn.Linear):
            continue

        handles.append(
            mod.register_forward_hook(
                make_hook(steps[name])
            )
        )

    print(f"[INFO] registered quant hooks: {len(handles)}")

    return handles


# ============================================================
# UNIFORM
# ============================================================

def compute_uniform_minmax_steps(
    model: nn.Module,
    layer_names,
    num_bits: int,
    device,
):
    model = model.to(device).eval()

    name2mod = dict(model.named_modules())

    steps = {}

    qmax = (2 ** num_bits) - 1

    with torch.no_grad():
        for name in layer_names:

            mod = name2mod.get(name, None)

            if not isinstance(mod, nn.Linear):
                continue

            w = mod.weight.detach().to(device)

            w_flat = w.view(w.size(0), -1)

            w_min = w_flat.min(dim=1).values
            w_max = w_flat.max(dim=1).values

            step = (w_max - w_min) / float(qmax)

            step = torch.clamp(step, min=1e-8)

            steps[name] = {
                "step": step.cpu(),
                "zero": w_min.cpu(),
                "qmin": 0,
                "qmax": qmax,
            }

    return steps


# ============================================================
# BRECQ
# ============================================================

def load_brecq_adaround_state(qnn, adaround_path: Path, device):
    state = torch.load(adaround_path, map_location="cpu")

    loaded = 0
    missing = 0

    for name, m in qnn.model.named_modules():

        if not isinstance(m, QuantModule):
            continue

        if name not in state:
            missing += 1
            continue

        item = state[name]

        q = m.weight_quantizer

        q.delta = item["delta"].to(device)
        q.zero_point = item["zero_point"].to(device)
        q.inited = True

        if "alpha" in item:

            ada_q = AdaRoundQuantizer(
                uaq=q,
                round_mode="learned_hard_sigmoid",
                weight_tensor=m.org_weight.data,
            )

            ada_q.alpha.data.copy_(item["alpha"].to(device))
            ada_q.soft_targets = False

            m.weight_quantizer = ada_q

        loaded += 1

    print(f"[INFO] Loaded BRECQ AdaRound state: loaded={loaded}, missing={missing}")


def build_brecq_model(fp_model, adaround_path: Path, device, n_bits_w: int):

    wq_params = {
        "n_bits": n_bits_w,
        "channel_wise": True,
        "scale_method": "max",
    }

    aq_params = {
        "n_bits": 8,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": False,
    }

    qnn = PoseidonQuantModel(
        model=fp_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
    ).to(device)

    qnn.eval()

    load_brecq_adaround_state(qnn, adaround_path, device)

    qnn.set_quant_state(True, False)

    return qnn


# ============================================================
# METRICS
# ============================================================

def denormalize_tensor(x: torch.Tensor, constants, dataset_name: str = ""):

    dataset_name = dataset_name.lower()

    if "wave" in dataset_name:

        out = x.clone()

        mean_u = torch.as_tensor(
            constants["mean"],
            dtype=x.dtype,
            device=x.device,
        )

        std_u = torch.as_tensor(
            constants["std"],
            dtype=x.dtype,
            device=x.device,
        )

        mean_c = torch.as_tensor(
            constants["mean_c"],
            dtype=x.dtype,
            device=x.device,
        )

        std_c = torch.as_tensor(
            constants["std_c"],
            dtype=x.dtype,
            device=x.device,
        )

        if x.ndim == 4:

            out[:, 0] = x[:, 0] * std_u + mean_u

            if x.shape[1] >= 2:
                out[:, 1] = x[:, 1] * std_c + mean_c

        elif x.ndim == 3:

            out[0] = x[0] * std_u + mean_u

            if x.shape[0] >= 2:
                out[1] = x[1] * std_c + mean_c

        return out

    mean = torch.as_tensor(
        constants["mean"],
        dtype=x.dtype,
        device=x.device,
    ).flatten()

    std = torch.as_tensor(
        constants["std"],
        dtype=x.dtype,
        device=x.device,
    ).flatten()

    if x.ndim == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

    elif x.ndim == 3:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)

    return x * std + mean


def spatial_first_order_sobolev(
    pred: torch.Tensor,
    target: torch.Tensor,
    constants=None,
    denorm=True,
    dataset_name: str = "",
):

    if denorm and constants is not None:

        pred = denormalize_tensor(
            pred,
            constants,
            dataset_name=dataset_name,
        )

        target = denormalize_tensor(
            target,
            constants,
            dataset_name=dataset_name,
        )

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    s0_err = float(np.mean(np.abs(pred_np - target_np)))
    s0_norm = float(np.mean(np.abs(target_np)))

    dx_pred = pred_np[..., :, 1:] - pred_np[..., :, :-1]
    dx_tgt = target_np[..., :, 1:] - target_np[..., :, :-1]

    dy_pred = pred_np[..., 1:, :] - pred_np[..., :-1, :]
    dy_tgt = target_np[..., 1:, :] - target_np[..., :-1, :]

    s1_err = float(
        np.mean(np.abs(dx_pred - dx_tgt))
        + np.mean(np.abs(dy_pred - dy_tgt))
    )

    s1_norm = float(
        np.mean(np.abs(dx_tgt))
        + np.mean(np.abs(dy_tgt))
    )

    sobolev_s01 = s0_err + s1_err

    rel_sobolev_s01 = sobolev_s01 / (s0_norm + s1_norm + 1e-12)

    return {
        "sobolev_s01": sobolev_s01,
        "rel_sobolev_s01": rel_sobolev_s01,
    }


def spatial_sobolev_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_order: int,
    constants=None,
    denorm=True,
    dataset_name: str = "",
):
    if max_order < 0:
        raise ValueError("max_order must be non-negative.")

    if denorm and constants is not None:
        pred = denormalize_tensor(
            pred,
            constants,
            dataset_name=dataset_name,
        )
        target = denormalize_tensor(
            target,
            constants,
            dataset_name=dataset_name,
        )

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    total_error = float(np.mean(np.abs(pred_np - target_np)))
    total_norm = float(np.mean(np.abs(target_np)))

    if max_order >= 1:
        dx_pred = pred_np[..., :, 1:] - pred_np[..., :, :-1]
        dx_tgt = target_np[..., :, 1:] - target_np[..., :, :-1]
        dy_pred = pred_np[..., 1:, :] - pred_np[..., :-1, :]
        dy_tgt = target_np[..., 1:, :] - target_np[..., :-1, :]

        total_error += float(
            np.mean(np.abs(dx_pred - dx_tgt))
            + np.mean(np.abs(dy_pred - dy_tgt))
        )
        total_norm += float(
            np.mean(np.abs(dx_tgt))
            + np.mean(np.abs(dy_tgt))
        )

    if max_order >= 2:
        dxx_pred = pred_np[..., :, 2:] - 2.0 * pred_np[..., :, 1:-1] + pred_np[..., :, :-2]
        dxx_tgt = target_np[..., :, 2:] - 2.0 * target_np[..., :, 1:-1] + target_np[..., :, :-2]
        dyy_pred = pred_np[..., 2:, :] - 2.0 * pred_np[..., 1:-1, :] + pred_np[..., :-2, :]
        dyy_tgt = target_np[..., 2:, :] - 2.0 * target_np[..., 1:-1, :] + target_np[..., :-2, :]
        dxy_pred = (
            pred_np[..., 1:, 1:]
            - pred_np[..., 1:, :-1]
            - pred_np[..., :-1, 1:]
            + pred_np[..., :-1, :-1]
        )
        dxy_tgt = (
            target_np[..., 1:, 1:]
            - target_np[..., 1:, :-1]
            - target_np[..., :-1, 1:]
            + target_np[..., :-1, :-1]
        )

        total_error += float(
            np.mean(np.abs(dxx_pred - dxx_tgt))
            + np.mean(np.abs(dyy_pred - dyy_tgt))
            + np.mean(np.abs(dxy_pred - dxy_tgt))
        )
        total_norm += float(
            np.mean(np.abs(dxx_tgt))
            + np.mean(np.abs(dyy_tgt))
            + np.mean(np.abs(dxy_tgt))
        )

    return {
        f"sobolev_s0{max_order}": total_error,
        f"rel_sobolev_s0{max_order}": total_error / (total_norm + 1e-12),
    }


def spatial_grads_np(f, dx=1.0 / 128, dy=1.0 / 128):

    original_ndim = f.ndim

    if original_ndim == 2:
        f = f[np.newaxis, ...]

    dy_f = np.zeros_like(f)
    dx_f = np.zeros_like(f)

    dy_f[..., 1:-1, :] = (
        f[..., 2:, :] - f[..., :-2, :]
    ) / (2 * dy)

    dx_f[..., :, 1:-1] = (
        f[..., :, 2:] - f[..., :, :-2]
    ) / (2 * dx)

    dy_f[..., 0, :] = (
        f[..., 1, :] - f[..., 0, :]
    ) / dy

    dy_f[..., -1, :] = (
        f[..., -1, :] - f[..., -2, :]
    ) / dy

    dx_f[..., :, 0] = (
        f[..., :, 1] - f[..., :, 0]
    ) / dx

    dx_f[..., :, -1] = (
        f[..., :, -1] - f[..., :, -2]
    ) / dx

    if original_ndim == 2:
        return dy_f[0], dx_f[0]

    return dy_f, dx_f


def ns_physical_metrics(pred, target, constants):

    pred = denormalize_tensor(pred, constants, dataset_name="ns")
    target = denormalize_tensor(target, constants, dataset_name="ns")

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    pred_np = np.swapaxes(pred_np, -2, -1)
    target_np = np.swapaxes(target_np, -2, -1)

    c_dim = 1 if pred_np.ndim == 4 else 0

    if pred_np.shape[c_dim] == 3:
        u_idx, v_idx = 0, 1
    else:
        u_idx, v_idx = 1, 2

    if pred_np.ndim == 4:

        u_pred, v_pred = pred_np[:, u_idx], pred_np[:, v_idx]
        u_gt, v_gt = target_np[:, u_idx], target_np[:, v_idx]

    else:

        u_pred, v_pred = pred_np[u_idx], pred_np[v_idx]
        u_gt, v_gt = target_np[u_idx], target_np[v_idx]

    _, du_dx = spatial_grads_np(u_pred)
    dv_dy, _ = spatial_grads_np(v_pred)

    div = float(np.mean(np.abs(du_dx + dv_dy)))

    _, dv_dx = spatial_grads_np(v_pred)
    du_dy, _ = spatial_grads_np(u_pred)

    vort_pred = dv_dx - du_dy

    _, dv_dx_gt = spatial_grads_np(v_gt)
    du_dy_gt, _ = spatial_grads_np(u_gt)

    vort_gt = dv_dx_gt - du_dy_gt

    vort_err = float(np.mean(np.abs(vort_pred - vort_gt)))

    return {
        "continuity": div,
        "vorticity": vort_err,
    }


def is_ns_dataset(dataset_name: str):
    return "incompressible" in dataset_name.lower()


# ============================================================
# ROLLOUT EVAL
# ============================================================

@torch.no_grad()
def evaluate_rollout_horizon(
    model,
    trainer,
    dataset,
    rollout_steps,
    constants,
    dataset_name,
):
    print("[DEBUG] len(dataset) =", len(dataset))
    preds, labels, _ = rollout(
        trainer,
        dataset,
        ar_steps=rollout_steps,
        output_all_steps=False,
    )
    print("[DEBUG] preds.shape =", preds.shape)
    print("[DEBUG] labels.shape =", labels.shape)

    preds = torch.from_numpy(preds)
    labels = torch.from_numpy(labels)

    rec = {
        "L1": float(
            np.mean(
                lp_error(
                    preds.numpy(),
                    labels.numpy(),
                    p=1,
                )
            )
        ),

        "RelL1": float(
            np.mean(
                relative_lp_error(
                    preds.numpy(),
                    labels.numpy(),
                    p=1,
                    return_percent=True,
                )
            )
        ),
    }

    rec.update(
        spatial_first_order_sobolev(
            preds,
            labels,
            constants=constants,
            denorm=True,
            dataset_name=dataset_name,
        )
    )
    rec.update(
        spatial_sobolev_loss(
            preds,
            labels,
            max_order=2,
            constants=constants,
            denorm=True,
            dataset_name=dataset_name,
        )
    )

    if is_ns_dataset(dataset_name):

        rec.update(
            ns_physical_metrics(
                preds,
                labels,
                constants,
            )
        )

    return rec


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate long-term Poseidon rollout using saved SBPQ step "
            "sizes, with optional old SAPQ/PPQ/BRECQ comparisons."
        )
    )
    parser.add_argument(
        "--sbpq-run-dir",
        type=Path,
        default=DEFAULT_SBPQ_RUN_DIR,
        help=(
            "SBPQ run directory containing sbpq_trainer_state.pt. "
            "Defaults to the current best NS-PwC-L run by validation L1."
        ),
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        nargs="+",
        default=ROLLOUT_STEPS,
        help="Autoregressive rollout horizons to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path. Defaults to "
            "eval_results_longterm_sbpq/<model>/<dataset>.json."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = PPQConfig()

    root = Path(cfg.repo_root)

    model_tag = Path(cfg.model_path).name
    dataset_tag = Path(cfg.data_path).name

    print("\n[INFO] Loading FP model...")

    model, device = load_poseidon_model(
        cfg.model_path,
        cfg.device,
    )

    layers = load_candidate_layers(
        model,
        Path(cfg.quant_layer_path),
    )

    results = {
        "meta": {
            "model_path": cfg.model_path,
            "data_path": cfg.data_path,
            "dataset_name": cfg.dataset_name,
            "rollout_steps": args.rollout_steps,
            "sbpq_run_dir": str(args.sbpq_run_dir),
        }
    }
    eval_dataset_name = cfg.dataset_name

    # ========================================================
    # FP
    # ========================================================

    results["FP"] = {}

    for step in args.rollout_steps:

        print(f"\n========== FP rollout={step} ==========")

        dataset = get_test_set(
            eval_dataset_name,
            cfg.data_path,
            initial_time=0,
            final_time=step,
        )
        print(f"[INFO] rollout dataset size = {len(dataset)}")

        constants = dataset.dataset.constants if hasattr(dataset, "dataset") else dataset.constants

        trainer = get_trainer(
            cfg.model_path,
            batch_size=cfg.val_batchsize,
            dataset=dataset,
        )

        trainer.model = model
        trainer.model_wrapped = model
        trainer.args.remove_unused_columns = False
        trainer._signature_columns = None

        metrics = evaluate_rollout_horizon(
            model=model,
            trainer=trainer,
            dataset=dataset,
            rollout_steps=step,
            constants=constants,
            dataset_name=cfg.dataset_name,
        )

        results["FP"][f"t{step}"] = metrics

    # ========================================================
    # SBPQ / SAPQ / PPQ
    # ========================================================

    saved_methods = []

    saved_methods.append(
        (
            "SBPQ",
            "sbpq",
            args.sbpq_run_dir,
        )
    )

    saved_methods.append(
        (
            "SAPQ",
            "step_file",
            root
            / "sapq_experiments"
            / model_tag
            / dataset_tag
            / "network_block_sens_sobo"
            / "raw"
            / "sapq_global_step_sizes.pt"
        )
    )

    saved_methods.append(
        (
            "PPQ",
            "step_file",
            root
            / "ppq_experiments"
            / model_tag
            / dataset_tag
            / "layerwise_ppq"
            / "sapq_layerwise_step_sizes.pt"
        )
    )

    for method_name, method_type, step_path in saved_methods:

        if method_type == "sbpq":
            checkpoint_path = step_path / "sbpq_trainer_state.pt"
            exists = checkpoint_path.exists()
        else:
            exists = step_path.exists()

        if not exists:
            print(f"[WARN] Missing {method_name}: {step_path}")
            continue

        print(f"\n[INFO] Loading {method_name} steps...")

        if method_type == "sbpq":
            steps, meta = load_sbpq_step_sizes(step_path)
            method_layers = list(steps.keys())
        else:
            steps, meta = load_saved_steps(step_path)
            method_layers = layers

        print(f"[DEBUG] {method_name} step_path = {step_path}")
        print(f"[DEBUG] exists = {exists}")
        print(f"[DEBUG] meta = {meta}")


        handles = register_quant_hooks(
            model,
            method_layers,
            steps,
        )

        results[method_name] = {}

        for step in args.rollout_steps:

            print(f"\n========== {method_name} rollout={step} ==========")

            dataset = get_test_set(
                eval_dataset_name,
                cfg.data_path,
                initial_time=0,
                final_time=step,
            )
            print(f"[INFO] rollout dataset size = {len(dataset)}")

            constants = dataset.dataset.constants if hasattr(dataset, "dataset") else dataset.constants

            trainer = get_trainer(
                cfg.model_path,
                batch_size=cfg.val_batchsize,
                dataset=dataset,
            )

            trainer.model = model
            trainer.model_wrapped = model
            trainer.args.remove_unused_columns = False
            trainer._signature_columns = None



            metrics = evaluate_rollout_horizon(
                model=model,
                trainer=trainer,
                dataset=dataset,
                rollout_steps=step,
                constants=constants,
                dataset_name=cfg.dataset_name,
            )

            results[method_name][f"t{step}"] = metrics

        for h in handles:
            h.remove()

    # ========================================================
    # UNIFORM
    # ========================================================

    for bits in UNIFORM_BITS_LIST:

        method_name = f"Fixed-w{bits}"

        uniform_steps = compute_uniform_minmax_steps(
            model,
            layers,
            bits,
            device,
        )

        handles = register_quant_hooks(
            model,
            layers,
            uniform_steps,
        )

        results[method_name] = {}

        for step in args.rollout_steps:

            print(f"\n========== {method_name} rollout={step} ==========")

            dataset = get_test_set(
                eval_dataset_name,
                cfg.data_path,
                initial_time=0,
                final_time=step,
            )
            print(f"[INFO] rollout dataset size = {len(dataset)}")

            constants = dataset.dataset.constants if hasattr(dataset, "dataset") else dataset.constants

            trainer = get_trainer(
                cfg.model_path,
                batch_size=cfg.val_batchsize,
                dataset=dataset,
            )

            trainer.model = model
            trainer.model_wrapped = model
            trainer.args.remove_unused_columns = False
            trainer._signature_columns = None

            metrics = evaluate_rollout_horizon(
                model=model,
                trainer=trainer,
                dataset=dataset,
                rollout_steps=step,
                constants=constants,
                dataset_name=cfg.dataset_name,
            )

            results[method_name][f"t{step}"] = metrics

        for h in handles:
            h.remove()

    # ========================================================
    # BRECQ
    # ========================================================

    for bits in BRECQ_BITS_LIST:
        for iters in BRECQ_ITERS_LIST:

            method_name = f"BRECQ-w{bits}-iters{iters}"

            adaround_path = (
                root
                / "brecq_artifacts"
                / model_tag
                / "recon"
                / f"w{bits}"
                / f"iters{iters}"
                / "adaround_state.pt"
            )

            if not adaround_path.exists():
                print(f"[WARN] Missing BRECQ path: {adaround_path}")
                continue

            print(f"\n[INFO] Building {method_name}...")

            fp_model, _ = load_poseidon_model(
                cfg.model_path,
                cfg.device,
            )

            brecq_model = build_brecq_model(
                fp_model=fp_model,
                adaround_path=adaround_path,
                device=device,
                n_bits_w=bits,
            )

            brecq_model.config = brecq_model.model.config

            results[method_name] = {}

            for step in args.rollout_steps:

                print(f"\n========== {method_name} rollout={step} ==========")

                dataset = get_test_set(
                    eval_dataset_name,
                    cfg.data_path,
                    initial_time=0,
                    final_time=step,
                )
                print(f"[INFO] rollout dataset size = {len(dataset)}")

                constants = dataset.dataset.constants if hasattr(dataset, "dataset") else dataset.constants

                trainer = get_trainer(
                    cfg.model_path,
                    batch_size=cfg.val_batchsize,
                    dataset=dataset,
                )

                trainer.model = brecq_model
                trainer.model_wrapped = model
                trainer.args.remove_unused_columns = False
                trainer._signature_columns = None

                metrics = evaluate_rollout_horizon(
                    model=brecq_model,
                    trainer=trainer,
                    dataset=dataset,
                    rollout_steps=step,
                    constants=constants,
                    dataset_name=cfg.dataset_name,
                )

                results[method_name][f"t{step}"] = metrics

    # ========================================================
    # SAVE
    # ========================================================

    save_dir = root / "eval_results_longterm_sbpq" / model_tag

    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = args.output or (save_dir / f"{dataset_tag}.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n========== RESULTS ==========")

    for method, vals in results.items():

        if method == "meta":
            continue

        print(f"\n[{method}]")

        for horizon, rec in vals.items():

            line = (
                f"{horizon:6s} | "
                f"L1={rec['L1']:.6e} | "
                f"RelL1={rec['RelL1']:.6e} | "
                f"Sob1={rec['sobolev_s01']:.6e} | "
                f"RelSob1={rec['rel_sobolev_s01']:.6e} | "
                f"Sob2={rec['sobolev_s02']:.6e} | "
                f"RelSob2={rec['rel_sobolev_s02']:.6e}"
            )

            if "continuity" in rec:
                line += (
                    f" | Div={rec['continuity']:.6e}"
                    f" | Vort={rec['vorticity']:.6e}"
                )

            print(line)

    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()
