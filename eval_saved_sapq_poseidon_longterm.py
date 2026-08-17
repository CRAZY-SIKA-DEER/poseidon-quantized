from __future__ import annotations

import json
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

BRECQ_BITS_LIST = [4]
BRECQ_ITERS_LIST = [10000]

UNIFORM_BITS_LIST = [4]


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

def main():

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
            "rollout_steps": ROLLOUT_STEPS,
        }
    }
    eval_dataset_name = cfg.dataset_name

    # ========================================================
    # FP
    # ========================================================

    results["FP"] = {}

    for step in ROLLOUT_STEPS:

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
    # SAPQ / PPQ
    # ========================================================

    step_paths = {
        "SAPQ": (
            root
            / "sapq_experiments"
            / model_tag
            / dataset_tag
            / "network_block_sens_sobo"
            / "raw"
            / "sapq_global_step_sizes.pt"
        ),

        "PPQ": (
            root
            / "ppq_experiments"
            / model_tag
            / dataset_tag
            / "layerwise_ppq"
            / "sapq_layerwise_step_sizes.pt"
        ),
    }

    for method_name, step_path in step_paths.items():

        if not step_path.exists():
            print(f"[WARN] Missing {method_name}: {step_path}")
            continue

        print(f"\n[INFO] Loading {method_name} steps...")

        steps, meta = load_saved_steps(step_path)
        print(f"[DEBUG] {method_name} step_path = {step_path}")
        print(f"[DEBUG] exists = {step_path.exists()}")


        handles = register_quant_hooks(
            model,
            layers,
            steps,
        )

        results[method_name] = {}

        for step in ROLLOUT_STEPS:

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

        method_name = f"Uniform-w{bits}"

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

        for step in ROLLOUT_STEPS:

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

            for step in ROLLOUT_STEPS:

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

    save_dir = root / "eval_results_longterm" / model_tag

    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{dataset_tag}.json"

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
                f"RelSob1={rec['rel_sobolev_s01']:.6e}"
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