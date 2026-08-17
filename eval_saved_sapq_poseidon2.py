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

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer

from scOT.metrics import relative_lp_error, lp_error
import os


def load_saved_steps(path: Path):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "step_sizes_dict" in obj:
        return obj["step_sizes_dict"], obj.get("meta", {})

    if isinstance(obj, dict):
        return obj, {}

    raise ValueError(f"Unsupported step-size file format: {path}")


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


# def compute_uniform_minmax_steps(model: nn.Module, layer_names, num_bits: int, device):
#     """
#     Per-channel min-max uniform affine weight quantization step:

#         step = (w_max - w_min) / (2^bits - 1)

#     For Linear weight shape [out_features, in_features],
#     each output channel gets one step size.
#     """
#     model = model.to(device).eval()
#     name2mod = dict(model.named_modules())

#     steps = {}
#     denom = (2 ** num_bits) - 1

#     with torch.no_grad():
#         for name in layer_names:
#             mod = name2mod.get(name, None)
#             if not isinstance(mod, nn.Linear):
#                 continue

#             w = mod.weight.detach().to(device)
#             w_flat = w.view(w.size(0), -1)

#             w_min = w_flat.min(dim=1).values
#             w_max = w_flat.max(dim=1).values

#             step = (w_max - w_min) / float(denom)
#             step = torch.clamp(step, min=1e-8)

#             steps[name] = (step.cpu(), None)

#     return steps



def compute_uniform_minmax_steps(model: nn.Module, layer_names, num_bits: int, device):
    """
    Per-channel uniform symmetric weight quantization step:

        step = max(abs(w_min), abs(w_max)) / (2^(bits-1) - 1)

    For Linear weight shape [out_features, in_features],
    each output channel gets one step size.
    """
    model = model.to(device).eval()
    name2mod = dict(model.named_modules())

    steps = {}
    qmax = (2 ** (num_bits - 1)) - 1

    with torch.no_grad():
        for name in layer_names:
            mod = name2mod.get(name, None)
            if not isinstance(mod, nn.Linear):
                continue

            w = mod.weight.detach().to(device)
            w_flat = w.view(w.size(0), -1)

            max_abs = w_flat.abs().max(dim=1).values

            step = max_abs / float(qmax)
            step = torch.clamp(step, min=1e-8)

            steps[name] = (step.cpu(), None)

    return steps


# def compute_uniform_minmax_steps(model: nn.Module, layer_names, num_bits: int, device):
#     """
#     Per-channel uniform affine weight quantization.

#         step = (w_max - w_min) / (2^bits - 1)
#         zero = w_min

#         q  = round((w - zero) / step)
#         wq = q * step + zero
#     """
#     model = model.to(device).eval()
#     name2mod = dict(model.named_modules())

#     steps = {}
#     qmax = (2 ** num_bits) - 1

#     with torch.no_grad():
#         for name in layer_names:
#             mod = name2mod.get(name, None)
#             if not isinstance(mod, nn.Linear):
#                 continue

#             w = mod.weight.detach().to(device)
#             w_flat = w.view(w.size(0), -1)

#             w_min = w_flat.min(dim=1).values
#             w_max = w_flat.max(dim=1).values

#             step = (w_max - w_min) / float(qmax)
#             step = torch.clamp(step, min=1e-8)

#             steps[name] = {
#                 "step": step.cpu(),
#                 "zero": w_min.cpu(),
#                 "qmin": 0,
#                 "qmax": qmax,
#             }

#     return steps




def mean_dict(records):
    acc = defaultdict(list)
    for rec in records:
        for k, v in rec.items():
            acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def denormalize_tensor(x: torch.Tensor, constants, dataset_name: str = ""):
    """
    Denormalize Poseidon output/label.

    Normal datasets:
        use constants["mean"], constants["std"]

    Wave datasets:
        channel 0: u uses mean/std
        channel 1: c uses mean_c/std_c
    """
    dataset_name = dataset_name.lower()

    if "wave" in dataset_name:
        out = x.clone()

        mean_u = torch.as_tensor(constants["mean"], dtype=x.dtype, device=x.device)
        std_u = torch.as_tensor(constants["std"], dtype=x.dtype, device=x.device)

        mean_c = torch.as_tensor(constants["mean_c"], dtype=x.dtype, device=x.device)
        std_c = torch.as_tensor(constants["std_c"], dtype=x.dtype, device=x.device)

        if x.ndim == 4:
            out[:, 0] = x[:, 0] * std_u + mean_u
            if x.shape[1] >= 2:
                out[:, 1] = x[:, 1] * std_c + mean_c

        elif x.ndim == 3:
            out[0] = x[0] * std_u + mean_u
            if x.shape[0] >= 2:
                out[1] = x[1] * std_c + mean_c

        return out

    mean = torch.as_tensor(constants["mean"], dtype=x.dtype, device=x.device).flatten()
    std = torch.as_tensor(constants["std"], dtype=x.dtype, device=x.device).flatten()

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
    """
    Space-only first-order Sobolev:
        S01 = L1(u - y) + L1(dx u - dx y) + L1(dy u - dy y)

    No temporal term.
    """
    if denorm and constants is not None:
        pred = denormalize_tensor(pred, constants, dataset_name=dataset_name)
        target = denormalize_tensor(target, constants, dataset_name=dataset_name)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    s0_err = float(np.mean(np.abs(pred_np - target_np)))
    s0_norm = float(np.mean(np.abs(target_np)))

    dx_pred = pred_np[..., :, 1:] - pred_np[..., :, :-1]
    dx_tgt = target_np[..., :, 1:] - target_np[..., :, :-1]

    dy_pred = pred_np[..., 1:, :] - pred_np[..., :-1, :]
    dy_tgt = target_np[..., 1:, :] - target_np[..., :-1, :]

    s1_err = float(np.mean(np.abs(dx_pred - dx_tgt)) + np.mean(np.abs(dy_pred - dy_tgt)))
    s1_norm = float(np.mean(np.abs(dx_tgt)) + np.mean(np.abs(dy_tgt)))

    sobolev_s01 = s0_err + s1_err
    rel_sobolev_s01 = sobolev_s01 / (s0_norm + s1_norm + 1e-12)

    return {
        "sobolev_s01": sobolev_s01,
        "rel_sobolev_s01": rel_sobolev_s01,
    }


def spatial_grads_np(f, dx=1.0 / 128, dy=1.0 / 128):
    """
    f shape: [B, H, W] or [H, W]
    returns: dy_f, dx_f
    """
    original_ndim = f.ndim
    if original_ndim == 2:
        f = f[np.newaxis, ...]

    dy_f = np.zeros_like(f)
    dx_f = np.zeros_like(f)

    dy_f[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * dy)
    dx_f[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / (2 * dx)

    dy_f[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / dy
    dy_f[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / dy

    dx_f[..., :, 0] = (f[..., :, 1] - f[..., :, 0]) / dx
    dx_f[..., :, -1] = (f[..., :, -1] - f[..., :, -2]) / dx

    if original_ndim == 2:
        return dy_f[0], dx_f[0]

    return dy_f, dx_f


def ns_physical_metrics(pred: torch.Tensor, target: torch.Tensor, constants, denorm=True):
    """
    NS-only:
    - continuity / divergence-free
    - vorticity error
    """
    if denorm and constants is not None:
        pred = denormalize_tensor(pred, constants, dataset_name="ns")
        target = denormalize_tensor(target, constants, dataset_name="ns")

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # match friend's code: swap last two spatial axes
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

    _, du_dx_gt = spatial_grads_np(u_gt)
    dv_dy_gt, _ = spatial_grads_np(v_gt)
    div_scale = float(np.mean(np.abs(du_dx_gt)) + np.mean(np.abs(dv_dy_gt)) + 1e-12)

    _, dv_dx = spatial_grads_np(v_pred)
    du_dy, _ = spatial_grads_np(u_pred)
    vort_pred = dv_dx - du_dy

    _, dv_dx_gt = spatial_grads_np(v_gt)
    du_dy_gt, _ = spatial_grads_np(u_gt)
    vort_gt = dv_dx_gt - du_dy_gt

    vort_err = float(np.mean(np.abs(vort_pred - vort_gt)))
    rel_vort_err = vort_err / (float(np.mean(np.abs(vort_gt))) + 1e-12)

    return {
        "continuity": div,
        "rel_continuity": div / div_scale,
        "vorticity": vort_err,
        "rel_vorticity": rel_vort_err,
    }


def is_ns_dataset(dataset_name: str) -> bool:
    return "incompressible" in dataset_name.lower()


def eval_model_all_metrics(
    model,
    val_loader,
    layer_names,
    steps,
    constants,
    device,
    dataset_name: str,
):
    """
    For FP / BRECQ:
        layer_names=[]
        steps={}

    For PPQ / SAPQ:
        layer_names=candidate layers
        steps=saved step_sizes_dict
    """
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

    for name in layer_names:
        if name not in steps:
            continue

        mod = name2mod.get(name, None)
        if not isinstance(mod, nn.Linear):
            continue

        item = steps[name]
        step = item[0] if isinstance(item, (tuple, list)) else item
        handles.append(mod.register_forward_hook(make_hook(step)))

    loader = val_loader() if callable(val_loader) else val_loader
    records = []

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            t = batch.get("time", None)
            pm = batch.get("pixel_mask", None)
            y = batch.get("labels", None)

            if y is None:
                continue

            if t is not None:
                t = t.to(device)
            if pm is not None:
                pm = pm.to(device)
            y = y.to(device)

            # First inference: x0 -> out1
            out1 = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
                return_dict=True,
            ).output

            # Second inference: out1 -> out2
            out2 = model(
                pixel_values=out1,
                time=t,
                pixel_mask=pm,
                labels=y,
                return_dict=True,
            ).output

            # Evaluate second-step prediction for now
            out = out2

            pred_np = out.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            rec = {
                "L1": float(np.mean(lp_error(pred_np, y_np, p=1))),
                "RelL1": float(np.mean(relative_lp_error(pred_np, y_np, p=1, return_percent=True))),
            }

            rec.update(
                spatial_first_order_sobolev(
                    pred=out,
                    target=y,
                    constants=constants,
                    denorm=True,
                    dataset_name=dataset_name,
                )
            )

            if is_ns_dataset(dataset_name):
                rec.update(
                    ns_physical_metrics(
                        pred=out,
                        target=y,
                        constants=constants,
                        denorm=True,
                    )
                )

            records.append(rec)

    for h in handles:
        h.remove()

    if len(records) == 0:
        return {}

    return mean_dict(records)


def main():
    cfg = PPQConfig()
    cfg.model_path = os.environ.get("PPQ_MODEL_PATH", cfg.model_path)
    cfg.data_path = os.environ.get("PPQ_DATA_PATH", cfg.data_path)
    cfg.dataset_name = os.environ.get("PPQ_DATASET_NAME", cfg.dataset_name)
    root = Path(cfg.repo_root)

    model_tag = Path(cfg.model_path).name
    dataset_tag = Path(cfg.data_path).name

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

    brecq_bits_list = [8]
    brecq_iters_list = [10000]
    uniform_bits_list = [8]

    brecq_paths = {}
    for bits in brecq_bits_list:
        for iters in brecq_iters_list:
            name = f"BRECQ-w{bits}-iters{iters}"
            path = (
                root
                / "brecq_artifacts"
                / model_tag
                / "recon"
                / f"w{bits}"
                / f"iters{iters}"
                / "adaround_state.pt"
            )
            brecq_paths[name] = {
                "path": path,
                "bits": bits,
                "iters": iters,
            }

    print("========== UNIFIED EVAL CONFIG ==========")
    print(f"model_path:    {cfg.model_path}")
    print(f"data_path:     {cfg.data_path}")
    print(f"dataset_name:  {cfg.dataset_name}")
    print(f"val_batchsize: {cfg.val_batchsize}")
    print(f"val_steps:     {cfg.val_steps}")

    for name, path in step_paths.items():
        print(f"{name}_path: {path}")

    for name, rec in brecq_paths.items():
        print(f"{name}_path: {rec['path']}")

    print("=========================================")

    print("\n[INFO] Loading Poseidon model...")
    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    print("\n[INFO] Building validation loader...")
    _, val_loader, _, val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=cfg.val_batchsize,
        val_steps=cfg.val_steps,
    )

    constants = val_loader.dataset.constants

    print("\n[INFO] Loading candidate Linear layers...")
    layers = load_candidate_layers(model, Path(cfg.quant_layer_path))

    results = {
        "meta": {
            "model_path": cfg.model_path,
            "data_path": cfg.data_path,
            "dataset_name": cfg.dataset_name,
            "quant_layer_path": str(cfg.quant_layer_path),
            "num_candidate_layers": len(layers),
            "step_paths": {k: str(v) for k, v in step_paths.items()},
            "brecq_paths": {k: str(v["path"]) for k, v in brecq_paths.items()},
            "metrics": [
                "L1",
                "RelL1",
                "sobolev_s01",
                "rel_sobolev_s01",
                "continuity / rel_continuity for NS only",
                "vorticity / rel_vorticity for NS only",
            ],
        }
    }

    print("\nEvaluating FP...")
    fp_metrics = eval_model_all_metrics(
        model=model,
        val_loader=val_iter,
        layer_names=[],
        steps={},
        constants=constants,
        device=device,
        dataset_name=cfg.dataset_name,
    )
    results["FP"] = fp_metrics

    for method_name, step_path in step_paths.items():
        if not step_path.exists():
            print(f"[WARN] Skip {method_name}, missing file: {step_path}")
            continue

        print(f"\n[INFO] Loading saved {method_name} step sizes...")
        steps, method_meta = load_saved_steps(step_path)
        print(f"[INFO] Loaded {method_name} step sizes for {len(steps)} layers.")

        print(f"\nEvaluating {method_name}...")
        metrics = eval_model_all_metrics(
            model=model,
            val_loader=val_iter,
            layer_names=layers,
            steps=steps,
            constants=constants,
            device=device,
            dataset_name=cfg.dataset_name,
        )

        results[method_name] = {
            **metrics,
            "num_step_layers": len(steps),
            "meta": method_meta,
        }



    for bits in uniform_bits_list:
        method_name = f"Uniform-w{bits}"

        print(f"\nEvaluating {method_name}...")

        uniform_steps = compute_uniform_minmax_steps(
            model=model,
            layer_names=layers,
            num_bits=bits,
            device=device,
        )

        metrics = eval_model_all_metrics(
            model=model,
            val_loader=val_iter,
            layer_names=layers,
            steps=uniform_steps,
            constants=constants,
            device=device,
            dataset_name=cfg.dataset_name,
        )

        results[method_name] = {
            **metrics,
            "bits": bits,
            "num_step_layers": len(uniform_steps),
            "method": "minmax_uniform_affine_weight_only",
        }



    for method_name, rec in brecq_paths.items():
        adaround_path = rec["path"]
        bits = rec["bits"]
        iters = rec["iters"]

        if not adaround_path.exists():
            print(f"[WARN] Skip {method_name}, missing file: {adaround_path}")
            continue

        print(f"\nEvaluating {method_name}...")

        brecq_base_model, _ = load_poseidon_model(cfg.model_path, cfg.device)

        brecq_model = build_brecq_model(
            fp_model=brecq_base_model,
            adaround_path=adaround_path,
            device=device,
            n_bits_w=bits,
        )

        brecq_metrics = eval_model_all_metrics(
            model=brecq_model,
            val_loader=val_iter,
            layer_names=[],
            steps={},
            constants=constants,
            device=device,
            dataset_name=cfg.dataset_name,
        )

        results[method_name] = {
            **brecq_metrics,
            "bits": bits,
            "iters": iters,
            "adaround_path": str(adaround_path),
        }

    model_tag = Path(cfg.model_path).name
    dataset_tag = Path(cfg.data_path).name

    save_dir = root / "eval_results" / model_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{dataset_tag}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n========== RESULTS ==========")
    for name, rec in results.items():
        if name == "meta":
            continue

        line = (
            f"{name:20s} | "
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