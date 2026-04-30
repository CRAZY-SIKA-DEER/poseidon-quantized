# SAPQ/eval_saved_steps_nspwc.py
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from PPQ.metrics import evaluate_with_stepsizes
from SAPQ.run_sapq_network_global import load_candidate_layers


# -----------------------------
# Physical metrics from friend's logic, simplified for incompressible NS-PwC
# -----------------------------
class IncompressibleLoss:
    def __init__(
        self,
        preds,
        labels,
        constants,
        Re=2500.0,
        transpose=False,
        denorm=True,
        dx=1.0 / 128,
        dy=1.0 / 128,
    ):
        self.constants = constants
        self.Re = Re
        self.transpose = transpose
        self.denorm = denorm
        self.dx = dx
        self.dy = dy

        self.preds = self._process_data(preds)
        self.labels = self._process_data(labels)

        self.u_pred, self.v_pred, self.p_pred = self._extract_uvp(self.preds)
        self.u_gt, self.v_gt, self.p_gt = self._extract_uvp(self.labels)

    def _denormalize(self, tensor):
        mean = np.asarray(self.constants["mean"], dtype=tensor.dtype).flatten()
        std = np.asarray(self.constants["std"], dtype=tensor.dtype).flatten()

        if tensor.ndim == 4:
            mean = mean.reshape(1, -1, 1, 1)
            std = std.reshape(1, -1, 1, 1)
        elif tensor.ndim == 3:
            mean = mean.reshape(-1, 1, 1)
            std = std.reshape(-1, 1, 1)

        return tensor * std + mean

    def _process_data(self, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if self.denorm:
            data = self._denormalize(data)

        if not self.transpose:
            data = np.swapaxes(data, -2, -1)

        return data

    def _extract_uvp(self, data):
        c_dim = 1 if data.ndim == 4 else 0

        if data.shape[c_dim] == 3:
            u_idx, v_idx, p_idx = 0, 1, 2
        else:
            u_idx, v_idx, p_idx = 1, 2, 3

        if data.ndim == 4:
            return data[:, u_idx], data[:, v_idx], data[:, p_idx]
        else:
            return data[u_idx], data[v_idx], data[p_idx]

    def _spatial_grads(self, f):
        original_ndim = f.ndim
        if original_ndim == 2:
            f = f[np.newaxis, ...]

        dy_f = np.zeros_like(f)
        dx_f = np.zeros_like(f)

        dy_f[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2 * self.dy)
        dx_f[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / (2 * self.dx)

        dy_f[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / self.dy
        dy_f[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / self.dy

        dx_f[..., :, 0] = (f[..., :, 1] - f[..., :, 0]) / self.dx
        dx_f[..., :, -1] = (f[..., :, -1] - f[..., :, -2]) / self.dx

        if original_ndim == 2:
            return dy_f[0], dx_f[0]
        return dy_f, dx_f

    def _calc_continuity(self):
        _, du_dx = self._spatial_grads(self.u_pred)
        dv_dy, _ = self._spatial_grads(self.v_pred)

        div = np.mean(np.abs(du_dx + dv_dy))

        _, du_dx_gt = self._spatial_grads(self.u_gt)
        dv_dy_gt, _ = self._spatial_grads(self.v_gt)

        scale = np.mean(np.abs(du_dx_gt)) + np.mean(np.abs(dv_dy_gt)) + 1e-12
        return div, div / scale

    def _calc_momentum(self):
        du_dy, du_dx = self._spatial_grads(self.u_pred)
        dv_dy, dv_dx = self._spatial_grads(self.v_pred)
        dp_dy, dp_dx = self._spatial_grads(self.p_pred)

        d2u_dy2, _ = self._spatial_grads(du_dy)
        _, d2u_dx2 = self._spatial_grads(du_dx)
        d2v_dy2, _ = self._spatial_grads(dv_dy)
        _, d2v_dx2 = self._spatial_grads(dv_dx)

        visc_u = (d2u_dx2 + d2u_dy2) / self.Re
        visc_v = (d2v_dx2 + d2v_dy2) / self.Re

        res_u = (self.u_pred * du_dx + self.v_pred * du_dy) + dp_dx - visc_u
        res_v = (self.u_pred * dv_dx + self.v_pred * dv_dy) + dp_dy - visc_v

        abs_val = np.mean(np.abs(res_u) + np.abs(res_v))

        scale = (
            np.mean(np.abs(self.u_pred * du_dx))
            + np.mean(np.abs(self.v_pred * du_dy))
            + np.mean(np.abs(dp_dx))
            + np.mean(np.abs(visc_u))
            + np.mean(np.abs(self.u_pred * dv_dx))
            + np.mean(np.abs(self.v_pred * dv_dy))
            + np.mean(np.abs(dp_dy))
            + np.mean(np.abs(visc_v))
            + 1e-12
        )

        return abs_val, abs_val / scale

    def _calc_vorticity_error(self):
        _, dv_dx = self._spatial_grads(self.v_pred)
        du_dy, _ = self._spatial_grads(self.u_pred)
        curl_pred = dv_dx - du_dy

        _, dv_dx_gt = self._spatial_grads(self.v_gt)
        du_dy_gt, _ = self._spatial_grads(self.u_gt)
        curl_gt = dv_dx_gt - du_dy_gt

        abs_val = np.mean(np.abs(curl_pred - curl_gt))
        rel_val = abs_val / (np.mean(np.abs(curl_gt)) + 1e-12)

        return abs_val, rel_val

    def _calc_h1_error(self):
        err = 0.0
        norm = 0.0

        for pred, gt in [(self.u_pred, self.u_gt), (self.v_pred, self.v_gt)]:
            dy_p, dx_p = self._spatial_grads(pred)
            dy_gt, dx_gt = self._spatial_grads(gt)

            err += np.mean(np.abs(dx_p - dx_gt)) + np.mean(np.abs(dy_p - dy_gt))
            norm += np.mean(np.abs(dx_gt)) + np.mean(np.abs(dy_gt))

        err /= 2.0
        norm /= 2.0

        return err, err / (norm + 1e-12)

    def _calc_sobolev_split(self):
        preds = self.preds
        targets = self.labels

        current_p = [preds]
        current_t = [targets]

        order0_err = float(np.mean(np.abs(preds - targets)))
        order0_norm = float(np.mean(np.abs(targets)))

        per_order_err = [order0_err]
        per_order_norm = [order0_norm]

        for _ in range(1, 3):
            next_p, next_t = [], []
            order_err = 0.0
            order_norm = 0.0

            for p, t in zip(current_p, current_t):
                dx_p = (p[..., 1:] - p[..., :-1]) / self.dx
                dx_t = (t[..., 1:] - t[..., :-1]) / self.dx

                dy_p = (p[..., 1:, :] - p[..., :-1, :]) / self.dy
                dy_t = (t[..., 1:, :] - t[..., :-1, :]) / self.dy

                order_err += np.mean(np.abs(dx_p - dx_t)) + np.mean(np.abs(dy_p - dy_t))
                order_norm += np.mean(np.abs(dx_t)) + np.mean(np.abs(dy_t))

                next_p.extend([dx_p, dy_p])
                next_t.extend([dx_t, dy_t])

            per_order_err.append(order_err)
            per_order_norm.append(order_norm)
            current_p, current_t = next_p, next_t

        s0_err = per_order_err[0]
        s0_norm = per_order_norm[0]

        s01_err = s0_err + per_order_err[1]
        s01_norm = s0_norm + per_order_norm[1]

        s012_err = s01_err + per_order_err[2]
        s012_norm = s01_norm + per_order_norm[2]

        def rel(e, n):
            return e / (n + 1e-12)

        return {
            "sobolev_s0": s0_err,
            "rel_sobolev_s0": rel(s0_err, s0_norm),
            "sobolev_s01": s01_err,
            "rel_sobolev_s01": rel(s01_err, s01_norm),
            "sobolev_s012": s012_err,
            "rel_sobolev_s012": rel(s012_err, s012_norm),
            "sobolev_time": 0.0,
            "rel_sobolev_time": 0.0,
            "sobolev_0": s0_err,
            "rel_sobolev_0": rel(s0_err, s0_norm),
            "sobolev_01": s01_err,
            "rel_sobolev_01": rel(s01_err, s01_norm),
            "sobolev_012": s012_err,
            "rel_sobolev_012": rel(s012_err, s012_norm),
        }

    def compute(self):
        cont, rel_cont = self._calc_continuity()
        mom, rel_mom = self._calc_momentum()
        vort, rel_vort = self._calc_vorticity_error()
        h1, rel_h1 = self._calc_h1_error()
        sob = self._calc_sobolev_split()

        out = {
            "continuity": cont,
            "rel_continuity": rel_cont,
            "momentum": mom,
            "rel_momentum": rel_mom,
            "vorticity_err": vort,
            "rel_vorticity_err": rel_vort,
            "h1_error": h1,
            "rel_h1_error": rel_h1,
        }
        out.update(sob)
        return out


# -----------------------------
# Quantized evaluation
# -----------------------------
def load_saved_step_sizes(step_path: Path):
    obj = torch.load(step_path, map_location="cpu")

    if "step_sizes_dict" not in obj:
        raise ValueError(f"No step_sizes_dict found in {step_path}")

    return obj["step_sizes_dict"], obj.get("meta", {})


def register_weight_quant_hooks(model, layer_names, step_sizes_dict, device):
    name2mod = dict(model.named_modules())
    handles = []

    def make_hook(w_step_tensor):
        def hook(mod, inp, out):
            x = inp[0]
            w = mod.weight

            w_flat = w.view(w.size(0), -1)
            step = w_step_tensor.view(-1, 1).to(w.device)

            w_quant = torch.round(w_flat / step) * step
            w_quant = w_quant.view_as(w)

            return torch.nn.functional.linear(x, w_quant, mod.bias)

        return hook

    for name in layer_names:
        mod = name2mod.get(name, None)
        if not isinstance(mod, nn.Linear):
            continue
        if name not in step_sizes_dict:
            continue

        item = step_sizes_dict[name]
        w_step = item[0] if isinstance(item, (tuple, list)) else item

        if isinstance(w_step, torch.nn.Parameter):
            w_step = w_step.detach()

        if not torch.is_tensor(w_step):
            w_step = torch.tensor(w_step)

        handles.append(mod.register_forward_hook(make_hook(w_step.to(device))))

    print(f"[INFO] Registered quant hooks: {len(handles)}")
    return handles


def mean_dict(list_of_dicts):
    acc = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            acc[k].append(float(v))
    return {k: float(np.mean(vs)) for k, vs in acc.items()}


def evaluate_physical_metrics(model, val_iter, layer_names, step_sizes_dict, constants, device):
    model = model.to(device).eval()

    handles = register_weight_quant_hooks(
        model=model,
        layer_names=layer_names,
        step_sizes_dict=step_sizes_dict,
        device=device,
    )

    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(val_iter()):
            x = batch["pixel_values"].to(device)
            t = batch["time"].to(device)
            pm = batch["pixel_mask"].to(device)
            y = batch["labels"].to(device)

            outputs = model(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )

            pred = outputs.output

            metric = IncompressibleLoss(
                preds=pred,
                labels=y,
                constants=constants,
                Re=2500.0,
                transpose=False,
                denorm=True,
            ).compute()

            all_metrics.append(metric)

            if (i + 1) % 20 == 0:
                print(f"[INFO] physical eval batch {i + 1}")

    for h in handles:
        h.remove()

    return mean_dict(all_metrics)


def main():
    cfg = PPQConfig()

    # force NS-PwC
    cfg.model_path = os.environ.get("PPQ_MODEL_PATH", "models/NS-PwC-L")
    cfg.data_path = os.environ.get("PPQ_DATA_PATH", "dataset/NS-PwC")
    cfg.dataset_name = os.environ.get(
        "PPQ_DATASET_NAME",
        "fluids.incompressible.PiecewiseConstants",
    )

    cfg.val_batchsize = int(os.environ.get("PPQ_VAL_BATCHSIZE", cfg.val_batchsize))
    cfg.val_steps = int(os.environ.get("PPQ_VAL_STEPS", cfg.val_steps))

    default_step_path = (
        Path(cfg.repo_root)
        / "SAPQ"
        / "artifacts_global"
        / Path(cfg.model_path).name
        / "network_block_sens_sobo"
        / "sapq_global_step_sizes.pt"
    )
    step_path = Path(os.environ.get("SAPQ_STEP_PATH", default_step_path))

    out_dir = (
        Path(cfg.repo_root)
        / "SAPQ"
        / "eval_saved_steps"
        / Path(cfg.model_path).name
        / step_path.parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== EVAL SAVED SAPQ STEPS ==========")
    print("[DEBUG] model_path   =", cfg.model_path)
    print("[DEBUG] data_path    =", cfg.data_path)
    print("[DEBUG] dataset_name =", cfg.dataset_name)
    print("[DEBUG] step_path    =", step_path)
    print("[DEBUG] out_dir      =", out_dir)
    print("===========================================")

    # ----------------------------
    # Load model + data
    # ----------------------------
    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    calib_loader, val_loader, calib_iter, val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=cfg.val_batchsize,
        val_steps=cfg.val_steps,
    )

    candidate_layers = load_candidate_layers(model, Path(cfg.quant_layer_path))
    step_sizes_dict, step_meta = load_saved_step_sizes(step_path)

    constants = val_loader.dataset.constants

    # =========================================================
    # 1. FP evaluation (no quant)
    # =========================================================
    print("\n[FP] Evaluating L1 / RelL1...")
    fp_l1 = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps={},   # no quant
        act_steps=None,
        layer_names=candidate_layers,
        device=device,
    )

    print("[FP] Evaluating physical metrics...")
    fp_phys = evaluate_physical_metrics(
        model=model,
        val_iter=val_iter,
        layer_names=candidate_layers,
        step_sizes_dict={},   # no quant
        constants=constants,
        device=device,
    )

    # =========================================================
    # 2. Quantized evaluation
    # =========================================================
    print("\n[Q] Evaluating L1 / RelL1...")
    q_l1 = evaluate_with_stepsizes(
        model=model,
        val_loader=val_iter,
        weight_steps=step_sizes_dict,
        act_steps=None,
        layer_names=candidate_layers,
        device=device,
    )

    print("[Q] Evaluating physical metrics...")
    q_phys = evaluate_physical_metrics(
        model=model,
        val_iter=val_iter,
        layer_names=candidate_layers,
        step_sizes_dict=step_sizes_dict,
        constants=constants,
        device=device,
    )

    # =========================================================
    # 3. Merge results
    # =========================================================
    results = {}

    # L1
    results["FP_L1"] = fp_l1["l1"]
    results["FP_RelL1"] = fp_l1["rel_l1"]
    results["Q_L1"] = q_l1["l1"]
    results["Q_RelL1"] = q_l1["rel_l1"]

    # physical metrics (prefix)
    for k, v in fp_phys.items():
        results[f"FP_{k}"] = v
    for k, v in q_phys.items():
        results[f"Q_{k}"] = v

    results["meta"] = {
        "model_path": cfg.model_path,
        "dataset_name": cfg.dataset_name,
        "step_path": str(step_path),
        "num_eval_samples": cfg.val_batchsize * cfg.val_steps,
    }

    # ----------------------------
    # save
    # ----------------------------
    save_path = out_dir / "eval_saved_steps_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    # ----------------------------
    # print
    # ----------------------------
    print("\n========== FINAL RESULTS ==========")
    print(f"FP   | L1={results['FP_L1']:.6e} | RelL1={results['FP_RelL1']:.6e}")
    print(f"Quant| L1={results['Q_L1']:.6e} | RelL1={results['Q_RelL1']:.6e}")

    print("\n--- Physical Metrics (FP vs Quant) ---")
    for k in fp_phys.keys():
        print(f"{k}: FP={results['FP_' + k]:.6e} | Q={results['Q_' + k]:.6e}")

    print(f"\n[INFO] Saved results -> {save_path}")

if __name__ == "__main__":
    main()