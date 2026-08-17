#coding:utf8
"""
Physics-informed evaluation metrics for VICON.

适配 VICON 的 7 通道统一布局:
    ch0: rho (density)       COMP/EULER active
    ch1: vx  (velocity-x)    all active
    ch2: vy  (velocity-y)    all active
    ch3: p   (pressure)      COMP/EULER active
    ch4: (unused)
    ch5: u   (scalar field)  NS2D active
    ch6: type (node_type)    not used in metrics

Metrics computed (absolute + relative):
    - Sobolev:     multi-order spatial derivative error (all datasets, supervised)
    - Continuity:  div(v) = 0 residual (NS2D only, unsupervised)
    - Vorticity:   curl error (NS2D only, supervised)
"""
import numpy as np
import torch


# =====================================================================
# Helpers
# =====================================================================

def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return t


# =====================================================================
# Per-dataset metric computers
# =====================================================================

def compute_ns2d_metrics(pred, label, dx=1.0/128, dy=1.0/128, order=2):
    """
    NS2D metrics. pred/label: [bs, pairs, 6, H, W] denormalized, no type channel.
    Active channels: ch1(vx), ch2(vy), ch5(u).
    Returns dict with both absolute and relative (rel_) versions.
    """
    pred, label = _to_numpy(pred), _to_numpy(label)

    vx_p, vy_p = pred[:, :, 1], pred[:, :, 2]
    vx_g, vy_g = label[:, :, 1], label[:, :, 2]

    # ---- Continuity: div(v) = 0 (central diff) ----
    dvx_dx_p = (vx_p[..., :, 2:] - vx_p[..., :, :-2]) / (2.0 * dx)
    dvy_dy_p = (vy_p[..., 2:, :] - vy_p[..., :-2, :]) / (2.0 * dy)
    div_pred = dvx_dx_p[..., 1:-1, :] + dvy_dy_p[..., :, 1:-1]
    continuity = np.mean(np.abs(div_pred))

    # Relative continuity: normalize by GT velocity gradient scale
    dvx_dx_g = (vx_g[..., :, 2:] - vx_g[..., :, :-2]) / (2.0 * dx)
    dvy_dy_g_full = (vy_g[..., 2:, :] - vy_g[..., :-2, :]) / (2.0 * dy)
    vel_grad_scale = (np.mean(np.abs(dvx_dx_g[..., 1:-1, :]))
                      + np.mean(np.abs(dvy_dy_g_full[..., :, 1:-1])) + 1e-12)
    rel_continuity = continuity / vel_grad_scale

    # ---- Vorticity error (central diff) ----
    dvy_dx_p = (vy_p[..., :, 2:] - vy_p[..., :, :-2]) / (2.0 * dx)
    dvx_dy_p = (vx_p[..., 2:, :] - vx_p[..., :-2, :]) / (2.0 * dy)
    curl_pred = dvy_dx_p[..., 1:-1, :] - dvx_dy_p[..., :, 1:-1]

    dvy_dx_g = (vy_g[..., :, 2:] - vy_g[..., :, :-2]) / (2.0 * dx)
    dvx_dy_g = (vx_g[..., 2:, :] - vx_g[..., :-2, :]) / (2.0 * dy)
    curl_gt = dvy_dx_g[..., 1:-1, :] - dvx_dy_g[..., :, 1:-1]
    vorticity_error = np.mean(np.abs(curl_pred - curl_gt))
    rel_vorticity_error = vorticity_error / (np.mean(np.abs(curl_gt)) + 1e-12)

    # ---- Sobolev (传入动态 order) ----
    sobolev, rel_sobolev = _calc_sobolev(
        pred[:, :, [1, 2, 5]], label[:, :, [1, 2, 5]], dx, dy, order=order
    )

    return {
        "sobolev":           float(sobolev),
        "rel_sobolev":       float(rel_sobolev),
        "continuity":        float(continuity),
        "rel_continuity":    float(rel_continuity),
        "vorticity_err":     float(vorticity_error),
        "rel_vorticity_err": float(rel_vorticity_error),
    }


def compute_compressible_metrics(pred, label, dx=1.0/128, dy=1.0/128, order=1):
    """
    COMPRESSIBLE2D / EULER2D metrics.
    Active channels: ch0(rho), ch1(vx), ch2(vy), ch3(p).
    """
    pred, label = _to_numpy(pred), _to_numpy(label)

    # ---- Sobolev (传入动态 order) ----
    sobolev, rel_sobolev = _calc_sobolev(
        pred[:, :, [0, 1, 2, 3]], label[:, :, [0, 1, 2, 3]], dx, dy, order=order
    )

    return {
        "sobolev":     float(sobolev),
        "rel_sobolev": float(rel_sobolev),
    }


def _calc_sobolev(pred, label, dx, dy, order=2):
    """
    Multi-order absolute Sobolev metric (finite diff, L1).

    Returns:
        (absolute, relative)
        absolute = sum over orders 0..order of mean|pred_deriv - label_deriv|
        relative = absolute / (sum over orders 0..order of mean|label_deriv|)
    """
    current_preds = [pred]
    current_labels = [label]

    # Order 0
    total = float(np.mean(np.abs(pred - label)))
    total_norm = float(np.mean(np.abs(label)))

    for k in range(1, order + 1):
        next_p, next_l = [], []
        order_loss = 0.0
        order_norm = 0.0

        for p, l in zip(current_preds, current_labels):
            dx_p = (p[..., :, 1:] - p[..., :, :-1]) / dx
            dx_l = (l[..., :, 1:] - l[..., :, :-1]) / dx
            dy_p = (p[..., 1:, :] - p[..., :-1, :]) / dy
            dy_l = (l[..., 1:, :] - l[..., :-1, :]) / dy

            order_loss += np.mean(np.abs(dx_p - dx_l)) + np.mean(np.abs(dy_p - dy_l))
            order_norm += np.mean(np.abs(dx_l)) + np.mean(np.abs(dy_l))

            if k < order:
                next_p.extend([dx_p, dy_p])
                next_l.extend([dx_l, dy_l])

        total += order_loss
        total_norm += order_norm
        current_preds = next_p
        current_labels = next_l

    rel = total / (total_norm + 1e-12)
    return total, rel


# =====================================================================
# Top-level dispatch
# =====================================================================

METRIC_DISPATCH = {
    "NS2D":            compute_ns2d_metrics,
    "COMPRESSIBLE2D":  compute_compressible_metrics,
    "EULER2D":         compute_compressible_metrics,
}


@torch.inference_mode()
def evaluate_physics_metrics(trainer, dataloaders, min_ex=None, check_gt=False, sobolev_order_map=None):
    """
    Evaluate physics metrics across all datasets.
    """
    trainer.model.eval()
    if min_ex is None:
        min_ex = trainer.loss_cfg.min_ex

    # 默认 mapping，防止外部调用未传递参数时报错
    if sobolev_order_map is None:
        sobolev_order_map = {"NS2D": 2, "COMPRESSIBLE2D": 1, "EULER2D": 1}

    all_metrics = {}
    gt_metrics = {}

    for dataset_type, loader in dataloaders.items():
        compute_fn = METRIC_DISPATCH.get(dataset_type)
        if compute_fn is None:
            print(f"  [Skip] No physics metrics defined for {dataset_type}")
            continue

        # 动态获取当前数据集的 Sobolev 阶数
        current_order = sobolev_order_map.get(dataset_type, 1)

        batch_metrics = []
        batch_gt_metrics = []

        for batch_cnt, batch in enumerate(loader):
            _, pairs, t_in, t_out, delta_t = batch
            pairs = trainer._move_to_device(pairs)

            data, mean, std = trainer._data_preprocess(pairs)
            output = trainer._model_forward(data)

            pred_denorm = (output[:, min_ex:, :-1] * std[:, :, :-1] + mean[:, :, :-1]).detach()
            label_raw = data[1]
            label_denorm = (label_raw[:, min_ex:, :-1] * std[:, :, :-1] + mean[:, :, :-1]).detach()

            # 将 order 作为参数传递下去
            metrics = compute_fn(pred_denorm, label_denorm, order=current_order)
            batch_metrics.append(metrics)

            if check_gt:
                # 同样也将 order 传给 GT 检查
                gt_check = compute_fn(label_denorm, label_denorm, order=current_order)
                batch_gt_metrics.append(gt_check)

        if batch_metrics:
            avg = {}
            for key in batch_metrics[0]:
                avg[key] = float(np.mean([m[key] for m in batch_metrics]))
            all_metrics[dataset_type] = avg

        if check_gt and batch_gt_metrics:
            avg_gt = {}
            for key in batch_gt_metrics[0]:
                avg_gt[key] = float(np.mean([m[key] for m in batch_gt_metrics]))
            gt_metrics[dataset_type] = avg_gt

    if check_gt and gt_metrics:
        print("\n" + "="*60)
        print("  SANITY CHECK: GT vs GT")
        print("="*60)
        for ds, metrics in gt_metrics.items():
            print(f"\n  Dataset: {ds}")
            for name, val in metrics.items():
                if name == "continuity":
                    print(f"    {name:>20s} = {val:.2e}  (baseline)")
                elif name.startswith("rel_"):
                    # relative of GT vs GT: should be 0 for error metrics, or small
                    status = "✅" if abs(val) < 1e-3 else "⚠️"
                    print(f"    {name:>20s} = {val:.2e}  {status}")
                else:
                    status = "✅" if abs(val) < 1e-5 else "❌ BUG"
                    print(f"    {name:>20s} = {val:.2e}  {status} (should be ~0)")
        print("="*60 + "\n")

    return all_metrics