# SAPQ/sapq_sobo_sensitivity.py
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.poseidon_quant_block import (
    QuantScOTLayer,
    QuantConvNeXtBlock,
    QuantResNetBlock,
)
from BRECQ.quant.poseidon_data_utils import get_model_output_tensor

from scOT.problems.fluids.normalization_constants import CONSTANTS



TARGET_BLOCK_TYPES = (QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock)


class BlockGradHook:
    def __init__(self):
        self.grad = None

    def __call__(self, module, grad_input, grad_output):
        self.grad = grad_output[0].detach()


def reduce_channel_sensitivity(grad: torch.Tensor) -> torch.Tensor:
    g2 = grad.pow(2)

    if g2.dim() == 3:
        return g2.mean(dim=(0, 1))
    elif g2.dim() == 4:
        return g2.mean(dim=(0, 2, 3))
    elif g2.dim() == 2:
        return g2.mean(dim=0)
    else:
        reduce_dims = tuple(range(g2.dim() - 1))
        return g2.mean(dim=reduce_dims)


def compute_sobolev_h1_loss_denorm(
    out_q: torch.Tensor,
    out_fp: torch.Tensor,
    dataset,
    dataset_name: str,
    sobolev_weight: float = 1.0,
    transpose: bool = False,
) -> torch.Tensor:
    """
    Clean Sobolev H1 loss with correct dataset-dependent normalization.

    L = ||q - fp||^2 + λ (||Dx(q-fp)||^2 + ||Dy(q-fp)||^2)

    Uses dataset.constants automatically.
    """

    if out_q.dim() != 4 or out_fp.dim() != 4:
        raise ValueError(
            f"Sobolev H1 expects [B,C,H,W], got {out_q.shape}"
        )

    device = out_q.device

    # ----------------------------
    # 1. Load dataset constants
    # ----------------------------
    constants = dataset.constants

    mean = constants["mean"]
    std = constants["std"]

    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, device=device, dtype=torch.float32)
    else:
        mean = mean.to(device=device, dtype=torch.float32)

    if not torch.is_tensor(std):
        std = torch.tensor(std, device=device, dtype=torch.float32)
    else:
        std = std.to(device=device, dtype=torch.float32)

    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)

    # ----------------------------
    # 2. Denormalize
    # ----------------------------
    q = out_q * std + mean
    fp = out_fp * std + mean

    if transpose:
        q = q.transpose(-2, -1)
        fp = fp.transpose(-2, -1)

    # ----------------------------
    # 3. Select meaningful channels
    # ----------------------------
    C = q.shape[1]
    name_lower = dataset_name.lower()

    if "incompressible" in name_lower or "ns" in name_lower:
        if C == 4:
            indices = [1, 2, 3]   # [rho,u,v,p] → [u,v,p]
        elif C == 3:
            indices = [0, 1, 2]
        elif C == 2:
            indices = [0, 1]
        else:
            indices = list(range(C))
    elif "compressible" in name_lower:
        indices = list(range(C))
    elif "wave" in name_lower:
        indices = list(range(C))
    else:
        indices = list(range(C))

    q = q[:, indices, ...].contiguous()
    fp = fp[:, indices, ...].contiguous()

    # ----------------------------
    # 4. Order 0 (value loss)
    # ----------------------------
    loss0 = torch.nn.functional.mse_loss(q, fp)

    # ----------------------------
    # 5. Order 1 (spatial gradients)
    # ----------------------------
    dx_q = q[..., 1:] - q[..., :-1]
    dx_fp = fp[..., 1:] - fp[..., :-1]

    dy_q = q[..., 1:, :] - q[..., :-1, :]
    dy_fp = fp[..., 1:, :] - fp[..., :-1, :]

    loss1 = (
        torch.nn.functional.mse_loss(dx_q, dx_fp) +
        torch.nn.functional.mse_loss(dy_q, dy_fp)
    )

    # ----------------------------
    # 6. Final Sobolev loss
    # ----------------------------
    return loss0 + sobolev_weight * loss1

def normalize_blockwise_minmax(
    sens_dict: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    out = {}
    for name, sens in sens_dict.items():
        smin = sens.min()
        smax = sens.max()
        out[name] = (sens - smin) / (smax - smin + eps)
    return out


def plot_layer_importance(layer_importance: dict[str, float], save_path: Path):
    names = list(layer_importance.keys())
    values = [layer_importance[n] for n in names]

    plt.figure(figsize=(18, 6))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
    plt.ylabel("Mean normalized channel sensitivity")
    plt.title("SAPQ Sobolev-H1 layer importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_topk_channel_sensitivity(
    sens_dict: dict[str, torch.Tensor],
    save_dir: Path,
    topk_layers: int = 12,
):
    layer_scores = {k: float(v.mean().item()) for k, v in sens_dict.items()}
    top_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)[:topk_layers]

    for layer_name, _ in top_layers:
        sens = sens_dict[layer_name].cpu()

        plt.figure(figsize=(10, 4))
        plt.plot(range(len(sens)), sens)
        plt.xlabel("Channel index")
        plt.ylabel("Normalized sensitivity")
        plt.title(layer_name)
        plt.tight_layout()

        safe_name = layer_name.replace(".", "__")
        plt.savefig(save_dir / f"{safe_name}.png", dpi=220)
        plt.close()


def build_block_sensitivity_to_inner_layer_dict(
    qmodel: PoseidonQuantModel,
    block_sens_dict: dict[str, torch.Tensor],
    fallback: str = "mean",
) -> dict[str, torch.Tensor]:
    from BRECQ.quant.quant_layer import QuantModule

    out = {}
    q_name2mod = dict(qmodel.named_modules())

    for block_wrapped_name, block_sens in block_sens_dict.items():
        block_mod = q_name2mod[block_wrapped_name]
        block_orig_prefix = (
            block_wrapped_name[len("model."):]
            if block_wrapped_name.startswith("model.")
            else block_wrapped_name
        )

        for local_name, local_mod in block_mod.named_modules():
            if not isinstance(local_mod, QuantModule):
                continue
            if local_name == "":
                continue

            orig_layer_name = f"{block_orig_prefix}.{local_name}"
            out_dim = local_mod.weight.shape[0]

            if block_sens.numel() == out_dim:
                out[orig_layer_name] = block_sens.clone().detach().cpu()
            else:
                if fallback == "mean":
                    out[orig_layer_name] = torch.full(
                        (out_dim,),
                        float(block_sens.mean().item()),
                        dtype=block_sens.dtype,
                    )
                elif fallback == "zeros":
                    out[orig_layer_name] = torch.zeros(out_dim, dtype=block_sens.dtype)
                else:
                    raise ValueError(f"Unsupported fallback: {fallback}")

    return out

def load_frozen_calibration_batches(cfg, device: torch.device):
    """
    Load pre-frozen calibration batches from:
        <repo_root>/ppq_artifacts/frozen_calibration_batches.pt

    Expected format:
        list[dict] with keys:
            pixel_values, labels, time, pixel_mask
    """
    #frozen_path = Path(cfg.repo_root) / "ppq_artifacts" / "frozen_calibration_batches.pt"
    #frozen_path = Path(cfg.repo_root) / "ppq_artifacts" / "fluids.incompressible.VortexSheet-calib" / "frozen_calibration_batches.pt"
    frozen_path = (
        Path(cfg.repo_root)
        / "ppq_artifacts"
        / f"{Path(cfg.data_path).name}-calib"
        / "frozen_calibration_batches.pt"
    )
    print(frozen_path)


    if not frozen_path.exists():
        raise FileNotFoundError(
            f"Frozen calibration file not found: {frozen_path}"
        )

    print(f"[INFO] Loading frozen calibration batches from: {frozen_path}")
    frozen_batches = torch.load(frozen_path, map_location="cpu")

    if not isinstance(frozen_batches, list):
        raise ValueError(
            f"Expected frozen_batches to be a list, got {type(frozen_batches)}"
        )
    if len(frozen_batches) == 0:
        raise ValueError("Frozen calibration batch list is empty.")

    first_batch = frozen_batches[0]

    dataset_tag = Path(cfg.data_path).name

    if dataset_tag in {"Wave-Layer", "Wave-Gauss"}:
        required_keys = {"pixel_values", "labels", "time"}
    else:
        required_keys = {"pixel_values", "labels", "time", "pixel_mask"}
    first_batch = frozen_batches[0]

    if not isinstance(first_batch, dict):
        raise ValueError(
            f"Expected each frozen batch to be a dict, got {type(first_batch)}"
        )

    missing = required_keys - set(first_batch.keys())
    if missing:
        raise ValueError(
            f"Frozen calibration batch missing keys: {missing}"
        )

    print(f"[INFO] Loaded {len(frozen_batches)} frozen calibration batches.")

    # move to device ONCE
    for batch in frozen_batches:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
    return frozen_batches


def main():
    cfg = PPQConfig()

    sobolev_weight = float(getattr(cfg, "sobolev_weight", 1.0))
    sobolev_transpose = bool(getattr(cfg, "sobolev_transpose", False))

    # ----------------------------
    # output paths
    # ----------------------------
    model_name = Path(cfg.model_path).name
    out_dir = Path(cfg.repo_root) / "SAPQ" / "prior_sensitivity_sobo" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n========== SOBO SENSITIVITY CONFIG ==========")
    print("[DEBUG] cfg.model_path       =", cfg.model_path)
    print("[DEBUG] cfg.data_path        =", cfg.data_path)
    print("[DEBUG] cfg.dataset_name     =", cfg.dataset_name)
    print("[DEBUG] frozen path will be  =", Path(cfg.repo_root) / "ppq_artifacts" / f"{Path(cfg.data_path).name}-calib" / "frozen_calibration_batches.pt")
    print("[DEBUG] save out_dir         =", out_dir)
    print("[DEBUG] real save out_dir    =", out_dir.resolve())
    print("============================================\n")
    print(out_dir)

    pt_path = out_dir / "prior_sensitivity.pt"
    json_path = out_dir / "prior_sensitivity.json"
    layer_plot_path = out_dir / "layer_importance.png"
    channel_plot_dir = out_dir / "channel_plots"
    channel_plot_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # load model and calibration data
    # ----------------------------
    print("Loading Poseidon model...")
    fp_model, device = load_poseidon_model(cfg.model_path, cfg.device)

    # --------------------------------------------------
    # Load fixed frozen calibration dataset from disk
    # --------------------------------------------------
    print("Loading frozen calibration batches...")
    frozen_batches = load_frozen_calibration_batches(cfg, device=device)

    def frozen_iter():
        for batch in frozen_batches:
            yield batch

    cali_batches = list(frozen_iter())
    print(f"[INFO] Frozen calibration batches: {len(cali_batches)}")

    # ----------------------------
    # denormalization constants
    # ----------------------------
 
    constants = CONSTANTS

    mean = constants["mean"]
    std = constants["std"]

    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, device=device, dtype=torch.float32)
    else:
        mean = mean.to(device=device, dtype=torch.float32)

    if not torch.is_tensor(std):
        std = torch.tensor(std, device=device, dtype=torch.float32)
    else:
        std = std.to(device=device, dtype=torch.float32)

    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1)
    if std.ndim == 1:
        std = std.view(1, -1, 1, 1)

    # ----------------------------
    # wrap quant model
    # ----------------------------
    print("Wrapping model with PoseidonQuantModel...")
    qmodel = PoseidonQuantModel(model=fp_model).to(device).eval()

    print("Initializing quantizer states with first calibration batch...")
    first_batch = cali_batches[0]
    with torch.no_grad():
        qmodel.set_quant_state(True, False)
        _ = qmodel(
            pixel_values=first_batch["pixel_values"].to(device),
            time=first_batch["time"].to(device),
            pixel_mask=(
                first_batch["pixel_mask"].to(device)
                if "pixel_mask" in first_batch
                else None
            ),
            labels=(
                first_batch["labels"].to(device)
                if first_batch.get("labels") is not None
                else None
            ),
        )
        qmodel.set_quant_state(False, False)

    # ----------------------------
    # find target blocks
    # ----------------------------
    target_blocks = []
    for name, module in qmodel.named_modules():
        if isinstance(module, TARGET_BLOCK_TYPES):
            target_blocks.append((name, module))

    print(f"Found {len(target_blocks)} target blocks.")

    # ----------------------------
    # accumulate block-output sensitivity
    # ----------------------------
    sens_sum = {}
    sens_count = defaultdict(int)

    for block_idx, (block_name, block) in enumerate(target_blocks, start=1):
        print(f"\n[{block_idx}/{len(target_blocks)}] Processing block: {block_name}")

        hook = BlockGradHook()
        handle = block.register_full_backward_hook(hook)

        for batch_idx, batch in enumerate(cali_batches, start=1):
            qmodel.zero_grad(set_to_none=True)

            x = batch["pixel_values"].to(device)
            t = batch["time"].to(device)
            pm = batch.get("pixel_mask")
            if pm is not None:
                pm = pm.to(device)
            y = batch.get("labels")
            if y is not None:
                y = y.to(device)

            # clean FP output
            qmodel.set_quant_state(False, False)
            out_fp = qmodel(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            out_fp = get_model_output_tensor(out_fp).detach()

            # quantized output
            qmodel.set_quant_state(True, False)
            out_q = qmodel(
                pixel_values=x,
                time=t,
                pixel_mask=pm,
                labels=y,
            )
            out_q = get_model_output_tensor(out_q)


            calib_loader, _, _, _ = build_poseidon_loaders(
                dataset_name=cfg.dataset_name,
                data_path=cfg.data_path,
                calib_batchsize=cfg.calib_batchsize,
                calib_steps=cfg.calib_steps,
                val_batchsize=cfg.val_batchsize,
                val_steps=cfg.val_steps,
            )

            dataset = calib_loader.dataset
            
            loss = compute_sobolev_h1_loss_denorm(
                out_q=out_q,
                out_fp=out_fp,
                dataset=dataset,                     # pass dataset
                dataset_name=cfg.dataset_name,
                sobolev_weight=sobolev_weight,
                transpose=sobolev_transpose,
            )
            loss.backward()

            if hook.grad is None:
                raise RuntimeError(f"Failed to capture gradient for block {block_name}")

            cur_sens = reduce_channel_sensitivity(hook.grad).detach().cpu()

            if block_name not in sens_sum:
                sens_sum[block_name] = cur_sens.clone()
            else:
                sens_sum[block_name] += cur_sens

            sens_count[block_name] += 1
            hook.grad = None

            if batch_idx % 10 == 0 or batch_idx == len(cali_batches):
                print(f"  batch {batch_idx}/{len(cali_batches)}")

        handle.remove()

    # average over calibration batches
    block_sens_raw = {
        name: sens_sum[name] / max(sens_count[name], 1)
        for name in sens_sum
    }

    # normalize per block
    block_sens_norm = normalize_blockwise_minmax(block_sens_raw)

    # map block sensitivity to inner QuantModule layers
    layer_sens_norm = build_block_sensitivity_to_inner_layer_dict(
        qmodel=qmodel,
        block_sens_dict=block_sens_norm,
        fallback="mean",
    )

    layer_sens_raw = build_block_sensitivity_to_inner_layer_dict(
        qmodel=qmodel,
        block_sens_dict=block_sens_raw,
        fallback="mean",
    )

    layer_importance = {
        name: float(s.mean().item())
        for name, s in layer_sens_norm.items()
    }

    # ----------------------------
    # save
    # ----------------------------
    save_obj = {
        "block_sensitivity_raw": block_sens_raw,
        "block_sensitivity_norm": block_sens_norm,
        "layer_sensitivity_raw": layer_sens_raw,
        "layer_sensitivity_norm": layer_sens_norm,
        "layer_importance": layer_importance,
        "meta": {
            "model_path": cfg.model_path,
            "dataset_name": cfg.dataset_name,
            "data_path": cfg.data_path,
            "num_calibration_batches": len(cali_batches),
            "calib_batchsize": cfg.calib_batchsize,
            "calib_steps": cfg.calib_steps,
            "loss_for_sensitivity": "sobolev_h1_denorm(final_quant_output, final_fp_output)",
            "sobolev_order": 1,
            "sobolev_weight": sobolev_weight,
            "sobolev_transpose": sobolev_transpose,
            "denormalized": True,
            "normalization_constants_source": "scOT.problems.fluids.normalization_constants.CONSTANTS",
            "normalization": "per-block minmax",
            "layer_mapping_fallback": "mean",
        },
    }
    torch.save(save_obj, pt_path)

    with open(json_path, "w") as f:
        json.dump(
            {
                "layer_sensitivity_norm": {
                    k: v.tolist() for k, v in layer_sens_norm.items()
                },
                "layer_importance": layer_importance,
                "meta": save_obj["meta"],
            },
            f,
            indent=2,
        )

    # ----------------------------
    # plots
    # ----------------------------
    plot_layer_importance(layer_importance, layer_plot_path)
    plot_topk_channel_sensitivity(layer_sens_norm, channel_plot_dir, topk_layers=12)

    print(f"\nSaved sensitivity pt   -> {pt_path}")
    print(f"Saved sensitivity json -> {json_path}")
    print(f"Saved layer plot       -> {layer_plot_path}")
    print(f"Saved channel plots    -> {channel_plot_dir}")


if __name__ == "__main__":
    main()