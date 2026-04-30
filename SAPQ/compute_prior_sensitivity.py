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


TARGET_BLOCK_TYPES = (QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock)


class BlockGradHook:
    def __init__(self):
        self.grad = None

    def __call__(self, module, grad_input, grad_output):
        # grad_output is usually a tuple; first item is grad wrt block output tensor
        self.grad = grad_output[0].detach()


def reduce_channel_sensitivity(grad: torch.Tensor) -> torch.Tensor:
    """
    Convert gradient wrt block output into one scalar sensitivity per channel.

    Expected block output shapes:
    - [B, HW, C]  for ScOT-like sequence features
    - [B, C, H, W] if some block returns image-like tensors

    Formula idea:
        sens_c = mean over non-channel dims of grad^2
    """
    g2 = grad.pow(2)

    if g2.dim() == 3:
        # [B, T, C] -> reduce over B,T
        sens = g2.mean(dim=(0, 1))   # [C]
    elif g2.dim() == 4:
        # [B, C, H, W] -> reduce over B,H,W
        sens = g2.mean(dim=(0, 2, 3))  # [C]
    elif g2.dim() == 2:
        # [B, C]
        sens = g2.mean(dim=0)
    else:
        # fallback: assume last dim is channel
        reduce_dims = tuple(range(g2.dim() - 1))
        sens = g2.mean(dim=reduce_dims)

    return sens


def normalize_blockwise_minmax(sens_dict: dict[str, torch.Tensor], eps: float = 1e-8):
    """
    Normalize each block's channel sensitivity independently to [0, 1].
    """
    out = {}
    for name, sens in sens_dict.items():
        smin = sens.min()
        smax = sens.max()
        out[name] = (sens - smin) / (smax - smin + eps)
    return out


def plot_layer_importance(layer_importance: dict[str, float], save_path: Path):
    names = list(layer_importance.keys())
    values = [layer_importance[n] for n in names]

    plt.figure(figsize=(16, 6))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
    plt.ylabel("Mean channel sensitivity")
    plt.title("SAPQ layer importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_topk_channel_sensitivity(
    sens_dict: dict[str, torch.Tensor],
    save_dir: Path,
    topk_layers: int = 12,
):
    """
    Plot per-channel sensitivity for the most important layers.
    """
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
        plt.savefig(save_dir / f"{safe_name}.png", dpi=200)
        plt.close()


def main():
    cfg = PPQConfig()

    # --------------------------------------------------
    # paths
    # --------------------------------------------------
    # this path stroes teh older sensitivity
    #out_dir = Path(cfg.repo_root) / "SAPQ" / "prior_sensitivity" / Path(cfg.model_path).name
    # this path is to store the sobolve sensitivity (divergence)
    out_dir = Path(cfg.repo_root) / "SAPQ" / "prior_sensitivity_div" / Path(cfg.model_path).name

    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = out_dir / "prior_sensitivity.pt"
    json_path = out_dir / "prior_sensitivity.json"
    layer_plot_path = out_dir / "layer_importance.png"
    channel_plot_dir = out_dir / "channel_plots"
    channel_plot_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # load model and data
    # --------------------------------------------------
    print("Loading Poseidon model...")
    fp_model, device = load_poseidon_model(cfg.model_path, cfg.device)

    print("Building calibration loader...")
    _calib_loader, _val_loader, calib_iter, _val_iter = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batchsize=cfg.calib_batchsize,
        calib_steps=cfg.calib_steps,
        val_batchsize=cfg.val_batchsize,
        val_steps=cfg.val_steps,
    )

    cali_batches = list(calib_iter())
    print(f"Collected {len(cali_batches)} calibration batches.")

    # --------------------------------------------------
    # wrap quant model
    # --------------------------------------------------
    print("Wrapping quant model...")
    qmodel = PoseidonQuantModel(model=fp_model).to(device).eval()

    # initialize quantizers once
    print("Initializing quantizer states with one batch...")
    first_batch = cali_batches[0]
    with torch.no_grad():
        qmodel.set_quant_state(True, False)
        _ = qmodel(
            pixel_values=first_batch["pixel_values"].to(device),
            time=first_batch["time"].to(device),
            pixel_mask=first_batch["pixel_mask"].to(device),
            labels=(first_batch["labels"].to(device) if first_batch.get("labels") is not None else None),
        )
        qmodel.set_quant_state(False, False)

    # --------------------------------------------------
    # find target blocks
    # --------------------------------------------------
    target_blocks = []
    for name, module in qmodel.named_modules():
        if isinstance(module, TARGET_BLOCK_TYPES):
            target_blocks.append((name, module))

    print(f"Found {len(target_blocks)} target blocks.")

    # --------------------------------------------------
    # accumulate sensitivities
    # --------------------------------------------------
    sens_sum = {}
    sens_count = defaultdict(int)

    for block_name, block in target_blocks:
        print(f"Processing block: {block_name}")
        block_hook = BlockGradHook()
        handle = block.register_full_backward_hook(block_hook)

        for batch_idx, batch in enumerate(cali_batches):
            qmodel.zero_grad(set_to_none=True)

            x = batch["pixel_values"].to(device)
            t = batch["time"].to(device)
            pm = batch["pixel_mask"].to(device)
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

            # final-output loss
            loss = F.mse_loss(out_q, out_fp)
            loss.backward()

            if block_hook.grad is None:
                raise RuntimeError(f"Failed to capture grad for block {block_name}")

            cur_sens = reduce_channel_sensitivity(block_hook.grad).detach().cpu()

            if block_name not in sens_sum:
                sens_sum[block_name] = cur_sens.clone()
            else:
                sens_sum[block_name] += cur_sens

            sens_count[block_name] += 1
            block_hook.grad = None

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(cali_batches):
                print(f"  batch {batch_idx + 1}/{len(cali_batches)}")

        handle.remove()

    # --------------------------------------------------
    # average across batches
    # --------------------------------------------------
    sens_avg = {}
    for block_name, s in sens_sum.items():
        sens_avg[block_name] = s / max(sens_count[block_name], 1)

    # normalize per block
    sens_norm = normalize_blockwise_minmax(sens_avg)

    # layer importance = mean over channels
    layer_importance = {
        name: float(s.mean().item())
        for name, s in sens_norm.items()
    }

    # --------------------------------------------------
    # save
    # --------------------------------------------------
    save_obj = {
        "raw_sensitivity": sens_avg,
        "normalized_sensitivity": sens_norm,
        "layer_importance": layer_importance,
        "meta": {
            "model_path": cfg.model_path,
            "dataset_name": cfg.dataset_name,
            "data_path": cfg.data_path,
            "num_calibration_batches": len(cali_batches),
            "calib_batchsize": cfg.calib_batchsize,
            "calib_steps": cfg.calib_steps,
            "loss": "mse(final_quant_output, final_fp_output)",
            "normalization": "per-block minmax",
        },
    }
    torch.save(save_obj, pt_path)

    with open(json_path, "w") as f:
        json.dump(
            {
                "normalized_sensitivity": {
                    k: v.tolist() for k, v in sens_norm.items()
                },
                "layer_importance": layer_importance,
                "meta": save_obj["meta"],
            },
            f,
            indent=2,
        )

    # --------------------------------------------------
    # plots
    # --------------------------------------------------
    plot_layer_importance(layer_importance, layer_plot_path)
    plot_topk_channel_sensitivity(sens_norm, channel_plot_dir, topk_layers=12)

    print(f"\nSaved sensitivity pt   -> {pt_path}")
    print(f"Saved sensitivity json -> {json_path}")
    print(f"Saved layer plot       -> {layer_plot_path}")
    print(f"Saved channel plots    -> {channel_plot_dir}")


if __name__ == "__main__":
    main()