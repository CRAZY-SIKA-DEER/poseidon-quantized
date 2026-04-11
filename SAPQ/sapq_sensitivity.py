# SAPQ/sapq_sensitivity.py
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
        # grad_output is usually a tuple; first entry is grad wrt block output tensor
        self.grad = grad_output[0].detach()


def reduce_channel_sensitivity(grad: torch.Tensor) -> torch.Tensor:
    """
    Convert gradient wrt block output into one scalar sensitivity per channel.

    Expected block output shapes:
    - [B, T, C]     (most ScOT blocks)
    - [B, C, H, W]  (fallback if some block returns image-like tensor)
    - [B, C]

    Returns:
        sens: [C]
    """
    g2 = grad.pow(2)

    if g2.dim() == 3:
        # [B, T, C] -> average over B,T
        sens = g2.mean(dim=(0, 1))
    elif g2.dim() == 4:
        # [B, C, H, W] -> average over B,H,W
        sens = g2.mean(dim=(0, 2, 3))
    elif g2.dim() == 2:
        # [B, C]
        sens = g2.mean(dim=0)
    else:
        # fallback: assume channel is last dim
        reduce_dims = tuple(range(g2.dim() - 1))
        sens = g2.mean(dim=reduce_dims)

    return sens


def normalize_blockwise_minmax(
    sens_dict: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
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

    plt.figure(figsize=(18, 6))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
    plt.ylabel("Mean normalized channel sensitivity")
    plt.title("SAPQ layer importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_topk_channel_sensitivity(
    sens_dict: dict[str, torch.Tensor],
    save_dir: Path,
    topk_layers: int = 12,
):
    """
    Plot per-channel sensitivity for the top-k most important layers.
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
        plt.savefig(save_dir / f"{safe_name}.png", dpi=220)
        plt.close()


def build_block_sensitivity_to_inner_layer_dict(
    qmodel: PoseidonQuantModel,
    block_sens_dict: dict[str, torch.Tensor],
    fallback: str = "mean",
) -> dict[str, torch.Tensor]:
    """
    Map block-output sensitivity to inner QuantModule layers.

    Output keys are in ORIGINAL PPQ namespace, e.g.
        encoder.layers.0.blocks.0.attention.self.query

    Current rule:
    - if inner QuantModule out_channels == len(block_sensitivity):
        use block sensitivity directly
    - else:
        fallback == "mean": fill all channels with block mean
        fallback == "zeros": fill with zeros
    """
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


def main():
    cfg = PPQConfig()

    # ----------------------------
    # output paths
    # ----------------------------
    model_name = Path(cfg.model_path).name
    out_dir = Path(cfg.repo_root) / "SAPQ" / "prior_sensitivity" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # ----------------------------
    # wrap quant model
    # ----------------------------
    print("Wrapping model with PoseidonQuantModel...")
    qmodel = PoseidonQuantModel(model=fp_model).to(device).eval()

    # initialize quantizer states once
    print("Initializing quantizer states with first calibration batch...")
    first_batch = cali_batches[0]
    with torch.no_grad():
        qmodel.set_quant_state(True, False)
        _ = qmodel(
            pixel_values=first_batch["pixel_values"].to(device),
            time=first_batch["time"].to(device),
            pixel_mask=first_batch["pixel_mask"].to(device),
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

            # final-output loss for sensitivity
            loss = F.mse_loss(out_q, out_fp)
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

    # map block sensitivity to inner QuantModule layers (original PPQ namespace)
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

    # layer importance = mean over channels
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
            "loss_for_sensitivity": "mse(final_quant_output, final_fp_output)",
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