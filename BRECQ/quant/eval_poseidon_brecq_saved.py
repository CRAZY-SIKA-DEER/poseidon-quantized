from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import torch
import numpy as np

from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from scOT.metrics import relative_lp_error, lp_error

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer


def move_batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def validate_poseidon(model, val_loader, device, name="model"):
    model.eval()
    loader = val_loader() if callable(val_loader) else val_loader

    rel_l1_list = []
    abs_l1_list = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        y = batch.get("labels", None)
        if y is None:
            continue

        outputs = model(
            pixel_values=batch["pixel_values"],
            time=batch.get("time", None),
            pixel_mask=batch.get("pixel_mask", None),
            labels=y,
            return_dict=True,
        )

        pred = outputs.output

        pred_np = pred.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        rel_l1_list.append(float(np.mean(relative_lp_error(pred_np, y_np, p=1, return_percent=True))))
        abs_l1_list.append(float(np.mean(lp_error(pred_np, y_np, p=1))))

    mean_l1 = float(sum(abs_l1_list) / len(abs_l1_list))
    mean_rel_l1 = float(sum(rel_l1_list) / len(rel_l1_list))

    print(f"[{name}] L1:     {mean_l1:.6e}")
    print(f"[{name}] RelL1:  {mean_rel_l1:.6e}")

    return {"l1": mean_l1, "rel_l1": mean_rel_l1}


def load_adaround_state(qnn, adaround_path: Path, device):
    state = torch.load(adaround_path, map_location="cpu")

    loaded = 0
    missing = 0

    for name, m in qnn.model.named_modules():
        if not isinstance(m, QuantModule):
            continue

        if name not in state:
            print(f"[WARN] missing AdaRound state for {name}")
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
        else:
            # layer was not AdaRound-reconstructed; keep normal quantizer
            pass
        loaded += 1

    print(f"[INFO] Loaded AdaRound state: {loaded}, missing: {missing}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="models/NS-PwC-L", type=str)
    parser.add_argument("--dataset_name", default="fluids.incompressible.PiecewiseConstants", type=str)
    parser.add_argument("--data_path", default="dataset/NS-PwC", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--n_bits_w", default=4, type=int)
    parser.add_argument("--channel_wise", action="store_true")

    parser.add_argument("--val_batchsize", default=2, type=int)
    parser.add_argument("--val_steps", default=20, type=int)

    parser.add_argument(
        "--adaround_path",
        default="brecq_artifacts/NS-PwC-L/recon/w4/iters10/adaround_state.pt",
        type=str,
    )

    args = parser.parse_args()

    print("Loading FP Poseidon model...")
    fp_model, device = load_poseidon_model(args.model_path, args.device)

    print("Building validation loader...")
    _, _, _, val_iter = build_poseidon_loaders(
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        calib_batchsize=2,
        calib_steps=1,
        val_batchsize=args.val_batchsize,
        val_steps=args.val_steps,
    )

    print("\nEvaluating FP model...")
    validate_poseidon(fp_model, val_iter, device, name="FP")

    print("\nBuilding BRECQ quantized model...")
    wq_params = {
        "n_bits": args.n_bits_w,
        "channel_wise": args.channel_wise,
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

    adaround_path = REPO_ROOT / args.adaround_path
    print(f"[INFO] Loading AdaRound state from: {adaround_path}")
    load_adaround_state(qnn, adaround_path, device)

    qnn.set_quant_state(True, False)

    print("\nEvaluating BRECQ quantized model...")
    validate_poseidon(qnn, val_iter, device, name="BRECQ")


if __name__ == "__main__":
    main()