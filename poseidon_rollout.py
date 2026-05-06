from __future__ import annotations
import os
os.environ["WANDB_DISABLED"] = "true"


import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from scOT.inference import get_trainer, rollout, get_trajectories, get_test_set
from scOT.metrics import relative_lp_error, lp_error

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.quant_layer import QuantModule
from BRECQ.quant.adaptive_rounding import AdaRoundQuantizer


def load_saved_steps(path: Path):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "step_sizes_dict" in obj:
        return obj["step_sizes_dict"]

    if isinstance(obj, dict):
        return obj

    raise ValueError(f"Unsupported step-size file format: {path}")


def normalize_saved_steps(saved_steps):
    """
    Convert saved SAPQ/PPQ step dict to:
        name -> step_tensor
    """
    steps = {}

    for name, item in saved_steps.items():
        if isinstance(item, (tuple, list)):
            step = item[0]
        elif isinstance(item, dict) and "step" in item:
            step = item["step"]
        else:
            step = item

        steps[name] = step.detach().cpu()

    return steps


def compute_uniform_symmetric_steps(model: nn.Module, bits: int):
    steps = {}
    qmax = (2 ** (bits - 1)) - 1

    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue

            w = mod.weight.detach()
            w_flat = w.view(w.size(0), -1)
            max_abs = w_flat.abs().max(dim=1).values
            step = torch.clamp(max_abs / float(qmax), min=1e-8)

            steps[name] = step

    return steps


def register_uniform_qdq_hooks(model: nn.Module, steps):
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

    for name, mod in model.named_modules():
        if name not in steps:
            continue
        if not isinstance(mod, nn.Linear):
            continue

        handles.append(mod.register_forward_hook(make_hook(steps[name])))

    return handles

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

    # Important for scOT Trainer rollout logic
    qnn.config = fp_model.config

    return qnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="fluids.incompressible.PiecewiseConstants")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--initial_time", type=int, default=0)
    parser.add_argument("--final_time", type=int, default=15)
    parser.add_argument("--ar_steps", type=int, default=3)
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--sapq_steps", type=str, default=None)
    parser.add_argument("--ppq_steps", type=str, default=None)
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--brecq_iters", type=int, default=10000)
    args = parser.parse_args()
    repo_root = Path(args.repo_root) if args.repo_root is not None else Path(args.model_path).parents[1]
    model_tag = Path(args.model_path).name

    assert (args.final_time - args.initial_time) % args.ar_steps == 0

    print("========== POSEIDON ROLLOUT CONFIG ==========")
    print(f"model_path:    {args.model_path}")
    print(f"data_path:     {args.data_path}")
    print(f"dataset:       {args.dataset}")
    print(f"initial_time:  {args.initial_time}")
    print(f"final_time:    {args.final_time}")
    print(f"ar_steps:      {args.ar_steps}")
    print(f"batch_size:    {args.batch_size}")
    print("=============================================")

    # Dataset gives:
    # input  = frame initial_time
    # label  = frame final_time
    # time   = (final_time - initial_time) / constants["time"]
    test_ds = get_test_set(
        dataset=args.dataset,
        data_path=args.data_path,
        initial_time=args.initial_time,
        final_time=args.final_time,
        dataset_kwargs={},
    )

    # Optional: limit number of trajectories
    if args.num_trajectories is not None:
        test_ds.length = min(test_ds.length, args.num_trajectories)

    all_preds = {}

    methods = [
        ("FP", "fp", None),
        ("Uniform-w8", "uniform", 8),
        ("Uniform-w4", "uniform", 4),
    ]

    if args.sapq_steps is not None:
        methods.append(("SAPQ", "saved", Path(args.sapq_steps)))

    if args.ppq_steps is not None:
        methods.append(("PPQ", "saved", Path(args.ppq_steps)))

    for bits in [8, 4]:
        brecq_path = (
            repo_root
            / "brecq_artifacts"
            / model_tag
            / "recon"
            / f"w{bits}"
            / f"iters{args.brecq_iters}"
            / "adaround_state.pt"
        )

    methods.append((f"BRECQ-w{bits}", "brecq", (bits, brecq_path)))

    for method_name, method_type, method_value in methods:
        print(f"\n[INFO] Running rollout: {method_name}")

        trainer = get_trainer(
            model_path=args.model_path,
            batch_size=args.batch_size,
            dataset=test_ds,
            output_all_steps=True,
        )
        trainer.args.remove_unused_columns = False

        if method_type == "brecq":
            bits, brecq_path = method_value

            if not brecq_path.exists():
                print(f"[WARN] Skip {method_name}, missing file: {brecq_path}")
                continue

            trainer.model = build_brecq_model(
                fp_model=trainer.model,
                adaround_path=brecq_path,
                device=trainer.model.device if hasattr(trainer.model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                n_bits_w=bits,
            )

            print(f"[INFO] Built {method_name}")
            print(f"[INFO] BRECQ path: {brecq_path}")

        handles = []

        if method_type == "uniform":
            bits = method_value
            steps = compute_uniform_symmetric_steps(trainer.model, bits=bits)
            handles = register_uniform_qdq_hooks(trainer.model, steps)
            print(f"[INFO] Registered Uniform-w{bits} hooks for {len(handles)} Linear layers")

        elif method_type == "saved":
            step_path = method_value
            saved_steps = load_saved_steps(step_path)
            steps = normalize_saved_steps(saved_steps)
            handles = register_uniform_qdq_hooks(trainer.model, steps)
            print(f"[INFO] Registered {method_name} hooks for {len(handles)} Linear layers")
            print(f"[INFO] Step path: {step_path}")

        preds, _, _ = rollout(
            trainer=trainer,
            dataset=test_ds,
            ar_steps=args.ar_steps,
            output_all_steps=True,
        )

        for h in handles:
            h.remove()

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        all_preds[method_name] = preds

    # labels shape:
    # [N, ar_steps, C, H, W]
    labels = get_trajectories(
        dataset=args.dataset,
        data_path=args.data_path,
        ar_steps=args.ar_steps,
        initial_time=args.initial_time,
        final_time=args.final_time,
        dataset_kwargs={},
    )

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    first_preds = next(iter(all_preds.values()))
    labels = labels[: first_preds.shape[0]]

    delta_t = (args.final_time - args.initial_time) // args.ar_steps

    print("\n========== ROLLOUT RESULTS ==========")

    for method_name, preds in all_preds.items():
        print(f"\n----- {method_name} -----")

        for step in range(args.ar_steps):
            curr_time = args.initial_time + (step + 1) * delta_t

            pred_step = preds[:, step]
            gt_step = labels[:, step]

            l1 = float(np.mean(lp_error(pred_step, gt_step, p=1)))
            rel_l1 = float(np.mean(relative_lp_error(pred_step, gt_step, p=1, return_percent=True)))

            print(
                f"t={curr_time:03d} | "
                f"L1={l1:.6e} | "
                f"RelL1={rel_l1:.6e}"
            )


if __name__ == "__main__":
    main()
