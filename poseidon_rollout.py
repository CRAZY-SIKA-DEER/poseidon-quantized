from __future__ import annotations

import argparse
import numpy as np
import torch

from scOT.inference import get_trainer, rollout, get_trajectories, get_test_set
from scOT.metrics import relative_lp_error, lp_error


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
    args = parser.parse_args()

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

    trainer = get_trainer(
        model_path=args.model_path,
        batch_size=args.batch_size,
        dataset=test_ds,
        output_all_steps=True,
    )

    # predictions shape:
    # [N, ar_steps, C, H, W]
    preds, _, _ = rollout(
        trainer=trainer,
        dataset=test_ds,
        ar_steps=args.ar_steps,
        output_all_steps=True,
    )

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

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

    labels = labels[: preds.shape[0]]

    delta_t = (args.final_time - args.initial_time) // args.ar_steps

    print("\n========== ROLLOUT RESULTS ==========")

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
