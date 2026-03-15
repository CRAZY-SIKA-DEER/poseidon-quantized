from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from PPQ.ranges import compute_data_ranges_poseidon
from PPQ.poseidon_utils import get_clean_outputs_poseidon
from PPQ.loss import compute_mc_loss_with_prior_layerwise
from PPQ.metrics import build_channel_param_weights, compute_avg_bits
from PPQ.optimize import (
    get_lr_for_epoch,
    clamp_step_sizes_,
    initialize_step_sizes,
    freeze_batches,
    get_compatible_linear_layers,
)


class PPQLayerwiseTrainer:
    """
    Trainer for layer-wise PPQ optimization on Poseidon.

    Current training path:
      - layer-wise MC likelihood
      - MDL prior on weight step sizes
      - no avg-bit cap prior
    """

    def __init__(
        self,
        model,
        config,
        layer_names,
        device="cuda",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

        self.layer_names = layer_names
        self.name2mod = dict(self.model.named_modules())

        self.channel_weights = build_channel_param_weights(self.model, self.layer_names)

    def train(
        self,
        dataloader,
        gamma: float = 0.0,
        ranges_dict=None,
        eval_callback=None,
    ):
        cfg = self.config

        # --------------------------------------------------
        # 1) Freeze calibration batches
        # --------------------------------------------------
        frozen_batches, frozen_iter = freeze_batches(dataloader)
        num_batches = len(frozen_batches)
        print(f"Number of frozen calibration batches: {num_batches}")

        # --------------------------------------------------
        # 2) Candidate Linear layers
        # --------------------------------------------------
        candidate_layers = [
            n for n in self.layer_names
            if isinstance(self.name2mod.get(n, None), nn.Linear)
        ]
        print(f"Candidate Linear layers: {len(candidate_layers)}")

        # --------------------------------------------------
        # 3) Compute or reuse ranges
        # --------------------------------------------------
        if ranges_dict is None:
            print(f"Computing ranges with percentile_prob={cfg.percentile_prob} ...")
            ranges_dict = compute_data_ranges_poseidon(
                model=self.model,
                dataloader=frozen_iter,
                device=self.device,
                layer_names=candidate_layers,
                percentile_prob=cfg.percentile_prob,
            )
        else:
            print("Using provided ranges_dict...")

        # --------------------------------------------------
        # 4) Keep only compatible Linear layers
        # --------------------------------------------------
        target_layers = get_compatible_linear_layers(
            model=self.model,
            candidate_layers=candidate_layers,
            ranges_dict=ranges_dict,
        )
        print(f"Optimizing {len(target_layers)} compatible layers.")
        if len(target_layers) == 0:
            raise ValueError("No compatible layers found.")

        # --------------------------------------------------
        # 5) Cache clean per-layer inputs/outputs
        # --------------------------------------------------
        print("Caching clean per-layer inputs/outputs ...")
        clean_inputs, clean_outputs = get_clean_outputs_poseidon(
            model=self.model,
            dataloader=frozen_batches,
            device=self.device,
            layer_names=target_layers,
        )

        # --------------------------------------------------
        # 6) Initialize step sizes
        # --------------------------------------------------
        step_sizes_dict, params = initialize_step_sizes(
            ranges_dict=ranges_dict,
            target_layers=target_layers,
            init_bits=cfg.init_bits,
            bmax_bits=cfg.bmax_bits,
            device=self.device,
        )

        # --------------------------------------------------
        # 7) Initial bitwidth log
        # --------------------------------------------------
        with torch.no_grad():
            avg_bits = compute_avg_bits(
                step_sizes_dict=step_sizes_dict,
                ranges_dict=ranges_dict,
                channel_weights=self.channel_weights,
            )
        print(f"[Init] AvgBits≈{avg_bits:.2f} (target={cfg.init_bits})")

        # --------------------------------------------------
        # 8) Optimizer
        # --------------------------------------------------
        optimizer = optim.Adam(params, lr=cfg.base_lr)
        history = []

        print(
            f"\nStarting LAYER-WISE optimization: epochs={cfg.num_epochs}, "
            f"mc_samples={cfg.num_mc_samples}, eta={cfg.eta}, gamma={gamma}, "
            f"base_lr={cfg.base_lr}, updates_per_batch={cfg.updates_per_batch}"
        )

        # ==================================================
        # 9) Main training loop
        # ==================================================
        for epoch in range(1, cfg.num_epochs + 1):
            lr_epoch = get_lr_for_epoch(
                epoch=epoch,
                base_lr=cfg.base_lr,
                num_epochs=cfg.num_epochs,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_epoch

            for batch_idx in range(num_batches):
                for _ in range(cfg.updates_per_batch):
                    optimizer.zero_grad()

                    total_loss, like_loss, prior_loss = compute_mc_loss_with_prior_layerwise(
                        model=self.model,
                        step_sizes_dict=step_sizes_dict,
                        clean_inputs=clean_inputs,
                        clean_outputs=clean_outputs,
                        ranges_dict=ranges_dict,
                        batch_idx=batch_idx,
                        num_mc_samples=cfg.num_mc_samples,
                        eta=cfg.eta,
                        gamma=gamma,
                        device=self.device,
                    )

                    total_loss.backward()
                    optimizer.step()

                    clamp_step_sizes_(
                        step_sizes_dict=step_sizes_dict,
                        ranges_dict=ranges_dict,
                        bmax_bits=cfg.bmax_bits,
                        device=self.device,
                    )

            with torch.no_grad():
                avg_bits = compute_avg_bits(
                    step_sizes_dict=step_sizes_dict,
                    ranges_dict=ranges_dict,
                    channel_weights=self.channel_weights,
                )

            if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.num_epochs:
                print(
                    f"[Epoch {epoch:4d}] "
                    f"LR={lr_epoch:.3e} | "
                    f"Total={total_loss.item():.6f} | "
                    f"Like={like_loss.item():.6f} | "
                    f"Prior={prior_loss.item():.6f} | "
                    f"AvgBits={avg_bits:.2f}"
                )

            history.append(
                {
                    "epoch": epoch,
                    "lr": float(lr_epoch),
                    "total_loss": float(total_loss.item()),
                    "likelihood_loss": float(like_loss.item()),
                    "prior_loss": float(prior_loss.item()),
                    "avg_bits": float(avg_bits),
                }
            )

            if (
                eval_callback is not None
                and cfg.eval_every is not None
                and epoch % cfg.eval_every == 0
            ):
                eval_callback(epoch, step_sizes_dict, ranges_dict)

        return step_sizes_dict, ranges_dict, history