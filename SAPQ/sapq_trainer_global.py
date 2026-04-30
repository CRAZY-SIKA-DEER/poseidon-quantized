# SAPQ/sapq_trainer_global.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from PPQ.ranges import (
    compute_data_ranges_poseidon,
    load_precalculated_ranges_if_exists,
)
from PPQ.poseidon_utils import get_clean_network_outputs_poseidon
from PPQ.metrics import (
    build_channel_param_weights,
    compute_avg_bits,
)
from PPQ.optimize import (
    get_lr_for_epoch,
    clamp_step_sizes_,
    initialize_step_sizes,
    freeze_batches,
    get_compatible_linear_layers,
)

from SAPQ.sapq_loss import compute_sapq_loss_with_prior_global


class SAPQTrainerGlobal:
    """
    Global / pure-network SAPQ trainer.

    Compared with block-wise SAPQ:
    - no block traversal
    - no block-local caching
    - optimize all channel step sizes together
    - likelihood uses final model output only
    - prior uses SAPQ sensitivity-aware prior
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

        # Precompute channel weights once for avg-bit metrics
        self.channel_weights = build_channel_param_weights(self.model, self.layer_names)

    def train(
        self,
        dataloader,
        ranges_dict=None,
        sens_dict=None,
        eval_callback=None,
    ):
        """
        Main global SAPQ optimization loop.

        Args:
            dataloader:
                calibration iterator / dataloader
            ranges_dict:
                optional precomputed ranges
            sens_dict:
                global channel-wise sensitivity dict in ORIGINAL namespace
            eval_callback:
                optional callback: eval_callback(epoch, step_sizes_dict, ranges_dict)

        Returns:
            step_sizes_dict, ranges_dict, history
        """
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
        # 3) Load / compute / reuse ranges
        # --------------------------------------------------
        if ranges_dict is None:
            ranges_dict = load_precalculated_ranges_if_exists(
                model_path=cfg.model_path,
                percentile_prob=cfg.percentile_prob,
                repo_root=cfg.repo_root,
                device=self.device,
            )

            if ranges_dict is None:
                print(f"Computing ranges with percentile_prob={cfg.percentile_prob} ...")
                ranges_dict = compute_data_ranges_poseidon(
                    model=self.model,
                    dataloader=frozen_iter,
                    device=self.device,
                    layer_names=candidate_layers,
                    percentile_prob=cfg.percentile_prob,
                )

                for name, value in ranges_dict.items():
                    value["weight_ranges"] = value["weight_ranges"].to(self.device)
                    value["activation_ranges"] = value["activation_ranges"].to(self.device)
        else:
            print("Using provided ranges_dict...")

        # --------------------------------------------------
        # 4) Cache clean final network outputs
        # --------------------------------------------------
        print("Caching clean network outputs (final model output) ...")
        clean_net_outputs = get_clean_network_outputs_poseidon(
            model=self.model,
            frozen_batches=frozen_batches,
            device=self.device,
        )

        # --------------------------------------------------
        # 5) Keep only compatible Linear layers
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
        # 6) Initialize step sizes
        # --------------------------------------------------
        step_sizes_dict, params = initialize_step_sizes(
            ranges_dict=ranges_dict,
            target_layers=target_layers,
            init_bits=cfg.init_bits,
            bmax_bits=cfg.bmax_bits,
            device=self.device,
            model_path=cfg.model_path,
            percentile_prob=cfg.percentile_prob,
            repo_root=cfg.repo_root,
            weight_only=cfg.weight_only,
        )

        # --------------------------------------------------
        # 7) Sensitivity dict
        # --------------------------------------------------
        if sens_dict is None:
            raise ValueError(
                "sens_dict must be provided to SAPQTrainerGlobal.train(...). "
                "Please precompute it with SAPQ/sapq_sensitivity.py."
            )

        # Optional: keep only sensitivities for target layers
        sens_dict = {
            name: sens_dict[name]
            for name in target_layers
            if name in sens_dict
        }
        print(f"Using sensitivity for {len(sens_dict)} target layers.")

        # --------------------------------------------------
        # 8) Initial bitwidth log
        # --------------------------------------------------
        with torch.no_grad():
            avg_bits = compute_avg_bits(
                step_sizes_dict=step_sizes_dict,
                ranges_dict=ranges_dict,
                channel_weights=self.channel_weights,
            )
        print(f"[Init] AvgBits≈{avg_bits:.2f} (target={getattr(cfg, 'target_bits', cfg.init_bits)})")

        # --------------------------------------------------
        # 9) Optimizer
        # --------------------------------------------------
        optimizer = optim.Adam(params, lr=cfg.base_lr)

        history = []

        print(
            f"\nStarting global SAPQ optimization: epochs={cfg.num_epochs}, "
            f"min_epochs={getattr(cfg, 'min_epochs', 10)}, "
            f"early_stop_bits={getattr(cfg, 'early_stop_bits', 4.0)}, "
            f"mc_samples={cfg.num_mc_samples}, eta={cfg.eta}, "
            f"base_lr={cfg.base_lr}, updates_per_batch={cfg.updates_per_batch}, "
            f"target_bits={getattr(cfg, 'target_bits', cfg.init_bits)}, "
            f"sigma0={getattr(cfg, 'sigma0', 0.5)}, "
            f"alpha={getattr(cfg, 'alpha', 1.0)}, "
            f"prior_scale={getattr(cfg, 'prior_scale', 1.0)}"
        )

        # ==================================================
        # 10) Main training loop
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

                    total_loss, like_loss, prior_loss = compute_sapq_loss_with_prior_global(
                        model=self.model,
                        step_sizes_dict=step_sizes_dict,
                        frozen_batches=frozen_batches,
                        clean_net_outputs=clean_net_outputs,
                        ranges_dict=ranges_dict,
                        sens_dict=sens_dict,
                        batch_idx=batch_idx,
                        num_mc_samples=cfg.num_mc_samples,
                        eta=cfg.eta,
                        prior_mode=str(getattr(cfg, "prior_mode", "block_sens")),
                        b_target=float(getattr(cfg, "target_bits", cfg.init_bits)),
                        sigma0=float(getattr(cfg, "sigma0", 0.5)),
                        alpha=float(getattr(cfg, "alpha", 1.0)),
                        prior_scale=float(getattr(cfg, "prior_scale", 1.0)),
                        device=self.device,
                    )

                    total_loss.backward()
                    optimizer.step()

                    clamp_step_sizes_(
                        step_sizes_dict=step_sizes_dict,
                        ranges_dict=ranges_dict,
                        bmax_bits=cfg.bmax_bits,
                        device=self.device,
                        weight_only=cfg.weight_only,
                    )

            # --------------------------------------------------
            # Epoch-end avg bits
            # --------------------------------------------------
            with torch.no_grad():
                avg_bits = compute_avg_bits(
                    step_sizes_dict=step_sizes_dict,
                    ranges_dict=ranges_dict,
                    channel_weights=self.channel_weights,
                )

            # --------------------------------------------------
            # Logging
            # --------------------------------------------------
            if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.num_epochs:
                print(
                    f"[Epoch {epoch:4d}] "
                    f"LR={lr_epoch:.3e} | "
                    f"Total={total_loss.item():.6f} | "
                    f"Like={like_loss.item():.6f} | "
                    f"Prior={prior_loss.item():.6f} | "
                    f"AvgBits={avg_bits:.2f}"
                )

            # --------------------------------------------------
            # Save training history
            # --------------------------------------------------
            history.append(
                {
                    "epoch": epoch,
                    "lr": float(lr_epoch),
                    "total_loss": float(total_loss.item()),
                    "likelihood_loss": float(like_loss.item()),
                    "prior_loss": float(prior_loss.item()),
                    "avg_bits": float(avg_bits),
                    "likelihood_mode": "network_global",
                    "prior_mode": str(getattr(cfg, "prior_mode", "block_sens")),
                }
            )

            # --------------------------------------------------
            # early stopping & make sure we optimize under 4 bits
            # --------------------------------------------------

            min_epochs = int(getattr(cfg, "min_epochs", 10))
            early_stop_bits = float(getattr(cfg, "early_stop_bits", 4.0))

            if epoch >= min_epochs and avg_bits < early_stop_bits:
                print(
                    f"[EARLY STOP] epoch={epoch} | "
                    f"AvgBits={avg_bits:.4f} < {early_stop_bits:.4f}"
                )
                break

            # --------------------------------------------------
            # Optional evaluation callback
            # --------------------------------------------------
            if (
                eval_callback is not None
                and cfg.eval_every is not None
                and epoch % cfg.eval_every == 0
            ):
                eval_callback(epoch, step_sizes_dict, ranges_dict)

        return step_sizes_dict, ranges_dict, history