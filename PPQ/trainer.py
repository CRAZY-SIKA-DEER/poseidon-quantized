from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


from PPQ.ranges import compute_data_ranges_poseidon
from PPQ.poseidon_utils import get_clean_network_outputs_poseidon
from PPQ.loss import (
    compute_mc_loss_with_prior,
    prior_weighted_avg_bits_cap_poseidon,
)
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



def load_precalculated_ranges_if_exists(model_path, percentile_prob, repo_root, device="cpu"):
    """
    Try loading cached ranges from:
      <repo_root>/precalculated_ranges/<model_name>/p<percentile>/ranges.pt

    Returns:
      ranges_dict or None
    """
    model_name = Path(model_path).name
    percentile_tag = f"p{float(percentile_prob):.0e}"
    ranges_path = (
        Path(repo_root)
        / "precalculated_ranges"
        / model_name
        / percentile_tag
        / "ranges.pt"
    )

    if not ranges_path.exists():
        print(f"[INFO] Precalculated ranges not found: {ranges_path}")
        return None

    print(f"[INFO] Loading precalculated ranges from: {ranges_path}")
    obj = torch.load(ranges_path, map_location="cpu")
    ranges_dict = obj["ranges_dict"]

    # move tensors to target device
    out = {}
    for name, value in ranges_dict.items():
        out[name] = {
            "weight_ranges": value["weight_ranges"].to(device),
            "activation_ranges": value["activation_ranges"].to(device),
        }

    print(f"[INFO] Loaded precalculated ranges for {len(out)} layers.")
    return out


class PPQTrainer:
    """
    Trainer for final PPQ optimization on Poseidon.

    Current training path:
      - network-wise MC likelihood
      - MDL prior on weight step sizes
      - optional average-bit cap prior

    Notes:
      - This keeps the original code behavior as much as possible.
      - Some parts are slightly redundant but intentionally preserved for now.
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

        # Precompute channel weights once for avg-bit metrics / avg-bit cap prior
        self.channel_weights = build_channel_param_weights(self.model, self.layer_names)

    def train(
        self,
        dataloader,
        gamma: float = 0.0,
        ranges_dict=None,
        eval_callback=None,
    ):
        """
        Main PPQ optimization loop.

        Args:
            dataloader:
                calibration iterator / dataloader
            gamma:
                MDL prior coefficient
            ranges_dict:
                optional precomputed ranges
            eval_callback:
                optional callback: eval_callback(epoch, step_sizes_dict, ranges_dict)

        Returns:
            step_sizes_dict, ranges_dict, history
        """
        cfg = self.config

        # --------------------------------------------------
        # 1) Freeze calibration batches
        # --------------------------------------------------
        # This is intentional and matches your original logic:
        # reuse the same small calibration set every epoch.
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

                # move computed ranges to target device once
                for name, value in ranges_dict.items():
                    value["weight_ranges"] = value["weight_ranges"].to(self.device)
                    value["activation_ranges"] = value["activation_ranges"].to(self.device)
        else:
            print("Using provided ranges_dict...")

        # --------------------------------------------------
        # 4) Cache clean final network outputs
        # --------------------------------------------------
        # This is required for the current network-wise loss.
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
            f"\nStarting optimization: epochs={cfg.num_epochs}, "
            f"mc_samples={cfg.num_mc_samples}, eta={cfg.eta}, gamma={gamma}, "
            f"base_lr={cfg.base_lr}, updates_per_batch={cfg.updates_per_batch}, "
            f"avg_cap_lam={cfg.avg_cap_lam}"
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

                    # --------------------------------------------------
                    # Network-wise MAP loss
                    # --------------------------------------------------
                    total_loss, like_loss, prior_loss = compute_mc_loss_with_prior(
                        model=self.model,
                        step_sizes_dict=step_sizes_dict,
                        frozen_batches=frozen_batches,
                        clean_net_outputs=clean_net_outputs,
                        ranges_dict=ranges_dict,
                        batch_idx=batch_idx,
                        num_mc_samples=cfg.num_mc_samples,
                        eta=cfg.eta,
                        gamma=gamma,
                        device=self.device,
                    )

                    # --------------------------------------------------
                    # Optional average-bit cap prior
                    # --------------------------------------------------
                    avg_cap_prior = torch.zeros_like(total_loss)
                    if cfg.avg_cap_bits is not None:
                        avg_cap_prior = prior_weighted_avg_bits_cap_poseidon(
                            step_sizes_dict=step_sizes_dict,
                            ranges_dict=ranges_dict,
                            channel_weights=self.channel_weights,
                            target_bits=float(cfg.avg_cap_bits),
                            lam=float(cfg.avg_cap_lam),
                            alpha=float(cfg.avg_cap_alpha),
                        )

                    # NOTE:
                    # This recomposition is slightly redundant because total_loss above
                    # already includes likelihood + prior.
                    # We keep it to match the original code style clearly.
                    total_loss = like_loss  + avg_cap_prior #+ prior_loss

                    total_loss.backward()
                    optimizer.step()

                    # --------------------------------------------------
                    # Clamp step sizes back into valid range
                    # --------------------------------------------------
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
                    f"CapPrior={avg_cap_prior.item():.6f} | "
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
                    "avg_cap_prior": float(avg_cap_prior.item()),
                    "avg_bits": float(avg_bits),
                }
            )

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