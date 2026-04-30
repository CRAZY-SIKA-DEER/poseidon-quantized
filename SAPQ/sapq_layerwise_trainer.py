from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from PPQ.ranges import (
    compute_data_ranges_poseidon,
    load_precalculated_ranges_if_exists,
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

from SAPQ.sapq_loss import compute_prior_by_mode
from SAPQ.sapq_layerwise_cache_utils import (
    get_model_layerio_root,
    load_cached_layer_io_for_layers,
    build_layer_name_to_file_map,
    load_single_layer_io,
)


class SAPQLayerwiseTrainer:
    """
    Layerwise-likelihood SAPQ trainer.

    Design:
    - use cached clean per-layer input/output from disk
    - optimize all target layer step sizes jointly
    - likelihood is layerwise reconstruction
    - prior is selectable by prior_mode:
          "ppq"
          "block_no_sens"
          "block_sens"

    Notes:
    - this trainer reuses cached layer IO generated on the shared frozen
      calibration dataset
    - current likelihood is plain layerwise reconstruction error, matching
      the PPQ layerwise style
    - sensitivity affects only the prior, not the likelihood
    """

    def __init__(
        self,
        model,
        config,
        layer_names,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

        self.layer_names = layer_names
        self.name2mod = dict(self.model.named_modules())

        self.channel_weights = build_channel_param_weights(self.model, self.layer_names)



    def _compute_single_layer_likelihood_streaming(
        self,
        layer_name: str,
        step_sizes_dict,
        layer_name_to_file,
        batch_idx: int,
        num_mc_samples: int = 10,
        eta: float = 1e-4,
    ):
        """
        Compute Monte Carlo layerwise reconstruction loss for ONE layer only.
        This is true layer-by-layer optimization.
        """
        device = self.device

        if layer_name not in step_sizes_dict:
            raise ValueError(f"{layer_name} not found in step_sizes_dict")

        if layer_name not in layer_name_to_file:
            raise ValueError(f"{layer_name} not found in cached layer IO")

        module = self.name2mod.get(layer_name, None)
        if module is None or not isinstance(module, nn.Linear):
            raise ValueError(f"{layer_name} is not an nn.Linear layer")

        layer_file = layer_name_to_file[layer_name]

        loaded_name, x_list, y_list = load_single_layer_io(
            layer_file=layer_file,
            to_float32=True,
        )

        if loaded_name != layer_name:
            raise ValueError(
                f"Layer name mismatch: requested '{layer_name}', loaded '{loaded_name}'"
            )

        if batch_idx >= len(x_list) or batch_idx >= len(y_list):
            del x_list, y_list
            return torch.zeros((), device=device)

        x_clean = x_list[batch_idx].to(device)
        y_clean = y_list[batch_idx].to(device)
        w_clean = module.weight.to(device)

        step_entry = step_sizes_dict[layer_name]
        w_step = step_entry[0].to(device) if isinstance(step_entry, tuple) else step_entry.to(device)

        if x_clean.shape[-1] != w_clean.shape[1]:
            raise ValueError(
                f"{layer_name}: x_clean last dim {x_clean.shape[-1]} != in_features {w_clean.shape[1]}"
            )

        if w_step.numel() != w_clean.shape[0]:
            raise ValueError(
                f"{layer_name}: w_step numel {w_step.numel()} != out_features {w_clean.shape[0]}"
            )

        mc_losses = []

        for _ in range(num_mc_samples):
            noise = torch.rand_like(w_clean) - 0.5
            w_noisy = w_clean + noise * w_step.view(-1, *([1] * (w_clean.dim() - 1)))

            y_noisy = torch.nn.functional.linear(x_clean, w_noisy, module.bias)

            loss_elem = torch.mean((y_noisy - y_clean) ** 2) / (2.0 * eta)
            mc_losses.append(loss_elem)

        loss = torch.stack(mc_losses).mean()

        del x_list, y_list, x_clean, y_clean, w_clean, w_step, mc_losses

        return loss

    # ------------------------------------------------------------------
    # internal: layerwise likelihood
    # ------------------------------------------------------------------

    def _compute_single_layer_likelihood_from_cached_batch(
        self,
        layer_name: str,
        cached_batch: dict,
        step_sizes_dict,
        num_mc_samples: int = 10,
        eta: float = 1e-4,
    ):
        device = self.device

        module = self.name2mod[layer_name]
        if not isinstance(module, nn.Linear):
            raise ValueError(f"{layer_name} is not nn.Linear")

        x_clean = cached_batch["x_pre"].to(device=device, dtype=torch.float32)
        y_clean = cached_batch["y_post"].to(device=device, dtype=torch.float32)

        w_clean = module.weight.to(device)
        bias = module.bias.to(device) if module.bias is not None else None

        step_entry = step_sizes_dict[layer_name]
        w_step = step_entry[0] if isinstance(step_entry, tuple) else step_entry
        w_step = w_step.to(device)

        mc_losses = []
        for _ in range(num_mc_samples):
            noise = torch.rand_like(w_clean) - 0.5
            w_noisy = w_clean + noise * w_step.view(-1, 1)

            y_noisy = torch.nn.functional.linear(x_clean, w_noisy, bias)

            loss = torch.mean((y_noisy - y_clean) ** 2) / (2.0 * eta)
            mc_losses.append(loss)

        return torch.stack(mc_losses).mean()

    # ------------------------------------------------------------------
    # main training
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader,
        ranges_dict=None,
        sens_dict=None,
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
        # 5) Build cached layer-IO file map only (do NOT load all layer IO)
        # --------------------------------------------------
        layerio_root = get_model_layerio_root(
            repo_root=cfg.repo_root,
            model_path=cfg.model_path,
        )
        print(f"[INFO] Using cached layer IO from: {layerio_root}")

        layer_name_to_file = build_layer_name_to_file_map(layerio_root)

        missing_layers = [name for name in target_layers if name not in layer_name_to_file]
        if missing_layers:
            raise ValueError(
                "The following target layers are missing from cached layer IO:\n"
                + "\n".join(missing_layers[:20])
                + ("\n..." if len(missing_layers) > 20 else "")
            )

        print(f"[INFO] Layer-IO files ready for {len(target_layers)} target layers.")

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
        if sens_dict is not None:
            sens_dict = {
                name: sens_dict[name]
                for name in target_layers
                if name in sens_dict
            }
            print(f"[INFO] Using sensitivity for {len(sens_dict)} target layers.")
        else:
            print("[INFO] sens_dict is None. This is okay for prior_mode='ppq' or 'block_no_sens'.")

        # --------------------------------------------------
        # 8) Initial bitwidth log
        # --------------------------------------------------
        with torch.no_grad():
            avg_bits = compute_avg_bits(
                step_sizes_dict=step_sizes_dict,
                ranges_dict=ranges_dict,
                channel_weights=self.channel_weights,
            )
        print(
            f"[Init] AvgBits≈{avg_bits:.2f} "
            f"(target={getattr(cfg, 'target_bits', cfg.init_bits)}) | "
            f"prior_mode={getattr(cfg, 'prior_mode', 'block_sens')}"
        )

        # --------------------------------------------------
        # 9) Optimizer
        # --------------------------------------------------
        #optimizer = optim.Adam(params, lr=cfg.base_lr)
        # --------------------------------------------------
        # 9) True layer-by-layer optimization
        # --------------------------------------------------
        history = []

        print(
            f"\nStarting TRUE SAPQ LAYER-BY-LAYER optimization: "
            f"layers={len(target_layers)}, epochs_per_layer={cfg.num_epochs}, "
            f"mc_samples={cfg.num_mc_samples}, eta={cfg.eta}, "
            f"base_lr={cfg.base_lr}, updates_per_batch={cfg.updates_per_batch}, "
            f"prior_mode={getattr(cfg, 'prior_mode', 'block_sens')}, "
            f"target_bits={getattr(cfg, 'target_bits', cfg.init_bits)}, "
            f"sigma0={getattr(cfg, 'sigma0', 0.5)}, "
            f"alpha={getattr(cfg, 'alpha', 1.0)}, "
            f"prior_scale={getattr(cfg, 'prior_scale', 1.0)}"
        )

        # ==================================================
        # 10) Main true layer-by-layer training loop
        # ==================================================
        for layer_idx, layer_name in enumerate(target_layers, start=1):
            print(
                f"\n========== Optimizing layer {layer_idx}/{len(target_layers)}: {layer_name} ==========",
                flush=True,
            )

            # ---------------- LOAD LAYER IO ONCE ----------------
            layer_file = layer_name_to_file[layer_name]
            print("[DEBUG] layer_file =", layer_file, flush=True)
            print("[DEBUG] file_size =", layer_file.stat().st_size, flush=True)
            obj = torch.load(layer_file, map_location="cpu")

            if obj["layer_name"] != layer_name:
                raise ValueError(
                    f"Layer mismatch: expected {layer_name}, got {obj['layer_name']}"
                )

            cached_batches = obj["batches"]

            print(
                f"[INFO] Loaded {len(cached_batches)} cached batches for {layer_name}",
                flush=True,
            )
            # ---------------------------------------------------

            # Optimize ONLY this layer's step size
            step_entry = step_sizes_dict[layer_name]
            layer_param = step_entry[0] if isinstance(step_entry, tuple) else step_entry

            optimizer = optim.Adam([layer_param], lr=cfg.base_lr)

            # Single-layer dicts for prior calculation
            single_step_sizes_dict = {layer_name: step_sizes_dict[layer_name]}
            single_ranges_dict = {layer_name: ranges_dict[layer_name]}

            if sens_dict is not None and layer_name in sens_dict:
                single_sens_dict = {layer_name: sens_dict[layer_name]}
            else:
                single_sens_dict = None

            last_total_loss = None
            last_like_loss = None
            last_prior_loss = None

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

                        like_loss = self._compute_single_layer_likelihood_from_cached_batch(
                            layer_name=layer_name,
                            cached_batch=cached_batches[batch_idx],
                            step_sizes_dict=step_sizes_dict,
                            num_mc_samples=cfg.num_mc_samples,
                            eta=cfg.eta,
                        )

                        prior_loss = compute_prior_by_mode(
                            prior_mode=str(getattr(cfg, "prior_mode", "block_sens")),
                            step_sizes_dict=single_step_sizes_dict,
                            ranges_dict=single_ranges_dict,
                            sens_dict=single_sens_dict,
                            gamma=float(getattr(cfg, "gamma", 0.005)),
                            b_target=float(getattr(cfg, "target_bits", cfg.init_bits)),
                            sigma0=float(getattr(cfg, "sigma0", 0.5)),
                            alpha=float(getattr(cfg, "alpha", 1.0)),
                            prior_scale=float(getattr(cfg, "prior_scale", 1.0)),
                        )

                        total_loss = like_loss + prior_loss
                        total_loss.backward()
                        optimizer.step()

                        clamp_step_sizes_(
                            step_sizes_dict=single_step_sizes_dict,
                            ranges_dict=single_ranges_dict,
                            bmax_bits=cfg.bmax_bits,
                            device=self.device,
                            weight_only=cfg.weight_only,
                        )

                        last_total_loss = total_loss
                        last_like_loss = like_loss
                        last_prior_loss = prior_loss

                # with torch.no_grad():
                #     avg_bits = compute_avg_bits(
                #         step_sizes_dict=step_sizes_dict,
                #         ranges_dict=ranges_dict,
                #         channel_weights=self.channel_weights,
                #     )

                if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.num_epochs:
                    print(
                        f"[Layer {layer_idx:4d}/{len(target_layers)} | Epoch {epoch:4d}] "
                        f"LR={lr_epoch:.3e} | "
                        f"Total={last_total_loss.item():.6f} | "
                        f"Like={last_like_loss.item():.6f} | "
                        f"Prior={last_prior_loss.item():.6f}", 
                        #f"GlobalAvgBits={avg_bits:.2f}",
                        flush=True,
                    )

                history.append(
                    {
                        "layer_idx": int(layer_idx),
                        "layer_name": layer_name,
                        "epoch": int(epoch),
                        "lr": float(lr_epoch),
                        "total_loss": float(last_total_loss.item()),
                        "likelihood_loss": float(last_like_loss.item()),
                        "prior_loss": float(last_prior_loss.item()),
                        #"avg_bits": float(avg_bits),
                        "prior_mode": str(getattr(cfg, "prior_mode", "block_sens")),
                        "likelihood_mode": "true_layer_by_layer",
                    }
                )

            # Optional evaluation after finishing each layer
            if (
                eval_callback is not None
                and cfg.eval_every is not None
                and layer_idx % cfg.eval_every == 0
            ):
                eval_callback(layer_idx, step_sizes_dict, ranges_dict)
        
        # --------------------------------------------------
        # 11) Final avg bitwidth
        # --------------------------------------------------
        with torch.no_grad():
            final_avg_bits = compute_avg_bits(
                step_sizes_dict=step_sizes_dict,
                ranges_dict=ranges_dict,
                channel_weights=self.channel_weights,
            )

        print(f"\n[Final] AvgBits≈{final_avg_bits:.4f}")

        history.append(
            {
                "stage": "final",
                "final_avg_bits": float(final_avg_bits),
                "prior_mode": str(getattr(cfg, "prior_mode", "block_sens")),
                "likelihood_mode": "true_layer_by_layer",
            }
        )

        return step_sizes_dict, ranges_dict, history