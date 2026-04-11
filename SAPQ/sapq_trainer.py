# SAPQ/sapq_trainer.py
from __future__ import annotations

import torch
import torch.optim as optim

from PPQ.ranges import (
    compute_data_ranges_poseidon,
    load_precalculated_ranges_if_exists,
)
from PPQ.optimize import (
    get_lr_for_epoch,
    clamp_step_sizes_,
    initialize_step_sizes,
    freeze_batches,
)
from PPQ.metrics import (
    build_channel_param_weights,
    compute_avg_bits,
)

from SAPQ.sapq_loss import compute_sapq_loss_with_prior

from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.poseidon_quant_block import (
    QuantScOTLayer,
    QuantConvNeXtBlock,
    QuantResNetBlock,
)
from BRECQ.quant.poseidon_data_utils import (
    save_inp_oup_data,
    save_grad_data,
)
from BRECQ.quant.quant_layer import QuantModule


class SAPQTrainer:
    """
    Structural-Aware Probabilistic Quantization (SAPQ) trainer.

    Design:
    - wrap Poseidon into quantized block structure
    - keep global PPQ-style per-channel step sizes
    - optimize block by block
    - likelihood is fixed to block-wise Fisher-diagonal geometry
    - prior uses precomputed sensitivity from final-output gradients

    Naming detail:
    - original PPQ namespace:
          encoder.layers.0.blocks.0.attention.self.query
    - wrapped PoseidonQuantModel namespace:
          model.encoder.layers.0.blocks.0.attention.self.query
    - this trainer handles the mapping internally
    """

    def __init__(
        self,
        model,
        config,
        layer_names,
        device: str = "cuda",
        weight_quant_params: dict | None = None,
        act_quant_params: dict | None = None,
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        weight_quant_params = {} if weight_quant_params is None else weight_quant_params
        act_quant_params = {} if act_quant_params is None else act_quant_params

        self.qmodel = PoseidonQuantModel(
            model=model,
            weight_quant_params=weight_quant_params,
            act_quant_params=act_quant_params,
        ).to(self.device).eval()

        self.layer_names = layer_names
        self.q_name2mod = dict(self.qmodel.named_modules())

        self.orig_to_wrapped = {}
        for wrapped_name in self.q_name2mod.keys():
            if wrapped_name.startswith("model."):
                self.orig_to_wrapped[wrapped_name[len("model."):]] = wrapped_name

        # global avg-bit reporting uses original namespace
        self.channel_weights = build_channel_param_weights(self.qmodel.model, self.layer_names)

    # ------------------------------------------------------------------
    # Name helpers
    # ------------------------------------------------------------------

    def _to_wrapped_name(self, orig_name: str) -> str:
        return orig_name if orig_name.startswith("model.") else f"model.{orig_name}"

    def _to_orig_name(self, wrapped_name: str) -> str:
        return wrapped_name[len("model."):] if wrapped_name.startswith("model.") else wrapped_name

    # ------------------------------------------------------------------
    # Layer / block discovery
    # ------------------------------------------------------------------

    def _get_target_layers(self, ranges_dict):
        """
        Keep only target layers that:
        - appear in ranges_dict
        - appear in wrapped model as QuantModule
        - have matching per-channel weight range size
        """
        target_layers = []

        for orig_name in self.layer_names:
            if orig_name not in ranges_dict:
                continue

            wrapped_name = self._to_wrapped_name(orig_name)
            mod = self.q_name2mod.get(wrapped_name, None)
            rec = ranges_dict[orig_name]

            if mod is None or not isinstance(mod, QuantModule):
                continue

            w_range = rec.get("weight_ranges", None)
            if w_range is None:
                continue

            if mod.weight is None:
                continue

            if w_range.numel() == mod.weight.shape[0]:
                target_layers.append(orig_name)

        return target_layers

    def _iter_target_blocks(self):
        target_types = (QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock)
        for wrapped_name, module in self.qmodel.named_modules():
            if isinstance(module, target_types):
                yield wrapped_name, module

    # ------------------------------------------------------------------
    # Block-local dict builders
    # ------------------------------------------------------------------

    def _collect_block_local_dicts(
        self,
        block_wrapped_name: str,
        block,
        global_step_sizes_dict,
        global_ranges_dict,
        global_sens_dict,
    ):
        """
        Build block-local dicts keyed by LOCAL QuantModule names inside current block.

        Example:
            block_wrapped_name:
                model.encoder.layers.0.blocks.0

            local QuantModule name:
                attention.self.query

            corresponding original global name:
                encoder.layers.0.blocks.0.attention.self.query
        """
        block_step_sizes_dict = {}
        block_ranges_dict = {}
        block_sens_local_dict = {}
        block_orig_names = []

        block_orig_prefix = self._to_orig_name(block_wrapped_name)

        for local_name, local_mod in block.named_modules():
            if not isinstance(local_mod, QuantModule):
                continue
            if local_name == "":
                continue

            orig_name = f"{block_orig_prefix}.{local_name}"

            if orig_name not in global_step_sizes_dict:
                continue
            if orig_name not in global_ranges_dict:
                continue

            block_step_sizes_dict[local_name] = global_step_sizes_dict[orig_name]
            block_ranges_dict[local_name] = global_ranges_dict[orig_name]
            block_orig_names.append(orig_name)

            if global_sens_dict is not None and orig_name in global_sens_dict:
                block_sens_local_dict[local_name] = global_sens_dict[orig_name]

        return (
            block_step_sizes_dict,
            block_ranges_dict,
            block_sens_local_dict,
            block_orig_names,
        )

    # ------------------------------------------------------------------
    # Main training
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
        print(f"Number of frozen calibration batches: {len(frozen_batches)}")

        # --------------------------------------------------
        # 2) Load / compute ranges in ORIGINAL namespace
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
                    model=self.qmodel.model,
                    dataloader=frozen_iter,
                    device=self.device,
                    layer_names=self.layer_names,
                    percentile_prob=cfg.percentile_prob,
                )

                for name, value in ranges_dict.items():
                    value["weight_ranges"] = value["weight_ranges"].to(self.device)
                    value["activation_ranges"] = value["activation_ranges"].to(self.device)
        else:
            print("Using provided ranges_dict...")

        # --------------------------------------------------
        # 3) Get target layers in ORIGINAL namespace
        # --------------------------------------------------
        target_layers = self._get_target_layers(ranges_dict)
        print(f"Optimizing {len(target_layers)} compatible QuantModule layers.")
        if len(target_layers) == 0:
            raise ValueError("No compatible QuantModule layers found.")

        # --------------------------------------------------
        # 4) Initialize global step sizes in ORIGINAL namespace
        # --------------------------------------------------
        step_sizes_dict, _ = initialize_step_sizes(
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
        # 5) Initial avg bits
        # --------------------------------------------------
        with torch.no_grad():
            avg_bits = compute_avg_bits(
                step_sizes_dict=step_sizes_dict,
                ranges_dict=ranges_dict,
                channel_weights=self.channel_weights,
            )
        print(f"[Init] AvgBits≈{avg_bits:.2f} (target={cfg.init_bits})")

        # --------------------------------------------------
        # 6) Sensitivity dict in ORIGINAL namespace
        # --------------------------------------------------
        if sens_dict is None:
            raise ValueError(
                "sens_dict must be provided to SAPQTrainer.train(...). "
                "Please precompute it with SAPQ/sapq_sensitivity.py."
            )

        history = []

        # --------------------------------------------------
        # 7) Traverse wrapped blocks
        # --------------------------------------------------
        target_blocks = list(self._iter_target_blocks())
        print(f"Found {len(target_blocks)} target blocks for SAPQ.")

        for block_idx, (block_wrapped_name, block) in enumerate(target_blocks, start=1):
            (
                block_step_sizes_dict,
                block_ranges_dict,
                block_sens_dict,
                block_orig_names,
            ) = self._collect_block_local_dicts(
                block_wrapped_name=block_wrapped_name,
                block=block,
                global_step_sizes_dict=step_sizes_dict,
                global_ranges_dict=ranges_dict,
                global_sens_dict=sens_dict,
            )

            if len(block_step_sizes_dict) == 0:
                print(f"[Skip] Block {block_wrapped_name}: no QuantModule step sizes found.")
                continue

            print(
                f"\n[Block {block_idx}/{len(target_blocks)}] {block_wrapped_name} | "
                f"local quant layers: {len(block_step_sizes_dict)}"
            )

            block_params = []
            for step_pair in block_step_sizes_dict.values():
                w_step = step_pair[0]
                if isinstance(w_step, torch.nn.Parameter):
                    block_params.append(w_step)

            if len(block_params) == 0:
                print(f"[Skip] Block {block_wrapped_name}: no trainable step sizes.")
                continue

            optimizer = optim.Adam(block_params, lr=cfg.base_lr)

            # --------------------------------------------------
            # 8) Cache block inputs / outputs
            # --------------------------------------------------
            cached_block_inputs, cached_block_outputs = save_inp_oup_data(
                model=self.qmodel,
                layer=block,
                cali_data=frozen_batches,
                asym=getattr(cfg, "asym", True),
                act_quant=False,
                batch_size=cfg.calib_batchsize,
            )

            # --------------------------------------------------
            # 9) Cache Fisher gradients (always needed now)
            # --------------------------------------------------
            cached_block_grads = save_grad_data(
                model=self.qmodel,
                layer=block,
                cali_data=frozen_batches,
                act_quant=False,
                batch_size=cfg.calib_batchsize,
            )

            # --------------------------------------------------
            # 10) Optimize this block
            # --------------------------------------------------
            num_cached_batches = len(cached_block_inputs)

            for epoch in range(1, cfg.num_epochs + 1):
                lr_epoch = get_lr_for_epoch(
                    epoch=epoch,
                    base_lr=cfg.base_lr,
                    num_epochs=cfg.num_epochs,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_epoch

                for batch_idx in range(num_cached_batches):
                    for _ in range(cfg.updates_per_batch):
                        optimizer.zero_grad()

                        total_loss, like_loss, prior_loss = compute_sapq_loss_with_prior(
                            block=block,
                            block_step_sizes_dict=block_step_sizes_dict,
                            cached_block_inputs=cached_block_inputs,
                            cached_block_outputs=cached_block_outputs,
                            cached_block_grads=cached_block_grads,
                            batch_idx=batch_idx,
                            block_ranges_dict=block_ranges_dict,
                            block_sens_dict=block_sens_dict,
                            num_mc_samples=cfg.num_mc_samples,
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

                with torch.no_grad():
                    avg_bits = compute_avg_bits(
                        step_sizes_dict=step_sizes_dict,
                        ranges_dict=ranges_dict,
                        channel_weights=self.channel_weights,
                    )

                if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.num_epochs:
                    print(
                        f"[Block {block_idx}/{len(target_blocks)} | Epoch {epoch:4d}] "
                        f"LR={lr_epoch:.3e} | "
                        f"Total={total_loss.item():.6f} | "
                        f"Like={like_loss.item():.6f} | "
                        f"Prior={prior_loss.item():.6f} | "
                        f"AvgBits={avg_bits:.2f}"
                    )

                history.append(
                    {
                        "block_idx": block_idx,
                        "block_name": block_wrapped_name,
                        "epoch": epoch,
                        "lr": float(lr_epoch),
                        "total_loss": float(total_loss.item()),
                        "likelihood_loss": float(like_loss.item()),
                        "prior_loss": float(prior_loss.item()),
                        "avg_bits": float(avg_bits),
                        "num_local_layers": len(block_step_sizes_dict),
                        "orig_layer_names": list(block_orig_names),
                    }
                )

                if (
                    eval_callback is not None
                    and cfg.eval_every is not None
                    and epoch % cfg.eval_every == 0
                ):
                    eval_callback(
                        block_idx,
                        block_wrapped_name,
                        epoch,
                        step_sizes_dict,
                        ranges_dict,
                    )

        return step_sizes_dict, ranges_dict, history