"""
Temporary SBPQ test runner for Poseidon.

This script tests the current pipeline:

    1. Load configuration.
    2. Load the Poseidon model.
    3. Build the calibration dataloader.
    4. Detect Poseidon structural blocks.
    5. Build the Sobolev-loss wrapper.
    6. Calculate block sensitivity using one calibration batch.
    7. Normalize and save the sensitivity values.

Run from the repository root with:

    python -m SBPQ.poseidon.run_sbpq_poseidon
"""

from __future__ import annotations

from pathlib import Path

import torch

from SBPQ.poseidon.config import SBPQConfig

from SBPQ.poseidon.poseidon_utils import (
    load_poseidon_model,
    build_poseidon_loaders,
)

from SBPQ.poseidon.blocks import (
    build_poseidon_block_mapping,
    print_poseidon_block_summary,
)

from SBPQ.poseidon.sobolev import (
    compute_poseidon_sobolev_loss,
)

from SBPQ.poseidon.sensitivity import (
    compute_poseidon_block_sensitivity,
    normalize_block_sensitivity,
    save_block_sensitivity,
)

from SBPQ.poseidon.beta_parameter_builder import (
    build_and_save_from_sensitivity_file,
)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    cfg = SBPQConfig()

    print("\n========== SBPQ TEST CONFIG ==========")
    print(f"Repository root: {cfg.repo_root}")
    print(f"Model path:      {cfg.model_path}")
    print(f"Data path:       {cfg.data_path}")
    print(f"Dataset name:    {cfg.dataset_name}")
    print(f"Device:          {cfg.device}")
    print(f"Sobolev order:   {cfg.sobolev_order}")
    print(f"Order weights:   {cfg.sobolev_order_weights}")
    print("======================================\n")

    # ---------------------------------------------------------
    # 2. Load Poseidon model
    # ---------------------------------------------------------
    print("[INFO] Loading Poseidon model...")

    model, device = load_poseidon_model(
        model_path=cfg.model_path,
        device=cfg.device,
    )

    model.eval()

    print(f"[INFO] Model loaded on device: {device}")

    # ---------------------------------------------------------
    # 3. Build calibration and validation loaders
    # ---------------------------------------------------------
    print("[INFO] Building Poseidon dataloaders...")

    (
        calib_loader,
        val_loader,
        calib_iterator,
        val_iterator,
    ) = build_poseidon_loaders(
        dataset_name=cfg.dataset_name,
        data_path=cfg.data_path,
        calib_batch_size=cfg.calib_batch_size,
        calib_steps=cfg.calib_steps,
        val_batch_size=cfg.val_batch_size,
        val_steps=cfg.val_steps,
    )

    calib_dataset = calib_loader.dataset
    val_dataset = val_loader.dataset

    print(
        f"[INFO] Calibration dataset: "
        f"{type(calib_dataset).__name__}"
    )

    # ---------------------------------------------------------
    # 4. Find Poseidon blocks and their Linear layers
    # ---------------------------------------------------------
    print("\n[INFO] Detecting Poseidon blocks...")

    (
        blocks,
        block_to_layers,
        layer_to_block,
        unassigned_layers,
    ) = build_poseidon_block_mapping(model)

    print_poseidon_block_summary(
        blocks=blocks,
        block_to_layers=block_to_layers,
        unassigned_layers=unassigned_layers,
    )

    if len(blocks) == 0:
        raise RuntimeError(
            "The block test failed because no Poseidon blocks were found."
        )

    # ---------------------------------------------------------
    # 5. Create the Sobolev-loss wrapper
    # ---------------------------------------------------------
    # sensitivity.py expects:
    #
    #     loss_function(prediction, reference)
    #
    # This wrapper supplies the additional Poseidon-specific settings.
    def sobolev_loss_function(
        prediction: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return compute_poseidon_sobolev_loss(
            prediction=prediction,
            reference=reference,
            dataset=calib_dataset,
            dataset_name=cfg.dataset_name,
            max_order=cfg.sobolev_order,
            order_weights=cfg.sobolev_order_weights,
            transpose_spatial_axes=cfg.sobolev_transpose,
        )

    # ---------------------------------------------------------
    # 6. Calculate block sensitivity
    # ---------------------------------------------------------
    print("\n[INFO] Starting block-sensitivity test...")

    # Use only one batch per block for the first test.
    raw_sensitivity = compute_poseidon_block_sensitivity(
        model=model,
        dataloader=calib_iterator,
        device=device,
        loss_function=sobolev_loss_function,
        max_batches=1,
    )

    # ---------------------------------------------------------
    # 7. Normalize sensitivity across blocks
    # ---------------------------------------------------------
    normalized_sensitivity = normalize_block_sensitivity(
        sensitivity=raw_sensitivity,
    )

    # ---------------------------------------------------------
    # 8. Print results
    # ---------------------------------------------------------
    print("\n========== BLOCK SENSITIVITY ==========")

    for block_name in raw_sensitivity:
        raw_value = raw_sensitivity[block_name].item()
        normalized_value = normalized_sensitivity[block_name].item()

        print(
            f"{block_name}\n"
            f"    raw        = {raw_value:.6e}\n"
            f"    normalized = {normalized_value:.6f}"
        )

    print("=======================================\n")

    # ---------------------------------------------------------
    # 9. Save test results
    # ---------------------------------------------------------
    save_path = (
        Path(cfg.repo_root)
        / "SBPQ"
        / "artifacts"
        / "poseidon"
        / "block_sensitivity_test.pt"
    )

    metadata = {
        "model_path": str(cfg.model_path),
        "data_path": str(cfg.data_path),
        "dataset_name": cfg.dataset_name,
        "sobolev_order": cfg.sobolev_order,
        "sobolev_order_weights": list(
            cfg.sobolev_order_weights
        ),
        "sobolev_transpose": cfg.sobolev_transpose,
        "calibration_batches_per_block": 1,
        "sensitivity_definition": (
            "mean squared gradient of Sobolev loss "
            "with respect to block output"
        ),
    }

    save_block_sensitivity(
        raw_sensitivity=raw_sensitivity,
        normalized_sensitivity=normalized_sensitivity,
        save_path=save_path,
        metadata=metadata,
    )

    # ---------------------------------------------------------
    # 10. Build and save Beta parameters
    # ---------------------------------------------------------
    beta_parameter_path = (
        Path(cfg.repo_root)
        / "SBPQ"
        / "artifacts"
        / "poseidon"
        / "beta_parameters_test.pt"
    )

    beta_parameters = build_and_save_from_sensitivity_file(
        sensitivity_path=save_path,
        save_path=beta_parameter_path,
        minimum_bits=cfg.minimum_bits,
        maximum_bits=cfg.maximum_bits,
        reference_bits=cfg.reference_bits,
        delta_bits=cfg.delta_bits,
        beta_kappa=cfg.beta_kappa,
        metadata={
            "model_path": str(cfg.model_path),
            "data_path": str(cfg.data_path),
            "dataset_name": cfg.dataset_name,
        },
    )

    print("[SUCCESS] Block and sensitivity pipeline test completed.")


if __name__ == "__main__":
    main()