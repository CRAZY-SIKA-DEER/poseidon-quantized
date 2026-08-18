"""Configuration for stage-one SBPQ experiments on Microsoft Aurora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AuroraSBPQConfig:
    artifacts_root: Path = Path("SBPQ/artifacts/aurora")
    hf_root: Path = Path("aurora_artifacts/huggingface/microsoft_aurora")
    model_name: str = "small"
    checkpoint_name: str | None = None
    input_pickle: str = "aurora-0.25-small-pretrained-test-input.pickle"
    target_pickle: str = "aurora-0.25-small-pretrained-test-output.pickle"
    static_pickle: str = "aurora-0.25-static.pickle"
    data_source: str = "pickle"
    era5_raw_dir: Path = Path("dataset/aurora/era5_025/raw")
    era5_days: tuple[str, ...] = ("2023-01-01", "2023-01-02")
    calib_samples: int = 1
    val_samples: int = 1

    minimum_bits: float = 2.0
    maximum_bits: float = 16.0
    reference_bits: float = 8.0
    init_bits: float = 8.0
    delta_bits: float = 2.0
    beta_kappa: float = 100.0
    beta_prior_scale: float = 1.0
    beta_epsilon: float = 1e-4
    beta_relative_epsilon: float = 1e-12

    learning_rate: float = 3e-5
    num_mc_samples: int = 1
    eta: float = 1e-3
    likelihood_scale: float = 1.0
    num_optimization_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    weight_decay: float = 0.0
    autocast_dtype: str | None = "bfloat16"

    range_percentile: float = 1e-4
    max_quant_layers: int | None = 12
    crop_height: int | None = None
    crop_width: int | None = None
    sensitivity_mode: str = "gradient"
    device: str = "cuda"

    run_name: str = "stage1_small"


def default_checkpoint_for_model(model_name: str) -> str:
    if model_name == "small":
        return "aurora-0.25-small-pretrained.ckpt"
    if model_name in {"pretrained", "full"}:
        return "aurora-0.25-pretrained.ckpt"
    raise ValueError(f"Unsupported Aurora model_name: {model_name}")
