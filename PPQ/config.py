"""
Configuration module for the PPQ project.

This file defines the `PPQConfig` dataclass, which centralizes all configuration
settings used in the PPQ pipeline, including:

1. Project directory structure
   - Automatically determines the project root based on the location of this file.
   - Constructs commonly used directories (e.g., inspect_layers, artifacts, dynamic_stats).

2. Model and dataset settings
   - Paths and identifiers for the Poseidon model and dataset.

3. Runtime configuration
   - Device selection (e.g., CUDA).

4. Calibration and validation parameters
   - Batch sizes and number of steps used during calibration and evaluation.

5. PPQ optimization parameters
   - Training epochs, Monte Carlo samples, learning rate, bit initialization,
     and related optimization hyperparameters.

6. Bit-cap prior configuration
   - Parameters controlling the average bit-width constraint.

7. Dynamic quantization baselines
   - Paths and settings for loading precomputed dynamic step sizes.

The class also exposes several computed properties (e.g., `quant_layer_path`,
`artifacts_dir`, `lr_dir`) that generate commonly used filesystem paths based
on the configuration.
"""
from dataclasses import dataclass, field
from pathlib import Path
from PPQ.path_utils import PROJECT_STORAGE_ROOT, get_repo_root, ensure_symlink_dir


@dataclass
class PPQConfig:
    # -------------------------
    # Project paths
    # -------------------------
    project_root: Path = Path(__file__).resolve().parent.parent
    inspect_dir: Path = field(init=False)
    artifacts_root: Path = field(init=False)
    dynamic_stats_dir: Path = field(init=False)

    # -------------------------
    # Model / data
    # -------------------------
    model_path: str = "models/NS-PwC-L"
    data_path: str = "dataset/NS-PwC"
    dataset_name: str = "fluids.incompressible.PiecewiseConstants"
    quant_layer_file: str = "L_quantize_layers.pt"

    # -------------------------
    # Device
    # -------------------------
    device: str = "cuda"

    # -------------------------
    # Calibration / validation
    # -------------------------
    calib_batchsize: int = 2
    calib_steps: int = 512 #256 #512 #128  #64
    val_batchsize: int = 2
    val_steps: int = 512

    # -------------------------#
    # PPQ optimization
    # -------------------------
    weight_only: bool = True
    num_epochs: int = 20
    num_mc_samples: int = 10
    base_lr: float = 9.1e-4
    eta: float = 1e-6
    gamma_list: list[float] = field(default_factory=lambda: [0.0])

    percentile_prob: float = 1e-4
    init_bits: int = 8
    bmax_bits: int = 20

    log_every: int = 10
    updates_per_batch: int = 1
    eval_every: int = 1

    # -------------------------
    # Avg bit cap prior
    # -------------------------
    avg_cap_bits: float = 8.0
    avg_cap_lam: float = 400.0
    avg_cap_alpha: float = 10.0

    # -------------------------
    # Dynamic baselines
    # -------------------------
    dyn4_json: str = "NS-PwC-T-dynamic-stepsizes-4.json"
    dyn8_bits: int = 8
    dyn16_bits: int = 16

    def __post_init__(self):
        self.inspect_dir = self.project_root / "inspect_layers"
        self.artifacts_root = self.project_root / "ppq_artifacts"
        self.dynamic_stats_dir = self.project_root / "dynamic_stats"

    @property
    def quant_layer_path(self) -> Path:
        return self.inspect_dir / self.quant_layer_file

    @property
    def dyn4_path(self) -> Path:
        return self.dynamic_stats_dir / self.dyn4_json

    @property
    def model_name(self) -> str:
        return Path(self.model_path).name

    @property
    def repo_root(self) -> Path:
        return get_repo_root()

    @property
    def project_storage_root(self) -> Path:
        return PROJECT_STORAGE_ROOT

    @property
    def artifacts_dir(self) -> Path:
        """
        Clean repo-facing output path, but physically stored under /lus/...
        Repo path:
            <repo>/ppq_artifacts
        Real target:
            /lus/lfs1aip2/projects/u6ey/yiheng.u6ey/poseidon/ppq_artifacts
        """
        repo_artifacts = self.repo_root / "ppq_artifacts"
        storage_artifacts = self.project_storage_root / "ppq_artifacts"
        ensure_symlink_dir(repo_artifacts, storage_artifacts)
        return repo_artifacts

    @property
    def lr_dir(self) -> Path:
        return self.artifacts_dir / "lr"