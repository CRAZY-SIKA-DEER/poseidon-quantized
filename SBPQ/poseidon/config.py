from dataclasses import dataclass, field
from pathlib import Path
import os


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    """
    Parse comma-separated floats from an environment variable.
    """
    values = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    if len(values) == 0:
        raise ValueError(
            "Expected at least one comma-separated float."
        )

    return tuple(float(item) for item in values)


@dataclass
class SBPQConfig:
    """
    Central configuration for Sobolev-Guided Beta
    Probabilistic Quantization.
    """

    # ==================================================
    # Project paths
    # ==================================================
    # __file__ returns the path of thsi file, and Path is a object that makes it easier to use .parent()/.mkdir function stuff, and resolve ensures the absolute path, where 3 parents give us the root of this repo
    repo_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    output_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    # ==================================================
    # Model and dataset
    # ==================================================
    model_path: str = "models/NS-PwC-L"
    data_path: str = "dataset/NS-PwC"
    dataset_name: str = "fluids.incompressible.PiecewiseConstants"

    quant_layer_file: str = "L_quantize_layers.pt"

    # ==================================================
    # Runtime
    # ==================================================
    device: str = "cuda"
    random_seed: int = 42
    run_group: str = ""

    # ==================================================
    # Calibration and validation
    # ==================================================
    calib_batch_size: int = 2
    calib_steps: int = 512

    val_batch_size: int = 512
    val_steps: int = 2
    num_workers: int = 0

    # ==================================================
    # Quantization bounds
    # ==================================================
    weight_only: bool = True

    init_bits: float = 8.0
    minimum_bits: float = 1.0
    maximum_bits: float = 16.0

    percentile_prob: float = 1e-4
    range_method: str = "percentile"

    # ==================================================
    # Beta prior
    # ==================================================
    reference_bits: float = 8.0      # B*
    delta_bits: float = 2.0          # Delta_B
    beta_kappa: float = 10.0         # concentration
    beta_prior_scale: float = 1.0

    beta_epsilon: float = 1e-6
    beta_relative_epsilon: float = 1e-12

    task_loss_weight: float = 1.0
    sobolev_loss_weight: float = 1.0

    sensitivity_epsilon: float = 1e-8
    sensitivity_batches: int | None = None
    max_blocks: int | None = None

    # ==================================================
    # PPQ likelihood
    # ==================================================
    eta: float = 1e-6
    num_mc_samples: int = 5
    likelihood_scale: float = 1.0

    # ==================================================
    # Optimization
    # ==================================================
    num_epochs: int = 20
    num_optimization_steps: int | None = None
    learning_rate: float = 1e-4
    updates_per_batch: int = 1
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None

    log_every: int = 1
    eval_every: int = 1


    # ==================================================
    # Sobolev setting
    # ==================================================
    sobolev_order: int = 1

    sobolev_order_weights: float | tuple[float, ...] = (
        1.0,  # order 0
        1.0,  # order 1
    )

    sobolev_norm: str = "l1"
    sobolev_transpose: bool = False

    def __post_init__(self):
        # Allow cluster commands to override important paths.
        self.model_path = os.environ.get(
            "SBPQ_MODEL_PATH",
            self.model_path,
        )
        self.data_path = os.environ.get(
            "SBPQ_DATA_PATH",
            self.data_path,
        )
        self.dataset_name = os.environ.get(
            "SBPQ_DATASET_NAME",
            self.dataset_name,
        )
        self.random_seed = int(os.environ.get(
            "SBPQ_RANDOM_SEED",
            self.random_seed,
        ))
        self.run_group = os.environ.get(
            "SBPQ_RUN_GROUP",
            self.run_group,
        )
        self.num_epochs = int(os.environ.get(
            "SBPQ_NUM_EPOCHS",
            self.num_epochs,
        ))
        self.calib_batch_size = int(os.environ.get(
            "SBPQ_CALIB_BATCH_SIZE",
            self.calib_batch_size,
        ))
        self.calib_steps = int(os.environ.get(
            "SBPQ_CALIB_STEPS",
            self.calib_steps,
        ))
        self.val_batch_size = int(os.environ.get(
            "SBPQ_VAL_BATCH_SIZE",
            self.val_batch_size,
        ))
        self.val_steps = int(os.environ.get(
            "SBPQ_VAL_STEPS",
            self.val_steps,
        ))
        self.num_workers = int(os.environ.get(
            "SBPQ_NUM_WORKERS",
            self.num_workers,
        ))
        if "SBPQ_SENSITIVITY_BATCHES" in os.environ:
            self.sensitivity_batches = int(
                os.environ["SBPQ_SENSITIVITY_BATCHES"]
            )
        if "SBPQ_MAX_BLOCKS" in os.environ:
            self.max_blocks = int(
                os.environ["SBPQ_MAX_BLOCKS"]
            )
        if "SBPQ_NUM_OPTIMIZATION_STEPS" in os.environ:
            self.num_optimization_steps = int(
                os.environ["SBPQ_NUM_OPTIMIZATION_STEPS"]
            )
        self.init_bits = float(os.environ.get(
            "SBPQ_INIT_BITS",
            self.init_bits,
        ))
        self.minimum_bits = float(os.environ.get(
            "SBPQ_MINIMUM_BITS",
            self.minimum_bits,
        ))
        self.maximum_bits = float(os.environ.get(
            "SBPQ_MAXIMUM_BITS",
            self.maximum_bits,
        ))
        self.reference_bits = float(os.environ.get(
            "SBPQ_REFERENCE_BITS",
            self.reference_bits,
        ))
        self.delta_bits = float(os.environ.get(
            "SBPQ_DELTA_BITS",
            self.delta_bits,
        ))
        self.beta_kappa = float(os.environ.get(
            "SBPQ_BETA_KAPPA",
            self.beta_kappa,
        ))
        self.beta_epsilon = float(os.environ.get(
            "SBPQ_BETA_EPSILON",
            self.beta_epsilon,
        ))
        self.beta_relative_epsilon = float(os.environ.get(
            "SBPQ_BETA_RELATIVE_EPSILON",
            self.beta_relative_epsilon,
        ))
        self.percentile_prob = float(os.environ.get(
            "SBPQ_PERCENTILE_PROB",
            self.percentile_prob,
        ))
        self.task_loss_weight = float(os.environ.get(
            "SBPQ_TASK_LOSS_WEIGHT",
            self.task_loss_weight,
        ))
        self.sobolev_loss_weight = float(os.environ.get(
            "SBPQ_SOBOLEV_LOSS_WEIGHT",
            self.sobolev_loss_weight,
        ))
        self.sensitivity_epsilon = float(os.environ.get(
            "SBPQ_SENSITIVITY_EPSILON",
            self.sensitivity_epsilon,
        ))
        self.learning_rate = float(os.environ.get(
            "SBPQ_LEARNING_RATE",
            self.learning_rate,
        ))
        self.beta_prior_scale = float(os.environ.get(
            "SBPQ_BETA_PRIOR_SCALE",
            self.beta_prior_scale,
        ))
        self.num_mc_samples = int(os.environ.get(
            "SBPQ_NUM_MC_SAMPLES",
            self.num_mc_samples,
        ))
        self.eta = float(os.environ.get(
            "SBPQ_ETA",
            self.eta,
        ))
        self.likelihood_scale = float(os.environ.get(
            "SBPQ_LIKELIHOOD_SCALE",
            self.likelihood_scale,
        ))
        self.weight_decay = float(os.environ.get(
            "SBPQ_WEIGHT_DECAY",
            self.weight_decay,
        ))
        if "SBPQ_GRADIENT_CLIP_NORM" in os.environ:
            raw_gradient_clip = os.environ[
                "SBPQ_GRADIENT_CLIP_NORM"
            ].strip().lower()
            self.gradient_clip_norm = (
                None
                if raw_gradient_clip in {"", "none", "null", "0"}
                else float(raw_gradient_clip)
            )
        self.range_method = os.environ.get(
            "SBPQ_RANGE_METHOD",
            self.range_method,
        )
        self.sobolev_order = int(os.environ.get(
            "SBPQ_SOBOLEV_ORDER",
            os.environ.get(
                "SBPQ_SOB_ORDER",
                self.sobolev_order,
            ),
        ))
        self.sobolev_norm = os.environ.get(
            "SBPQ_SOBOLEV_NORM",
            self.sobolev_norm,
        )
        if "SBPQ_SOBOLEV_ORDER_WEIGHTS" in os.environ:
            self.sobolev_order_weights = _parse_float_tuple(
                os.environ["SBPQ_SOBOLEV_ORDER_WEIGHTS"]
            )
        elif "SBPQ_SOB_WEIGHTS" in os.environ:
            self.sobolev_order_weights = _parse_float_tuple(
                os.environ["SBPQ_SOB_WEIGHTS"]
            )
        elif (
            not isinstance(self.sobolev_order_weights, (int, float))
            and len(self.sobolev_order_weights) != self.sobolev_order + 1
        ):
            self.sobolev_order_weights = 1.0

        if self.minimum_bits >= self.maximum_bits:
            raise ValueError(
                "SBPQ minimum_bits must be smaller than maximum_bits."
            )

        if not self.minimum_bits <= self.reference_bits <= self.maximum_bits:
            raise ValueError(
                "SBPQ reference_bits must lie between minimum_bits "
                "and maximum_bits."
            )

        self.output_dir = self.repo_root / "sbpq_outputs"
        self.cache_dir = self.repo_root / "sbpq_cache"

    @property
    def quant_layer_path(self) -> Path:
        return (
            self.repo_root
            / "inspect_layers"
            / self.quant_layer_file
        )

    @property
    def poseidon_artifact_dir(self) -> Path:
        return (
            self.repo_root
            / "SBPQ"
            / "artifacts"
            / "poseidon"
            / Path(self.model_path).name
        )

    def create_directories(self) -> None:
        """Create output and cache directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.poseidon_artifact_dir.mkdir(parents=True, exist_ok=True)
