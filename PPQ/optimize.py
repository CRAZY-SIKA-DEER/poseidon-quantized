"""
Utility functions for preparing and managing step-size optimization in the PPQ pipeline.

This module provides helper functions used during training and setup of the
probabilistic post-training quantization (PPQ) process. These utilities handle
learning-rate scheduling, step-size initialization and constraints, calibration
batch freezing, and selection of valid quantization layers.

Functions
---------

get_lr_for_epoch(...)
    Returns a piecewise learning rate based on the current epoch. The schedule
    gradually reduces the learning rate across four phases of training to help
    stabilize step-size optimization.

clamp_step_sizes_(...)
    Performs an in-place clamp of learned step sizes so that they stay within
    valid per-channel bounds. The minimum step corresponds to the maximum
    allowed bit precision (bmax_bits), while the maximum step corresponds to
    the full channel range.

initialize_step_sizes(...)
    Initializes learnable per-channel step sizes for both weights and
    activations using precomputed range statistics. The initial step size is
    derived from a target bit-width and then clamped to valid bounds.

freeze_batches(...)
    Converts a dataloader or iterator into a fixed list of batches. This is
    used to create deterministic calibration inputs so that model evaluations
    during optimization always run on the same data.

get_compatible_linear_layers(...)
    Filters candidate layer names and keeps only Linear layers whose stored
    range statistics match the actual model dimensions. This ensures that
    quantization parameters align correctly with layer inputs and outputs.
"""


import torch
import os

def get_lr_for_epoch(
    epoch: int,
    base_lr: float = 1e-6,
    num_epochs: int = 4,
) -> float:
    """
    Piecewise learning-rate schedule over training epochs.

    Current schedule:
      first 25%   -> 1.0   * base_lr
      second 25%  -> 0.1   * base_lr
      third 25%   -> 0.01  * base_lr
      last 25%    -> 0.001 * base_lr
    """
    if epoch <= num_epochs * 0.25:
        return 1.0 * base_lr
    elif epoch <= num_epochs * 0.50:
        return 0.1 * base_lr
    elif epoch <= num_epochs * 0.75:
        return 0.01 * base_lr
    else:
        return 0.001 * base_lr


def clamp_step_sizes_(
    step_sizes_dict,
    ranges_dict,
    bmax_bits: int,
    device: str | torch.device = "cuda",
    eps: float = 1e-8,
    weight_only: bool = True,
):
    """
    In-place clamp of step sizes into valid per-channel ranges.

    For each layer:
      min step = range / 2^bmax_bits
      max step = range
    """
    #device = torch.device(device)

    with torch.no_grad():
        for name, (w_step, a_step) in step_sizes_dict.items():
            if name not in ranges_dict:
                continue

            w_range = ranges_dict[name]["weight_ranges"]
            w_min = w_range / (2 ** bmax_bits)
            w_min.clamp_(min=eps)
            #w_min = torch.maximum(w_min, torch.full_like(w_min, eps))
            w_step.clamp_(min=w_min, max=w_range)

            if (not weight_only) and isinstance(a_step, torch.nn.Parameter):
                a_range = ranges_dict[name]["activation_ranges"]
                a_min = a_range / (2 ** bmax_bits)
                a_min.clamp_(min=eps)
                #a_min = torch.maximum(a_min, torch.full_like(a_min, eps))
                a_step.clamp_(min=a_min, max=a_range)


# def initialize_step_sizes(
#     ranges_dict,
#     target_layers,
#     init_bits: int,
#     bmax_bits: int,
#     device: str | torch.device = "cuda",
#     eps: float = 1e-8,
# ):
#     """
#     Initialize learnable per-channel step sizes from ranges_dict.

#     Init:
#       step_init = range / 2^init_bits

#     Clamp:
#       step in [range / 2^bmax_bits, range]
#     """
#     device = torch.device(device)
#     step_sizes_dict = {}
#     params = []

#     for name in target_layers:
#         if name not in ranges_dict:
#             continue

#         w_range = ranges_dict[name]["weight_ranges"].to(device)
#         a_range = ranges_dict[name]["activation_ranges"].to(device)

#         w_step_init = w_range / (2 ** init_bits)
#         a_step_init = a_range / (2 ** init_bits)

#         w_step_min = w_range / (2 ** bmax_bits)
#         a_step_min = a_range / (2 ** bmax_bits)

#         w_step_min = torch.maximum(w_step_min, torch.full_like(w_step_min, eps))
#         a_step_min = torch.maximum(a_step_min, torch.full_like(a_step_min, eps))

#         w_step_init = torch.clamp(w_step_init, min=w_step_min, max=w_range)
#         a_step_init = torch.clamp(a_step_init, min=a_step_min, max=a_range)

#         w_step = torch.nn.Parameter(w_step_init.clone().detach())
#         a_step = torch.nn.Parameter(a_step_init.clone().detach())

#         step_sizes_dict[name] = (w_step, a_step)
#         params.extend([w_step, a_step])

#     return step_sizes_dict, params


def initialize_step_sizes(
    ranges_dict,
    target_layers,
    init_bits: int,
    bmax_bits: int,
    device: str | torch.device = "cuda",
    eps: float = 1e-8,
    *,
    model_path: str | None = None,
    percentile_prob: float | None = None,
    repo_root: str | os.PathLike | None = None,
    weight_only: bool = True,
):
    """
    Initialize learnable per-channel step sizes.

    Priority:
      1) If cached initial step sizes exist, load them.
      2) Otherwise compute from ranges_dict.

    Cached file layout:
      <repo_root>/initial_step_sizes/<model_name>/p<percentile>/<bits>bit.pt

    Init formula (fallback compute path):
      step_init = range / 2^init_bits

    Clamp:
      step in [range / 2^bmax_bits, range]
    """
    from pathlib import Path

    device = torch.device(device)
    step_sizes_dict = {}
    params = []

    loaded_cache = False
    cached_steps = None

    if model_path is not None and percentile_prob is not None and repo_root is not None:
        model_name = Path(model_path).name
        percentile_tag = f"p{float(percentile_prob):.0e}"
        cache_path = (
            Path(repo_root)
            / "initial_step_sizes"
            / model_name
            / percentile_tag
            / f"{int(init_bits)}bit.pt"
        )

        if cache_path.exists():
            print(f"[INFO] Loading cached initial step sizes from: {cache_path}")
            cache_obj = torch.load(cache_path, map_location="cpu")
            cached_steps = cache_obj["step_sizes"]
            loaded_cache = True
            '''
            loaded_cache = True if cache exists otherwise it stays False
            '''
        else:
            print(f"[INFO] Cached initial step sizes not found: {cache_path}")
            print("[INFO] Falling back to on-the-fly initialization from ranges_dict.")

    if loaded_cache:
        for name in target_layers:
            if name not in cached_steps:
                continue

            w_step_init, a_step_init = cached_steps[name]
            w_step_init = w_step_init.to(device)
            a_step_init = a_step_init.to(device)

            if name in ranges_dict:
                w_range = ranges_dict[name]["weight_ranges"]#.to(device)    #no need to move the tensor id ranegs_dict is already onto device
                a_range = ranges_dict[name]["activation_ranges"]#.to(device)

                w_step_min = w_range / (2 ** bmax_bits)
                a_step_min = a_range / (2 ** bmax_bits)

                #w_step_min = torch.maximum(w_step_min, torch.full_like(w_step_min, eps))
                #above way of clamping will create a new empty tensor
                w_step_min.clamp_(min=eps)
                #a_step_min = torch.maximum(a_step_min, torch.full_like(a_step_min, eps))
                a_step_min.clamp_(min=eps)

                w_step_init = torch.clamp(w_step_init, min=w_step_min, max=w_range)
                a_step_init = torch.clamp(a_step_init, min=a_step_min, max=a_range)

            w_step = torch.nn.Parameter(w_step_init.clone().detach())

            if weight_only:
                a_step = a_step_init.clone().detach()
                step_sizes_dict[name] = (w_step, a_step)
                params.append(w_step)
                '''
                params is the list passed into:
                optimizer = optim.Adam(params, lr=cfg.base_lr)
                So only tensors inside params are treated as trainable optimizer parameters.
                If you do:
                params.append(w_step)
                but do not add a_step, then Adam only updates w_step.
                '''
            else:
                a_step = torch.nn.Parameter(a_step_init.clone().detach())
                step_sizes_dict[name] = (w_step, a_step)
                params.extend([w_step, a_step])

        print(f"[INFO] Loaded initial step sizes for {len(step_sizes_dict)} layers from cache.")
        return step_sizes_dict, params

    for name in target_layers:
        if name not in ranges_dict:
            continue

        w_range = ranges_dict[name]["weight_ranges"]#.to(device)     #no need to move the tensor id ranegs_dict is already onto device
        a_range = ranges_dict[name]["activation_ranges"]#.to(device)

        #here the initial step sizes are really calculated
        w_step_init = w_range / (2 ** init_bits)
        a_step_init = a_range / (2 ** init_bits)

        w_step_min = w_range / (2 ** bmax_bits)
        a_step_min = a_range / (2 ** bmax_bits)

        w_step_min = torch.maximum(w_step_min, torch.full_like(w_step_min, eps))
        a_step_min = torch.maximum(a_step_min, torch.full_like(a_step_min, eps))

        w_step_init = torch.clamp(w_step_init, min=w_step_min, max=w_range)
        a_step_init = torch.clamp(a_step_init, min=a_step_min, max=a_range)

        w_step = torch.nn.Parameter(w_step_init.clone().detach())
        
        if weight_only:
            a_step = a_step_init.clone().detach()
            step_sizes_dict[name] = (w_step, a_step)
            params.append(w_step)
        else:
            a_step = torch.nn.Parameter(a_step_init.clone().detach())
            step_sizes_dict[name] = (w_step, a_step)
            params.extend([w_step, a_step])

    print(f"[INFO] Computed initial step sizes for {len(step_sizes_dict)} layers on the fly.")
    return step_sizes_dict, params


def freeze_batches(dataloader):
    """
    Materialize a dataloader / iterator into a fixed list of batches.

    Returns:
      frozen_batches: list of batch dicts
      frozen_iter: callable that re-yields the same frozen batches
    """
    if callable(dataloader):
        frozen_batches = list(dataloader())
    else:
        frozen_batches = list(iter(dataloader))

    def frozen_iter():
        for batch in frozen_batches:
            yield batch

    return frozen_batches, frozen_iter


def get_compatible_linear_layers(
    model,
    candidate_layers,
    ranges_dict,
):
    """
    Keep only Linear layers whose ranges match module dimensions:
      activation_ranges -> in_features
      weight_ranges     -> out_features
    """
    name2mod = dict(model.named_modules())
    target_layers = []

    for name in candidate_layers:
        mod = name2mod.get(name, None)
        rec = ranges_dict.get(name, None)

        if mod is None or rec is None:
            continue
        if not isinstance(mod, torch.nn.Linear):
            continue

        a_ranges = rec.get("activation_ranges", None)
        w_ranges = rec.get("weight_ranges", None)

        if a_ranges is None or w_ranges is None:
            continue

        if a_ranges.numel() == mod.in_features and w_ranges.numel() == mod.out_features:
            target_layers.append(name)

    return target_layers