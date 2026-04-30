from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def get_model_layerio_root(repo_root: str | Path, model_path: str | Path) -> Path:
    """
    Return the model-specific cached layer-IO root.

    Example:
        repo_root = /home/.../poseidon-quantized
        model_path = models/NS-PwC-L

    returns:
        /home/.../poseidon-quantized/ppq_artifacts/NS-PwC-L_layerio
    """
    repo_root = Path(repo_root)
    model_name = Path(model_path).name
    return repo_root / "ppq_artifacts" / f"{model_name}_layerio"


def load_layerio_meta(layerio_root: str | Path) -> dict:
    """
    Load meta.json from cached layer-IO root.
    """
    layerio_root = Path(layerio_root)
    meta_path = layerio_root / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Layer-IO meta file not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    required_keys = {"num_layers", "num_batches", "save_dtype", "layers"}
    missing = required_keys - set(meta.keys())
    if missing:
        raise ValueError(f"Layer-IO meta missing keys: {missing}")

    return meta


def build_layer_name_to_file_map(layerio_root: str | Path) -> Dict[str, Path]:
    """
    Build a mapping:
        layer_name -> layer_XXXXX.pt

    from meta.json.
    """
    layerio_root = Path(layerio_root)
    meta = load_layerio_meta(layerio_root)
    layer_dir = layerio_root / "layer_io"

    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer-IO directory not found: {layer_dir}")

    name_to_file: Dict[str, Path] = {}

    for entry in meta["layers"]:
        if "layer_index" not in entry or "layer_name" not in entry:
            raise ValueError(f"Invalid meta layer entry: {entry}")

        layer_index = int(entry["layer_index"])
        layer_name = str(entry["layer_name"])
        layer_file = layer_dir / f"layer_{layer_index:05d}.pt"

        if not layer_file.exists():
            raise FileNotFoundError(
                f"Layer cache file for '{layer_name}' not found: {layer_file}"
            )

        name_to_file[layer_name] = layer_file

    return name_to_file


def _validate_layer_cache_obj(obj: dict, layer_file: Path):
    required_keys = {"layer_index", "layer_name", "num_batches", "save_dtype", "batches"}
    missing = required_keys - set(obj.keys())
    if missing:
        raise ValueError(f"{layer_file} missing keys: {missing}")

    if not isinstance(obj["batches"], list):
        raise ValueError(f"{layer_file}: 'batches' must be a list")

    if int(obj["num_batches"]) != len(obj["batches"]):
        raise ValueError(
            f"{layer_file}: num_batches={obj['num_batches']} "
            f"but len(batches)={len(obj['batches'])}"
        )


def load_single_layer_io(
    layer_file: str | Path,
    to_float32: bool = True,
) -> Tuple[str, List[torch.Tensor], List[torch.Tensor]]:
    """
    Load one cached layer file and return:

        layer_name, clean_inputs_list, clean_outputs_list

    where:
        clean_inputs_list[batch_idx]  = x_pre
        clean_outputs_list[batch_idx] = y_post

    Notes:
    - cached tensors were typically saved as float16
    - by default we convert to float32 for safer math during training
    """
    layer_file = Path(layer_file)

    if not layer_file.exists():
        raise FileNotFoundError(f"Layer cache file not found: {layer_file}")

    obj = torch.load(layer_file, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"{layer_file}: expected dict, got {type(obj)}")

    _validate_layer_cache_obj(obj, layer_file)

    layer_name = str(obj["layer_name"])
    clean_inputs: List[torch.Tensor] = []
    clean_outputs: List[torch.Tensor] = []

    for batch_idx, batch_obj in enumerate(obj["batches"]):
        if not isinstance(batch_obj, dict):
            raise ValueError(
                f"{layer_file}: batch {batch_idx} expected dict, got {type(batch_obj)}"
            )

        if "x_pre" not in batch_obj or "y_post" not in batch_obj:
            raise ValueError(
                f"{layer_file}: batch {batch_idx} missing x_pre or y_post"
            )

        x_pre = batch_obj["x_pre"]
        y_post = batch_obj["y_post"]

        if not torch.is_tensor(x_pre) or not torch.is_tensor(y_post):
            raise ValueError(
                f"{layer_file}: batch {batch_idx} x_pre/y_post must be tensors"
            )

        if to_float32:
            x_pre = x_pre.float()
            y_post = y_post.float()

        clean_inputs.append(x_pre)
        clean_outputs.append(y_post)

    return layer_name, clean_inputs, clean_outputs


def load_cached_layer_io_for_layers(
    layerio_root: str | Path,
    target_layers: List[str],
    to_float32: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    """
    Load cached clean layer IO for the requested target layers.

    Returns:
        clean_inputs:  dict[layer_name] -> list[tensor]
        clean_outputs: dict[layer_name] -> list[tensor]

    Compatible with the structure expected by the old PPQ layerwise logic:
        clean_inputs[layer_name][batch_idx]
        clean_outputs[layer_name][batch_idx]
    """
    layerio_root = Path(layerio_root)
    name_to_file = build_layer_name_to_file_map(layerio_root)

    clean_inputs: Dict[str, List[torch.Tensor]] = {}
    clean_outputs: Dict[str, List[torch.Tensor]] = {}

    missing_layers = [name for name in target_layers if name not in name_to_file]
    if missing_layers:
        raise ValueError(
            "The following target layers are missing from cached layer IO:\n"
            + "\n".join(missing_layers[:20])
            + ("\n..." if len(missing_layers) > 20 else "")
        )

    for idx, layer_name in enumerate(target_layers):
        layer_file = name_to_file[layer_name]
        loaded_name, x_list, y_list = load_single_layer_io(
            layer_file=layer_file,
            to_float32=to_float32,
        )

        if loaded_name != layer_name:
            raise ValueError(
                f"Layer name mismatch: requested '{layer_name}', "
                f"but file contains '{loaded_name}'"
            )

        clean_inputs[layer_name] = x_list
        clean_outputs[layer_name] = y_list

        if verbose and ((idx + 1) % 20 == 0 or (idx + 1) == len(target_layers)):
            print(
                f"[INFO] Loaded cached layer IO for {idx + 1}/{len(target_layers)} layers"
            )

    return clean_inputs, clean_outputs


def inspect_cached_layer_io_summary(
    layerio_root: str | Path,
    target_layers: List[str] | None = None,
):
    """
    Small helper for debugging / sanity checking.
    Prints a short summary only.
    """
    layerio_root = Path(layerio_root)
    meta = load_layerio_meta(layerio_root)
    name_to_file = build_layer_name_to_file_map(layerio_root)

    print(f"[INFO] layerio_root = {layerio_root}")
    print(f"[INFO] num_layers(meta) = {meta['num_layers']}")
    print(f"[INFO] num_batches(meta) = {meta['num_batches']}")
    print(f"[INFO] save_dtype(meta) = {meta['save_dtype']}")

    if target_layers is None:
        target_layers = list(name_to_file.keys())[:3]

    print(f"[INFO] inspect target count = {len(target_layers)}")

    for layer_name in target_layers[:3]:
        layer_file = name_to_file[layer_name]
        loaded_name, x_list, y_list = load_single_layer_io(layer_file, to_float32=False)
        print(f"\n[layer] {loaded_name}")
        print(f"  file: {layer_file.name}")
        print(f"  num_batches: {len(x_list)}")
        if len(x_list) > 0:
            print(f"  x_pre[0]: shape={tuple(x_list[0].shape)}, dtype={x_list[0].dtype}")
            print(f"  y_post[0]: shape={tuple(y_list[0].shape)}, dtype={y_list[0].dtype}")