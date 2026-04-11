# SAPQ/check_wrapped_layer_names.py
from __future__ import annotations

import json
from pathlib import Path

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model
from BRECQ.quant.poseidon_quant_model import PoseidonQuantModel
from BRECQ.quant.quant_layer import QuantModule


def load_layer_names(layer_path: Path):
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer file not found: {layer_path}")

    if layer_path.suffix == ".json":
        with open(layer_path, "r") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "quantize_layers" in obj:
                return obj["quantize_layers"]
            raise ValueError(f"JSON file does not contain 'quantize_layers': {layer_path}")
        raise TypeError(f"Unsupported JSON content type: {type(obj)}")

    if layer_path.suffix == ".pt":
        import torch
        obj = torch.load(layer_path, map_location="cpu")
        if isinstance(obj, dict) and "quantize_layers" in obj:
            return obj["quantize_layers"]
        raise ValueError(f"PT file does not contain 'quantize_layers': {layer_path}")

    raise ValueError(f"Unsupported layer file type: {layer_path}")


def main():
    cfg = PPQConfig()

    model_path = Path(cfg.model_path)
    device = cfg.device

    # choose JSON first if it exists, otherwise use cfg.quant_layer_path
    json_layer_path = cfg.repo_root / "inspect_layers" / "L_quantize_layers.json"
    layer_path = json_layer_path if json_layer_path.exists() else Path(cfg.quant_layer_path)

    print(f"Loading Poseidon model from: {model_path}")
    model, device = load_poseidon_model(str(model_path), device)

    print("Wrapping model with PoseidonQuantModel...")
    qmodel = PoseidonQuantModel(model=model)
    qmodel = qmodel.to(device).eval()

    print(f"Loading layer names from: {layer_path}")
    layer_names = load_layer_names(layer_path)

    wrapped_named_modules = dict(qmodel.named_modules())

    found_names = []
    quantmodule_names = []
    missing_names = []
    non_quantmodule_names = []

    for name in layer_names:
        wrapped_name = name if name.startswith("model.") else f"model.{name}"
        mod = wrapped_named_modules.get(wrapped_name, None)
        if mod is None:
            missing_names.append((name, wrapped_name))
        else:
            found_names.append((name, wrapped_name))
            if isinstance(mod, QuantModule):
                quantmodule_names.append((name, wrapped_name))
            else:
                non_quantmodule_names.append((name, wrapped_name, type(mod).__name__))

    all_quantmodules = [
        name for name, mod in wrapped_named_modules.items()
        if isinstance(mod, QuantModule)
    ]

    print("\n========== CHECK RESULT ==========")
    print(f"Layer names total             : {len(layer_names)}")
    print(f"Found in wrapped model        : {len(found_names)}")
    print(f"Found and are QuantModule     : {len(quantmodule_names)}")
    print(f"Found but NOT QuantModule     : {len(non_quantmodule_names)}")
    print(f"Missing in wrapped model      : {len(missing_names)}")
    print(f"Total QuantModule in qmodel   : {len(all_quantmodules)}")

    print("\n----- Sample matched QuantModule names -----")
    for orig_name, wrapped_name in quantmodule_names[:20]:
        print(f"{orig_name}  -->  {wrapped_name}")

    print("\n----- Sample found but not QuantModule -----")
    for orig_name, wrapped_name, mod_type in non_quantmodule_names[:20]:
        print(f"{orig_name}  -->  {wrapped_name}  -->  {mod_type}")

    print("\n----- Sample missing names -----")
    for orig_name, wrapped_name in missing_names[:20]:
        print(f"{orig_name}  -->  expected wrapped name: {wrapped_name}")

    print("\n----- Sample all QuantModule names in qmodel -----")
    for name in all_quantmodules[:30]:
        print(name)

    print("\nDone.")


if __name__ == "__main__":
    main()