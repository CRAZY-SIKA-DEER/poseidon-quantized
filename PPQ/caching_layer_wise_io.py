from __future__ import annotations
'''
I am running this project on an HPC cluster.

My setup is:
- Code repo stays in: /home/u6ey/yiheng.u6ey/poseidon-quantized
- Large files must be saved in project storage: /lus/lfs1aip2/projects/u6ey/yiheng.u6ey/poseidon
- In the repo, I already use symlinks when possible, for example:
  - dataset -> /lus/lfs1aip2/projects/u6ey/yiheng.u6ey/poseidon/dataset
  - models  -> /lus/lfs1aip2/projects/u6ey/yiheng.u6ey/poseidon/models

Please modify my code so that any results generated during running (for example json, pt, csv, logs, checkpoints, saved statistics, plots, or output folders) are saved to the project directory, not under /home.

Requirements:
1. Keep my code paths clean and preferably relative from the repo root.
2. If suitable, use a symlink-based approach so code can still write to a normal repo path like results/... or ppq_artifacts/..., but the real files go to /lus/lfs1aip2/projects/u6ey/yiheng.u6ey/poseidon/...
3. Do not change dataset/model loading logic unless necessary.
4. Give me the exact code changes, not just explanation.
5. If a new output directory is needed, choose a clean structure under:
   /lus/lfs1aip2/projects/u6ey/yiheng.u6ey/poseidon

Please assume I want the safest HPC-style solution for saving runtime outputs.
'''


'''
below is the full caching code for the clean structure:

ppq_artifacts/
  frozen_calibration_batches.pt
  calibration_meta.json

  NS-PwC-T_layerio/
    meta.json
    run_meta.json
    layer_io/
      layer_00000.pt
      layer_00001.pt
      ...
'''



import gc
import json
from pathlib import Path

import torch
import torch.nn as nn

from PPQ.config import PPQConfig
from PPQ.poseidon_utils import load_poseidon_model, build_poseidon_loaders
from PPQ.optimize import freeze_batches


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_candidate_linear_layers(model, quant_layer_path: Path):
    """
    Load candidate quantization layers from file, then keep only real Linear layers.
    """
    print(f"[INFO] Loading quantize layer list from: {quant_layer_path}")
    layer_data = torch.load(quant_layer_path, map_location="cpu")

    name2mod = dict(model.named_modules())
    candidate_layers = [
        name for name in layer_data["quantize_layers"]
        if isinstance(name2mod.get(name, None), nn.Linear)
    ]

    print(f"[INFO] {len(candidate_layers)} candidate Linear layers")
    return candidate_layers


def get_model_cache_name(cfg: PPQConfig) -> str:
    """
    Decide the model-specific cache folder name.
    Change this if you want a different naming rule.
    """
    # For your current case:
    model_name = Path(cfg.model_path).name
    return f"{model_name}_layerio"
    


def save_shared_frozen_calibration(
    frozen_batches,
    artifacts_root: Path,
    calib_batchsize: int,
    calib_steps: int,
    dataset_name: str,
):
    """
    Save the shared frozen calibration dataset once at the top level:
        ppq_artifacts/frozen_calibration_batches.pt
        ppq_artifacts/calibration_meta.json
    """
    ensure_dir(artifacts_root)

    frozen_path = artifacts_root / "frozen_calibration_batches.pt"
    meta_path = artifacts_root / "calibration_meta.json"

    torch.save(frozen_batches, frozen_path)

    meta = {
        "dataset_name": str(dataset_name),
        "calib_batchsize": int(calib_batchsize),
        "calib_steps": int(calib_steps),
        "num_calibration_samples": int(calib_batchsize * calib_steps),
        "num_frozen_batches": int(len(frozen_batches)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved shared frozen calibration -> {frozen_path}")
    print(f"[INFO] Saved shared calibration meta -> {meta_path}")

    return frozen_path, meta_path


def cache_clean_outputs_poseidon_to_disk(
    model,
    frozen_batches,
    device,
    layer_names,
    model_cache_root: Path,
    save_dtype: torch.dtype = torch.float16,
):
    """
    Streamingly cache clean per-layer inputs/outputs to disk.

    Structure:
        model_cache_root/
            meta.json
            run_meta.json
            layer_io/
                layer_00000.pt
                layer_00001.pt
                ...

    Each layer file stores:
        {
            "layer_index": int,
            "layer_name": str,
            "num_batches": int,
            "save_dtype": str,
            "batches": [
                {"x_pre": tensor, "y_post": tensor},
                {"x_pre": tensor, "y_post": tensor},
                ...
            ]
        }

    Note:
        This still collects one layer's full batch list in RAM before saving that layer file.
        That is much safer than collecting all layers together, but if one single layer is still too big,
        later we can switch to shard files per layer.
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)

    name2mod = dict(model.named_modules())
    valid_layers = [
        name for name in layer_names
        if isinstance(name2mod.get(name, None), nn.Linear)
    ]

    layer_io_root = model_cache_root / "layer_io"
    ensure_dir(layer_io_root)

    print(f"[INFO] Start caching clean IO for {len(valid_layers)} layers")
    print(f"[INFO] Number of frozen batches: {len(frozen_batches)}")
    print(f"[INFO] Save dtype: {save_dtype}")

    cache_meta = {
        "num_layers": len(valid_layers),
        "num_batches": len(frozen_batches),
        "save_dtype": str(save_dtype),
        "layers": [
            {"layer_index": idx, "layer_name": name}
            for idx, name in enumerate(valid_layers)
        ],
    }
    with open(model_cache_root / "meta.json", "w") as f:
        json.dump(cache_meta, f, indent=2)

    with torch.inference_mode():
        for layer_idx, layer_name in enumerate(valid_layers):
            mod = name2mod[layer_name]
            print(f"\n[INFO] Caching layer {layer_idx + 1}/{len(valid_layers)}: {layer_name}")

            layer_batches = []

            for batch_idx, batch in enumerate(frozen_batches):
                layer_io = {}

                def hook_fn(module, inp, out):
                    x_pre = inp[0].detach().to("cpu", dtype=save_dtype)
                    y_post = out.detach().to("cpu", dtype=save_dtype)
                    layer_io["x_pre"] = x_pre
                    layer_io["y_post"] = y_post

                handle = mod.register_forward_hook(hook_fn)

                x = batch["pixel_values"].to(device)
                t = batch.get("time", None)
                pm = batch.get("pixel_mask", None)
                y = batch.get("labels", None)

                if t is not None:
                    t = t.to(device)
                if pm is not None:
                    pm = pm.to(device)
                if y is not None:
                    y = y.to(device)

                _ = model(
                    pixel_values=x,
                    time=t,
                    pixel_mask=pm,
                    labels=y,
                )

                handle.remove()

                if "x_pre" not in layer_io or "y_post" not in layer_io:
                    raise RuntimeError(
                        f"[ERROR] Hook did not capture IO for layer={layer_name}, batch_idx={batch_idx}"
                    )

                layer_batches.append(
                    {
                        "x_pre": layer_io["x_pre"],
                        "y_post": layer_io["y_post"],
                    }
                )

                del layer_io
                del x
                del t
                del pm
                del y
                _ = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(frozen_batches):
                    print(
                        f"[INFO]   layer={layer_name} | "
                        f"processed batch {batch_idx + 1}/{len(frozen_batches)}"
                    )

            layer_obj = {
                "layer_index": layer_idx,
                "layer_name": layer_name,
                "num_batches": len(layer_batches),
                "save_dtype": str(save_dtype),
                "batches": layer_batches,
            }

            layer_path = layer_io_root / f"layer_{layer_idx:05d}.pt"
            torch.save(layer_obj, layer_path)
            print(f"[INFO] Saved layer cache -> {layer_path}")

            del layer_batches
            del layer_obj

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print("\n[INFO] Finished caching all layer IO.")


def main():
    cfg = PPQConfig()

    # --------------------------------------------------
    # Recommended setup for 1024 calibration samples
    # safer memory choice:
    #   batch size small
    #   steps large
    # --------------------------------------------------
    calib_batchsize = 2
    calib_steps = 512

    artifacts_root = cfg.artifacts_dir
    ensure_dir(artifacts_root)

    model_cache_name = get_model_cache_name(cfg)
    model_cache_root = artifacts_root / model_cache_name
    ensure_dir(model_cache_root)

    # --------------------------------------------------
    # 1) Load model
    # --------------------------------------------------
    model, device = load_poseidon_model(cfg.model_path, cfg.device)

    # # --------------------------------------------------
    # # 2) Build calibration loader
    # # --------------------------------------------------
    # _calib_loader, _val_loader, calib_iter, _val_iter = build_poseidon_loaders(
    #     dataset_name=cfg.dataset_name,
    #     data_path=cfg.data_path,
    #     calib_batchsize=calib_batchsize,
    #     calib_steps=calib_steps,
    #     val_batchsize=cfg.val_batchsize,
    #     val_steps=cfg.val_steps,
    # )

    # # --------------------------------------------------
    # # 3) Freeze exact calibration dataset
    # # --------------------------------------------------
    # frozen_batches, _ = freeze_batches(calib_iter)


    frozen_batches = torch.load(cfg.artifacts_dir / "frozen_calibration_batches.pt")
    # print(
    #     f"[INFO] Frozen calibration samples = "
    #     f"{len(frozen_batches)} x {calib_batchsize} = {len(frozen_batches) * calib_batchsize}"
    # )
    print(f"[INFO] Loaded frozen calibration batches: {len(frozen_batches)}")

    # frozen_path, calib_meta_path = save_shared_frozen_calibration(
    #     frozen_batches=frozen_batches,
    #     artifacts_root=artifacts_root,
    #     calib_batchsize=calib_batchsize,
    #     calib_steps=calib_steps,
    #     dataset_name=cfg.dataset_name,
    # )

    # --------------------------------------------------
    # 4) Load candidate Linear layers
    # --------------------------------------------------
    candidate_layers = load_candidate_linear_layers(
        model=model,
        quant_layer_path=cfg.quant_layer_path,
    )

    # --------------------------------------------------
    # 5) Save model/run-specific metadata
    # --------------------------------------------------

    # run_meta = {
    #     "model_path": str(cfg.model_path),
    #     "dataset_name": str(cfg.dataset_name),
    #     "data_path": str(cfg.data_path),
    #     "quant_layer_path": str(cfg.quant_layer_path),
    #     "device": str(device),
    #     "calib_batchsize": calib_batchsize,
    #     "calib_steps": calib_steps,
    #     "num_calibration_samples": calib_batchsize * calib_steps,
    #     "num_candidate_layers": len(candidate_layers),
    #     "shared_frozen_batches_path": str(frozen_path),
    #     "shared_calibration_meta_path": str(calib_meta_path),
    #     "model_cache_name": model_cache_name,
    # }

    shared_frozen_path = cfg.artifacts_dir / "frozen_calibration_batches.pt"
    shared_calib_meta_path = cfg.artifacts_dir / "calibration_meta.json"

    run_meta = {
        "model_path": str(cfg.model_path),
        "dataset_name": str(cfg.dataset_name),
        "data_path": str(cfg.data_path),
        "quant_layer_path": str(cfg.quant_layer_path),
        "device": str(device),
        "calib_batchsize": calib_batchsize,
        "calib_steps": calib_steps,
        "num_calibration_samples": calib_batchsize * calib_steps,
        "num_candidate_layers": len(candidate_layers),
        "shared_frozen_batches_path": str(shared_frozen_path),
        "shared_calibration_meta_path": str(shared_calib_meta_path),
        "model_cache_name": model_cache_name,
    }


    with open(model_cache_root / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # --------------------------------------------------
    # 6) Cache clean layer IO
    # --------------------------------------------------
    cache_clean_outputs_poseidon_to_disk(
        model=model,
        frozen_batches=frozen_batches,
        device=device,
        layer_names=candidate_layers,
        model_cache_root=model_cache_root,
        save_dtype=torch.float16,
    )


if __name__ == "__main__":
    main()