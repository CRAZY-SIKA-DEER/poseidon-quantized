import os
import torch
from torch.utils.data import DataLoader

from scOT.model import ScOT
from scOT.problems.base import get_dataset


def load_poseidon_model(model_path: str, device: str = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = ScOT.from_pretrained(model_path).to(device)
    model.eval()

    torch.set_float32_matmul_precision("high")

    print(f"Model loaded on device: {device}")
    #print(f"Model type: {type(model)}")

    return model, device


def build_poseidon_loaders(
    dataset_name: str,
    data_path: str,
    calib_batchsize: int = 8,
    calib_steps: int = 8,
    val_batchsize: int = 16,
    val_steps: int = 50,
):
    train_ds = get_dataset(
        dataset_name,
        which="train",
        num_trajectories=2048,
        data_path=data_path,
    )

    try:
        val_ds = get_dataset(
            dataset_name,
            which="val",
            num_trajectories=256,
            data_path=data_path,
        )
    except Exception:
        val_ds = get_dataset(
            dataset_name,
            which="test",
            num_trajectories=256,
            data_path=data_path,
        )

    calib_loader = DataLoader(
        train_ds,
        batch_size=calib_batchsize,
        shuffle=True,
        num_workers=min(os.cpu_count() or 0, 16),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batchsize,
        shuffle=False,
        num_workers=min(os.cpu_count() or 0, 16),
        pin_memory=True,
    )

    def take(loader, steps):
        for i, batch in enumerate(loader):
            if i >= steps:
                break
            yield batch

    calib_iter = lambda: take(calib_loader, calib_steps)
    val_iter = lambda: take(val_loader, val_steps)

    return calib_loader, val_loader, calib_iter, val_iter


def poseidon_forward(model, batch, device):
    x = batch["pixel_values"].to(device)
    t = batch["time"].to(device)
    pm = batch["pixel_mask"].to(device)
    y = batch.get("labels")

    out = model(
        pixel_values=x,
        time=t,
        pixel_mask=pm,
        labels=(y.to(device) if y is not None else None),
    )
    return out.output