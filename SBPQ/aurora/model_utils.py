"""Model-loading helpers for Aurora."""

from __future__ import annotations

import torch
from aurora import AuroraPretrained, AuroraSmallPretrained


def load_aurora_model(
    model_name: str = "small",
    checkpoint_name: str | None = None,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    if model_name == "small":
        model = AuroraSmallPretrained()
        checkpoint_name = checkpoint_name or "aurora-0.25-small-pretrained.ckpt"
    elif model_name in {"pretrained", "full"}:
        model = AuroraPretrained()
        checkpoint_name = checkpoint_name or "aurora-0.25-pretrained.ckpt"
    else:
        raise ValueError(f"Unsupported Aurora model_name: {model_name}")

    model.load_checkpoint("microsoft/aurora", checkpoint_name)
    model.eval()
    return model.to(torch.device(device))


def freeze_model(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

