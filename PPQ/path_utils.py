from __future__ import annotations

import os
from pathlib import Path


PROJECT_STORAGE_ROOT = Path("/lus/lfs1aip2/projects/u6ey/yiheng.u6ey/poseidon")


def get_repo_root() -> Path:
    # PPQ/path_utils.py -> repo root is one level above PPQ/
    return Path(__file__).resolve().parents[1]


def ensure_storage_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_symlink_dir(link_path: Path, target_path: Path) -> Path:
    """
    Make link_path a symlink to target_path.

    Safe HPC-style behavior:
    - real data lives under /lus/...
    - repo path remains clean and relative
    """
    target_path = ensure_storage_dir(target_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.exists() or link_path.is_symlink():
        # already correct symlink
        if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
            return link_path

        # existing normal directory/file -> do not silently destroy
        raise RuntimeError(
            f"[ERROR] {link_path} already exists and is not the expected symlink to {target_path}. "
            f"Please move/remove it manually first."
        )

    os.symlink(target_path, link_path, target_is_directory=True)
    return link_path