from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """
    Compute the project root directory for `support-copilot-ai/`.

    Returns:
        Path pointing to the `support-copilot-ai` directory.
    """
    # support-copilot-ai/app/core/config.py -> parents[0]=core, [1]=app, [2]=support-copilot-ai
    return Path(__file__).resolve().parents[2]


def data_dirs() -> tuple[Path, Path]:
    """
    Resolve raw and processed data directories.

    Env overrides:
        - DATA_DIR: base directory (defaults to "<project_root>/data")

    Returns:
        (raw_dir, processed_dir)
    """
    base = Path(os.getenv("DATA_DIR", str(project_root() / "data"))).resolve()
    raw_dir = base / "raw"
    processed_dir = base / "processed"
    return raw_dir, processed_dir

