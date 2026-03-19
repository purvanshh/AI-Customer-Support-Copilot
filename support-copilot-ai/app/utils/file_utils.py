from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

from fastapi import UploadFile


def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists.

    Args:
        path: Directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str) -> str:
    """
    Make a filename safe for filesystem usage by stripping path components.

    Args:
        filename: Original filename.

    Returns:
        A safe filename.
    """
    return os.path.basename(filename) if filename else "uploaded_file"


async def save_upload_file(upload_file: UploadFile, dest_path: Path) -> None:
    """
    Save an uploaded file to disk.

    Args:
        upload_file: FastAPI UploadFile.
        dest_path: Destination path including filename.
    """
    ensure_dir(dest_path.parent)

    upload_file.file.seek(0)
    with dest_path.open("wb") as f:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def open_binary(path: Path) -> BinaryIO:
    """
    Open a file in binary mode.

    Args:
        path: File path.

    Returns:
        A file object opened in 'rb' mode.
    """
    return path.open("rb")

