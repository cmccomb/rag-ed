"""Shared utilities for loader modules."""

from __future__ import annotations

import os
import tempfile
import zipfile
from typing import List


def extract_zip(path: str) -> List[str]:
    """Extract ``path`` and return absolute paths of contained files.

    Parameters
    ----------
    path:
        Filesystem path to a ``.zip`` archive.

    Returns
    -------
    list[str]
        Paths to all files contained in the archive. The archive is extracted to
        a temporary directory, which is not automatically cleaned up.
    """
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(temp_dir)
    file_paths: list[str] = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths
