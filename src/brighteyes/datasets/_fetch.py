"""Resolve packaged inputs and cache verified remote files."""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import requests

data_dir = Path(__file__).with_name("data")
catalog = json.loads(Path(__file__).with_name("_catalog.json").read_text("utf-8"))
cache_dir = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    if os.name == "nt"
    else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
) / "brighteyes"
logger = logging.getLogger(__name__)


def verify(path: Path, size: int, checksum: str) -> None:
    """Raise ValueError when a file's size or checksum differs."""
    if path.stat().st_size != size:
        raise ValueError(f"dataset size mismatch: {path.name}")
    algorithm, expected = checksum.split(":", 1)
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"dataset checksum mismatch: {path.name}")


def fetch(name: str) -> Path:
    """Return a verified example input, downloading remote entries if needed.

    Small inputs are packaged locally. Full measurements download into the
    user cache and are reused when their size and checksum match.

    Raises
    ------
    ValueError
        If the name is unknown or the file fails integrity checks.
    requests.RequestException
        If a remote request fails.
    OSError
        If the input or cache cannot be read or written.
    """
    if name not in catalog:
        raise ValueError(f"unknown dataset: {name}")
    entry = catalog[name]
    if "path" in entry:
        path = data_dir / entry["path"]
        verify(path, entry["size"], "sha256:" + entry["sha256"])
        return path

    checksum = entry["checksum"]
    path = cache_dir / checksum.replace(":", "-") / entry["filename"]
    if path.exists():
        try:
            verify(path, entry["size"], checksum)
        except ValueError:
            pass
        else:
            return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.warning("Downloading %s (%s bytes)", name, entry["size"])
    temporary = None
    try:
        with requests.get(
            entry["url"],
            stream=True,
            timeout=(10, 60),
            headers={"User-Agent": "brighteyes-dev"},
        ) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                algorithm, _ = checksum.split(":", 1)
                digest = hashlib.new(algorithm, usedforsecurity=False)
                size = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    size += len(chunk)
                    if size > entry["size"]:
                        raise ValueError(f"dataset size mismatch: {name}")
                    stream.write(chunk)
                    digest.update(chunk)
            if size != entry["size"]:
                raise ValueError(f"dataset size mismatch: {name}")
            if digest.hexdigest() != checksum.split(":", 1)[1]:
                raise ValueError(f"dataset checksum mismatch: {name}")
            temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
