"""Capture CPU example results from a fixed source revision."""

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from brighteyes import datasets

root = Path(__file__).resolve().parent.parent
source_revision = "9d7f078"


def crop_spatial(
    source: np.ndarray, axes: tuple[int, int], size: int = 32
) -> tuple[np.ndarray, list[list[int]]]:
    """Select a central spatial region while preserving other dimensions."""
    selection = [[0, length] for length in source.shape]
    for axis in axes:
        length = min(source.shape[axis], size)
        start = (source.shape[axis] - length) // 2
        selection[axis] = [start, start + length]
    return source[
        tuple(slice(start, end) for start, end in selection)
    ].copy(), selection


def file_info(path: Path) -> dict:
    """Describe a capture file without recording its local directory."""
    return {
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_capture(
    directory: Path,
    inputs: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    """Write numerical inputs, outputs, and provenance into a new directory.

    Raises
    ------
    FileExistsError
        If the destination already exists.
    ValueError
        If an input or output contains Python objects.
    """
    for arrays in (inputs, outputs):
        if any(np.asarray(array).dtype.hasobject for array in arrays.values()):
            raise ValueError("capture arrays must not contain Python objects")
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "inputs.npz", **inputs)
    np.savez_compressed(directory / "outputs.npz", **outputs)
    metadata = {
        **metadata,
        "inputs": file_info(directory / "inputs.npz"),
        "outputs": file_info(directory / "outputs.npz"),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def capture(example: str, cache: Path, output: Path) -> None:
    """Run a registered example against the fixed source revision."""
    cache = cache.resolve()
    output = output.resolve()
    manifest = json.loads((root / "tests/data/manifest.json").read_text("utf-8"))
    if example not in manifest:
        raise ValueError(f"unknown example: {example}")
    entry = manifest[example]
    if entry["status"] == "unavailable":
        raise ValueError(entry["reason"])
    if output.exists():
        raise FileExistsError(output)
    cache.mkdir(parents=True, exist_ok=True)
    input_path = datasets.fetch(entry["dataset"])
    with np.load(input_path, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}

    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        work = Path(temporary)
        archived = subprocess.run(
            ["git", "archive", "--format=zip", source_revision, "src"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        with ZipFile(io.BytesIO(archived)) as archive:
            archive.extractall(work)
        result_path = work / "outputs.npz"
        environment = {
            **os.environ,
            "PYTHONPATH": str(work / "src"),
            "BRIGHTEYES_REFERENCE_ROOT": str(work / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "QT_QPA_PLATFORM": "offscreen",
            "MPLBACKEND": "Agg",
        }
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                str(root / "tests/regression/_examples.py"),
                example,
                str(input_path),
                str(result_path),
                json.dumps(entry["parameters"]),
            ],
            cwd=work,
            env=environment,
            check=True,
        )
        with np.load(result_path, allow_pickle=False) as archive:
            outputs = {key: archive[key] for key in archive.files}
    write_capture(
        output,
        inputs,
        outputs,
        {
            **entry,
            "status": "captured",
            "example": example,
            "source_revision": source_revision,
            "seeds": {"numpy": 0, "torch": 0, "python": 0},
            "environment": {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
                "device": "cpu",
                "threads": 1,
                "packages": {
                    name: version(name) for name in ("numpy", "scipy", "torch", "h5py")
                },
            },
            "comparison": {"exact": True},
        },
    )


def main() -> None:
    """Capture one named example into a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    capture(args.example, args.cache, args.output)


if __name__ == "__main__":
    main()
