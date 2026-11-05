"""Check preservation of numerical reference data."""

import os
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest

from brighteyes import datasets


@pytest.fixture
def capture():
    """Load the capture command without executing its entry point."""
    return runpy.run_path(str(Path("scripts/capture_example_results.py")))


def test_crop_preserves_nonspatial_axes(capture):
    """Crop spatial axes without reducing time bins or detector channels."""
    source = np.arange(2 * 3 * 40 * 44 * 7 * 9).reshape(2, 3, 40, 44, 7, 9)
    cropped, selection = capture["crop_spatial"](source, (2, 3), size=32)
    np.testing.assert_array_equal(cropped, source[:, :, 4:36, 6:38, :, :])
    assert selection == [[0, 2], [0, 3], [4, 36], [6, 38], [0, 7], [0, 9]]


def test_capture_refuses_existing_results(capture, tmp_path):
    """Refuse to replace a previous reference result."""
    output = tmp_path / "baseline"
    output.mkdir()
    original = output / "outputs.npz"
    np.savez(original, image=np.ones((2, 3)))
    content = original.read_bytes()
    with pytest.raises(FileExistsError):
        capture["write_capture"](
            output,
            {"input": np.zeros((2, 3))},
            {"image": np.zeros((2, 3))},
            {"source_revision": "9d7f078"},
        )
    assert original.read_bytes() == content


def test_capture_preserves_output_values(capture, tmp_path):
    """Persist outputs separately from the inputs used to compute them."""
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    capture["write_capture"](
        tmp_path / "baseline",
        {"input": expected},
        {"image": expected + 1},
        {"source_revision": "9d7f078"},
    )
    with np.load(tmp_path / "baseline" / "outputs.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["image"], expected + 1, strict=True)


def test_capture_uses_reference_source_with_relative_cache(tmp_path):
    """Run the fixed source from a cache path relative to the checkout."""
    root = Path(__file__).parents[2]
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "scripts/capture_example_results.py",
            "--example",
            "ism-simulation",
            "--cache",
            os.path.relpath(tmp_path / "cache", root),
            "--output",
            str(tmp_path / "capture"),
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    with (
        np.load(tmp_path / "capture/outputs.npz", allow_pickle=False) as output,
        np.load(datasets.fetch("ism-simulation-small"), allow_pickle=False) as sample,
    ):
        np.testing.assert_array_equal(output["image"], sample["image"], strict=True)
