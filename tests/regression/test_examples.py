"""Compare example computations with fixed numerical reference outputs."""

import hashlib
import json
import random
import shutil
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import torch
from _examples import execute
from threadpoolctl import threadpool_limits

from brighteyes import datasets

data_dir = Path(__file__).parents[1] / "data"
manifest = json.loads((data_dir / "manifest.json").read_text("utf-8"))


def check_capture(name: str, entry: dict, input_path: Path) -> None:
    """Check that provenance describes the frozen files and computation."""
    metadata = json.loads((data_dir / entry["capture"]).read_text("utf-8"))
    assert metadata["example"] == name
    assert metadata["source_revision"] == "9d7f078"
    for key, value in entry.items():
        assert metadata[key] == value, key
    assert metadata["comparison"] == {"exact": True}
    assert metadata["seeds"] == {"numpy": 0, "torch": 0, "python": 0}
    assert metadata["environment"]["device"] == "cpu"
    for key, path in (
        ("inputs", input_path),
        ("outputs", data_dir / entry["expected"]),
    ):
        recorded = metadata[key]
        assert recorded["filename"] == path.name
        assert recorded["size"] == path.stat().st_size
        assert recorded["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(data_dir / entry["expected"], allow_pickle=False) as expected:
        assert metadata["outputs"]["arrays"] == {
            key: {"shape": list(expected[key].shape), "dtype": str(expected[key].dtype)}
            for key in expected.files
        }


@pytest.fixture(autouse=True)
def cpu_environment() -> Generator[None, None, None]:
    """Restore random generators and CPU thread settings after each example."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(threads)


@pytest.mark.parametrize("name, entry", tuple(manifest.items()), ids=tuple(manifest))
def test_example_matches_saved_cpu_result(name: str, entry: dict) -> None:
    """Reproduce each saved numerical example result from its fixed input."""
    if entry["status"] == "unavailable":
        pytest.skip(entry["reason"])
    input_path = datasets.fetch(entry["dataset"])
    with np.load(input_path, allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    if entry["status"] == "known_failure":
        with pytest.raises(RuntimeError, match=entry["error"]["match"]):
            execute(name, inputs, entry["parameters"])
        return
    check_capture(name, entry, input_path)
    actual = execute(name, inputs, entry["parameters"])
    with np.load(data_dir / entry["expected"], allow_pickle=False) as expected:
        assert set(actual) == set(expected.files)
        for key, value in actual.items():
            np.testing.assert_array_equal(value, expected[key], strict=True)


@pytest.mark.parametrize("field", ("parameters", "sha256", "shape"))
def test_capture_rejects_changed_metadata(field, tmp_path, monkeypatch):
    """Reject provenance that no longer describes the recorded computation."""
    entry = manifest["ism-simulation"]
    metadata = json.loads((data_dir / entry["capture"]).read_text("utf-8"))
    match field:
        case "parameters":
            metadata["parameters"]["signal"] += 1
        case "sha256":
            metadata["outputs"]["sha256"] = "0" * 64
        case "shape":
            metadata["outputs"]["arrays"]["image"]["shape"][0] += 1
    (tmp_path / "imaging").mkdir()
    (tmp_path / entry["capture"]).write_text(json.dumps(metadata), encoding="utf-8")
    shutil.copyfile(data_dir / entry["expected"], tmp_path / entry["expected"])
    monkeypatch.setattr(f"{__name__}.data_dir", tmp_path)
    with pytest.raises(AssertionError):
        check_capture("ism-simulation", entry, datasets.fetch(entry["dataset"]))


def test_regression_rejects_changed_result(monkeypatch):
    """Fail the example regression when one computed pixel changes."""
    original = execute

    def changed_result(*args, **kwargs):
        result = original(*args, **kwargs)
        result["image"][0, 0, 0] += 1
        return result

    monkeypatch.setattr(f"{__name__}.execute", changed_result)
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        test_example_matches_saved_cpu_result(
            "ism-simulation", manifest["ism-simulation"]
        )
