"""Compare example computations with fixed numerical reference outputs."""

import json
import random
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
    with np.load(datasets.fetch(entry["dataset"]), allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    if entry["status"] == "known_failure":
        with pytest.raises(RuntimeError, match=entry["error"]["match"]):
            execute(name, inputs, entry["parameters"])
        return
    actual = execute(name, inputs, entry["parameters"])
    with np.load(data_dir / entry["expected"], allow_pickle=False) as expected:
        assert set(actual) == set(expected.files)
        for key, value in actual.items():
            np.testing.assert_array_equal(value, expected[key], strict=True)
