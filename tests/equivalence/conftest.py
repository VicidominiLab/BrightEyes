from collections.abc import Generator

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures() -> Generator[None, None, None]:
    yield
    plt.close("all")
