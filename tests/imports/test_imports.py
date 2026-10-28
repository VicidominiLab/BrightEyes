import importlib
import pkgutil

import pytest

import brighteyes


def _raise(name: str) -> None:
    raise ImportError(name)


MODULE_NAMES = sorted(
    info.name
    for info in pkgutil.walk_packages(
        brighteyes.__path__, prefix="brighteyes.", onerror=_raise
    )
)


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_module_imports(name: str) -> None:
    """Import every module of the brighteyes package."""
    importlib.import_module(name)
