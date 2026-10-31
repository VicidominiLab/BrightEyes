import importlib
import pkgutil
import sys

import pytest

import brighteyes


def _raise(name: str) -> None:
    raise sys.exception()


MODULES = [
    module.name
    for module in pkgutil.walk_packages(
        brighteyes.__path__, prefix="brighteyes.", onerror=_raise
    )
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    """Import every module of the brighteyes package."""
    importlib.import_module(name)
