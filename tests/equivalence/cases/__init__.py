import importlib
import pkgutil

from .._registry import Case

ALL_CASES: list[Case] = []
ALL_SKIPPED: dict[str, str] = {}


def _collect() -> None:
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{info.name}")
        ALL_CASES.extend(module.CASES)
        for path, reason in module.SKIPPED.items():
            if path in ALL_SKIPPED:
                raise ValueError(f"{path} is skipped twice")
            ALL_SKIPPED[path] = reason


_collect()
