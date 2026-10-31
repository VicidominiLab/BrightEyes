from types import ModuleType

from _registry import Case

_MODULES: tuple[ModuleType, ...] = ()

ALL_CASES: list[Case] = [case for module in _MODULES for case in module.CASES]
ALL_SKIPPED: dict[str, str] = {}

for _module in _MODULES:
    for _path, _reason in _module.SKIPPED.items():
        if _path in ALL_SKIPPED:
            raise ValueError(f"{_path} is skipped twice")
        ALL_SKIPPED[_path] = _reason
