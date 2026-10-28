import importlib
import inspect
import pkgutil

from .cases import ALL_CASES, ALL_SKIPPED

OLD_PACKAGES = ("brighteyes_ism", "brighteyes_flism", "brighteyes_ffs")


def _raise(name: str) -> None:
    raise ImportError(name)


def _public_names() -> set[str]:
    names = set()
    for package_name in OLD_PACKAGES:
        package = importlib.import_module(package_name)
        module_names = [package_name]
        module_names += [
            info.name for info in pkgutil.walk_packages(package.__path__, prefix=package_name + ".", onerror=_raise)
        ]
        for module_name in module_names:
            module = importlib.import_module(module_name)
            for name, value in vars(module).items():
                is_defined_here = (inspect.isfunction(value) or inspect.isclass(value)) and value.__module__ == module_name
                if is_defined_here and not name.startswith("_"):
                    names.add(f"{module_name}:{name}")
    return names


def test_every_public_name_is_covered() -> None:
    """Give every public name of the old packages a case or a skip reason."""
    found = _public_names()
    declared = {case.old for case in ALL_CASES} | set(ALL_SKIPPED)
    missing = sorted(found - declared)
    stale = sorted(declared - found)
    assert not (missing or stale), f"missing {len(missing)}: {missing}\nstale {len(stale)}: {stale}"
