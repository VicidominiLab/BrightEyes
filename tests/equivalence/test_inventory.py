import inspect
import pkgutil
from importlib import import_module

import brighteyes_ffs
import brighteyes_flism
import brighteyes_ism
from cases import ALL_CASES, ALL_SKIPPED


def _raise(name: str) -> None:
    raise


def public_names() -> set[str]:
    """Return the `module:name` path of every public function and class of the old packages."""
    names = set()
    for package in (brighteyes_ism, brighteyes_flism, brighteyes_ffs):
        modules = [package.__name__]
        modules += [
            info.name for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".", onerror=_raise)
        ]
        for module_name in modules:
            for name, obj in vars(import_module(module_name)).items():
                if (
                    not name.startswith("_")
                    and (inspect.isfunction(obj) or inspect.isclass(obj))
                    and obj.__module__ == module_name
                ):
                    names.add(f"{module_name}:{name}")
    return names


def test_every_public_name_is_covered():
    """Give every public name of the old packages a case or a skip reason."""
    declared = {case.old for case in ALL_CASES} | ALL_SKIPPED.keys()
    covered = {f"{module}:{qualname.partition('.')[0]}" for module, qualname in (p.split(":") for p in declared)}
    inventory = public_names()
    missing = sorted(inventory - covered)
    stale = sorted(covered - inventory)
    assert not missing and not stale, (
        f"{len(missing)} names have no case:\n"
        + "\n".join(missing)
        + f"\n{len(stale)} paths name nothing:\n"
        + "\n".join(stale)
    )
