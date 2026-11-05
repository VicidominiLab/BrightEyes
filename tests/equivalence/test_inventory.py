import inspect
import pkgutil
import sys
from importlib import import_module

import pytest
from _registry import Case, new_path, resolve
from _relocations import removed
from cases import ALL_CASES, ALL_SKIPPED

OLD_PACKAGES = (
    "brighteyes_ism",
    "brighteyes_flism",
    "brighteyes_ffs",
    "brighteyes_mcs_reader",
    "brighteyes_mcs_dataprep",
    "libttp",
    "s2ism",
    "varreg",
    "psf_ism_distiller",
    "birfi",
)
OLD_MODULES = ("flus2ISM_lib", "libttp.ttpCython")


def _raise(name: str) -> None:
    raise sys.exception()


def public_names() -> set[str]:
    """Return the `module:name` path of every public function and class of the old packages."""
    names = set()
    modules = list(OLD_MODULES)
    for name in OLD_PACKAGES:
        package = import_module(name)
        modules.append(name)
        modules += [
            info.name
            for info in pkgutil.walk_packages(
                package.__path__, name + ".", onerror=_raise
            )
        ]
    for module_name in modules:
        for name, obj in vars(import_module(module_name)).items():
            if (
                not name.startswith("_")
                and (inspect.isroutine(obj) or inspect.isclass(obj))
                and obj.__module__ == module_name
            ):
                names.add(f"{module_name}:{name}")
    return names


def check_inventory(
    inventory: set[str], cases: list[Case], skipped: dict[str, str]
) -> None:
    """Require coverage or a declared removal for each original definition."""
    declared = {case.old for case in cases} | skipped.keys()
    covered = {
        f"{module}:{qualname.partition('.')[0]}"
        for module, qualname in (p.split(":") for p in declared)
    }
    assert not covered & removed.keys(), "removed names still registered"
    covered |= removed.keys()
    missing = sorted(inventory - covered)
    stale = sorted(covered - inventory)
    unresolved = []
    for path in sorted(skipped):
        try:
            resolve(new_path(path))
        except (ImportError, AttributeError) as error:
            unresolved.append(f"{path}: {error!r}")
    assert not missing and not stale and not unresolved, (
        f"{len(missing)} names have no case:\n"
        + "\n".join(missing)
        + f"\n{len(stale)} paths name nothing:\n"
        + "\n".join(stale)
        + f"\n{len(unresolved)} skipped paths missing from brighteyes:\n"
        + "\n".join(unresolved)
    )


def test_every_public_name_is_covered():
    """Give each original definition coverage or an explicit removal reason."""
    check_inventory(public_names(), ALL_CASES, ALL_SKIPPED)


def test_removed_name_cannot_remain_in_cases(monkeypatch):
    """Reject a removed name that still registers an equivalence case."""
    monkeypatch.setattr(
        "test_inventory.removed", {"sample.old:removed": "removed from imaging"}
    )
    case = Case("sample.old:removed", lambda target: target())
    with pytest.raises(AssertionError, match="removed names still registered"):
        check_inventory({"sample.old:removed"}, [case], {})


def test_unregistered_public_name_fails():
    """Report a retained original definition without an equivalence case."""
    with pytest.raises(AssertionError, match="names have no case"):
        check_inventory({"sample.old:uncovered"}, [], {})
