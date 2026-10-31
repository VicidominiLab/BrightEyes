from types import ModuleType

from _registry import Case

from . import (
    imaging_analysis,
    imaging_flism,
    imaging_s2ism,
    imaging_simulation,
    mcs_alignment,
    mcs_calibrator,
    mcs_dataprep,
    mcs_reader,
    spectroscopy_fcs,
    spectroscopy_other,
    ttm,
    ttm_kernels,
)

_MODULES: tuple[ModuleType, ...] = (
    imaging_analysis,
    imaging_flism,
    imaging_s2ism,
    imaging_simulation,
    mcs_alignment,
    mcs_calibrator,
    mcs_dataprep,
    mcs_reader,
    spectroscopy_fcs,
    spectroscopy_other,
    ttm,
    ttm_kernels,
)

ALL_CASES: list[Case] = [case for module in _MODULES for case in module.CASES]
ALL_SKIPPED: dict[str, str] = {}

for _module in _MODULES:
    for _path, _reason in _module.SKIPPED.items():
        if _path in ALL_SKIPPED:
            raise ValueError(f"{_path} is skipped twice")
        ALL_SKIPPED[_path] = _reason

for _case in ALL_CASES:
    if _case.old in ALL_SKIPPED:
        raise ValueError(f"{_case.old} has a case and is skipped")
