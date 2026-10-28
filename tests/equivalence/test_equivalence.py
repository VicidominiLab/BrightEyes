import pytest

from ._registry import Case, run
from .cases import ALL_CASES


@pytest.mark.parametrize(
    "case", ALL_CASES, ids=lambda case: f"{case.old}[{case.label}]"
)
def test_equivalent(case: Case) -> None:
    """Return the same result from the old and the moved code."""
    run(case)
