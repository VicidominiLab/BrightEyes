import pytest
from _registry import Case, run
from cases import ALL_CASES


@pytest.mark.parametrize(
    "case", ALL_CASES, ids=[f"{case.old}[{case.label}]" for case in ALL_CASES]
)
def test_equivalent(case: Case):
    """Return the same result from the old and the moved code."""
    run(case)
