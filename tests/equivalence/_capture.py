import contextlib
import io
import warnings
from collections.abc import Callable
from typing import Any

ERRORS = (
    ArithmeticError,
    ImportError,
    LookupError,
    NotImplementedError,
    OSError,
    TypeError,
    ValueError,
)


def error(call: Callable[[], Any]) -> str:
    """Return the name of the exception `call` raises, and fail when it raises none."""
    try:
        call()
    except ERRORS as exc:
        return type(exc).__name__
    raise AssertionError("no exception raised")


def warned(call: Callable[[], Any]) -> tuple[Any, list[tuple[str, str]]]:
    """Return the result of `call` with the category and text of each warning it issued.

    The text has the old module name of the MCS reader replaced by its new
    one, because the move rewrote it inside a warning message.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = call()
    return result, [
        (
            item.category.__name__,
            str(item.message).replace("brighteyes_mcs_reader", "brighteyes.mcs.reader"),
        )
        for item in record
    ]


def printed(call: Callable[[], Any]) -> tuple[Any, str]:
    """Return the result of `call` with the text it printed."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result = call()
    return result, stdout.getvalue()
