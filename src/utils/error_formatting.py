"""Error formatting utilities."""

from __future__ import annotations

import traceback


def format_error(
    exc: BaseException,
    *,
    max_length: int = 4000,
    include_traceback: bool = False,
) -> str:
    """Return a human-readable description of *exc*."""
    type_name = type(exc).__name__

    text = str(exc)
    if not text:
        text = repr(exc)

    formatted = f"{type_name}: {text}"

    if include_traceback and exc.__traceback__:
        tb_lines = traceback.format_tb(exc.__traceback__, limit=10)
        tb_str = "".join(tb_lines)
        formatted = f"{formatted}\nTraceback:\n{tb_str}"

    if len(formatted) > max_length:
        suffix = "... [truncated]"
        formatted = formatted[: max_length - len(suffix)] + suffix

    return formatted
