"""Tests for src/utils/error_formatting.py -- format_error utility."""

from __future__ import annotations

from src.utils.error_formatting import format_error

# ── Basic formatting ──────────────────────────────────────────────


class TestBasicFormatting:
    def test_simple_exception(self):
        result = format_error(ValueError("bad value"))
        assert result == "ValueError: bad value"

    def test_type_prefix(self):
        result = format_error(RuntimeError("oops"))
        assert result.startswith("RuntimeError: ")

    def test_custom_exception(self):
        class MyError(Exception):
            pass

        result = format_error(MyError("custom"))
        assert result == "MyError: custom"


# ── Empty str fallback ────────────────────────────────────────────


class TestEmptyStrFallback:
    def test_bare_runtime_error(self):
        exc = RuntimeError()
        result = format_error(exc)
        assert "RuntimeError" in result
        # Should not be empty after type prefix
        assert len(result) > len("RuntimeError: ")

    def test_empty_string_message(self):
        exc = ValueError("")
        result = format_error(exc)
        assert "ValueError" in result
        assert len(result) > len("ValueError: ")

    def test_repr_fallback_content(self):
        exc = RuntimeError()
        result = format_error(exc)
        # repr(RuntimeError()) == "RuntimeError()"
        assert "RuntimeError()" in result


# ── Truncation ────────────────────────────────────────────────────


class TestTruncation:
    def test_long_message_truncated(self):
        exc = ValueError("x" * 5000)
        result = format_error(exc, max_length=100)
        assert len(result) <= 100
        assert result.endswith("... [truncated]")

    def test_exact_length_not_truncated(self):
        exc = ValueError("short")
        result = format_error(exc, max_length=4000)
        assert "truncated" not in result

    def test_custom_max_length(self):
        exc = ValueError("a" * 200)
        result = format_error(exc, max_length=50)
        assert len(result) <= 50


# ── Traceback ─────────────────────────────────────────────────────


class TestTraceback:
    def test_no_traceback_by_default(self):
        try:
            raise ValueError("test")
        except ValueError as exc:
            result = format_error(exc)
        assert "Traceback" not in result

    def test_include_traceback(self):
        try:
            raise ValueError("test")
        except ValueError as exc:
            result = format_error(exc, include_traceback=True)
        assert "Traceback:" in result
        # Python 3.14's traceback.format_tb() may omit source lines depending
        # on how the file is loaded; the stable signal is the frame ref
        # (filename + function name).
        assert "test_include_traceback" in result

    def test_traceback_without_tb_object(self):
        exc = ValueError("no tb")
        result = format_error(exc, include_traceback=True)
        assert "Traceback" not in result

    def test_traceback_truncated(self):
        try:
            raise ValueError("x" * 3000)
        except ValueError as exc:
            result = format_error(exc, include_traceback=True, max_length=100)
        assert len(result) <= 100
        assert result.endswith("... [truncated]")
