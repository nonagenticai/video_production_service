"""Partial ratchet over swallowed failures on the compile path.

WHAT THIS IS -- AND IS NOT
--------------------------
This is NOT a completeness gate, and a green run here is NOT evidence that
every drop is reported. It catches exactly ONE swallow shape:

    an ``except`` handler, in one of five modules, that calls
    ``logger.warning``/``logger.error`` and does not re-raise.

Measured against handlers appended to ``_effects.py`` (each run in isolation):

    A  except + logger.warning, no record   -> 1 failed   CAUGHT
    B  same but logger.info                 -> 6 passed   MISSED
    C  except: return, no log at all        -> 6 passed   MISSED
    D  contextlib.suppress(...)             -> 6 passed   MISSED
    E  early-return drop, no except         -> 6 passed   MISSED

Four of five shapes pass. All four occur in this codebase: ``_mix_background
_music`` was shape C (a bare return with no log at all), and
``title_card_generator.py`` uses ``contextlib.suppress``. Shape E is how a
missing-config drop is usually written.

``_ALLOWED`` is an escape hatch, not a proof obligation: adding one plausible
entry silences a real finding, and ``test_allowlist_has_no_stale_entries`` will
not object so long as the named function still swallows something. Anyone
adding an entry is making a claim on their own authority.

KNOWN UNCOVERED, BY MODULE
--------------------------
``title_card_generator.py`` is OUTSIDE the five modules scanned and contains
three unreported drops -- ``_build_background`` (silent gradient fallback),
``generate_image`` (logo dropped), ``generate_video`` (static instead of
animated). They are reached with the very same ``background_image`` /
``logo_path`` / ``animation`` inputs whose guard-refusal paths ARE reported, so
the same feature is reported when refused and silent when it fails to render.

So: use this to stop a regression in the one shape it measures. Do not cite it
as evidence that the degradation channel is complete. It is not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_COMPILATION_DIR = Path(__file__).resolve().parents[2] / "src" / "services" / "compilation"

# The modules that BUILD THE DELIVERABLE -- i.e. the ones whose swallows can
# change the video the caller downloads, which is all `compilation_report.status`
# makes a claim about.
#
# Every other module is OUT OF SCOPE FOR THIS GUARD, and that is a statement
# about this guard's reach, NOT a judgement that their swallows are harmless.
# `batch_service.py` in particular derives a terminal job status of its own.
# That status WAS wrong -- a job lost to a BaseException incremented neither
# counter and the batch landed 'completed' -- and #171 fixed it there, at the
# aggregate, by deriving from what was produced. It stays out of this guard's
# reach because this guard measures per-feature drops on the compile path, not
# batch aggregation. Its absence here is a statement about scope, not a verdict.
_DELIVERABLE_MODULES = (
    "_assembly.py",
    "_effects.py",
    "_finalize.py",
    "_segment_builders.py",
    "compiler_service.py",
)

# (module, enclosing function, justification). Every entry is a handler that logs
# but is NOT a degradation of the deliverable.
_ALLOWED: dict[tuple[str, str], str] = {
    (
        "_effects.py",
        "_apply_subtitles",
    ): "Records via ctx.subtitle_error, which _finalize folds into degradations. "
    "It is no longer a separate API field -- degradations is the only channel "
    "a caller reads. Reported, just not through this call.",
    (
        "_segment_builders.py",
        "_materialize_image_ref",
    ): "Documented fail-soft resolver returning None; the caller drops the part "
    "and counts it in ctx.segments_excluded, already a report field.",
    (
        "_segment_builders.py",
        "_resolve_recording_video_path",
    ): "Same shape: returns None, caller skips the part and increments ctx.segments_excluded.",
    (
        "compiler_service.py",
        "_fire_webhook",
    ): "Webhook delivery is an operational concern about NOTIFYING the caller, "
    "not a property of the video. A failed webhook must not mark the "
    "compile itself partially successful.",
    (
        "compiler_service.py",
        "_progress_heartbeat",
    ): "The heartbeat writes `updated_at` so the stale reaper measures silence "
    "rather than elapsed time. A failed heartbeat WRITE changes nothing about the "
    "video -- the compile carries on untouched -- so it is not a degradation of "
    "the deliverable. Same class as _fire_webhook above: an operational concern "
    "about the job RECORD, not a property of the output. Its real consequence is "
    "that the row ages toward the reaper, which the log line says, and which is "
    "the correct fail-direction (a job that cannot prove it is alive should be "
    "reapable).",
    (
        "_finalize.py",
        "_finalize",
    ): "Not a partial success: the job already left 'processing' (cancelled, or "
    "reaped as stale), so `mark_completed` wrote NOTHING and `_finalize` returns "
    "before building any compilation_report at all. There is no report for a "
    "degradation to appear in, and nothing is delivered to degrade -- the caller "
    "is told 'failed' by the webhook. Recording a degradation here would describe "
    "a deliverable that does not exist.",
}

_LOG_DEGRADE_LEVELS = {"warning", "error"}


def _is_log_call(node: ast.AST, levels: set[str]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in levels
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
    )


def _contains(body: list[ast.stmt], predicate) -> bool:
    return any(predicate(n) for stmt in body for n in ast.walk(stmt))


def _swallowing_handlers() -> list[tuple[str, str, int, bool]]:
    """Return (module, function, lineno, records) for every swallowing handler."""
    found: list[tuple[str, str, int, bool]] = []
    for name in _DELIVERABLE_MODULES:
        path = _COMPILATION_DIR / name
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for handler in ast.walk(fn):
                if not isinstance(handler, ast.ExceptHandler):
                    continue
                # Attribute the handler to its INNERMOST enclosing function.
                inner = [
                    f
                    for f in ast.walk(fn)
                    if isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef)
                    and any(handler is h for h in ast.walk(f))
                ]
                owner = min(inner, key=lambda f: (f.end_lineno or 0) - f.lineno).name
                if _contains(handler.body, lambda n: isinstance(n, ast.Raise)):
                    continue  # re-raises: not swallowed
                if not _contains(handler.body, lambda n: _is_log_call(n, _LOG_DEGRADE_LEVELS)):
                    continue  # no degrade log: not the shape this rule targets
                records = _contains(
                    handler.body,
                    lambda n: (
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "record_degradation"
                    ),
                )
                found.append((path.name, owner, handler.lineno, records))
    return found


def test_the_scan_actually_finds_handlers() -> None:
    """Guard the guard: an AST rule that matches nothing would pass vacuously."""
    handlers = _swallowing_handlers()
    assert len(handlers) >= 10, f"scan found only {len(handlers)} handlers -- rule is broken"
    assert any(records for *_, records in handlers), "scan never detects record_degradation"


def test_every_swallowing_handler_reports_a_degradation() -> None:
    """A swallowed failure that records nothing is published to the caller as success."""
    unreported = [
        (mod, fn, lineno)
        for mod, fn, lineno, records in _swallowing_handlers()
        if not records and (mod, fn) not in _ALLOWED
    ]
    assert not unreported, (
        "These handlers swallow a failure and log it, but record no degradation -- "
        "so _finalize will report the compile as fully successful:\n"
        + "\n".join(f"  {mod}:{lineno} in {fn}()" for mod, fn, lineno in unreported)
        + "\n\nEither call ctx.record_degradation(feature, <fixed literal reason>) "
        "in the handler, or add (module, function) to _ALLOWED with a justification."
    )


def test_allowlist_has_no_stale_entries() -> None:
    """An _ALLOWED entry naming a function that no longer swallows is dead weight."""
    live = {(mod, fn) for mod, fn, _, _ in _swallowing_handlers()}
    stale = sorted(set(_ALLOWED) - live)
    assert not stale, f"_ALLOWED names handlers that no longer exist/swallow: {stale}"


@pytest.mark.parametrize(
    ("module", "function"),
    [
        ("_effects.py", "_mix_background_music"),
        ("_effects.py", "_apply_watermark"),
        ("_segment_builders.py", "_process_title_card"),
    ],
)
def test_known_non_except_drops_are_wired(module: str, function: str) -> None:
    """Drops expressed as an early ``return`` rather than an ``except``.

    The AST rule above cannot see these -- ``_mix_background_music`` returned on
    a missing file with no log at all -- so the specific ones found by hand are
    pinned here by name.
    """
    tree = ast.parse((_COMPILATION_DIR / module).read_text())
    fns = [
        f
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef) and f.name == function
    ]
    assert fns, f"{module}:{function} no longer exists -- update this guard"
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "record_degradation"
        for n in ast.walk(fns[0])
    ), f"{module}:{function}() drops a requested feature without recording it"
