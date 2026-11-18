"""Regression-prevention AST test: forbid bare ``asyncio.run`` calls
under ``src/services/compilation/``.

Background
----------
Commit ``5158e74`` (P2.1) consolidated the sync→async storage boundary
behind ``StorageService.retrieve_uri_sync``. Commit ``11134a1`` (P3.2's
god-module split) extracted ``_materialize_image_ref`` into
``_segment_builders.py`` and silently reverted the call site back to a
bare ``asyncio.run(self._storage.retrieve_uri(ref))``. Bare
``asyncio.run`` is unsafe: it deadlocks if invoked from inside a running
event loop and bypasses the guarded wrapper that exists precisely to
catch that case.

This test scans every Python file under ``src/services/compilation/``
and fails if it finds any ``asyncio.run(...)`` call. The single
intentional ``asyncio.run`` call in this codebase lives inside
``StorageService.retrieve_uri_sync`` in ``src/services/storage_service.py``
— that file is *not* under ``src/services/compilation/`` so no exception
is needed here.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_asyncio_run_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr != "run":
        return False
    value = func.value
    return isinstance(value, ast.Name) and value.id == "asyncio"


def test_no_bare_asyncio_run_in_compilation() -> None:
    """No file under src/services/compilation/ may call ``asyncio.run(...)``.

    Use ``self._storage.retrieve_uri_sync(...)`` from a worker thread, or
    ``await self._storage.retrieve_uri(...)`` from an async context.
    """
    root = _repo_root() / "src" / "services" / "compilation"
    assert root.is_dir(), f"compilation dir not found: {root}"

    offenders: list[str] = []
    for py in sorted(root.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_asyncio_run_call(node):
                offenders.append(f"{py.relative_to(_repo_root())}:{node.lineno}")

    assert offenders == [], (
        "Bare asyncio.run() leaked into src/services/compilation/. "
        "Use storage.retrieve_uri_sync(...) (from a worker thread) or "
        "await storage.retrieve_uri(...) (from async context). Offenders: "
        f"{offenders}"
    )


def test_retrieve_uri_sync_is_only_asyncio_run_in_services() -> None:
    """Across all of ``src/services/``, the ONLY ``asyncio.run(...)`` call
    permitted is the one inside ``StorageService.retrieve_uri_sync``
    in ``src/services/storage_service.py``.

    The whole point of P2.1 is that there is exactly one well-defined
    sync→async boundary. This test enforces that invariant globally.
    """
    services_root = _repo_root() / "src" / "services"
    storage_file = services_root / "storage_service.py"
    assert services_root.is_dir(), f"services dir not found: {services_root}"
    assert storage_file.is_file(), f"storage_service.py not found: {storage_file}"

    offenders: list[str] = []
    for py in sorted(services_root.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_asyncio_run_call(node)):
                continue
            if py == storage_file:
                # The single intentional call site. Verify it is inside
                # ``retrieve_uri_sync`` and not some new leak.
                enclosing = _enclosing_function_name(tree, node)
                if enclosing == "retrieve_uri_sync":
                    continue
                offenders.append(
                    f"{py.relative_to(_repo_root())}:{node.lineno} "
                    f"(in function {enclosing!r}; only retrieve_uri_sync may call asyncio.run)"
                )
                continue
            offenders.append(f"{py.relative_to(_repo_root())}:{node.lineno}")

    assert offenders == [], (
        f"Bare asyncio.run() found outside StorageService.retrieve_uri_sync. Offenders: {offenders}"
    )


def _enclosing_function_name(tree: ast.AST, target: ast.AST) -> str | None:
    """Return the name of the (Async)FunctionDef that lexically encloses
    ``target``, or ``None`` if ``target`` is at module scope.
    """
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    node: ast.AST | None = target
    while node is not None:
        parent = parents.get(id(node))
        if parent is None:
            return None
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return parent.name
        node = parent
    return None
