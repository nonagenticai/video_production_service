"""Tenant provenance across background jobs — the OUTCOME, not the mechanism.

These tests assert what the service needs to be true, not how ``w2v_common``
achieves it. The previous version of this file asserted the mechanism ("the bare
call clobbers provenance") and therefore went red the moment the library was
fixed — a test that fails on an improvement is a test that blocks improvements.

The contract, as of w2v-common 0.67.7, is three-way. ``set_identity_context``
distinguishes an OMITTED ``tenant_source`` from an explicit ``None`` via a
``KEEP_TENANT_SOURCE`` sentinel default::

    omitted, SAME tenant as ambient    → provenance PRESERVED
    omitted, DIFFERENT tenant          → provenance DROPPED to None
    tenant_source=None   (explicit)    → cleared
    tenant_source=<value> (explicit)   → ALWAYS wins

Both call-site families in this service therefore use the SAME bare
three-argument call, and get opposite — and individually correct — behaviour:

* ``src/api/routes/`` ``_bg`` entrypoints re-seed the tenant they INHERITED (for
  the Phase-D RLS bind), so omission PRESERVES the request's provenance and a
  metered outbound hop stays vouched. Without it ``build_outbound_identity``
  reports ``None`` as ``'unrecorded'`` and refuses the destination — the defect
  measured 2026-08-28 in the w2v_frontend E2E matrix, fixed upstream in
  ai_content_generation_service #109.
* ``*/cleanup_service.py`` reapers rebind a DIFFERENT tenant per row over a
  cross-tenant scan, so the same omission DROPS provenance every iteration and
  the loop fails closed by itself.

⚠️ The security half is ``test_bare_call_on_a_different_tenant_fails_closed``.
A naive "always preserve what was inherited" implementation would pass the
preservation tests and FAIL that one: it would carry the first tenant's proof
onto a rebind of somebody else's tenant. Any future library change must keep
both halves green.

Passing ``tenant_source=get_tenant_source_from_context()`` at a call site is
NOT the safe option and is deliberately absent from this service: an explicit
value always wins, so it bypasses the different-tenant drop entirely.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
import uuid

from w2v_common.auth.identity_context import (
    clear_identity_context,
    get_tenant_source_from_context,
    set_identity_context,
)
from w2v_common.tenant import TenantSource

TENANT = uuid.UUID("e148e8af-3751-49bf-a0fa-56ab46d31ddc")
OTHER_TENANT = uuid.UUID("0d9c3b2a-71f4-4a55-9c8e-2b6f1d34ac07")
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def test_detached_task_keeps_inherited_provenance() -> None:
    """A bg task that re-seeds ITS OWN tenant still has vouched provenance.

    Models exactly what the ``_bg`` entrypoints do: the request scope binds
    provenance, ``asyncio.create_task`` copies the context, and the task re-seeds
    the same tenant so the Phase-D RLS bind applies to its fresh session. The
    outcome that matters is that a metered hop from inside the task would not
    fail closed.
    """
    seen: dict[str, object] = {}

    async def bg() -> None:
        # the real call shape in src/api/routes/ — no tenant_source keyword
        set_identity_context(TENANT, None, None)
        seen["source"] = get_tenant_source_from_context()

    async def drive() -> None:
        clear_identity_context()
        set_identity_context(TENANT, None, None, tenant_source=TenantSource.LICENSE_TOKEN)
        await asyncio.gather(asyncio.create_task(bg()))
        clear_identity_context()

    asyncio.run(drive())

    assert seen["source"] is TenantSource.LICENSE_TOKEN, (
        "a background job re-seeding its own tenant lost provenance; every metered "
        "hop from it will be refused as 'unrecorded'"
    )


def test_bare_call_on_a_different_tenant_fails_closed() -> None:
    """The security half: rebinding ANOTHER tenant must not inherit the proof.

    This is what stops a reaper loop from vouching tenant A's licence for tenant
    B's row. A naive "always preserve" fix passes every other test in this file
    and fails this one.
    """
    clear_identity_context()
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.LICENSE_TOKEN)
    assert get_tenant_source_from_context() is TenantSource.LICENSE_TOKEN

    set_identity_context(OTHER_TENANT, None, None)
    assert get_tenant_source_from_context() is None, (
        "rebinding a DIFFERENT tenant carried the previous tenant's provenance "
        "forward — that is a cross-tenant escalation, not a preserve"
    )
    clear_identity_context()


def test_explicit_none_clears_even_for_the_same_tenant() -> None:
    """An explicit ``tenant_source=None`` is a deliberate clear, not an omission."""
    clear_identity_context()
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.LICENSE_TOKEN)
    assert get_tenant_source_from_context() is TenantSource.LICENSE_TOKEN

    set_identity_context(TENANT, None, None, tenant_source=None)
    assert get_tenant_source_from_context() is None, (
        "an explicit tenant_source=None no longer clears provenance — callers can "
        "no longer deliberately drop a vouch"
    )
    clear_identity_context()


def test_absence_of_context_does_not_invent_provenance() -> None:
    """With nothing bound, the bare call must yield ``None``.

    Absence must never be upgraded into a vouch: a job started outside any
    request has no proof of anything, and must fail closed at the first metered
    destination.
    """
    clear_identity_context()
    set_identity_context(TENANT, None, None)
    assert get_tenant_source_from_context() is None, (
        "provenance was INVENTED from an empty context — an unauthenticated "
        "background job would be treated as vouched"
    )
    clear_identity_context()


def _call_sites(root: pathlib.Path) -> list[tuple[str, bool]]:
    """Return ``(file:line, passes_tenant_source)`` for every real call under ``root``.

    Parsed with ``ast``, deliberately, not matched with a regex. A regex over the
    source text also matches the call shape quoted in a docstring or a comment — and
    the module docstrings in this repo quote BOTH shapes to explain the difference.
    A text scan therefore reports the explanation as an offender.
    """
    found: list[tuple[str, bool]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "set_identity_context":
                continue
            explicit = any(kw.arg == "tenant_source" for kw in node.keywords)
            found.append((f"{path.relative_to(_SRC)}:{node.lineno}", explicit))
    return found


def test_reaper_loops_never_pass_tenant_source() -> None:
    """Cleanup reapers must not escalate by pinning an explicit provenance.

    Source-level ON PURPOSE, and only in the forbidding direction: an explicit
    ``tenant_source=`` ALWAYS wins, so adding one inside a loop that rebinds a
    different tenant per row would bypass the library's tenant-change drop and
    vouch one tenant's licence for another tenant's data. No behavioural test of
    one reaper generalises across the other three.
    """
    reapers = [w for w, _ in _call_sites(_SRC / "services") if "cleanup_service" in w]
    escalations = [
        where
        for where, explicit in _call_sites(_SRC / "services")
        if "cleanup_service" in where and explicit
    ]
    # Vacuity floor: the assertion below passes trivially if the files are renamed
    # or the calls disappear, so pin the population it is measuring.
    assert len(reapers) == 6, f"expected 6 reaper seeding calls, saw {reapers}"
    assert not escalations, (
        "a cleanup reaper now passes tenant_source — it iterates rows across tenants, "
        f"so an explicit value vouches one tenant for another's row: {escalations}"
    )
