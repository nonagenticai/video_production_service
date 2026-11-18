"""Request-tenant resolution — the one place a request becomes a ``uuid.UUID``.

w2v-common 0.67.0 deleted every tenant fallback it used to offer
(``AuthSettings.default_tenant_id``, ``DEFAULT_TENANT_ID``,
``UNATTRIBUTED_TENANT_ID``) and flipped ``tenant_id`` to ``uuid.UUID``.  This
service had its own spelling of the same lie: three route helpers returned the
literal string ``"anonymous"`` when ``request.state.tenant_id`` was absent.

That string is now doubly wrong.  ``compilation_jobs.tenant_id`` and friends are
``uuid`` columns (migration ``004``), so writing it raises a ``DataError`` at the
driver; and every unattributed caller collapsing onto ONE identity is exactly the
mis-attribution the 0.67.0 ruling exists to remove — RLS would isolate them into
a shared bucket each of them can read.

So there is no replacement value.  An unresolved tenant is refused.

**401, not 400.**  ``request.state.tenant_id`` is set in exactly one place —
``w2v_common.auth.middleware`` (three assignments, all ``str(tenant_id)`` of an
already-parsed UUID).  Its absence therefore never means "the caller sent a bad
field"; it means no authenticated identity reached the handler at all (auth
middleware not installed, or the route wrongly listed as public).  That is the
condition 401 names, and it is the shape ``slide_service/src/api/routes.py``
chose at the same seam.  ``video_compiler_service``'s 400 is right for ITS seam:
there the tenant arrives in a job payload, so a bad one really is a bad request.
"""

from __future__ import annotations

import uuid

from fastapi import HTTPException, Request
from w2v_common.tenant import InvalidTenantIdError, TenantError, parse_tenant_id

__all__ = [
    "NilTenantIdError",
    "parse_request_tenant",
    "require_request_tenant",
]

_UNAUTHENTICATED_DETAIL = (
    "No tenant identity on this request. Every tenant-scoped call must carry a "
    "real registered tenant UUID — there is no default and no anonymous tenant."
)


class NilTenantIdError(InvalidTenantIdError):
    """The all-zeros UUID was supplied as a tenant.

    It is the exact sentinel 0.67.0 removed, and it is *registered* in
    ``licenses.tenants`` as ``default-anon``, so it passes both a type check and a
    dirty-row audit while mis-attributing every unowned row to one bucket.  A
    ``None``-only guard would leave that hole wide open, so it is rejected BY
    VALUE — which also covers the undashed, braced and ``urn:uuid:`` spellings,
    since ``uuid.UUID`` normalises all four before ``.int`` is read.

    **Defence-in-depth as of w2v-common 0.68.0, no longer the active gate.**
    ``parse_tenant_id`` used to accept the nil uuid, which is what this subclass
    existed to correct.  It now refuses it on both of its own paths (an
    already-constructed ``uuid.UUID`` and text), so it raises
    :class:`InvalidTenantIdError` before this type can ever be constructed.  It is
    kept, unreachable, so that a regression in the library does not silently
    reopen the hole — and it subclasses ``InvalidTenantIdError`` precisely so
    callers and tests need not care which layer refused.
    """


def parse_request_tenant(value: object, *, source: str = "tenant_id") -> uuid.UUID:
    """Coerce ``value`` to a real tenant UUID, or raise a :class:`TenantError`.

    Never returns a sentinel and never returns ``None``.

    The nil-UUID rejection below is now unreachable: since w2v-common 0.68.0
    ``parse_tenant_id`` refuses the nil uuid itself and raises first.  It is
    retained as defence-in-depth — see :class:`NilTenantIdError`.
    """
    tenant = parse_tenant_id(value, source=source)
    if tenant.int == 0:
        raise NilTenantIdError(
            f"{source}: the nil UUID is a removed sentinel, not a registered tenant"
        )
    return tenant


def require_request_tenant(request: Request) -> uuid.UUID:
    """Return the caller's tenant as a ``uuid.UUID``, or raise ``HTTPException(401)``.

    The single entry point for route code.  Fails loud: there is deliberately no
    ``default=`` parameter, because a caller able to supply a default is a caller
    able to reintroduce the sentinel.
    """
    try:
        return parse_request_tenant(
            getattr(request.state, "tenant_id", None),
            source="request.state.tenant_id",
        )
    except TenantError as exc:
        raise HTTPException(status_code=401, detail=_UNAUTHENTICATED_DETAIL) from exc
