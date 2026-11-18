"""Pins for ``src.api.tenant`` — the one place a request becomes a tenant UUID.

The guard tests for the ROUTE seam live next to their routes
(``tests/unit/compilation/test_routes.py::TestGetTenantId`` for the 401 and the
four nil-uuid spellings). What is pinned HERE is the property of the exception
hierarchy that makes those guards easy to defeat by accident.
"""

from __future__ import annotations

import uuid
from typing import cast
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from w2v_common.tenant import InvalidTenantIdError, MissingTenantError, TenantError

from src.api.tenant import NilTenantIdError, parse_request_tenant, require_request_tenant

_NIL = "00000000-0000-0000-0000-000000000000"
_REAL = uuid.UUID("019de900-0000-7000-8000-0000000000d1")


class TestTenantErrorsAreValueErrors:
    """``except ValueError`` is a live foot-gun, not a hypothetical one.

    ``w2v_common.tenant``'s errors subclass ``ValueError``, and the idiom around
    uuid parsing is ``try: uuid.UUID(v) except ValueError:``. Any such block
    wrapped around a tenant parse therefore CATCHES the refusal. That is fine
    when the handler re-refuses (``src/api/events.py`` turns it into a 403) and
    catastrophic when the handler substitutes a fallback — which is exactly what
    0.67.0 removed. These assertions exist so that the day someone narrows
    ``except TenantError`` to ``except ValueError``, or widens a refusal into a
    default, the reason is in front of them.
    """

    @pytest.mark.parametrize(
        "exc_type", [TenantError, MissingTenantError, InvalidTenantIdError, NilTenantIdError]
    )
    def test_every_tenant_error_is_a_value_error(self, exc_type: type[Exception]) -> None:
        assert issubclass(exc_type, ValueError)

    def test_nil_tenant_error_is_an_invalid_tenant_id_error(self) -> None:
        """So a handler catching the library's own type also catches ours."""
        assert issubclass(NilTenantIdError, InvalidTenantIdError)

    def test_a_bare_except_value_error_would_swallow_the_refusal(self) -> None:
        """Demonstrates the swallow, so the guard below is not mistaken for luck."""
        swallowed = False
        try:
            parse_request_tenant(None, source="demo")
        except ValueError:
            swallowed = True
        assert swallowed, "MissingTenantError stopped being a ValueError — re-read this module"


class TestParseRequestTenant:
    def test_returns_the_uuid_for_a_real_tenant(self) -> None:
        assert parse_request_tenant(str(_REAL)) == _REAL

    def test_passes_a_uuid_object_through(self) -> None:
        assert parse_request_tenant(_REAL) == _REAL

    def test_missing_raises_missing_tenant_error_not_a_sentinel(self) -> None:
        with pytest.raises(MissingTenantError):
            parse_request_tenant(None)

    @pytest.mark.parametrize(
        "spelling",
        [
            _NIL,
            _NIL.replace("-", ""),
            "{" + _NIL + "}",
            "urn:uuid:" + _NIL,
        ],
    )
    def test_the_nil_uuid_is_refused_by_value_in_every_spelling(self, spelling: str) -> None:
        """The nil uuid is refused BY VALUE, so all four spellings fail one check.

        WHICH layer refuses moved in w2v-common 0.68.0: ``parse_tenant_id`` used
        to accept the nil uuid and this service's ``NilTenantIdError`` was the
        active gate. The library now rejects it on both of its own paths, so it
        raises first, with :class:`InvalidTenantIdError` and its own wording.

        This asserts the *reason* (nil), not the *owner*, and deliberately does
        not name ``NilTenantIdError``: the base type is what both layers raise
        (``NilTenantIdError`` subclasses it), so the pin holds whichever one
        fires and would survive the guard moving back. ``match`` keeps it strict
        — a generic "tenant id is not a UUID" refusal does not satisfy it, so
        this cannot pass for the wrong reason.
        """
        with pytest.raises(InvalidTenantIdError, match=r"(?i)nil uuid"):
            parse_request_tenant(spelling)

    def test_a_nil_uuid_object_is_refused_too(self) -> None:
        """The str lane is not the only lane; ``uuid.UUID(int=0)`` short-circuits it."""
        with pytest.raises(InvalidTenantIdError, match=r"(?i)nil uuid"):
            parse_request_tenant(uuid.UUID(int=0))


class TestRequireRequestTenant:
    def _request(self, value: object) -> Request:
        """A stand-in for the Request the auth middleware has already stamped.

        ``value=None`` models the UNAUTHENTICATED case, so ``state`` must not
        merely hold ``None`` — the attribute has to be genuinely absent, which
        is why the spec is emptied rather than the value set.
        """
        request = MagicMock(spec=Request)
        if value is None:
            request.state = MagicMock(spec=[])
        else:
            request.state.tenant_id = value
        return cast(Request, request)

    def test_returns_the_uuid_the_middleware_stashed(self) -> None:
        assert require_request_tenant(self._request(str(_REAL))) == _REAL

    @pytest.mark.parametrize("value", [None, "", "anonymous", _NIL, "not-a-uuid"])
    def test_every_unusable_identity_becomes_a_401(self, value: object) -> None:
        """401, not 400: ``request.state.tenant_id`` has exactly one producer.

        The auth middleware. Its absence or unusability therefore never means
        "the caller sent a bad field" — it means no authenticated identity
        reached the handler.
        """
        with pytest.raises(HTTPException) as exc:
            require_request_tenant(self._request(value))
        assert exc.value.status_code == 401, value

    def test_it_never_leaks_a_bare_value_error_to_the_caller(self) -> None:
        """The whole point of the helper: route code must not need to know.

        If a refusal escaped as a plain ``ValueError``, FastAPI would turn it
        into a 500 and an operator would go looking for a crash instead of an
        unauthenticated caller.
        """
        with pytest.raises(HTTPException):
            require_request_tenant(self._request(None))
