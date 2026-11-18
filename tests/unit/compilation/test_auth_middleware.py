"""Unit tests for auth middleware."""

from __future__ import annotations

import ipaddress
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import w2v_common.auth.middleware as auth_middleware
from pydantic import ValidationError
from w2v_common.tenant import MissingTenantError

from src.auth.middleware import (
    PUBLIC_ENDPOINTS,
    PUBLIC_PREFIXES,
    AuthMiddleware,
    _get_license_cache,
    _introspect_license,
    _is_trusted_network,
)
from src.auth.models import User

#: A real tenant uuid. w2v-common 0.67.0 made ``tenant_id`` a ``uuid.UUID``
#: everywhere, so the old ``"tenant-1"`` / ``"t1"`` slugs are no longer tenants.
_TENANT = uuid.UUID("019de900-0000-7000-8000-0000000000a1")

# ---------------------------------------------------------------------------
# User model tests
# ---------------------------------------------------------------------------


class TestUserModel:
    def test_user_creation(self):
        user = User(sub="tenant-1", preferred_username="admin", tenant_id=_TENANT)
        assert user.sub == "tenant-1"
        assert user.preferred_username == "admin"
        assert user.tenant_id == _TENANT

    def test_user_tenant_id_rejects_a_slug(self):
        """w2v-common 0.67.0: ``User.tenant_id`` is a ``uuid.UUID``, not a slug."""
        with pytest.raises(ValidationError):
            # ``model_validate`` (not the typed ctor) so the bad value reaches
            # pydantic instead of being rejected by the type checker first.
            User.model_validate({"sub": "sub-1", "tenant_id": "tenant-1"})

    def test_effective_tenant_id_with_tenant(self):
        user = User(sub="sub-1", tenant_id=_TENANT)
        assert user.effective_tenant_id == _TENANT

    def test_effective_tenant_id_refuses_to_fall_back_to_sub(self):
        """Was ``assert user.effective_tenant_id == "sub-1"``.

        Falling back to ``sub`` made a SUBJECT id masquerade as a TENANT id — the
        same substitution class w2v-common 0.67.0 removed, and one that a uuid
        column cannot even store. It now raises, so the assertion is inverted.
        """
        user = User(sub="sub-1", tenant_id=None)
        with pytest.raises(MissingTenantError):
            _ = user.effective_tenant_id

    def test_default_username(self):
        user = User(sub="sub-1")
        assert user.preferred_username == "anonymous"


# ---------------------------------------------------------------------------
# Trusted network tests
# ---------------------------------------------------------------------------


class TestTrustDefaultIsEmpty:
    """w2v-common >= 0.89.0: with TRUSTED_CIDRS unset, NO peer is trusted.

    Every other trusted-network test here opts in via the
    ``explicit_trusted_cidrs`` fixture; this one pins the library default
    itself, so a regression back to an implicit loopback/private-range default
    goes red. Runs in a fresh interpreter so no module is reloaded mid-suite.
    """

    def test_unset_env_trusts_nothing(self):
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "TRUSTED_CIDRS"}
        probe = (
            "import w2v_common.auth.middleware as m\n"
            "assert m.TRUSTED_NETWORKS == [], m.TRUSTED_NETWORKS\n"
            "for h in ('127.0.0.1', '::1', '10.0.0.1', '10.42.5.10', '172.16.0.1', '192.168.1.1'):\n"
            "    assert m._is_trusted_network(h) is False, h\n"
            "print('EMPTY-DEFAULT-OK')\n"
        )
        proc = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter + a constant probe
            [sys.executable, "-c", probe],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "EMPTY-DEFAULT-OK" in proc.stdout


@pytest.mark.usefixtures("explicit_trusted_cidrs")
class TestTrustedNetwork:
    def test_trusted_networks_populated(self):
        # Read the live module global (an import-time name binding would not
        # see the configured lanes) and assert loopback is actually covered.
        assert len(auth_middleware.TRUSTED_NETWORKS) > 0
        assert any(
            ipaddress.ip_address("127.0.0.1") in net for net in auth_middleware.TRUSTED_NETWORKS
        )

    def test_localhost_is_trusted(self):
        assert _is_trusted_network("127.0.0.1") is True

    def test_ipv6_localhost_is_trusted(self):
        assert _is_trusted_network("::1") is True

    def test_private_10_is_trusted(self):
        assert _is_trusted_network("10.0.0.1") is True

    def test_private_172_is_trusted(self):
        assert _is_trusted_network("172.16.0.1") is True

    def test_private_192_not_trusted(self):
        # 192.168.0.0/16 is not among the configured lanes: it was dropped to
        # close the SSR-proxy source-IP bypass (see
        # project_orchestrator_trusted_networks_bypass_2026-04-18).
        assert _is_trusted_network("192.168.1.1") is False

    def test_public_ip_not_trusted(self):
        assert _is_trusted_network("8.8.8.8") is False

    def test_none_host_not_trusted(self):
        assert _is_trusted_network(None) is False

    def test_empty_string_not_trusted(self):
        assert _is_trusted_network("") is False

    def test_invalid_host_not_trusted(self):
        assert _is_trusted_network("not-an-ip") is False


# ---------------------------------------------------------------------------
# Public endpoints tests
# ---------------------------------------------------------------------------


class TestPublicEndpoints:
    def test_health_is_public(self):
        assert "/health" in PUBLIC_ENDPOINTS

    def test_health_ready_is_public(self):
        assert "/health/ready" in PUBLIC_ENDPOINTS

    def test_health_live_is_public(self):
        assert "/health/live" in PUBLIC_ENDPOINTS

    def test_docs_is_public(self):
        assert "/docs" in PUBLIC_ENDPOINTS

    def test_openapi_json_is_public(self):
        assert "/openapi.json" in PUBLIC_ENDPOINTS

    def test_public_prefixes(self):
        assert "/docs" in PUBLIC_PREFIXES
        assert "/health/" in PUBLIC_PREFIXES


# ---------------------------------------------------------------------------
# License introspection tests
# ---------------------------------------------------------------------------


class TestIntrospectLicense:
    def setup_method(self):
        _get_license_cache().clear()

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_successful_introspection(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "tenant_id": "t1"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await _introspect_license("valid-token")
        assert result is not None
        assert result["active"] is True
        assert result["tenant_id"] == "t1"

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_inactive_license(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": False, "reason": "expired"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await _introspect_license("expired-token")
        assert result is not None
        assert result["active"] is False

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_introspection_failure_returns_none(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("connection error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await _introspect_license("any-token")
        assert result is None

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_introspection_caches_active_result(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "tenant_id": "t1"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        # First call populates cache
        await _introspect_license("cached-token")
        # Second call should use cache
        result = await _introspect_license("cached-token")
        assert result is not None
        assert result["active"] is True
        # Only one HTTP call should have been made (second was cached)
        assert mock_client.post.await_count == 1


# ---------------------------------------------------------------------------
# AuthMiddleware dispatch tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("explicit_trusted_cidrs")
class TestAuthMiddleware:
    def _make_middleware(self):
        app = MagicMock()
        return AuthMiddleware(app)

    def _make_request(
        self,
        path: str = "/api/compile",
        auth_header: str | None = None,
        x_tenant_id: str | None = None,
        client_host: str | None = "8.8.8.8",
    ):
        request = MagicMock()
        request.url.path = path
        request.state = MagicMock(spec=[])

        headers = {}
        if auth_header:
            headers["Authorization"] = auth_header
        if x_tenant_id:
            headers["X-Tenant-Id"] = x_tenant_id
        request.headers = MagicMock()
        request.headers.get = lambda k, d=None: headers.get(k, d)

        if client_host:
            request.client = MagicMock()
            request.client.host = client_host
        else:
            request.client = None

        return request

    async def test_public_endpoint_passes_through(self):
        middleware = self._make_middleware()
        request = self._make_request(path="/health")
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        await middleware.dispatch(request, call_next)
        call_next.assert_awaited_once_with(request)

    async def test_public_prefix_passes_through(self):
        middleware = self._make_middleware()
        request = self._make_request(path="/docs/something")
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        await middleware.dispatch(request, call_next)
        call_next.assert_awaited_once_with(request)

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_valid_bearer_token(self, mock_introspect):
        mock_introspect.return_value = {
            "active": True,
            "tenant_id": str(_TENANT),
            "subject_id": "user1",
        }
        middleware = self._make_middleware()
        request = self._make_request(x_tenant_id="Bearer valid-token")
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        await middleware.dispatch(request, call_next)
        call_next.assert_awaited_once()
        assert request.state.tenant_id == str(_TENANT)

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_invalid_bearer_token(self, mock_introspect):
        mock_introspect.return_value = {"active": False, "reason": "expired"}
        middleware = self._make_middleware()
        request = self._make_request(x_tenant_id="Bearer invalid-token")
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401
        call_next.assert_not_awaited()

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_introspection_failure_returns_401(self, mock_introspect):
        mock_introspect.return_value = None
        middleware = self._make_middleware()
        request = self._make_request(x_tenant_id="Bearer any-token")
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401

    async def test_trusted_network_no_token_fail_closed_default(self):
        """w2v_common 0.6.9: trusted-network unauth fails closed (401) by default."""
        middleware = self._make_middleware()
        request = self._make_request(client_host="127.0.0.1")
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401
        call_next.assert_not_awaited()

    @patch("w2v_common.auth.middleware._trust_incluster_identity", new=lambda: True)
    async def test_trusted_network_no_token_is_refused_not_defaulted(self):
        """w2v-common 0.67.0 deleted ``DEFAULT_TENANT_ID``.

        This test used to assert the SUBSTITUTION (``state.tenant_id ==
        "default-tenant"``). The trusted network vouches for the CALLER, never
        for the TENANT, so an in-cluster call that forgets to forward the
        requestor's tenant is now refused instead of being attributed to a fake
        one. Asserting the refusal is the same test, inverted — not a weaker one.
        """
        middleware = self._make_middleware()
        request = self._make_request(client_host="127.0.0.1")
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401
        call_next.assert_not_awaited()
        assert not hasattr(request.state, "tenant_id")

    @patch("w2v_common.auth.middleware._trust_incluster_identity", new=lambda: True)
    async def test_trusted_network_with_x_tenant_header_opt_in(self):
        """A REAL tenant uuid on the wire is still accepted on the opt-in lane."""
        middleware = self._make_middleware()
        request = self._make_request(client_host="127.0.0.1", x_tenant_id=str(_TENANT))
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        await middleware.dispatch(request, call_next)
        call_next.assert_awaited_once()
        assert request.state.tenant_id == str(_TENANT)

    @patch("w2v_common.auth.middleware._trust_incluster_identity", new=lambda: True)
    async def test_trusted_network_with_slug_x_tenant_header_is_refused(self):
        """A legacy slug is not a tenant: refused, never coerced or truncated."""
        middleware = self._make_middleware()
        # Deliberately a SLUG, not a uuid — that is the whole point of this test.
        request = self._make_request(client_host="127.0.0.1", x_tenant_id="my-tenant")
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401
        call_next.assert_not_awaited()

    async def test_no_token_untrusted_network_returns_401(self):
        middleware = self._make_middleware()
        request = self._make_request(client_host="8.8.8.8")
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401
        call_next.assert_not_awaited()

    async def test_no_client_returns_401(self):
        middleware = self._make_middleware()
        request = self._make_request(client_host=None)
        call_next = AsyncMock()

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 401

    def test_extract_license_token_from_opaque_authorization_bearer(self):
        """An OPAQUE `Authorization: Bearer` IS a license source (w2v_common 0.74.0).

        0.74.0 deliberately reversed the old FR-401 invariant: `Authorization`
        is now a second accepted licence lane, because the platform's
        `license-authz` ext-authz gate reads the token ONLY from
        `Authorization`. Sitting behind that gate requires accepting it here.
        """
        middleware = self._make_middleware()
        request = self._make_request(auth_header="Bearer my-token")
        assert middleware._extract_license_token(request) == "my-token"

    def test_extract_license_token_ignores_jwt_shaped_authorization_bearer(self):
        """A JWT-shaped `Authorization: Bearer` is still NOT a licence token.

        This is the half of the old FR-401 invariant that SURVIVES 0.74.0 and is
        still worth pinning: the new lane must not swallow Keycloak JWTs into
        licence introspection. `_looks_like_jwt` (three non-empty dot segments)
        is what keeps the two apart — a JWT goes down the JWT path instead.
        Mixing the two caused the 2026-04-26 prod 401s on /api/assets/*.
        """
        middleware = self._make_middleware()
        request = self._make_request(auth_header="Bearer header.payload.signature")
        assert middleware._extract_license_token(request) is None
        # ...and it is still reachable as a JWT candidate.
        assert middleware._extract_jwt_bearer(request) == "header.payload.signature"

    def test_extract_jwt_bearer_from_authorization_header(self):
        """Authorization Bearer feeds JWT validation only, never license introspection."""
        middleware = self._make_middleware()
        request = self._make_request(auth_header="Bearer my-token")
        assert middleware._extract_jwt_bearer(request) == "my-token"

    def test_extract_license_token_from_x_tenant_bearer(self):
        middleware = self._make_middleware()
        request = self._make_request(x_tenant_id="Bearer tenant-token")
        token = middleware._extract_license_token(request)
        assert token == "tenant-token"

    def test_extract_license_token_none_when_missing(self):
        middleware = self._make_middleware()
        request = self._make_request()
        token = middleware._extract_license_token(request)
        assert token is None

    async def test_trailing_slash_stripped(self):
        """A trailing slash must resolve to the public endpoint, not to a guarded path.

        The path is ``/metrics/`` on purpose. ``/health/`` is ALREADY covered by
        ``PUBLIC_PREFIXES`` and passes with or without the normalisation, so it
        cannot witness this behaviour; ``/metrics`` is an exact-match entry only.
        Without the ``rstrip("/")`` the scraper URL misses PUBLIC_ENDPOINTS and
        the middleware answers 401. So what this pins is the RESPONSE: the
        route's own 200 comes back, not a challenge the middleware synthesised.
        """
        middleware = self._make_middleware()
        request = self._make_request(path="/metrics/")
        response = MagicMock(status_code=200)
        call_next = AsyncMock(return_value=response)

        result = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert result is response
        assert result.status_code == 200

    async def test_trailing_slash_does_not_make_a_guarded_path_public(self):
        """The normalisation must not turn an unauthenticated API call into a pass.

        Positive control for the test above: strip the slash, yes — but
        ``/api/compile/`` still has to be rejected.
        """
        middleware = self._make_middleware()
        request = self._make_request(path="/api/compile/")
        call_next = AsyncMock()

        result = await middleware.dispatch(request, call_next)

        assert result.status_code == 401
        call_next.assert_not_awaited()
