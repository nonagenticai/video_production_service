"""Tests for src/auth/middleware.py -- auth middleware, helpers, introspection."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.auth.middleware import (
    PUBLIC_ENDPOINTS,
    PUBLIC_PREFIXES,
    AuthMiddleware,
    _get_license_cache,
    _introspect_license,
    _is_trusted_network,
)
from tests.conftest import TENANT_U1

#: A real tenant uuid. w2v-common 0.67.0 deleted ``DEFAULT_TENANT_ID`` (the
#: nil-uuid sentinel this file used to import) and made every tenant a
#: ``uuid.UUID``, so slugs like ``"my-tenant"`` / ``"lic-t"`` are refused now.
_TENANT = uuid.UUID("019de900-0000-7000-8000-0000000000b1")

# ── PUBLIC_ENDPOINTS / PUBLIC_PREFIXES ─────────────────────────────


class TestPublicPaths:
    def test_health_in_public_endpoints(self):
        assert "/health" in PUBLIC_ENDPOINTS

    def test_metrics_in_public_endpoints(self):
        assert "/metrics" in PUBLIC_ENDPOINTS

    def test_docs_in_public_endpoints(self):
        assert "/docs" in PUBLIC_ENDPOINTS
        assert "/redoc" in PUBLIC_ENDPOINTS

    def test_public_prefixes_contains_health(self):
        assert "/health/" in PUBLIC_PREFIXES
        assert "/api/health/" in PUBLIC_PREFIXES

    def test_root_in_public_endpoints(self):
        assert "/" in PUBLIC_ENDPOINTS

    def test_openapi_json_in_public_endpoints(self):
        assert "/openapi.json" in PUBLIC_ENDPOINTS


# ── _is_trusted_network ───────────────────────────────────────────


@pytest.mark.usefixtures("explicit_trusted_cidrs")
class TestIsTrustedNetwork:
    def test_localhost_trusted(self):
        assert _is_trusted_network("127.0.0.1") is True

    def test_private_10_trusted(self):
        assert _is_trusted_network("10.0.0.5") is True

    def test_public_ip_not_trusted(self):
        assert _is_trusted_network("8.8.8.8") is False

    def test_none_not_trusted(self):
        assert _is_trusted_network(None) is False

    def test_invalid_string_not_trusted(self):
        assert _is_trusted_network("not-an-ip") is False

    def test_ipv6_loopback_trusted(self):
        assert _is_trusted_network("::1") is True


# ── _extract_license_token (via AuthMiddleware instance) ──────────


class _FakeHeaders(dict):
    """Dict subclass that behaves like Starlette's Headers for .get()."""

    pass


def _make_request(
    headers: dict | None = None, path: str = "/api/record/part", client_host: str = "127.0.0.1"
):
    """Create a minimal mock request."""
    req = MagicMock()
    req.url.path = path
    h = _FakeHeaders(headers or {})
    req.headers = h
    req.client = MagicMock()
    req.client.host = client_host
    req.state = MagicMock(spec=[])  # empty spec so setattr works freely
    return req


class TestExtractLicenseToken:
    def setup_method(self):
        self.mw = AuthMiddleware(app=MagicMock())

    def test_opaque_authorization_bearer_is_a_license_source(self):
        """An OPAQUE `Authorization: Bearer` IS a license source (w2v_common 0.74.0).

        0.74.0 deliberately reversed the old FR-401 invariant: `Authorization`
        became a second accepted licence lane, because the platform's
        `license-authz` ext-authz gate reads the token ONLY from
        `Authorization`. Sitting behind that gate requires accepting it here.
        """
        req = _make_request({"Authorization": "Bearer my-token-123"})
        assert self.mw._extract_license_token(req) == "my-token-123"

    def test_jwt_shaped_authorization_bearer_is_not_a_license_source(self):
        """A JWT-shaped `Authorization: Bearer` is still NOT a licence token.

        The surviving half of FR-401, and the reason the new lane is safe: a
        JWT must never reach licence introspection, it goes down the JWT path
        via `_extract_jwt_bearer`. `_looks_like_jwt` (three non-empty dot
        segments) separates the lanes. Mixing the two caused the 2026-04-26
        prod 401s on /api/assets/*.
        """
        req = _make_request({"Authorization": "Bearer header.payload.signature"})
        assert self.mw._extract_license_token(req) is None
        assert self.mw._extract_jwt_bearer(req) == "header.payload.signature"

    def test_jwt_bearer_extracted_from_authorization(self):
        """Authorization Bearer feeds JWT validation only, never license introspection."""
        req = _make_request({"Authorization": "Bearer my-token-123"})
        assert self.mw._extract_jwt_bearer(req) == "my-token-123"

    def test_bearer_in_x_tenant_id(self):
        req = _make_request({"X-Tenant-Id": "Bearer tenant-tok"})
        assert self.mw._extract_license_token(req) == "tenant-tok"

    def test_no_token(self):
        req = _make_request({})
        assert self.mw._extract_license_token(req) is None

    def test_authorization_without_bearer(self):
        req = _make_request({"Authorization": "Basic abc"})
        assert self.mw._extract_license_token(req) is None

    def test_x_tenant_id_without_bearer_prefix(self):
        """Plain X-Tenant-Id (no bearer prefix) is NOT extracted as a license token."""
        req = _make_request({"X-Tenant-Id": "plain-tenant"})
        assert self.mw._extract_license_token(req) is None


# ── _introspect_license ───────────────────────────────────────────


class TestIntrospectLicense:
    def setup_method(self):
        _get_license_cache().clear()

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_success(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "tenant_id": str(TENANT_U1)}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await _introspect_license("tok-abc")
        assert result is not None
        assert result["active"] is True

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_failure_returns_none(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("conn refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await _introspect_license("tok-bad")
        assert result is None

    @patch("w2v_common.auth.middleware.httpx.AsyncClient")
    async def test_caching(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "tenant_id": "cached-t"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        r1 = await _introspect_license("tok-cache")
        r2 = await _introspect_license("tok-cache")
        assert r1 == r2
        # The underlying HTTP client should only have been called once
        assert mock_client.post.call_count == 1


# ── AuthMiddleware.dispatch ───────────────────────────────────────


@pytest.mark.usefixtures("explicit_trusted_cidrs")
class TestAuthMiddlewareDispatch:
    def setup_method(self):
        self.mw = AuthMiddleware(app=MagicMock())
        _get_license_cache().clear()

    async def test_public_endpoint_passthrough(self):
        req = _make_request(path="/health")
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        await self.mw.dispatch(req, call_next)
        call_next.assert_awaited_once_with(req)

    async def test_trusted_network_no_token_fail_closed_default(self):
        """w2v_common 0.6.9: trusted-network unauth fails closed (401) by default."""
        req = _make_request(path="/api/record/part", client_host="127.0.0.1")
        call_next = AsyncMock()

        response = await self.mw.dispatch(req, call_next)
        assert response.status_code == 401
        call_next.assert_not_awaited()

    async def test_trusted_network_no_x_tenant_id_is_refused_not_defaulted(self):
        """The opt-in lane no longer substitutes a tenant — it refuses.

        This test used to assert ``req.state.tenant_id == DEFAULT_TENANT_ID``,
        i.e. that a caller who forwarded NO tenant was bound to the nil-uuid
        sentinel. w2v-common 0.67.0 deleted that sentinel: the trusted network
        vouches for the CALLER, never for the TENANT. The invariant the old test
        was protecting — "a request must never reach a route with no bound
        tenant" — is now upheld by refusing the request outright, so that is what
        is asserted. Both halves still matter: 401 AND nothing bound.
        """
        req = _make_request(path="/api/record/part", client_host="127.0.0.1")
        call_next = AsyncMock()

        with patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True):
            response = await self.mw.dispatch(req, call_next)

        assert response.status_code == 401
        call_next.assert_not_awaited()
        assert not hasattr(req.state, "tenant_id")

    async def test_trusted_network_with_x_tenant_id_opt_in(self):
        """A REAL forwarded tenant uuid is still accepted on the opt-in lane."""
        req = _make_request(
            path="/api/record/part",
            headers={"X-Tenant-Id": str(_TENANT)},
            client_host="10.0.0.1",
        )
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        with patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True):
            await self.mw.dispatch(req, call_next)
        call_next.assert_awaited_once()
        assert req.state.tenant_id == str(_TENANT)

    async def test_trusted_network_with_slug_x_tenant_id_is_refused(self):
        """A legacy slug is not a tenant: refused, never coerced into a uuid column."""
        req = _make_request(
            path="/api/record/part",
            headers={"X-Tenant-Id": "my-tenant"},
            client_host="10.0.0.1",
        )
        call_next = AsyncMock()

        with patch("w2v_common.auth.middleware._trust_incluster_identity", return_value=True):
            response = await self.mw.dispatch(req, call_next)

        assert response.status_code == 401
        call_next.assert_not_awaited()

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_valid_bearer_token(self, mock_introspect):
        mock_introspect.return_value = {"active": True, "tenant_id": str(_TENANT)}
        req = _make_request(
            path="/api/record/part",
            headers={"X-Tenant-Id": "Bearer valid-tok"},
            client_host="8.8.8.8",
        )
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        await self.mw.dispatch(req, call_next)
        call_next.assert_awaited_once()
        assert req.state.tenant_id == str(_TENANT)

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_invalid_bearer_token(self, mock_introspect):
        mock_introspect.return_value = {"active": False, "reason": "expired"}
        req = _make_request(
            path="/api/record/part",
            headers={"X-Tenant-Id": "Bearer bad-tok"},
            client_host="8.8.8.8",
        )
        call_next = AsyncMock()

        resp = await self.mw.dispatch(req, call_next)
        assert resp.status_code == 401
        call_next.assert_not_awaited()

    @patch("w2v_common.auth.middleware._introspect_license")
    async def test_expired_license(self, mock_introspect):
        mock_introspect.return_value = None
        req = _make_request(
            path="/api/record/part",
            headers={"X-Tenant-Id": "Bearer expired-tok"},
            client_host="8.8.8.8",
        )
        call_next = AsyncMock()

        resp = await self.mw.dispatch(req, call_next)
        assert resp.status_code == 401

    async def test_missing_token_untrusted_returns_401(self):
        req = _make_request(path="/api/record/part", client_host="8.8.8.8")
        call_next = AsyncMock(return_value=MagicMock())

        resp = await self.mw.dispatch(req, call_next)
        assert resp.status_code == 401
        call_next.assert_not_awaited()
