"""SSRF guard on the caller-supplied URL fields of the recording schemas.

`ipaddress.ip_address("100.64.0.5").is_private` is **False** — RFC 6598
CGNAT `100.64.0.0/10` is not RFC 1918. That /10 is this estate's Tailscale
tailnet (postgres 100.64.0.5, docker registry 100.64.0.6, k3s master
100.64.0.7), so a hand-rolled `is_private` check would have waved every one
of those through. These tests pin the real boundary: the deny is the **/10**,
not a `startswith("100.")`.

`webhook_url` is an outbound POST target, so it is guarded identically to
`url` — both validators are exercised here.

DNS: literal-IP URLs need no patching (`getaddrinfo` resolves them numerically,
no network). Hostname cases patch `socket.getaddrinfo`, which is what
`w2v_common.net.safe_fetch` calls.
"""

from __future__ import annotations

import socket

import pytest
from pydantic import ValidationError

from src.api.schemas.recording import (
    RecordUrlRequest,
    _validate_url_value,
    _validate_webhook_url_value,
)

_REAL_GETADDRINFO = socket.getaddrinfo


@pytest.fixture
def patch_dns(monkeypatch: pytest.MonkeyPatch):
    """Map hostname -> resolved IPv4 string; anything unmapped resolves for real.

    Delegating unmapped names to the real resolver keeps the literal-IP cases
    honest (they exercise the actual numeric parse) while making hostname cases
    hermetic.
    """

    def _apply(mapping: dict[str, str]) -> None:
        def _fake(host, port, *args, **kwargs):
            if host in mapping:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mapping[host], port or 0))]
            return _REAL_GETADDRINFO(host, port, *args, **kwargs)

        monkeypatch.setattr(socket, "getaddrinfo", _fake)

    return _apply


# ── url: blocked targets ──────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://100.64.0.1/x", id="cgnat-low-edge"),
        pytest.param("http://100.64.0.5/x", id="tailnet-postgres-primary"),
        pytest.param("http://100.64.0.6/x", id="tailnet-docker-registry"),
        pytest.param("http://[::ffff:100.64.0.5]/x", id="ipv4-mapped-ipv6-postgres"),
        pytest.param("http://169.254.169.254/latest/meta-data/", id="cloud-metadata"),
        pytest.param("http://127.0.0.1/x", id="loopback"),
        pytest.param("http://10.42.0.7/x", id="k8s-clusterip"),
    ],
)
def test_url_rejects_internal_targets(url: str) -> None:
    with pytest.raises(ValueError, match="internal or reserved network address"):
        _validate_url_value(url)


@pytest.mark.unit
def test_url_rejects_hostname_resolving_into_tailnet(patch_dns) -> None:
    """A public-looking name that resolves into the /10 is still rejected."""
    patch_dns({"postgres.internal": "100.64.0.5"})
    with pytest.raises(ValueError, match="internal or reserved network address"):
        _validate_url_value("https://postgres.internal/x")


@pytest.mark.unit
def test_url_fails_closed_on_dns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    with pytest.raises(ValueError, match="internal or reserved network address"):
        _validate_url_value("https://does-not-resolve.example/x")


# ── url: still accepted (the guard must not over-block) ───────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://8.8.8.8/x", id="public-literal"),
        # The /10 runs 100.64.0.0 - 100.127.255.255. These two straddle it and
        # prove the deny is the network, not a `startswith("100.")`.
        pytest.param("http://100.63.255.255/x", id="just-below-the-slash-10"),
        pytest.param("http://100.128.0.0/x", id="just-above-the-slash-10"),
    ],
)
def test_url_accepts_public_literals(url: str) -> None:
    assert _validate_url_value(url) == url


@pytest.mark.unit
def test_url_accepts_public_hostname(patch_dns) -> None:
    patch_dns({"example.com": "93.184.216.34"})
    assert _validate_url_value("https://example.com/v.mp4") == "https://example.com/v.mp4"


# ── url: pre-existing behaviour preserved ─────────────────────────────────


@pytest.mark.unit
def test_url_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="must start with"):
        _validate_url_value("ftp://example.com/v.mp4")


@pytest.mark.unit
def test_url_rejects_over_2048_chars() -> None:
    long_url = "https://example.com/" + ("a" * 2100)
    assert len(long_url) > 2048
    with pytest.raises(ValueError, match="must not exceed 2048 characters"):
        _validate_url_value(long_url)


# ── webhook_url: the outbound POST / exfiltration sink ────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://100.64.0.7/hook", id="tailnet-k3s-master"),
        pytest.param("http://100.64.0.1/hook", id="cgnat-low-edge"),
        pytest.param("http://[::ffff:100.64.0.5]/hook", id="ipv4-mapped-ipv6"),
        pytest.param("http://169.254.169.254/hook", id="cloud-metadata"),
    ],
)
def test_webhook_url_rejects_internal_targets(url: str) -> None:
    with pytest.raises(ValueError, match="internal or reserved network address"):
        _validate_webhook_url_value(url)


@pytest.mark.unit
def test_webhook_url_allows_none() -> None:
    assert _validate_webhook_url_value(None) is None


@pytest.mark.unit
def test_webhook_url_accepts_public(patch_dns) -> None:
    patch_dns({"hooks.example.com": "93.184.216.34"})
    url = "https://hooks.example.com/cb"
    assert _validate_webhook_url_value(url) == url


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://100.63.255.255/hook", id="just-below-the-slash-10"),
        pytest.param("http://100.128.0.0/hook", id="just-above-the-slash-10"),
    ],
)
def test_webhook_url_accepts_slash_10_boundary(url: str) -> None:
    assert _validate_webhook_url_value(url) == url


@pytest.mark.unit
def test_webhook_url_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="must start with"):
        _validate_webhook_url_value("ftp://example.com/hook")


@pytest.mark.unit
def test_webhook_url_rejects_over_2048_chars() -> None:
    long_url = "https://example.com/" + ("a" * 2100)
    with pytest.raises(ValueError, match="must not exceed 2048 characters"):
        _validate_webhook_url_value(long_url)


# ── model level: proves the wiring returns 422, not 500 ───────────────────


@pytest.mark.unit
def test_record_url_request_rejects_internal_url() -> None:
    """A blocked target must surface as a pydantic field error (=> HTTP 422)."""
    with pytest.raises(ValidationError) as exc_info:
        RecordUrlRequest(url="http://100.64.0.5/x", part_id="p1")
    assert "internal or reserved network address" in str(exc_info.value)
    assert ("url",) in [e["loc"] for e in exc_info.value.errors()]


@pytest.mark.unit
def test_record_url_request_rejects_internal_webhook_url(patch_dns) -> None:
    patch_dns({"example.com": "93.184.216.34"})
    with pytest.raises(ValidationError) as exc_info:
        RecordUrlRequest(
            url="https://example.com/v.mp4",
            part_id="p1",
            webhook_url="http://100.64.0.7/hook",
        )
    assert ("webhook_url",) in [e["loc"] for e in exc_info.value.errors()]


@pytest.mark.unit
def test_record_url_request_accepts_public_pair(patch_dns) -> None:
    patch_dns({"example.com": "93.184.216.34", "hooks.example.com": "93.184.216.34"})
    req = RecordUrlRequest(
        url="https://example.com/v.mp4",
        part_id="p1",
        webhook_url="https://hooks.example.com/cb",
    )
    assert req.url == "https://example.com/v.mp4"
    assert req.webhook_url == "https://hooks.example.com/cb"
