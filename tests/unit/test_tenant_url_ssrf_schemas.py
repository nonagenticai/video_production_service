"""Admission-layer SSRF proof for the tenant-supplied URL fields.

Why these assertions exist
--------------------------
``ipaddress.ip_address("100.64.0.5").is_private`` is **False**. CPython
classifies RFC 6598 shared address space (``100.64.0.0/10``) as neither private
nor reserved, and that range is this fleet's entire Tailscale tailnet — the
bare-metal prod Postgres primary (``100.64.0.5``), the Docker registry
(``100.64.0.6``) and the k3s control plane (``100.64.0.7``). Any guard
built from an RFC 1918 list, or from ``not is_private``, waves all of them
through.

PR #106 closed this for ``RecordingRequest.url`` and its ``webhook_url``. Five
sibling fields were missed, and the worst of them was in the *same file* one
class down:

* ``UrlAuthInfo.login_url`` — the recorder does ``page.goto(login_url)`` and
  then ``page.fill(...)``s the tenant's username and password into that page.
  Unguarded, it does not merely *fetch* an internal address, it **posts the
  tenant's credentials** to whatever host the request names, and the session
  cookies persist into the recorded video. This is the highest-severity field
  on the model, and it had no test at all until this file.
* ``CompileRequest.webhook_url`` — the compile schema imported no validator
  whatsoever, so the job payload was POSTable anywhere.
* ``RecordingInput.video_path``, ``ShotListPartInput.image_url`` /
  ``slide_image_url`` / ``background_image`` — overloaded fields that the
  compiler downloads when they are http(s).

Every test here asserts the *range* is the discriminator, not the string prefix:
``100.63.255.255`` and ``100.128.0.0`` bracket the ``/10`` and must stay
ALLOWED, so a lazy ``startswith("100.")`` implementation fails these cases.

DNS is stubbed. A real lookup under this repo's xdist run is flaky, and
``--reruns`` would paper over it.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from w2v_common.net import safe_fetch

from src.api.schemas.compilation import (
    CompileRequest,
    RecordingInput,
    ShotListInput,
    ShotListPartInput,
)
from src.api.schemas.recording import UrlAuthInfo

# Live fleet addresses, all inside 100.64.0.0/10, plus the classics.
BLOCKED_TARGETS: tuple[str, ...] = (
    "http://100.64.0.5:5432/",  # bare-metal prod Postgres primary
    "http://100.64.0.6:5000/v2/_catalog",  # Docker registry
    "http://100.64.0.7:6443/",  # k3s control plane
    "http://[::ffff:100.64.0.5]/",  # the same address, smuggled as v4-in-v6
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    "http://127.0.0.1:8080/",  # loopback
    "http://10.0.0.5/",  # RFC 1918
)

# Must stay reachable. The two /10 neighbours are the load-bearing cases: they
# prove the deny is the range and not ``startswith("100.")``.
ALLOWED_TARGETS: tuple[str, ...] = (
    "https://100.63.255.255/",  # one below the /10
    "https://100.128.0.0/",  # one above the /10
    "https://8.8.8.8/",
    "https://tenant-site.example/login",
)

_PUBLIC_IP = "93.184.216.34"


def _shot_list() -> ShotListInput:
    """Minimal valid shot list — this file is about URL fields, nothing else."""
    return ShotListInput(parts=[ShotListPartInput(id="p1", part_type="title_card")])


class _DnsStub:
    """Stand-in for the ``socket`` module as ``safe_fetch`` uses it.

    The guard is patched by REBINDING ``safe_fetch.socket`` to this object,
    never by ``patch("...safe_fetch.socket.getaddrinfo")``. That spelling
    reaches through to the *global* ``socket`` module — ``safe_fetch.socket``
    is the real module object — so it also hijacks psycopg's resolution and
    hangs every DB-backed test in the process. Rebinding the name touches only
    the guard.
    """

    AF_INET = socket.AF_INET
    AF_INET6 = socket.AF_INET6
    SOCK_STREAM = socket.SOCK_STREAM
    gaierror = socket.gaierror

    @staticmethod
    def getaddrinfo(host: str, *_args: Any, **_kwargs: Any) -> list[Any]:
        """Resolve names to a public address; resolve IP literals to themselves.

        Returning the public address for literals too would let every blocked
        case pass the guard, and the suite would be asserting this stub rather
        than the production rules.
        """
        try:
            literal = ipaddress.ip_address(str(host).strip("[]"))
        except ValueError:
            literal = None
        if literal is not None:
            family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
            return [(family, socket.SOCK_STREAM, 6, "", (str(literal), 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_PUBLIC_IP, 0))]


@pytest.fixture(autouse=True)
def _stub_dns() -> Iterator[None]:
    """Make the SSRF guard's DNS deterministic without touching global sockets."""
    with patch.object(safe_fetch, "socket", _DnsStub):
        yield


class TestLoginUrlIsGuarded:
    """``UrlAuthInfo.login_url`` — the credential sink."""

    @pytest.mark.parametrize("target", BLOCKED_TARGETS)
    def test_internal_login_target_is_rejected(self, target: str) -> None:
        """The tenant's credentials must never be typed into an estate host."""
        with pytest.raises(ValidationError):
            UrlAuthInfo(type="basic", login_url=target, username="u", password="p")

    @pytest.mark.parametrize("target", ALLOWED_TARGETS)
    def test_public_login_target_is_accepted(self, target: str) -> None:
        """A genuine external login page still works."""
        auth = UrlAuthInfo(type="basic", login_url=target, username="u", password="p")
        assert auth.login_url == target

    def test_none_login_url_is_allowed(self) -> None:
        """Interactive login is opt-in; unset must stay legal."""
        assert UrlAuthInfo(type="none", login_url=None).login_url is None

    def test_non_http_scheme_is_rejected(self) -> None:
        """``file://`` would make the browser read the pod's filesystem."""
        with pytest.raises(ValidationError):
            UrlAuthInfo(type="basic", login_url="file:///etc/passwd", username="u", password="p")


class TestCompileWebhookIsGuarded:
    """``CompileRequest.webhook_url`` — an exfil sink as well as an SSRF one."""

    @pytest.mark.parametrize("target", BLOCKED_TARGETS)
    def test_internal_webhook_is_rejected(self, target: str) -> None:
        with pytest.raises(ValidationError):
            CompileRequest(shot_list=_shot_list(), recordings=[], webhook_url=target)

    def test_public_webhook_is_accepted(self) -> None:
        req = CompileRequest(
            shot_list=_shot_list(), recordings=[], webhook_url="https://hooks.example/cb"
        )
        assert req.webhook_url == "https://hooks.example/cb"

    def test_no_webhook_is_allowed(self) -> None:
        assert CompileRequest(shot_list=_shot_list(), recordings=[]).webhook_url is None


class TestOverloadedAssetFieldsAreGuarded:
    """Fields that may be a local path OR a URL: only the URL branch is guarded."""

    @pytest.mark.parametrize("target", BLOCKED_TARGETS)
    def test_internal_video_path_url_is_rejected(self, target: str) -> None:
        with pytest.raises(ValidationError):
            RecordingInput(part_id="r1", video_path=target)

    @pytest.mark.parametrize(
        "local_value",
        [
            "/var/lib/recordings/r1.mp4",
            "s3://bucket/key.mp4",
            "recordings/r1.mp4",
        ],
    )
    def test_local_and_s3_paths_pass_through_untouched(self, local_value: str) -> None:
        """The common case has no outbound-request surface and must not break."""
        assert RecordingInput(part_id="r1", video_path=local_value).video_path == local_value

    @pytest.mark.parametrize("field", ["image_url", "slide_image_url", "background_image"])
    def test_internal_asset_url_is_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            ShotListPartInput.model_validate(
                {"id": "p1", "part_type": "title_card", field: BLOCKED_TARGETS[0]}
            )

    @pytest.mark.parametrize("field", ["image_url", "slide_image_url", "background_image"])
    def test_data_uri_asset_passes_through(self, field: str) -> None:
        """``data:`` refs are materialised locally — no request, no guard."""
        data_uri = "data:image/png;base64,iVBORw0KGgo="
        part = ShotListPartInput.model_validate(
            {"id": "p1", "part_type": "title_card", field: data_uri}
        )
        assert getattr(part, field) == data_uri
