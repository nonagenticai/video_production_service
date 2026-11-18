"""Shared SSRF guard helpers for tenant-supplied URL fields.

WHY this module exists: several tenant-controlled fields in this service are
either fetched (``image_url``, ``slide_image_url``, ``video_path``,
``narration_audio_dir``) or POSTed to (``webhook_url``), and two separate
schema modules plus three service-layer sinks need the *same* boundary. Copying
the check into each of them is how five of them ended up missing it in the first
place, so the single definition lives here.

The boundary itself is ``w2v_common.net.validate_public_url``: it resolves the
hostname and rejects private / reserved / loopback / link-local / multicast
targets — critically including RFC 6598 CGNAT ``100.64.0.0/10``, which is this
estate's Tailscale tailnet (prod postgres ``100.64.0.5``, the docker registry
``100.64.0.6``, the k3s control plane ``100.64.0.7``) and which
``ipaddress.ip_address(...).is_private`` reports as **False**. A hand-rolled
``is_private`` check would wave every one of those through.

Two-layer defence, both required:

* **Schema layer** (:func:`reject_internal_target`,
  :func:`validate_optional_remote_url`) refuses the value at the pydantic
  boundary so a job row is never created for an internal target — the API
  answers 422, not 500, because ``SSRFError`` subclasses ``ValueError`` and
  pydantic converts ``ValueError`` in a validator into a field error.
* **Sink layer** (:func:`artifact_allow_hosts` + ``w2v_common.net.safe_get`` /
  ``safe_get_sync`` / ``safe_post``) is *not* redundant: those calls follow
  redirects, so a schema-approved public host can 302 straight into the tailnet.
  The safe_* helpers re-validate every hop. They also defend a poisoned/loosened
  DB row being replayed long after the request was accepted.

The LFI half of the same problem
--------------------------------
Those same overloaded fields also accept ``file://`` and bare on-disk paths, and
the compiler reads them off the pod and composites them into a **downloadable**
video. That is a local-file-inclusion primitive on exactly the fields the SSRF
guard above covers, so it lives here rather than in a third module.

The boundary is not "reject every local path" — this service *publishes* local
paths. ``GET /api/record/jobs/{id}/result`` returns
``{STORAGE_LOCAL_PATH}/{job_id}/{part_id}.mp4`` (``storage_service`` local
backend), prod runs ``STORAGE_BACKEND=local`` with ``/app/data`` on a PVC shared
by both replicas, and re-submitting that path on ``recordings[].video_path`` is
the documented record → compile round trip. Blanket-denying bare paths would
break it.

The boundary is therefore **confinement**: with the gate closed, a local
reference is accepted only when it resolves inside the configured storage root —
precisely the set of files this service produced itself. ``/etc/passwd``,
``/app/.env`` and ``/var/run/secrets/...`` are refused, the round trip is
untouched. ``.resolve()`` runs before the check, so a symlink planted inside the
root that points outside it is refused too.

``compilation.allow_local_file_inputs`` (default **False in code**, not merely in
gitops — "only the gitops half landed" is how a sibling repo's identical guard
sat disarmed in prod) opens the gate for dev and for tests whose fixtures live in
a tmpdir outside the storage root.

The rest of the local-path surface
----------------------------------
The same confinement — :func:`resolve_local_asset_path`, unchanged — also decides
the six remaining tenant-supplied on-disk references that reached ``Path()`` with
no validator at all: ``watermark.image_path``, ``subtitles.file_path``,
``clips_dir``, ``shot_list.parts[].logo_path`` and the local branch of
``narration_audio_dir``. There is deliberately **one** confinement mechanism, not
two: a second one is how the first set drifted out of sync.

Two of these do NOT get the storage root, for reasons that are load-bearing:

* ``theme.font_path`` / ``theme.font_bold_path`` are **not** tenant inputs at the
  sink. ``title_card_generator._get_font`` reads ``settings.theme.font_*`` — the
  operator's ``THEME_FONT_*`` env — and never the request's theme dict. Fonts
  legitimately point at files baked into the image
  (``/usr/share/fonts/truetype/brand``), which is *outside* the storage root, so
  confining them there would break rendering. See
  :func:`validate_local_asset_ref`'s docstring for what the schema layer does
  with the request-side spelling of those fields.
* ``narration_audio_dir``'s http(s) branch downloads into a directory this
  service creates under the job work dir. That path is **internally produced**
  and must not be pushed through the tenant guard — only the ``else`` branch,
  where the value is the raw tenant string, is confined.

Path *components*
-----------------
``part_id`` is the other half of the same primitive: it is a tenant string that
is f-string-joined onto directories the compiler owns (``{clips_dir}/{part_id}
.mp4``, ``{normalized_dir}/{part_id}.mp4``, ``{edited_dir}/{part_id}_trim.mp4``).
Confining the *directory* is worth nothing if the component can climb back out,
so :func:`validate_path_component` refuses separators and traversal at admission
— the same ``^[a-zA-Z0-9_\\-\\.]+$`` shape the recording schemas have enforced on
``part_id`` since they were written.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from w2v_common.net import SSRFError, validate_public_url

_MAX_URL_LEN = 2048
_MAX_PATH_LEN = 4096
_FILE_SCHEME = "file://"

# Mirrors ``src/api/schemas/recording._PART_ID_RE`` — one shape for a path
# component across both domains, so a record → compile hand-off cannot produce
# an id that one side accepts and the other rejects.
_PATH_COMPONENT_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


# --------------------------------------------------------------------------
# Per-site response-body ceilings for the allow-listed artifact fetches.
#
# w2v_common 0.55.0 gave `safe_get`/`safe_get_sync` a 64 MiB DEFAULT
# `max_fetch_bytes` (previously `None` == unbounded). That default is NOT safe
# to inherit here. Every artifact fetch in this service is deliberately
# fail-soft — it sits in an `except (... ValueError ...)` handler that logs and
# skips — and `ResponseTooLargeError` subclasses `ValueError`. So an artifact
# over the cap does not raise; it SILENTLY becomes a missing segment, a missing
# background, or a compilation with no narration. The failure is invisible in
# the output and only a warning line in the log.
#
# The ceilings are therefore stated per site, sized to what that site actually
# fetches, rather than taken from one shared knob: a source recording is not a
# thumbnail. All four are still bounded — the point is to bound them HONESTLY,
# not to opt out. The bound is also a memory bound: each site buffers the whole
# body via `resp.content` before writing it, in a pod with `requests.memory=2Gi`
# / `limits.memory=8Gi` that also runs ffmpeg.

#: Title-card and visual-asset image refs (`_materialize_image_ref`, and the
#: title-card `background_image`). These are stills handed to Pillow. 64 MiB is
#: already far above any real slide, screenshot or logo — a 4K PNG lands in the
#: low tens of MiB — so this matches the upstream default. It is stated anyway
#: so the value is pinned at the site: if w2v_common ever retunes its default,
#: an image ref must not silently start or stop fitting.
ARTIFACT_IMAGE_MAX_BYTES = 64 * 1024 * 1024

#: The full narration track (`narration_audio_dir` pointing at a single
#: `narration_full.mp3`). This one is genuinely mis-served by the 64 MiB
#: default: at a typical 128 kbps mp3, 64 MiB is only ~70 minutes of audio, and
#: a long-form course narration exceeds that. Losing it is the quiet case — the
#: handler sets `narration_audio_dir = None` and the compile proceeds SILENT.
#: 256 MiB is ~4.5 hours at the same bitrate, past any single-job narration.
ARTIFACT_AUDIO_MAX_BYTES = 256 * 1024 * 1024

#: The source screen recording (`_resolve_recording_video_path`). This is the
#: one genuinely unbounded source in the service: the record → compile round
#: trip re-submits whatever the recorder produced, and recording length is
#: driven by the tenant's content, not by any setting here. A 1080p screencast
#: at ~8 Mbps passes 64 MiB in about a minute, so the inherited default would
#: have skipped essentially every real recording. 1 GiB is the largest ceiling
#: that still leaves working headroom: the body is buffered whole by
#: `resp.content` inside the 8Gi pod limit alongside ffmpeg, and parts are
#: resolved sequentially, so ~1 GiB resident is recoverable where an unbounded
#: read is not. Oversize now surfaces as a logged, bounded skip instead of an
#: OOMKill that takes the whole compilation with it.
ARTIFACT_RECORDING_MAX_BYTES = 1024 * 1024 * 1024


def artifact_allow_hosts() -> tuple[str, ...]:
    """Return the artifact hosts that may legitimately resolve to an internal address.

    Presigned MinIO URLs point at the in-cluster object store, which lives in a
    blocked range by design. Blocking them would break every legitimate
    compilation, so the configured MinIO endpoints (internal and, when set, the
    external rewrite target) are trusted by name. This is an *allow-list of
    hosts*, not a bypass: ``validate_public_url`` still enforces the http(s)
    scheme, and a trusted host that redirects to a non-allowed internal address
    is still rejected on the next hop.

    Returns:
        Host or ``host:port`` keys accepted by ``allow_hosts``. Empty when the
        service runs on the ``local`` storage backend.
    """
    from src.config import get_settings

    storage = get_settings().storage
    hosts: list[str] = []
    for raw in (storage.minio_endpoint, storage.minio_external_endpoint):
        value = (raw or "").strip()
        if not value:
            continue
        # Accept both "host:port" and "scheme://host:port" spellings.
        netloc = urlparse(value).netloc if "//" in value else value
        netloc = netloc.split("@")[-1].strip().lower()
        if netloc:
            hosts.append(netloc)
            hostname = netloc.rsplit(":", 1)[0]
            if hostname and hostname != netloc:
                hosts.append(hostname)
    return tuple(dict.fromkeys(hosts))


def is_remote_url(value: str) -> bool:
    """Return True when ``value`` is an http(s) URL rather than a local/S3 path."""
    return value.startswith(("http://", "https://"))


def local_file_inputs_allowed() -> bool:
    """Whether ``file://`` / bare-path references may be read off local disk freely.

    Defaults False (see the module docstring). Read through ``get_settings`` at
    call time rather than captured at import, so a test can open the gate for
    one case without leaking it into the next.
    """
    from src.config import get_settings

    return bool(get_settings().compilation.allow_local_file_inputs)


def _storage_root() -> Path | None:
    """The configured local storage root, resolved, or ``None`` if unusable.

    ``None`` makes :func:`resolve_local_asset_path` fail **closed**: an
    unreadable storage config must refuse local refs, never wave them through.
    """
    from src.config import get_settings

    try:
        return Path(str(get_settings().storage.local_path)).resolve()
    except OSError, RuntimeError, ValueError, TypeError:
        return None


def is_local_file_ref(value: str) -> bool:
    """Return True when ``value`` names a local file rather than a remote object.

    ``data:`` payloads are materialised in-process and ``s3://`` goes through the
    storage client, so neither touches the filesystem by name. Everything else
    that is not http(s) — ``file://`` and every bare path — does.
    """
    return not value.startswith(("http://", "https://", "s3://", "data:"))


def resolve_local_asset_path(ref: str) -> Path | None:
    """Resolve a ``file://`` / bare-path asset reference, or refuse it.

    This is the **sink-layer** LFI guard. With the gate closed the reference is
    accepted only when it resolves inside the configured storage root — the set
    of files this service itself wrote and published — so the record → compile
    round trip keeps working while ``file:///etc/passwd`` does not.

    Args:
        ref: The raw reference, with or without a ``file://`` prefix.

    Returns:
        The resolved path when the reference is permitted, else ``None``. The
        caller treats ``None`` exactly like "asset not available" and skips the
        segment, which keeps both sinks fail-soft: one hostile part must not
        abort a whole compilation.

    Known residual (accepted, not closed here): this resolves and returns a
    path, and the sink opens it afterwards, so a symlink swapped into the root
    between the two wins. It is not theoretical — ``/app/data`` is a
    ReadWriteMany PVC mounted by both replicas — but closing it needs the whole
    read side moved onto an ``O_NOFOLLOW`` open + ``fstat`` device/inode
    re-check, and the two biggest consumers (ffmpeg and Pillow) take a *path*,
    not a file descriptor, so it cannot be closed without re-plumbing them. It
    is a strictly smaller window than the unconfined read this replaces.
    """
    raw = ref[len(_FILE_SCHEME) :] if ref.startswith(_FILE_SCHEME) else ref
    if not raw:
        return None
    if local_file_inputs_allowed():
        return Path(raw)
    root = _storage_root()
    if root is None:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        # A relative ref is a storage key (the minio backend's reference shape),
        # so it is resolved against the root rather than the process CWD.
        candidate = root / candidate
    try:
        # ``resolve`` before comparing: a symlink planted inside the root that
        # points outside it must be refused, and ``..`` must not escape.
        resolved = candidate.resolve()
    except OSError, RuntimeError, ValueError:
        return None
    if resolved == root or root in resolved.parents:
        return resolved
    return None


def reject_local_file_uri(value: str, *, field: str) -> None:
    """Refuse an explicit ``file://`` reference at the pydantic boundary.

    Only the ``file://`` *spelling* is refused here, never a bare path. Nothing
    in this service ever emits a ``file://`` reference — the recording API
    publishes bare absolute paths — so there is no legitimate producer to break,
    and rejecting it at admission turns an LFI attempt into a 422 before a job
    row exists. Bare paths are decided at the sink by
    :func:`resolve_local_asset_path`, which needs the storage root to judge them.

    Raises:
        ValueError: The value is a ``file://`` URI and the gate is closed.
    """
    if not value.startswith(_FILE_SCHEME):
        return
    if local_file_inputs_allowed():
        return
    msg = (
        f"{field} must not be a local file:// reference; "
        "supply an http(s), s3:// or data: reference instead"
    )
    raise ValueError(msg)


def validate_local_asset_ref(v: object, *, field: str) -> object:
    """Admission guard for a field that names a file/directory on the pod.

    ``watermark.image_path``, ``subtitles.file_path``, ``clips_dir``,
    ``logo_path`` and ``theme.background_image`` are all "a path, or occasionally
    a URL" in the same way the fields above are, so they get the same admission
    treatment: ``file://`` is refused outright (nothing in this service ever
    *emits* a ``file://`` reference, so there is no producer to break), an
    http(s) value is SSRF-checked, and a bare path is handed on for the sink to
    decide against the storage root.

    ``theme.font_path`` / ``theme.font_bold_path`` are validated here too even
    though the font loader ignores them (it reads ``settings.theme.font_*``).
    That is on purpose: the fields exist on the public request model, a reader
    will reasonably assume they work, and the day someone wires them up the
    guard must already be on the field rather than be remembered. The bare-path
    branch stays permissive precisely because the *correct* root for a font is
    the image's bundled font dir, not the storage root — so this layer refuses
    only the shapes that are wrong under every root.

    Runs in ``mode="before"``; non-strings pass through to pydantic's own type
    check so a wrong-typed field stays a 422 rather than becoming a 500.

    Args:
        v: Candidate value; ``None``, non-strings and bare paths pass through.
        field: Field name for the error message.

    Returns:
        The value unchanged when it is acceptable.

    Raises:
        ValueError: Over-long, a blocked ``file://`` URI, or an http(s) URL
            targeting an internal / reserved address.
    """
    if not isinstance(v, str):
        return v
    if len(v) > _MAX_PATH_LEN:
        msg = f"{field} must not exceed {_MAX_PATH_LEN} characters"
        raise ValueError(msg)
    return validate_optional_remote_url(v, field=field)


def validate_path_component(v: object, *, field: str) -> object:
    """Refuse a tenant string that would escape the directory it is joined to.

    ``part_id`` is interpolated straight into filenames under directories the
    compiler owns — ``{clips_dir}/{part_id}.mp4`` and
    ``{narration_audio_dir}/{part_id}.mp3`` are reads, but
    ``{normalized_dir}/{part_id}.mp4``, ``{title_cards_dir}/{part_id}.mp4`` and
    ``{edited_dir}/{part_id}_trim.mp4`` are **writes**. A ``part_id`` of
    ``../../../app/src/x`` therefore both escapes the confinement applied to
    those directories and gives a write primitive, so the component is
    constrained rather than the join sites being individually re-guarded.

    The accepted shape is the one the recording schemas already enforce, so
    every id produced by ``POST /api/record/*`` — the ids a compile request
    actually carries — passes unchanged.

    Args:
        v: Candidate value; non-strings pass through to pydantic's type check.
        field: Field name for the error message.

    Returns:
        The value unchanged when it is acceptable.

    Raises:
        ValueError: Empty, over-long, traversal, or containing a character that
            is not ``[A-Za-z0-9_-.]`` (which is what excludes ``/`` and ``\\``).
    """
    if not isinstance(v, str):
        return v
    if not 1 <= len(v) <= 128:
        msg = f"{field} must be between 1 and 128 characters"
        raise ValueError(msg)
    if ".." in v or v.strip(".") == "":
        msg = f"{field} must not contain '..' or be all dots"
        raise ValueError(msg)
    if not _PATH_COMPONENT_RE.match(v):
        msg = (
            f"{field} contains invalid characters: {v!r} "
            r"(must match ^[a-zA-Z0-9_\-\.]+$ — it is used as a filename)"
        )
        raise ValueError(msg)
    return v


def reject_internal_target(value: str, *, field: str) -> None:
    """Run the shared fleet SSRF guard, re-raised as a pydantic-friendly ``ValueError``.

    Args:
        value: The candidate URL.
        field: Field name, used only to build a readable error message.

    Raises:
        ValueError: Always, when the target is blocked or malformed. Both
            branches raise a plain ``ValueError`` because that is what pydantic
            converts into a field error — i.e. the API answers **422**, not a
            500. ``SSRFError`` subclasses ``ValueError`` and so is caught first.
    """
    try:
        validate_public_url(value, allow_hosts=artifact_allow_hosts())
    except SSRFError as exc:
        msg = f"{field} must not target an internal or reserved network address ({exc})"
        raise ValueError(msg) from exc
    except ValueError as exc:
        msg = f"{field} is not a valid public HTTP(S) URL ({exc})"
        raise ValueError(msg) from exc


def validate_webhook_url(v: object, *, field: str = "webhook_url") -> object:
    """Validate an outbound-POST webhook target.

    A webhook URL is both an SSRF sink and a data-exfiltration sink: an
    unguarded value lets a tenant make this service POST the whole job payload
    at any tailnet or cluster-internal address. Unlike the "path or URL" fields
    below, a webhook is *always* a URL, so a non-http(s) value is an error
    rather than a pass-through.

    Runs in ``mode="before"``; a non-string value is handed back so pydantic's
    own type check produces the 422 rather than this helper raising on ``len``.

    Args:
        v: Candidate value; ``None`` is allowed (webhooks are optional).
        field: Field name for the error message.

    Returns:
        The value unchanged when it is acceptable.

    Raises:
        ValueError: Over-long, non-http(s), or internal-network target.
    """
    if not isinstance(v, str):
        return v
    if len(v) > _MAX_URL_LEN:
        msg = f"{field} must not exceed {_MAX_URL_LEN} characters"
        raise ValueError(msg)
    if not is_remote_url(v):
        msg = f"{field} must start with 'http://' or 'https://'"
        raise ValueError(msg)
    reject_internal_target(v, field=field)
    return v


def validate_optional_remote_url(v: object, *, field: str) -> object:
    """Guard a field that accepts EITHER a local/S3 path OR a remote URL.

    ``video_path``, ``narration_audio_dir``, ``image_url`` and
    ``slide_image_url`` are all overloaded: callers legitimately pass a local
    filesystem path, an ``s3://`` URI, or a ``data:`` URI, and the compiler
    dispatches on the prefix. Only the http(s) branch reaches an outbound
    request, so only that branch is validated — a plain path is returned
    untouched. Rejecting non-URL values here would break every existing
    local-path caller, and there is no SSRF surface to protect in that branch.

    Runs in ``mode="before"``, so ``v`` is whatever the caller sent. Non-string
    values pass straight through to pydantic's own type check rather than
    blowing up here — that keeps a wrong-typed field a 422, not a 500.

    The non-http(s) branch is not a pass-through *entirely*: an explicit
    ``file://`` reference is refused here (see :func:`reject_local_file_uri`).
    Bare paths still pass, because only the sink knows the storage root that
    decides them.

    Args:
        v: Candidate value; ``None``, non-strings and bare paths pass through.
        field: Field name for the error message.

    Returns:
        The value unchanged.

    Raises:
        ValueError: The value is an http(s) URL that is over-long or targets an
            internal / reserved address, or it is a blocked ``file://`` URI.
    """
    if not isinstance(v, str):
        return v
    if not is_remote_url(v):
        reject_local_file_uri(v, field=field)
        return v
    if len(v) > _MAX_URL_LEN:
        msg = f"{field} must not exceed {_MAX_URL_LEN} characters"
        raise ValueError(msg)
    reject_internal_target(v, field=field)
    return v
