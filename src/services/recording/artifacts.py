"""What KIND of artifact a recorder handed back, and the integrity check for it.

``RecorderService._handle_success`` must verify the recorder's output before it
marks a job completed and meters the tenant for it. The verification is not one
check: it depends on what the artifact *is*.

Every recorder returns its output under the ``video_path`` key -- that key name
is historical and is now a lie for two of the modes. The per-screen capture
modes (:meth:`UrlRecorder.capture_screens` and
:meth:`UrlRecorder.capture_screen_clips`) return a **zip** of PNG stills / MP4
reveals there, reusing the same result shape so the success/storage path is
shared. ffprobe cannot read a zip, so probing one always reports
``probed=False``.

That is exactly why the artifact kind is declared EXPLICITLY by the producer
rather than sniffed at the consumer. "This is not a video and was never meant to
be probed" and "this is a video whose probe failed" are different facts, and
only the recorder knows which one it is holding. Inferring it from a failed
probe -- catching the probe failure and continuing -- would collapse the two
back together and re-open the hole that the probe veto exists to close: a
corrupt recording reported as a zero-second success and billed.

So: the producer states ``artifact_kind``; the consumer dispatches on it; and
EVERY kind gets a real integrity check. A zip is not exempt from verification,
it is verified as a zip.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: A real video container. Verified with ``FFmpegOps.probe_all``.
ARTIFACT_KIND_VIDEO = "video"

#: A zip of per-screen stills (``<id>.png``) and/or reveal clips (``<id>.mp4``),
#: written by ``UrlRecorder._zip_screens`` / ``._zip_clips``. Verified as an
#: archive, never with ffprobe.
ARTIFACT_KIND_SCREENS_ZIP = "screens_zip"

#: Entry suffixes that count as delivering a screen. ``<id>.track.json``
#: sidecars are deliberately NOT here: they are optional metadata and a zip that
#: contained only sidecars would have delivered no screen at all.
_SCREEN_MEMBER_SUFFIXES = (".png", ".mp4")


def artifact_kind_of(result: Mapping[str, Any]) -> str:
    """Return the artifact kind a recorder result declares.

    Defaults to :data:`ARTIFACT_KIND_VIDEO`, so a recorder that declares nothing
    keeps the strict ffprobe verification. Only a producer that explicitly says
    otherwise opts out of it.
    """
    declared = result.get("artifact_kind")
    if isinstance(declared, str) and declared:
        return declared
    return ARTIFACT_KIND_VIDEO


#: Media types for the video-container suffixes a recorder can store. Only
#: consulted for :data:`ARTIFACT_KIND_VIDEO`; a non-video kind never reaches it.
_VIDEO_MEDIA_TYPES = {"mp4": "video/mp4", "webm": "video/webm", "gif": "image/gif"}

#: The one media type a screens archive can have.
_SCREENS_ZIP_MEDIA_TYPE = "application/zip"


def artifact_media_type(kind: str | None, suffix: str | None) -> str:
    """Return the ``Content-Type`` for a stored artifact of ``kind``.

    Dispatches on the PRODUCER-declared kind first, and only falls through to the
    stored file suffix for a real video. That ordering is the fix: the download
    route used to consult the suffix table alone, which has no ``"zip"`` key, so
    a screens archive -- stored with its ``.zip`` suffix intact by
    ``store_video`` -- was served as ``video/webm``. A client asking for a
    download got a video content type wrapped around a zip.

    ``kind`` of ``None`` resolves to :data:`ARTIFACT_KIND_VIDEO`, matching
    :func:`artifact_kind_of`: a producer that declares nothing, and every row
    written before ``artifact_kind`` was persisted, keeps exactly the previous
    behaviour. The default stays the strict/video case in both directions.

    Args:
        kind: The persisted ``artifact_kind``, or ``None`` for an undeclared row.
        suffix: The stored file's extension WITHOUT the dot (e.g. ``"mp4"``), or
            ``None``. Unknown suffixes keep the historical ``video/webm``
            fallback.
    """
    if kind == ARTIFACT_KIND_SCREENS_ZIP:
        return _SCREENS_ZIP_MEDIA_TYPE
    return _VIDEO_MEDIA_TYPES.get(suffix or "", "video/webm")


def screens_zip_defect(path: Path, screen_ids: Iterable[str] | None = None) -> str | None:
    """Return why ``path`` is not a usable screens archive, or ``None`` if it is.

    The zip equivalent of the ffprobe veto -- an archive that is empty, is not a
    zip at all, has a CRC-corrupt entry, or does not actually contain the
    screens the recorder claims it captured, is a FAILED capture and must not be
    completed or billed.

    Args:
        path: The archive on disk (the caller has already checked it exists).
        screen_ids: The ``screen_ids`` the recorder reported. Every one of them
            must be backed by a non-empty ``<id>.png`` or ``<id>.mp4`` member.
    """
    try:
        if path.stat().st_size == 0:
            return "the archive is empty (0 bytes)"
        if not zipfile.is_zipfile(path):
            return "the file is not a valid zip archive"
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if not infos:
                return "the archive contains no entries"
            corrupt = zf.testzip()
            if corrupt is not None:
                return f"the archive has a corrupt entry ({corrupt!r} failed its CRC)"
            sizes = {info.filename: info.file_size for info in infos}
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        return f"the archive could not be read ({exc})"

    if not any(size > 0 for size in sizes.values()):
        return "every entry in the archive is empty"

    missing = [
        screen_id
        for screen_id in (screen_ids or ())
        if not any(sizes.get(f"{screen_id}{suffix}", 0) > 0 for suffix in _SCREEN_MEMBER_SUFFIXES)
    ]
    if missing:
        return (
            f"the archive is missing a non-empty .png/.mp4 entry for "
            f"{len(missing)} reported screen(s): {missing!r}"
        )
    return None
