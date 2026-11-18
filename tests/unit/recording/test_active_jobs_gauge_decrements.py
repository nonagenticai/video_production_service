"""``active_recording_jobs`` must be re-published on RELEASE, not only on acquire.

``set_active_jobs`` was called on the acquire side for all five producers, but on
the release side only in ``_attempt_recording``. The content / animation / url /
screens paths released the slot and left the gauge pinned at its high-water mark,
so an idle pod reported saturation that did not exist.

That is an observability FAIL-OPEN -- the reading is wrong in the direction that
invents a problem -- and it plausibly corroborated a false "capacity" theory
during the 2026-09-03 outage investigation, when the real cause was a leaked
fleet slot.

Asserted as a pairing invariant over the source rather than by driving all five
Chromium/ffmpeg recording paths: the defect was a MISSING line at four of five
call sites, so the property worth pinning is "every release is followed by a
re-publish", which catches a regression at any site including ones added later.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path("src/services/recording/recorder_service.py")
_RELEASE = re.compile(r"await self\._concurrency\.release\(")
_PUBLISH = re.compile(r"set_active_jobs\(self\._concurrency\.active_count\)")


def _release_sites() -> list[tuple[int, str]]:
    lines = _SRC.read_text().splitlines()
    return [(i, ln) for i, ln in enumerate(lines) if _RELEASE.search(ln)]


class TestGaugePairing:
    def test_the_premise_holds(self):
        """If the release sites ever stop matching, the checks below are vacuous."""
        sites = _release_sites()
        assert len(sites) == 5, f"expected 5 concurrency release sites, found {len(sites)}"

    def test_every_release_republishes_the_gauge(self):
        lines = _SRC.read_text().splitlines()
        missing: list[int] = []
        for idx, _ in _release_sites():
            # The re-publish must be the next STATEMENT, inside the same block --
            # comments and blank lines in between are fine, another statement is
            # not. (A raw next-two-lines window is what a comment silences.)
            nxt = ""
            for ln in lines[idx + 1 : idx + 15]:
                stripped = ln.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                nxt = ln
                break
            if not _PUBLISH.search(nxt):
                missing.append(idx + 1)
        assert not missing, (
            f"concurrency slots are released without re-publishing "
            f"active_recording_jobs at line(s) {missing}; the gauge stays pinned at "
            f"its peak on an idle pod and reports saturation that does not exist"
        )

    def test_acquire_still_publishes(self):
        """The other half of the pair, so a 'fix' that deleted both would fail."""
        src = _SRC.read_text()
        assert _PUBLISH.search(src)
        assert "await self._concurrency.acquire(job_id)" in src
