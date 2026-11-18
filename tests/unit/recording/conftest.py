"""Unit test fixtures."""

from __future__ import annotations

import os
import shutil

import pytest


@pytest.fixture
def require_ffmpeg() -> None:
    """Gate for the real-ffmpeg tests: skip locally, FAIL under CI.

    A skip is not a pass. These tests are the only ones that touch the binary the
    product ships with (the Dockerfile installs ffmpeg), and they sat silently
    skipped in CI because the test job had none. The wrapper now asks the
    archetype for it (``extra_apt: ffmpeg`` in ``.github/workflows/ci-cd.yaml``);
    this fixture makes losing that input a red build instead of a quiet skip.
    """
    if shutil.which("ffmpeg") is not None:
        return
    if os.environ.get("CI"):
        pytest.fail(
            "ffmpeg is not on PATH but CI is set: the real-ffmpeg tests must run in CI. "
            "Restore `extra_apt: ffmpeg` in .github/workflows/ci-cd.yaml.",
            pytrace=False,
        )
    pytest.skip("ffmpeg not on PATH")
