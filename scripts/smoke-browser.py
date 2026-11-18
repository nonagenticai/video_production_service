"""Fail the image build unless the production Playwright can capture a page."""

import tempfile
from importlib.metadata import version
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    """Exercise Chromium, screenshot capture, and the bundled video encoder."""
    with (
        tempfile.TemporaryDirectory(prefix="browser-smoke-") as output_dir,
        sync_playwright() as playwright,
    ):
        # Match the URL recorder's default launch, including the headless shell.
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 320, "height": 240},
            record_video_dir=output_dir,
        )
        page = context.new_page()
        page.set_content("<body style='background:#123456'>browser capture smoke</body>")
        screenshot = page.screenshot()
        if not screenshot.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("Chromium did not produce a PNG screenshot")
        video = page.video
        context.close()
        if video is None or Path(video.path()).stat().st_size == 0:
            raise RuntimeError("Chromium did not produce a video recording")
        browser.close()
    print(f"Playwright {version('playwright')}: Chromium screenshot and video capture passed")


if __name__ == "__main__":
    main()
