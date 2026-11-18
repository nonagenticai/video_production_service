"""Page grounding travels with the screenshot, using the same authenticated DOM."""

from __future__ import annotations

import json
import zipfile

from src.services.recording.url_recorder import CapturedScreens, UrlRecorder


def test_still_and_clip_archives_carry_the_same_page_sidecar(tmp_path):
    shots = CapturedScreens()
    shots.append(("screen-1", b"png"))
    shots.page_contents["screen-1"] = {
        "text": "Rendered text",
        "html": "<main>Rendered text</main>",
    }
    for archive in (
        UrlRecorder._zip_screens(shots, tmp_path),
        UrlRecorder._zip_clips([], shots, {}, tmp_path),
    ):
        with zipfile.ZipFile(archive) as bundle:
            assert bundle.read("screen-1.png") == b"png"
            assert json.loads(bundle.read("screen-1.page.json"))["text"] == "Rendered text"


async def test_rendered_content_bundle_excludes_credentials_in_real_chromium(tmp_path):
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        try:
            # Synthetic values verify that captured markup excludes credentials.
            await page.set_content(
                """<main><section><h1>Authenticated dashboard</h1>
                <p>Report a bottleneck</p><input type='password' value='private-credential'>
                """
                "<script>const secret = 'private-script-value';</script>"  # pragma: allowlist secret
                """
                <div hidden>hidden-data</div><a href='/?token=private-token' onclick='alert(1)'>Solutions</a>
                </section></main>"""
            )
            content = await UrlRecorder._page_content(page)
            assert "Authenticated dashboard" in content["text"]
            assert "private-" not in content["html"]
            assert "hidden-data" not in content["html"]
            assert "onclick" not in content["html"]
            shots = await UrlRecorder._walk_and_capture(
                page, [{"action": "wait", "capture_id": "home", "settle_ms": 0}]
            )
            archive = UrlRecorder._zip_screens(shots, tmp_path)
            with zipfile.ZipFile(archive) as bundle:
                assert bundle.read("home.png").startswith(b"\x89PNG")
                assert json.loads(bundle.read("home.page.json"))["text"] == content["text"]
        finally:
            await browser.close()
