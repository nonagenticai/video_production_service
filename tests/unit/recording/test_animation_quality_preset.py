"""POST /api/record/animation forwards ``quality_preset`` to the recorder's CRF.

Ken Burns clips reach VP through this route; without the field every clip was
encoded at the ``standard`` preset (CRF 23) regardless of the order's tier.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from src.services.recording.quality_presets import QUALITY_PRESETS
from tests.unit.recording.test_animation_recorder import _make_recorder
from tests.unit.recording.test_routes import _fake_create_task, _make_client

_URL = "/api/record/animation"
_HTML = "<html><body><script>window.RENDER_DURATION_MS=1000</script></body></html>"


def _post(payload_extra: dict):
    service = AsyncMock()
    service.create_animation_job = AsyncMock(return_value=str(uuid.uuid4()))
    client, _ = _make_client(recorder_service=service)
    with patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task):
        resp = client.post(_URL, json={"html": _HTML, "part_id": "kb_1", **payload_extra})
    return resp, service


@pytest.mark.parametrize("name", sorted(QUALITY_PRESETS))
def test_preset_reaches_recorder_crf(name):
    resp, service = _post({"quality_preset": name})
    assert resp.status_code in (200, 202), resp.text
    cfg = service.create_animation_job.await_args.kwargs["recording_config"]
    assert cfg["quality_preset"] == name
    assert _make_recorder()._resolve_config(cfg)["crf"] == QUALITY_PRESETS[name].crf


def test_2k_is_crf_18_not_standard_23():
    _, service = _post({"quality_preset": "2k"})
    cfg = service.create_animation_job.await_args.kwargs["recording_config"]
    assert _make_recorder()._resolve_config(cfg)["crf"] == 18


def test_omitted_preset_is_unchanged_standard():
    resp, service = _post({})
    assert resp.status_code in (200, 202), resp.text
    cfg = service.create_animation_job.await_args.kwargs["recording_config"]
    # Byte-identical to the pre-field config: no quality_preset key at all.
    assert cfg == {
        "animation_config": {
            "params": None,
            "fps": 30,
            "duration_ms": None,
            "output_format": "mp4",
            "transparent": False,
        }
    }
    assert _make_recorder()._resolve_config(cfg)["crf"] == 23


@pytest.mark.parametrize("bad", ["ultra", "2K", "", "Standard"])
def test_unknown_preset_is_422_and_no_job(bad):
    resp, service = _post({"quality_preset": bad})
    assert resp.status_code == 422, resp.text
    service.create_animation_job.assert_not_awaited()


# The exact names the orchestrator's ken_burns stage sends (resolved via
# site_capture's _resolve_recording_quality_preset; an empty snapshot -> "2k").
# Pinned literally so a registry rename cannot silently 422 every Ken Burns render.
@pytest.mark.parametrize(("name", "crf"), [("2k", 18), ("4k", 18), ("high_quality", 18)])
def test_orchestrator_ken_burns_presets_accepted_below_standard_crf(name, crf):
    resp, service = _post({"quality_preset": name})
    assert resp.status_code in (200, 202), resp.text
    cfg = service.create_animation_job.await_args.kwargs["recording_config"]
    resolved = _make_recorder()._resolve_config(cfg)["crf"]
    assert resolved == crf
    assert resolved < QUALITY_PRESETS["standard"].crf == 23
