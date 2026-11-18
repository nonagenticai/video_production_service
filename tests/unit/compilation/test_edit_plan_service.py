"""Unit tests for edit plan service (WM-routed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.compilation.edit_plan_service import EditPlanService


def _make_service(wm_response: object | None = None) -> EditPlanService:
    """Build an EditPlanService whose WMClient is a no-call mock by default."""
    wm = MagicMock()
    wm.message = AsyncMock(return_value=wm_response)
    wm.aclose = AsyncMock()
    return EditPlanService(wm_client=wm)


class TestExtractJson:
    def test_direct_json_array(self):
        result = EditPlanService._extract_json('[{"part_id": "p1"}]')
        assert isinstance(result, list)
        assert result[0]["part_id"] == "p1"

    def test_wrapped_object(self):
        result = EditPlanService._extract_json('{"decisions": [{"part_id": "p1"}]}')
        assert isinstance(result, dict)
        assert result["decisions"][0]["part_id"] == "p1"

    def test_markdown_code_fence(self):
        raw = 'Here is the plan:\n```json\n[{"part_id": "p1"}]\n```\nDone.'
        result = EditPlanService._extract_json(raw)
        assert isinstance(result, list)
        assert result[0]["part_id"] == "p1"

    def test_plain_code_fence_without_json_tag(self):
        raw = 'Result:\n```\n[{"part_id": "p2"}]\n```'
        result = EditPlanService._extract_json(raw)
        assert isinstance(result, list)
        assert result[0]["part_id"] == "p2"

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            EditPlanService._extract_json("not json at all")


class TestParseDecisions:
    def test_parse_dict_format(self):
        svc = _make_service()
        decisions = svc._parse_decisions({"decisions": [{"part_id": "p1", "speed_factor": 1.5}]})
        assert len(decisions) == 1
        assert decisions[0].part_id == "p1"
        assert decisions[0].speed_factor == 1.5

    def test_parse_list_format(self):
        svc = _make_service()
        decisions = svc._parse_decisions(
            [{"part_id": "p1", "speed_factor": 1.5, "trim_start_s": 0.5}]
        )
        assert len(decisions) == 1
        assert decisions[0].speed_factor == 1.5
        assert decisions[0].trim_start_s == 0.5

    def test_parse_string_array_format(self):
        svc = _make_service()
        decisions = svc._parse_decisions('[{"part_id":"p1","speed_factor":1.5}]')
        assert len(decisions) == 1
        assert decisions[0].part_id == "p1"

    def test_parse_string_wrapper_format(self):
        svc = _make_service()
        decisions = svc._parse_decisions('{"decisions":[{"part_id":"p2","remove_dead_time":true}]}')
        assert len(decisions) == 1
        assert decisions[0].part_id == "p2"
        assert decisions[0].remove_dead_time is True

    def test_parse_malformed_string_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError):
            svc._parse_decisions("not json")

    def test_parse_with_defaults(self):
        svc = _make_service()
        decisions = svc._parse_decisions([{"part_id": "p1"}])
        assert decisions[0].speed_factor == 1.0
        assert decisions[0].trim_start_s == 0.0
        assert decisions[0].trim_end_s == 0.0
        assert decisions[0].remove_dead_time is False
        assert decisions[0].notes == ""

    def test_parse_unexpected_shape_raises(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="Unexpected JSON shape"):
            svc._parse_decisions(42)  # type: ignore[arg-type]


class TestDefaultDecisions:
    def test_returns_defaults_for_successful(self):
        recordings = [
            {"part_id": "p1", "success": True},
            {"part_id": "p2", "success": False},
            {"part_id": "p3", "success": True},
        ]
        decisions = EditPlanService._default_decisions(recordings)
        assert len(decisions) == 2
        assert all(d.speed_factor == 1.0 for d in decisions)
        assert {d.part_id for d in decisions} == {"p1", "p3"}

    def test_empty_recordings(self):
        assert EditPlanService._default_decisions([]) == []

    def test_all_failed_recordings(self):
        recordings = [
            {"part_id": "p1", "success": False},
            {"part_id": "p2", "success": False},
        ]
        assert EditPlanService._default_decisions(recordings) == []

    def test_default_notes_mention_llm_unavailable(self):
        recordings = [{"part_id": "p1", "success": True}]
        decisions = EditPlanService._default_decisions(recordings)
        assert "Default" in decisions[0].notes


class TestEstimateDuration:
    def test_with_speed_factor(self):
        from src.api.schemas.compilation import EditPlanDecisionResponse

        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 10.0}]
        decisions = [EditPlanDecisionResponse(part_id="p1", speed_factor=2.0)]
        assert EditPlanService._estimate_duration(recordings, decisions) == 5.0

    def test_with_trim(self):
        from src.api.schemas.compilation import EditPlanDecisionResponse

        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 10.0}]
        decisions = [EditPlanDecisionResponse(part_id="p1", trim_start_s=1.0, trim_end_s=2.0)]
        assert EditPlanService._estimate_duration(recordings, decisions) == 7.0

    def test_with_speed_and_trim(self):
        from src.api.schemas.compilation import EditPlanDecisionResponse

        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 10.0}]
        decisions = [
            EditPlanDecisionResponse(
                part_id="p1", speed_factor=2.0, trim_start_s=1.0, trim_end_s=1.0
            )
        ]
        assert EditPlanService._estimate_duration(recordings, decisions) == 4.0

    def test_no_duration_returns_none(self):
        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 0.0}]
        assert EditPlanService._estimate_duration(recordings, []) is None

    def test_multiple_parts(self):
        from src.api.schemas.compilation import EditPlanDecisionResponse

        recordings = [
            {"part_id": "p1", "success": True, "duration_seconds": 10.0},
            {"part_id": "p2", "success": True, "duration_seconds": 20.0},
        ]
        decisions = [
            EditPlanDecisionResponse(part_id="p1", speed_factor=2.0),
            EditPlanDecisionResponse(part_id="p2", speed_factor=1.0),
        ]
        assert EditPlanService._estimate_duration(recordings, decisions) == 25.0

    def test_no_decision_for_part_uses_raw_duration(self):
        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 10.0}]
        assert EditPlanService._estimate_duration(recordings, []) == 10.0

    def test_failed_recordings_skipped(self):
        recordings = [
            {"part_id": "p1", "success": False, "duration_seconds": 100.0},
            {"part_id": "p2", "success": True, "duration_seconds": 5.0},
        ]
        assert EditPlanService._estimate_duration(recordings, []) == 5.0


class TestBuildPrompt:
    def test_builds_prompt_with_parts(self):
        svc = _make_service()
        recordings = [
            {
                "part_id": "p1",
                "success": True,
                "duration_seconds": 10.0,
                "width": 1920,
                "height": 1080,
            }
        ]
        shot_list = {
            "parts": [{"id": "p1", "title": "Test", "part_type": "browser_recording"}],
            "recording_order": ["p1"],
        }
        prompt = svc._build_prompt("Template", recordings, shot_list)
        assert "p1" in prompt
        assert "Template" in prompt
        assert "Test" in prompt
        assert "browser_recording" in prompt

    def test_includes_target_duration(self):
        svc = _make_service()
        prompt = svc._build_prompt("Template", [], {"parts": []}, target_duration_minutes=5.0)
        assert "5.0 minutes" in prompt
        assert "300 seconds" in prompt

    def test_no_target_duration(self):
        svc = _make_service()
        prompt = svc._build_prompt("Template", [], {"parts": []})
        assert "Target Duration" not in prompt

    def test_skips_failed_recordings(self):
        svc = _make_service()
        recordings = [
            {"part_id": "p1", "success": True, "duration_seconds": 10.0},
            {"part_id": "p2", "success": False, "duration_seconds": 5.0},
        ]
        shot_list = {
            "parts": [
                {"id": "p1", "title": "Good"},
                {"id": "p2", "title": "Bad"},
            ],
        }
        prompt = svc._build_prompt("Template", recordings, shot_list)
        assert "p1" in prompt
        assert '"part_id": "p2"' not in prompt


class TestGenerateEditPlan:
    async def test_generate_edit_plan_success(self):
        wm_response = MagicMock()
        wm_response.output = {"decisions": [{"part_id": "p1", "speed_factor": 1.5}]}
        wm_response.usage_metrics = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        svc = _make_service(wm_response)

        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 10.0}]
        shot_list = {"parts": [{"id": "p1", "title": "Test"}]}

        result = await svc.generate_edit_plan(recordings, shot_list)
        assert len(result.decisions) == 1
        assert result.decisions[0].part_id == "p1"
        assert result.decisions[0].speed_factor == 1.5
        assert result.token_usage == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

    async def test_generate_edit_plan_passes_schema_and_worker(self):
        wm_response = MagicMock()
        wm_response.output = {"decisions": [{"part_id": "p1"}]}
        wm_response.usage_metrics = None
        wm = MagicMock()
        wm.message = AsyncMock(return_value=wm_response)
        wm.aclose = AsyncMock()
        svc = EditPlanService(wm_client=wm)

        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 10.0}]
        shot_list = {"parts": [{"id": "p1", "title": "Test"}]}

        await svc.generate_edit_plan(recordings, shot_list)

        wm.message.assert_awaited_once()
        await_args = wm.message.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["worker_name"] == "edit-plan-generator"
        assert "system_prompt" not in kwargs
        assert kwargs["output_schema"]["type"] == "object"
        assert "decisions" in kwargs["output_schema"]["properties"]

    async def test_generate_edit_plan_llm_failure_falls_back(self):
        wm = MagicMock()
        wm.message = AsyncMock(side_effect=RuntimeError("WM down"))
        wm.aclose = AsyncMock()
        svc = EditPlanService(wm_client=wm)

        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 10.0}]
        shot_list = {"parts": [{"id": "p1", "title": "Test"}]}

        result = await svc.generate_edit_plan(recordings, shot_list)
        assert len(result.decisions) == 1
        assert result.decisions[0].speed_factor == 1.0
        assert "Default" in result.decisions[0].notes
        assert result.token_usage is None

    async def test_generate_edit_plan_string_output_parsed(self):
        """WM returning a stringified JSON (provider ignored output_schema) is tolerated."""
        wm_response = MagicMock()
        wm_response.output = '[{"part_id":"p1","speed_factor":2.0}]'
        wm_response.usage_metrics = None
        svc = _make_service(wm_response)

        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 10.0}]
        shot_list = {"parts": [{"id": "p1", "title": "Test"}]}

        result = await svc.generate_edit_plan(recordings, shot_list)
        assert len(result.decisions) == 1
        assert result.decisions[0].speed_factor == 2.0

    async def test_generate_edit_plan_ignores_llm_config(self):
        """llm_config is accepted for back-compat but never reaches WMClient."""
        wm_response = MagicMock()
        wm_response.output = {"decisions": [{"part_id": "p1"}]}
        wm_response.usage_metrics = None
        wm = MagicMock()
        wm.message = AsyncMock(return_value=wm_response)
        wm.aclose = AsyncMock()
        svc = EditPlanService(wm_client=wm)

        recordings = [{"part_id": "p1", "success": True, "duration_seconds": 10.0}]
        shot_list = {"parts": [{"id": "p1", "title": "Test"}]}

        # Some random llm_config object — must not propagate to WM.
        llm_config = MagicMock(model="ignored", base_url="ignored", api_key_env="X")
        await svc.generate_edit_plan(recordings, shot_list, llm_config=llm_config)

        await_args = wm.message.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert "model_override" not in kwargs or kwargs.get("model_override") is None


class TestAclose:
    async def test_aclose_propagates_to_wm(self):
        wm = MagicMock()
        wm.message = AsyncMock()
        wm.aclose = AsyncMock()
        svc = EditPlanService(wm_client=wm)
        await svc.aclose()
        wm.aclose.assert_awaited_once()
