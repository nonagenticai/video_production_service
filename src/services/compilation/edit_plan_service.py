"""AI-powered edit plan generation routed through worker_management (WM).

w2v services NEVER call openai / anthropic / inference-router directly — every
LLM call goes through ``worker_management.POST /workers/{name}/message``
so cost is attributed per tenant and providers can be swapped server-side.
See memory ``wm-as-sole-llm-gateway.md`` for the canonical pattern.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from w2v_common.clients import WMClient

from src.api.schemas import EditPlanDecisionResponse, EditPlanResponse

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent / "templates" / "video_edit_plan.md"

_WORKER_NAME = "edit-plan-generator"

_EDIT_PLAN_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string"},
                    "speed_factor": {"type": "number"},
                    "trim_start_s": {"type": "number"},
                    "trim_end_s": {"type": "number"},
                    "remove_dead_time": {"type": "boolean"},
                    "notes": {"type": "string"},
                },
                "required": ["part_id"],
            },
        },
    },
    "required": ["decisions"],
}


class EditPlanService:
    """Generate AI edit plans for video compilation via WM."""

    def __init__(self, *, wm_client: WMClient | None = None) -> None:
        # One shared WMClient per service — its underlying httpx.AsyncClient is
        # connection-pooled and the identity event hook reads request-scoped
        # ContextVars, so the same instance serves concurrent end-user requests
        # correctly.
        self._wm = wm_client if wm_client is not None else WMClient()
        self._last_token_usage: dict[str, int] | None = None

    async def generate_edit_plan(
        self,
        recordings: list[dict[str, Any]],
        shot_list: dict[str, Any],
        target_duration_minutes: float | None = None,
        llm_config: Any | None = None,
    ) -> EditPlanResponse:
        """Generate an AI edit plan for the given recordings and shot list.

        Args:
            recordings: List of recording dicts (from RecordingInput.model_dump()).
            shot_list: Shot list dict (from ShotListInput.model_dump()).
            target_duration_minutes: Optional target duration for the final video.
            llm_config: Deprecated — accepted for API back-compat but ignored.
                LLM provider/model selection is owned by worker_management;
                services no longer choose providers per-call.

        Returns:
            EditPlanResponse with decisions for each successful recording.
        """
        template = self._load_template()
        prompt = self._build_prompt(template, recordings, shot_list, target_duration_minutes)

        llm_start = time.time()
        try:
            decisions = await self._call_llm(prompt)
            token_usage = self._last_token_usage
            logger.info(
                "WM edit plan generated in %.2fs: decisions=%d",
                time.time() - llm_start,
                len(decisions),
            )
        except OSError, RuntimeError, ValueError, TimeoutError:
            logger.exception(
                "WM edit plan generation failed after %.2fs, returning defaults",
                time.time() - llm_start,
            )
            decisions = self._default_decisions(recordings)
            token_usage = None

        estimated = self._estimate_duration(recordings, decisions)

        return EditPlanResponse(
            decisions=decisions,
            estimated_duration_seconds=estimated,
            token_usage=token_usage,
        )

    def _load_template(self) -> str:
        """Load the prompt template from disk."""
        try:
            return _TEMPLATE_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("Template not found at %s, using inline fallback", _TEMPLATE_PATH)
            return (
                "Analyze the recorded video segments and generate editing decisions "
                "for each part as a JSON array."
            )

    def _build_prompt(
        self,
        template: str,
        recordings: list[dict[str, Any]],
        shot_list: dict[str, Any],
        target_duration_minutes: float | None = None,
    ) -> str:
        """Build the full prompt from template, recordings, and shot list."""
        parts_by_id: dict[str, dict[str, Any]] = {p["id"]: p for p in shot_list.get("parts", [])}

        parts_info: list[dict[str, Any]] = []
        for rec in recordings:
            if not rec.get("success", True):
                continue
            part = parts_by_id.get(rec.get("part_id", ""))
            parts_info.append(
                {
                    "part_id": rec.get("part_id", ""),
                    "duration_seconds": rec.get("duration_seconds", 0.0),
                    "file_size_bytes": rec.get("file_size_bytes", 0),
                    "dimensions": f"{rec.get('width', 1920)}x{rec.get('height', 1080)}",
                    "has_audio": rec.get("has_audio", False),
                    "title": part.get("title", "") if part else "",
                    "part_type": part.get("part_type", "unknown") if part else "unknown",
                }
            )

        prompt = (
            f"{template}\n\n## Recorded Parts\n\n```json\n{json.dumps(parts_info, indent=2)}\n```"
        )

        if target_duration_minutes is not None and target_duration_minutes > 0:
            total_secs = target_duration_minutes * 60
            prompt += (
                f"\n\n## Target Duration\n\n"
                f"Target: **{target_duration_minutes:.1f} minutes "
                f"({total_secs:.0f} seconds)**."
            )

        return prompt

    async def _call_llm(self, prompt: str) -> list[EditPlanDecisionResponse]:
        """Call WM with the structured-output schema and return parsed decisions."""
        response = await self._wm.message(
            worker_name=_WORKER_NAME,
            message=prompt,
            output_schema=_EDIT_PLAN_OUTPUT_SCHEMA,
        )

        # Capture token usage if WM returned it.
        usage = response.usage_metrics
        if usage:
            self._last_token_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
        else:
            self._last_token_usage = None

        return self._parse_decisions(response.output)

    def _parse_decisions(self, raw: Any) -> list[EditPlanDecisionResponse]:
        """Parse decisions from a WM ``output`` payload.

        WM with ``output_schema`` set typically returns a ``dict`` already
        matching the schema. Tolerate alternate shapes (string / list / wrapper
        object) for providers that ignored the schema.
        """
        data: Any
        data = self._extract_json(raw) if isinstance(raw, str) else raw

        if isinstance(data, dict):
            items: list[Any] = list(data.get("decisions", []))
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError(f"Unexpected JSON shape: {type(data).__name__}")

        decisions: list[EditPlanDecisionResponse] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            decisions.append(
                EditPlanDecisionResponse(
                    part_id=str(item.get("part_id", "")),
                    speed_factor=(
                        float(item["speed_factor"]) if item.get("speed_factor") is not None else 1.0
                    ),
                    trim_start_s=(
                        float(item["trim_start_s"]) if item.get("trim_start_s") is not None else 0.0
                    ),
                    trim_end_s=(
                        float(item["trim_end_s"]) if item.get("trim_end_s") is not None else 0.0
                    ),
                    remove_dead_time=bool(item.get("remove_dead_time", False)),
                    notes=str(item.get("notes", "")),
                )
            )
        return decisions

    @staticmethod
    def _extract_json(raw: str) -> Any:
        """Extract and parse JSON from a raw string response.

        Tries direct parsing first, then looks for JSON inside markdown
        code fences (```json ... ``` or ``` ... ```).
        """
        import re

        stripped = raw.strip()

        # Try direct parse
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fences
        fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
        for match in fence_pattern.finditer(stripped):
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not parse JSON from response: {stripped[:200]}")

    @staticmethod
    def _default_decisions(
        recordings: list[dict[str, Any]],
    ) -> list[EditPlanDecisionResponse]:
        """Return safe default decisions (no edits) for all successful recordings."""
        return [
            EditPlanDecisionResponse(
                part_id=rec.get("part_id", ""),
                speed_factor=1.0,
                trim_start_s=0.0,
                trim_end_s=0.0,
                remove_dead_time=False,
                notes="Default (LLM unavailable)",
            )
            for rec in recordings
            if rec.get("success", True)
        ]

    @staticmethod
    def _estimate_duration(
        recordings: list[dict[str, Any]],
        decisions: list[EditPlanDecisionResponse],
    ) -> float | None:
        """Estimate resulting duration based on decisions.

        Returns None if no recordings have duration info.
        """
        decision_map = {d.part_id: d for d in decisions}
        total = 0.0
        has_duration = False

        for rec in recordings:
            if not rec.get("success", True):
                continue
            dur = rec.get("duration_seconds", 0.0)
            if dur <= 0:
                continue
            has_duration = True
            part_id = rec.get("part_id", "")
            decision = decision_map.get(part_id)
            if decision:
                effective = max(
                    0.0,
                    dur - decision.trim_start_s - decision.trim_end_s,
                )
                if decision.speed_factor > 0:
                    effective /= decision.speed_factor
                total += effective
            else:
                total += dur

        return round(total, 2) if has_duration else None

    async def aclose(self) -> None:
        """Close the underlying WM client."""
        await self._wm.aclose()
