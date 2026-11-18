"""Converts findings/review data into a shot list for video compilation."""

from __future__ import annotations

import logging
import uuid

logger = logging.getLogger(__name__)

# Severity ordering: lower index = higher priority.
_SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}

_MIN_DURATION_SECONDS = 4.0
_MAX_DURATION_SECONDS = 8.0
# Approximate characters per second for readable title cards.
_CHARS_PER_SECOND = 15.0
_BLOCKED_APPROVAL_STATUSES = {"rejected", "blocked", "failed"}
_ASSET_KIND_ALIASES = {
    "slide_preview": "slide_preview",
    "slide": "slide_preview",
    "deck_preview": "slide_preview",
    "comparison_frame": "comparison_frame",
    "comparison": "comparison_frame",
    "generated_image": "generated_image",
    "generated": "generated_image",
    "image": "generated_image",
}


def _estimate_duration(text: str) -> float:
    """Estimate a title-card duration from the total text length.

    Clamps the result between ``_MIN_DURATION_SECONDS`` and
    ``_MAX_DURATION_SECONDS``.
    """
    char_count = len(text)
    seconds = max(char_count / _CHARS_PER_SECOND, _MIN_DURATION_SECONDS)
    return min(seconds, _MAX_DURATION_SECONDS)


def _severity_sort_key(finding: dict) -> int:
    """Return a numeric key so that higher-severity findings sort first."""
    return _SEVERITY_ORDER.get(
        (finding.get("severity") or "info").lower(),
        len(_SEVERITY_ORDER),
    )


def _first_image_ref(*values: object) -> str:
    for value in values:
        ref = str(value or "").strip()
        if ref:
            return ref
    return ""


def _normalize_asset_kind(kind: object) -> str:
    normalized = str(kind or "").strip().lower().replace(" ", "_")
    return _ASSET_KIND_ALIASES.get(normalized, "")


def _build_visual_part(finding: dict, *, finding_id: str, duration: float) -> dict | None:
    approval_status = str(finding.get("approval_status") or "").strip().lower()
    if approval_status in _BLOCKED_APPROVAL_STATUSES:
        return None

    asset_kind = _normalize_asset_kind(
        finding.get("asset_kind")
        or finding.get("visual_asset_type")
        or finding.get("evidence_type")
    )
    slide_ref = _first_image_ref(
        finding.get("slide_image_url"),
        finding.get("slide_preview_url"),
        finding.get("preview_image_url"),
    )
    comparison_ref = _first_image_ref(
        finding.get("comparison_frame_url"),
        finding.get("comparison_image_url"),
    )
    generic_ref = _first_image_ref(
        finding.get("generated_image_url"),
        finding.get("image_url"),
        finding.get("evidence_path"),
    )

    part_type = "generated_image"
    image_ref = generic_ref
    if slide_ref or asset_kind == "slide_preview":
        part_type = "slide_preview"
        image_ref = slide_ref or generic_ref
    elif comparison_ref or asset_kind == "comparison_frame":
        part_type = "comparison_frame"
        image_ref = comparison_ref or generic_ref
    elif asset_kind == "generated_image":
        part_type = "generated_image"

    if not image_ref:
        return None

    part = {
        "id": finding_id,
        "title": finding.get("title") or "",
        "subtitle": finding.get("description") or "",
        "part_type": part_type,
        "estimated_duration_seconds": duration,
        "slide_role": finding.get("slide_role") or "finding_evidence",
        "asset_id": finding.get("asset_id"),
        "asset_kind": part_type,
        "asset_source": finding.get("asset_source"),
        "approval_status": finding.get("approval_status"),
        "provenance": finding.get("provenance"),
        "motion_preset": finding.get("motion_preset"),
    }
    if part_type == "slide_preview":
        part["slide_image_url"] = image_ref
    else:
        part["image_url"] = image_ref
    return part


class FindingsVideoService:
    """Build a ``ShotListInput``-compatible dict from a list of findings."""

    def build_shot_list(
        self,
        findings: list[dict],
        title: str,
        subtitle: str,
    ) -> dict:
        """Convert findings into a shot list with visual-first finding segments.

        Parameters
        ----------
        findings:
            Each dict should have at least ``title``, ``description``,
            ``severity`` (critical/high/medium/low/info), and ``category``.
        title:
            Report-level title shown on the opening title card.
        subtitle:
            Report-level subtitle shown on the opening title card.

        Returns
        -------
        dict
            A ``ShotListInput``-compatible dict with ``parts`` (list) and
            ``recording_order`` (list of part ids).
        """
        parts: list[dict] = []
        recording_order: list[str] = []

        # -- Opening title card --
        opening_id = str(uuid.uuid7())
        combined_text = f"{title} {subtitle}"
        parts.append(
            {
                "id": opening_id,
                "title": title,
                "subtitle": subtitle,
                "part_type": "title_card",
                "estimated_duration_seconds": _estimate_duration(combined_text),
            }
        )
        recording_order.append(opening_id)

        # -- Group findings by category --
        sorted_findings = sorted(findings, key=_severity_sort_key)

        categories: dict[str, list[dict]] = {}
        for finding in sorted_findings:
            cat = finding.get("category") or "general"
            categories.setdefault(cat, []).append(finding)

        # -- Build parts per category --
        for category, cat_findings in categories.items():
            # Category title card
            cat_id = str(uuid.uuid7())
            cat_title = category
            cat_subtitle = f"{len(cat_findings)} finding{'s' if len(cat_findings) != 1 else ''}"
            cat_text = f"{cat_title} {cat_subtitle}"
            parts.append(
                {
                    "id": cat_id,
                    "title": cat_title,
                    "subtitle": cat_subtitle,
                    "part_type": "title_card",
                    "estimated_duration_seconds": _estimate_duration(cat_text),
                }
            )
            recording_order.append(cat_id)

            # Individual finding segments. If the caller provided a deck/image
            # asset, emit a visual part; otherwise fall back to a title card.
            for finding in cat_findings:
                finding_id = str(uuid.uuid7())
                f_title = finding.get("title") or ""
                f_desc = finding.get("description") or ""
                combined = f"{f_title} {f_desc}"
                duration = _estimate_duration(combined)
                visual_part = _build_visual_part(finding, finding_id=finding_id, duration=duration)
                if visual_part is not None:
                    parts.append(visual_part)
                else:
                    parts.append(
                        {
                            "id": finding_id,
                            "title": f_title,
                            "subtitle": f_desc,
                            "part_type": "title_card",
                            "slide_role": "finding_fallback_card",
                            "estimated_duration_seconds": duration,
                        }
                    )
                recording_order.append(finding_id)

        logger.info(
            "Built shot list with %d parts from %d findings",
            len(parts),
            len(findings),
        )

        return {
            "parts": parts,
            "recording_order": recording_order,
        }
