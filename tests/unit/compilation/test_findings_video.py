"""Unit tests for findings_video_service."""

from __future__ import annotations

from unittest.mock import patch

from src.services.compilation.findings_video_service import (
    FindingsVideoService,
    _estimate_duration,
    _severity_sort_key,
)


class TestEstimateDuration:
    def test_short_text_clamps_to_minimum(self):
        """Text shorter than 60 chars (60/15=4.0) should clamp to _MIN_DURATION_SECONDS."""
        text = "short"  # 5 chars -> 5/15 = 0.33, clamped to 4.0
        result = _estimate_duration(text)
        assert result == 4.0

    def test_long_text_clamps_to_maximum(self):
        """Text longer than 120 chars (120/15=8.0) should clamp to _MAX_DURATION_SECONDS."""
        text = "x" * 200  # 200 chars -> 200/15 = 13.33, clamped to 8.0
        result = _estimate_duration(text)
        assert result == 8.0

    def test_mid_range_text_is_proportional(self):
        """Text that produces a value between min and max should be chars / 15.0."""
        text = "a" * 90  # 90 chars -> 90/15 = 6.0
        result = _estimate_duration(text)
        assert result == 6.0


class TestSeveritySortKey:
    def test_known_severities_return_correct_order(self):
        """Each known severity maps to the expected numeric position."""
        assert _severity_sort_key({"severity": "critical"}) == 0
        assert _severity_sort_key({"severity": "high"}) == 1
        assert _severity_sort_key({"severity": "medium"}) == 2
        assert _severity_sort_key({"severity": "low"}) == 3
        assert _severity_sort_key({"severity": "info"}) == 4

    def test_unknown_severity_returns_sentinel_value(self):
        """An unrecognised severity should return len(_SEVERITY_ORDER) as a sentinel."""
        result = _severity_sort_key({"severity": "banana"})
        assert result == 5  # len({"critical","high","medium","low","info"})


class TestBuildShotList:
    @patch("src.services.compilation.findings_video_service.uuid.uuid7", side_effect=["id-0"])
    def test_empty_findings_returns_only_opening_card(self, _mock_uuid):
        """An empty findings list should produce exactly one opening title card."""
        svc = FindingsVideoService()
        result = svc.build_shot_list(findings=[], title="Report", subtitle="Sub")

        assert len(result["parts"]) == 1
        assert result["parts"][0]["id"] == "id-0"
        assert result["parts"][0]["title"] == "Report"
        assert result["parts"][0]["subtitle"] == "Sub"
        assert result["parts"][0]["part_type"] == "title_card"
        assert result["recording_order"] == ["id-0"]

    @patch(
        "src.services.compilation.findings_video_service.uuid.uuid7",
        side_effect=["opening-id", "cat-id", "finding-id"],
    )
    def test_single_finding_produces_three_parts(self, _mock_uuid):
        """One finding should produce: opening card + category card + finding card."""
        findings = [
            {
                "title": "Issue A",
                "description": "Desc A",
                "severity": "high",
                "category": "security",
            },
        ]
        svc = FindingsVideoService()
        result = svc.build_shot_list(findings=findings, title="T", subtitle="S")

        assert len(result["parts"]) == 3
        assert result["parts"][0]["id"] == "opening-id"
        assert result["parts"][1]["id"] == "cat-id"
        assert result["parts"][1]["title"] == "security"
        assert result["parts"][2]["id"] == "finding-id"
        assert result["parts"][2]["title"] == "Issue A"
        assert result["parts"][2]["part_type"] == "title_card"
        assert result["recording_order"] == ["opening-id", "cat-id", "finding-id"]

    @patch(
        "src.services.compilation.findings_video_service.uuid.uuid7",
        side_effect=["opening-id", "cat-id", "finding-id"],
    )
    def test_finding_with_evidence_path_uses_visual_asset_part(self, _mock_uuid):
        """Finding-level evidence should emit a visual segment instead of a title card."""
        findings = [
            {
                "title": "Issue A",
                "description": "Desc A",
                "severity": "high",
                "category": "security",
                "evidence_path": "s3://bucket/findings/issue-a.png",
            },
        ]
        svc = FindingsVideoService()
        result = svc.build_shot_list(findings=findings, title="T", subtitle="S")

        finding_part = result["parts"][2]
        assert finding_part["id"] == "finding-id"
        assert finding_part["part_type"] == "generated_image"
        assert finding_part["image_url"] == "s3://bucket/findings/issue-a.png"
        assert finding_part["slide_role"] == "finding_evidence"

    @patch(
        "src.services.compilation.findings_video_service.uuid.uuid7",
        side_effect=["opening-id", "cat-id", "finding-id"],
    )
    def test_slide_preview_asset_uses_slide_preview_part_type(self, _mock_uuid):
        """Deck preview assets should keep their slide_preview semantic in the shot list."""
        findings = [
            {
                "title": "Issue A",
                "description": "Desc A",
                "severity": "high",
                "category": "security",
                "slide_preview_url": "https://cdn.example/slides/01.png",
                "asset_kind": "slide_preview",
            },
        ]
        svc = FindingsVideoService()
        result = svc.build_shot_list(findings=findings, title="T", subtitle="S")

        finding_part = result["parts"][2]
        assert finding_part["part_type"] == "slide_preview"
        assert finding_part["slide_image_url"] == "https://cdn.example/slides/01.png"
        assert "image_url" not in finding_part

    @patch(
        "src.services.compilation.findings_video_service.uuid.uuid7",
        side_effect=["opening-id", "cat-id", "finding-id"],
    )
    def test_rejected_visual_asset_falls_back_to_title_card(self, _mock_uuid):
        """Rejected visuals should use the deterministic title-card fallback."""
        findings = [
            {
                "title": "Issue A",
                "description": "Desc A",
                "severity": "high",
                "category": "security",
                "evidence_path": "https://cdn.example/asset.png",
                "approval_status": "rejected",
            },
        ]
        svc = FindingsVideoService()
        result = svc.build_shot_list(findings=findings, title="T", subtitle="S")

        finding_part = result["parts"][2]
        assert finding_part["part_type"] == "title_card"
        assert finding_part["slide_role"] == "finding_fallback_card"

    @patch(
        "src.services.compilation.findings_video_service.uuid.uuid7",
        side_effect=[
            "opening-id",
            "cat-net-id",
            "finding-1-id",
            "cat-sec-id",
            "finding-2-id",
        ],
    )
    def test_multiple_categories_grouped_correctly(self, _mock_uuid):
        """Findings from different categories produce separate category groups."""
        findings = [
            {
                "title": "F1",
                "description": "D1",
                "severity": "high",
                "category": "network",
            },
            {
                "title": "F2",
                "description": "D2",
                "severity": "medium",
                "category": "security",
            },
        ]
        svc = FindingsVideoService()
        result = svc.build_shot_list(findings=findings, title="T", subtitle="S")

        # 1 opening + 2 categories * (1 cat card + 1 finding card) = 5 parts
        assert len(result["parts"]) == 5
        # Category titles should appear in order
        category_titles = [
            p["title"] for p in result["parts"] if p["title"] in ("network", "security")
        ]
        assert category_titles == ["network", "security"]

    @patch(
        "src.services.compilation.findings_video_service.uuid.uuid7",
        side_effect=[
            "opening-id",
            "cat-id",
            "finding-crit-id",
            "finding-low-id",
        ],
    )
    def test_severity_ordering_critical_before_low(self, _mock_uuid):
        """Within the same category, critical findings should come before low ones."""
        findings = [
            {
                "title": "Low Issue",
                "description": "desc",
                "severity": "low",
                "category": "security",
            },
            {
                "title": "Critical Issue",
                "description": "desc",
                "severity": "critical",
                "category": "security",
            },
        ]
        svc = FindingsVideoService()
        result = svc.build_shot_list(findings=findings, title="T", subtitle="S")

        # Parts: opening, category, critical finding, low finding
        finding_titles = [p["title"] for p in result["parts"][2:]]
        assert finding_titles == ["Critical Issue", "Low Issue"]

    @patch(
        "src.services.compilation.findings_video_service.uuid.uuid7",
        side_effect=[
            "opening-id",
            "cat-single-id",
            "finding-single-id",
            "cat-multi-id",
            "finding-multi-1-id",
            "finding-multi-2-id",
        ],
    )
    def test_pluralization_singular_and_plural(self, _mock_uuid):
        """Category subtitle should read '1 finding' (singular) or '2 findings' (plural)."""
        findings = [
            {
                "title": "A",
                "description": "d",
                "severity": "high",
                "category": "alpha",
            },
            {
                "title": "B",
                "description": "d",
                "severity": "medium",
                "category": "beta",
            },
            {
                "title": "C",
                "description": "d",
                "severity": "low",
                "category": "beta",
            },
        ]
        svc = FindingsVideoService()
        result = svc.build_shot_list(findings=findings, title="T", subtitle="S")

        # Collect category card subtitles
        cat_subtitles = {}
        for part in result["parts"]:
            if part["title"] in ("alpha", "beta"):
                cat_subtitles[part["title"]] = part["subtitle"]

        assert cat_subtitles["alpha"] == "1 finding"
        assert cat_subtitles["beta"] == "2 findings"
