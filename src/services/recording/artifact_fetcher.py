"""Pull Playwright scripts from artifact_service into a recording-ready dir.

This is the bridge between content_service (which uploads each generated .ts
to artifact_service) and the existing PlaywrightRecorder, which expects a
filesystem-resident scripts_dir layout::

    <scripts_base_dir>/<job_id>/
      ├── parts/
      │     ├── part-001.demo.spec.ts
      │     └── …
      ├── playwright.config.ts
      └── package.json

We don't run ``npm install`` here — the vps image either (a) ships
``node_modules`` next to the vendored scaffold, or (b) Playwright's CLI
resolves the global install. Both work today; npm install is **not** part
of this fetcher and **must not** be added.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from uuid import UUID

from w2v_common.clients import ArtifactClient

logger = logging.getLogger(__name__)


class ArtifactBatchError(Exception):
    """One or more parts in a batch download failed.

    Carries per-part failure detail so the caller can map it to an HTTP status
    and tell the user *which* parts to retry. Without this, a single bad
    artifact_id silently cancelled every concurrent download in the same
    ``asyncio.gather`` and the caller only saw the first traceback.
    """

    def __init__(
        self,
        failed: list[tuple[str, BaseException]],
        succeeded: list[str],
    ) -> None:
        self.failed = failed
        self.succeeded = succeeded
        total = len(failed) + len(succeeded)
        detail = ", ".join(f"{pid}={exc!r}" for pid, exc in failed)
        super().__init__(f"{len(failed)}/{total} parts failed: {detail}")


class ArtifactFetcher:
    """Materializes a recording-ready directory from artifact_service IDs."""

    def __init__(self, artifact_client: ArtifactClient, scripts_base_dir: Path) -> None:
        self._artifact = artifact_client
        self._base = scripts_base_dir.resolve()

    async def materialize(
        self,
        *,
        job_id: UUID,
        project_id: UUID,
        parts: list[tuple[str, UUID]],
        scaffold_src: Path | None = None,
    ) -> Path:
        """Download each (part_id, artifact_id) into <base>/<job_id>/parts/.

        Returns the resolved scripts_dir suitable for PlaywrightRecorder.
        """
        scripts_dir = (self._base / str(job_id)).resolve()
        if scripts_dir != self._base and not scripts_dir.is_relative_to(self._base):
            # Defence in depth: any path-traversal sneak attempt blows up here.
            raise ValueError(f"scripts_dir escaped base: {scripts_dir}")
        parts_dir = scripts_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        # Optional: copy a vendored Playwright scaffold (playwright.config.ts,
        # package.json, helpers/) so the part files have a project to run in.
        if scaffold_src is not None and scaffold_src.exists():
            for entry in scaffold_src.iterdir():
                target = scripts_dir / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(entry, target)

        # Synchronously validate every part_id before kicking off any downloads.
        # part_id is already regex-validated upstream; this is defence in depth,
        # and it must stay synchronous so an unsafe input fails fast and loud
        # (not buried inside an ArtifactBatchError alongside real download faults).
        for part_id, _ in parts:
            if "/" in part_id or "\\" in part_id or ".." in part_id:
                raise ValueError(f"unsafe part_id: {part_id!r}")

        await self._download_all(parts, parts_dir, project_id, job_id)
        logger.info("Materialized %d scripts into %s for job %s", len(parts), scripts_dir, job_id)
        return scripts_dir

    async def _download_all(
        self,
        parts: list[tuple[str, UUID]],
        parts_dir: Path,
        project_id: UUID,
        job_id: UUID,
    ) -> None:
        """Download every part concurrently; raise ArtifactBatchError on partial failure.

        ``return_exceptions=True`` is the whole point: without it, one bad part
        cancels every sibling mid-download (half-written .ts files; opaque
        single traceback). With it, every download either completes or is
        captured per-part so the caller learns *which* parts to retry.
        """

        async def _one(part_id: str, artifact_id: UUID) -> Path:
            dest = parts_dir / f"{part_id}.demo.spec.ts"
            return await self._artifact.download_to_path(
                project_id=project_id,
                artifact_id=artifact_id,
                dest=dest,
            )

        results = await asyncio.gather(
            *(_one(p, a) for p, a in parts),
            return_exceptions=True,
        )
        failed: list[tuple[str, BaseException]] = []
        succeeded: list[str] = []
        for (part_id, _), result in zip(parts, results, strict=True):
            if isinstance(result, BaseException):
                failed.append((part_id, result))
            else:
                succeeded.append(part_id)
        if failed:
            logger.warning(
                "Batch materialize: %d/%d parts failed for job %s",
                len(failed),
                len(parts),
                job_id,
            )
            raise ArtifactBatchError(failed=failed, succeeded=succeeded)
