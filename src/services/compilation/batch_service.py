"""Handles batch compilation orchestration."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import CompilationBatch
from src.services.compilation.compiler_service import CompilerService
from src.utils.error_formatting import format_error

logger = logging.getLogger(__name__)

# The two statuses a batch can sit in while it is still someone's responsibility.
# Anything else ('completed', 'partial', 'failed') is TERMINAL and must never be
# rewritten by a reaper: the owning task getting there first is the good outcome.
_NON_TERMINAL_BATCH_STATUSES = ("pending", "processing")

# What a reap attempt actually did. A bool could not tell "someone else owns this
# row right now" apart from "it finished first", and those need different logs.
BatchReapOutcome = Literal["reaped", "terminal", "locked", "missing"]

# Marks an ``errors`` entry as written by the reaper rather than by a child job, so
# ``process_batch`` can strip it if the batch finishes after being reaped. Without
# this a SUCCESSFUL long batch keeps a permanent "the owning process is gone" entry.
_REAPER_ERROR_SOURCE = "reaper"

# Marks an ``errors`` entry for a job that reached NEITHER counter — the task raised a
# BaseException (``CancelledError``) that ``_run_job``'s ``except Exception`` cannot
# catch, or the shortfall reconciliation found jobs it cannot even name. Distinct from
# a child job's own error entry so a caller can tell "this job failed and told us why"
# apart from "this job vanished and we only know it never finished".
_LOST_ERROR_SOURCE = "lost"


class BatchService:
    """Orchestrate batch compilation of multiple jobs."""

    def __init__(
        self,
        compiler_service: CompilerService,
        db_session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ) -> None:
        self._compiler_service = compiler_service
        self._db_session_factory = db_session_factory

    async def create_batch(
        self,
        db: AsyncSession,
        tenant_id: uuid.UUID,
        configs: list[dict],
        max_concurrent: int = 3,
    ) -> str:
        """Create a batch record and individual compilation jobs.

        Parameters
        ----------
        db:
            Active async database session.
        tenant_id:
            Tenant that owns the batch.
        configs:
            List of per-job configuration dicts (each passed as kwargs to
            ``compiler_service.create_job``).
        max_concurrent:
            Maximum number of jobs to process concurrently (stored for
            reference but not enforced until ``process_batch``).

        Returns
        -------
        str
            The ``batch_id`` for the newly created batch.
        """
        batch_id = str(uuid.uuid7())

        batch = CompilationBatch(
            batch_id=batch_id,
            tenant_id=tenant_id,
            status="pending",
            total_jobs=len(configs),
            completed_jobs=0,
            failed_jobs=0,
            job_ids=[],
            errors=[],
        )
        db.add(batch)
        await db.flush()

        job_ids: list[str] = []
        for config in configs:
            # Work on a copy so we never mutate the caller's dicts
            config = dict(config)
            # Remap 'theme' to 'theme_config' to match create_job() signature
            theme_data = config.pop("theme", None)
            if theme_data is not None:
                config["theme_config"] = theme_data
            # Also remap 'template' which isn't a create_job param
            config.pop("template", None)
            config.pop("llm_config", None)
            job_id = await self._compiler_service.create_job(
                db,
                tenant_id=tenant_id,
                **config,
            )
            job_ids.append(job_id)

        batch.job_ids = job_ids
        await db.flush()

        logger.info(
            "Created batch %s with %d jobs for tenant %s",
            batch_id,
            len(job_ids),
            tenant_id,
        )

        return batch_id

    async def process_batch(
        self,
        db: AsyncSession,
        batch_id: str,
        tenant_id: uuid.UUID,
        max_concurrent: int = 3,
    ) -> None:
        """Process all jobs in a batch with bounded concurrency.

        Parameters
        ----------
        db:
            Active async database session.
        batch_id:
            Identifier of the batch to process.
        tenant_id:
            Tenant that owns the batch.
        max_concurrent:
            Maximum number of jobs executing at the same time.
        """
        batch = await self._get_batch(db, batch_id)
        if batch is None:
            raise ValueError(f"Batch not found: {batch_id}")

        batch.status = "processing"
        batch.updated_at = datetime.now(UTC)
        await db.flush()
        await db.commit()

        semaphore = asyncio.Semaphore(max_concurrent)
        batch_lock = asyncio.Lock()
        job_ids: list[str] = list(batch.job_ids or [])

        async def _run_job(job_id: str) -> None:
            async with semaphore:
                try:
                    async with self._db_session_factory() as job_db:
                        await self._compiler_service.process_job(
                            job_db,
                            job_id,
                            tenant_id,
                        )
                        await job_db.commit()
                    async with batch_lock:
                        batch.completed_jobs = (batch.completed_jobs or 0) + 1
                        batch.updated_at = datetime.now(UTC)
                        await db.flush()
                except Exception as exc:
                    logger.error(
                        "Batch %s job %s failed: %s",
                        batch_id,
                        job_id,
                        exc,
                        exc_info=True,
                    )
                    async with batch_lock:
                        batch.failed_jobs = (batch.failed_jobs or 0) + 1
                        errors = list(batch.errors or [])
                        errors.append(
                            {
                                "job_id": job_id,
                                "error": format_error(exc, max_length=2000),
                            }
                        )
                        batch.errors = errors
                        batch.updated_at = datetime.now(UTC)
                        await db.flush()

        tasks = [asyncio.create_task(_run_job(jid)) for jid in job_ids]
        # ⛔ `return_exceptions=True` here is what makes a LOST job invisible, so the
        # results are reconciled below rather than discarded. A child that raised a
        # BaseException (CancelledError is the live one) never reached either counter:
        # `_run_job`'s `except Exception` cannot catch it. NOTE this does NOT swallow a
        # cancellation of THIS coroutine — `return_exceptions=True` only captures the
        # CHILDREN's exceptions; an outer cancel still raises out of the await.
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ⛔ Re-read `errors` from the database before writing the final state.
        #
        # This transaction has been open since the 'processing' flip and every
        # progress write above is a bare flush(), so a reaper CAN have committed a
        # failure into this row in the window before the first child job flushed
        # (after that the row lock is held here and the reaper skips it). If it did,
        # and nothing in this task ever reassigned `errors`, the column is simply not
        # part of the UPDATE below — so the status becomes 'completed' while the
        # reaper's "the owning process is gone" entry survives FOREVER on a batch that
        # actually succeeded. Refreshing here reads that committed value (READ
        # COMMITTED, plus this transaction's own flushed writes), and the filter drops
        # exactly the reaper's entries while keeping every real per-job error.
        await db.refresh(batch, ["errors"])
        batch.errors = [
            err for err in (batch.errors or []) if err.get("source") != _REAPER_ERROR_SOURCE
        ]

        # ── Reconcile: every job must have reached a recorded outcome ──────────
        #
        # A job is only "accounted" if it incremented completed_jobs or failed_jobs.
        # Anything else is LOST, and a lost job is a video the caller asked for and
        # never got. `GET /api/compile/batches/{id}` serves this status verbatim and
        # the orchestrator stops polling on a terminal one, so reporting 'completed'
        # over a shortfall is a silent data loss, not a cosmetic inaccuracy.
        #
        # ONE detector, and it is mechanism-specific by construction: `results` names
        # the exact job_ids whose task raised a BaseException — the CancelledError
        # class that escapes `_run_job`'s `except Exception`.
        #
        # There is deliberately NO count-shortfall backstop here. `_run_job` has exactly
        # three terminations — `completed_jobs += 1`, the `except Exception` handler's
        # `failed_jobs += 1`, and a BaseException that lands in `results` — so
        # `completed + failed + len(lost) >= total_jobs` always holds and a shortfall is
        # arithmetically unreachable. A branch that cannot fire is not a backstop; it
        # reads as coverage while testing nothing. The reachable accounting failure is
        # the OPPOSITE sign — a job counted twice — and it is handled below.
        total_jobs = len(job_ids)
        errors = list(batch.errors or [])
        for job_id, outcome in zip(job_ids, results, strict=True):
            if not isinstance(outcome, BaseException):
                continue
            logger.error(
                "Batch %s job %s was LOST (%s) and recorded no outcome",
                batch_id,
                job_id,
                type(outcome).__name__,
            )
            errors.append(
                {
                    "job_id": job_id,
                    "error": (
                        f"job ended without recording an outcome "
                        f"({type(outcome).__name__}: {format_error(outcome, max_length=2000)})"
                    ),
                    "source": _LOST_ERROR_SOURCE,
                }
            )

        completed_jobs = batch.completed_jobs or 0
        failed_jobs = batch.failed_jobs or 0

        # ── A job can be counted TWICE, and that inflates completed_jobs ───────
        #
        # `_run_job` increments `completed_jobs` and only THEN flushes. If that flush
        # raises, the `except Exception` handler increments `failed_jobs` for the SAME
        # job, so the counters sum to more than `total_jobs`. `completed_jobs` then no
        # longer proves a video was produced — its success write never landed — and
        # `completed_jobs >= total_jobs` would report 'completed' for a batch whose only
        # job failed. That is the optimistic terminal status this whole reconciliation
        # exists to remove, so the surplus is subtracted before the status is derived:
        # `verified_completed` is a LOWER BOUND on jobs that actually produced a video.
        double_counted = max(completed_jobs + failed_jobs - total_jobs, 0)
        verified_completed = completed_jobs - double_counted
        if double_counted > 0:
            logger.error(
                "Batch %s: %d of %d job(s) were counted twice (completed=%d failed=%d); "
                "at most %d job(s) actually produced a video",
                batch_id,
                double_counted,
                total_jobs,
                completed_jobs,
                failed_jobs,
                verified_completed,
            )
            errors.append(
                {
                    "job_id": None,
                    "error": (
                        f"{double_counted} of {total_jobs} job(s) were counted twice "
                        f"(completed={completed_jobs} failed={failed_jobs}); at most "
                        f"{verified_completed} job(s) produced a video"
                    ),
                    "source": _LOST_ERROR_SOURCE,
                }
            )
        batch.errors = errors

        # Terminal status is derived from what was actually PRODUCED, never from
        # failed_jobs alone: a lost job is not a success, so it can only drag the
        # status down. 'completed' now requires every job to have succeeded, counted
        # once.
        if total_jobs == 0 or verified_completed >= total_jobs:
            batch.status = "completed"
        elif verified_completed <= 0:
            batch.status = "failed"
        else:
            batch.status = "partial"

        now = datetime.now(UTC)
        batch.updated_at = now
        batch.completed_at = now
        await db.flush()
        await db.commit()

        logger.info(
            "Batch %s finished: status=%s completed=%d failed=%d",
            batch_id,
            batch.status,
            batch.completed_jobs or 0,
            batch.failed_jobs or 0,
        )

    @staticmethod
    async def get_batch_status(
        db: AsyncSession,
        batch_id: str,
    ) -> CompilationBatch | None:
        """Look up a batch record by its ``batch_id``.

        Returns
        -------
        CompilationBatch | None
            The batch record, or ``None`` if not found.
        """
        result = await db.execute(
            select(CompilationBatch).where(
                CompilationBatch.batch_id == batch_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def mark_batch_failed(
        db: AsyncSession,
        batch_id: str,
        reason: str,
    ) -> BatchReapOutcome:
        """Move a NON-TERMINAL batch to ``failed``, appending ``reason`` to ``errors``.

        The reaper's mutation path, and the recovery path for a background batch task
        that died. Commits internally, exactly like ``CompilationJobManager.mark_failed``.

        Returns which of four things actually happened, so the caller can count real
        reclaims and log an honest reason for every skip:

        ``"reaped"``    this call performed the transition.
        ``"terminal"``  the row reached ``completed`` / ``partial`` / ``failed`` first.
        ``"locked"``    another transaction holds the row lock — see below.
        ``"missing"``   no such ``batch_id``.

        ⛔ ``skip_locked=True`` is LOAD-BEARING, not a micro-optimisation.

        ``process_batch`` opens ONE transaction for the whole life of a batch and does
        not commit again between the ``'processing'`` flip and the final status write:
        every ``_run_job`` progress write is a bare ``flush()``. So from the moment the
        first child job finishes, the owning transaction holds this row's lock until the
        batch ends — for hours, legitimately. A plain ``FOR UPDATE`` here would block on
        that lock, and because ``_reap_stale_batches`` is awaited INLINE in the cleanup
        loop it would take ``_cleanup_cycle`` and ``_reap_stale_jobs`` down with it: a
        fix for ONE immortal row wedging the sweeper that reaps all the others. There is
        no ``lock_timeout`` anywhere in ``src/`` to bound that wait, and adding one would
        only convert an indefinite block into a per-row stall plus an error.

        ``SKIP LOCKED`` is also the semantically RIGHT answer, not merely the
        non-blocking one: a held row lock is positive evidence that a live transaction
        still owns this batch, which is exactly the case the reaper must not touch. The
        row is simply retried next sweep.

        The status is still re-checked under the lock we DO get: without it, a sweep
        racing the owning task's final flush could overwrite a batch that had JUST
        legitimately completed, turning a success into a reported failure.

        Entries written here are tagged ``"source": "reaper"`` so ``process_batch`` can
        strip them if it finishes after all — see the finalisation block there.
        """
        result = await db.execute(
            select(CompilationBatch)
            .where(CompilationBatch.batch_id == batch_id)
            .with_for_update(skip_locked=True)
        )
        batch = result.scalar_one_or_none()
        if batch is None:
            # SKIP LOCKED cannot distinguish "no such row" from "someone else holds the
            # lock" — both come back empty. Classify with an unlocked read so the
            # caller's log names the real reason instead of guessing "already terminal".
            present = (
                await db.execute(
                    select(CompilationBatch.status).where(CompilationBatch.batch_id == batch_id)
                )
            ).scalar_one_or_none()
            return "missing" if present is None else "locked"
        if batch.status not in _NON_TERMINAL_BATCH_STATUSES:
            return "terminal"

        errors = list(batch.errors or [])
        errors.append({"error": reason, "source": _REAPER_ERROR_SOURCE})
        batch.errors = errors
        batch.status = "failed"
        now = datetime.now(UTC)
        batch.updated_at = now
        batch.completed_at = now
        await db.commit()
        return "reaped"

    @staticmethod
    async def delete_batches(db: AsyncSession, batch_ids: list[str]) -> int:
        """Delete batch rows by ``batch_id``; returns the number removed.

        The prune backstop's mutation path. Runs tenant-scoped like every other
        mutation here, so RLS still applies under enforce.
        """
        if not batch_ids:
            return 0
        result = await db.execute(
            delete(CompilationBatch).where(CompilationBatch.batch_id.in_(batch_ids))
        )
        await db.commit()
        # ``AsyncSession.execute`` is typed as returning ``Result[Any]``, which has no
        # ``rowcount``; only the ``CursorResult`` a DML statement actually returns
        # does. ``getattr`` keeps the count honest without a cast that would assert a
        # runtime type the annotation does not promise — and a 0 here would only
        # under-report the prune log line, never delete the wrong rows.
        return int(getattr(result, "rowcount", 0) or 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_batch(
        db: AsyncSession,
        batch_id: str,
    ) -> CompilationBatch | None:
        result = await db.execute(
            select(CompilationBatch).where(
                CompilationBatch.batch_id == batch_id,
            )
        )
        return result.scalar_one_or_none()
