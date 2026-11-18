"""Shared helpers and background-task wrappers for compilation routes.

Tenant provenance in the ``_bg`` wrappers
-----------------------------------------
Both wrappers below are dispatched with ``asyncio.create_task`` from a route
handler (``compilation.py`` and ``_compilation_extras.py``), each carrying ONE
tenant taken from request state. They seed ``set_identity_context`` so the
Phase-D RLS bind (``after_begin`` re-binds every transaction) applies to the
fresh session they open, and they OMIT ``tenant_source`` deliberately.

``asyncio.create_task`` COPIES the caller's context, so ``AuthMiddleware``'s
provenance is ALREADY PRESENT inside these tasks. Since w2v-common 0.67.7 the
``tenant_source`` parameter carries a ``KEEP_TENANT_SOURCE`` sentinel default,
so omitting it means "keep what I inherited, unless the tenant changes":

* omitted + SAME tenant as the ambient context → provenance PRESERVED
* omitted + DIFFERENT tenant                   → provenance DROPPED to ``None``
* ``tenant_source=None`` explicitly            → cleared
* ``tenant_source=<value>`` explicitly         → ALWAYS wins

These wrappers rebind the SAME tenant they inherited, so omission preserves the
request's provenance and every metered outbound hop stays vouched. Passing
``tenant_source=get_tenant_source_from_context()`` would be strictly LESS safe:
an explicit value always wins, so it bypasses the different-tenant drop that is
the library's safety net. Do NOT delete the seeding call either — it exists for
the RLS bind, and removing it breaks multi-commit jobs.

The bare call is also correct in a reaper/worker loop that iterates rows for
DIFFERENT tenants — see ``src/services/compilation/cleanup_service.py``, where
the tenant-change drop clears provenance per row automatically.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import Request
from w2v_common.auth.identity_context import set_identity_context

from src.api.tenant import require_request_tenant

if TYPE_CHECKING:
    from src.services.compilation.batch_service import BatchService
    from src.services.compilation.compiler_service import CompilerService
    from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

# Track fire-and-forget background tasks to prevent garbage collection.
_background_tasks: set[asyncio.Task] = set()


def _get_db_session() -> Any:
    """Late-bound accessor for ``get_db_session``.

    Tests patch ``src.api.routes.compilation.get_db_session``; resolving the
    lookup via the ``compilation`` module ensures helpers see the patch too.
    """
    from src.api.routes import compilation

    return compilation.get_db_session


def _get_tenant_id(request: Request) -> uuid.UUID:
    """Return the caller's tenant UUID, or raise 401. No "anonymous" fallback."""
    return require_request_tenant(request)


def _get_compiler_service(request: Request) -> CompilerService:
    """Retrieve the CompilerService instance from the application state."""
    return request.app.state.compiler_service


def _get_storage(request: Request) -> StorageService:
    """Retrieve the StorageService instance from the application state."""
    return request.app.state.storage_service


async def _process_job_bg(
    service: CompilerService,
    job_id: str,
    tenant_id: uuid.UUID,
) -> None:
    """Background task wrapper that creates its own DB session for job processing."""
    # Phase D RLS: bind the tenant for this bg task's sessions (after_begin re-binds each txn).
    # tenant_source is OMITTED on purpose — see the module docstring. Omission keeps the
    # provenance this task inherited (create_task copies the request's context) because the
    # tenant is unchanged, and would drop it to None if the tenant ever differed. Passing
    # tenant_source= explicitly always wins and would bypass that drop.
    set_identity_context(tenant_id, None, None)
    progress_callback = None
    try:
        from src.api.ws import ProgressReporter

        ws_manager = getattr(service, "_ws_manager", None)
        if ws_manager:
            progress_callback = ProgressReporter(ws_manager, job_id)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        logger.warning("Failed to set up WebSocket progress reporter: %s", exc)
    try:
        async with _get_db_session()() as db:
            await service.process_job(db, job_id, tenant_id, progress_callback=progress_callback)
    except asyncio.CancelledError:
        logger.info("Job %s was cancelled", job_id)
    except OSError, ValueError, RuntimeError, TimeoutError:
        logger.exception("Background compilation job %s failed", job_id)
    finally:
        service.unregister_task(job_id)


async def _process_batch_bg(
    batch_svc: BatchService,
    batch_id: str,
    tenant_id: uuid.UUID,
    max_concurrent: int,
) -> None:
    """Background task wrapper that creates its own DB session for batch processing."""
    # Phase D RLS: bind the tenant for this bg task's sessions (after_begin re-binds each txn).
    # tenant_source is OMITTED on purpose — see the module docstring. Omission keeps the
    # provenance this task inherited (create_task copies the request's context) because the
    # tenant is unchanged, and would drop it to None if the tenant ever differed. Passing
    # tenant_source= explicitly always wins and would bypass that drop.
    set_identity_context(tenant_id, None, None)
    try:
        async with _get_db_session()() as db_session:
            await batch_svc.process_batch(db_session, batch_id, tenant_id, max_concurrent)
    except Exception as exc:
        # ⛔ NOT a curated tuple. This guard used to read
        # `except (OSError, ValueError, RuntimeError, TimeoutError)`, and
        # `SQLAlchemyError` inherits from NONE of those — so the single most likely
        # failure of a function whose entire body is database work escaped the
        # handler, the batch was never marked `failed`, and the row stayed
        # 'processing' forever with the client polling it. A curated tuple here
        # encodes "the failures I thought of"; the only correct net for a
        # fire-and-forget task that OWNS a row's terminal state is everything.
        # `asyncio.CancelledError` is a BaseException and still propagates, so a
        # SIGTERM is not misreported as a batch failure — the reaper collects that
        # row instead (src/services/compilation/cleanup_service.py).
        logger.exception("Background batch %s failed", batch_id)
        try:
            async with _get_db_session()() as db_session:
                from datetime import UTC, datetime

                from sqlalchemy import select

                from src.database.models import CompilationBatch

                result = await db_session.execute(
                    select(CompilationBatch).where(CompilationBatch.batch_id == batch_id)
                )
                batch = result.scalar_one_or_none()
                if batch is not None:
                    batch.status = "failed"
                    batch.errors = [{"error": str(exc)}]
                    batch.updated_at = datetime.now(UTC)
                    await db_session.commit()
        except Exception:
            # Same defect, same fix: the recovery write is itself pure database work,
            # so a curated tuple that excludes SQLAlchemyError made the LAST-CHANCE
            # handler unable to catch the error it exists for. Nothing may escape a
            # background task's recovery path.
            logger.exception("Failed to mark batch %s as failed", batch_id)
