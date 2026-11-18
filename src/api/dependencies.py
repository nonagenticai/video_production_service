"""FastAPI dependency injection providers."""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.database.session import get_db

from src.services.compilation.job_manager import CompilationJobManager
from src.services.recording.job_manager import RecordingJobManager


async def get_recording_job_manager(db: AsyncSession = Depends(get_db)) -> RecordingJobManager:
    """Provide a RecordingJobManager instance wrapping the request-scoped DB session."""
    return RecordingJobManager(db)


async def get_compilation_job_manager(
    db: AsyncSession = Depends(get_db),
) -> type[CompilationJobManager]:
    """Provide the CompilationJobManager class for static method access.

    The CompilationJobManager uses static methods that accept a DB session
    parameter, so we return the class itself rather than an instance.
    """
    return CompilationJobManager
