"""Shared structural typing protocols for service-layer callbacks.

These protocols allow service-layer code to depend on a callback contract
without importing concrete classes from the API/transport layer (e.g.,
``src.api.ws.ProgressReporter``). This keeps the dependency arrow pointing
from API → services, not the reverse.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressReporterProtocol(Protocol):
    """Structural protocol for a job progress reporter.

    Implemented by ``src.api.ws.ProgressReporter``. Used by the compilation
    service to emit ``progress``, ``complete``, and ``error`` events to
    interested subscribers (typically WebSocket clients).
    """

    async def report(
        self,
        stage: str,
        progress_pct: float,
        message: str = "",
        current_part: int | None = None,
        total_parts: int | None = None,
    ) -> None:
        """Broadcast an in-flight progress update."""
        ...

    async def complete(self) -> None:
        """Broadcast a terminal completion event."""
        ...

    async def error(self, message: str) -> None:
        """Broadcast a terminal error event."""
        ...
