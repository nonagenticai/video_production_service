"""API routes for video_production_service.

Imports both recording and compilation routers for inclusion in the FastAPI app.
"""

from src.api.routes.compilation import router as compilation_router
from src.api.routes.compilation import router_ws as compilation_ws_router
from src.api.routes.recording import router as recording_router

__all__ = [
    "compilation_router",
    "compilation_ws_router",
    "recording_router",
]
