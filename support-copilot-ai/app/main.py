from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.logger import get_logger


logger = get_logger()

app = FastAPI(title="SupportCopilot AI (Foundation)")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """
    Basic health check endpoint.
    """
    return {"status": "ok"}

