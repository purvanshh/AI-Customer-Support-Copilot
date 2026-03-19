from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.db import init_db
from app.core.logger import configure_logger

logger = configure_logger()
settings = get_settings()

app = FastAPI(title="SupportCopilot AI")

app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    settings.resolve_paths()
    init_db()
    logger.info("Startup complete. DB initialized at %s", settings.db_path)

