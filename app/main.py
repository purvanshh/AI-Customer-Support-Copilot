from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.db import init_db
from app.core.logger import configure_logger

logger = configure_logger()
settings = get_settings()

app = FastAPI(
    title="SupportCopilot AI",
    description="AI-powered customer support copilot with RAG, feedback learning, and analytics.",
    version="1.0.0",
)

# CORS — allows the Streamlit frontend (or any client) to talk to the API.
origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    settings.resolve_paths()
    init_db()
    logger.info("Startup complete. DB initialized at %s", settings.db_path)
