from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.core.logger import configure_logger
from app.api.schemas import (
    AnalyticsResponse,
    FeedbackRequest,
    FeedbackResponse,
    UploadDocsResponse,
    QueryRequest,
    QueryResponse,
    UploadTicketsResponse,
)
from app.rag.pipeline import (
    get_analytics,
    query_support,
    submit_feedback,
    upload_docs,
    upload_tickets,
)

logger = configure_logger()

router = APIRouter()


@router.post("/upload-tickets", response_model=UploadTicketsResponse)
async def upload_tickets_endpoint(
    file: UploadFile = File(...),
):
    # FastAPI's UploadFile is a streaming interface; we persist it to a temp file for
    # parsers that expect a real filepath.
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    logger.info("Ingesting tickets file: %s", file.filename)
    return await upload_tickets(tmp_path=tmp_path, original_filename=file.filename)


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    return await query_support(req)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
    return await submit_feedback(req)


@router.post("/upload-docs", response_model=UploadDocsResponse)
async def upload_docs_endpoint(docs: list[UploadFile] = File(...)) -> UploadDocsResponse:
    paths: list[tuple[str, str]] = []
    for d in docs:
        suffix = Path(d.filename or "").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await d.read()
            tmp.write(content)
            paths.append((tmp.name, d.filename or "document"))

    return await upload_docs(doc_paths=paths)


@router.get("/analytics", response_model=AnalyticsResponse)
async def analytics_endpoint() -> AnalyticsResponse:
    return await get_analytics()

