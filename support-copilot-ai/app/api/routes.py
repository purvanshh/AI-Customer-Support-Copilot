from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.api.schemas import (
    AnalyticsResponse,
    FeedbackRequest,
    FeedbackResponse,
    GenerateResponseRequest,
    GenerateResponseResponse,
    UploadDocsResponse,
    UploadTicketsResponse,
)
from app.services.ingestion_service import ingest_docs, ingest_tickets
from app.services.rag_service import RagService
from app.services.feedback_service import FeedbackService
from app.services.analytics_service import AnalyticsService


router = APIRouter()
rag_service = RagService()
feedback_service = FeedbackService()
analytics_service = AnalyticsService()


@router.post("/upload-tickets", response_model=UploadTicketsResponse)
async def upload_tickets(file: UploadFile = File(...)) -> UploadTicketsResponse:
    """
    Upload a CSV/JSON dataset of historical tickets.
    """
    return await ingest_tickets(file=file)


@router.post("/upload-docs", response_model=UploadDocsResponse)
async def upload_docs(file: UploadFile = File(...)) -> UploadDocsResponse:
    """
    Upload a document (.txt or .pdf) for knowledge-base ingestion.
    """
    return await ingest_docs(file=file)


@router.post("/generate-response", response_model=GenerateResponseResponse)
async def generate_response(req: GenerateResponseRequest) -> GenerateResponseResponse:
    """
    Generate an AI response using RAG over ingested tickets + documents.
    """
    return rag_service.generate_response(req)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """
    Store human feedback for an AI-generated response.
    """
    corrected = req.corrected_response if req.corrected_response is not None else ""
    fid = feedback_service.save_feedback(
        query=req.query,
        ai_response=req.ai_response,
        corrected_response=corrected,
        rating=req.rating,
    )
    return FeedbackResponse(id=fid)


@router.get("/analytics", response_model=AnalyticsResponse)
async def analytics() -> AnalyticsResponse:
    """
    Compute analytics derived from user feedback.
    """
    return analytics_service.compute()

