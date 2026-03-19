from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.api.schemas import GenerateResponseRequest, GenerateResponseResponse, UploadDocsResponse, UploadTicketsResponse
from app.services.ingestion_service import ingest_docs, ingest_tickets
from app.services.rag_service import RagService


router = APIRouter()
rag_service = RagService()


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

