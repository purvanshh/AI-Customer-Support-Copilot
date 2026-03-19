from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.api.schemas import UploadDocsResponse, UploadTicketsResponse
from app.services.ingestion_service import ingest_docs, ingest_tickets


router = APIRouter()


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

