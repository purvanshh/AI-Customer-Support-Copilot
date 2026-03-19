from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TicketRecord(BaseModel):
    """
    Normalized ticket record stored after ingestion.
    """

    id: str = Field(..., description="Unique ticket identifier.")
    query: str = Field(..., description="Customer question / problem statement.")
    response: Optional[str] = Field(None, description="Agent response (optional).")
    timestamp: Optional[str] = Field(None, description="Event timestamp (optional).")


class UploadTicketsResponse(BaseModel):
    """
    Response returned by `POST /upload-tickets`.
    """

    message: str
    raw_file_path: str
    processed_file_path: str
    records_received: int
    records_ingested: int


class UploadDocsResponse(BaseModel):
    """
    Response returned by `POST /upload-docs`.
    """

    message: str
    raw_file_path: str
    processed_file_path: str
    chunks_saved: int


class ApiError(BaseModel):
    """
    Standard error payload for documentation/testing convenience.
    """

    detail: str
    error_type: Literal["invalid_file", "empty_file", "unsupported_format", "parsing_error"]
    extra: Optional[dict[str, Any]] = None

