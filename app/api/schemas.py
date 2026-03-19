from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class UploadTicketsResponse(BaseModel):
    tickets_ingested: int
    vector_items_added: int
    index_rebuilt: bool


class UploadDocsResponse(BaseModel):
    docs_processed: int
    chunks_added: int
    index_rebuilt: bool


class QueryRequest(BaseModel):
    customer_query: str = Field(..., min_length=1)
    top_k: Optional[int] = None
    tags: Optional[list[str]] = None
    llm_override: Optional[str] = None  # optional: openai | gemini | mock


class SourceReference(BaseModel):
    source_type: Literal["ticket", "doc"]
    source_ref: str
    tags: Optional[list[str]] = None
    score: float


class QueryResponse(BaseModel):
    generation_id: int
    suggestion_text: str
    confidence: float
    sources: list[SourceReference] = []
    model_used: str


class FeedbackRequest(BaseModel):
    generation_id: int
    user_action: Literal["accepted", "rejected", "corrected"]
    corrected_text: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: Literal["ok"]


class AnalyticsResponse(BaseModel):
    tickets_processed: int
    percent_auto_resolved: float
    avg_response_time_ms: float
    accuracy: float
    feedback_count: int

