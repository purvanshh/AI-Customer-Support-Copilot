from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from fastapi import HTTPException

from app.api.schemas import GenerateResponseRequest, GenerateResponseResponse, SourceReference
from app.core.config import data_dirs, get_settings
from app.core.logger import get_logger
from app.embeddings.vector_store import SourceType, VectorDocument, VectorStore
from app.rag.pipeline import RagPipeline
from app.services.classification_service import ClassificationService
from app.services.llm_service import LLMService
from app.services.feedback_service import FeedbackService


logger = get_logger()


class RagService:
    """
    Orchestrates:
      - building/loading the vector index from /data/processed/
      - running the RAG pipeline
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.llm = LLMService()
        self.pipeline = RagPipeline(vector_store=self.vector_store, llm=self.llm)
        self.classifier = ClassificationService()
        self.feedback_service = FeedbackService()

    def _processed_inputs_manifest(self) -> dict[str, Any]:
        """
        Build a manifest of processed input files (tickets + docs).
        Used for index staleness checks.
        """
        _, processed_dir = data_dirs()
        tickets_files = sorted(processed_dir.glob("tickets_processed_*.json"))
        docs_meta_files = sorted(processed_dir.glob("docs_processed_*.json"))
        docs_txt_files = sorted(processed_dir.glob("docs_processed_*.txt"))

        def mtimes(files: Iterable[Path]) -> dict[str, float]:
            out: dict[str, float] = {}
            for f in files:
                try:
                    out[str(f)] = f.stat().st_mtime
                except FileNotFoundError:
                    continue
            return out

        return {
            "tickets_processed_files": [str(p) for p in tickets_files],
            "docs_processed_meta_files": [str(p) for p in docs_meta_files],
            "docs_processed_txt_files": [str(p) for p in docs_txt_files],
            "input_mtimes": {
                **mtimes(tickets_files),
                **mtimes(docs_meta_files),
                **mtimes(docs_txt_files),
            },
        }

    def _load_ticket_documents(self, tickets_file: Path) -> list[VectorDocument]:
        """
        Convert processed tickets JSON into vector documents.
        """
        items = json.loads(tickets_file.read_text(encoding="utf-8"))
        documents: list[VectorDocument] = []

        for it in items:
            ticket_id = str(it.get("id", "")).strip()
            query = str(it.get("query", "")).strip()
            response = it.get("response", None)
            timestamp = it.get("timestamp", None)
            if not ticket_id or not query:
                continue

            response_s = str(response).strip() if response not in (None, "") else ""
            timestamp_s = str(timestamp).strip() if timestamp not in (None, "") else ""
            combined_text = (
                f"Ticket ID: {ticket_id}\n"
                f"Timestamp: {timestamp_s}\n"
                f"Customer Query: {query}\n"
                f"Agent Response: {response_s}\n"
            ).strip()

            documents.append(
                VectorDocument(source_type="ticket", source_ref=f"{ticket_id}", text=combined_text)
            )

        return documents

    def _load_doc_documents_from_meta(self, docs_meta_file: Path) -> list[VectorDocument]:
        """
        Convert structured docs metadata JSON into vector documents.
        """
        chunk_entries = json.loads(docs_meta_file.read_text(encoding="utf-8"))
        source_doc = docs_meta_file.stem.replace("docs_processed_", "")
        documents: list[VectorDocument] = []

        for entry in chunk_entries:
            source_filename = str(entry.get("source_filename", "")).strip() or source_doc
            chunk_index = int(entry.get("chunk_index", 0))
            chunk_text = str(entry.get("chunk_text", "")).strip()
            if not chunk_text:
                continue

            documents.append(
                VectorDocument(
                    source_type="doc",
                    source_ref=f"{source_filename}#{chunk_index}",
                    text=chunk_text,
                )
            )

        return documents

    def _load_doc_documents_from_txt(self, docs_txt_file: Path) -> list[VectorDocument]:
        """
        Fallback doc conversion for older ingest outputs (no JSON metadata).
        """
        txt = docs_txt_file.read_text(encoding="utf-8")
        chunks = [c.strip() for c in txt.split("\n\n") if c.strip()]
        documents: list[VectorDocument] = []
        for i, c in enumerate(chunks):
            documents.append(
                VectorDocument(
                    source_type="doc",
                    source_ref=f"{docs_txt_file.stem}#{i}",
                    text=c,
                )
            )
        return documents

    def _build_documents_from_processed(self) -> tuple[list[VectorDocument], dict[str, Any]]:
        """
        Read /data/processed and convert into vector documents.
        Returns:
            (documents, manifest)
        """
        manifest = self._processed_inputs_manifest()
        _, processed_dir = data_dirs()

        documents: list[VectorDocument] = []
        seen_refs: set[str] = set()

        # Tickets
        for tf in manifest["tickets_processed_files"]:
            path = Path(tf)
            docs = self._load_ticket_documents(path)
            for d in docs:
                key = f"{d.source_type}:{d.source_ref}"
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                documents.append(d)

        # Docs (metadata first)
        if manifest["docs_processed_meta_files"]:
            for df in manifest["docs_processed_meta_files"]:
                path = Path(df)
                docs = self._load_doc_documents_from_meta(path)
                for d in docs:
                    key = f"{d.source_type}:{d.source_ref}"
                    if key in seen_refs:
                        continue
                    seen_refs.add(key)
                    documents.append(d)
        else:
            for df in manifest["docs_processed_txt_files"]:
                path = Path(df)
                docs = self._load_doc_documents_from_txt(path)
                for d in docs:
                    key = f"{d.source_type}:{d.source_ref}"
                    if key in seen_refs:
                        continue
                    seen_refs.add(key)
                    documents.append(d)

        return documents, manifest

    def _should_rebuild(self, *, force_rebuild_index: bool) -> bool:
        """
        Determine whether to rebuild the index.
        """
        if force_rebuild_index:
            return True
        if not self.vector_store.is_index_ready():
            return True
        if not self.settings.auto_rebuild_index:
            return False

        # Compare staleness via metadata manifest.
        manifest = self._processed_inputs_manifest()
        try:
            meta_path = Path(self.settings.faiss_meta_path)
            if not meta_path.exists():
                return True
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            stored_mtimes = stored.get("built_from", {}).get("input_mtimes", {})
            current_mtimes = manifest.get("input_mtimes", {})
            return stored_mtimes != current_mtimes
        except Exception:
            # If anything goes wrong, prefer rebuild to avoid serving stale retrieval.
            return True

    def ensure_index_built(self, *, force_rebuild_index: bool = False) -> None:
        """
        Ensure the vector index exists and is up to date.
        """
        should_rebuild = self._should_rebuild(force_rebuild_index=force_rebuild_index)
        if not should_rebuild:
            return

        documents, manifest = self._build_documents_from_processed()
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No processed tickets/docs found. Upload tickets and docs first.",
            )

        logger.info("Building vector index from processed inputs (documents=%s)...", len(documents))
        self.vector_store.build_index(documents, built_from=manifest)
        # Load index in memory for subsequent searches.
        self.vector_store.load()

    def generate_response(self, req: GenerateResponseRequest) -> GenerateResponseResponse:
        """
        Generate an AI response using RAG.
        """
        classification = self.classifier.classify(req.query)

        # Ensure retrieval assets exist (vector index).
        self.ensure_index_built(force_rebuild_index=req.force_rebuild_index)

        top_k = int(req.top_k or self.settings.rag_top_k_default)
        source_types = req.source_types

        # 1) Feedback adaptation: if we have a high-rated similar query, prioritize the corrected response.
        feedback_match = self.feedback_service.find_best_match(req.query)
        if feedback_match is not None:
            retrieved = self.vector_store.similarity_search(
                req.query,
                top_k=top_k,
                source_types=source_types,
            )
            sources = [
                SourceReference(
                    source_type=s.source_type,
                    source_ref=s.source_ref,
                    score=s.score,
                    snippet=s.text[:400].strip(),
                )
                for s in retrieved
            ]

            logger.info(
                "RAG feedback override. query=%r category=%s match_similarity=%.3f rating=%s",
                req.query,
                classification.label,
                feedback_match.similarity,
                feedback_match.rating,
            )

            return GenerateResponseResponse(
                response=feedback_match.corrected_response,
                sources=sources,
                confidence=feedback_match.similarity,
                category=classification.label,
            )

        # 2) Normal path: retrieve -> generate.
        generation = self.pipeline.run(query=req.query, top_k=top_k, source_types=source_types)

        # Debug visibility: show retrieved chunks + scores.
        src_preview = [
            {"source_type": s.source_type, "source_ref": s.source_ref, "score": round(s.score, 4)}
            for s in generation.sources[: min(5, len(generation.sources))]
        ]
        logger.info(
            "RAG generated response. query=%r category=%s confidence=%.3f top_sources=%s",
            req.query,
            classification.label,
            generation.confidence,
            src_preview,
        )

        sources = [
            SourceReference(
                source_type=s.source_type,
                source_ref=s.source_ref,
                score=s.score,
                snippet=s.text[:400].strip(),
            )
            for s in generation.sources
        ]

        return GenerateResponseResponse(
            response=generation.response,
            sources=sources,
            confidence=generation.confidence,
            category=classification.label,
        )

