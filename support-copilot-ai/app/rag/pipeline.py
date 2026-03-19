from __future__ import annotations

from typing import Optional, Sequence

from app.embeddings.vector_store import SearchResult, SourceType, VectorStore
from app.rag.generator import GenerationResult, Generator
from app.rag.retriever import RetrievedContext, Retriever
from app.services.llm_service import LLMService


class RagPipeline:
    """
    Full RAG pipeline: retrieve -> generate.
    """

    def __init__(self, *, vector_store: VectorStore, llm: LLMService) -> None:
        self.retriever = Retriever(vector_store)
        self.generator = Generator(llm)

    def run(
        self,
        *,
        query: str,
        top_k: int,
        source_types: Optional[Sequence[SourceType]] = None,
    ) -> GenerationResult:
        """
        Run the RAG pipeline.
        """
        retrieved: RetrievedContext = self.retriever.retrieve(query, top_k=top_k, source_types=source_types)
        return self.generator.generate(query=query, retrieved_results=retrieved.results)

