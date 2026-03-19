import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Storage
    data_dir: str = "data"
    log_dir: str = "logs"
    db_path: str | None = None
    faiss_index_path: str | None = None
    faiss_meta_path: str | None = None

    # Embeddings
    embedding_backend: str = "sentence_transformers"  # sentence_transformers | openai | mock
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLMs
    llm_backend: str = "mock"  # openai | gemini | mock
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2
    openai_max_tokens: int = 400

    gemini_model: str = "gemini-1.5-pro"
    gemini_temperature: float = 0.2

    # Retrieval
    top_k_default: int = 5
    max_context_chars: int = 8000

    # Chunking (for docs)
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 150

    # CORS
    cors_allow_origins: str = "*"  # comma-separated

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"

    def resolve_paths(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        processed_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        self.db_path = self.db_path or os.path.join(self.data_dir, "supportcopilot.db")
        self.faiss_index_path = self.faiss_index_path or os.path.join(processed_dir, "faiss.index")
        self.faiss_meta_path = self.faiss_meta_path or os.path.join(processed_dir, "faiss_meta.json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.resolve_paths()
    return s

