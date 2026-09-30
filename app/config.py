"""Application settings loaded from environment variables."""

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.chunking import DEFAULT_CHUNKING_VERSION, validate_chunking_config
from app.prompting import (
    PRODUCTION_PROMPT_ID,
    PromptResolutionError,
    resolve_prompt_identity,
)

EMBEDDING_DIMENSION = 1024


class DatabaseSettings(BaseSettings):
    """Settings required by database tooling and sessions."""

    database_url: str = "postgresql+psycopg://ddrag:ddrag_dev_password@127.0.0.1:55432/ddrag"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Settings(DatabaseSettings):
    """Settings required by the application."""

    app_name: str = "DDRAG"
    app_env: str = "development"
    log_level: str = "INFO"
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    document_storage_path: str = "storage/documents"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    chunk_size_chars: int = 1000
    chunk_overlap_chars: int = 200
    chunking_version: str = DEFAULT_CHUNKING_VERSION
    embedding_model: str = "qwen3-embedding:0.6b"
    generation_model: str = "qwen2.5:1.5b"
    generation_prompt_id: str = PRODUCTION_PROMPT_ID
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_dimension: int = EMBEDDING_DIMENSION
    retrieval_default_top_k: int = 5

    @field_validator("generation_prompt_id")
    @classmethod
    def _validate_generation_prompt_id(cls, value: str) -> str:
        normalized_id = value.strip()
        try:
            resolve_prompt_identity(normalized_id)
        except PromptResolutionError as exc:
            raise ValueError(str(exc)) from exc
        return normalized_id

    @model_validator(mode="after")
    def _validate_chunking_settings(self) -> "Settings":
        validate_chunking_config(self.chunk_size_chars, self.chunk_overlap_chars)
        if not self.chunking_version.strip():
            raise ValueError("chunking_version must not be blank")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be blank")
        if not self.generation_model.strip():
            raise ValueError("generation_model must not be blank")
        candidate_base_url = self.ollama_base_url.strip()
        if not candidate_base_url:
            raise ValueError("ollama_base_url must not be blank")
        parsed = urlparse(candidate_base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("ollama_base_url must include scheme and host")
        if self.embedding_dimension != EMBEDDING_DIMENSION:
            raise ValueError(f"embedding_dimension must be {EMBEDDING_DIMENSION}")
        if not 1 <= self.retrieval_default_top_k <= 50:
            raise ValueError("retrieval_default_top_k must be between 1 and 50")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()
