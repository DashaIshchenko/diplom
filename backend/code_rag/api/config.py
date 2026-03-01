"""
Конфигурация API сервера.
"""

from functools import lru_cache
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Настройки приложения."""

    # API настройки
    api_title: str = "Code RAG API"
    api_version: str = "0.1.0"
    api_description: str = "API для индексации и поиска кода с RAG"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # CORS
    cors_origins: List[str] = ["*"]

    # Embeddings
    embedding_model: str = Field(default="nomic-embed-text:latest", env="EMBEDDING_MODEL")
    embedding_device: str = "cpu"
    embedding_base_url: str = Field(default="http://ai.parma.ru/local", env="EMBEDDING_BASE_URL")
    embedding_timeout: int = Field(default=60, env="EMBEDDING_TIMEOUT")
    embedding_request_timeout: int = 30
    embedding_retry_attempts: int = 3
    embedding_backoff_factor: int = 1
    embedding_normalize: bool = True
    embedding_dimension: int = Field(default=768, env="EMBEDDING_DIMENSION")
    embedding_max_tokens: int = Field(default=8192, env="EMBEDDING_MAX_TOKENS")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_timeout: int = 30
    collection_name: str = Field(default="code_collection", env="COLLECTION_NAME")

    # RAG
    use_reranking: bool = True
    default_top_k: int = 10

    # Qwen
    qwen_api_key: Optional[str] = Field(default=None, env="QWEN_API_KEY")
    qwen_base_url: Optional[str] = Field(default="https://ai.parma.ru/cloud/v1", env="QWEN_BASE_URL")
    qwen_model: str = Field(default="Qwen/Qwen3-Coder-30B-A3B-Instruct", env="QWEN_MODEL")
    qwen_temperature: float = Field(default=0.4, env="QWEN_TEMPERATURE")
    qwen_max_tokens: int = Field(default=8096, env="QWEN_MAX_TOKENS")

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Получение настроек (кэшируется).

    Returns:
        Settings instance
    """
    return Settings()
