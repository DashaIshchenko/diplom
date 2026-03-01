"""
FastAPI приложение для Code RAG.
"""

from .app import app, create_app
from .config import Settings, get_settings
from .dependencies import (
    get_embedder,
    get_qdrant_client,
    get_rag_retriever,
    get_qwen,
    cleanup_dependencies,
)

__all__ = [
    "app",
    "create_app",
    "Settings",
    "get_settings",
    "get_embedder",
    "get_qdrant_client",
    "get_rag_retriever",
    "get_qwen",
    "cleanup_dependencies",
]
