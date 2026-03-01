"""
Модуль моделей эмбеддингов.

Предоставляет абстракцию для различных моделей эмбеддингов
с возможностью легкой замены через Factory паттерн.
"""

from .base import BaseEmbeddingModel, EmbeddingConfig
from .nomic_embed import NomicEmbedModel, create_nomic_embed_model
from .remote_embed import OllamaEmbedModel, create_ollama_embed_model
from .factory import EmbeddingModelFactory, EmbeddingModelType
from .code_embedder import CodeEmbedder, create_code_embedder

__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "NomicEmbedModel",
    "create_nomic_embed_model",
    "EmbeddingModelFactory",
    "EmbeddingModelType",
    "CodeEmbedder",
    "create_code_embedder",
    "OllamaEmbedModel",
    "create_ollama_embed_model",
]
