"""
Зависимости FastAPI для dependency injection.
"""

from typing import Generator, Optional
import logging

from ..core.embeddings import CodeEmbedder, EmbeddingModelType
from ..core.vector_db import QdrantClient
from ..core.rag import RAGRetriever, QwenIntegration
from ..core.git_handler.repository import RepositoryManager
from ..config import get_settings, Settings

logger = logging.getLogger(__name__)

# ==================== Singleton instances ====================

_embedder_instance: Optional[CodeEmbedder] = None
_qdrant_client_instance: Optional[QdrantClient] = None
_rag_retriever_instance: Optional[RAGRetriever] = None
_qwen_instance: Optional[QwenIntegration] = None


# ==================== Embedder ====================

def get_embedder() -> CodeEmbedder:
    """
    Получение экземпляра CodeEmbedder (singleton).

    Returns:
        CodeEmbedder instance
    """
    global _embedder_instance

    if _embedder_instance is None:
        settings = get_settings()

        logger.info(f"Инициализация CodeEmbedder: {settings.embedding_model}")

        _embedder_instance = CodeEmbedder(
            model_type=EmbeddingModelType.REMOTE,
            model_name=settings.embedding_model,
            base_url=settings.embedding_base_url,
            dimension=settings.embedding_dimension,
            max_tokens=settings.embedding_max_tokens,
            timeout=settings.embedding_timeout,
            request_timeout=settings.embedding_request_timeout,
            retry_attempts=settings.embedding_retry_attempts,
            backoff_factor=settings.embedding_backoff_factor,
            normalize=settings.embedding_normalize
        )

        logger.info(f"CodeEmbedder инициализирован, dim={_embedder_instance.embedding_dim}")

    return _embedder_instance


# ==================== Qdrant Client ====================

def get_qdrant_client() -> Generator[QdrantClient, None, None]:
    """Получение QdrantClient instance."""
    settings = get_settings()
    from ..core.vector_db import QdrantClient
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        use_https=settings.qdrant_use_https
    )
    yield client


# ==================== RAG Retriever ====================

def get_rag_retriever() -> RAGRetriever:
    """
    Получение экземпляра RAGRetriever (singleton).

    Returns:
        RAGRetriever instance
    """
    global _rag_retriever_instance

    if _rag_retriever_instance is None:
        settings = get_settings()
        embedder = get_embedder()
        # Исправлено: получаем клиент через зависимость
        from .dependencies import get_qdrant_client
        qdrant_client = next(get_qdrant_client())

        logger.info(f"Инициализация RAGRetriever для коллекции: {settings.collection_name}")

        _rag_retriever_instance = RAGRetriever(
            collection_name=settings.qdrant_collection_name,
            embedder=embedder,
            qdrant_client=qdrant_client,
            use_reranking=settings.use_reranking
        )

    return _rag_retriever_instance


# ==================== Qwen Integration ====================

def get_qwen() -> QwenIntegration:
    """
    Получение экземпляра QwenIntegration (singleton).

    Returns:
        QwenIntegration instance
    """
    global _qwen_instance

    if _qwen_instance is None:
        settings = get_settings()

        if not settings.qwen_api_key:
            raise ValueError("QWEN_API_KEY не установлен в настройках")

        logger.info(f"Инициализация QwenIntegration: {settings.qwen_model}")

        _qwen_instance = QwenIntegration(
            api_key=settings.qwen_api_key,
            model=settings.qwen_model,
            temperature=settings.qwen_temperature,
            max_tokens=settings.qwen_max_tokens,
            base_url=settings.qwen_base_url
        )

    return _qwen_instance

def get_repository_manager() -> Generator[RepositoryManager, None, None]:
    """Получение RepositoryManager instance."""
    settings = get_settings()
    manager = RepositoryManager(settings.repo_base_path)
    yield manager


# ==================== Cleanup ====================

def cleanup_dependencies():
    """Очистка всех singleton экземпляров."""
    global _embedder_instance, _qdrant_client_instance, _rag_retriever_instance, _qwen_instance

    logger.info("Очистка dependencies...")

    if _qdrant_client_instance:
        _qdrant_client_instance.close()

    _embedder_instance = None
    _qdrant_client_instance = None
    _rag_retriever_instance = None
    _qwen_instance = None

    logger.info("Dependencies очищены")
