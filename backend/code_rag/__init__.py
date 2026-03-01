"""
Code RAG - Система векторизации и семантического поиска исходного кода.

Основные компоненты:
- Parser: Парсинг исходного кода на множественных языках
- Embeddings: Векторизация кода с использованием Nomic Embed
- Vector DB: Интеграция с Qdrant для хранения и поиска векторов
- RAG: Retrieval-Augmented Generation с Qwen-Coder
- Git Handler: Работа с Git репозиториями и мониторинг изменений
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Импорты основных компонентов для удобного доступа
from core.embeddings import (
    CodeEmbedder,
    NomicEmbedModel,
    EmbeddingModelFactory,
    EmbeddingModelType,
)

from core.parser import (
    ParserFactory,
    ProgrammingLanguage,
    CodeElementType,
    CodeElement,
    ModuleInfo,
)

from core.vector_db import (
    QdrantClient,
    VectorizationPipeline,
    CollectionSchema,
    SearchFilters,
    CodePayload,
)

from core.rag import (
    RAGRetriever,
    SearchResult,
    RAGConfig,
    RAGResponse,
    QwenIntegration,
    create_retriever,
)

from core.git_handler import (
    RepositoryManager,
    RepositoryInfo,
    RepositoryMonitor,
    CommitInfo,
    ChangeEvent,
)

# Утилиты
from utils import (
    setup_logger,
    load_config,
    save_config,
)

# Экспорт всех публичных компонентов
__all__ = [
    # Version
    "__version__",

    # Embeddings
    "CodeEmbedder",
    "NomicEmbedModel",
    "EmbeddingModelFactory",
    "EmbeddingModelType",

    # Parser
    "ParserFactory",
    "ProgrammingLanguage",
    "CodeElementType",
    "CodeElement",
    "ModuleInfo",

    # Vector DB
    "QdrantClient",
    "VectorizationPipeline",
    "CollectionSchema",
    "SearchFilters",
    "CodePayload",

    # RAG
    "RAGRetriever",
    "SearchResult",
    "RAGConfig",
    "RAGResponse",
    "QwenIntegration",
    "create_retriever",

    # Git Handler
    "RepositoryManager",
    "RepositoryInfo",
    "RepositoryMonitor",
    "CommitInfo",
    "ChangeEvent",

    # Utils
    "setup_logger",
    "load_config",
    "save_config",
]

# Настройка логирования при импорте
import logging

# Базовая настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Создаем logger для всего пакета
logger = logging.getLogger(__name__)
logger.info(f"Code RAG v{__version__} initialized")


# Проверка зависимостей при импорте
def _check_dependencies():
    """Проверка наличия критичных зависимостей."""
    missing_deps = []

    try:
        import torch
    except ImportError:
        missing_deps.append("torch")

    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")

    try:
        import qdrant_client
    except ImportError:
        missing_deps.append("qdrant-client")

    try:
        import tree_sitter
    except ImportError:
        missing_deps.append("tree-sitter")

    if missing_deps:
        logger.warning(
            f"Missing optional dependencies: {', '.join(missing_deps)}. "
            f"Some features may not be available."
        )

    return len(missing_deps) == 0


# Проверяем зависимости
_check_dependencies()


# Удобные функции для быстрого старта
def quick_start(
        collection_name: str = "code-rag",
        qdrant_url: str = "http://localhost:6333",
        device: str = "cpu"
):
    """
    Быстрая инициализация всех компонентов.

    Args:
        collection_name: Имя коллекции в Qdrant
        qdrant_url: URL Qdrant сервера
        device: Устройство для embeddings ('cpu' или 'cuda')

    Returns:
        dict с инициализированными компонентами

    Example:
        >>> components = quick_start()
        >>> pipeline = components['pipeline']
        >>> retriever = components['retriever']
    """
    logger.info("Initializing Code RAG components...")

    # Embedder
    embedder = CodeEmbedder(device=device)
    logger.info(f"Embedder initialized on {device}")

    # Qdrant Client
    qdrant_client = QdrantClient(url=qdrant_url)
    logger.info(f"Connected to Qdrant at {qdrant_url}")

    # Collection Schema
    schema = CollectionSchema(
        collection_name=collection_name,
        vector_size=embedder.embedding_dim
    )

    # Создаем коллекцию если не существует
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(schema)
        logger.info(f"Created collection '{collection_name}'")

    # Vectorization Pipeline
    pipeline = VectorizationPipeline(
        collection_name=collection_name,
        embedder=embedder,
        qdrant_client=qdrant_client
    )

    # RAG Retriever
    retriever = RAGRetriever(
        collection_name=collection_name,
        embedder=embedder,
        qdrant_client=qdrant_client
    )

    logger.info("All components initialized successfully")

    return {
        'embedder': embedder,
        'qdrant_client': qdrant_client,
        'pipeline': pipeline,
        'retriever': retriever,
        'collection_name': collection_name
    }


# Добавляем в __all__
__all__.append("quick_start")


# Информация о пакете
def get_info():
    """Получение информации о пакете."""
    return {
        "name": "code-rag",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "RAG система для векторизации и поиска исходного кода",
        "components": {
            "embeddings": "Nomic Embed Text v1.5",
            "vector_db": "Qdrant",
            "llm": "Qwen-Coder",
            "parser": "Tree-sitter"
        }
    }


__all__.append("get_info")
