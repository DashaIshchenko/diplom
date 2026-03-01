"""
Ядро системы CODE RAG.

Включает модули:
- parser: Парсинг кода на разных языках
- embeddings: Векторизация кода
- vector_db: Работа с Qdrant векторной БД
- rag: RAG система (retrieval и generation)
"""

# Парсер
from .parser import (
    # Структуры данных
    CodeElement,
    CodeElementType,
    CodeLocation,
    ModuleInfo,
    ParseResult,
    ProgrammingLanguage,

    # Базовый класс и фабрика
    BaseParser,
    ParserFactory,

    # Парсеры языков
    PythonParser,
    JavaParser,
    JavaScriptParser,
    TypeScriptParser,
    CSharpParser,
    KotlinParser,
    HTMLParser,
    CSSParser,

    # Утилиты парсера
    get_parser_info,
    list_supported_languages,
    list_supported_extensions,
)

# Embeddings
from .embeddings import (
    BaseEmbeddingModel,
    EmbeddingConfig,
    NomicEmbedModel,
    EmbeddingModelFactory,
    EmbeddingModelType,
    CodeEmbedder,
    create_code_embedder,
    OllamaEmbedModel,
    create_ollama_embed_model,
)

# Vector DB
from .vector_db import (
    # Клиент
    QdrantClient,
    create_qdrant_client,

    # Схемы
    CodePayload,
    CollectionSchema,
    IndexConfig,
    PointData,
    BatchInsertData,
    SearchFilters,
    VectorDistance,
    DEFAULT_CODE_SCHEMA,
    LARGE_CODE_SCHEMA,
    FAST_CODE_SCHEMA,
    create_collection_schema,

    # Pipeline векторизации
    VectorizationPipeline,
    VectorizationConfig,
    VectorizationResult,
    create_vectorization_pipeline,
)

# RAG
from .rag import (
    # Retriever
    RAGRetriever,
    SearchResult,
    create_retriever,

    # Qwen Integration
    QwenIntegration,
    QwenResponse,
    create_qwen_integration,
)

__all__ = [
    # ==================== PARSER ====================
    # Структуры данных
    "CodeElement",
    "CodeElementType",
    "CodeLocation",
    "ModuleInfo",
    "ParseResult",
    "ProgrammingLanguage",

    # Базовый класс и фабрика
    "BaseParser",
    "ParserFactory",

    # Парсеры языков
    "PythonParser",
    "JavaParser",
    "JavaScriptParser",
    "TypeScriptParser",
    "CSharpParser",
    "KotlinParser",
    "HTMLParser",
    "CSSParser",

    # Утилиты
    "get_parser_info",
    "list_supported_languages",
    "list_supported_extensions",

    # ==================== EMBEDDINGS ====================
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "NomicEmbedModel",
    "EmbeddingModelFactory",
    "EmbeddingModelType",
    "CodeEmbedder",
    "create_code_embedder",
    "OllamaEmbedModel",
    "create_ollama_embed_model",

    # ==================== VECTOR DB ====================
    # Клиент
    "QdrantClient",
    "create_qdrant_client",

    # Схемы
    "CodePayload",
    "CollectionSchema",
    "IndexConfig",
    "PointData",
    "BatchInsertData",
    "SearchFilters",
    "VectorDistance",
    "DEFAULT_CODE_SCHEMA",
    "LARGE_CODE_SCHEMA",
    "FAST_CODE_SCHEMA",
    "create_collection_schema",

    # Pipeline
    "VectorizationPipeline",
    "VectorizationConfig",
    "VectorizationResult",
    "create_vectorization_pipeline",

    # ==================== RAG ====================
    # Retriever
    "RAGRetriever",
    "SearchResult",
    "create_retriever",

    # Qwen Integration
    "QwenIntegration",
    "QwenResponse",
    "create_qwen_integration",
]

# Версия модуля
__version__ = "0.1.0"

# Краткая справка по использованию
__doc__ = """
CODE RAG - Система для индексации и поиска кода с использованием RAG.

Быстрый старт:
==============

1. Парсинг кода:
   >>> from code_rag.core import ParserFactory
   >>> parser = ParserFactory.create_parser_for_file(Path("file.py"))
   >>> module = parser.parse_file(Path("file.py"))

2. Создание векторной БД:
   >>> from code_rag.core import QdrantClient, DEFAULT_CODE_SCHEMA
   >>> client = QdrantClient(url="http://localhost:6333")
   >>> client.create_collection(DEFAULT_CODE_SCHEMA)

3. Индексация кода:
   >>> from code_rag.core import VectorizationPipeline, CodeEmbedder
   >>> embedder = CodeEmbedder()
   >>> pipeline = VectorizationPipeline("code_collection", embedder, client)
   >>> result = pipeline.process_repository(Path("/path/to/repo"), "my-project")

4. Поиск кода:
   >>> from code_rag.core import RAGRetriever
   >>> retriever = RAGRetriever("code_collection", embedder, client)
   >>> results = retriever.search("authentication function", top_k=5)

5. Генерация ответов:
   >>> from code_rag.core import QwenIntegration
   >>> qwen = QwenIntegration(api_key="...", model="qwen2.5-coder-32b-instruct")
   >>> response = qwen.answer_question_with_rag("How is auth implemented?", results)
   >>> print(response.content)

Поддерживаемые языки:
=====================
Python, Java, JavaScript, TypeScript, C#, Kotlin, HTML, CSS

Документация: https://github.com/your-org/code-rag
"""


def print_module_info():
    """Вывод информации о модуле."""
    print("=" * 70)
    print("CODE RAG CORE MODULE")
    print("=" * 70)
    print(f"Version: {__version__}")
    print()
    print("Available components:")
    print("  - Parser: Поддержка 8 языков программирования")
    print("  - Embeddings: Векторизация кода (Nomic Embed)")
    print("  - Vector DB: Qdrant интеграция")
    print("  - RAG: Retrieval и генерация с Qwen")
    print()
    print(f"Поддерживаемые языки: {', '.join(list_supported_languages())}")
    print(f"Всего экспортов: {len(__all__)}")
    print("=" * 70)


if __name__ == "__main__":
    print_module_info()
