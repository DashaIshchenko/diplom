"""
Модуль RAG (Retrieval-Augmented Generation).

Обеспечивает:
- Интеллектуальный поиск контекста
- Интеграцию с Qwen-Coder
- Генерацию кода и ответов
"""

from .rag_retriever import RAGRetriever, SearchResult, RAGConfig, create_retriever,RAGResponse
from .qwen_integration import QwenIntegration, QwenResponse, create_qwen_integration

# Алиасы для совместимости
CodeRAGRetriever = RAGRetriever
RetrievedContext = SearchResult
QwenCoderAgent = QwenIntegration
GenerationConfig = QwenResponse

__all__ = [
    # Новые имена
    "RAGRetriever",
    "SearchResult",
    "RAGConfig",
    "RAGResponse",
    "create_retriever",
    "QwenIntegration",
    "QwenResponse",
    "create_qwen_integration",

    # Алиасы для совместимости
    "CodeRAGRetriever",
    "RetrievedContext",
    "QwenCoderAgent",
    "GenerationConfig",
]
