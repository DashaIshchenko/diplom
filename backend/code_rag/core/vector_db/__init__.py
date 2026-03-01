"""
Модуль работы с векторной базой данных Qdrant.

Включает:
- Схемы данных для хранения
- Клиент Qdrant
- Пайплайн векторизации
"""

from .schemas import (
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
)
from .qdrant_client import QdrantClient, create_qdrant_client
from .vectorization_pipeline import (
    VectorizationPipeline,
    VectorizationConfig,
    VectorizationResult,
    create_vectorization_pipeline,
)

# Алиасы для совместимости со старым кодом
CodeVectorPayload = CodePayload
SearchFilter = SearchFilters
QdrantCodeClient = QdrantClient
CodeVectorizationPipeline = VectorizationPipeline
create_default_pipeline = create_vectorization_pipeline
VectorPoint = PointData

__all__ = [
    # Основные классы (новые имена)
    "CodePayload",
    "CollectionSchema",
    "IndexConfig",
    "PointData",
    "BatchInsertData",
    "SearchFilters",
    "VectorDistance",

    # Предопределенные схемы
    "DEFAULT_CODE_SCHEMA",
    "LARGE_CODE_SCHEMA",
    "FAST_CODE_SCHEMA",
    "create_collection_schema",

    # Клиент
    "QdrantClient",
    "create_qdrant_client",

    # Pipeline
    "VectorizationPipeline",
    "VectorizationConfig",
    "VectorizationResult",
    "create_vectorization_pipeline",

    # Алиасы для совместимости
    "CodeVectorPayload",
    "SearchFilter",
    "QdrantCodeClient",
    "CodeVectorizationPipeline",
    "create_default_pipeline",
    "VectorPoint",
]
