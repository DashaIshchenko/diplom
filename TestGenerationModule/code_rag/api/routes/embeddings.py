"""
API роуты для работы с эмбеддингами кода.
Предоставляет эндпоинты для векторизации кода и текста.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
import logging

from ...core.embeddings import CodeEmbedder, EmbeddingModelType
from ..dependencies import get_embedder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# ==================== Request/Response Models ====================

class EmbedTextRequest(BaseModel):
    """Запрос на векторизацию текста."""
    text: str = Field(..., description="Текст для векторизации", min_length=1)
    normalize: bool = Field(True, description="Нормализовать эмбеддинг")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "def calculate_sum(a, b): return a + b",
                "normalize": True
            }
        }


class EmbedBatchRequest(BaseModel):
    """Запрос на векторизацию батча текстов."""
    texts: List[str] = Field(..., description="Список текстов", min_items=1, max_items=100)
    batch_size: Optional[int] = Field(32, description="Размер батча", ge=1, le=128)
    normalize: bool = Field(True, description="Нормализовать эмбеддинги")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "def add(a, b): return a + b",
                    "def subtract(a, b): return a - b"
                ],
                "batch_size": 32,
                "normalize": True
            }
        }


class EmbeddingResponse(BaseModel):
    """Ответ с эмбеддингом."""
    embedding: List[float] = Field(..., description="Вектор эмбеддинга")
    dimension: int = Field(..., description="Размерность вектора")
    model: str = Field(..., description="Использованная модель")

    class Config:
        json_schema_extra = {
            "example": {
                "embedding": [0.123, -0.456, 0.789],
                "dimension": 768,
                "model": "nomic-embed-text-v1.5"
            }
        }


class BatchEmbeddingResponse(BaseModel):
    """Ответ с батчем эмбеддингов."""
    embeddings: List[List[float]] = Field(..., description="Список векторов")
    count: int = Field(..., description="Количество эмбеддингов")
    dimension: int = Field(..., description="Размерность векторов")
    model: str = Field(..., description="Использованная модель")

    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": [
                    [0.123, -0.456, 0.789],
                    [0.321, 0.654, -0.987]
                ],
                "count": 2,
                "dimension": 768,
                "model": "nomic-embed-text-v1.5"
            }
        }


class SimilarityRequest(BaseModel):
    """Запрос на вычисление схожести."""
    text1: str = Field(..., description="Первый текст")
    text2: str = Field(..., description="Второй текст")

    class Config:
        json_schema_extra = {
            "example": {
                "text1": "def add(a, b): return a + b",
                "text2": "def sum_numbers(x, y): return x + y"
            }
        }


class SimilarityResponse(BaseModel):
    """Ответ с коэффициентом схожести."""
    similarity: float = Field(..., description="Косинусная схожесть (0-1)", ge=0.0, le=1.0)
    text1_length: int = Field(..., description="Длина первого текста")
    text2_length: int = Field(..., description="Длина второго текста")

    class Config:
        json_schema_extra = {
            "example": {
                "similarity": 0.92,
                "text1_length": 32,
                "text2_length": 38
            }
        }


class ModelInfoResponse(BaseModel):
    """Информация о модели эмбеддингов."""
    model_name: str = Field(..., description="Название модели")
    embedding_dimension: int = Field(..., description="Размерность эмбеддингов")
    max_sequence_length: int = Field(..., description="Максимальная длина последовательности")
    device: str = Field(..., description="Устройство (cpu/cuda)")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "nomic-embed-text-v1.5",
                "embedding_dimension": 768,
                "max_sequence_length": 8192,
                "device": "cpu"
            }
        }


# ==================== Endpoints ====================

@router.post(
    "/embed",
    response_model=EmbeddingResponse,
    summary="Векторизация текста",
    description="Создает векторное представление (эмбеддинг) для текста кода или запроса"
)
async def embed_text(
        request: EmbedTextRequest,
        embedder: CodeEmbedder = Depends(get_embedder)
) -> EmbeddingResponse:
    """
    Векторизация одного текста.

    Args:
        request: Запрос с текстом
        embedder: Экземпляр embedder (dependency)

    Returns:
        EmbeddingResponse с вектором
    """
    try:
        # Векторизуем текст
        embedding = embedder.encode_text(request.text)

        return EmbeddingResponse(
            embedding=embedding.tolist(),
            dimension=len(embedding),
            model=embedder.model.__class__.__name__
        )

    except Exception as e:
        logger.error(f"Ошибка векторизации текста: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка векторизации: {str(e)}"
        )


@router.post(
    "/embed/batch",
    response_model=BatchEmbeddingResponse,
    summary="Векторизация батча текстов",
    description="Создает векторные представления для списка текстов (эффективнее чем по одному)"
)
async def embed_batch(
        request: EmbedBatchRequest,
        embedder: CodeEmbedder = Depends(get_embedder)
) -> BatchEmbeddingResponse:
    """
    Векторизация батча текстов.

    Args:
        request: Запрос с текстами
        embedder: Экземпляр embedder (dependency)

    Returns:
        BatchEmbeddingResponse с векторами
    """
    try:
        # Векторизуем батч
        embeddings = embedder.encode_batch(
            request.texts,
            batch_size=request.batch_size
        )

        return BatchEmbeddingResponse(
            embeddings=embeddings.tolist(),
            count=len(embeddings),
            dimension=embeddings.shape[1] if len(embeddings) > 0 else 0,
            model=embedder.model.__class__.__name__
        )

    except Exception as e:
        logger.error(f"Ошибка векторизации батча: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка векторизации батча: {str(e)}"
        )


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    summary="Вычисление схожести",
    description="Вычисляет косинусную схожесть между двумя текстами"
)
async def calculate_similarity(
        request: SimilarityRequest,
        embedder: CodeEmbedder = Depends(get_embedder)
) -> SimilarityResponse:
    """
    Вычисление схожести двух текстов.

    Args:
        request: Запрос с двумя текстами
        embedder: Экземпляр embedder (dependency)

    Returns:
        SimilarityResponse с коэффициентом схожести
    """
    try:
        import numpy as np

        # Векторизуем оба текста
        embeddings = embedder.encode_batch([request.text1, request.text2])

        # Вычисляем косинусную схожесть
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]

        # Косинусная схожесть
        similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        return SimilarityResponse(
            similarity=float(similarity),
            text1_length=len(request.text1),
            text2_length=len(request.text2)
        )

    except Exception as e:
        logger.error(f"Ошибка вычисления схожести: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка вычисления схожести: {str(e)}"
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Информация о модели",
    description="Возвращает информацию об используемой модели эмбеддингов"
)
async def get_model_info(
        embedder: CodeEmbedder = Depends(get_embedder)
) -> ModelInfoResponse:
    """
    Получение информации о модели.

    Args:
        embedder: Экземпляр embedder (dependency)

    Returns:
        ModelInfoResponse с информацией о модели
    """
    try:
        return ModelInfoResponse(
            model_name=embedder.model.__class__.__name__,
            embedding_dimension=embedder.embedding_dim,
            max_sequence_length=getattr(embedder.model, 'max_seq_length', 8192),
            device=getattr(embedder.model, 'device', 'cpu')
        )

    except Exception as e:
        logger.error(f"Ошибка получения информации о модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка получения информации: {str(e)}"
        )


@router.post(
    "/embed/code",
    response_model=EmbeddingResponse,
    summary="Векторизация кода",
    description="Специализированная векторизация фрагмента кода"
)
async def embed_code(
        request: EmbedTextRequest,
        embedder: CodeEmbedder = Depends(get_embedder)
) -> EmbeddingResponse:
    """
    Векторизация кода (алиас для embed_text).

    Args:
        request: Запрос с кодом
        embedder: Экземпляр embedder (dependency)

    Returns:
        EmbeddingResponse с вектором
    """
    # Используем тот же метод что и для текста
    return await embed_text(request, embedder)


@router.post(
    "/embed/query",
    response_model=EmbeddingResponse,
    summary="Векторизация запроса",
    description="Векторизация поискового запроса"
)
async def embed_query(
        request: EmbedTextRequest,
        embedder: CodeEmbedder = Depends(get_embedder)
) -> EmbeddingResponse:
    """
    Векторизация поискового запроса.

    Args:
        request: Запрос с текстом запроса
        embedder: Экземпляр embedder (dependency)

    Returns:
        EmbeddingResponse с вектором
    """
    try:
        # Векторизуем запрос
        embedding = embedder.encode_query(request.text)

        return EmbeddingResponse(
            embedding=embedding.tolist(),
            dimension=len(embedding),
            model=embedder.model.__class__.__name__
        )

    except Exception as e:
        logger.error(f"Ошибка векторизации запроса: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка векторизации запроса: {str(e)}"
        )


@router.get(
    "/health",
    summary="Проверка здоровья",
    description="Проверка доступности сервиса эмбеддингов"
)
async def health_check(embedder: CodeEmbedder = Depends(get_embedder)) -> Dict[str, Any]:
    """
    Проверка здоровья сервиса эмбеддингов.

    Args:
        embedder: Экземпляр embedder (dependency)

    Returns:
        Словарь со статусом
    """
    try:
        # Проверяем что модель работает
        test_embedding = embedder.encode_text("test")

        return {
            "status": "healthy",
            "model": embedder.model.__class__.__name__,
            "embedding_dim": embedder.embedding_dim,
            "test_embedding_shape": test_embedding.shape
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embeddings service unhealthy: {str(e)}"
        )
