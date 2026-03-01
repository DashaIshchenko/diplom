"""
API роуты для health checks и статуса системы.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import logging
from datetime import datetime

from ...core.embeddings import CodeEmbedder
from ...core.vector_db import QdrantClient
from ..dependencies import get_embedder, get_qdrant_client, get_settings
from ..config import Settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


# ==================== Models ====================

class HealthResponse(BaseModel):
    """Ответ health check."""
    status: str
    timestamp: str
    version: str
    components: Dict[str, Any]


class SystemInfoResponse(BaseModel):
    """Информация о системе."""
    version: str
    embedder_model: str
    embedding_dimension: int
    qdrant_url: str
    collection_name: str
    components_status: Dict[str, str]


# ==================== Endpoints ====================

@router.get("/health", response_model=HealthResponse)
async def health_check(
        embedder: CodeEmbedder = Depends(get_embedder),
        qdrant_client: QdrantClient = Depends(get_qdrant_client),
        settings: Settings = Depends(get_settings)
) -> HealthResponse:
    """
    Проверка здоровья всех компонентов системы.

    Args:
        embedder: CodeEmbedder instance
        qdrant_client: QdrantClient instance
        settings: Settings instance
    """
    components = {}
    overall_status = "healthy"

    # Проверка embedder
    try:
        test_embedding = embedder.encode_text("test")
        components["embedder"] = {
            "status": "healthy",
            "model": embedder.model.__class__.__name__,
            "dimension": embedder.embedding_dim
        }
    except Exception as e:
        logger.error(f"Embedder health check failed: {e}")
        components["embedder"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "unhealthy"

    # Проверка Qdrant
    try:
        qdrant_healthy = qdrant_client.health_check()
        if qdrant_healthy:
            components["qdrant"] = {
                "status": "healthy",
                "url": qdrant_client.url
            }
        else:
            components["qdrant"] = {
                "status": "unhealthy",
                "url": qdrant_client.url
            }
            overall_status = "degraded"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        components["qdrant"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "unhealthy"

    # Проверка коллекции
    try:
        collection_exists = qdrant_client.collection_exists(settings.collection_name)
        if collection_exists:
            info = qdrant_client.get_collection_info(settings.collection_name)
            components["collection"] = {
                "status": "healthy",
                "name": settings.collection_name,
                "points_count": info.get("points_count", 0)
            }
        else:
            components["collection"] = {
                "status": "not_found",
                "name": settings.collection_name
            }
            overall_status = "degraded"
    except Exception as e:
        logger.error(f"Collection check failed: {e}")
        components["collection"] = {
            "status": "error",
            "error": str(e)
        }

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version=settings.api_version,
        components=components
    )


@router.get("/")
async def root() -> Dict[str, str]:
    """Корневой эндпоинт."""
    return {
        "message": "Code RAG API",
        "status": "running",
        "docs": "/docs"
    }


@router.get("/info", response_model=SystemInfoResponse)
async def system_info(
        embedder: CodeEmbedder = Depends(get_embedder),
        qdrant_client: QdrantClient = Depends(get_qdrant_client),
        settings: Settings = Depends(get_settings)
) -> SystemInfoResponse:
    """
    Информация о системе.

    Args:
        embedder: CodeEmbedder instance
        qdrant_client: QdrantClient instance
        settings: Settings instance
    """
    # Проверяем статусы компонентов
    embedder_status = "unknown"
    qdrant_status = "unknown"
    collection_status = "unknown"

    try:
        embedder.encode_text("test")
        embedder_status = "healthy"
    except:
        embedder_status = "unhealthy"

    try:
        if qdrant_client.health_check():
            qdrant_status = "healthy"
        else:
            qdrant_status = "unhealthy"
    except:
        qdrant_status = "error"

    try:
        if qdrant_client.collection_exists(settings.collection_name):
            collection_status = "exists"
        else:
            collection_status = "not_found"
    except:
        collection_status = "error"

    return SystemInfoResponse(
        version=settings.api_version,
        embedder_model=embedder.model.__class__.__name__,
        embedding_dimension=embedder.embedding_dim,
        qdrant_url=settings.qdrant_url,
        collection_name=settings.collection_name,
        components_status={
            "embedder": embedder_status,
            "qdrant": qdrant_status,
            "collection": collection_status
        }
    )


@router.get("/ping")
async def ping() -> Dict[str, str]:
    """Простой ping эндпоинт."""
    return {"message": "pong"}


@router.get("/version")
async def version(settings: Settings = Depends(get_settings)) -> Dict[str, str]:
    """Версия API."""
    return {
        "version": settings.api_version,
        "title": settings.api_title
    }


@router.get("/health/detailed")
async def detailed_health_check(
        qdrant_client: QdrantClient = Depends(get_qdrant_client),
        embedder: CodeEmbedder = Depends(get_embedder)
):
    """Детальная проверка здоровья системы."""
    health = {
        "status": "healthy",
        "components": {}
    }

    # Проверка Qdrant
    try:
        collections = qdrant_client.list_collections()
        health["components"]["qdrant"] = {
            "status": "up",
            "collections": len(collections)
        }
    except Exception as e:
        health["status"] = "degraded"
        health["components"]["qdrant"] = {
            "status": "down",
            "error": str(e)
        }

    # Проверка Embedder
    try:
        test_vector = embedder.encode_text("test")
        health["components"]["embedder"] = {
            "status": "up",
            "dimension": len(test_vector)
        }
    except Exception as e:
        health["status"] = "degraded"
        health["components"]["embedder"] = {
            "status": "down",
            "error": str(e)
        }

    return health
