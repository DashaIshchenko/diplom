"""
FastAPI приложение.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .config import get_settings
from .routes import embeddings, search, indexing, health, generation
from .dependencies import cleanup_dependencies

from monitoring_module.router import router as monitoring_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Создание FastAPI приложения.

    Returns:
        FastAPI app instance
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Роуты
    app.include_router(health.router)
    app.include_router(embeddings.router)
    app.include_router(search.router)
    app.include_router(indexing.router)
    app.include_router(generation.router)
    app.include_router(monitoring_router, prefix="/monitor", tags=["Monitoring"])

    # События
    @app.on_event("startup")
    async def startup_event():
        logger.info("Запуск Code RAG API...")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Остановка Code RAG API...")
        cleanup_dependencies()

    return app


# Экземпляр приложения
app = create_app()
