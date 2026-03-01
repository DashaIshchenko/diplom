"""
API роуты для индексации кода.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
import tempfile
import shutil
import logging

from ...core.vector_db import VectorizationPipeline, VectorizationConfig, VectorizationResult
from ...core.vector_db import QdrantClient, CollectionSchema, DEFAULT_CODE_SCHEMA
from ...core.embeddings import CodeEmbedder
from ...core.parser import ParserFactory
from ..dependencies import get_embedder, get_qdrant_client, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/indexing", tags=["indexing"])


# ==================== Models ====================

class IndexFileRequest(BaseModel):
    """Запрос на индексацию файла."""
    file_path: str = Field(..., description="Путь к файлу")
    repository_name: Optional[str] = Field(None, description="Имя репозитория")

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/file.py",
                "repository_name": "my-project"
            }
        }


class IndexDirectoryRequest(BaseModel):
    """Запрос на индексацию директории."""
    directory_path: str = Field(..., description="Путь к директории")
    repository_name: str = Field(..., description="Имя репозитория")
    recursive: bool = Field(True, description="Рекурсивный поиск")

    class Config:
        json_schema_extra = {
            "example": {
                "directory_path": "/path/to/project",
                "repository_name": "my-project",
                "recursive": True
            }
        }


class IndexingResultResponse(BaseModel):
    """Результат индексации."""
    status: str
    total_files: int
    parsed_files: int
    failed_files: int
    total_elements: int
    indexed_elements: int
    success_rate: float
    errors: List[Dict[str, str]] = []


class CollectionStatusResponse(BaseModel):
    """Статус коллекции."""
    name: str
    exists: bool
    points_count: Optional[int] = None
    vectors_count: Optional[int] = None
    vector_size: Optional[int] = None


# ==================== Endpoints ====================

@router.post("/file", response_model=IndexingResultResponse)
async def index_file(
        request: IndexFileRequest,
        embedder: CodeEmbedder = Depends(get_embedder),
        qdrant_client: QdrantClient = Depends(get_qdrant_client)
) -> IndexingResultResponse:
    """
    Индексация одного файла.

    Args:
        request: Параметры индексации
        embedder: CodeEmbedder instance
        qdrant_client: QdrantClient instance
    """
    try:
        file_path = Path(request.file_path)

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Файл не найден: {request.file_path}")

        if not ParserFactory.can_parse_file(file_path):
            raise HTTPException(status_code=400, detail=f"Неподдерживаемый тип файла: {file_path.suffix}")

        settings = get_settings()

        # Создаем pipeline
        pipeline = VectorizationPipeline(
            collection_name=settings.collection_name,
            embedder=embedder,
            qdrant_client=qdrant_client
        )

        # Обрабатываем файл
        repository_info = None
        if request.repository_name:
            repository_info = {"repository_name": request.repository_name}

        result = pipeline.process_file(file_path, repository_info)

        return IndexingResultResponse(
            status="completed",
            total_files=result.total_files,
            parsed_files=result.parsed_files,
            failed_files=result.failed_files,
            total_elements=result.total_elements,
            indexed_elements=result.indexed_elements,
            success_rate=result.success_rate(),
            errors=[{"file": e["file"], "error": e["error"]} for e in result.errors]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка индексации файла: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/directory", response_model=IndexingResultResponse)
async def index_directory(
        request: IndexDirectoryRequest,
        background_tasks: BackgroundTasks,
        embedder: CodeEmbedder = Depends(get_embedder),
        qdrant_client: QdrantClient = Depends(get_qdrant_client)
) -> IndexingResultResponse:
    """
    Индексация директории.

    Args:
        request: Параметры индексации
        background_tasks: FastAPI background tasks
        embedder: CodeEmbedder instance
        qdrant_client: QdrantClient instance
    """
    try:
        directory_path = Path(request.directory_path)

        if not directory_path.exists():
            raise HTTPException(status_code=404, detail=f"Директория не найдена: {request.directory_path}")

        if not directory_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Не является директорией: {request.directory_path}")

        settings = get_settings()

        # Создаем pipeline
        pipeline = VectorizationPipeline(
            collection_name=settings.collection_name,
            embedder=embedder,
            qdrant_client=qdrant_client
        )

        # Обрабатываем директорию
        repository_info = {"repository_name": request.repository_name}

        result = pipeline.process_directory(
            directory=directory_path,
            repository_info=repository_info,
            recursive=request.recursive
        )

        return IndexingResultResponse(
            status="completed",
            total_files=result.total_files,
            parsed_files=result.parsed_files,
            failed_files=result.failed_files,
            total_elements=result.total_elements,
            indexed_elements=result.indexed_elements,
            success_rate=result.success_rate(),
            errors=[{"file": e["file"], "error": e["error"]} for e in result.errors]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка индексации директории: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_and_index(
        file: UploadFile = File(...),
        repository_name: Optional[str] = None,
        embedder: CodeEmbedder = Depends(get_embedder),
        qdrant_client: QdrantClient = Depends(get_qdrant_client)
) -> IndexingResultResponse:
    """
    Загрузка и индексация файла.

    Args:
        file: Загружаемый файл
        repository_name: Имя репозитория
        embedder: CodeEmbedder instance
        qdrant_client: QdrantClient instance
    """
    try:
        # Проверяем расширение
        file_suffix = Path(file.filename).suffix
        if not ParserFactory.is_extension_supported(file_suffix):
            raise HTTPException(status_code=400, detail=f"Неподдерживаемый тип файла: {file_suffix}")

        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = Path(tmp_file.name)

        try:
            settings = get_settings()

            # Создаем pipeline
            pipeline = VectorizationPipeline(
                collection_name=settings.collection_name,
                embedder=embedder,
                qdrant_client=qdrant_client
            )

            # Обрабатываем файл
            repository_info = None
            if repository_name:
                repository_info = {"repository_name": repository_name}

            result = pipeline.process_file(tmp_path, repository_info)

            # Обрабатываем ошибки, учитывая возможное None
            errors = []
            if result.errors:
                errors = [{"file": e["file"], "error": e["error"]} for e in result.errors]

            return IndexingResultResponse(
                status="completed",
                total_files=result.total_files,
                parsed_files=result.parsed_files,
                failed_files=result.failed_files,
                total_elements=result.total_elements,
                indexed_elements=result.indexed_elements,
                success_rate=result.success_rate(),
                errors=errors
            )

        finally:
            # Удаляем временный файл
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    except HTTPException:
        # Перебросить HTTP исключения
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки и индексации: {e}")
        # Удаление временного файла в случае ошибки
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/status", response_model=CollectionStatusResponse)
async def get_collection_status(
        qdrant_client: QdrantClient = Depends(get_qdrant_client)
) -> CollectionStatusResponse:
    """
    Статус коллекции.

    Args:
        qdrant_client: QdrantClient instance
    """
    try:
        settings = get_settings()
        collection_name = settings.collection_name

        exists = qdrant_client.collection_exists(collection_name)

        if not exists:
            return CollectionStatusResponse(
                name=collection_name,
                exists=False
            )

        info = qdrant_client.get_collection_info(collection_name)

        return CollectionStatusResponse(
            name=collection_name,
            exists=True,
            points_count=info.get("points_count"),
            vectors_count=info.get("vectors_count"),
            vector_size=info.get("vector_size")
        )

    except Exception as e:
        logger.error(f"Ошибка получения статуса коллекции: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collection/create")
async def create_collection(
        recreate: bool = False,
        qdrant_client: QdrantClient = Depends(get_qdrant_client),
        embedder: CodeEmbedder = Depends(get_embedder)
) -> Dict[str, str]:
    """
    Создание коллекции.

    Args:
        recreate: Пересоздать если существует
        qdrant_client: QdrantClient instance
        embedder: CodeEmbedder instance
    """
    try:
        settings = get_settings()

        # Создаем схему с правильным размером вектора
        schema = CollectionSchema(
            collection_name=settings.collection_name,
            vector_size=embedder.embedding_dim
        )

        success = qdrant_client.create_collection(schema, recreate=recreate)

        if success:
            return {"status": "created", "collection_name": settings.collection_name}
        else:
            raise HTTPException(status_code=500, detail="Ошибка создания коллекции")

    except Exception as e:
        logger.error(f"Ошибка создания коллекции: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collection")
async def delete_collection(
        qdrant_client: QdrantClient = Depends(get_qdrant_client)
) -> Dict[str, str]:
    """
    Удаление коллекции.

    Args:
        qdrant_client: QdrantClient instance
    """
    try:
        settings = get_settings()

        success = qdrant_client.delete_collection(settings.collection_name)

        if success:
            return {"status": "deleted", "collection_name": settings.collection_name}
        else:
            raise HTTPException(status_code=500, detail="Ошибка удаления коллекции")

    except Exception as e:
        logger.error(f"Ошибка удаления коллекции: {e}")
        raise HTTPException(status_code=500, detail=str(e))
