"""
API роуты для работы с репозиториями.
Индексация, управление и статистика по репозиториям.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel, Field
import logging

from ...config import settings
from ...core.vector_db import VectorizationPipeline, VectorizationResult
from ...core.vector_db import QdrantClient
from ...core.embeddings import CodeEmbedder
from ...core.git_handler.repository import RepositoryManager, RepositoryProvider, RepositoryInfo
from ...core.git_handler.repository import AzureDevOpsClient
from ..dependencies import get_embedder, get_qdrant_client, get_settings, get_repository_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/repositories", tags=["repositories"])


# ==================== Models ====================

class IndexRepositoryRequest(BaseModel):
    """Запрос на индексацию репозитория."""
    repository_path: str = Field(..., description="Путь к репозиторию")
    repository_name: str = Field(..., description="Имя репозитория")
    branch: str = Field("main", description="Ветка")
    commit_hash: Optional[str] = Field(None, description="Хэш коммита")
    provider: str = Field("local", description="Провайдер (local, github, gitlab, azure_devops)")
    exclude_dirs: Optional[List[str]] = Field(None, description="Исключаемые директории")
    # Azure DevOps специфичные параметры
    azure_connection: Optional[str] = Field(None, description="Имя Azure DevOps подключения")
    azure_project: Optional[str] = Field(None, description="Имя проекта Azure DevOps")
    azure_repository: Optional[str] = Field(None, description="Имя репозитория в Azure DevOps")

    class Config:
        json_schema_extra = {
            "example": {
                "repository_path": "/path/to/repo",
                "repository_name": "my-project",
                "branch": "main",
                "provider": "local"
            }
        }


class IndexingStatusResponse(BaseModel):
    """Статус индексации."""
    status: str = Field(..., description="queued, processing, completed, failed")
    repository_name: str
    progress: Optional[Dict[str, Any]] = None


class RepositoryInfoResponse(BaseModel):
    """Информация о репозитории."""
    repository_name: str
    branch: str
    total_files: int
    total_elements: int
    indexed_at: str
    languages: List[str]


class AzureDevOpsConnectionRequest(BaseModel):
    """Запрос на регистрацию Azure DevOps подключения."""
    name: str = Field(..., description="Имя подключения")
    base_url: str = Field(..., description="Базовый URL Azure DevOps")
    collection: str = Field("DefaultCollection", description="Название коллекции")
    personal_access_token: str = Field(..., description="Personal Access Token")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "company_azure",
                "base_url": "https://dev.azure.com/myorg",
                "collection": "DefaultCollection",
                "personal_access_token": "your_pat_here"
            }
        }


class AzureDevOpsRepositoryListResponse(BaseModel):
    """Список репозиториев из Azure DevOps."""
    repositories: List[Dict[str, str]]


# ==================== Endpoints ====================

@router.post("/index", response_model=IndexingStatusResponse)
async def index_repository(
        request: IndexRepositoryRequest,
        background_tasks: BackgroundTasks,
        embedder: CodeEmbedder = Depends(get_embedder),
        qdrant_client: QdrantClient = Depends(get_qdrant_client),
        repository_manager: RepositoryManager = Depends(get_repository_manager)
):
    """
    Индексация репозитория в фоновом режиме.

    Args:
        request: Параметры индексации
        background_tasks: FastAPI background tasks
        embedder: CodeEmbedder instance
        qdrant_client: QdrantClient instance
        repository_manager: RepositoryManager instance
    """
    try:
        # Проверка на Azure DevOps
        if request.provider == RepositoryProvider.AZURE_DEVOPS:
            if not request.azure_connection or not request.azure_project:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Для Azure DevOps необходимо указать azure_connection и azure_project"
                )

            # Проверка наличия репозитория
            if not request.azure_repository:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Для Azure DevOps необходимо указать azure_repository"
                )

            # Проверка существования Azure DevOps подключения
            if request.azure_connection not in repository_manager._azure_clients:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Azure DevOps подключение '{request.azure_connection}' не зарегистрировано"
                )

            # Используем background task для Azure DevOps
            background_tasks.add_task(
                _index_azure_repository_task,
                request,
                embedder,
                qdrant_client,
                repository_manager
            )
        else:
            # Обычная обработка для локальных/других репозиториев
            repository_path = Path(request.repository_path)

            if not repository_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Репозиторий не найден: {request.repository_path}"
                )

            settings = get_settings()

            # Создаем pipeline
            pipeline = VectorizationPipeline(
                collection_name=settings.collection_name,
                embedder=embedder,
                qdrant_client=qdrant_client
            )

            # Запускаем в фоне
            background_tasks.add_task(
                _index_repository_task,
                pipeline,
                repository_path,
                request.repository_name,
                request.branch,
                request.commit_hash,
                request.provider,
                request.exclude_dirs
            )

        return IndexingStatusResponse(
            status="queued",
            repository_name=request.repository_name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка запуска индексации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/azure/register-connection")
async def register_azure_connection(
        request: AzureDevOpsConnectionRequest,
        repository_manager: RepositoryManager = Depends(get_repository_manager)
) -> Dict[str, str]:
    """
    Регистрация Azure DevOps подключения.

    Args:
        request: Данные подключения
        repository_manager: RepositoryManager instance
    """
    try:
        repository_manager.register_azure_devops(
            name=request.name,
            base_url=request.base_url,
            collection=request.collection,
            personal_access_token=request.personal_access_token
        )

        return {"status": "registered", "connection_name": request.name}
    except Exception as e:
        logger.error(f"Ошибка регистрации Azure DevOps подключения: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/azure/list-repositories")
async def list_azure_repositories(
        azure_connection: str,
        project: str,
        repository_manager: RepositoryManager = Depends(get_repository_manager)
) -> AzureDevOpsRepositoryListResponse:
    """
    Получение списка репозиториев из Azure DevOps.

    Args:
        azure_connection: Имя зарегистрированного Azure подключения
        project: Название проекта
        repository_manager: RepositoryManager instance
    """
    try:
        # Проверка существования подключения
        if azure_connection not in repository_manager._azure_clients:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Azure DevOps подключение '{azure_connection}' не зарегистрировано"
            )

        repos = repository_manager.list_azure_repositories(azure_connection, project)
        return AzureDevOpsRepositoryListResponse(repositories=repos)
    except Exception as e:
        logger.error(f"Ошибка получения списка репозиториев: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_repositories(
        qdrant_client: QdrantClient = Depends(get_qdrant_client)
) -> List[str]:
    """Список проиндексированных репозиториев."""
    try:
        settings = get_settings()

        # Получаем уникальные имена репозиториев через scroll
        from ...core.vector_db import SearchFilters

        repositories = set()
        offset = None

        while True:
            results, next_offset = qdrant_client.scroll(
                collection_name=settings.collection_name,
                limit=100,
                offset=offset
            )

            for result in results:
                repo_name = result.get("payload", {}).get("repository_name")
                if repo_name:
                    repositories.add(repo_name)

            if next_offset is None:
                break

            offset = next_offset

        return sorted(list(repositories))

    except Exception as e:
        logger.error(f"Ошибка получения списка репозиториев: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repository_name}/info", response_model=RepositoryInfoResponse)
async def get_repository_info(
        repository_name: str,
        qdrant_client: QdrantClient = Depends(get_qdrant_client)
) -> RepositoryInfoResponse:
    """Информация о репозитории."""
    try:
        settings = get_settings()
        from ...core.vector_db import SearchFilters

        # Фильтруем по репозиторию
        filters = SearchFilters(repository_name=repository_name)

        # Считаем элементы
        total_elements = qdrant_client.count_points(
            collection_name=settings.collection_name,
            filters=filters
        )

        if total_elements == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Репозиторий не найден: {repository_name}"
            )

        # Получаем sample для метаданных
        results, _ = qdrant_client.scroll(
            collection_name=settings.collection_name,
            limit=100,
            filters=filters
        )

        # Собираем метаданные
        languages = set()
        files = set()
        branch = "unknown"
        indexed_at = None

        for result in results:
            payload = result.get("payload", {})

            lang = payload.get("language")
            if lang:
                languages.add(lang)

            file_path = payload.get("file_path")
            if file_path:
                files.add(file_path)

            if not branch or branch == "unknown":
                branch = payload.get("branch", "unknown")

            if not indexed_at:
                indexed_at = payload.get("indexed_at")

        return RepositoryInfoResponse(
            repository_name=repository_name,
            branch=branch,
            total_files=len(files),
            total_elements=total_elements,
            indexed_at=indexed_at or "unknown",
            languages=sorted(list(languages))
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения информации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{repository_name}")
async def delete_repository(
        repository_name: str,
        qdrant_client: QdrantClient = Depends(get_qdrant_client)
) -> Dict[str, str]:
    """Удаление репозитория из индекса."""
    try:
        settings = get_settings()
        from ...core.vector_db import SearchFilters

        filters = SearchFilters(repository_name=repository_name)

        success = qdrant_client.delete_by_filter(
            collection_name=settings.collection_name,
            filters=filters
        )

        if success:
            return {"status": "deleted", "repository_name": repository_name}
        else:
            raise HTTPException(status_code=500, detail="Ошибка удаления")

    except Exception as e:
        logger.error(f"Ошибка удаления репозитория: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repository_name}/statistics")
async def get_repository_statistics(
        repository_name: str,
        qdrant_client: QdrantClient = Depends(get_qdrant_client)
) -> Dict[str, Any]:
    """Статистика по репозиторию."""
    try:
        settings = get_settings()
        from ...core.vector_db import SearchFilters

        filters = SearchFilters(repository_name=repository_name)

        # Получаем все элементы
        all_results = []
        offset = None

        while True:
            results, next_offset = qdrant_client.scroll(
                collection_name=settings.collection_name,
                limit=100,
                filters=filters,
                offset=offset
            )

            all_results.extend(results)

            if next_offset is None:
                break

            offset = next_offset

        if not all_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Репозиторий не найден: {repository_name}"
            )

        # Подсчет статистики
        stats = {
            "total_elements": len(all_results),
            "by_language": {},
            "by_type": {},
            "by_file": {},
            "total_lines": 0,
            "avg_complexity": 0
        }

        total_complexity = 0

        for result in all_results:
            payload = result.get("payload", {})

            # По языкам
            lang = payload.get("language", "unknown")
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1

            # По типам
            elem_type = payload.get("element_type", "unknown")
            stats["by_type"][elem_type] = stats["by_type"].get(elem_type, 0) + 1

            # По файлам
            file_path = payload.get("file_path", "unknown")
            stats["by_file"][file_path] = stats["by_file"].get(file_path, 0) + 1

            # Строки и сложность
            stats["total_lines"] += payload.get("line_count", 0)
            total_complexity += payload.get("complexity", 0)

        if len(all_results) > 0:
            stats["avg_complexity"] = round(total_complexity / len(all_results), 2)

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{repository_name}/update")
async def update_repository_incremental(
        repository_name: str,
        background_tasks: BackgroundTasks,
        embedder: CodeEmbedder = Depends(get_embedder),
        qdrant_client: QdrantClient = Depends(get_qdrant_client),
        repository_manager: RepositoryManager = Depends(get_repository_manager)
):
    """Инкрементальное обновление репозитория."""
    try:
        # Получаем изменения
        changes = repository_manager.update_repository(repository_name)

        if not changes["changed"] and not changes["deleted"]:
            return {"status": "no_changes"}

        # Удаляем старые версии файлов
        for file_path in changes["deleted"] + changes["changed"]:
            from code_rag import SearchFilters
            filters = SearchFilters(
                repository_name=repository_name,
                file_path=str(file_path)
            )
            qdrant_client.delete_by_filter(settings.collection_name, filters)

        # Переиндексируем изменённые файлы
        repo_info = repository_manager.get_repository_info(repository_name)
        pipeline = VectorizationPipeline(...)

        for file_path in changes["changed"]:
            full_path = repo_info.local_path / file_path
            if full_path.exists():
                pipeline.process_file(full_path, {"repository_name": repository_name})

        return {
            "status": "updated",
            "changed": len(changes["changed"]),
            "deleted": len(changes["deleted"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Background Task ====================

def _index_repository_task(
        pipeline: VectorizationPipeline,
        repository_path: Path,
        repository_name: str,
        branch: str,
        commit_hash: Optional[str],
        provider: str,
        exclude_dirs: Optional[List[str]]
):
    """Фоновая задача индексации."""
    try:
        logger.info(f"Начало индексации репозитория: {repository_name}")

        result = pipeline.process_repository(
            repository_path=repository_path,
            repository_name=repository_name,
            branch=branch,
            commit_hash=commit_hash,
            provider=provider,
            exclude_dirs=exclude_dirs
        )

        logger.info(f"Индексация завершена: {result}")

    except Exception as e:
        logger.error(f"Ошибка индексации: {e}")
        raise


def _index_azure_repository_task(
        request: IndexRepositoryRequest,
        embedder: CodeEmbedder,
        qdrant_client: QdrantClient,
        repository_manager: RepositoryManager
):
    """Фоновая задача индексации Azure DevOps репозитория."""
    try:
        logger.info(f"Начало индексации Azure DevOps репозитория: {request.repository_name}")

        # Проверка параметров
        if not request.azure_connection or not request.azure_project or not request.azure_repository:
            raise ValueError("Недостаточно параметров для индексации Azure DevOps репозитория")

        # Проверка существования подключения
        if request.azure_connection not in repository_manager._azure_clients:
            raise ValueError(f"Azure DevOps подключение '{request.azure_connection}' не зарегистрировано")

        # Клонируем репозиторий
        repo_info = repository_manager.clone_azure_repository(
            azure_connection=request.azure_connection,
            project=request.azure_project,
            repository_name=request.azure_repository,
            branch=request.branch,
            tags=[f"indexed_{request.repository_name}"]
        )

        # Используем путь к локальному репозиторию для индексации
        settings = get_settings()
        pipeline = VectorizationPipeline(
            collection_name=settings.collection_name,
            embedder=embedder,
            qdrant_client=qdrant_client
        )

        # Индексируем
        result = pipeline.process_repository(
            repository_path=repo_info.local_path,
            repository_name=request.repository_name,
            branch=request.branch,
            commit_hash=repo_info.last_commit,
            provider=RepositoryProvider.AZURE_DEVOPS,
            exclude_dirs=request.exclude_dirs
        )

        logger.info(f"Индексация Azure DevOps репозитория завершена: {result}")

    except Exception as e:
        logger.error(f"Ошибка индексации Azure DevOps репозитория: {e}")
        # Не выбрасываем исключение, чтобы не прерывать фоновую задачу
        # В реальном приложении можно отправить уведомление об ошибке
        raise