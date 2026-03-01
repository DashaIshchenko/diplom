"""
API роуты для поиска кода.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import logging

from ...core.rag import RAGRetriever, SearchResult
from ...core.parser import ProgrammingLanguage, CodeElementType
from ..dependencies import get_rag_retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


# ==================== Models ====================

class SearchRequest(BaseModel):
    """Запрос на поиск."""
    query: str = Field(..., description="Поисковый запрос", min_length=1)
    top_k: int = Field(10, description="Количество результатов", ge=1, le=100)
    language: Optional[str] = Field(None, description="Фильтр по языку")
    element_type: Optional[str] = Field(None, description="Фильтр по типу элемента")
    repository_name: Optional[str] = Field(None, description="Фильтр по репозиторию")
    min_score: Optional[float] = Field(None, description="Минимальный score", ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "authentication function",
                "top_k": 10,
                "language": "python",
                "element_type": "function"
            }
        }


class CodeElementResponse(BaseModel):
    """Элемент кода в результатах."""
    name: str
    qualified_name: str
    type: str
    language: str
    source_code: str
    signature: Optional[str] = None
    docstring: Optional[str] = None
    file_path: str
    start_line: int
    end_line: int
    complexity: int
    repository_name: Optional[str] = None


class SearchResultResponse(BaseModel):
    """Результат поиска."""
    score: float
    element: CodeElementResponse


class SearchResponse(BaseModel):
    """Ответ поиска."""
    results: List[SearchResultResponse]
    total: int
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "score": 0.92,
                        "element": {
                            "name": "authenticate_user",
                            "type": "function",
                            "language": "python",
                            "source_code": "def authenticate_user(...)..."
                        }
                    }
                ],
                "total": 1,
                "query": "authentication"
            }
        }


# ==================== Endpoints ====================

@router.post("/", response_model=SearchResponse)
async def search_code(
        request: SearchRequest,
        retriever: RAGRetriever = Depends(get_rag_retriever)
) -> SearchResponse:
    """
    Поиск кода по запросу.

    Args:
        request: Параметры поиска
        retriever: RAGRetriever instance
    """
    try:
        # Конвертируем language
        language = None
        if request.language:
            try:
                language = ProgrammingLanguage[request.language.upper()]
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Неизвестный язык: {request.language}"
                )

        # Конвертируем element_type
        element_type = None
        if request.element_type:
            try:
                element_type = CodeElementType[request.element_type.upper()]
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Неизвестный тип: {request.element_type}"
                )

        # Выполняем поиск
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            language=language,
            element_type=element_type,
            repository_name=request.repository_name,
            score_threshold=request.min_score
        )

        # Форматируем результаты
        formatted_results = []
        for result in results:
            element = result.element

            formatted_results.append(
                SearchResultResponse(
                    score=result.score,
                    element=CodeElementResponse(
                        name=element.name,
                        qualified_name=element.qualified_name,
                        type=element.type.value,
                        language=element.language.value,
                        source_code=element.source_code,
                        signature=element.signature,
                        docstring=element.docstring,
                        file_path=str(element.location.file_path),
                        start_line=element.location.start_line,
                        end_line=element.location.end_line,
                        complexity=element.complexity,
                        repository_name=element.repository_name
                    )
                )
            )

        return SearchResponse(
            results=formatted_results,
            total=len(formatted_results),
            query=request.query
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{element_id}", response_model=SearchResponse)
async def search_similar(
        source_code: str,
        top_k: int = Query(10, ge=1, le=100),
        retriever: RAGRetriever = Depends(get_rag_retriever)
) -> SearchResponse:
    """
    Поиск похожих элементов кода.

    Args:
        source_code: Текст кода
        top_k: Количество результатов
        retriever: RAGRetriever instance
    """
    try:
        results = retriever.search_similar_code(
            source_code=source_code,
            top_k=top_k
        )

        formatted_results = []
        for result in results:
            element = result.element

            formatted_results.append(
                SearchResultResponse(
                    score=result.score,
                    element=CodeElementResponse(
                        name=element.name,
                        qualified_name=element.qualified_name,
                        type=element.type.value,
                        language=element.language.value,
                        source_code=element.source_code,
                        signature=element.signature,
                        docstring=element.docstring,
                        file_path=str(element.location.file_path),
                        start_line=element.location.start_line,
                        end_line=element.location.end_line,
                        complexity=element.complexity,
                        repository_name=getattr(element, 'repository_name', None)
                    )
                )
            )

        return SearchResponse(
            results=formatted_results,
            total=len(formatted_results),
            query=f"similar_to:{source_code}"
        )

    except Exception as e:
        logger.error(f"Ошибка поиска похожих: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
        request: SearchRequest,
        retriever: RAGRetriever = Depends(get_rag_retriever)
) -> SearchResponse:
    """
    Семантический поиск кода (алиас для search).

    Args:
        request: Параметры поиска
        retriever: RAGRetriever instance
    """
    return await search_code(request, retriever)


@router.get("/by-file")
async def search_by_file(
        file_path: str = Query(..., description="Путь к файлу"),
        repository_name: Optional[str] = Query(None, description="Имя репозитория"),
        retriever: RAGRetriever = Depends(get_rag_retriever)
) -> SearchResponse:
    """
    Поиск элементов из конкретного файла.

    Args:
        file_path: Путь к файлу
        repository_name: Имя репозитория
        retriever: RAGRetriever instance
    """
    try:
        if repository_name:
            results = retriever.search_by_file(
                file_path=file_path,
                repository_name=repository_name
            )
        else:
            results = retriever.search_by_file(file_path=file_path, repository_name=None)

        if not results:
            return SearchResponse(results=[], total=len(results), query=f"file:{file_path}")

        formatted_results = []
        for result in results:
            element = result.element

            formatted_results.append(
                SearchResultResponse(
                    score=result.score,
                    element=CodeElementResponse(
                        name=element.name,
                        qualified_name=element.qualified_name,
                        type=element.type.value,
                        language=element.language.value,
                        source_code=element.source_code,
                        signature=element.signature,
                        docstring=element.docstring,
                        file_path=str(element.location.file_path),
                        start_line=element.location.start_line,
                        end_line=element.location.end_line,
                        complexity=element.complexity,
                        repository_name=getattr(element, 'repository_name', None)
                    )
                )
            )

        return SearchResponse(
            results=formatted_results,
            total=len(formatted_results),
            query=f"file:{file_path}"
        )

    except Exception as e:
        logger.error(f"Ошибка поиска по файлу: {e}")
        raise HTTPException(status_code=500, detail=str(e))
