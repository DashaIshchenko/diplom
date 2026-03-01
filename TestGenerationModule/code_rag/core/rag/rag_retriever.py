"""
RAG Retriever для поиска релевантных фрагментов кода.
Использует Qdrant для векторного поиска и реранкинг для улучшения результатов.
"""

from typing import List, Optional, Dict, Any, Union
import logging
from dataclasses import dataclass, field

from ..parser.code_structure import CodeElement, ProgrammingLanguage, CodeElementType
from ..embeddings import CodeEmbedder
from ..vector_db.qdrant_client import QdrantClient
from ..vector_db.schemas import SearchFilters

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Конфигурация для RAG Retriever."""
    top_k: int = 5
    score_threshold: float = 0.0
    use_reranking: bool = True
    reranker_model: Optional[str] = None
    max_context_length: int = 8000
    min_score: float = 0.0
    rerank_weights: tuple = (0.3, 0.7)

    def __post_init__(self):
        """Валидация конфигурации."""
        if self.top_k <= 0:
            raise ValueError("top_k должен быть больше 0")

        if not (0.0 <= self.score_threshold <= 1.0):
            raise ValueError("score_threshold должен быть в диапазоне [0.0, 1.0]")

        if self.max_context_length <= 0:
            raise ValueError("max_context_length должен быть больше 0")

        # ✅ Валидация rerank_weights
        if len(self.rerank_weights) != 2:
            raise ValueError("rerank_weights должен содержать 2 значения")
        if not all(0.0 <= w <= 1.0 for w in self.rerank_weights):
            raise ValueError("Веса в rerank_weights должны быть в диапазоне [0.0, 1.0]")
        if sum(self.rerank_weights) != 1.0:
            logger.warning(f"Сумма весов {sum(self.rerank_weights)} != 1.0, "
                           f"результаты могут быть масштабированы")

@dataclass
class SearchResult:
    """Результат поиска кода."""
    element: CodeElement
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def element_name(self) -> str:
        """Имя элемента."""
        return self.element.name

    @property
    def element_type(self) -> str:
        """Тип элемента."""
        return self.element.type.value

    @property
    def language(self) -> str:
        """Язык программирования."""
        return self.element.language.value

    @property
    def source_preview(self) -> str:
        """Превью исходного кода (первые 200 символов)."""
        source = self.element.source_code
        if len(source) > 200:
            return source[:200] + "..."
        return source

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "element_name": self.element_name,
            "element_type": self.element_type,
            "language": self.language,
            "score": self.score,
            "rank": self.rank,
            "signature": self.element.signature,
            "docstring": self.element.docstring,
            "source_preview": self.source_preview,
            "location": self.element.location.to_dict(),
            "metadata": self.metadata,
            "element": self.element.to_dict(),
        }

    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, {self.element_name})"


class SearchHit:
    """Обёртка для унификации доступа к результатам поиска."""

    def __init__(self, data: Union[dict, Any]):
        self._data = data

    @property
    def payload(self):
        if isinstance(self._data, dict):
            return self._data.get('payload', {})
        return getattr(self._data, 'payload', {})

    @property
    def score(self):
        if isinstance(self._data, dict):
            return self._data.get('score', 0.0)
        return getattr(self._data, 'score', 0.0)

    @property
    def id(self):
        if isinstance(self._data, dict):
            return self._data.get('id', '')
        return getattr(self._data, 'id', '')


@dataclass
class RAGResponse:
    """Ответ RAG-системы с источниками и сгенерированным ответом."""

    query: str
    answer: str
    sources: List[SearchResult]
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [
                {
                    "element_name": src.element.name,
                    "element_type": src.element.type.value,
                    "score": src.score,
                    "rank": src.rank,
                    "signature": src.element.signature,
                    "file_path": str(src.element.location.file_path),
                }
                for src in self.sources
            ],
            "confidence": self.confidence,
            "metadata": self.metadata
        }

    @property
    def top_source(self) -> Optional[SearchResult]:
        """Получить самый релевантный источник."""
        return self.sources[0] if self.sources else None

    @property
    def has_sources(self) -> bool:
        """Проверить наличие источников."""
        return len(self.sources) > 0

    def __repr__(self) -> str:
        return (
            f"RAGResponse(query='{self.query[:50]}...', "
            f"sources={len(self.sources)}, confidence={self.confidence:.2f})"
        )

class RAGRetriever:
    """
    RAG Retriever для поиска релевантных фрагментов кода.

    Использует:
    - Векторный поиск в Qdrant
    - Фильтрацию по языку, типу элемента, репозиторию
    - Реранкинг результатов
    - Гибридный поиск (векторный + текстовый)
    """

    def __init__(
            self,
            collection_name: str,
            embedder: CodeEmbedder,
            qdrant_client: Optional[QdrantClient] = None,
            qdrant_location: Optional[str] = None,
            qdrant_url: str = "http://localhost:6333",
            qdrant_api_key: Optional[str] = None,
            use_reranking: bool = True,
            reranker_model: Optional[str] = None
    ):
        """
        Инициализация RAG Retriever.

        Args:
            collection_name: Имя коллекции в Qdrant
            embedder: Embedder для векторизации запросов
            qdrant_client: Клиент Qdrant (опционально)
            qdrant_url: URL Qdrant сервера
            qdrant_api_key: API ключ для Qdrant Cloud
            use_reranking: Использовать реранкинг
            reranker_model: Модель для реранкинга (опционально)
        """
        self.collection_name = collection_name
        self.embedder = embedder
        self.use_reranking = use_reranking
        self.reranker_model = reranker_model

        # Инициализация Qdrant клиента
        if qdrant_client:
            self.qdrant_client = qdrant_client
        elif qdrant_location:
            self.qdrant_client = QdrantClient(location=qdrant_location)
        else:
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )

        # Инициализация reranker если нужен
        self.reranker = None
        if use_reranking:
            self._init_reranker(reranker_model)

        logger.info(f"RAG Retriever инициализирован для коллекции '{collection_name}'")

    def _init_reranker(self, model_name: Optional[str] = None):
        """Инициализация reranker модели."""
        try:
            from sentence_transformers import CrossEncoder

            # Используем cross-encoder для реранкинга
            model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.reranker = CrossEncoder(model_name)
            logger.info(f"Reranker инициализирован: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers не установлен. Реранкинг недоступен.")
            self.use_reranking = False
        except Exception as e:
            logger.error(f"Ошибка инициализации reranker: {e}")
            self.use_reranking = False

    def search(
            self,
            query: str,
            top_k: int = 10,
            language: Optional[ProgrammingLanguage] = None,
            element_type: Optional[CodeElementType] = None,
            repository_name: Optional[str] = None,
            score_threshold: float = 0.0,
            rerank: bool = True,
            rerank_weights: tuple = (0.3, 0.7)
    ) -> List[SearchResult]:
        """
        Поиск релевантных фрагментов кода.

        Args:
            query: Текстовый запрос
            top_k: Количество результатов
            language: Фильтр по языку программирования
            element_type: Фильтр по типу элемента
            repository_name: Фильтр по репозиторию
            score_threshold: Минимальный score для результатов. (минимальная оценка результата). Ниже её результаты не будут возвращены.
            rerank: Применять реранкинг
            rerank_weights: Веса для комбинирования (vector_weight, rerank_weight)

        Returns:
            Список SearchResult, отсортированный по релевантности
        """

        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}")

        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []

        if top_k <= 0:
            raise ValueError(f"top_k = {top_k} должен быть больше 0")

        if score_threshold < 0.0:
            raise ValueError(f"score_threshold = {score_threshold} не может быть отрицательным числом")

        try:
            # Векторизуем запрос
            query_vector = self.embedder.encode_query(query)

            # Формируем фильтры
            filters = self._build_filters(language, element_type, repository_name)

            # Увеличиваем количество результатов для реранкинга
            search_limit = top_k * 3 if rerank and self.use_reranking else top_k

            # Выполняем векторный поиск
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=search_limit,
                filters=filters,
                score_threshold=score_threshold
            )

            # ✅ Оборачиваем результаты для унификации доступа
            wrapped_results = [SearchHit(hit) for hit in search_results]

            # Преобразуем в SearchResult
            results = []
            for idx, hit in enumerate(wrapped_results):
                element = self._reconstruct_element(hit.payload)
                result = SearchResult(
                    element=element,
                    score=hit.score,
                    rank=idx + 1,
                    metadata={
                        'vector_score': hit.score,
                        'point_id': hit.id
                    }
                )
                results.append(result)

            # Применяем реранкинг
            if rerank and self.use_reranking and len(results) > 0:
                results = self._rerank_results(query, results, top_k, rerank_weights)
            else:
                results = results[:top_k]

            logger.info(f"Найдено {len(results)} результатов для запроса: '{query[:50]}...'")

            return results

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []

    def search_similar_code(
            self,
            source_code: str,
            top_k: int = 10,
            exclude_self: bool = True,
            same_language_only: bool = True
    ) -> List[SearchResult]:
        """
        Поиск похожих фрагментов кода.

        Args:
            source_code: код для поиска похожих
            top_k: Количество результатов
            exclude_self: Исключить сам элемент из результатов
            same_language_only: Искать только в том же языке

        Returns:
            Список похожих элементов
        """
        # Векторизуем исходный элемент
        query_vector = self.embedder.encode_text(source_code)

        # Поиск
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k + 1 if exclude_self else top_k,
        )

        # ✅ Оборачиваем результаты для унификации доступа
        wrapped_results = [SearchHit(hit) for hit in search_results]

        # Преобразуем результаты
        results = []
        for idx, hit in enumerate(wrapped_results):
            element = self._reconstruct_element(hit.payload)

            result = SearchResult(
                element=element,
                score=hit.score,
                rank=idx + 1,
                metadata={'similarity_search': True}
            )
            results.append(result)

        return results[:top_k]

    def search_by_signature(
            self,
            signature: str,
            language: Optional[ProgrammingLanguage] = None,
            fuzzy: bool = True,
            top_k: int = 5
    ) -> List[SearchResult]:
        """
        Поиск по сигнатуре функции/метода.

        Args:
            signature: Сигнатура для поиска
            language: Язык программирования
            fuzzy: Нечеткий поиск (через векторизацию)
            top_k: Количество результатов

        Returns:
            Список найденных элементов
        """
        filters = Optional[SearchFilters] = None
        if fuzzy:
            # Используем векторный поиск
            query = f"Function signature: {signature}"
            return self.search(
                query=query,
                top_k=top_k,
                language=language,
                element_type=CodeElementType.FUNCTION
            )
        else:
            # Точный поиск по метаданным
            filters = SearchFilters(signature=signature)

            if language:
                filters.language = language

            # Получаем все результаты с этой сигнатурой
            results, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                filters=filters,
                limit=top_k
            )

            # ✅ Оборачиваем результаты для унификации доступа
            wrapped_results = [SearchHit(hit) for hit in results]

            search_results = []
            for idx, point in enumerate(wrapped_results):
                element = self._reconstruct_element(point.payload)
                result = SearchResult(
                    element=element,
                    score=1.0,
                    rank=idx + 1,
                    metadata={'exact_match': True}
                )
                search_results.append(result)

            return search_results

    def search_by_docstring(
            self,
            docstring_query: str,
            language: Optional[ProgrammingLanguage] = None,
            top_k: int = 10
    ) -> List[SearchResult]:
        """
        Поиск по содержимому документации.

        Args:
            docstring_query: Запрос по документации
            language: Язык программирования
            top_k: Количество результатов

        Returns:
            Список элементов с релевантной документацией
        """
        query = f"Documentation: {docstring_query}"
        return self.search(
            query=query,
            top_k=top_k,
            language=language
        )

    def search_by_complexity(
            self,
            min_complexity: int = 0,
            max_complexity: int = 100,
            language: Optional[ProgrammingLanguage] = None,
            top_k: int = 10
    ) -> List[SearchResult]:
        """
        Поиск элементов по сложности.

        Args:
            min_complexity: Минимальная сложность
            max_complexity: Максимальная сложность
            language: Язык программирования
            top_k: Количество результатов

        Returns:
            Список элементов в диапазоне сложности
        """
        filters = SearchFilters(min_complexity=min_complexity, max_complexity=max_complexity)

        if language:
            filters.language = language

        # Получаем результаты
        results, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            filters=filters,
            limit=top_k
        )

        # ✅ Оборачиваем результаты для унификации доступа
        wrapped_results = [SearchHit(hit) for hit in results]

        search_results = []
        for idx, point in enumerate(wrapped_results):
            element = self._reconstruct_element(point.payload)
            result = SearchResult(
                element=element,
                score=1.0,
                rank=idx + 1,
                metadata={'complexity': element.complexity}
            )
            search_results.append(result)

        # Сортируем по сложности (по убыванию)
        search_results.sort(key=lambda x: x.element.complexity, reverse=True)

        return search_results

    def _build_filters(
            self,
            language: Optional[ProgrammingLanguage] = None,
            element_type: Optional[CodeElementType] = None,
            repository_name: Optional[str] = None,
            file_path: Optional[str] = None,
    ) -> Optional[SearchFilters]:
        """Построение фильтров для Qdrant."""
        filters = SearchFilters()

        if language:
            filters.language = language

        if element_type:
            filters.element_type = element_type

        if repository_name:
            filters.repository_name = repository_name

        if file_path:
            filters.file_path = file_path

        return filters

    def _reconstruct_element(self, payload: Dict[str, Any]) -> CodeElement:
        """
        Восстановление CodeElement из payload Qdrant.

        Args:
            payload: Payload точки в Qdrant

        Returns:
            CodeElement
        """
        from ..parser.code_structure import CodeElement

        # Простая десериализация (можно улучшить)
        return CodeElement.from_dict(payload)

    def _rerank_results(
            self,
            query: str,
            results: List[SearchResult],
            top_k: int,
            rerank_weights: tuple = (0.3, 0.7)
    ) -> List[SearchResult]:
        """
        Реранкинг результатов с использованием cross-encoder.

        Args:
            query: Исходный запрос
            results: Результаты векторного поиска
            top_k: Количество итоговых результатов
            rerank_weights: Кортеж весов (vector_weight, rerank_weight)
                        По умолчанию (0.3, 0.7)

        Returns:
            Переранжированные результаты
        """
        if not self.reranker or len(results) == 0:
            return results

        try:
            # Подготавливаем пары (query, document) для reranker
            pairs = []
            for result in results:
                # Формируем текст документа для реранкинга
                doc_text = f"{result.element.signature}\n{result.element.docstring or ''}"
                pairs.append([query, doc_text])

            # Получаем scores от reranker
            rerank_scores = self.reranker.predict(pairs)
            vector_weight, rerank_weight = rerank_weights

            # Обновляем scores и пересортируем
            for result, rerank_score in zip(results, rerank_scores):
                result.metadata['rerank_score'] = float(rerank_score)
                result.metadata['vector_score'] = result.score  # Сохраняем оригинальный score
                # Комбинированный score с настраиваемыми весами
                result.score = vector_weight * result.score + rerank_weight * float(rerank_score)

            # Сортируем по новым scores
            results.sort(key=lambda x: x.score, reverse=True)

            # Обновляем ранги
            for idx, result in enumerate(results[:top_k]):
                result.rank = idx + 1

            logger.debug(f"Reranked {len(results)} results with weights {rerank_weights}")

            return results[:top_k]

        except Exception as e:
            logger.error(f"Ошибка при reranking: {e}")
            return results[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.

        Returns:
            Словарь со статистикой
        """
        try:
            collection_info = self.qdrant_client.get_collection_info(self.collection_name)

            return {
                "collection_name": collection_info.get('name', 'unknown'),
                "vectors_count": collection_info.get('vectors_count', 'unknown'),
                "points_count": collection_info.get('points_count', 'unknown'),
                "status": collection_info.get('status', 'unknown'),
                "optimizer_status": collection_info.get('optimizer_status', 'unknown'),
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}

    def __repr__(self) -> str:
        return f"RAGRetriever(collection='{self.collection_name}', reranking={self.use_reranking})"

    def search_by_file(self, file_path, repository_name) -> List[SearchResult]:
        """
        Выполняет поиск в коллекции Qdrant по указанному файлу и имени репозитория.

        Функция строит фильтры на основе переданных параметров, осуществляет запрос к Qdrant-клиенту
        и возвращает список найденных элементов, обернутых в унифицированный формат SearchResult.

        :param file_path: Путь к файлу, по которому осуществляется поиск
        :param repository_name: Имя репозитория, в котором осуществляется поиск
        :return: Список объектов SearchResult, содержащих найденные элементы с оценками и рангами

        Пример использования:
            results = rag_retriever.search_by_file(
                file_path="src/main.py",
                repository_name="my_project"
            )

        Возвращает список результатов с полями:
            - element: найденный элемент данных
            - score: оценка соответствия (всегда 1.0 в текущей реализации)
            - rank: порядковый номер результата
        """
        filters = self._build_filters(repository_name=repository_name, file_path=file_path)
        # Получаем результаты
        results, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            filters=filters
        )

        # ✅ Оборачиваем результаты для унификации доступа
        wrapped_results = [SearchHit(hit) for hit in results]

        search_results = []
        for idx, point in enumerate(wrapped_results):
            element = self._reconstruct_element(point.payload)
            result = SearchResult(
                element=element,
                score=1.0,
                rank=idx + 1
            )
            search_results.append(result)

        return search_results



# Удобная функция для создания retriever
def create_retriever(
        collection_name: str,
        embedder: CodeEmbedder,
        **kwargs
) -> RAGRetriever:
    """
    Создание RAG Retriever.

    Args:
        collection_name: Имя коллекции
        embedder: Embedder для векторизации
        **kwargs: Дополнительные параметры для RAGRetriever

    Returns:
        RAGRetriever
    """
    return RAGRetriever(
        collection_name=collection_name,
        embedder=embedder,
        **kwargs
    )
