"""
Клиент для работы с Qdrant векторной базой данных.
Управляет коллекциями, индексацией и поиском кода.
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
import logging
import uuid
from datetime import datetime

import numpy as np
from qdrant_client import QdrantClient as QdrantClientBase
from qdrant_client.http.models import PointVectors, SearchParams
from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue
    )

from .schemas import (
    CodePayload, CollectionSchema, IndexConfig,
    PointData, BatchInsertData, SearchFilters,
    DEFAULT_CODE_SCHEMA
)
from ..parser.code_structure import CodeElement, ModuleInfo

logger = logging.getLogger(__name__)


class QdrantClient:
    """
    Обертка над Qdrant клиентом для работы с кодом.

    Упрощает:
    - Создание коллекций
    - Индексацию кода
    - Поиск с фильтрацией
    - Батч-операции
    - Управление коллекциями
    """

    def __init__(
            self,
            location: Optional[str] = None,
            url: Optional[str] = None,
            host: Optional[str] = None,
            port: Optional[int] = None,
            path: Optional[str] = None,
            api_key: Optional[str] = None,
            timeout: int = 30,
            prefer_grpc: bool = False,
            grpc_port: int = 6334,
            **kwargs
    ):
        """
        Инициализация Qdrant клиента.

        Args:
            location: If `":memory:"` - use in-memory Qdrant instance.
                        If `str` - use it as a `url` parameter.
                        If `None` - use default values for `host` and `port`.
            url: URL Qdrant сервера
            api_key: API ключ (для Qdrant Cloud)
            timeout: Таймаут запросов (секунды)
            prefer_grpc: Использовать gRPC вместо HTTP
            grpc_port: Порт для gRPC
        """
        self.timeout = timeout
        # Инициализация базового клиента
        client_params = { }

        if location is not None:
            client_params["location"] = location
        elif url is not None:
            client_params["url"] = url
        elif path is not None:
            client_params["path"] = path
        elif host is not None:
            client_params["host"] = host
            if port is not None:
                client_params["port"] = port
        else:
            # По умолчанию подключаемся к localhost
            client_params["host"] = "localhost"
            client_params["port"] = 6333

        if api_key:
            client_params["api_key"] = api_key

        if prefer_grpc:
            client_params["prefer_grpc"] = True
            client_params["grpc_port"] = grpc_port

        # Добавляем дополнительные параметры
        client_params.update(kwargs)

        try:
            self.client = QdrantClientBase(**client_params)
            logger.info(f"Подключен к Qdrant: {client_params}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {e}")
            raise

    def create_collection(
            self,
            schema: CollectionSchema,
            recreate: bool = False,
            unindexed_filtering_retrieve: bool = False,
            indexing_threshold: Optional[int] = 10000,
    ) -> bool:
        """
        Создание коллекции в Qdrant.

        Args:
            schema: Схема коллекции
            recreate: Пересоздать если существует
            unindexed_filtering_retrieve: Если для параметра unindexed_filtering_retrieve установлено значение false, фильтрация по неиндексированному ключу полезной нагрузки будет невозможна
            indexing_threshold: позволяет выполнять индексацию сегментов, в которых хранится более 10 000 КБ векторов

        Returns:
            True если успешно создана
        """
        try:
            collection_name = schema.collection_name

            # Проверяем существование
            if self.collection_exists(collection_name):
                if recreate:
                    logger.info(f"Удаление существующей коллекции: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    # Вместо тихого возврата — кидаем исключение
                    raise Exception(f"Коллекция '{collection_name}' уже существует")

            # Создаем коллекцию
            logger.info(f"Создание коллекции: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=schema.to_vector_params(),
                hnsw_config=schema.to_hnsw_config_params(),
                strict_mode_config=schema.to_strict_config_params(unindexed_filtering_retrieve=unindexed_filtering_retrieve),
                optimizers_config=schema.to_optimizers_config(indexing_threshold=indexing_threshold),
            )

            # Создаем индексы
            self._create_indexes(collection_name)

            logger.info(f"Коллекция {collection_name} успешно создана")
            return True

        except Exception as e:
            logger.error(f"Ошибка создания коллекции: {e}")
            # Передаём исключение дальше, чтобы тест его ловил
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """
        Проверка существования коллекции.

        Args:
            collection_name: Имя коллекции

        Returns:
            True если коллекция существует
        """
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.error(f"Ошибка проверки коллекции: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """
        Удаление коллекции.

        Args:
            collection_name: Имя коллекции

        Returns:
            True если успешно удалена
        """
        try:
            result = self.client.delete_collection(collection_name)
            if result:
                logger.info(f"Коллекция {collection_name} удалена")
            else:
                logger.info(f"Ошибка удаления коллекции: {collection_name}")
            return result
        except Exception as e:
            logger.error(f"Ошибка удаления коллекции: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о коллекции.

        Args:
            collection_name: Имя коллекции

        Returns:
            Словарь с информацией или None
        """
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
                "optimizer_status": info.optimizer_status.value if info.optimizer_status else "unknown",
                "vector_size": info.config.params.vectors.size if info.config else None,
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о коллекции: {e}")
            return None

    def list_collections(self) -> List[str]:
        """
        Список всех коллекций.

        Returns:
            Список имен коллекций
        """
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Ошибка получения списка коллекций: {e}")
            return []

    def _create_indexes(self, collection_name: str) -> None:
        """Создание индексов для коллекции."""
        indexes = IndexConfig.get_default_indexes()

        for index_config in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=index_config.field_name,
                    field_schema=index_config.schema_type
                )
                logger.debug(f"Создан индекс: {index_config.field_name}")
            except Exception as e:
                logger.warning(f"Не удалось создать индекс {index_config.field_name}: {e}")

    def insert_code_element(
            self,
            collection_name: str,
            element: CodeElement,
            vector: List[float],
            element_id: Optional[str] = None,
            repository_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Вставка одного элемента кода.

        Args:
            collection_name: Имя коллекции
            element: Элемент кода
            vector: Вектор элемента
            element_id: ID элемента (генерируется автоматически если None)
            repository_info: Информация о репозитории

        Returns:
            ID вставленного элемента
        """
        # Генерируем ID если не указан
        if not element_id:
            element_id = str(uuid.uuid4())

        # Создаем payload
        payload = CodePayload.from_code_element(element, repository_info)

        # Создаем точку
        point = PointStruct(
            id=element_id,
            vector=vector,
            payload=payload.to_dict()
        )

        # Вставляем
        self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        logger.debug(f"Вставлен элемент: {element.name} (ID: {element_id})")

        return element_id

    def insert_point(
            self,
            collection_name: str,
            vector: Union[List[float], np.ndarray],
            payload: Dict[str, Any],
            point_id: Optional[str] = None
    ) -> str:
        """
        Вставка одной точки по сигнатуре тестов:
          - готовый payload
          - point_id вместо element_id
        """
        if point_id is None:
            point_id = str(uuid.uuid4())

        # Приводим numpy.ndarray к list
        vec: List[float] = vector.tolist() if isinstance(vector, np.ndarray) else vector

        point = PointStruct(
            id=point_id,
            vector=vec,
            payload=payload
        )

        self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        logger.debug(f"Inserted point {point_id} into {collection_name}")
        return point_id

    import numpy as np
    import uuid

    def insert_batch(
            self,
            collection_name: str,
            batch_data: Optional[BatchInsertData] = None,
            vectors: Optional[Union[Dict[str, List[float]], List[List[float]], np.ndarray]] = None,
            payloads: Optional[Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
            batch_size: int = 100
    ) -> Tuple[int, int]:
        """
        Батч-вставка элементов.

        Args:
        collection_name: Имя коллекции
        batch_data: BatchInsertData
        vectors:
          - dict {id: vector}
          - list of vectors
          - numpy.ndarray shape (n, dim)
        payloads:
          - dict {id: payload}
          - list of payload dicts (one per vector)
        batch_size: Размер батча

        Returns:
        - при dict-входе: (inserted, total)
        """
        points: List[PointStruct] = []

        if batch_data is not None:
            # Случай BatchInsertData
            points = batch_data.to_point_structs()


        elif isinstance(vectors, dict):
            # Словарь {id: vector}
            for vec_id, vec in vectors.items():
                pid = str(uuid.uuid4())
                pl = payloads.get(vec_id) if isinstance(payloads, dict) else {}
                points.append(PointStruct(id=pid, vector=vec, payload=pl))
            # Для dict возвращаем (inserted, total) позже

        elif isinstance(vectors, np.ndarray):
            # Случай ndarray + список payloads
            if not isinstance(payloads, list):
                raise ValueError("Для numpy.ndarray vectors требуется список payloads той же длины")
            if vectors.shape[0] != len(payloads):
                raise ValueError("Число векторов и payloads должно совпадать")

            for i, vector in enumerate(vectors):
                pd = payloads[i]
                # Игнорируем pd["id"], всегда генерируем UUID
                point_id = str(uuid.uuid4())
                # Удаляем ключ 'id' из payload для хранения в Qdrant
                cleaned_payload = {k: v for k, v in pd.items() if k != "id"} if isinstance(pd, dict) else {}
                points.append(PointStruct(id=point_id, vector=vector.tolist(), payload=cleaned_payload))

        elif isinstance(vectors, list):
            # Список векторов + список payloads
            if not isinstance(payloads, list):
                raise ValueError("Для списка векторов требуется список payloads той же длины")
            if len(vectors) != len(payloads):
                raise ValueError("Число векторов и payloads должно совпадать")
            for vec, pd in zip(vectors, payloads):
                # Игнорируем pd["id"], всегда генерируем UUID
                pid = str(uuid.uuid4())
                pl = {k: v for k, v in pd.items() if k != "id"} if isinstance(pd, dict) else {}
                points.append(PointStruct(id=pid, vector=vec, payload=pl))

        else:
            raise ValueError("insert_batch требует либо batch_data, либо vectors (dict или ndarray)")

        total = len(points)
        inserted = 0
        failed_batches = []

        for i in range(0, total, batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                inserted += len(batch)
                logger.debug(f"Вставлен батч: {inserted}/{total}")
            except Exception as e:
                logger.error(f"Ошибка вставки батча {i}-{i + batch_size}: {e}")
                failed_batches.append((i, batch))

        # Retry failed batches
        for i, batch in failed_batches:
            try:
                self.client.upsert(collection_name=collection_name, points=batch)
                inserted += len(batch)
            except Exception as e:
                logger.error(f"Повторная ошибка батча {i}: {e}")

        logger.info(f"Вставлено {inserted}/{total} элементов")
        return inserted, total

    def insert_module(
            self,
            collection_name: str,
            module: ModuleInfo,
            vectors: Dict[str, List[float]],
            repository_info: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Вставка всех элементов из модуля.

        Args:
            collection_name: Имя коллекции
            module: ModuleInfo с элементами кода
            vectors: Словарь {element_id: vector}
            repository_info: Информация о репозитории

        Returns:
            Количество вставленных элементов
        """
        points = []

        for element in module.all_elements:
            element_id = f"{module.module_name}::{element.qualified_name}"

            if element_id not in vectors:
                logger.warning(f"Вектор не найден для: {element_id}")
                continue

            payload = CodePayload.from_code_element(element, repository_info)

            point = PointData(
                id=element_id,
                vector=vectors[element_id],
                payload=payload
            )

            points.append(point)

        if points:
            batch_data = BatchInsertData(points=points)
            inserted, total = self.insert_batch(collection_name, batch_data)
            return inserted

        return 0

    def search(
            self,
            collection_name: str,
            query_vector: Union[List[float], np.ndarray],
            limit: int = 10,
            filters: Optional[SearchFilters] = None,
            score_threshold: Optional[float] = None,
            with_payload: bool = True,
            with_vectors: bool = False,
            indexed_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Векторный поиск в коллекции.

        Args:
            collection_name: Имя коллекции
            query_vector: Вектор запроса
            limit: Количество результатов
            filters: Фильтры поиска
            score_threshold: Минимальный score (минимальная оценка результата). Ниже её результаты не будут возвращены.
            with_payload: Включить payload в результаты
            with_vectors: Включить векторы в результаты
            indexed_only: По умолчанию True - поиск только по индексам

        Returns:
            Список результатов поиска
        """
        try:
            # Конвертируем фильтры
            qdrant_filter = filters.to_qdrant_filter() if filters else None

            # Приводим numpy.ndarray к list
            query_vec: List[float] = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

            search_params = None
            if not indexed_only:
                search_params = SearchParams(indexed_only=indexed_only,hnsw_ef = 128, exact=False)

            # Выполняем поиск
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vec,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors,
                search_params=search_params
            )

            # ✅ Универсальное извлечение результатов
            if hasattr(response, 'points'):
                # query_points() возвращает QueryResponse с .points
                hits = response.points
            elif hasattr(response, 'result'):
                # Некоторые методы возвращают .result
                hits = response.result
            elif isinstance(response, list):
                # Старые версии могли возвращать список напрямую
                hits = response
            else:
                hits = []

            # Форматируем результаты
            formatted_results = []
            for hit in hits:
                result = {
                    "id": str(hit.id),  # ✅ Преобразуем UUID в строку
                    "score": hit.score,
                }

                if with_payload and hit.payload:
                    result["payload"] = hit.payload

                if with_vectors and hit.vector:
                    result["vector"] = hit.vector

                formatted_results.append(result)

            logger.debug(f"Найдено {len(formatted_results)} результатов")

            return formatted_results

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []

    def search_by_id(
            self,
            collection_name: str,
            point_id: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих элементов по ID точки.

        Args:
            collection_name: Имя коллекции
            point_id: ID точки для поиска похожих
            limit: Количество результатов

        Returns:
            Список похожих элементов
        """
        try:
            # Получаем точку
            point = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_vectors=True
            )

            if not point:
                logger.warning(f"Точка {point_id} не найдена")
                return []

            # Используем её вектор для поиска
            query_vector = point[0].vector

            return self.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit + 1  # +1 чтобы исключить саму точку
            )[1:]  # Исключаем первый результат (сама точка)

        except Exception as e:
            logger.error(f"Ошибка поиска по ID: {e}")
            return []

    def get_by_id(
            self,
            collection_name: str,
            point_id: str,
            with_vector: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Получение элемента по ID.

        Args:
            collection_name: Имя коллекции
            point_id: ID точки
            with_vector: Включить вектор

        Returns:
            Данные элемента или None
        """
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_vectors=with_vector,
                with_payload=True
            )

            if points:
                point = points[0]
                result = {
                    "id": point.id,
                    "payload": point.payload
                }

                if with_vector:
                    result["vector"] = point.vector

                return result

            return None

        except Exception as e:
            logger.error(f"Ошибка получения элемента: {e}")
            return None

    def delete_by_id(
            self,
            collection_name: str,
            point_ids: List[str]
    ) -> bool:
        """
        Удаление элементов по ID.

        Args:
            collection_name: Имя коллекции
            point_ids: Список ID для удаления

        Returns:
            True если успешно
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )
            logger.info(f"Удалено {len(point_ids)} элементов")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления элементов: {e}")
            return False

    def delete_by_filter(
            self,
            collection_name: str,
            filters: SearchFilters
    ) -> bool:
        """
        Удаление элементов по фильтру.

        Args:
            collection_name: Имя коллекции
            filters: Фильтры

        Returns:
            True если успешно
        """
        try:
            qdrant_filter = filters.to_qdrant_filter()

            if not qdrant_filter:
                logger.warning("Пустой фильтр для удаления")
                return False

            self.client.delete(
                collection_name=collection_name,
                points_selector=qdrant_filter
            )
            logger.info("Элементы удалены по фильтру")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления по фильтру: {e}")
            return False

    def count_points(
            self,
            collection_name: str,
            filters: Optional[SearchFilters] = None
    ) -> int:
        """
        Подсчет количества точек.

        Args:
            collection_name: Имя коллекции
            filters: Фильтры (опционально)

        Returns:
            Количество точек
        """
        try:
            qdrant_filter = filters.to_qdrant_filter() if filters else None

            result = self.client.count(
                collection_name=collection_name,
                count_filter=qdrant_filter
            )

            return result.count
        except Exception as e:
            logger.error(f"Ошибка подсчета точек: {e}")
            return 0

    def scroll(
            self,
            collection_name: str,
            limit: int = 100,
            filters: Optional[SearchFilters] = None,
            offset: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Скроллинг по коллекции.

        Args:
            collection_name: Имя коллекции
            limit: Количество элементов
            filters: Фильтры
            offset: Offset для пагинации

        Returns:
            Tuple (список элементов, следующий offset)
        """
        try:
            qdrant_filter = filters.to_qdrant_filter() if filters else None

            results, next_offset = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                scroll_filter=qdrant_filter,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            formatted_results = [
                {"id": point.id, "payload": point.payload}
                for point in results
            ]

            return formatted_results, next_offset

        except Exception as e:
            logger.error(f"Ошибка скроллинга: {e}")
            return [], None

    def update_payload(
            self,
            collection_name: str,
            point_id: str,
            payload: Dict[str, Any],
            merge: bool = True
    ) -> bool:
        """
        Обновление payload точки.

        Args:
            collection_name: Имя коллекции
            point_id: ID точки
            payload: Новый payload
            merge: Объединить с существующим (True) или заменить (False)

        Returns:
            True если успешно
        """
        try:
            if merge:
                self.client.set_payload(
                    collection_name=collection_name,
                    payload=payload,
                    points=[point_id]
                )
            else:
                self.client.overwrite_payload(
                    collection_name=collection_name,
                    payload=payload,
                    points=[point_id]
                )

            logger.debug(f"Payload обновлен для {point_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка обновления payload: {e}")
            return False

    def update_vector(
            self,
            collection_name: str,
            point_id: str,
            vector: Union[List[float], np.ndarray],
    ) -> bool:
        """
        Обновление вектора существующей точки.

        Args:
            collection_name: Имя коллекции
            point_id: ID точки
            vector: Новый вектор

        Returns:
            True если успешно
        """
        try:
            vec: List[float] = vector.tolist() if isinstance(vector, np.ndarray) else vector
            self.client.update_vectors(
                collection_name=collection_name,
                points=[PointVectors(id=point_id, vector=vec)]
            )
            logger.debug(f"Vector updated for {point_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка обновления вектора: {e}")
            return False

    def health_check(self) -> bool:
        """
        Проверка здоровья Qdrant сервера.

        Returns:
            True если сервер доступен
        """
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant недоступен: {e}")
            return False

    def close(self) -> None:
        """Закрытие соединения."""
        try:
            self.client.close()
            logger.info("Соединение с Qdrant закрыто")
        except Exception as e:
            logger.error(f"Ошибка закрытия соединения: {e}")

    def __enter__(self):
        """Контекстный менеджер: вход."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход."""
        self.close()

    def __repr__(self) -> str:
        return f"QdrantClient(url='{self.url}')"


# Удобная функция для создания клиента
def create_qdrant_client(
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        **kwargs
) -> QdrantClient:
    """
    Создание Qdrant клиента.

    Args:
        url: URL Qdrant сервера
        api_key: API ключ
        **kwargs: Дополнительные параметры

    Returns:
        QdrantClient
    """
    return QdrantClient(url=url, api_key=api_key, **kwargs)
