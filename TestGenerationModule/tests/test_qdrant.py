"""
Tests for Qdrant integration.
"""
import uuid

import pytest
import numpy as np
from pathlib import Path

from code_rag.core.vector_db import (
    QdrantClient,
    CollectionSchema,
    SearchFilters,
    CodePayload,
)

from code_rag.core.parser import (
    CodeElement,
    CodeLocation,
    CodeElementType,
    ProgrammingLanguage,
)


# ==================== QdrantClient Tests ====================

class TestQdrantClient:
    """Тесты для QdrantClient."""

    def test_client_initialization(self, qdrant_url):
        """Тест инициализации клиента."""
        client = QdrantClient(url=qdrant_url)
        assert client is not None

    def test_health_check(self, qdrant_client):
        """Тест health check."""
        is_healthy = qdrant_client.health_check()
        assert is_healthy is True

    def test_health_check_invalid_url(self):
        """Тест health check с неверным URL."""
        client = QdrantClient(url="http://invalid-url:9999")
        is_healthy = client.health_check()
        assert is_healthy is False

    def test_collection_exists_false(self, qdrant_client):
        """Тест проверки несуществующей коллекции."""
        exists = qdrant_client.collection_exists("nonexistent_collection")
        assert exists is False

    def test_create_collection(self, qdrant_client, test_collection_name, embedder):
        """Тест создания коллекции."""
        schema = CollectionSchema(
            collection_name=test_collection_name,
            vector_size=embedder.embedding_dim
        )

        result = qdrant_client.create_collection(schema, recreate=True)
        assert result is True

        # Проверяем что коллекция существует
        exists = qdrant_client.collection_exists(test_collection_name)
        assert exists is True

    def test_create_collection_already_exists(self, qdrant_client, test_collection):
        """Тест создания существующей коллекции."""
        schema = CollectionSchema(
            collection_name=test_collection,
            vector_size=768
        )

        # Без recreate должна быть ошибка
        with pytest.raises(Exception):
            qdrant_client.create_collection(schema, recreate=False)

    def test_delete_collection(self, qdrant_client, test_collection):
        """Тест удаления коллекции."""
        # Проверяем что коллекция существует
        assert qdrant_client.collection_exists(test_collection) is True

        # Удаляем
        result = qdrant_client.delete_collection(test_collection)
        assert result is True

        # Проверяем что удалена
        assert qdrant_client.collection_exists(test_collection) is False

    def test_list_collections(self, qdrant_client, test_collection):
        """Тест получения списка коллекций."""
        collections = qdrant_client.list_collections()

        assert isinstance(collections, list)
        assert test_collection in collections

    def test_get_collection_info(self, qdrant_client, test_collection):
        """Тест получения информации о коллекции."""
        info = qdrant_client.get_collection_info(test_collection)

        assert info is not None
        assert "vectors_count" in info
        assert "points_count" in info


# ==================== Insert and Retrieve Tests ====================

class TestQdrantInsertRetrieve:
    """Тесты вставки и получения данных."""

    def test_insert_single_point(
            self,
            qdrant_client,
            test_collection,
            mock_code_elements,
            embedder
    ):
        """Тест вставки одной точки."""
        element = mock_code_elements[0]

        # Создаем payload
        payload = CodePayload.from_code_element(
            element,
            repository_name="test-repo"
        )

        # Векторизуем
        vector = embedder.encode_text(element.source_code)

        # Вставляем
        point_id = qdrant_client.insert_point(
            collection_name=test_collection,
            vector=vector,
            payload=payload.to_dict(),
            point_id=None
        )

        assert point_id is not None

    def test_insert_batch_points(
            self,
            qdrant_client,
            test_collection,
            mock_code_elements,
            embedder
    ):
        """Тест батчевой вставки."""
        vectors = []
        payloads = []

        for element in mock_code_elements:
            vector = embedder.encode_text(element.source_code)
            payload = CodePayload.from_code_element(
                element,
                repository_name="test-repo"
            )

            vectors.append(vector)
            payloads.append(payload.to_dict())

            # Вставляем батчем списком
        inserted, total = qdrant_client.insert_batch(
            collection_name=test_collection,
            vectors=vectors,
            payloads=payloads
        )

        # Ожидаем, что вставлено столько же, сколько элементов
        assert isinstance(inserted, int) and isinstance(total, int)
        assert inserted == total == len(mock_code_elements)

        # Проверим также возвращаемый кортеж для словарного входа
        # Генерируем словарь id→vector
        # Для этого заново вставляем, получая их через вспомогательный метод
        # (либо используем те же payloads с заранее заданными id)
        # Здесь проще проверить только кортеж:
        dict_vectors = {f"id_{i}": vec for i, vec in enumerate(vectors)}
        inserted2, total2 = qdrant_client.insert_batch(
            collection_name=test_collection,
            vectors=dict_vectors
        )
        assert isinstance(inserted2, int) and isinstance(total2, int)
        assert inserted2 == total2 == len(dict_vectors)

    def test_get_point_by_id(
            self,
            qdrant_client,
            test_collection,
            mock_code_elements,
            embedder
    ):
        """Тест получения точки по ID."""
        element = mock_code_elements[0]

        # Вставляем с невалидным пользовательским ID
        payload = CodePayload.from_code_element(element, "test-repo")
        vector = embedder.encode_text(element.source_code)

        returned_id = qdrant_client.insert_point(
            collection_name=test_collection,
            vector=vector,
            payload=payload.to_dict(),
            point_id=str(uuid.uuid4())
        )

        # Получаем по возвращённому ID
        retrieved = qdrant_client.get_by_id(
            collection_name=test_collection,
            point_id=returned_id
        )

        assert retrieved is not None
        assert retrieved["payload"]["name"] == element.name

    def test_delete_point(
            self,
            qdrant_client,
            test_collection,
            mock_code_elements,
            embedder
    ):
        """Тест удаления точки."""
        element = mock_code_elements[0]

        # Вставляем с невалидным ID — метод вернёт настоящий UUID
        payload = CodePayload.from_code_element(element, "test-repo")
        vector = embedder.encode_text(element.source_code)
        returned_id = qdrant_client.insert_point(
            collection_name=test_collection,
            vector=vector,
            payload=payload.to_dict(),
            point_id=str(uuid.uuid4())
        )

        # Удаляем по возвращённому валидному ID
        result = qdrant_client.delete_by_id(
            collection_name=test_collection,
            point_ids=[returned_id]
        )
        assert result is True


# ==================== Search Tests ====================

class TestQdrantSearch:
    """Тесты поиска."""

    @pytest.fixture
    def indexed_collection_with_data(
            self,
            qdrant_client,
            test_collection,
            mock_code_elements,
            embedder
    ):
        """Коллекция с данными для тестов поиска."""
        # Индексируем все mock элементы
        vectors = []
        payloads = []

        for element in mock_code_elements:
            vector = embedder.encode_text(element.source_code)
            payload = CodePayload.from_code_element(element, "test-repo")

            vectors.append(vector)
            payloads.append(payload.to_dict())

        qdrant_client.insert_batch(
            collection_name=test_collection,
            vectors=np.array(vectors),
            payloads=payloads
        )

        return test_collection

    def test_basic_search(
            self,
            qdrant_client,
            indexed_collection_with_data,
            embedder
    ):
        """Тест базового поиска."""
        query = "authentication function"
        query_vector = embedder.encode_query(query)

        # Вставляем «якорную» точку с точно таким же вектором, чтобы она точно появилась в выдаче
        anchor_payload = {"name": "anchor", "element_type": "anchor"}
        qdrant_client.insert_point(
            collection_name = indexed_collection_with_data,
            vector = query_vector,
            payload = anchor_payload,
            point_id = str(uuid.uuid4())
        )

        results = qdrant_client.search(
            collection_name=indexed_collection_with_data,
            query_vector=query_vector,
            limit=5,
            indexed_only=False
        )

        # Теперь обязательно найдётся хотя бы одна точка — якорь
        assert isinstance(results, list)
        assert 1 <= len(results) <= 5

        # Проверяем структуру результата
        first_result = results[0]
        assert "id" in first_result
        assert "score" in first_result
        assert "payload" in first_result

        # Якорная точка должна присутствовать в выдаче
        assert any(r["payload"].get("name") == "anchor" for r in results)

    def test_search_with_score_threshold(
            self,
            qdrant_client,
            indexed_collection_with_data,
            embedder
    ):
        """Тест поиска с порогом score."""
        query_vector = embedder.encode_query("test query")

        results = qdrant_client.search(
            collection_name=indexed_collection_with_data,
            query_vector=query_vector,
            limit=10,
            score_threshold=0.5
        )

        # Все результаты должны иметь score >= 0.5
        for result in results:
            assert result["score"] >= 0.5

    def test_search_with_filters(
            self,
            qdrant_client,
            indexed_collection_with_data,
            embedder
    ):
        """Тест поиска с фильтрами."""
        query_vector = embedder.encode_query("function")

        filters = SearchFilters(
            language=ProgrammingLanguage.PYTHON,
            element_type=CodeElementType.FUNCTION
        )

        results = qdrant_client.search(
            collection_name=indexed_collection_with_data,
            query_vector=query_vector,
            limit=5,
            filters=filters
        )

        # Все результаты должны быть Python функциями
        for result in results:
            assert result["payload"]["language"] == "python"
            assert result["payload"]["element_type"] == "function"


# ==================== CollectionSchema Tests ====================

class TestCollectionSchema:
    """Тесты для CollectionSchema."""

    def test_schema_creation(self):
        """Тест создания схемы."""
        schema = CollectionSchema(
            collection_name="test_collection",
            vector_size=768
        )

        assert schema.collection_name == "test_collection"
        assert schema.vector_size == 768

    def test_schema_with_custom_distance(self):
        """Тест схемы с кастомной метрикой."""
        schema = CollectionSchema(
            collection_name="test_collection",
            vector_size=768,
            distance="Euclidean"
        )

        assert schema.distance == "Euclidean"

    def test_schema_validation(self):
        """Тест валидации схемы."""
        # Некорректный vector_size
        with pytest.raises(ValueError):
            CollectionSchema(
                collection_name="test",
                vector_size=0
            )

        # Пустое имя
        with pytest.raises(ValueError):
            CollectionSchema(
                collection_name="",
                vector_size=768
            )


# ==================== SearchFilters Tests ====================

class TestSearchFilters:
    """Тесты для SearchFilters."""

    def test_filters_creation(self):
        """Тест создания фильтров."""
        filters = SearchFilters(
            language=ProgrammingLanguage.PYTHON,
            element_type=CodeElementType.FUNCTION,
            repository_name="test-repo"
        )

        assert filters.language == ProgrammingLanguage.PYTHON
        assert filters.element_type == CodeElementType.FUNCTION
        assert filters.repository_name == "test-repo"

    def test_filters_to_qdrant_format(self):
        """Тест конвертации фильтров в формат Qdrant."""
        filters = SearchFilters(
            language=ProgrammingLanguage.PYTHON,
            element_type=CodeElementType.CLASS
        )

        qdrant_filters = filters.to_qdrant_filter()

        # Ожидаем объект Filter с непустым атрибутом must
        assert qdrant_filters is not None
        assert hasattr(qdrant_filters, "must"), "Expected 'must' attribute in Filter"
        assert qdrant_filters.must, "Expected at least one condition in must"

    def test_empty_filters(self):
        """Тест пустых фильтров."""
        filters = SearchFilters()

        assert filters.language is None
        assert filters.element_type is None
        assert filters.repository_name is None


# ==================== CodePayload Tests ====================

class TestCodePayload:
    """Тесты для CodePayload."""

    def test_payload_from_code_element(self, mock_code_elements):
        """Тест создания payload из CodeElement."""
        element = mock_code_elements[0]

        payload = CodePayload.from_code_element(
            element,
            repository_name="test-repo",
            branch="main"
        )

        assert payload.name == element.name
        assert payload.qualified_name == element.qualified_name
        assert payload.element_type == element.type.value
        assert payload.language == element.language.value
        assert payload.repository_name == "test-repo"
        assert payload.branch == "main"

    def test_payload_to_dict(self, mock_code_elements):
        """Тест конвертации payload в словарь."""
        element = mock_code_elements[0]
        payload = CodePayload.from_code_element(element, "test-repo")

        payload_dict = payload.to_dict()

        assert isinstance(payload_dict, dict)
        assert "name" in payload_dict
        assert "qualified_name" in payload_dict
        assert "element_type" in payload_dict
        assert "language" in payload_dict
        assert "source_code" in payload_dict
        assert "repository_name" in payload_dict

    def test_payload_with_optional_fields(self, mock_code_elements):
        """Тест payload с опциональными полями."""
        element = mock_code_elements[0]

        payload = CodePayload.from_code_element(
            element,
            repository_name="test-repo",
            branch="develop",
            commit_hash="abc123"
        )

        payload_dict = payload.to_dict()

        assert payload_dict["branch"] == "develop"
        assert payload_dict.get("commit_hash") == "abc123"


# ==================== Scroll Tests ====================

class TestQdrantScroll:
    """Тесты для scroll (пагинация)."""

    def test_scroll_through_collection(
            self,
            qdrant_client,
            test_collection,
            mock_code_elements,
            embedder
    ):
        """Тест scroll через коллекцию."""
        # Вставляем данные
        vectors = []
        payloads = []

        for element in mock_code_elements:
            vector = embedder.encode_text(element.source_code)
            payload = CodePayload.from_code_element(element, "test-repo")

            vectors.append(vector)
            payloads.append(payload.to_dict())

        qdrant_client.insert_batch(
            collection_name=test_collection,
            vectors=np.array(vectors),
            payloads=payloads
        )

        # Scroll
        all_points = []
        offset = None

        while True:
            points, next_offset = qdrant_client.scroll(
                collection_name=test_collection,
                limit=2,
                offset=offset
            )

            all_points.extend(points)

            if next_offset is None:
                break

            offset = next_offset

        assert len(all_points) >= len(mock_code_elements)

    def test_scroll_with_filters(
            self,
            qdrant_client,
            test_collection,
            mock_code_elements,
            embedder
    ):
        """Тест scroll с фильтрами."""
        # Вставляем данные
        vectors = []
        payloads = []

        for element in mock_code_elements:
            vector = embedder.encode_text(element.source_code)
            payload = CodePayload.from_code_element(element, "test-repo")

            vectors.append(vector)
            payloads.append(payload.to_dict())

        qdrant_client.insert_batch(
            collection_name=test_collection,
            vectors=np.array(vectors),
            payloads=payloads
        )

        # Scroll с фильтром по типу
        filters = SearchFilters(element_type=CodeElementType.FUNCTION)

        points, _ = qdrant_client.scroll(
            collection_name=test_collection,
            limit=10,
            filters=filters
        )

        # Все точки должны быть функциями
        for point in points:
            assert point["payload"]["element_type"] == "function"


# ==================== Update Tests ====================

class TestQdrantUpdate:
    """Тесты обновления данных."""

    def test_update_point_payload(
            self,
            qdrant_client,
            test_collection,
            mock_code_elements,
            embedder
    ):
        """Тест обновления payload точки."""
        element = mock_code_elements[0]

        # Вставляем
        payload = CodePayload.from_code_element(element, "test-repo")
        vector = embedder.encode_text(element.source_code)

        point_id = qdrant_client.insert_point(
            collection_name=test_collection,
            vector=vector,
            payload=payload.to_dict(),
            point_id=str(uuid.uuid4())
        )

        # Обновляем payload
        new_payload = payload.to_dict()
        new_payload["updated"] = True

        result = qdrant_client.update_payload(
            collection_name=test_collection,
            point_id=point_id,
            payload=new_payload
        )

        assert result is True

        # Проверяем обновление
        retrieved = qdrant_client.get_by_id(
            collection_name=test_collection,
            point_id=point_id
        )

        assert retrieved["payload"]["updated"] is True

    def test_update_point_vector(
            self,
            qdrant_client,
            test_collection,
            embedder
    ):
        """Тест обновления вектора точки."""
        # Вставляем
        vector1 = embedder.encode_text("original code")
        payload = {"name": "test", "type": "function"}

        point_id = qdrant_client.insert_point(
            collection_name=test_collection,
            vector=vector1,
            payload=payload,
            point_id=str(uuid.uuid4())
        )

        # Обновляем вектор через метод update_vector
        vector2 = embedder.encode_text("updated code")

        result = qdrant_client.update_vector(
            collection_name=test_collection,
            point_id=point_id,
            vector=vector2
        )

        assert result is True


# ==================== Performance Tests ====================

@pytest.mark.slow
class TestQdrantPerformance:
    """Тесты производительности Qdrant."""

    def test_bulk_insert_performance(
            self,
            qdrant_client,
            test_collection,
            embedder,
            benchmark_timer
    ):
        """Тест производительности bulk insert."""
        # Создаем много векторов
        num_vectors = 1000

        vectors = []
        payloads = []

        for i in range(num_vectors):
            vector = embedder.encode_text(f"code example {i}")
            payload = {
                "name": f"element_{i}",
                "type": "function",
                "language": "python"
            }

            vectors.append(vector)
            payloads.append(payload)

        # Измеряем время вставки
        benchmark_timer.start("bulk_insert")

        qdrant_client.insert_batch(
            collection_name=test_collection,
            vectors=np.array(vectors),
            payloads=payloads,
            batch_size=100
        )

        benchmark_timer.stop("bulk_insert")

        # Должно быть быстрее 30 секунд
        benchmark_timer.assert_faster_than("bulk_insert", 30.0)

    def test_search_performance(
            self,
            qdrant_client,
            test_collection,
            embedder,
            benchmark_timer
    ):
        """Тест производительности поиска."""
        # Вставляем данные
        vectors = []
        payloads = []

        for i in range(100):
            vector = embedder.encode_text(f"code {i}")
            payload = {"name": f"item_{i}", "type": "function"}

            vectors.append(vector)
            payloads.append(payload)

        qdrant_client.insert_batch(
            collection_name=test_collection,
            vectors=np.array(vectors),
            payloads=payloads
        )

        # Измеряем время поиска
        query_vector = embedder.encode_query("test query")

        benchmark_timer.start("search")

        for _ in range(10):
            results = qdrant_client.search(
                collection_name=test_collection,
                query_vector=query_vector,
                limit=10
            )

        benchmark_timer.stop("search")

        # 10 поисков должны быть быстрее 5 секунд
        benchmark_timer.assert_faster_than("search", 5.0)


# ==================== Edge Cases Tests ====================

class TestQdrantEdgeCases:
    """Тесты граничных случаев."""

    def test_insert_empty_batch(self, qdrant_client, test_collection):
        """Тест вставки пустого батча."""
        result = qdrant_client.insert_batch(
            collection_name=test_collection,
            vectors=[],
            payloads=[]
        )

        assert result == (0,0)

    def test_search_with_zero_vector(self, qdrant_client, test_collection):
        """Тест поиска с нулевым вектором."""
        zero_vector = np.zeros(768, dtype=np.float32)

        results = qdrant_client.search(
            collection_name=test_collection,
            query_vector=zero_vector,
            limit=5
        )

        # Должен вернуть пустой список или результаты с низким score
        assert isinstance(results, list)

    def test_insert_with_very_long_payload(
            self,
            qdrant_client,
            test_collection,
            embedder
    ):
        """Тест вставки с очень большим payload."""
        vector = embedder.encode_text("test")

        # Очень длинный source_code
        long_code = "def func():\n" + ("    pass\n" * 10000)

        payload = {
            "name": "long_function",
            "type": "function",
            "source_code": long_code
        }

        point_id = qdrant_client.insert_point(
            collection_name=test_collection,
            vector=vector,
            payload=payload
        )

        assert point_id is not None

    def test_search_in_empty_collection(
            self,
            qdrant_client,
            test_collection,
            embedder
    ):
        """Тест поиска в пустой коллекции."""
        query_vector = embedder.encode_query("test")

        results = qdrant_client.search(
            collection_name=test_collection,
            query_vector=query_vector,
            limit=10
        )

        assert len(results) == 0

    def test_insert_duplicate_ids(
            self,
            qdrant_client,
            test_collection,
            embedder
    ):
        """Тест вставки с дублирующимися ID."""
        vector = embedder.encode_text("test")
        payload = {"name": "test"}
        point_id = str(uuid.uuid4())

        # Первая вставка
        qdrant_client.insert_point(
            collection_name=test_collection,
            vector=vector,
            payload=payload,
            point_id=point_id
        )

        # Вторая вставка с тем же ID (должна перезаписать)
        result = qdrant_client.insert_point(
            collection_name=test_collection,
            vector=vector,
            payload={"name": "updated"},
            point_id=point_id
        )

        # Проверяем что обновилось
        retrieved = qdrant_client.get_by_id(
            collection_name=test_collection,
            point_id=point_id
        )

        assert retrieved["payload"]["name"] == "updated"


# ==================== Filter Combinations Tests ====================

class TestQdrantFilterCombinations:
    """Тесты комбинаций фильтров."""

    @pytest.fixture
    def collection_with_diverse_data(
            self,
            qdrant_client,
            test_collection,
            embedder
    ):
        """Коллекция с разнообразными данными."""
        data = [
            ("func1", "function", "python", "repo-a"),
            ("func2", "function", "javascript", "repo-a"),
            ("Class1", "class", "python", "repo-a"),
            ("func3", "function", "python", "repo-b"),
            ("Class2", "class", "javascript", "repo-b"),
        ]

        vectors = []
        payloads = []

        for name, elem_type, language, repo in data:
            vector = embedder.encode_text(f"code {name}")
            payload = {
                "name": name,
                "type": elem_type,
                "language": language,
                "repository_name": repo
            }

            vectors.append(vector)
            payloads.append(payload)

        qdrant_client.insert_batch(
            collection_name=test_collection,
            vectors=np.array(vectors),
            payloads=payloads
        )

        return test_collection

    def test_filter_by_language_and_type(
            self,
            qdrant_client,
            collection_with_diverse_data,
            embedder
    ):
        """Тест фильтрации по языку и типу."""
        filters = SearchFilters(
            language=ProgrammingLanguage.PYTHON,
            element_type=CodeElementType.FUNCTION
        )

        query_vector = embedder.encode_query("function")

        results = qdrant_client.search(
            collection_name=collection_with_diverse_data,
            query_vector=query_vector,
            limit=10,
            filters=filters
        )

        # Должны быть только Python функции
        for result in results:
            assert result["payload"]["language"] == "python"
            assert result["payload"]["type"] == "function"

    def test_filter_by_repository(
            self,
            qdrant_client,
            collection_with_diverse_data,
            embedder
    ):
        """Тест фильтрации по репозиторию."""
        filters = SearchFilters(repository_name="repo-a")

        query_vector = embedder.encode_query("code")

        results = qdrant_client.search(
            collection_name=collection_with_diverse_data,
            query_vector=query_vector,
            limit=10,
            filters=filters
        )

        # Все результаты из repo-a
        for result in results:
            assert result["payload"]["repository_name"] == "repo-a"

    def test_complex_filter_combination(
            self,
            qdrant_client,
            collection_with_diverse_data,
            embedder
    ):
        """Тест сложной комбинации фильтров."""
        filters = SearchFilters(
            language=ProgrammingLanguage.PYTHON,
            element_type=CodeElementType.FUNCTION,
            repository_name="repo-b"
        )

        query_vector = embedder.encode_query("function")

        results = qdrant_client.search(
            collection_name=collection_with_diverse_data,
            query_vector=query_vector,
            limit=10,
            filters=filters
        )

        # Только Python функции из repo-b
        for result in results:
            assert result["payload"]["language"] == "python"
            assert result["payload"]["type"] == "function"
            assert result["payload"]["repository_name"] == "repo-b"


# ==================== Error Handling Tests ====================

class TestQdrantErrorHandling:
    """Тесты обработки ошибок."""

    def test_search_nonexistent_collection(self, qdrant_client, embedder):
        """Тест поиска в несуществующей коллекции."""
        query_vector = embedder.encode_query("test")

        result = qdrant_client.search(
                collection_name="nonexistent_collection",
                query_vector=query_vector,
                limit=5
            )

        assert len(result) == 0

    def test_insert_wrong_vector_size(
            self,
            qdrant_client,
            test_collection
    ):
        """Тест вставки вектора неверного размера."""
        wrong_size_vector = np.random.rand(512).astype(np.float32)
        payload = {"name": "test"}

        with pytest.raises(Exception):
            qdrant_client.insert_point(
                collection_name=test_collection,
                vector=wrong_size_vector,
                payload=payload
            )

    def test_get_nonexistent_point(self, qdrant_client, test_collection):
        """Тест получения несуществующей точки."""
        result = qdrant_client.get_by_id(
            collection_name=test_collection,
            point_id=str(uuid.uuid4())
        )

        assert result is None

    def test_delete_nonexistent_collection(self, qdrant_client):
        """Тест удаления несуществующей коллекции."""
        result = qdrant_client.delete_collection("nonexistent_collection")

        # Может вернуть True(inmemory qdrant) или raise исключение
        assert result is True or isinstance(result, Exception)


# ==================== Integration Tests ====================

@pytest.mark.integration
class TestQdrantIntegration:
    """Интеграционные тесты Qdrant."""

    def test_full_workflow(
            self,
            qdrant_client,
            test_collection,
            sample_python_code,
            python_parser,
            embedder,
            temp_dir
    ):
        """Тест полного workflow: parse -> embed -> insert -> search."""
        # 1. Parse
        file_path = temp_dir / "sample.py"
        file_path.write_text(sample_python_code)

        module = python_parser.parse_file(file_path)

        # 2. Embed and Insert
        for element in module.all_elements:
            vector = embedder.encode_text(element.source_code)
            payload = CodePayload.from_code_element(element, "test-repo")

            qdrant_client.insert_point(
                collection_name=test_collection,
                vector=vector,
                payload=payload.to_dict()
            )

        # 3. Search
        query = "calculator function"
        query_vector = embedder.encode_query(query)

        results = qdrant_client.search(
            collection_name=test_collection,
            query_vector=query_vector,
            limit=5
        )

        assert len(results) > 0

        # 4. Verify results
        for result in results:
            assert "name" in result["payload"]
            assert "source_code" in result["payload"]


# ==================== Parametrized Tests ====================

@pytest.mark.parametrize("limit", [1, 5, 10, 50])
def test_search_with_different_limits(
        qdrant_client,
        indexed_collection,
        embedder,
        limit
):
    """Параметризованный тест поиска с разными лимитами."""
    query_vector = embedder.encode_query("test")

    results = qdrant_client.search(
        collection_name=indexed_collection,
        query_vector=query_vector,
        limit=limit
    )

    assert len(results) <= limit


@pytest.mark.parametrize("vector_size", [128, 384, 768, 1536])
def test_create_collections_with_different_sizes(
        qdrant_client,
        vector_size
):
    """Параметризованный тест создания коллекций разных размеров."""
    collection_name = f"test_collection_{vector_size}"

    schema = CollectionSchema(
        collection_name=collection_name,
        vector_size=vector_size
    )

    try:
        result = qdrant_client.create_collection(schema, recreate=True)
        assert result is True

        # Проверяем что создалась
        assert qdrant_client.collection_exists(collection_name) is True
    finally:
        # Cleanup
        qdrant_client.delete_collection(collection_name)


# ==================== Concurrency Tests ====================

@pytest.mark.slow
class TestQdrantConcurrency:
    """Тесты конкурентного доступа."""

    def test_concurrent_inserts(
            self,
            qdrant_client,
            test_collection,
            embedder
    ):
        """Тест конкурентных вставок."""
        import threading

        def insert_data(thread_id):
            for i in range(10):
                vector = embedder.encode_text(f"thread {thread_id} item {i}")
                payload = {"name": f"item_{thread_id}_{i}"}

                qdrant_client.insert_point(
                    collection_name=test_collection,
                    vector=vector,
                    payload=payload
                )

        # Запускаем 5 потоков
        threads = []
        for i in range(5):
            thread = threading.Thread(target=insert_data, args=(i,))
            threads.append(thread)
            thread.start()

        # Ждем завершения
        for thread in threads:
            thread.join()

        # Проверяем что все вставилось
        info = qdrant_client.get_collection_info(test_collection)
        assert info["points_count"] == 50


# ==================== Cleanup Tests ====================

class TestQdrantCleanup:
    """Тесты очистки данных."""

    def test_delete_by_filter(
            self,
            qdrant_client,
            test_collection,
            embedder
    ):
        """Тест удаления по фильтру."""
        # Вставляем данные
        for i in range(10):
            vector = embedder.encode_text(f"code {i}")
            payload = {
                "name": f"item_{i}",
                "repository_name": "repo-a" if i < 5 else "repo-b"
            }

            qdrant_client.insert_point(
                collection_name=test_collection,
                vector=vector,
                payload=payload
            )

        # Удаляем по фильтру
        filters = SearchFilters(repository_name="repo-a")

        result = qdrant_client.delete_by_filter(
            collection_name=test_collection,
            filters=filters
        )

        # Проверяем что удалены только из repo-a
        query_vector = embedder.encode_query("test")
        remaining = qdrant_client.search(
            collection_name=test_collection,
            query_vector=query_vector,
            limit=20
        )

        # Все оставшиеся должны быть из repo-b
        for result in remaining:
            assert result["payload"]["repository_name"] == "repo-b"
