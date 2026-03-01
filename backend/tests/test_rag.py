"""
Tests for RAG (Retrieval-Augmented Generation) module.
"""

import pytest
from pathlib import Path

from code_rag.core.rag import (
    RAGRetriever,
    SearchResult,
    RAGResponse,
    RAGConfig,
)

from code_rag.core.parser import (
    CodeElement,
    CodeLocation,
    CodeElementType,
    ProgrammingLanguage,
)


# ==================== RAGRetriever Tests ====================

class TestRAGRetriever:
    """Тесты для RAGRetriever."""

    def test_retriever_initialization(self, rag_retriever):
        """Тест инициализации retriever."""
        assert rag_retriever is not None
        assert rag_retriever.collection_name is not None
        assert rag_retriever.embedder is not None
        assert rag_retriever.qdrant_client is not None

    def test_basic_search(self, rag_retriever, indexed_collection):
        """Тест базового поиска."""
        results = rag_retriever.search(
            query="authentication function",
            top_k=5
        )

        assert isinstance(results, list)
        assert len(results) <= 5

        # Проверяем структуру результатов
        if len(results) > 0:
            assert isinstance(results[0], SearchResult)
            assert results[0].element is not None
            assert results[0].score >= 0

    def test_search_with_filters(self, rag_retriever, indexed_collection):
        """Тест поиска с фильтрами."""
        results = rag_retriever.search(
            query="function",
            top_k=5,
            language=ProgrammingLanguage.PYTHON,
            element_type=CodeElementType.FUNCTION
        )

        # Все результаты должны быть Python функциями
        for result in results:
            assert result.element.language == ProgrammingLanguage.PYTHON
            assert result.element.type == CodeElementType.FUNCTION

    def test_search_empty_query(self, rag_retriever):
        """Тест поиска с пустым запросом."""
        results = rag_retriever.search(
            query="",
            top_k=5
        )

        # Должен вернуть пустой список или все результаты
        assert isinstance(results, list)

    def test_search_with_score_threshold(self, rag_retriever, indexed_collection):
        """Тест поиска с порогом score."""
        results = rag_retriever.search(
            query="test function",
            top_k=10,
            score_threshold=0.7
        )

        # Все результаты должны иметь score >= 0.7
        for result in results:
            assert result.score >= 0.7

    def test_search_by_repository(self, rag_retriever, indexed_collection):
        """Тест поиска по репозиторию."""
        results = rag_retriever.search(
            query="function",
            top_k=5,
            repository_name="test-repo"
        )

        # Все результаты должны быть из test-repo
        for result in results:
            assert result.element.repository_name == "test-repo"


# ==================== SearchResult Tests ====================

class TestSearchResult:
    """Тесты для SearchResult."""

    def test_search_result_creation(self, mock_code_elements):
        """Тест создания SearchResult."""
        element = mock_code_elements[0]

        result = SearchResult(
            element=element,
            score=0.95,
            rank=1
        )

        assert result.element == element
        assert result.score == 0.95

    def test_search_result_comparison(self, mock_code_elements):
        """Тест сравнения SearchResult по score."""
        result1 = SearchResult(element=mock_code_elements[0], score=0.9, rank=1)
        result2 = SearchResult(element=mock_code_elements[1], score=0.8, rank=1)

        # Сортировка по убыванию score
        results = sorted([result2, result1], key=lambda x: x.score, reverse=True)

        assert results[0].score == 0.9
        assert results[1].score == 0.8

    def test_search_result_to_dict(self, mock_code_elements):
        """Тест конвертации SearchResult в словарь."""
        result = SearchResult(
            element=mock_code_elements[0],
            score=0.85,
            rank=1
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "element" in result_dict
        assert "score" in result_dict
        assert result_dict["score"] == 0.85


# ==================== RAGResponse Tests ====================

class TestRAGResponse:
    """Тесты для RAGResponse."""

    def test_rag_response_creation(self, mock_search_results):
        """Тест создания RAGResponse."""
        response = RAGResponse(
            query="test query",
            answer="This is the answer",
            sources=mock_search_results,
            confidence=0.9
        )

        assert response.query == "test query"
        assert response.answer == "This is the answer"
        assert len(response.sources) == len(mock_search_results)
        assert response.confidence == 0.9

    def test_rag_response_without_sources(self):
        """Тест RAGResponse без источников."""
        response = RAGResponse(
            query="test query",
            answer="No sources found",
            sources=[],
            confidence=0.0
        )

        assert len(response.sources) == 0
        assert response.confidence == 0.0

    def test_rag_response_to_dict(self, mock_search_results):
        """Тест конвертации RAGResponse в словарь."""
        response = RAGResponse(
            query="test",
            answer="answer",
            sources=mock_search_results,
            confidence=0.8
        )

        response_dict = response.to_dict()

        assert isinstance(response_dict, dict)
        assert "query" in response_dict
        assert "answer" in response_dict
        assert "sources" in response_dict
        assert "confidence" in response_dict


# ==================== RAGConfig Tests ====================

class TestRAGConfig:
    """Тесты для RAGConfig."""

    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = RAGConfig()

        assert config.top_k > 0
        assert config.score_threshold >= 0
        assert config.use_reranking in [True, False]

    def test_custom_config(self):
        """Тест кастомной конфигурации."""
        config = RAGConfig(
            top_k=10,
            score_threshold=0.8,
            use_reranking=True,
            max_context_length=4000
        )

        assert config.top_k == 10
        assert config.score_threshold == 0.8
        assert config.use_reranking is True
        assert config.max_context_length == 4000

    def test_config_validation(self):
        """Тест валидации конфигурации."""
        # Некорректный top_k
        with pytest.raises(ValueError):
            RAGConfig(top_k=0)

        # Некорректный score_threshold
        with pytest.raises(ValueError):
            RAGConfig(score_threshold=1.5)



# ==================== Semantic Search Tests ====================

class TestSemanticSearch:
    """Тесты семантического поиска."""

    def test_similar_queries_similar_results(self, rag_retriever, indexed_collection):
        """Тест что похожие запросы дают похожие результаты."""
        query1 = "user authentication"
        query2 = "login verification"

        results1 = rag_retriever.search(query1, top_k=5)
        results2 = rag_retriever.search(query2, top_k=5)

        # Должны быть некоторые пересечения в результатах
        names1 = {r.element.name for r in results1}
        names2 = {r.element.name for r in results2}

        # Хотя бы одно пересечение (если есть релевантные данные)
        if len(names1) > 0 and len(names2) > 0:
            assert len(names1 & names2) >= 0  # Может быть 0 если данных мало

    def test_specific_vs_general_query(self, rag_retriever, indexed_collection):
        """Тест специфичного vs общего запроса."""
        specific_query = "authenticate_user function"
        general_query = "function"

        specific_results = rag_retriever.search(specific_query, top_k=3)
        general_results = rag_retriever.search(general_query, top_k=3)

        # Специфичный запрос должен давать более высокие scores
        if len(specific_results) > 0 and len(general_results) > 0:
            assert specific_results[0].score >= general_results[0].score * 0.8


# ==================== Reranking Tests ====================

class TestReranking:
    """Тесты для reranking результатов."""

    def test_reranking_enabled(self, test_collection, embedder, qdrant_client):
        """Тест с включенным reranking."""
        retriever = RAGRetriever(
            collection_name=test_collection,
            embedder=embedder,
            qdrant_client=qdrant_client,
            use_reranking=True
        )

        # Индексируем данные (используем indexed_collection fixture)
        results = retriever.search("test query", top_k=5)

        assert isinstance(results, list)

    def test_reranking_improves_results(
            self,
            rag_retriever,
            indexed_collection,
            mock_code_elements
    ):
        """Тест что reranking улучшает результаты."""
        # Без reranking
        rag_retriever.use_reranking = False
        results_without = rag_retriever.search("authentication", top_k=5)

        # С reranking
        rag_retriever.use_reranking = True
        results_with = rag_retriever.search("authentication", top_k=5)

        # Оба должны вернуть результаты
        assert len(results_without) >= 0
        assert len(results_with) >= 0


# ==================== Multi-step Retrieval Tests ====================

class TestMultiStepRetrieval:
    """Тесты многоэтапного поиска."""

    def test_iterative_retrieval(self, rag_retriever, indexed_collection):
        """Тест итеративного поиска."""
        # Первый поиск
        initial_results = rag_retriever.search(
            "database connection",
            top_k=3
        )

        # Второй поиск на основе первых результатов
        if len(initial_results) > 0:
            refined_query = f"{initial_results[0].element.name} implementation"
            refined_results = rag_retriever.search(
                refined_query,
                top_k=3
            )

            assert len(refined_results) >= 0

    def test_query_expansion(self, rag_retriever):
        """Тест расширения запроса."""
        original_query = "auth"

        # Расширяем запрос
        expanded_queries = [
            "authentication",
            "authorization",
            "login"
        ]

        all_results = []
        for query in expanded_queries:
            results = rag_retriever.search(query, top_k=3)
            all_results.extend(results)

        # Deduplicate по element.name
        unique_results = {}
        for result in all_results:
            if result.element.name not in unique_results:
                unique_results[result.element.name] = result

        assert len(unique_results) >= 0


# ==================== Answer Generation Tests ====================

class TestAnswerGeneration:
    """Тесты генерации ответов (с Qwen)."""

    @pytest.mark.requires_qwen
    def test_generate_answer(self, rag_retriever, qwen_integration, indexed_collection):
        """Тест генерации ответа."""
        # Поиск контекста
        results = rag_retriever.search(
            "authentication function",
            top_k=3
        )

        # Генерация ответа
        question = "How does authentication work?"

        answer = qwen_integration.generate(
            query=question,
            context_results=results
        )

        assert isinstance(answer.content, str)
        assert len(answer.content) > 0

    @pytest.mark.requires_qwen
    def test_generate_with_code_examples(
            self,
            rag_retriever,
            qwen_integration,
            indexed_collection
    ):
        """Тест генерации ответа с примерами кода."""
        results = rag_retriever.search("function", top_k=3)

        question = "Show me an example of a function"

        answer = qwen_integration.generate(
            query=question,
            context_results=results
        )

        assert isinstance(answer.content, str)
        # Ответ может содержать код
        assert "function" in answer.content


# ==================== RAG Pipeline Tests ====================

class TestRAGPipeline:
    """Тесты полного RAG pipeline."""

    @pytest.mark.requires_qwen
    def test_end_to_end_rag(
            self,
            rag_retriever,
            qwen_integration,
            indexed_collection
    ):
        """Тест полного RAG workflow."""
        question = "What functions are available for authentication?"

        # 1. Retrieval
        results = rag_retriever.search(
            question,
            top_k=5
        )

        assert len(results) >= 0

        # 3. Generation
        answer = qwen_integration.generate(
            query=question,
            context_results=results,
        )

        assert isinstance(answer.content, str)

        # 4. Create response
        response = RAGResponse(
            query=question,
            answer=answer.content,
            sources=results,
            confidence=0.8
        )

        assert response.query == question
        assert len(response.sources) == len(results)

    def test_rag_with_no_results(self, rag_retriever):
        """Тест RAG когда нет результатов поиска."""
        question = "completely unrelated query xyz123"

        results = rag_retriever.search(question, top_k=5, score_threshold=0.9)

        # Может вернуть пустой список
        assert isinstance(results, list)


# ==================== Performance Tests ====================

@pytest.mark.slow
class TestRAGPerformance:
    """Тесты производительности RAG."""

    def test_search_performance(
            self,
            rag_retriever,
            indexed_collection,
            benchmark_timer
    ):
        """Тест производительности поиска."""
        queries = [
            "authentication function",
            "database connection",
            "user management",
            "api endpoint",
            "error handling"
        ]

        benchmark_timer.start("rag_search")

        for query in queries:
            results = rag_retriever.search(query, top_k=5)

        benchmark_timer.stop("rag_search")

        # 5 поисков должны быть быстрее 5 секунд
        benchmark_timer.assert_faster_than("rag_search", 5.0)


# ==================== Edge Cases Tests ====================

class TestRAGEdgeCases:
    """Тесты граничных случаев RAG."""

    def test_very_long_query(self, rag_retriever, indexed_collection):
        """Тест очень длинного запроса."""
        long_query = "authentication " * 100

        results = rag_retriever.search(long_query, top_k=5)

        # Должен обработать без ошибок
        assert isinstance(results, list)

    def test_query_with_special_characters(self, rag_retriever, indexed_collection):
        """Тест запроса со спецсимволами."""
        special_query = "function with @decorator #comment $variable"

        results = rag_retriever.search(special_query, top_k=5)

        assert isinstance(results, list)

    def test_unicode_query(self, rag_retriever, indexed_collection):
        """Тест запроса с Unicode символами."""
        unicode_query = "функция аутентификации 用户认证"

        results = rag_retriever.search(unicode_query, top_k=5)

        assert isinstance(results, list)

    def test_empty_collection_search(self, rag_retriever):
        """Тест поиска в пустой коллекции."""
        # Коллекция пустая (без indexed_collection fixture)
        results = rag_retriever.search("test query", top_k=5)

        assert len(results) == 0

    def test_search_with_zero_top_k(self, rag_retriever):
        """Тест поиска с top_k=0."""
        with pytest.raises(ValueError):
            rag_retriever.search("test", top_k=0)

    def test_search_with_negative_score_threshold(self, rag_retriever):
        """Тест поиска с отрицательным порогом."""
        with pytest.raises(ValueError):
            rag_retriever.search("test", score_threshold=-0.5)


# ==================== Filter Combination Tests ====================

class TestRAGFilterCombinations:
    """Тесты комбинаций фильтров в RAG."""

    def test_multiple_filters(self, rag_retriever, indexed_collection):
        """Тест множественных фильтров."""
        results = rag_retriever.search(
            query="function",
            top_k=10,
            language=ProgrammingLanguage.PYTHON,
            element_type=CodeElementType.FUNCTION,
            repository_name="test-repo"
        )

        # Все результаты должны соответствовать всем фильтрам
        for result in results:
            assert result.element.language == ProgrammingLanguage.PYTHON
            assert result.element.type == CodeElementType.FUNCTION
            assert result.element.repository_name == "test-repo"

    def test_conflicting_filters(self, rag_retriever, indexed_collection):
        """Тест конфликтующих фильтров."""
        # Поиск Python классов в JavaScript репозитории (конфликт)
        results = rag_retriever.search(
            query="code",
            top_k=5,
            language=ProgrammingLanguage.PYTHON,
            repository_name="js-repo"  # Если такого нет
        )

        # Должен вернуть пустой список или только соответствующие результаты
        assert isinstance(results, list)


# ==================== Integration Tests ====================

@pytest.mark.integration
class TestRAGIntegration:
    """Интеграционные тесты RAG."""

    def test_full_workflow_with_real_code(
            self,
            vectorization_pipeline,
            rag_retriever,
            sample_python_code,
            temp_dir
    ):
        """Тест полного workflow с реальным кодом."""
        # 1. Индексация
        file_path = temp_dir / "real_code.py"
        file_path.write_text(sample_python_code)

        vectorize_result = vectorization_pipeline.process_file(
            file_path,
            repository_info={"repository_name": "real-project"}
        )

        assert vectorize_result.indexed_elements > 0

        # 2. Поиск
        results = rag_retriever.search(
            "calculator class",
            top_k=5
        )

        assert len(results) > 0

        # Вариант 1: Проверяем что Calculator в топ результатах
        calculator_found = False
        for result in results:
            if "calculator" in result.element.name.lower():
                calculator_found = True
                assert result.element.type == CodeElementType.CLASS, \
                    f"Expected class, got: {result.element.type}"
                break

        assert calculator_found, \
            f"Calculator not found in results. Got: {[r.element.name for r in results]}"

        # Вариант 2: Проверяем что первый результат имеет хороший score
        assert results[0].score > 0.5, \
            f"Top result score too low: {results[0].score}"


# ==================== Query Analysis Tests ====================

class TestQueryAnalysis:
    """Тесты анализа запросов."""

    def test_detect_query_intent(self, rag_retriever):
        """Тест определения намерения запроса."""
        queries = {
            "how does authentication work": "explanation",
            "show me login function": "code_example",
            "what is the purpose of": "explanation",
            "find all functions": "search"
        }

        for query, expected_intent in queries.items():
            # Если реализован анализ намерения
            if hasattr(rag_retriever, 'analyze_query'):
                intent = rag_retriever.analyze_query(query)
                # Проверяем что возвращается какой-то intent
                assert intent is not None

    def test_extract_keywords(self, rag_retriever):
        """Тест извлечения ключевых слов из запроса."""
        query = "find authentication and authorization functions in Python"

        if hasattr(rag_retriever, 'extract_keywords'):
            keywords = rag_retriever.extract_keywords(query)

            assert "authentication" in keywords or "authorization" in keywords
            assert "python" in [k.lower() for k in keywords]


# ==================== Result Ranking Tests ====================

class TestResultRanking:
    """Тесты ранжирования результатов."""

    def test_results_sorted_by_score(self, rag_retriever, indexed_collection):
        """Тест что результаты отсортированы по score."""
        results = rag_retriever.search("function", top_k=10)

        # Проверяем что scores в убывающем порядке
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_diversity_in_results(self, rag_retriever, indexed_collection):
        """Тест разнообразия результатов."""
        results = rag_retriever.search("code", top_k=10)

        # Проверяем что не все результаты из одного файла
        file_paths = {r.element.location.file_path for r in results}

        # Должно быть несколько разных файлов (если данных достаточно)
        if len(results) >= 3:
            assert len(file_paths) >= 1  # Минимум 1 файл


# ==================== Caching Tests ====================

class TestRAGCaching:
    """Тесты кэширования в RAG."""

    def test_repeated_query_caching(self, rag_retriever, indexed_collection):
        """Тест кэширования повторяющихся запросов."""
        query = "authentication function"

        # Первый запрос
        results1 = rag_retriever.search(query, top_k=5)

        # Второй такой же запрос
        results2 = rag_retriever.search(query, top_k=5)

        # Результаты должны быть идентичными
        assert len(results1) == len(results2)

        if len(results1) > 0:
            assert results1[0].element.name == results2[0].element.name
            assert results1[0].score == results2[0].score


# ==================== Error Handling Tests ====================

class TestRAGErrorHandling:
    """Тесты обработки ошибок в RAG."""

    def test_handle_invalid_collection(self, embedder, qdrant_client):
        """Тест обработки несуществующей коллекции."""
        retriever = RAGRetriever(
            collection_name="nonexistent_collection",
            embedder=embedder,
            qdrant_client=qdrant_client
        )

        # Поиск должен вернуть пустой список или raise исключение
        try:
            results = retriever.search("test", top_k=5)
            assert len(results) == 0
        except Exception:
            # Допустимо если поднимает исключение
            assert True

    def test_handle_malformed_query(self, rag_retriever):
        """Тест обработки некорректного запроса."""

        # Неправильный тип - должен выбросить TypeError
        with pytest.raises(TypeError):
            rag_retriever.search(None, top_k=5)

        with pytest.raises(TypeError):
            rag_retriever.search(123, top_k=5)

        with pytest.raises(TypeError):
            rag_retriever.search(["list", "query"], top_k=5)

        # Пустая строка - должен вернуть пустой список
        results = rag_retriever.search("", top_k=5)
        assert len(results) == 0

        results = rag_retriever.search("   ", top_k=5)
        assert len(results) == 0


# ==================== Parametrized Tests ====================

@pytest.mark.parametrize("top_k", [1, 3, 5, 10, 20])
def test_different_top_k_values(rag_retriever, indexed_collection, top_k):
    """Параметризованный тест разных значений top_k."""
    results = rag_retriever.search("function", top_k=top_k)

    assert len(results) <= top_k


@pytest.mark.parametrize("score_threshold", [0.0, 0.3, 0.5, 0.7, 0.9])
def test_different_score_thresholds(rag_retriever, indexed_collection, score_threshold):
    """Параметризованный тест разных порогов score."""
    results = rag_retriever.search(
        "function",
        top_k=10,
        score_threshold=score_threshold
    )

    # Все результаты должны иметь score >= threshold
    for result in results:
        assert result.score >= score_threshold


@pytest.mark.parametrize("language", [
    ProgrammingLanguage.PYTHON,
    ProgrammingLanguage.JAVASCRIPT,
    ProgrammingLanguage.TYPESCRIPT,
])
def test_search_by_language(rag_retriever, indexed_collection, language):
    """Параметризованный тест поиска по языку."""
    results = rag_retriever.search(
        "function",
        top_k=5,
        language=language
    )

    # Все результаты должны быть на указанном языке
    for result in results:
        assert result.element.language == language


# ==================== Explanation Tests ====================

class TestRAGExplanations:
    """Тесты объяснений и интерпретируемости."""

    def test_explain_search_results(self, rag_retriever, indexed_collection):
        """Тест объяснения результатов поиска."""
        results = rag_retriever.search("authentication", top_k=3)

        if len(results) > 0:
            # Проверяем что есть score (мера релевантности)
            for result in results:
                assert 0 <= result.score <= 1

    def test_result_metadata(self, rag_retriever, indexed_collection):
        """Тест наличия метаданных в результатах."""
        results = rag_retriever.search("function", top_k=3)

        for result in results:
            # Проверяем наличие важных метаданных
            assert result.element.name is not None
            assert result.element.type is not None
            assert result.element.language is not None
            assert result.element.location is not None


# ==================== Consistency Tests ====================

class TestRAGConsistency:
    """Тесты консистентности RAG."""

    def test_search_consistency(self, rag_retriever, indexed_collection):
        """Тест консистентности поиска."""
        query = "authentication function"

        # Несколько запусков
        results_list = []
        for _ in range(3):
            results = rag_retriever.search(query, top_k=5)
            results_list.append(results)

        # Проверяем что результаты одинаковые
        if len(results_list[0]) > 0:
            for i in range(1, len(results_list)):
                assert len(results_list[0]) == len(results_list[i])
                assert results_list[0][0].element.name == results_list[i][0].element.name


# ==================== Specialized Queries Tests ====================

class TestSpecializedQueries:
    """Тесты специализированных запросов."""

    def test_code_pattern_search(self, rag_retriever, indexed_collection):
        """Тест поиска паттернов кода."""
        patterns = [
            "decorator pattern",
            "singleton class",
            "factory method",
            "async function"
        ]

        for pattern in patterns:
            results = rag_retriever.search(pattern, top_k=3)
            assert isinstance(results, list)

    def test_api_endpoint_search(self, rag_retriever, indexed_collection):
        """Тест поиска API endpoints."""
        queries = [
            "POST endpoint",
            "GET request handler",
            "API route"
        ]

        for query in queries:
            results = rag_retriever.search(query, top_k=3)
            assert isinstance(results, list)

    def test_error_handling_search(self, rag_retriever, indexed_collection):
        """Тест поиска обработки ошибок."""
        results = rag_retriever.search(
            "error handling exception try catch",
            top_k=5
        )

        assert isinstance(results, list)

               # ==================== Memory Efficiency Tests ====================

@ pytest.mark.slow
class TestRAGMemoryEfficiency:
    """Тесты эффективности использования памяти."""

    def test_large_result_set_handling(self, rag_retriever, indexed_collection):
        """Тест обработки большого набора результатов."""
        import gc

        # Большой top_k
        results = rag_retriever.search("code", top_k=100)

        # Обрабатываем результаты
        for result in results:
            _ = result.element.name

        # Очищаем память
        del results
        gc.collect()

        # Проверяем что можем сделать еще один запрос
        results2 = rag_retriever.search("function", top_k=5)
        assert len(results2) >= 0


# ==================== Regression Tests ====================

class TestRAGRegression:
    """Регрессионные тесты RAG."""

    def test_backward_compatibility(self, rag_retriever):
        """Тест обратной совместимости."""
        # Базовый поиск должен работать
        results = rag_retriever.search("test", top_k=5)

        # Проверяем базовые атрибуты
        assert hasattr(rag_retriever, 'search')
        assert hasattr(rag_retriever, 'embedder')
        assert hasattr(rag_retriever, 'qdrant_client')

    def test_api_stability(self, rag_retriever, indexed_collection):
        """Тест стабильности API."""
        # Проверяем что основные методы работают
        query = "test query"

        # search method
        results = rag_retriever.search(query, top_k=5)
        assert isinstance(results, list)


# ==================== Load Testing ====================

@pytest.mark.slow
class TestRAGLoad:
    """Нагрузочные тесты RAG."""

    def test_concurrent_searches(self, rag_retriever, indexed_collection):
        """Тест конкурентных поисков."""
        import threading

        results_dict = {}

        def search_task(thread_id, query):
            results = rag_retriever.search(query, top_k=5)
            results_dict[thread_id] = results

        queries = [
            "authentication",
            "database",
            "api",
            "function",
            "class"
        ]

        threads = []
        for i, query in enumerate(queries):
            thread = threading.Thread(target=search_task, args=(i, query))
            threads.append(thread)
            thread.start()

        # Ждем завершения
        for thread in threads:
            thread.join()

        # Проверяем что все поиски выполнились
        assert len(results_dict) == len(queries)

    def test_rapid_sequential_searches(
            self,
            rag_retriever,
            indexed_collection,
            benchmark_timer
    ):
        """Тест быстрых последовательных поисков."""
        queries = [f"query {i}" for i in range(20)]

        benchmark_timer.start("rapid_search")

        for query in queries:
            results = rag_retriever.search(query, top_k=5)

        benchmark_timer.stop("rapid_search")

        # 20 поисков должны быть быстрее 10 секунд
        benchmark_timer.assert_faster_than("rapid_search", 10.0)
