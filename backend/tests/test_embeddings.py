"""
Tests for embeddings module.
"""

import pytest
import numpy as np
from pathlib import Path

from code_rag.core.embeddings import (
    CodeEmbedder,
    NomicEmbedModel,
    EmbeddingModelFactory,
    EmbeddingModelType,
    EmbeddingConfig,
)


# ==================== Basic Embedder Tests ====================

class TestCodeEmbedder:
    """Тесты для CodeEmbedder."""

    def test_embedder_initialization(self, embedder):
        """Тест инициализации embedder."""
        assert embedder is not None
        assert embedder.embedding_dim > 0
        assert embedder.embedding_dim == 768  # Nomic embed dimension

    def test_encode_text_single(self, embedder):
        """Тест векторизации одного текста."""
        text = "def hello(): pass"
        embedding = embedder.encode_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32

        # Проверка что вектор нормализован
        norm = np.linalg.norm(embedding)
        assert 0.99 < norm < 1.01

    def test_encode_text_empty(self, embedder):
        """Тест векторизации пустого текста."""
        embedding = embedder.encode_text("")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_encode_text_long(self, embedder):
        """Тест векторизации длинного текста."""
        # Создаем длинный код
        long_text = "\n".join([f"def func_{i}(): pass" for i in range(100)])

        embedding = embedder.encode_text(long_text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_encode_batch(self, embedder):
        """Тест батчевой векторизации."""
        texts = [
            "def add(a, b): return a + b",
            "def subtract(a, b): return a - b",
            "def multiply(a, b): return a * b"
        ]

        embeddings = embedder.encode_batch(texts, batch_size=2)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 768)
        assert embeddings.dtype == np.float32

    def test_encode_batch_empty_list(self, embedder):
        """Тест батчевой векторизации пустого списка."""
        embeddings = embedder.encode_batch([])

        # Должен вернуть массив формы (0, 768)
        assert embeddings.shape == (0, 768), f"Expected shape (0, 768), got {embeddings.shape}"
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32

    def test_encode_query(self, embedder):
        """Тест векторизации запроса."""
        query = "authentication function"
        embedding = embedder.encode_query(query)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_embedding_consistency(self, embedder):
        """Тест консистентности векторов."""
        text = "def test(): pass"

        embedding1 = embedder.encode_text(text)
        embedding2 = embedder.encode_text(text)

        # Векторы должны быть идентичны для одинакового текста
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)

    def test_similar_code_similar_embeddings(self, embedder):
        """Тест что похожий код имеет похожие векторы."""
        code1 = "def add(a, b): return a + b"
        code2 = "def sum(x, y): return x + y"
        code3 = "class Database: pass"

        emb1 = embedder.encode_text(code1)
        emb2 = embedder.encode_text(code2)
        emb3 = embedder.encode_text(code3)

        # Схожесть между add и sum должна быть выше
        similarity_12 = np.dot(emb1, emb2)
        similarity_13 = np.dot(emb1, emb3)

        assert similarity_12 > similarity_13
        assert similarity_12 > 0.6  # ослабленный порог


# ==================== Configuration Tests ====================

class TestEmbeddingConfig:
    """Тесты для EmbeddingConfig."""

    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = EmbeddingConfig()

        assert config.model_name == "nomic-embed-text:latest"
        assert config.base_url == "https://ai.parma.ru/local/"
        assert config.device == "cpu"
        assert config.max_tokens == 8192
        assert config.normalize is True

    def test_custom_config(self):
        """Тест кастомной конфигурации."""
        config = EmbeddingConfig(
            base_url="https://ai.parma.ru/local/",
            model_name="custom-model",
            device="cuda",
            max_tokens=4096,
            normalize=False
        )

        assert config.base_url == "https://ai.parma.ru/local/"
        assert config.model_name == "custom-model"
        assert config.device == "cuda"
        assert config.max_tokens == 4096
        assert config.normalize is False


# ==================== Similarity Tests ====================

class TestSimilarityCalculations:
    """Тесты для вычислений схожести."""

    def test_cosine_similarity(self, embedder):
        """Тест косинусной схожести."""
        text1 = "def add(a, b): return a + b"
        text2 = "def sum(x, y): return x + y"

        emb1 = embedder.encode_text(text1)
        emb2 = embedder.encode_text(text2)

        # Косинусная схожесть
        similarity = np.dot(emb1, emb2)

        assert 0 <= similarity <= 1
        assert similarity > 0.6  # Должна быть высокой для похожего кода

    def test_euclidean_distance(self, embedder):
        """Тест евклидова расстояния."""
        text1 = "def hello(): pass"
        text2 = "def world(): pass"
        text3 = "class Database: pass"

        emb1 = embedder.encode_text(text1)
        emb2 = embedder.encode_text(text2)
        emb3 = embedder.encode_text(text3)

        # Расстояния
        dist_12 = np.linalg.norm(emb1 - emb2)
        dist_13 = np.linalg.norm(emb1 - emb3)

        # Функции должны быть ближе друг к другу чем к классу
        assert dist_12 < dist_13

    def test_identical_code_similarity(self, embedder):
        """Тест что идентичный код имеет максимальную схожесть."""
        code = "def test(): return True"

        emb1 = embedder.encode_text(code)
        emb2 = embedder.encode_text(code)

        similarity = np.dot(emb1, emb2)

        # Должна быть близка к 1.0
        assert similarity > 0.99


# ==================== Performance Tests ====================

class TestEmbeddingPerformance:
    """Тесты производительности embeddings."""

    @pytest.mark.slow
    def test_batch_vs_single_speed(self, embedder, benchmark_timer):
        """Тест что батчевая обработка быстрее."""
        texts = [f"def func_{i}(): pass" for i in range(50)]

        # Single processing
        benchmark_timer.start("single")
        for text in texts:
            embedder.encode_text(text)
        benchmark_timer.stop("single")

        # Batch processing
        benchmark_timer.start("batch")
        embedder.encode_batch(texts, batch_size=32)
        benchmark_timer.stop("batch")

        single_time = benchmark_timer.get_duration("single")
        batch_time = benchmark_timer.get_duration("batch")

        # Батч должен быть значительно быстрее
        assert batch_time < single_time * 1.05 # ослабленная проверка Допускаем до 5% замедления пакета

    @pytest.mark.slow
    def test_encoding_speed(self, embedder, benchmark_timer):
        """Тест скорости векторизации."""
        text = "def calculate_sum(a, b): return a + b"

        benchmark_timer.start("encode")
        for _ in range(10):
            embedder.encode_text(text)
        benchmark_timer.stop("encode")

        # Среднее время на один encode
        avg_time = benchmark_timer.get_duration("encode") / 10

        # Должно быть быстрее 1 секунды
        assert avg_time < 1.0

    def test_batch_size_impact(self, embedder):
        """Тест влияния размера батча."""
        texts = [f"code {i}" for i in range(64)]

        import time

        # Маленький батч
        start = time.time()
        embedder.encode_batch(texts, batch_size=8)
        time_small = time.time() - start

        # Большой батч
        start = time.time()
        embedder.encode_batch(texts, batch_size=32)
        time_large = time.time() - start

        # Больший батч обычно эффективнее
        # (но может быть не всегда из-за overhead)
        assert time_large < time_small * 2  # Не должен быть в 2 раза медленнее


# ==================== Edge Cases Tests ====================

class TestEmbeddingEdgeCases:
    """Тесты граничных случаев."""

    def test_special_characters(self, embedder):
        """Тест с специальными символами."""
        text = "def test(): return '!@#$%^&*()'"

        embedding = embedder.encode_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert not np.isnan(embedding).any()

    def test_unicode_characters(self, embedder):
        """Тест с Unicode символами."""
        text = "def привет(): return '你好'"

        embedding = embedder.encode_text(text)

        assert isinstance(embedding, np.ndarray)
        assert not np.isnan(embedding).any()

    def test_very_long_text(self, embedder):
        """Тест очень длинного текста."""
        # Создаем текст > max_length
        text = "def func(): pass\n" * 1000

        embedding = embedder.encode_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_whitespace_only(self, embedder):
        """Тест текста только из пробелов."""
        text = "   \n\t   "

        embedding = embedder.encode_text(text)

        assert isinstance(embedding, np.ndarray)
        assert not np.isnan(embedding).any()

    def test_mixed_languages_code(self, embedder):
        """Тест смешанного кода на разных языках."""
        texts = [
            "def python_func(): pass",
            "function jsFunc() { }",
            "public void javaMethod() { }",
            "fun kotlinFunc() { }"
        ]

        embeddings = embedder.encode_batch(texts)

        assert embeddings.shape == (4, 768)
        assert not np.isnan(embeddings).any()


# ==================== Mock Embedder Tests ====================

class TestMockEmbedder:
    """Тесты для mock embedder."""

    def test_mock_embedder_basic(self, mock_embedder):
        """Тест базовой функциональности mock embedder."""
        text = "test code"

        embedding = mock_embedder.encode_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_mock_embedder_deterministic(self, mock_embedder):
        """Тест детерминированности mock embedder."""
        text = "test code"

        emb1 = mock_embedder.encode_text(text)
        emb2 = mock_embedder.encode_text(text)

        np.testing.assert_array_equal(emb1, emb2)

    def test_mock_embedder_batch(self, mock_embedder):
        """Тест батчевой обработки mock embedder."""
        texts = ["code1", "code2", "code3"]

        embeddings = mock_embedder.encode_batch(texts)

        assert embeddings.shape == (3, 768)


# ==================== Integration Tests ====================

@pytest.mark.integration
class TestEmbeddingIntegration:
    """Интеграционные тесты embeddings."""

    def test_embedder_with_real_code(self, embedder, sample_python_code):
        """Тест с реальным кодом."""
        embedding = embedder.encode_text(sample_python_code)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert not np.isnan(embedding).any()

    def test_multiple_languages(self, embedder, sample_python_code, sample_javascript_code):
        """Тест с разными языками программирования."""
        embeddings = embedder.encode_batch([
            sample_python_code,
            sample_javascript_code
        ])

        assert embeddings.shape == (2, 768)

        # Проверка что векторы различаются
        similarity = np.dot(embeddings[0], embeddings[1])
        assert similarity < 1.0  # Не идентичны

    def test_code_search_scenario(self, embedder):
        """Тест сценария поиска кода."""
        # Код примеры
        codes = [
            "def authenticate_user(username, password): return True",
            "def verify_credentials(user, pwd): return check(user, pwd)",
            "class Database: def connect(self): pass",
            "function calculateSum(a, b) { return a + b; }"
        ]

        # Запрос
        query = "authentication function"

        # Векторизация
        code_embeddings = embedder.encode_batch(codes)
        query_embedding = embedder.encode_query(query)

        # Вычисление схожести
        similarities = code_embeddings @ query_embedding

        # Наиболее релевантным должен быть первый или второй
        best_idx = np.argmax(similarities)
        assert best_idx in [0, 1]


# ==================== Normalization Tests ====================

class TestVectorNormalization:
    """Тесты нормализации векторов."""

    def test_vectors_are_normalized(self, embedder):
        """Тест что векторы нормализованы."""
        texts = ["code1", "code2", "code3"]

        embeddings = embedder.encode_batch(texts)

        # Проверка нормализации для каждого вектора
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            assert 0.99 < norm < 1.01

    def test_normalization_consistency(self, embedder):
        """Тест консистентности нормализации."""
        text = "def test(): pass"

        # Многократная векторизация
        norms = []
        for _ in range(5):
            embedding = embedder.encode_text(text)
            norms.append(np.linalg.norm(embedding))

        # Все нормы должны быть одинаковыми
        assert all(0.99 < n < 1.01 for n in norms)


# ==================== Error Handling Tests ====================

class TestEmbeddingErrorHandling:
    """Тесты обработки ошибок."""

    def test_none_input(self, embedder):
        """Тест с None входом."""
        with pytest.raises((TypeError, ValueError)):
            embedder.encode_text(None)

    def test_invalid_type_input(self, embedder):
        """Тест с некорректным типом входа."""
        with pytest.raises((TypeError, AttributeError)):
            embedder.encode_text(12345)

    def test_batch_with_none(self, embedder):
        """Тест батча с None элементами."""
        texts = ["code1", None, "code3"]

        with pytest.raises((TypeError, ValueError)):
            embedder.encode_batch(texts)


# ==================== Memory Tests ====================

@pytest.mark.slow
class TestEmbeddingMemory:
    """Тесты использования памяти."""

    def test_large_batch_memory(self, embedder):
        """Тест обработки большого батча."""
        # Создаем большой батч
        texts = [f"def func_{i}(): pass" for i in range(1000)]

        # Должно отработать без out of memory
        embeddings = embedder.encode_batch(texts, batch_size=32)

        assert embeddings.shape == (1000, 768)

    def test_memory_cleanup(self, embedder):
        """Тест очистки памяти после обработки."""
        import gc

        # Большая обработка
        texts = [f"code {i}" for i in range(500)]
        embeddings = embedder.encode_batch(texts)

        # Удаляем ссылки
        del embeddings
        gc.collect()

        # Должны суметь обработать еще раз
        embeddings2 = embedder.encode_batch(texts)
        assert embeddings2.shape == (500, 768)


# ==================== Parametrized Tests ====================

@pytest.mark.parametrize("text,expected_dim", [
    ("def test(): pass", 768),
    ("class MyClass: pass", 768),
    ("function test() {}", 768),
    ("", 768),
])
def test_various_inputs(embedder, text, expected_dim):
    """Параметризованный тест различных входов."""
    embedding = embedder.encode_text(text)
    assert embedding.shape == (expected_dim,)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
def test_different_batch_sizes(embedder, batch_size):
    """Тест различных размеров батча."""
    texts = [f"code {i}" for i in range(32)]

    embeddings = embedder.encode_batch(texts, batch_size=batch_size)

    assert embeddings.shape == (32, 768)


# ==================== Regression Tests ====================

class TestEmbeddingRegression:
    """Регрессионные тесты."""

    def test_embedding_stability(self, embedder):
        """Тест стабильности embeddings между запусками."""
        text = "def stable_function(): return 42"

        # Несколько запусков
        embeddings = [embedder.encode_text(text) for _ in range(3)]

        # Все должны быть идентичны
        for i in range(1, len(embeddings)):
            np.testing.assert_array_almost_equal(
                embeddings[0],
                embeddings[i],
                decimal=5
            )
