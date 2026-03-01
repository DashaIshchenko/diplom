"""
Продвинутые техники RAG.

Демонстрирует:
1. Гибридный поиск (semantic + keyword)
2. Reranking результатов
3. Контекстное окно с приоритетами
4. Цепочки запросов (chain of thought)
5. Самопроверка и улучшение ответов
6. Кэширование контекста
"""
import logging
from pathlib import Path
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import hashlib

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..code_rag.core import (
    CodeEmbedder,
    QdrantClient,
    RAGRetriever,
    QwenIntegration,
    SearchResult,
    ProgrammingLanguage,
)

from ..code_rag.utils import (
    setup_logger,
    format_duration,
)


@dataclass
class AdvancedRAGConfig:
    """Конфигурация для Advanced RAG."""
    # Поиск
    top_k: int = 10
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    use_reranking: bool = True
    rerank_top_k: int = 5

    # Контекст
    max_context_tokens: int = 8000
    context_prioritization: str = "score"  # score, recency, diversity

    # Генерация
    use_chain_of_thought: bool = True
    self_reflection: bool = True
    temperature: float = 0.7

    # Кэш
    cache_enabled: bool = True
    cache_ttl: int = 3600  # секунд


class AdvancedRAGPipeline:
    """Продвинутый RAG pipeline."""

    def __init__(
            self,
            collection_name: str,
            embedder: CodeEmbedder,
            qdrant_client: QdrantClient,
            qwen: Optional[QwenIntegration] = None,
            config: Optional[AdvancedRAGConfig] = None
    ):
        """
        Инициализация Advanced RAG.

        Args:
            collection_name: Имя коллекции
            embedder: Code embedder
            qdrant_client: Qdrant client
            qwen: Qwen integration (опционально)
            config: Конфигурация
        """
        self.collection_name = collection_name
        self.embedder = embedder
        self.qdrant_client = qdrant_client
        self.qwen = qwen
        self.config = config or AdvancedRAGConfig()

        self.logger = logging.getLogger(__name__)

        # Retriever
        self.retriever = RAGRetriever(
            collection_name=collection_name,
            embedder=embedder,
            qdrant_client=qdrant_client,
            use_reranking=self.config.use_reranking
        )

        # Кэш контекста
        self.context_cache: Dict[str, Tuple[List[SearchResult], float]] = {}

        self.logger.info("🚀 Advanced RAG Pipeline инициализирован")

    def _cache_key(self, query: str) -> str:
        """Генерация ключа кэша."""
        return hashlib.md5(query.encode()).hexdigest()

    def _get_cached_context(self, query: str) -> Optional[List[SearchResult]]:
        """Получение контекста из кэша."""
        if not self.config.cache_enabled:
            return None

        cache_key = self._cache_key(query)

        if cache_key in self.context_cache:
            results, timestamp = self.context_cache[cache_key]

            # Проверяем TTL
            if time.time() - timestamp < self.config.cache_ttl:
                self.logger.info("  💾 Контекст из кэша")
                return results
            else:
                # Удаляем устаревший
                del self.context_cache[cache_key]

        return None

    def _cache_context(self, query: str, results: List[SearchResult]):
        """Сохранение контекста в кэш."""
        if not self.config.cache_enabled:
            return

        cache_key = self._cache_key(query)
        self.context_cache[cache_key] = (results, time.time())

    def hybrid_search(
            self,
            query: str,
            language: Optional[ProgrammingLanguage] = None
    ) -> List[SearchResult]:
        """
        Гибридный поиск (semantic + keyword).

        Args:
            query: Поисковый запрос
            language: Фильтр по языку

        Returns:
            Список результатов
        """
        self.logger.info(f"🔍 Гибридный поиск: '{query}'")

        # Проверяем кэш
        cached = self._get_cached_context(query)
        if cached:
            return cached

        # Семантический поиск
        semantic_results = self.retriever.search(
            query=query,
            top_k=self.config.top_k,
            language=language
        )

        # Keyword поиск (упрощенная версия - через filter)
        # В реальной реализации можно использовать BM25 или другие алгоритмы
        keyword_results = self._keyword_search(query, language)

        # Объединяем результаты с весами
        combined = self._combine_results(
            semantic_results,
            keyword_results,
            semantic_weight=self.config.semantic_weight,
            keyword_weight=self.config.keyword_weight
        )

        # Reranking
        if self.config.use_reranking and len(combined) > self.config.rerank_top_k:
            combined = self._rerank_results(query, combined)[:self.config.rerank_top_k]

        # Кэшируем
        self._cache_context(query, combined)

        self.logger.info(f"  ✓ Найдено результатов: {len(combined)}")

        return combined

    def _keyword_search(
            self,
            query: str,
            language: Optional[ProgrammingLanguage]
    ) -> List[SearchResult]:
        """Keyword поиск (упрощенная версия)."""
        # В реальной реализации здесь был бы BM25 или full-text search
        # Для примера просто возвращаем пустой список
        return []

    def _combine_results(
            self,
            semantic: List[SearchResult],
            keyword: List[SearchResult],
            semantic_weight: float,
            keyword_weight: float
    ) -> List[SearchResult]:
        """Объединение результатов с весами."""
        # Собираем уникальные результаты
        seen = {}

        for result in semantic:
            key = result.element.qualified_name
            if key not in seen:
                seen[key] = result
                # Применяем вес
                result.score *= semantic_weight

        for result in keyword:
            key = result.element.qualified_name
            if key in seen:
                # Увеличиваем score
                seen[key].score += result.score * keyword_weight
            else:
                result.score *= keyword_weight
                seen[key] = result

        # Сортируем по score
        combined = sorted(seen.values(), key=lambda x: x.score, reverse=True)

        return combined

    def _rerank_results(
            self,
            query: str,
            results: List[SearchResult]
    ) -> List[SearchResult]:
        """Reranking результатов."""
        self.logger.info("  🔄 Reranking результатов...")

        # Здесь можно использовать более сложные алгоритмы:
        # - Cross-encoder модели
        # - LLM-based reranking
        # - Custom scoring

        # Для примера просто возвращаем как есть
        return results

    def prioritize_context(
            self,
            results: List[SearchResult],
            strategy: str = "score"
    ) -> List[SearchResult]:
        """
        Приоритизация контекста.

        Args:
            results: Результаты поиска
            strategy: Стратегия (score, recency, diversity)

        Returns:
            Приоритизированные результаты
        """
        self.logger.info(f"  📊 Приоритизация контекста: {strategy}")

        if strategy == "score":
            # Уже отсортировано по score
            return results

        elif strategy == "recency":
            # Сортируем по времени индексации
            return sorted(
                results,
                key=lambda x: getattr(x.element, 'indexed_at', ''),
                reverse=True
            )

        elif strategy == "diversity":
            # Максимизируем разнообразие (файлы, типы)
            return self._diversify_results(results)

        return results

    def _diversify_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Увеличение разнообразия результатов."""
        diversified = []
        seen_files = set()
        seen_types = defaultdict(int)

        # Берем по одному из каждого файла/типа
        for result in results:
            file_path = str(result.element.location.file_path)
            elem_type = result.element.type.value

            # Добавляем с учетом разнообразия
            if file_path not in seen_files or seen_types[elem_type] < 2:
                diversified.append(result)
                seen_files.add(file_path)
                seen_types[elem_type] += 1

        return diversified

    def generate_with_chain_of_thought(
            self,
            question: str,
            context_results: List[SearchResult]
    ) -> str:
        """
        Генерация с chain of thought.

        Args:
            question: Вопрос
            context_results: Контекст

        Returns:
            Ответ
        """
        if not self.qwen:
            raise ValueError("Qwen не инициализирован")

        self.logger.info("🤔 Генерация с Chain of Thought...")

        # Шаг 1: Анализ вопроса
        analysis_prompt = f"""Analyze this question step by step:
Question: {question}

Break down:
1. What is being asked?
2. What information is needed?
3. What type of answer is expected?

Provide brief analysis:"""

        analysis = self.qwen.generate(analysis_prompt)
        self.logger.info(f"  📝 Анализ вопроса:\n{analysis.content[:200]}...")

        # Шаг 2: Генерация ответа с контекстом
        response = self.qwen.answer_question_with_rag(
            question=question,
            context_results=context_results
        )

        return response.content

    def self_reflect_and_improve(
            self,
            question: str,
            initial_answer: str,
            context_results: List[SearchResult]
    ) -> str:
        """
        Самопроверка и улучшение ответа.

        Args:
            question: Вопрос
            initial_answer: Первоначальный ответ
            context_results: Контекст

        Returns:
            Улучшенный ответ
        """
        if not self.qwen:
            return initial_answer

        self.logger.info("🔄 Самопроверка и улучшение...")

        # Проверка ответа
        reflection_prompt = f"""Review this answer and identify issues:

Question: {question}

Answer: {initial_answer}

Check for:
1. Accuracy
2. Completeness
3. Clarity
4. Code examples quality

Provide brief critique:"""

        reflection = self.qwen.generate(reflection_prompt)
        self.logger.info(f"  💭 Рефлексия:\n{reflection.content[:200]}...")

        # Если есть проблемы, генерируем улучшенную версию
        if "issue" in reflection.content.lower() or "improve" in reflection.content.lower():
            self.logger.info("  ✏️ Улучшение ответа...")

            improvement_prompt = f"""Improve this answer based on the critique:

Question: {question}
Original Answer: {initial_answer}
Critique: {reflection.content}

Provide improved answer:"""

            improved = self.qwen.generate(improvement_prompt)
            return improved.content

        return initial_answer

    def advanced_query(
            self,
            question: str,
            language: Optional[ProgrammingLanguage] = None
    ) -> Dict[str, Any]:
        """
        Полный продвинутый RAG pipeline.

        Args:
            question: Вопрос
            language: Фильтр по языку

        Returns:
            Результат с метаданными
        """
        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"❓ Вопрос: {question}")
        self.logger.info(f"{'=' * 70}")

        start_time = time.time()

        # 1. Гибридный поиск
        results = self.hybrid_search(question, language)

        # 2. Приоритизация
        results = self.prioritize_context(
            results,
            strategy=self.config.context_prioritization
        )

        if not self.qwen:
            return {
                "answer": "Qwen не инициализирован",
                "context_count": len(results),
                "duration": time.time() - start_time
            }

        # 3. Генерация
        if self.config.use_chain_of_thought:
            answer = self.generate_with_chain_of_thought(question, results)
        else:
            response = self.qwen.answer_question_with_rag(question, results)
            answer = response.content

        # 4. Самопроверка
        if self.config.self_reflection:
            answer = self.self_reflect_and_improve(question, answer, results)

        duration = time.time() - start_time

        # Результат
        result = {
            "answer": answer,
            "context_count": len(results),
            "duration": duration,
            "context_sources": [
                {
                    "name": r.element.qualified_name,
                    "file": str(r.element.location.file_path),
                    "score": r.score
                }
                for r in results[:5]
            ]
        }

        # Вывод
        self.logger.info(f"\n💬 Ответ:\n{answer[:500]}...")
        self.logger.info(f"\n📊 Метаданные:")
        self.logger.info(f"  ⏱️ Время: {format_duration(duration)}")
        self.logger.info(f"  📚 Контекст: {len(results)} элементов")

        return result


def main():
    """Главная функция."""
    import os

    # Настройки
    COLLECTION_NAME = "advanced_rag_demo"
    QDRANT_URL = "http://localhost:6333"
    QWEN_API_KEY = os.getenv("QWEN_API_KEY")

    try:
        # Инициализация компонентов
        logger = logging.getLogger(__name__)

        logger.info("⚙️ Инициализация компонентов...")

        embedder = CodeEmbedder()
        qdrant_client = QdrantClient(url=QDRANT_URL)

        if not qdrant_client.health_check():
            logger.error("❌ Qdrant недоступен!")
            return

        # Qwen (опционально)
        qwen = None
        if QWEN_API_KEY:
            qwen = QwenIntegration(
                api_key=QWEN_API_KEY,
                model="qwen2.5-coder-32b-instruct"
            )
        else:
            logger.warning("⚠️ QWEN_API_KEY не установлен")

        # Конфигурация Advanced RAG
        config = AdvancedRAGConfig(
            top_k=10,
            use_reranking=True,
            rerank_top_k=5,
            use_chain_of_thought=True,
            self_reflection=True,
            cache_enabled=True
        )

        # Создаем pipeline
        pipeline = AdvancedRAGPipeline(
            collection_name=COLLECTION_NAME,
            embedder=embedder,
            qdrant_client=qdrant_client,
            qwen=qwen,
            config=config
        )

        # Примеры запросов
        questions = [
            "How is authentication implemented in this codebase?",
            "Show me examples of database queries",
            "What design patterns are used?",
        ]

        for question in questions:
            result = pipeline.advanced_query(question)

            logger.info("\n" + "=" * 70 + "\n")
            time.sleep(2)  # Пауза между запросами

        logger.info("✅ Advanced RAG пример завершён!")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
