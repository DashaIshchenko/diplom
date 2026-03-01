"""
Полный рабочий процесс Code RAG.

Демонстрирует реальный workflow:
1. Клонирование/подготовка репозитория
2. Создание и настройка коллекции
3. Полная индексация репозитория
4. Различные виды поиска
5. Генерация с RAG
6. Статистика и анализ
"""
import logging
from pathlib import Path
import sys
import time
import shutil

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..code_rag.core import (
    ParserFactory,
    CodeEmbedder,
    QdrantClient,
    VectorizationPipeline,
    VectorizationConfig,
    CollectionSchema,
    RAGRetriever,
    QwenIntegration,
    ProgrammingLanguage,
    CodeElementType,
)

from ..code_rag.utils import (
    setup_logger,
    format_duration,
    format_bytes,
    format_number,
)


class CodeRAGWorkflow:
    """Класс для управления полным workflow."""

    def __init__(
            self,
            collection_name: str = "my_code_collection",
            qdrant_url: str = "http://localhost:6333",
            log_level: str = "INFO"
    ):
        """
        Инициализация workflow.

        Args:
            collection_name: Имя коллекции Qdrant
            qdrant_url: URL Qdrant сервера
            log_level: Уровень логирования
        """
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)

        self.logger.info("=" * 70)
        self.logger.info("🚀 Code RAG - Полный Workflow")
        self.logger.info("=" * 70)

        # Инициализация компонентов
        self._init_components(qdrant_url)

    def _init_components(self, qdrant_url: str):
        """Инициализация всех компонентов."""
        self.logger.info("\n⚙️ Инициализация компонентов...")

        # Embedder
        self.logger.info("  📊 Загрузка embedder модели...")
        start = time.time()
        self.embedder = CodeEmbedder()
        self.logger.info(f"  ✓ Embedder готов (dim={self.embedder.embedding_dim}, "
                         f"{format_duration(time.time() - start)})")

        # Qdrant
        self.logger.info(f"  🔗 Подключение к Qdrant ({qdrant_url})...")
        self.qdrant_client = QdrantClient(url=qdrant_url)

        if not self.qdrant_client.health_check():
            raise ConnectionError(
                "❌ Qdrant недоступен!\n"
                "Запустите: docker run -p 6333:6333 qdrant/qdrant"
            )

        self.logger.info("  ✓ Qdrant подключен")

        # Pipeline
        self.pipeline = None
        self.retriever = None

    def setup_collection(self, recreate: bool = False):
        """
        Настройка коллекции.

        Args:
            recreate: Пересоздать если существует
        """
        self.logger.info(f"\n📦 Настройка коллекции: {self.collection_name}")

        exists = self.qdrant_client.collection_exists(self.collection_name)

        if exists and recreate:
            self.logger.info("  🗑️ Удаление существующей коллекции...")
            self.qdrant_client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.logger.info("  📝 Создание коллекции...")
            schema = CollectionSchema(
                collection_name=self.collection_name,
                vector_size=self.embedder.embedding_dim
            )
            self.qdrant_client.create_collection(schema)
            self.logger.info("  ✓ Коллекция создана")
        else:
            self.logger.info("  ✓ Коллекция существует")

        # Информация о коллекции
        info = self.qdrant_client.get_collection_info(self.collection_name)
        self.logger.info(f"  📊 Точек в коллекции: {format_number(info.get('points_count', 0))}")

    def index_repository(
            self,
            repo_path: Path,
            repository_name: str,
            branch: str = "main",
            exclude_dirs: list = None
    ):
        """
        Индексация репозитория.

        Args:
            repo_path: Путь к репозиторию
            repository_name: Имя репозитория
            branch: Ветка
            exclude_dirs: Директории для исключения
        """
        self.logger.info(f"\n📥 Индексация репозитория: {repository_name}")
        self.logger.info(f"  📁 Путь: {repo_path}")

        if not repo_path.exists():
            raise FileNotFoundError(f"Репозиторий не найден: {repo_path}")

        # Настройка pipeline
        config = VectorizationConfig(
            batch_size=100,
            embedding_batch_size=32,
            skip_errors=True,
            verbose=True
        )

        self.pipeline = VectorizationPipeline(
            collection_name=self.collection_name,
            embedder=self.embedder,
            qdrant_client=self.qdrant_client,
            config=config
        )

        # Прогресс callback
        def progress_callback(message: str, current: int, total: int):
            percent = (current / total) * 100 if total > 0 else 0
            self.logger.info(f"  [{current}/{total}] ({percent:.1f}%) {message}")

        # Индексация
        start_time = time.time()

        result = self.pipeline.process_repository(
            repository_path=repo_path,
            repository_name=repository_name,
            branch=branch,
            exclude_dirs=exclude_dirs or [
                ".git", "node_modules", "venv", ".venv",
                "__pycache__", "build", "dist"
            ],
            progress_callback=progress_callback
        )

        duration = time.time() - start_time

        # Результаты
        self.logger.info(f"\n✅ Индексация завершена за {format_duration(duration)}")
        self.logger.info(f"  📄 Файлов обработано: {result.parsed_files}/{result.total_files}")
        self.logger.info(f"  🔖 Элементов проиндексировано: {format_number(result.indexed_elements)}")
        self.logger.info(f"  ✓ Success rate: {result.success_rate():.1f}%")

        if result.errors:
            self.logger.warning(f"  ⚠️ Ошибок: {len(result.errors)}")
            for error in result.errors[:5]:  # Показываем первые 5
                self.logger.warning(f"    - {error['file']}: {error['error']}")

    def search_examples(self):
        """Примеры поиска."""
        self.logger.info("\n🔎 Примеры поиска кода...")

        # Создаем retriever
        self.retriever = RAGRetriever(
            collection_name=self.collection_name,
            embedder=self.embedder,
            qdrant_client=self.qdrant_client
        )

        # Примеры запросов
        queries = [
            ("Базовый поиск", "authentication function", None, None),
            ("По языку", "database connection", ProgrammingLanguage.PYTHON, None),
            ("По типу", "user class", None, CodeElementType.CLASS),
            ("API эндпоинты", "REST API endpoints", None, CodeElementType.FUNCTION),
        ]

        for title, query, language, element_type in queries:
            self.logger.info(f"\n  📝 {title}: '{query}'")

            if language:
                self.logger.info(f"     Язык: {language.value}")
            if element_type:
                self.logger.info(f"     Тип: {element_type.value}")

            results = self.retriever.search(
                query=query,
                top_k=3,
                language=language,
                element_type=element_type
            )

            if not results:
                self.logger.info("     ℹ️ Результатов не найдено")
                continue

            for i, result in enumerate(results, 1):
                element = result.element
                self.logger.info(
                    f"     {i}. {element.qualified_name} "
                    f"({element.type.value}, score: {result.score:.3f})"
                )
                self.logger.info(f"        📁 {element.location.file_path}:{element.location.start_line}")

    def rag_examples(self, api_key: str = None):
        """
        Примеры RAG генерации.

        Args:
            api_key: Qwen API ключ
        """
        if not api_key:
            self.logger.warning("\n⚠️ QWEN_API_KEY не установлен, пропускаем RAG примеры")
            return

        self.logger.info("\n🤖 Примеры RAG генерации...")

        try:
            qwen = QwenIntegration(
                api_key=api_key,
                model="qwen2.5-coder-32b-instruct"
            )

            # Примеры вопросов
            questions = [
                "How is authentication implemented in this codebase?",
                "What database operations are available?",
                "Show me API endpoints for user management",
            ]

            for question in questions:
                self.logger.info(f"\n  ❓ Вопрос: {question}")

                # Получаем контекст
                context_results = self.retriever.search(query=question, top_k=5)
                self.logger.info(f"     📚 Найдено контекстных элементов: {len(context_results)}")

                # Генерируем ответ
                start = time.time()
                response = qwen.answer_question_with_rag(
                    question=question,
                    context_results=context_results
                )
                duration = time.time() - start

                # Выводим ответ (первые 300 символов)
                self.logger.info(f"     💬 Ответ ({format_duration(duration)}):")
                answer_preview = response.content[:300].replace('\n', '\n        ')
                self.logger.info(f"        {answer_preview}...")
                self.logger.info(f"     🔢 Токенов: {response.tokens_used}")

        except Exception as e:
            self.logger.error(f"  ❌ Ошибка RAG: {e}")

    def show_statistics(self):
        """Показать статистику коллекции."""
        self.logger.info("\n📊 Статистика коллекции...")

        info = self.qdrant_client.get_collection_info(self.collection_name)

        self.logger.info(f"  📦 Коллекция: {self.collection_name}")
        self.logger.info(f"  🔢 Точек: {format_number(info.get('points_count', 0))}")
        self.logger.info(f"  📏 Размер вектора: {info.get('vector_size', 0)}")
        self.logger.info(f"  💾 Векторов: {format_number(info.get('vectors_count', 0))}")

    def cleanup(self, delete_collection: bool = False):
        """
        Очистка ресурсов.

        Args:
            delete_collection: Удалить коллекцию
        """
        self.logger.info("\n🧹 Очистка...")

        if delete_collection:
            self.logger.info(f"  🗑️ Удаление коллекции {self.collection_name}...")
            self.qdrant_client.delete_collection(self.collection_name)
            self.logger.info("  ✓ Коллекция удалена")

        self.logger.info("  ✓ Очистка завершена")


def main():
    """Главная функция."""
    import os

    # Настройки
    COLLECTION_NAME = "demo_collection"
    QDRANT_URL = "http://localhost:6333"
    QWEN_API_KEY = os.getenv("QWEN_API_KEY")

    # Путь к репозиторию для индексации
    # Можно использовать текущий проект или любой другой
    REPO_PATH = Path(__file__).parent.parent  # Корень проекта
    REPO_NAME = "code-rag-demo"

    try:
        # Создаем workflow
        workflow = CodeRAGWorkflow(
            collection_name=COLLECTION_NAME,
            qdrant_url=QDRANT_URL,
            log_level="INFO"
        )

        # 1. Настройка коллекции
        workflow.setup_collection(recreate=True)

        # 2. Индексация репозитория
        workflow.index_repository(
            repo_path=REPO_PATH,
            repository_name=REPO_NAME,
            branch="main"
        )

        # 3. Примеры поиска
        workflow.search_examples()

        # 4. RAG примеры (если есть API ключ)
        workflow.rag_examples(api_key=QWEN_API_KEY)

        # 5. Статистика
        workflow.show_statistics()

        # 6. Очистка (опционально)
        # workflow.cleanup(delete_collection=True)

        workflow.logger.info("\n" + "=" * 70)
        workflow.logger.info("✅ Workflow успешно завершён!")
        workflow.logger.info("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
