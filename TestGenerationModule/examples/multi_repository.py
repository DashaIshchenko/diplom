"""
Работа с множественными репозиториями.

Демонстрирует:
1. Индексация нескольких репозиториев в одну коллекцию
2. Поиск по всем репозиториям
3. Фильтрация по репозиторию
4. Сравнение реализаций между репозиториями
5. Статистика по репозиториям
"""
import logging
from pathlib import Path
import sys
import time
from typing import List, Dict, Any
from dataclasses import dataclass

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_rag.core import (
    CodeEmbedder,
    QdrantClient,
    VectorizationPipeline,
    VectorizationConfig,
    CollectionSchema,
    RAGRetriever,
    ProgrammingLanguage,
)

from code_rag.utils import (
    setup_logger,
    format_duration,
    format_number,
)


@dataclass
class RepositoryConfig:
    """Конфигурация репозитория."""
    name: str
    path: Path
    branch: str = "main"
    provider: str = "local"
    exclude_dirs: List[str] = None

    def __post_init__(self):
        if self.exclude_dirs is None:
            self.exclude_dirs = [
                ".git", "node_modules", "venv", ".venv",
                "__pycache__", "build", "dist", "target"
            ]


class MultiRepositoryManager:
    """Менеджер для работы с несколькими репозиториями."""

    def __init__(
            self,
            collection_name: str = "multi_repo_collection",
            qdrant_url: str = "http://localhost:6333"
    ):
        """
        Инициализация менеджера.

        Args:
            collection_name: Имя коллекции
            qdrant_url: URL Qdrant
        """
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)

        self.logger.info("=" * 70)
        self.logger.info("🗂️ Multi-Repository Manager")
        self.logger.info("=" * 70)

        # Инициализация компонентов
        self._init_components(qdrant_url)

        # Статистика по репозиториям
        self.repository_stats: Dict[str, Dict[str, Any]] = {}

    def _init_components(self, qdrant_url: str):
        """Инициализация компонентов."""
        self.logger.info("\n⚙️ Инициализация компонентов...")

        # Embedder
        self.logger.info("  📊 Загрузка embedder...")
        start = time.time()
        self.embedder = CodeEmbedder()
        self.logger.info(f"  ✓ Embedder готов ({format_duration(time.time() - start)})")

        # Qdrant
        self.logger.info(f"  🔗 Подключение к Qdrant...")
        self.qdrant_client = QdrantClient(url=qdrant_url)

        if not self.qdrant_client.health_check():
            raise ConnectionError("Qdrant недоступен!")

        self.logger.info("  ✓ Qdrant подключен")

        # RAG Retriever
        self.retriever = None

    def setup_collection(self, recreate: bool = False):
        """Настройка коллекции."""
        self.logger.info(f"\n📦 Настройка коллекции: {self.collection_name}")

        exists = self.qdrant_client.collection_exists(self.collection_name)

        if exists and recreate:
            self.logger.info("  🗑️ Удаление существующей коллекции...")
            self.qdrant_client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            schema = CollectionSchema(
                collection_name=self.collection_name,
                vector_size=self.embedder.embedding_dim
            )
            self.qdrant_client.create_collection(schema)
            self.logger.info("  ✓ Коллекция создана")
        else:
            self.logger.info("  ✓ Коллекция существует")

    def index_repository(self, config: RepositoryConfig):
        """
        Индексация репозитория.

        Args:
            config: Конфигурация репозитория
        """
        self.logger.info(f"\n📥 Индексация: {config.name}")
        self.logger.info(f"  📁 Путь: {config.path}")

        if not config.path.exists():
            self.logger.error(f"  ❌ Репозиторий не найден: {config.path}")
            return

        # Pipeline config
        pipeline_config = VectorizationConfig(
            batch_size=100,
            embedding_batch_size=32,
            skip_errors=True,
            verbose=False
        )

        pipeline = VectorizationPipeline(
            collection_name=self.collection_name,
            embedder=self.embedder,
            qdrant_client=self.qdrant_client,
            config=pipeline_config
        )

        # Индексация
        start_time = time.time()

        result = pipeline.process_repository(
            repository_path=config.path,
            repository_name=config.name,
            branch=config.branch,
            provider=config.provider,
            exclude_dirs=config.exclude_dirs
        )

        duration = time.time() - start_time

        # Статистика
        self.repository_stats[config.name] = {
            "files": result.parsed_files,
            "elements": result.indexed_elements,
            "duration": duration,
            "success_rate": result.success_rate()
        }

        # Результаты
        self.logger.info(f"  ✓ Завершено за {format_duration(duration)}")
        self.logger.info(f"    📄 Файлов: {result.parsed_files}/{result.total_files}")
        self.logger.info(f"    🔖 Элементов: {format_number(result.indexed_elements)}")
        self.logger.info(f"    ✓ Success: {result.success_rate():.1f}%")

    def index_multiple(self, configs: List[RepositoryConfig]):
        """
        Индексация нескольких репозиториев.

        Args:
            configs: Список конфигураций
        """
        self.logger.info(f"\n📚 Индексация {len(configs)} репозиториев...")

        total_start = time.time()

        for i, config in enumerate(configs, 1):
            self.logger.info(f"\n[{i}/{len(configs)}] Обработка {config.name}")
            self.index_repository(config)

        total_duration = time.time() - total_start

        self.logger.info(f"\n✅ Все репозитории проиндексированы за {format_duration(total_duration)}")

    def setup_retriever(self):
        """Настройка retriever."""
        if self.retriever is None:
            self.retriever = RAGRetriever(
                collection_name=self.collection_name,
                embedder=self.embedder,
                qdrant_client=self.qdrant_client
            )

    def search_all_repositories(self, query: str, top_k: int = 5):
        """
        Поиск по всем репозиториям.

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
        """
        self.setup_retriever()

        self.logger.info(f"\n🔎 Поиск по всем репозиториям: '{query}'")

        results = self.retriever.search(query=query, top_k=top_k)

        if not results:
            self.logger.info("  ℹ️ Результатов не найдено")
            return

        # Группируем по репозиториям
        by_repo: Dict[str, List] = {}

        for result in results:
            repo_name = getattr(result.element, 'repository_name', 'unknown')
            if repo_name not in by_repo:
                by_repo[repo_name] = []
            by_repo[repo_name].append(result)

        # Выводим результаты
        for repo_name, repo_results in by_repo.items():
            self.logger.info(f"\n  📦 {repo_name} ({len(repo_results)} результатов):")

            for result in repo_results:
                element = result.element
                self.logger.info(
                    f"    • {element.qualified_name} "
                    f"({element.type.value}, score: {result.score:.3f})"
                )
                self.logger.info(f"      📁 {element.location.file_path}:{element.location.start_line}")

    def search_in_repository(self, query: str, repository_name: str, top_k: int = 5):
        """
        Поиск в конкретном репозитории.

        Args:
            query: Поисковый запрос
            repository_name: Имя репозитория
            top_k: Количество результатов
        """
        self.setup_retriever()

        self.logger.info(f"\n🔎 Поиск в '{repository_name}': '{query}'")

        results = self.retriever.search(
            query=query,
            top_k=top_k,
            repository_name=repository_name
        )

        if not results:
            self.logger.info("  ℹ️ Результатов не найдено")
            return

        for i, result in enumerate(results, 1):
            element = result.element
            self.logger.info(
                f"  {i}. {element.qualified_name} "
                f"({element.type.value}, score: {result.score:.3f})"
            )
            self.logger.info(f"     📁 {element.location.file_path}:{element.location.start_line}")

    def compare_implementations(self, query: str, top_k: int = 3):
        """
        Сравнение реализаций между репозиториями.

        Args:
            query: Поисковый запрос
            top_k: Количество результатов на репозиторий
        """
        self.setup_retriever()

        self.logger.info(f"\n🔄 Сравнение реализаций: '{query}'")

        # Получаем результаты по всем репозиториям
        all_results = self.retriever.search(query=query, top_k=top_k * len(self.repository_stats))

        # Группируем по репозиториям
        by_repo: Dict[str, List] = {}

        for result in all_results:
            repo_name = getattr(result.element, 'repository_name', 'unknown')
            if repo_name not in by_repo:
                by_repo[repo_name] = []
            if len(by_repo[repo_name]) < top_k:
                by_repo[repo_name].append(result)

        # Выводим сравнение
        for repo_name, results in by_repo.items():
            self.logger.info(f"\n  📦 {repo_name}:")

            for result in results:
                element = result.element
                self.logger.info(f"    • {element.qualified_name} (score: {result.score:.3f})")

                # Показываем сигнатуру или первую строку
                if element.signature:
                    self.logger.info(f"      {element.signature}")
                elif element.source_code:
                    first_line = element.source_code.split('\n')[0][:80]
                    self.logger.info(f"      {first_line}...")

    def show_repository_statistics(self):
        """Показать статистику по всем репозиториям."""
        self.logger.info("\n📊 Статистика репозиториев:")
        self.logger.info("=" * 70)

        total_files = 0
        total_elements = 0

        for repo_name, stats in self.repository_stats.items():
            self.logger.info(f"\n  📦 {repo_name}:")
            self.logger.info(f"    📄 Файлов: {format_number(stats['files'])}")
            self.logger.info(f"    🔖 Элементов: {format_number(stats['elements'])}")
            self.logger.info(f"    ⏱️ Время: {format_duration(stats['duration'])}")
            self.logger.info(f"    ✓ Success: {stats['success_rate']:.1f}%")

            total_files += stats['files']
            total_elements += stats['elements']

        self.logger.info(f"\n  {'─' * 68}")
        self.logger.info(f"  📊 Всего:")
        self.logger.info(f"    📄 Файлов: {format_number(total_files)}")
        self.logger.info(f"    🔖 Элементов: {format_number(total_elements)}")
        self.logger.info("=" * 70)

    def list_repositories(self) -> List[str]:
        """Получить список репозиториев в коллекции."""
        self.logger.info("\n📋 Список репозиториев:")

        from code_rag.core.vector_db import SearchFilters

        repositories = set()
        offset = None

        while True:
            results, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
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

        repo_list = sorted(list(repositories))

        for i, repo in enumerate(repo_list, 1):
            self.logger.info(f"  {i}. {repo}")

        return repo_list


def main():
    """Главная функция."""

    # Конфигурации репозиториев для индексации
    repositories = [
        RepositoryConfig(
            name="project-a",
            path=Path("path/to/project-a"),
            branch="main"
        ),
        RepositoryConfig(
            name="project-b",
            path=Path("path/to/project-b"),
            branch="develop"
        ),
        RepositoryConfig(
            name="project-c",
            path=Path("path/to/project-c"),
            branch="main"
        ),
    ]

    # Для демо используем текущий проект несколько раз
    demo_path = Path(__file__).parent.parent / "code_rag"

    if demo_path.exists():
        repositories = [
            RepositoryConfig(name="core-module", path=demo_path / "core"),
            RepositoryConfig(name="api-module", path=demo_path / "api"),
            RepositoryConfig(name="utils-module", path=demo_path / "utils"),
        ]

    try:
        # Создаем менеджер
        manager = MultiRepositoryManager(
            collection_name="multi_repo_demo",
            qdrant_url="http://localhost:6333"
        )

        # 1. Настройка коллекции
        manager.setup_collection(recreate=True)

        # 2. Индексация всех репозиториев
        manager.index_multiple(repositories)

        # 3. Статистика
        manager.show_repository_statistics()

        # 4. Список репозиториев
        manager.list_repositories()

        # 5. Поиск по всем репозиториям
        manager.search_all_repositories("parse function", top_k=6)

        # 6. Поиск в конкретном репозитории
        if repositories:
            manager.search_in_repository(
                "logger setup",
                repository_name=repositories[0].name,
                top_k=3
            )

        # 7. Сравнение реализаций
        manager.compare_implementations("configuration class", top_k=2)

        manager.logger.info("\n✅ Пример завершён!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
