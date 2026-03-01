"""
Автоматическая реиндексация при изменении файлов.

Мониторинг файловой системы и автоматическая реиндексация
измененных файлов кода.
"""

from pathlib import Path
import sys
import time
import hashlib
from typing import Dict, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..code_rag.core import (
    ParserFactory,
    CodeEmbedder,
    QdrantClient,
    VectorizationPipeline,
    CollectionSchema,
)

from ..code_rag.utils import (
    setup_logger,
    calculate_file_hash,
    format_duration,
)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ watchdog не установлен. Установите: pip install watchdog")


@dataclass
class FileIndex:
    """Индекс файла для отслеживания изменений."""
    path: Path
    hash: str
    last_modified: datetime
    indexed_at: datetime
    element_ids: Set[str] = field(default_factory=set)


class AutoReindexer:
    """Автоматическая реиндексация при изменении файлов."""

    def __init__(
            self,
            watch_path: Path,
            collection_name: str,
            repository_name: str,
            qdrant_url: str = "http://localhost:6333",
            debounce_seconds: float = 2.0
    ):
        """
        Инициализация auto-reindexer.

        Args:
            watch_path: Путь для мониторинга
            collection_name: Имя коллекции
            repository_name: Имя репозитория
            qdrant_url: URL Qdrant
            debounce_seconds: Задержка перед реиндексацией
        """
        self.watch_path = watch_path
        self.collection_name = collection_name
        self.repository_name = repository_name
        self.debounce_seconds = debounce_seconds

        self.logger = logging.getLogger(__name__)
        self.logger.info("🔄 Инициализация Auto-Reindexer...")

        # Индекс файлов
        self.file_index: Dict[str, FileIndex] = {}

        # Очередь изменений (для debounce)
        self.pending_changes: Dict[str, float] = {}

        # Компоненты
        self._init_components(qdrant_url)

        # Observer
        self.observer = None

    def _init_components(self, qdrant_url: str):
        """Инициализация компонентов."""
        self.logger.info("  📊 Загрузка embedder...")
        self.embedder = CodeEmbedder()

        self.logger.info("  🔗 Подключение к Qdrant...")
        self.qdrant_client = QdrantClient(url=qdrant_url)

        if not self.qdrant_client.health_check():
            raise ConnectionError("Qdrant недоступен!")

        # Pipeline
        self.pipeline = VectorizationPipeline(
            collection_name=self.collection_name,
            embedder=self.embedder,
            qdrant_client=self.qdrant_client
        )

        self.logger.info("  ✓ Компоненты готовы")

    def build_initial_index(self):
        """Построение начального индекса файлов."""
        self.logger.info(f"\n📋 Построение индекса файлов в {self.watch_path}...")

        count = 0
        for file_path in self.watch_path.rglob("*"):
            if file_path.is_file() and ParserFactory.can_parse_file(file_path):
                self._add_to_index(file_path)
                count += 1

        self.logger.info(f"  ✓ Проиндексировано файлов: {count}")

    def _add_to_index(self, file_path: Path):
        """Добавление файла в индекс."""
        try:
            file_hash = calculate_file_hash(file_path)
            file_stat = file_path.stat()

            self.file_index[str(file_path)] = FileIndex(
                path=file_path,
                hash=file_hash,
                last_modified=datetime.fromtimestamp(file_stat.st_mtime),
                indexed_at=datetime.now()
            )
        except Exception as e:
            self.logger.warning(f"  ⚠️ Ошибка индексирования {file_path}: {e}")

    def _remove_from_index(self, file_path: Path):
        """Удаление файла из индекса."""
        file_key = str(file_path)
        if file_key in self.file_index:
            del self.file_index[file_key]

    def check_file_changed(self, file_path: Path) -> bool:
        """
        Проверка изменения файла.

        Args:
            file_path: Путь к файлу

        Returns:
            True если файл изменился
        """
        file_key = str(file_path)

        if not file_path.exists():
            return False

        try:
            current_hash = calculate_file_hash(file_path)

            if file_key not in self.file_index:
                return True  # Новый файл

            old_hash = self.file_index[file_key].hash
            return current_hash != old_hash

        except Exception as e:
            self.logger.warning(f"  ⚠️ Ошибка проверки {file_path}: {e}")
            return False

    def reindex_file(self, file_path: Path):
        """
        Реиндексация файла.

        Args:
            file_path: Путь к файлу
        """
        self.logger.info(f"\n🔄 Реиндексация: {file_path.name}")

        try:
            # Удаляем старые записи
            file_key = str(file_path)
            if file_key in self.file_index:
                old_element_ids = self.file_index[file_key].element_ids
                if old_element_ids:
                    self.logger.info(f"  🗑️ Удаление {len(old_element_ids)} старых элементов...")
                    # TODO: удалить точки из Qdrant по ID

            # Индексируем заново
            repository_info = {"repository_name": self.repository_name}
            result = self.pipeline.process_file(file_path, repository_info)

            # Обновляем индекс
            self._add_to_index(file_path)

            self.logger.info(
                f"  ✓ Проиндексировано элементов: "
                f"{result.indexed_elements}/{result.total_elements}"
            )

        except Exception as e:
            self.logger.error(f"  ❌ Ошибка реиндексации: {e}")

    def delete_file_from_index(self, file_path: Path):
        """
        Удаление файла из индекса и Qdrant.

        Args:
            file_path: Путь к файлу
        """
        self.logger.info(f"\n🗑️ Удаление из индекса: {file_path.name}")

        file_key = str(file_path)
        if file_key in self.file_index:
            element_ids = self.file_index[file_key].element_ids
            if element_ids:
                self.logger.info(f"  🗑️ Удаление {len(element_ids)} элементов из Qdrant...")
                # TODO: удалить точки из Qdrant

            self._remove_from_index(file_path)
            self.logger.info("  ✓ Удалено")

    def process_pending_changes(self):
        """Обработка отложенных изменений (debounce)."""
        current_time = time.time()
        to_process = []

        for file_path, timestamp in list(self.pending_changes.items()):
            if current_time - timestamp >= self.debounce_seconds:
                to_process.append(file_path)
                del self.pending_changes[file_path]

        for file_path_str in to_process:
            file_path = Path(file_path_str)

            if not file_path.exists():
                self.delete_file_from_index(file_path)
            elif self.check_file_changed(file_path):
                self.reindex_file(file_path)

    def start_watching(self):
        """Запуск мониторинга файловой системы."""
        if not WATCHDOG_AVAILABLE:
            self.logger.error("❌ watchdog не установлен!")
            return

        self.logger.info(f"\n👀 Начало мониторинга: {self.watch_path}")
        self.logger.info(f"  ⏱️ Debounce: {self.debounce_seconds}s")

        # Event handler
        event_handler = FileChangeHandler(self)

        # Observer
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.watch_path), recursive=True)
        self.observer.start()

        self.logger.info("  ✓ Мониторинг запущен. Нажмите Ctrl+C для остановки.\n")

        try:
            while True:
                time.sleep(1)
                self.process_pending_changes()

        except KeyboardInterrupt:
            self.logger.info("\n\n⚠️ Остановка мониторинга...")
            self.observer.stop()

        self.observer.join()
        self.logger.info("✓ Мониторинг остановлен")


class FileChangeHandler(FileSystemEventHandler):
    """Обработчик событий файловой системы."""

    def __init__(self, reindexer: AutoReindexer):
        self.reindexer = reindexer
        self.logger = reindexer.logger

    def on_modified(self, event):
        """Обработка изменения файла."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not ParserFactory.can_parse_file(file_path):
            return

        self.logger.info(f"📝 Изменен: {file_path.name}")

        # Добавляем в очередь (debounce)
        self.reindexer.pending_changes[str(file_path)] = time.time()

    def on_created(self, event):
        """Обработка создания файла."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not ParserFactory.can_parse_file(file_path):
            return

        self.logger.info(f"➕ Создан: {file_path.name}")

        # Добавляем в очередь
        self.reindexer.pending_changes[str(file_path)] = time.time()

    def on_deleted(self, event):
        """Обработка удаления файла."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        self.logger.info(f"➖ Удален: {file_path.name}")

        # Добавляем в очередь
        self.reindexer.pending_changes[str(file_path)] = time.time()


def main():
    """Главная функция."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-reindexing для Code RAG")
    parser.add_argument("path", type=Path, help="Путь для мониторинга")
    parser.add_argument("--collection", default="auto_reindex_collection", help="Имя коллекции")
    parser.add_argument("--repo-name", default="monitored-repo", help="Имя репозитория")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="URL Qdrant")
    parser.add_argument("--debounce", type=float, default=2.0, help="Debounce секунды")
    parser.add_argument("--no-initial-index", action="store_true", help="Не строить начальный индекс")

    args = parser.parse_args()

    if not args.path.exists():
        print(f"❌ Путь не существует: {args.path}")
        return

    try:
        # Создаем reindexer
        reindexer = AutoReindexer(
            watch_path=args.path,
            collection_name=args.collection,
            repository_name=args.repo_name,
            qdrant_url=args.qdrant_url,
            debounce_seconds=args.debounce
        )

        # Построение начального индекса
        if not args.no_initial_index:
            reindexer.build_initial_index()

        # Запуск мониторинга
        reindexer.start_watching()

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
