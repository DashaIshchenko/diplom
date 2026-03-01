"""
Pipeline для векторизации и индексации кода в Qdrant.
Объединяет парсинг, эмбеддинг и хранение в векторной БД.
"""

from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

from ..parser.factory import ParserFactory
from ..parser.code_structure import (
    CodeElement, ModuleInfo, ProgrammingLanguage, ParseResult
)
from ..embeddings import CodeEmbedder
from .qdrant_client import QdrantClient
from .schemas import (
    CodePayload, CollectionSchema, PointData, BatchInsertData,
    DEFAULT_CODE_SCHEMA
)

logger = logging.getLogger(__name__)


@dataclass
class VectorizationConfig:
    """Конфигурация pipeline векторизации."""
    # Настройки батчинга
    batch_size: int = 100
    max_workers: int = 4

    # Фильтры элементов
    min_lines: int = 1
    max_lines: int = 1000
    include_types: Optional[List[str]] = None
    exclude_types: Optional[List[str]] = None

    # Настройки эмбеддинга
    embedding_batch_size: int = 32

    # Флаги
    skip_errors: bool = True
    verbose: bool = True


@dataclass
class VectorizationResult:
    """Результат векторизации."""
    total_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0

    total_elements: int = 0
    vectorized_elements: int = 0
    indexed_elements: int = 0
    skipped_elements: int = 0

    errors: List[Dict[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Процент успешно обработанных файлов."""
        if self.total_files == 0:
            return 0.0
        return (self.parsed_files / self.total_files) * 100

    @property
    def indexing_rate(self) -> float:
        """Процент проиндексированных элементов."""
        if self.total_elements == 0:
            return 0.0
        return (self.indexed_elements / self.total_elements) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "total_files": self.total_files,
            "parsed_files": self.parsed_files,
            "failed_files": self.failed_files,
            "total_elements": self.total_elements,
            "vectorized_elements": self.vectorized_elements,
            "indexed_elements": self.indexed_elements,
            "skipped_elements": self.skipped_elements,
            "success_rate": self.success_rate,
            "indexing_rate": self.indexing_rate,
            "errors_count": len(self.errors)
        }

    def __repr__(self) -> str:
        return (
            f"VectorizationResult("
            f"files={self.parsed_files}/{self.total_files}, "
            f"indexed={self.indexed_elements}/{self.total_elements}, "
            f"success_rate={self.success_rate:.1f}%)"
        )


class VectorizationPipeline:
    """
    Pipeline для векторизации и индексации кода.

    Этапы:
    1. Парсинг файлов кода
    2. Фильтрация элементов
    3. Векторизация (эмбеддинг)
    4. Индексация в Qdrant
    """

    def __init__(
            self,
            collection_name: str,
            embedder: CodeEmbedder,
            qdrant_client: QdrantClient,
            config: Optional[VectorizationConfig] = None
    ):
        """
        Инициализация pipeline.

        Args:
            collection_name: Имя коллекции Qdrant
            embedder: Embedder для векторизации
            qdrant_client: Клиент Qdrant
            config: Конфигурация pipeline
        """
        self.collection_name = collection_name
        self.embedder = embedder
        self.qdrant_client = qdrant_client
        self.config = config or VectorizationConfig()

        logger.info(f"VectorizationPipeline инициализирован для коллекции '{collection_name}'")

    def process_directory(
            self,
            directory: Path,
            repository_info: Optional[Dict[str, Any]] = None,
            recursive: bool = True,
            file_pattern: str = "*",
            progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> VectorizationResult:
        """
        Обработка директории с кодом.

        Args:
            directory: Путь к директории
            repository_info: Информация о репозитории
            recursive: Рекурсивный поиск файлов
            file_pattern: Паттерн для поиска файлов
            progress_callback: Callback для прогресса (message, current, total)

        Returns:
            VectorizationResult с результатами
        """
        logger.info(f"Начало обработки директории: {directory}")

        result = VectorizationResult()

        # Находим все файлы
        if recursive:
            files = list(directory.rglob(file_pattern))
        else:
            files = list(directory.glob(file_pattern))

        result.total_files = len(files)
        logger.info(f"Найдено {result.total_files} файлов для в каталоге {directory}")

        # Фильтруем файлы по расширению
        code_files = [
            f for f in files
            if f.is_file() and ParserFactory.can_parse_file(f)
        ]



        if result.total_files == 0:
            return result

        # Обрабатываем файлы
        for idx, file_path in enumerate(code_files, 1):
            if progress_callback:
                progress_callback(f"Обработка {file_path.name}", idx, result.total_files)

            try:
                file_result = self.process_file(file_path, repository_info)

                result.parsed_files += 1
                result.total_elements += file_result.total_elements
                result.vectorized_elements += file_result.vectorized_elements
                result.indexed_elements += file_result.indexed_elements
                result.skipped_elements += file_result.skipped_elements
                result.failed_files += file_result.failed_files
                result.errors += file_result.errors

            except Exception as e:
                result.failed_files += 1
                error_msg = f"Ошибка обработки {file_path}: {e}"
                logger.error(error_msg)
                result.errors.append({
                    "file": str(file_path),
                    "error": str(e)
                })

                if not self.config.skip_errors:
                    raise

        logger.info(f"Обработка завершена: {result}")

        return result

    def process_file(
            self,
            file_path: Path,
            repository_info: Optional[Dict[str, Any]] = None
    ) -> VectorizationResult:
        """
        Обработка одного файла.

        Args:
            file_path: Путь к файлу
            repository_info: Информация о репозитории

        Returns:
            VectorizationResult для файла
        """
        result = VectorizationResult()
        result.total_files = 1

        try:
            # 1. Парсинг файла
            parser = ParserFactory.create_parser_for_file(file_path)
            if not parser:
                logger.warning(f"Парсер не найден для {file_path}")
                result.failed_files += 1
                return result

            module = parser.parse_file(
                file_path,
                repository_name=repository_info.get("repository_name") if repository_info else None,
                branch=repository_info.get("branch") if repository_info else None,
                commit_hash=repository_info.get("commit_hash") if repository_info else None,
                provider=repository_info.get("provider") if repository_info else None
            )

            if not module:
                logger.warning(f"Не удалось распарсить {file_path}")
                result.failed_files += 1
                result.errors.append({
                    "file": str(file_path),
                    "error": "module does not exist for parser.parse_file"
                })
                return result

            result.parsed_files += 1

            # 2. Фильтрация элементов
            elements = self._filter_elements(module.all_elements)
            result.total_elements = len(elements)

            if result.total_elements == 0:
                logger.debug(f"Нет элементов для индексации в {file_path}")
                return result

            # 3. Векторизация и индексация
            indexed = self._vectorize_and_index(elements, repository_info)
            result.vectorized_elements = len(indexed)
            result.indexed_elements = len(indexed)
            result.skipped_elements = result.total_elements - result.indexed_elements

            if self.config.verbose:
                logger.info(
                    f"Файл {file_path.name}: проиндексировано {result.indexed_elements}/{result.total_elements}"
                )

        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {e}")
            result.failed_files += 1
            result.errors.append({
                "file": str(file_path),
                "error": str(e)
            })

            if not self.config.skip_errors:
                raise

        return result

    def process_repository(
            self,
            repository_path: Path,
            repository_name: str,
            branch: str = "main",
            commit_hash: Optional[str] = None,
            provider: str = "local",
            exclude_dirs: Optional[List[str]] = None,
            progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> VectorizationResult:
        """
        Обработка всего репозитория.

        Args:
            repository_path: Путь к репозиторию
            repository_name: Имя репозитория
            branch: Ветка
            commit_hash: Хэш коммита
            provider: Провайдер (github, gitlab, local)
            exclude_dirs: Директории для исключения
            progress_callback: Callback для прогресса

        Returns:
            VectorizationResult
        """
        exclude_dirs = exclude_dirs or [
            ".git", ".svn", "node_modules", "venv", ".venv",
            "__pycache__", "build", "dist", "target"
        ]

        repository_info = {
            "repository_name": repository_name,
            "branch": branch,
            "commit_hash": commit_hash,
            "provider": provider
        }

        logger.info(f"Обработка репозитория: {repository_name} ({branch})")

        # Собираем все файлы, исключая ненужные директории
        all_files = []
        for file_path in repository_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Проверяем что файл не в исключенных директориях
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue

            if ParserFactory.can_parse_file(file_path):
                all_files.append(file_path)

        result = VectorizationResult()
        result.total_files = len(all_files)

        logger.info(f"Найдено {result.total_files} файлов в репозитории")

        # Обрабатываем файлы
        for idx, file_path in enumerate(all_files, 1):
            if progress_callback:
                rel_path = file_path.relative_to(repository_path)
                progress_callback(f"Обработка {rel_path}", idx, result.total_files)

            try:
                file_result = self.process_file(file_path, repository_info)

                result.parsed_files += file_result.parsed_files
                result.failed_files += file_result.failed_files
                result.total_elements += file_result.total_elements
                result.vectorized_elements += file_result.vectorized_elements
                result.indexed_elements += file_result.indexed_elements
                result.skipped_elements += file_result.skipped_elements
                result.errors.extend(file_result.errors)

            except Exception as e:
                result.failed_files += 1
                error_msg = f"Ошибка обработки {file_path}: {e}"
                logger.error(error_msg)
                result.errors.append({
                    "file": str(file_path),
                    "error": str(e)
                })

                if not self.config.skip_errors:
                    raise

        logger.info(f"Репозиторий обработан: {result}")

        return result

    def _filter_elements(self, elements: List[CodeElement]) -> List[CodeElement]:
        """Фильтрация элементов по конфигурации."""
        filtered = []

        for element in elements:
            # Фильтр по количеству строк
            if element.location.line_count < self.config.min_lines:
                continue
            if element.location.line_count > self.config.max_lines:
                continue

            # Фильтр по типу
            if self.config.include_types:
                if element.type.value not in self.config.include_types:
                    continue

            if self.config.exclude_types:
                if element.type.value in self.config.exclude_types:
                    continue

            filtered.append(element)

        return filtered

    def _vectorize_and_index(
            self,
            elements: List[CodeElement],
            repository_info: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Векторизация и индексация элементов.

        Args:
            elements: Элементы кода
            repository_info: Информация о репозитории

        Returns:
            Список ID проиндексированных элементов
        """
        if not elements:
            return []

        indexed_ids = []

        # Обрабатываем батчами
        for i in range(0, len(elements), self.config.embedding_batch_size):
            batch = elements[i:i + self.config.embedding_batch_size]

            try:
                # Векторизуем батч
                texts = [element.text_representation for element in batch]
                vectors = self.embedder.encode_batch(texts)

                # Создаем точки для Qdrant
                points = []
                for element, vector in zip(batch, vectors):
                    element_id = self._generate_element_id(element, repository_info)
                    payload = CodePayload.from_code_element(element, repository_info)

                    point = PointData(
                        id=element_id,
                        vector=vector.tolist(),
                        payload=payload
                    )
                    points.append(point)

                # Индексируем в Qdrant
                batch_data = BatchInsertData(points=points)
                inserted, total = self.qdrant_client.insert_batch(
                    self.collection_name,
                    batch_data,
                    batch_size=self.config.batch_size
                )

                indexed_ids.extend([p.id for p in points[:inserted]])

            except Exception as e:
                logger.error(f"Ошибка векторизации батча: {e}")
                if not self.config.skip_errors:
                    raise

        return indexed_ids

    def _generate_element_id(
            self,
            element: CodeElement,
            repository_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Генерация уникального ID для элемента.

        Args:
            element: Элемент кода
            repository_info: Информация о репозитории

        Returns:
            Уникальный ID
        """
        parts = []

        if repository_info:
            repo_name = repository_info.get("repository_name")
            if repo_name:
                parts.append(repo_name)

        # Путь к файлу (относительный)
        file_path = str(element.location.file_path)
        parts.append(file_path)

        # Квалифицированное имя элемента
        parts.append(element.qualified_name)

        # Строка начала (для уникальности)
        parts.append(str(element.location.start_line))

        # Объединяем части
        id_string = "::".join(parts)

        # Генерируем UUID на основе строки (детерминированно)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики коллекции."""
        return self.qdrant_client.get_collection_info(self.collection_name)


def create_vectorization_pipeline(
        collection_name: str,
        embedder: CodeEmbedder,
        qdrant_client: QdrantClient,
        **config_kwargs
) -> VectorizationPipeline:
    """
    Создание pipeline векторизации.

    Args:
        collection_name: Имя коллекции
        embedder: Embedder
        qdrant_client: Клиент Qdrant
        **config_kwargs: Параметры конфигурации

    Returns:
        VectorizationPipeline
    """
    config = VectorizationConfig(**config_kwargs)
    return VectorizationPipeline(
        collection_name=collection_name,
        embedder=embedder,
        qdrant_client=qdrant_client,
        config=config
    )
