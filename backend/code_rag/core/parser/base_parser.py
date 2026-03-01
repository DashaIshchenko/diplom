"""
Базовый класс для парсеров разных языков.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from pathlib import Path
import logging

from .code_structure import ModuleInfo, ParseResult, ProgrammingLanguage

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Абстрактный базовый класс для всех парсеров языков."""

    def __init__(self, max_chunk_size: int = 8192):
        """
        Args:
            max_chunk_size: Максимальный размер чанка
        """
        self.max_chunk_size = max_chunk_size

    @property
    @abstractmethod
    def language(self) -> ProgrammingLanguage:
        """Язык программирования парсера."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Поддерживаемые расширения файлов."""
        pass

    @abstractmethod
    def parse_file(
            self,
            file_path: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None
    ) -> Optional[ModuleInfo]:
        """
        Парсинг одного файла.

        Args:
            file_path: Путь к файлу
            repository_name: Имя репозитория
            branch: Ветка
            commit_hash: Хэш коммита
            provider: Провайдер (github, gitlab, azure_devops)

        Returns:
            ModuleInfo или None при ошибке
        """
        pass

    def parse_directory(
            self,
            directory: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None,
            recursive: bool = True
    ) -> ParseResult:
        """
        Парсинг директории.

        Args:
            directory: Путь к директории
            repository_name: Имя репозитория
            branch: Ветка
            commit_hash: Хэш коммита
            provider: Провайдер
            recursive: Рекурсивный поиск

        Returns:
            ParseResult с результатами
        """
        result = ParseResult()

        # Поиск файлов с подходящими расширениями
        all_files = []
        for ext in self.file_extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            all_files.extend(directory.glob(pattern))

        logger.info(
            f"Найдено {len(all_files)} {self.language.value} файлов в {directory}"
        )

        for file_path in all_files:
            if self._should_skip_file(file_path):
                continue

            module_info = self.parse_file(
                file_path,
                repository_name=repository_name,
                branch=branch,
                commit_hash=commit_hash,
                provider=provider
            )

            if module_info:
                result.modules.append(module_info)
            else:
                result.errors.append({
                    "file": str(file_path),
                    "language": self.language.value,
                    "error": "Failed to parse"
                })

        logger.info(
            f"Парсинг {self.language.value} завершен: "
            f"{len(result.modules)} модулей, {result.total_elements} элементов"
        )

        return result

    def _should_skip_file(self, file_path: Path) -> bool:
        """Проверка, нужно ли пропустить файл."""
        skip_patterns = [
            '__pycache__', 'node_modules', 'venv', '.venv',
            'build', 'dist', 'target', 'bin', 'obj',
            '.git', '.svn', 'vendor', 'packages',
            '.idea', '.vs', '.vscode'
        ]

        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)

    def _get_module_name(self, file_path: Path) -> str:
        """Получение имени модуля из пути."""
        return file_path.stem

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(language={self.language.value})"
