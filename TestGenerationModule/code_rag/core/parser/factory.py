"""
Factory для создания парсеров языков программирования.
"""

from typing import Optional, Dict, Type, List
from pathlib import Path
import logging

from .base_parser import BaseParser
from .code_structure import ProgrammingLanguage
from .parsers.python_parser import PythonParser
from .parsers.java_parser import JavaParser
from .parsers.javascript_parser import JavaScriptParser
from .parsers.typescript_parser import TypeScriptParser
from .parsers.csharp_parser import CSharpParser
from .parsers.kotlin_parser import KotlinParser
from .parsers.html_parser import HTMLParser
from .parsers.css_parser import CSSParser

logger = logging.getLogger(__name__)


class ParserFactory:
    """Factory для создания парсеров разных языков."""

    # Реестр парсеров
    _parsers: Dict[ProgrammingLanguage, Type[BaseParser]] = {
        ProgrammingLanguage.PYTHON: PythonParser,
        ProgrammingLanguage.JAVA: JavaParser,
        ProgrammingLanguage.JAVASCRIPT: JavaScriptParser,
        ProgrammingLanguage.TYPESCRIPT: TypeScriptParser,
        ProgrammingLanguage.CSHARP: CSharpParser,
        ProgrammingLanguage.KOTLIN: KotlinParser,
        ProgrammingLanguage.HTML: HTMLParser,
        ProgrammingLanguage.CSS: CSSParser,
    }

    # Маппинг расширений на языки
    _extension_map: Dict[str, ProgrammingLanguage] = {
        # Python
        '.py': ProgrammingLanguage.PYTHON,
        '.pyi': ProgrammingLanguage.PYTHON,

        # Java
        '.java': ProgrammingLanguage.JAVA,

        # JavaScript
        '.js': ProgrammingLanguage.JAVASCRIPT,
        '.jsx': ProgrammingLanguage.JAVASCRIPT,
        '.mjs': ProgrammingLanguage.JAVASCRIPT,
        '.cjs': ProgrammingLanguage.JAVASCRIPT,

        # TypeScript
        '.ts': ProgrammingLanguage.TYPESCRIPT,
        '.tsx': ProgrammingLanguage.TYPESCRIPT,
        '.mts': ProgrammingLanguage.TYPESCRIPT,
        '.cts': ProgrammingLanguage.TYPESCRIPT,

        # C#
        '.cs': ProgrammingLanguage.CSHARP,

        # Kotlin
        '.kt': ProgrammingLanguage.KOTLIN,
        '.kts': ProgrammingLanguage.KOTLIN,

        # HTML
        '.html': ProgrammingLanguage.HTML,
        '.htm': ProgrammingLanguage.HTML,
        '.xhtml': ProgrammingLanguage.HTML,
        '.vue': ProgrammingLanguage.HTML,  # Vue components
        '.svelte': ProgrammingLanguage.HTML,  # Svelte components

        # CSS
        '.css': ProgrammingLanguage.CSS,
        '.scss': ProgrammingLanguage.CSS,
        '.sass': ProgrammingLanguage.CSS,
        '.less': ProgrammingLanguage.CSS,
    }

    @classmethod
    def create_parser(
            cls,
            language: ProgrammingLanguage,
            max_chunk_size: int = 8192
    ) -> BaseParser:
        """
        Создание парсера для конкретного языка.

        Args:
            language: Язык программирования
            max_chunk_size: Максимальный размер чанка

        Returns:
            Экземпляр парсера

        Raises:
            ValueError: Если язык не поддерживается
        """
        if language not in cls._parsers:
            available = ", ".join([lang.value for lang in cls._parsers.keys()])
            raise ValueError(
                f"Язык {language.value} не поддерживается. "
                f"Доступные: {available}"
            )

        parser_class = cls._parsers[language]
        return parser_class(max_chunk_size=max_chunk_size)

    @classmethod
    def create_parser_for_file(
            cls,
            file_path: Path,
            max_chunk_size: int = 8192
    ) -> Optional[BaseParser]:
        """
        Создание парсера на основе расширения файла.

        Args:
            file_path: Путь к файлу
            max_chunk_size: Максимальный размер чанка

        Returns:
            Парсер или None если расширение не поддерживается
        """
        extension = file_path.suffix.lower()

        if extension not in cls._extension_map:
            logger.warning(f"Расширение {extension} не поддерживается")
            return None

        language = cls._extension_map[extension]
        return cls.create_parser(language, max_chunk_size)

    @classmethod
    def get_supported_languages(cls) -> List[ProgrammingLanguage]:
        """Получение списка поддерживаемых языков."""
        return list(cls._parsers.keys())

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Получение списка поддерживаемых расширений."""
        return list(cls._extension_map.keys())

    @classmethod
    def detect_language(cls, file_path: Path) -> Optional[ProgrammingLanguage]:
        """
        Определение языка программирования по файлу.

        Args:
            file_path: Путь к файлу

        Returns:
            ProgrammingLanguage или None
        """
        extension = file_path.suffix.lower()
        return cls._extension_map.get(extension)

    @classmethod
    def is_programming_language(cls, language: ProgrammingLanguage) -> bool:
        """
        Проверка, является ли язык программирования (не разметки).

        Args:
            language: Язык

        Returns:
            True если это язык программирования
        """
        markup_languages = {
            ProgrammingLanguage.HTML,
            ProgrammingLanguage.CSS,
        }
        return language not in markup_languages

    @classmethod
    def is_markup_language(cls, language: ProgrammingLanguage) -> bool:
        """
        Проверка, является ли язык разметки.

        Args:
            language: Язык

        Returns:
            True если это язык разметки/стилей
        """
        return language in {ProgrammingLanguage.HTML, ProgrammingLanguage.CSS}

    @classmethod
    def register_parser(
            cls,
            language: ProgrammingLanguage,
            parser_class: Type[BaseParser],
            extensions: List[str]
    ) -> None:
        """
        Регистрация кастомного парсера.

        Args:
            language: Язык программирования
            parser_class: Класс парсера
            extensions: Расширения файлов
        """
        cls._parsers[language] = parser_class

        for ext in extensions:
            cls._extension_map[ext.lower()] = language

        logger.info(f"Зарегистрирован парсер для {language.value}")

    @classmethod
    def get_language_info(cls) -> Dict[str, Dict]:
        """
        Получение информации о всех поддерживаемых языках.

        Returns:
            Словарь с информацией о языках
        """
        info = {}

        for language in cls.get_supported_languages():
            extensions = [
                ext for ext, lang in cls._extension_map.items()
                if lang == language
            ]

            info[language.value] = {
                "language": language.value,
                "extensions": extensions,
                "is_programming": cls.is_programming_language(language),
                "is_markup": cls.is_markup_language(language),
                "parser_class": cls._parsers[language].__name__,
            }

        return info

    @classmethod
    def can_parse_file(cls, file_path: Path) -> bool:
        """
        Проверка возможности парсинга файла.

        Args:
            file_path: Путь к файлу

        Returns:
            True если файл может быть распарсен
        """
        if not file_path.exists():
            return False

        if not file_path.is_file():
            return False

        extension = file_path.suffix.lower()
        return extension in cls._extension_map

    @classmethod
    def get_language_from_extension(cls, extension: str) -> Optional[ProgrammingLanguage]:
        """
        Определение языка программирования по расширению файла.

        Args:
            extension: Расширение файла (например, '.py', '.js')

        Returns:
            ProgrammingLanguage или None
        """
        return cls._extension_map.get(extension.lower())

    @classmethod
    def list_supported_languages(cls) -> List[ProgrammingLanguage]:
        """
        Получение списка поддерживаемых языков (алиас для get_supported_languages).

        Returns:
            Список ProgrammingLanguage
        """
        return cls.get_supported_languages()

    @classmethod
    def get_extensions_for_language(cls, language: ProgrammingLanguage) -> List[str]:
        """
        Получение всех расширений для конкретного языка.

        Args:
            language: Язык программирования

        Returns:
            Список расширений файлов
        """
        return [
            ext for ext, lang in cls._extension_map.items()
            if lang == language
        ]

    @classmethod
    def is_extension_supported(cls, extension: str) -> bool:
        """
        Проверка поддержки расширения файла.

        Args:
            extension: Расширение файла

        Returns:
            True если расширение поддерживается
        """
        return extension.lower() in cls._extension_map

    @classmethod
    def is_language_supported(cls, language: ProgrammingLanguage) -> bool:
        """
        Проверка поддержки языка.

        Args:
            language: Язык программирования

        Returns:
            True если язык поддерживается
        """
        return language in cls._parsers

    @classmethod
    def get_parser_class(cls, language: ProgrammingLanguage) -> Optional[Type[BaseParser]]:
        """
        Получение класса парсера для языка.

        Args:
            language: Язык программирования

        Returns:
            Класс парсера или None
        """
        return cls._parsers.get(language)

    @classmethod
    def batch_create_parsers(
            cls,
            languages: List[ProgrammingLanguage],
            max_chunk_size: int = 8192
    ) -> Dict[ProgrammingLanguage, BaseParser]:
        """
        Создание нескольких парсеров одновременно.

        Args:
            languages: Список языков
            max_chunk_size: Максимальный размер чанка

        Returns:
            Словарь {язык: парсер}
        """
        parsers = {}

        for language in languages:
            try:
                parsers[language] = cls.create_parser(language, max_chunk_size)
            except ValueError as e:
                logger.warning(f"Failed to create parser for {language.value}: {e}")

        return parsers

    @classmethod
    def get_parsers_for_directory(
            cls,
            directory: Path,
            recursive: bool = True
    ) -> Dict[ProgrammingLanguage, BaseParser]:
        """
        Создание парсеров для всех поддерживаемых языков в директории.

        Args:
            directory: Путь к директории
            recursive: Рекурсивный поиск файлов

        Returns:
            Словарь с парсерами для найденных языков
        """
        if not directory.exists() or not directory.is_dir():
            return {}

        # Найти все языки в директории
        found_languages = set()

        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                language = cls.detect_language(file_path)
                if language:
                    found_languages.add(language)

        # Создать парсеры для найденных языков
        return cls.batch_create_parsers(list(found_languages))

    @classmethod
    def validate_parser_registration(cls) -> Dict[str, List[str]]:
        """
        Валидация всех зарегистрированных парсеров.

        Returns:
            Словарь с ошибками валидации
        """
        errors = {}

        for language, parser_class in cls._parsers.items():
            language_errors = []

            # Проверка что класс наследуется от BaseParser
            if not issubclass(parser_class, BaseParser):
                language_errors.append(f"{parser_class.__name__} не наследуется от BaseParser")

            # Проверка что есть хотя бы одно расширение
            extensions = cls.get_extensions_for_language(language)
            if not extensions:
                language_errors.append(f"Нет зарегистрированных расширений для {language.value}")

            if language_errors:
                errors[language.value] = language_errors

        return errors

    @classmethod
    def unregister_parser(cls, language: ProgrammingLanguage) -> bool:
        """
        Удаление парсера из реестра.

        Args:
            language: Язык программирования

        Returns:
            True если парсер был удален
        """
        if language not in cls._parsers:
            return False

        # Удаляем парсер
        del cls._parsers[language]

        # Удаляем связанные расширения
        extensions_to_remove = [
            ext for ext, lang in cls._extension_map.items()
            if lang == language
        ]

        for ext in extensions_to_remove:
            del cls._extension_map[ext]

        logger.info(f"Удален парсер для {language.value}")
        return True

    @classmethod
    def reset_parsers(cls) -> None:
        """
        Сброс всех парсеров к начальному состоянию.
        """
        # Сохраняем оригинальные значения
        original_parsers = {
            ProgrammingLanguage.PYTHON: PythonParser,
            ProgrammingLanguage.JAVA: JavaParser,
            ProgrammingLanguage.JAVASCRIPT: JavaScriptParser,
            ProgrammingLanguage.TYPESCRIPT: TypeScriptParser,
            ProgrammingLanguage.CSHARP: CSharpParser,
            ProgrammingLanguage.KOTLIN: KotlinParser,
            ProgrammingLanguage.HTML: HTMLParser,
            ProgrammingLanguage.CSS: CSSParser,
        }

        original_extensions = {
            '.py': ProgrammingLanguage.PYTHON,
            '.pyi': ProgrammingLanguage.PYTHON,
            '.java': ProgrammingLanguage.JAVA,
            '.js': ProgrammingLanguage.JAVASCRIPT,
            '.jsx': ProgrammingLanguage.JAVASCRIPT,
            '.mjs': ProgrammingLanguage.JAVASCRIPT,
            '.cjs': ProgrammingLanguage.JAVASCRIPT,
            '.ts': ProgrammingLanguage.TYPESCRIPT,
            '.tsx': ProgrammingLanguage.TYPESCRIPT,
            '.mts': ProgrammingLanguage.TYPESCRIPT,
            '.cts': ProgrammingLanguage.TYPESCRIPT,
            '.cs': ProgrammingLanguage.CSHARP,
            '.kt': ProgrammingLanguage.KOTLIN,
            '.kts': ProgrammingLanguage.KOTLIN,
            '.html': ProgrammingLanguage.HTML,
            '.htm': ProgrammingLanguage.HTML,
            '.xhtml': ProgrammingLanguage.HTML,
            '.vue': ProgrammingLanguage.HTML,
            '.svelte': ProgrammingLanguage.HTML,
            '.css': ProgrammingLanguage.CSS,
            '.scss': ProgrammingLanguage.CSS,
            '.sass': ProgrammingLanguage.CSS,
            '.less': ProgrammingLanguage.CSS,
        }

        cls._parsers = original_parsers.copy()
        cls._extension_map = original_extensions.copy()

        logger.info("Парсеры сброшены к начальному состоянию")

    @classmethod
    def get_statistics(cls) -> Dict[str, any]:
        """
        Получение подробной статистики по парсерам.

        Returns:
            Словарь со статистикой
        """
        languages = cls.get_supported_languages()

        programming_languages = [
            lang for lang in languages
            if cls.is_programming_language(lang)
        ]

        markup_languages = [
            lang for lang in languages
            if cls.is_markup_language(lang)
        ]

        extensions_by_type = {}
        for ext, lang in cls._extension_map.items():
            lang_type = "programming" if cls.is_programming_language(lang) else "markup"
            if lang_type not in extensions_by_type:
                extensions_by_type[lang_type] = []
            extensions_by_type[lang_type].append(ext)

        return {
            "total_languages": len(languages),
            "programming_languages": {
                "count": len(programming_languages),
                "list": [lang.value for lang in programming_languages]
            },
            "markup_languages": {
                "count": len(markup_languages),
                "list": [lang.value for lang in markup_languages]
            },
            "total_extensions": len(cls._extension_map),
            "extensions_by_type": extensions_by_type,
            "extensions_per_language": {
                lang.value: len(cls.get_extensions_for_language(lang))
                for lang in languages
            }
        }


# Удобные функции
def parse_file(file_path: str, **kwargs) -> Optional:
    """
    Быстрый парсинг файла с автоопределением языка.

    Args:
        file_path: Путь к файлу
        **kwargs: Дополнительные параметры

    Returns:
        ModuleInfo или None
    """
    path = Path(file_path)
    parser = ParserFactory.create_parser_for_file(path)

    if parser:
        return parser.parse_file(path, **kwargs)

    return None


def parse_directory(directory_path: str, **kwargs):
    """
    Парсинг директории с автоопределением языков.

    Args:
        directory_path: Путь к директории
        **kwargs: Дополнительные параметры

    Returns:
        ParseResult со всеми найденными языками
    """
    from .code_structure import ParseResult

    directory = Path(directory_path)
    combined_result = ParseResult()

    # Парсим каждый поддерживаемый язык
    for language in ParserFactory.get_supported_languages():
        parser = ParserFactory.create_parser(language)
        result = parser.parse_directory(directory, **kwargs)

        combined_result.modules.extend(result.modules)
        combined_result.errors.extend(result.errors)

    return combined_result


def get_parser_statistics():
    """
    Получение статистики по парсерам.

    Returns:
        Словарь со статистикой
    """
    info = ParserFactory.get_language_info()

    programming_langs = [
        lang for lang, data in info.items()
        if data['is_programming']
    ]

    markup_langs = [
        lang for lang, data in info.items()
        if data['is_markup']
    ]

    return {
        "total_languages": len(info),
        "programming_languages": len(programming_langs),
        "markup_languages": len(markup_langs),
        "total_extensions": len(ParserFactory.get_supported_extensions()),
        "languages": info,
    }



# Пример использования
if __name__ == "__main__":
    print("=== Поддерживаемые языки ===")

    stats = get_parser_statistics()
    print(f"Всего языков: {stats['total_languages']}")
    print(f"Языки программирования: {stats['programming_languages']}")
    print(f"Языки разметки: {stats['markup_languages']}")
    print(f"Расширений файлов: {stats['total_extensions']}")

    print("\n=== Детали по языкам ===")
    for lang_name, lang_info in stats['languages'].items():
        print(f"\n{lang_name.upper()}:")
        print(f"  Расширения: {', '.join(lang_info['extensions'])}")
        print(f"  Тип: {'Программирование' if lang_info['is_programming'] else 'Разметка'}")
        print(f"  Парсер: {lang_info['parser_class']}")
