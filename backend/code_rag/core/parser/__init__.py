"""
Модуль парсинга исходного кода (мультиязычный).

Поддерживаемые языки:
- Python (AST)
- Java (tree-sitter)
- JavaScript (tree-sitter)
- TypeScript (tree-sitter)
- C# (tree-sitter)
- Kotlin (tree-sitter)
- HTML (tree-sitter)
- CSS (tree-sitter)
"""

from .code_structure import (
    CodeElement,
    CodeElementType,
    CodeLocation,
    ModuleInfo,
    ParseResult,
    ProgrammingLanguage,
)
from .base_parser import BaseParser
from .factory import ParserFactory

# Парсеры для разных языков
from .parsers.python_parser import PythonParser
from .parsers.java_parser import JavaParser
from .parsers.javascript_parser import JavaScriptParser
from .parsers.typescript_parser import TypeScriptParser
from .parsers.csharp_parser import CSharpParser
from .parsers.kotlin_parser import KotlinParser
from .parsers.html_parser import HTMLParser
from .parsers.css_parser import CSSParser

__all__ = [
    # Структуры данных
    "CodeElement",
    "CodeElementType",
    "CodeLocation",
    "ModuleInfo",
    "ParseResult",
    "ProgrammingLanguage",

    # Базовый класс
    "BaseParser",

    # Factory
    "ParserFactory",

    # Парсеры
    "PythonParser",
    "JavaParser",
    "JavaScriptParser",
    "TypeScriptParser",
    "CSharpParser",
    "KotlinParser",
    "HTMLParser",
    "CSSParser",
]

# Версия модуля
__version__ = "0.1.0"

# Информация о поддерживаемых языках
SUPPORTED_LANGUAGES = {
    "python": {
        "parser": PythonParser,
        "extensions": [".py", ".pyi"],
        "method": "AST",
        "description": "Python 3.x support with AST parsing"
    },
    "java": {
        "parser": JavaParser,
        "extensions": [".java"],
        "method": "tree-sitter",
        "description": "Java support with classes, interfaces, and annotations"
    },
    "javascript": {
        "parser": JavaScriptParser,
        "extensions": [".js", ".jsx", ".mjs", ".cjs"],
        "method": "tree-sitter",
        "description": "JavaScript/JSX support with ES6+ syntax"
    },
    "typescript": {
        "parser": TypeScriptParser,
        "extensions": [".ts", ".tsx", ".mts", ".cts"],
        "method": "tree-sitter",
        "description": "TypeScript/TSX support with generics and decorators"
    },
    "csharp": {
        "parser": CSharpParser,
        "extensions": [".cs"],
        "method": "tree-sitter",
        "description": "C# support with records, properties, and async/await"
    },
    "kotlin": {
        "parser": KotlinParser,
        "extensions": [".kt", ".kts"],
        "method": "tree-sitter",
        "description": "Kotlin support with data classes and coroutines"
    },
    "html": {
        "parser": HTMLParser,
        "extensions": [".html", ".htm", ".xhtml", ".vue", ".svelte"],
        "method": "tree-sitter",
        "description": "HTML support with semantic tags and accessibility"
    },
    "css": {
        "parser": CSSParser,
        "extensions": [".css", ".scss", ".sass", ".less"],
        "method": "tree-sitter",
        "description": "CSS support with selectors, media queries, and keyframes"
    }
}


def get_parser_info(language: str = None) -> dict:
    """
    Получение информации о парсерах.

    Args:
        language: Имя языка (опционально). Если None, возвращает все языки.

    Returns:
        Словарь с информацией о парсере(ах)

    Examples:
        >>> info = get_parser_info("python")
        >>> print(info["extensions"])
        ['.py', '.pyi']

        >>> all_info = get_parser_info()
        >>> print(len(all_info))
        8
    """
    if language:
        return SUPPORTED_LANGUAGES.get(language.lower())
    return SUPPORTED_LANGUAGES


def list_supported_languages() -> list:
    """
    Получение списка поддерживаемых языков.

    Returns:
        Список имен языков

    Examples:
        >>> languages = list_supported_languages()
        >>> print(languages)
        ['python', 'java', 'javascript', 'typescript', 'csharp', 'kotlin', 'html', 'css']
    """
    return list(SUPPORTED_LANGUAGES.keys())


def list_supported_extensions() -> list:
    """
    Получение списка всех поддерживаемых расширений файлов.

    Returns:
        Список расширений

    Examples:
        >>> extensions = list_supported_extensions()
        >>> print('.py' in extensions)
        True
    """
    extensions = []
    for lang_info in SUPPORTED_LANGUAGES.values():
        extensions.extend(lang_info["extensions"])
    return sorted(set(extensions))


def get_parser_for_extension(extension: str):
    """
    Получение класса парсера по расширению файла.

    Args:
        extension: Расширение файла (с точкой)

    Returns:
        Класс парсера или None

    Examples:
        >>> parser_class = get_parser_for_extension(".py")
        >>> print(parser_class.__name__)
        'PythonParser'
    """
    extension = extension.lower()
    for lang_info in SUPPORTED_LANGUAGES.values():
        if extension in lang_info["extensions"]:
            return lang_info["parser"]
    return None


def print_supported_languages():
    """
    Вывод информации о всех поддерживаемых языках в консоль.

    Examples:
        >>> print_supported_languages()
        Supported Languages:
        ====================

        Python (AST)
          Extensions: .py, .pyi
          Description: Python 3.x support with AST parsing
        ...
    """
    print("Supported Languages:")
    print("=" * 50)
    print()

    for lang_name, lang_info in SUPPORTED_LANGUAGES.items():
        print(f"{lang_name.upper()} ({lang_info['method']})")
        print(f"  Extensions: {', '.join(lang_info['extensions'])}")
        print(f"  Description: {lang_info['description']}")
        print(f"  Parser: {lang_info['parser'].__name__}")
        print()


# Примеры использования
if __name__ == "__main__":
    print("=" * 70)
    print("CODE RAG PARSER MODULE")
    print("=" * 70)
    print()

    print_supported_languages()

    print("=" * 70)
    print("QUICK EXAMPLES")
    print("=" * 70)
    print()

    print("1. Using ParserFactory:")
    print("   >>> from code_rag.core.parser import ParserFactory")
    print("   >>> parser = ParserFactory.create_parser_for_file(Path('file.py'))")
    print("   >>> module = parser.parse_file(Path('file.py'))")
    print()

    print("2. Using specific parser:")
    print("   >>> from code_rag.core.parser import PythonParser")
    print("   >>> parser = PythonParser()")
    print("   >>> module = parser.parse_file(Path('file.py'))")
    print()

    print("3. Getting parser info:")
    print("   >>> from code_rag.core.parser import get_parser_info")
    print("   >>> info = get_parser_info('python')")
    print("   >>> print(info['extensions'])")
    print()

    print("4. Parse directory:")
    print("   >>> from code_rag.core.parser import ParserFactory")
    print("   >>> parser = ParserFactory.create_parser(ProgrammingLanguage.PYTHON)")
    print("   >>> result = parser.parse_directory(Path('/path/to/code'))")
    print("   >>> print(result.get_statistics())")
    print()

    print("=" * 70)
    print(f"Total supported languages: {len(SUPPORTED_LANGUAGES)}")
    print(f"Total supported extensions: {len(list_supported_extensions())}")
    print(f"Version: {__version__}")
    print("=" * 70)
