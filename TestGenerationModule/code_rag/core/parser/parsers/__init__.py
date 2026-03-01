"""
Парсеры для различных языков программирования.

Поддерживаемые языки:
- Python (AST)
- Java (tree-sitter)
- JavaScript/JSX (tree-sitter)
- TypeScript/TSX (tree-sitter)
- C# (tree-sitter)
- Kotlin (tree-sitter)
- HTML (tree-sitter)
- CSS (tree-sitter)
"""

from .python_parser import PythonParser
from .java_parser import JavaParser
from .javascript_parser import JavaScriptParser
from .typescript_parser import TypeScriptParser
from .csharp_parser import CSharpParser
from .kotlin_parser import KotlinParser
from .html_parser import HTMLParser
from .css_parser import CSSParser

__all__ = [
    "PythonParser",
    "JavaParser",
    "JavaScriptParser",
    "TypeScriptParser",
    "CSharpParser",
    "KotlinParser",
    "HTMLParser",
    "CSSParser",
]
