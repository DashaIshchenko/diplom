"""
Утилиты для парсеров.
"""

from .tree_sitter_helper import (
    TreeSitterHelper,
    get_tree_sitter_helper,
    parse_code,
    is_tree_sitter_available,
    get_supported_languages as get_tree_sitter_languages,
)

from .docstring_extractor import (
    extract_docstring,
    extract_python_docstring,
    extract_javadoc,
    extract_jsdoc,
    extract_xmldoc,
    parse_javadoc,
    parse_jsdoc,
    parse_xmldoc,
    format_docstring_for_embedding,
    extract_code_examples_from_docstring,
)

from .complexity_calculator import (
    calculate_complexity,
    get_complexity_rating,
    get_complexity_recommendation,
    calculate_maintainability_index,
)

__all__ = [
    # Tree-sitter helper
    "TreeSitterHelper",
    "get_tree_sitter_helper",
    "parse_code",
    "is_tree_sitter_available",
    "get_tree_sitter_languages",

    # Docstring extraction
    "extract_docstring",
    "extract_python_docstring",
    "extract_javadoc",
    "extract_jsdoc",
    "extract_xmldoc",
    "parse_javadoc",
    "parse_jsdoc",
    "parse_xmldoc",
    "format_docstring_for_embedding",
    "extract_code_examples_from_docstring",

    # Complexity calculation
    "calculate_complexity",
    "get_complexity_rating",
    "get_complexity_recommendation",
    "calculate_maintainability_index",
]
