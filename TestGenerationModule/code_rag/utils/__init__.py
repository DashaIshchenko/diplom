"""
Утилиты для Code RAG.
"""

from .logger import (
    setup_logger,
    get_logger,
    set_log_level,
    add_file_handler,
    setup_development_logger,
    setup_production_logger,
    setup_testing_logger,
    LoggerContext,
    with_log_level,
    default_logger,
)

from .helpers import (
    # Хеширование
    calculate_file_hash,
    calculate_string_hash,

    # Файлы
    read_file,
    write_file,
    get_file_size,
    get_file_modified_time,

    # JSON
    load_json,
    save_json,
    json_serializable,

    # Текст
    truncate_text,
    normalize_whitespace,
    remove_comments,
    extract_identifier,

    # Списки
    chunk_list,
    flatten_list,
    deduplicate_list,

    # Словари
    deep_merge,
    flatten_dict,

    # Время
    format_duration,
    get_timestamp,

    # Валидация
    is_valid_identifier,
    is_valid_path,

    # Форматирование
    format_bytes,
    format_number,
)

from .config import load_config, save_config, get_config

__all__ = [
    # Logger
    "setup_logger",
    "get_logger",
    "set_log_level",
    "add_file_handler",
    "setup_development_logger",
    "setup_production_logger",
    "setup_testing_logger",
    "LoggerContext",
    "with_log_level",
    "default_logger",

    # Helpers - Хеширование
    "calculate_file_hash",
    "calculate_string_hash",

    # Helpers - Файлы
    "read_file",
    "write_file",
    "get_file_size",
    "get_file_modified_time",

    # Helpers - JSON
    "load_json",
    "save_json",
    "json_serializable",

    # Helpers - Текст
    "truncate_text",
    "normalize_whitespace",
    "remove_comments",
    "extract_identifier",

    # Helpers - Списки
    "chunk_list",
    "flatten_list",
    "deduplicate_list",

    # Helpers - Словари
    "deep_merge",
    "flatten_dict",

    # Helpers - Время
    "format_duration",
    "get_timestamp",

    # Helpers - Валидация
    "is_valid_identifier",
    "is_valid_path",

    # Helpers - Форматирование
    "format_bytes",
    "format_number",

     # Config
    "load_config",
    "save_config",
    "get_config",
]
