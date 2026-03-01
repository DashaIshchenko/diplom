"""
Утилиты для работы с конфигурацией.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML или JSON файла.

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        Словарь с конфигурацией

    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если формат файла не поддерживается
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Определяем формат по расширению
    suffix = config_path.suffix.lower()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {suffix}")

        # Подстановка переменных окружения
        config = _substitute_env_vars(config)

        logger.info(f"Config loaded from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str | Path) -> None:
    """
    Сохранение конфигурации в YAML или JSON файл.

    Args:
        config: Словарь с конфигурацией
        config_path: Путь для сохранения
    """
    config_path = Path(config_path)

    # Создаём директорию если не существует
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Определяем формат по расширению
    suffix = config_path.suffix.lower()

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif suffix == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config format: {suffix}")

        logger.info(f"Config saved to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        raise


def get_config(
        config_path: Optional[str | Path] = None,
        default_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Получение конфигурации с fallback на значения по умолчанию.

    Args:
        config_path: Путь к файлу конфигурации (опционально)
        default_config: Конфигурация по умолчанию

    Returns:
        Словарь с конфигурацией
    """
    # Конфигурация по умолчанию
    if default_config is None:
        default_config = _get_default_config()

    # Если путь не указан, ищем в стандартных местах
    if config_path is None:
        config_path = _find_config_file()

    # Если файл найден, загружаем и объединяем с default
    if config_path and Path(config_path).exists():
        try:
            user_config = load_config(config_path)
            # Рекурсивное объединение конфигов
            return _deep_merge(default_config, user_config)
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            return default_config

    logger.info("Using default configuration")
    return default_config


def _substitute_env_vars(config: Any) -> Any:
    """
    Рекурсивная подстановка переменных окружения в конфигурации.

    Поддерживает синтаксис: ${ENV_VAR} или ${ENV_VAR:default_value}
    """
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}

    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]

    elif isinstance(config, str):
        # Ищем паттерн ${VAR} или ${VAR:default}
        import re
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_value)

        return re.sub(pattern, replace_var, config)

    else:
        return config


def _find_config_file() -> Optional[Path]:
    """
    Поиск конфигурационного файла в стандартных местах.

    Порядок поиска:
    1. ./config.yaml
    2. ./config.yml
    3. ./config.json
    4. ~/.code_rag/config.yaml
    5. /etc/code_rag/config.yaml
    """
    search_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path.cwd() / "config.json",
        Path.home() / ".code_rag" / "config.yaml",
        Path("/etc/code_rag/config.yaml"),
    ]

    for path in search_paths:
        if path.exists():
            logger.debug(f"Found config at {path}")
            return path

    return None


def _get_default_config() -> Dict[str, Any]:
    """Конфигурация по умолчанию."""
    return {
        "embeddings": {
            "model_name": "nomic-ai/nomic-embed-text-v1.5",
            "device": "cpu",
            "max_length": 8192,
            "batch_size": 32,
        },
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": None,
            "collection_name": "code-rag",
            "vector_size": 768,
            "distance": "Cosine",
        },
        "rag": {
            "top_k": 10,
            "score_threshold": 0.6,
            "use_reranking": True,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_context_length": 8000,
        },
        "qwen": {
            "api_key": None,
            "model": "qwen2.5-coder-32b-instruct",
            "temperature": 0.7,
            "max_tokens": 2000,
        },
        "parser": {
            "languages": ["python", "javascript", "typescript", "java", "csharp", "kotlin"],
            "extract_complexity": True,
            "extract_imports": True,
        },
        "git": {
            "monitor_interval": 60,
            "auto_reindex": True,
            "exclude_patterns": [
                "*.test.py",
                "*.spec.js",
                "node_modules",
                "__pycache__",
                ".git",
            ],
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None,
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Глубокое объединение двух словарей.

    Args:
        base: Базовый словарь
        override: Словарь с переопределениями

    Returns:
        Объединённый словарь
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result
