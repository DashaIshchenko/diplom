"""
Вспомогательные утилиты для Code RAG.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import re


# ==================== Хеширование ====================

def calculate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Вычисление хеша файла.

    Args:
        file_path: Путь к файлу
        algorithm: Алгоритм хеширования (md5, sha1, sha256)

    Returns:
        Хеш строка
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def calculate_string_hash(text: str, algorithm: str = "md5") -> str:
    """
    Вычисление хеша строки.

    Args:
        text: Текст для хеширования
        algorithm: Алгоритм хеширования

    Returns:
        Хеш строка
    """
    return hashlib.new(algorithm, text.encode('utf-8')).hexdigest()


# ==================== Работа с файлами ====================

def read_file(file_path: Path, encoding: str = "utf-8") -> str:
    """
    Чтение файла с обработкой ошибок.

    Args:
        file_path: Путь к файлу
        encoding: Кодировка

    Returns:
        Содержимое файла
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Пробуем другие кодировки
        for enc in ["utf-8-sig", "latin-1", "cp1251"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except:
                continue
        raise


def write_file(file_path: Path, content: str, encoding: str = "utf-8"):
    """
    Запись в файл.

    Args:
        file_path: Путь к файлу
        content: Содержимое
        encoding: Кодировка
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)


def get_file_size(file_path: Path) -> int:
    """Размер файла в байтах."""
    return file_path.stat().st_size


def get_file_modified_time(file_path: Path) -> datetime:
    """Время последней модификации файла."""
    return datetime.fromtimestamp(file_path.stat().st_mtime)


# ==================== JSON ====================

def load_json(file_path: Path) -> Dict[str, Any]:
    """Загрузка JSON из файла."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path: Path, data: Dict[str, Any], indent: int = 2):
    """Сохранение в JSON файл."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def json_serializable(obj: Any) -> Any:
    """
    Преобразование объекта в JSON-сериализуемый формат.

    Args:
        obj: Объект для преобразования

    Returns:
        JSON-сериализуемый объект
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        return json_serializable(obj.__dict__)
    else:
        return str(obj)


# ==================== Текст ====================

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Обрезка текста до максимальной длины.

    Args:
        text: Исходный текст
        max_length: Максимальная длина
        suffix: Суффикс для обрезанного текста

    Returns:
        Обрезанный текст
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def normalize_whitespace(text: str) -> str:
    """Нормализация пробельных символов."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_comments(code: str, language: str) -> str:
    """
    Удаление комментариев из кода (простая версия).

    Args:
        code: Исходный код
        language: Язык программирования

    Returns:
        Код без комментариев
    """
    if language.lower() in ["python"]:
        # Удаляем однострочные комментарии
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # Удаляем многострочные строки (docstrings)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

    elif language.lower() in ["javascript", "typescript", "java", "csharp", "kotlin"]:
        # Удаляем однострочные комментарии
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Удаляем многострочные комментарии
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    return code


def extract_identifier(text: str) -> str:
    """
    Извлечение идентификатора из текста.

    Args:
        text: Текст

    Returns:
        Валидный идентификатор
    """
    # Заменяем недопустимые символы на подчеркивание
    identifier = re.sub(r'[^a-zA-Z0-9_]', '_', text)
    # Убираем множественные подчеркивания
    identifier = re.sub(r'_+', '_', identifier)
    # Убираем подчеркивания в начале и конце
    identifier = identifier.strip('_')
    return identifier or "unnamed"


# ==================== Списки ====================

def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Разбиение списка на чанки.

    Args:
        items: Список элементов
        chunk_size: Размер чанка

    Returns:
        Список чанков
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Разворачивание вложенного списка.

    Args:
        nested_list: Вложенный список

    Returns:
        Плоский список
    """
    return [item for sublist in nested_list for item in sublist]


def deduplicate_list(items: List[Any], key: Optional[callable] = None) -> List[Any]:
    """
    Удаление дубликатов из списка с сохранением порядка.

    Args:
        items: Список элементов
        key: Функция для извлечения ключа

    Returns:
        Список без дубликатов
    """
    seen = set()
    result = []

    for item in items:
        item_key = key(item) if key else item
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)

    return result


# ==================== Словари ====================

def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Глубокое слияние словарей.

    Args:
        dict1: Первый словарь
        dict2: Второй словарь (приоритетный)

    Returns:
        Объединенный словарь
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Разворачивание вложенного словаря.

    Args:
        d: Вложенный словарь
        parent_key: Родительский ключ
        sep: Разделитель

    Returns:
        Плоский словарь
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ==================== Время ====================

def format_duration(seconds: float) -> str:
    """
    Форматирование длительности в читаемый вид.

    Args:
        seconds: Количество секунд

    Returns:
        Форматированная строка
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_timestamp() -> str:
    """Получение текущего timestamp в ISO формате."""
    return datetime.utcnow().isoformat()


# ==================== Валидация ====================

def is_valid_identifier(name: str) -> bool:
    """Проверка валидности идентификатора."""
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def is_valid_path(path: Union[str, Path]) -> bool:
    """Проверка валидности пути."""
    try:
        Path(path)
        return True
    except:
        return False


# ==================== Размеры ====================

def format_bytes(size: int) -> str:
    """
    Форматирование размера в байтах.

    Args:
        size: Размер в байтах

    Returns:
        Форматированная строка
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_number(num: int) -> str:
    """
    Форматирование числа с разделителями.

    Args:
        num: Число

    Returns:
        Форматированная строка
    """
    return f"{num:,}".replace(',', ' ')
