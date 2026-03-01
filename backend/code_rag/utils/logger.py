"""
Настройка логирования для Code RAG.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


# Цвета для консольного вывода
class ColoredFormatter(logging.Formatter):
    """Форматтер с цветами для консоли."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
        name: str = "code_rag",
        level: str = "INFO",
        log_file: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5,
        console_output: bool = True,
        colored: bool = True
) -> logging.Logger:
    """
    Настройка логгера.

    Args:
        name: Имя логгера
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу логов (опционально)
        log_dir: Директория для логов (опционально)
        max_bytes: Максимальный размер файла лога
        backup_count: Количество backup файлов
        console_output: Выводить в консоль
        colored: Использовать цветной вывод в консоли

    Returns:
        Настроенный logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Удаляем существующие handlers
    logger.handlers.clear()

    # Формат логов
    detailed_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    simple_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if colored and sys.stdout.isatty():
            console_formatter = ColoredFormatter(simple_format, datefmt=date_format)
        else:
            console_formatter = logging.Formatter(simple_format, datefmt=date_format)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file or log_dir:
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Файл записывает все
        file_formatter = logging.Formatter(detailed_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "code_rag") -> logging.Logger:
    """
    Получение логгера по имени.

    Args:
        name: Имя логгера

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: str):
    """
    Изменение уровня логирования.

    Args:
        logger: Logger instance
        level: Новый уровень (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.setLevel(getattr(logging, level.upper()))
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(getattr(logging, level.upper()))


def add_file_handler(
        logger: logging.Logger,
        log_file: Path,
        level: str = "DEBUG",
        max_bytes: int = 10485760,
        backup_count: int = 5
):
    """
    Добавление файлового handler к логгеру.

    Args:
        logger: Logger instance
        log_file: Путь к файлу
        level: Уровень логирования
        max_bytes: Максимальный размер файла
        backup_count: Количество backup файлов
    """
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# Предустановленные конфигурации
def setup_development_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    """Логгер для разработки (DEBUG, цветной вывод)."""
    return setup_logger(
        name="code_rag",
        level="DEBUG",
        log_dir=log_dir,
        console_output=True,
        colored=True
    )


def setup_production_logger(log_dir: Path) -> logging.Logger:
    """Логгер для продакшена (INFO, файл + консоль)."""
    return setup_logger(
        name="code_rag",
        level="INFO",
        log_dir=log_dir,
        console_output=True,
        colored=False
    )


def setup_testing_logger() -> logging.Logger:
    """Логгер для тестов (WARNING, только консоль)."""
    return setup_logger(
        name="code_rag",
        level="WARNING",
        console_output=True,
        colored=False
    )


# Утилиты для логирования
class LoggerContext:
    """Контекстный менеджер для временного изменения уровня логирования."""

    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level

    def __enter__(self):
        set_log_level(self.logger, self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def with_log_level(level: str):
    """
    Декоратор для временного изменения уровня логирования.

    Args:
        level: Временный уровень логирования
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            old_level = logger.level
            try:
                set_log_level(logger, level)
                return func(*args, **kwargs)
            finally:
                logger.setLevel(old_level)

        return wrapper

    return decorator


# Инициализация логгера по умолчанию
default_logger = setup_logger(
    name="code_rag",
    level="INFO",
    console_output=True,
    colored=True
)
