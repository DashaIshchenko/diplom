"""
Модуль работы с git репозиториями.

Обеспечивает:
- Клонирование и обновление репозиториев
- Мониторинг изменений
- Отслеживание коммитов
"""

from .repository import RepositoryManager, RepositoryInfo, AzureDevOpsClient,RepositoryProvider
from .monitor import (
    RepositoryMonitor,
    CommitInfo,
    ChangeEvent,
    ChangeType,
)




__all__ = [
    # Основные классы
    "RepositoryManager",
    "RepositoryInfo",
    "AzureDevOpsClient",
    "RepositoryProvider",
    "RepositoryMonitor",
    "CommitInfo",
    "ChangeEvent",
    "ChangeType",
]
