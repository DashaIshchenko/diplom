"""
Модуль конфигурации для Code RAG.

Экспортирует настройки и конфигурационные классы.
"""

from .settings import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
]
