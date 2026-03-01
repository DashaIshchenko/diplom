"""
Test suite for Code RAG.

Содержит тесты для всех компонентов системы:
- Embeddings (векторизация кода)
- Parser (парсинг исходного кода)
- Vector DB (Qdrant интеграция)
- Vectorization Pipeline (индексация кода)
- RAG (поиск и генерация)
- Git Handler (работа с репозиториями)
- Utils (вспомогательные функции)
"""

__version__ = "0.1.0"

# Настройка pytest
def pytest_configure(config):
    """Настройка pytest."""
    # Регистрация кастомных маркеров
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "requires_qdrant: marks tests that require Qdrant server"
    )
    config.addinivalue_line(
        "markers",
        "requires_qwen: marks tests that require Qwen API key"
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: marks tests that require GPU"
    )


# Экспорт для использования в тестах (опционально)
__all__ = [
    "pytest_configure",
]
