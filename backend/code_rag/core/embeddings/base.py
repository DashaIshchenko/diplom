"""
Абстрактный базовый класс для моделей эмбеддингов.
Этот класс определяет интерфейс, который должны реализовать все модели эмбеддингов.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Конфигурация модели эмбеддингов."""
    base_url: str = "https://ai.parma.ru/local/"
    dimension: int = 768
    max_tokens: int = 8192
    model_path: str = "nomic-embed-text:latest"
    device: str = "cpu"  # "cpu" или "cuda"
    batch_size: int = 32
    normalize: bool = True
    additional_params: Optional[Dict[str, Any]] = None
    api_key: str = None
    model_name: str = "nomic-embed-text:latest"
    timeout: int = 30,
    request_timeout: int = 30
    retry_attempts: int = 3
    backoff_factor: int = 1


class BaseEmbeddingModel(ABC):
    """
    Абстрактный базовый класс для всех моделей эмбеддингов.

    Этот класс определяет общий интерфейс для генерации эмбеддингов
    из текста и кода. Все конкретные реализации (nomic-embed-text,
    Qwen3-Embedding и т.д.) должны наследовать этот класс.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Инициализация модели эмбеддингов.

        Args:
            config: Конфигурация модели
        """
        self.config = config
        self.model = None
        self._is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """
        Загрузка модели в память.
        Должна быть реализована в подклассах.
        """
        pass

    @abstractmethod
    def embed_text(self, texts: List[str], prefix: Optional[str] = None) -> np.ndarray:
        """
        Генерация эмбеддингов для текстовых описаний.

        Args:
            texts: Список текстов для эмбеддинга
            prefix: Опциональный префикс для модели (например, "search_query:")

        Returns:
            Numpy массив эмбеддингов размерности (len(texts), dimension)
        """
        pass

    @abstractmethod
    def embed_code(self, code_snippets: List[str], prefix: Optional[str] = None) -> np.ndarray:
        """
        Генерация эмбеддингов для фрагментов кода.

        Args:
            code_snippets: Список фрагментов кода для эмбеддинга
            prefix: Опциональный префикс для модели (например, "search_document:")

        Returns:
            Numpy массив эмбеддингов размерности (len(code_snippets), dimension)
        """
        pass

    def embed_batch(
            self,
            items: List[str],
            batch_size: Optional[int] = None,
            prefix: Optional[str] = None,
            is_code: bool = True
    ) -> np.ndarray:
        """
        Генерация эмбеддингов пакетами для оптимизации производительности.

        Args:
            items: Список элементов для эмбеддинга
            batch_size: Размер батча (если None, используется из конфига)
            prefix: Опциональный префикс для модели
            is_code: Флаг, указывающий что элементы - это код (True) или текст (False)

        Returns:
            Numpy массив всех эмбеддингов
        """
        # Обработка пустого списка
        if not items:
            logger.warning("Пустой список для векторизации")
            return np.zeros((0, self.config.dimension), dtype=np.float32)


        if not self._is_loaded:
            self.load_model()

        batch_size = batch_size or self.config.batch_size
        all_embeddings = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            if is_code:
                embeddings = self.embed_code(batch, prefix)
            else:
                embeddings = self.embed_text(batch, prefix)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    @property
    def dimension(self) -> int:
        """Размерность векторов эмбеддингов."""
        return self.config.dimension

    @property
    def max_tokens(self) -> int:
        """Максимальное количество токенов для обработки."""
        return self.config.max_tokens

    @property
    def model_name(self) -> str:
        """Имя модели."""
        return self.config.model_name

    def unload_model(self) -> None:
        """Выгрузка модели из памяти."""
        self.model = None
        self._is_loaded = False

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"model={self.config.model_name}, "
                f"dim={self.config.dimension}, "
                f"max_tokens={self.config.max_tokens})")
