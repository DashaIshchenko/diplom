"""
Обертка для использования моделей эмбеддингов в pipeline.
"""

from typing import List, Optional, Union
import numpy as np
import logging

from .base import BaseEmbeddingModel
from .factory import EmbeddingModelFactory, EmbeddingModelType

logger = logging.getLogger(__name__)


class CodeEmbedder:
    """
    Обертка для embedder моделей.

    Предоставляет унифицированный интерфейс для VectorizationPipeline.
    """

    def __init__(
            self,
            model: Optional[BaseEmbeddingModel] = None,
            model_type: EmbeddingModelType = EmbeddingModelType.REMOTE,
            **kwargs
    ):
        """
        Инициализация CodeEmbedder.

        Args:
            model: Готовая модель (опционально)
            model_type: Тип модели для создания
            **kwargs: Параметры для создания модели
        """
        if model:
            self.model = model
        else:
            self.model = EmbeddingModelFactory.create_model(model_type.value, **kwargs)

        logger.info(f"CodeEmbedder инициализирован с моделью: {self.model.__class__.__name__}")

    @property
    def embedding_dim(self) -> int:
        """Размерность эмбеддингов."""
        return self.model.dimension

    def encode_text(self, text: str) -> np.ndarray:
        """
        Векторизация одного текста.

        Args:
            text: Текст для векторизации

        Returns:
            Numpy array с эмбеддингом
        """
        # Валидация
        if text is None:
            raise TypeError("text must be a string, not None")
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text)}")
        if not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embeddings = self.model.embed_text([text])

        # Если 2D массив с одним элементом, возвращаем 1D
        if embeddings.ndim == 2 and embeddings.shape[0] == 1:
            return embeddings[0]

        return embeddings

    def encode_batch(
            self,
            texts: List[str],
            batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Векторизация батча текстов.

        Args:
            texts: Список текстов
            batch_size: Размер батча
            show_progress: Показывать прогресс

        Returns:
            Numpy array с эмбеддингами
        """

        # Валидация
        if texts is None:
            raise TypeError("texts list must not be None")
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list, got {type(texts)}")
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if any(t is None for t in texts):
            raise ValueError("None element in texts list")

        return self.model.embed_batch(
            texts,
            batch_size=batch_size
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Векторизация поискового запроса.

        Args:
            query: Поисковый запрос

        Returns:
            Numpy array с эмбеддингом
        """
        # Используем тот же метод что и для текста
        return self.encode_text(query)

    def __repr__(self) -> str:
        return f"CodeEmbedder(model={self.model.__class__.__name__}, dim={self.embedding_dim})"


def create_code_embedder(
        model_type: EmbeddingModelType = EmbeddingModelType.REMOTE,
        **kwargs
) -> CodeEmbedder:
    """
    Создание CodeEmbedder.

    Args:
        model_type: Тип модели
        **kwargs: Параметры модели

    Returns:
        CodeEmbedder
    """
    return CodeEmbedder(model_type=model_type, **kwargs)
