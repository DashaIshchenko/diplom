"""
Реализация модели nomic-embed-text для генерации эмбеддингов.
Использует sentence-transformers для работы с моделью.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging

from .base import BaseEmbeddingModel, EmbeddingConfig

logger = logging.getLogger(__name__)


class NomicEmbedModel(BaseEmbeddingModel):
    """
    Реализация nomic-embed-text модели.

    Характеристики:
    - 137M параметров
    - 8192 токенов контекста
    - 768 размерность (или меньше с Matryoshka)
    - Поддержка префиксов для retrieval задач
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Инициализация Nomic Embed модели.

        Args:
            config: Конфигурация модели
        """
        super().__init__(config)
        self.use_matryoshka = config.additional_params.get("matryoshka_dim") if config.additional_params else None

    def load_model(self) -> None:
        """Загрузка модели nomic-embed-text."""
        if self._is_loaded:
            return

        logger.info(f"Загрузка модели {self.config.model_path}...")

        # Определяем устройство
        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA недоступна, используется CPU")
            device = "cpu"

        # Загружаем модель
        self.model = SentenceTransformer(
            self.config.model_path,
            device=device,
            trust_remote_code=True
        )

        self._is_loaded = True
        logger.info(f"Модель загружена на устройство: {device}")

    def _prepare_text_with_prefix(
            self,
            texts: List[str],
            prefix: Optional[str] = None
    ) -> List[str]:
        """
        Подготовка текста с префиксом для модели.

        Nomic Embed поддерживает префиксы:
        - "search_query:" для поисковых запросов
        - "search_document:" для документов/кода
        - "clustering:" для кластеризации
        - "classification:" для классификации

        Args:
            texts: Список текстов
            prefix: Префикс (если None, не добавляется)

        Returns:
            Список текстов с префиксом
        """
        if prefix:
            return [f"{prefix} {text}" for text in texts]
        return texts

    def embed_text(self, texts: List[str], prefix: Optional[str] = None) -> np.ndarray:
        """
        Генерация эмбеддингов для текстовых описаний.

        Args:
            texts: Список текстов для эмбеддинга
            prefix: Опциональный префикс (например, "search_query:")

        Returns:
            Numpy массив эмбеддингов
        """
        if not self._is_loaded:
            self.load_model()

        # Подготавливаем тексты с префиксом
        prepared_texts = self._prepare_text_with_prefix(texts, prefix)

        # Генерируем эмбеддинги
        embeddings = self.model.encode(
            prepared_texts,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Применяем Matryoshka сжатие, если настроено
        if self.use_matryoshka:
            embeddings = embeddings[:, :self.use_matryoshka]

        return embeddings

    def embed_code(self, code_snippets: List[str], prefix: Optional[str] = None) -> np.ndarray:
        """
        Генерация эмбеддингов для фрагментов кода.

        По умолчанию используется префикс "search_document:" для кода,
        так как код будет индексироваться в базе данных.

        Args:
            code_snippets: Список фрагментов кода
            prefix: Опциональный префикс (по умолчанию "search_document:")

        Returns:
            Numpy массив эмбеддингов
        """
        if not self._is_loaded:
            self.load_model()

        # Если префикс не указан, используем search_document для индексации
        if prefix is None:
            prefix = "search_document:"

        return self.embed_text(code_snippets, prefix)


def create_nomic_embed_model(
        model_path: str = "nomic-ai/nomic-embed-text-v1.5",
        dimension: int = 768,
        max_tokens: int = 8192,
        device: str = "cpu",
        batch_size: int = 32,
        matryoshka_dim: Optional[int] = None
) -> NomicEmbedModel:
    """
    Фабричная функция для создания NomicEmbedModel.

    Args:
        model_path: Путь к модели на HuggingFace
        dimension: Размерность эмбеддингов (768 для полной размерности)
        max_tokens: Максимальное количество токенов (8192 для nomic-embed-text)
        device: Устройство для инференса ("cpu" или "cuda")
        batch_size: Размер батча для обработки
        matryoshka_dim: Размерность для Matryoshka сжатия (64-768, None для полной)

    Returns:
        Экземпляр NomicEmbedModel
    """
    config = EmbeddingConfig(
        model_name="nomic-embed-text",
        dimension=matryoshka_dim or dimension,
        max_tokens=max_tokens,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        normalize=True,
        additional_params={"matryoshka_dim": matryoshka_dim} if matryoshka_dim else None
    )

    return NomicEmbedModel(config)
