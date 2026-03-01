"""
Реализация модели Qwen3-Embedding для генерации эмбеддингов.
Поддержка моделей Qwen3-Embedding (0.6B, 4B, 8B).
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging

from .base import BaseEmbeddingModel, EmbeddingConfig

logger = logging.getLogger(__name__)


class QwenEmbedModel(BaseEmbeddingModel):
    """
    Реализация Qwen3-Embedding модели.

    Характеристики:
    - 0.6B/4B/8B параметров (зависит от версии)
    - 4096 токенов контекста
    - 2000 размерность
    - Мультиязычная поддержка
    - Высокое качество на code-specific задачах
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Инициализация Qwen3-Embedding модели.

        Args:
            config: Конфигурация модели
        """
        super().__init__(config)
        self.model_size = config.additional_params.get("model_size", "8B") if config.additional_params else "8B"

    def load_model(self) -> None:
        """Загрузка модели Qwen3-Embedding."""
        if self._is_loaded:
            return

        logger.info(f"Загрузка модели {self.config.model_path}...")

        # Определяем устройство
        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA недоступна, используется CPU")
            device = "cpu"

        try:
            # Загружаем модель через sentence-transformers
            self.model = SentenceTransformer(
                self.config.model_path,
                device=device,
                trust_remote_code=True
            )

            self._is_loaded = True
            logger.info(f"Модель Qwen3-Embedding {self.model_size} загружена на {device}")

        except Exception as e:
            logger.error(f"Ошибка загрузки Qwen3-Embedding: {e}")
            logger.info("Попытка загрузки через transformers...")

            # Альтернативный способ загрузки
            try:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True
                ).to(device)

                self._is_loaded = True
                self._use_transformers = True
                logger.info(f"Модель загружена через transformers на {device}")

            except Exception as e2:
                logger.error(f"Не удалось загрузить модель: {e2}")
                raise

    def embed_text(self, texts: List[str], prefix: Optional[str] = None) -> np.ndarray:
        """
        Генерация эмбеддингов для текстовых описаний.

        Args:
            texts: Список текстов для эмбеддинга
            prefix: Опциональный префикс (Qwen3 не использует префиксы)

        Returns:
            Numpy массив эмбеддингов
        """
        if not self._is_loaded:
            self.load_model()

        # Qwen3-Embedding обычно не использует префиксы
        # но поддерживаем для совместимости с интерфейсом
        if prefix and not hasattr(self, '_use_transformers'):
            texts = [f"{prefix} {text}" for text in texts]

        if hasattr(self, '_use_transformers'):
            # Используем transformers напрямую
            embeddings = self._encode_with_transformers(texts)
        else:
            # Используем sentence-transformers
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=self.config.batch_size
            )

        return embeddings

    def embed_code(self, code_snippets: List[str], prefix: Optional[str] = None) -> np.ndarray:
        """
        Генерация эмбеддингов для фрагментов кода.

        Qwen3-Embedding специально обучена на code tasks,
        поэтому хорошо работает с кодом без специальных префиксов.

        Args:
            code_snippets: Список фрагментов кода
            prefix: Опциональный префикс (не используется для Qwen3)

        Returns:
            Numpy массив эмбеддингов
        """
        if not self._is_loaded:
            self.load_model()

        return self.embed_text(code_snippets, prefix=None)

    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """
        Генерация эмбеддингов через transformers (альтернативный метод).

        Args:
            texts: Список текстов

        Returns:
            Numpy массив эмбеддингов
        """
        # Токенизация
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_tokens,
            return_tensors="pt"
        ).to(self.model.device)

        # Генерация эмбеддингов
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Используем last_hidden_state и берем [CLS] токен или mean pooling
            if hasattr(outputs, 'last_hidden_state'):
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = outputs[0].mean(dim=1)

        # Нормализация
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


def create_qwen_embed_model(
        model_size: str = "8B",
        device: str = "cpu",
        batch_size: int = 16,
        max_tokens: int = 4096
) -> QwenEmbedModel:
    """
    Фабричная функция для создания QwenEmbedModel.

    Args:
        model_size: Размер модели ("0.6B", "4B", "8B")
        device: Устройство для инференса ("cpu" или "cuda")
        batch_size: Размер батча для обработки
        max_tokens: Максимальное количество токенов (4096 для Qwen3)

    Returns:
        Экземпляр QwenEmbedModel

    Examples:
        >>> # Малая модель для CPU
        >>> model = create_qwen_embed_model(model_size="0.6B", device="cpu")

        >>> # Большая модель для GPU
        >>> model = create_qwen_embed_model(model_size="8B", device="cuda")
    """
    # Маппинг размеров на пути моделей
    model_paths = {
        "0.6B": "Qwen/Qwen3-Embedding-0.6B",
        "4B": "Qwen/Qwen3-Embedding-4B",
        "8B": "Qwen/Qwen3-Embedding-8B"
    }

    model_path = model_paths.get(model_size, model_paths["8B"])

    config = EmbeddingConfig(
        model_name=f"qwen3-embedding-{model_size.lower()}",
        dimension=2000,  # Qwen3-Embedding использует 2000 размерность
        max_tokens=max_tokens,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        normalize=True,
        additional_params={"model_size": model_size}
    )

    return QwenEmbedModel(config)
