"""
Factory для создания моделей эмбеддингов.
Позволяет легко переключаться между различными моделями.
"""

from typing import Dict, Type, Optional
from enum import Enum
import logging

from .base import BaseEmbeddingModel, EmbeddingConfig
from .nomic_embed import NomicEmbedModel, create_nomic_embed_model
from .qwen_embed import QwenEmbedModel, create_qwen_embed_model
from .remote_embed import OllamaEmbedModel, create_ollama_embed_model

logger = logging.getLogger(__name__)


class EmbeddingModelType(Enum):
    """Поддерживаемые типы моделей эмбеддингов."""
    NOMIC_EMBED_TEXT = "nomic-embed-text"
    NOMIC_EMBED_CODE = "nomic-embed-code"
    QWEN3_EMBEDDING_06B = "qwen3-embedding-0.6b"
    QWEN3_EMBEDDING_4B = "qwen3-embedding-4b"
    QWEN3_EMBEDDING_8B = "qwen3-embedding-8b"
    CUSTOM = "custom"
    REMOTE = "remote"


class EmbeddingModelFactory:
    """
    Factory для создания моделей эмбеддингов.

    Использует паттерн проектирования Factory для инкапсуляции
    логики создания различных моделей эмбеддингов.
    """

    # Реестр зарегистрированных моделей
    _registry: Dict[str, Type[BaseEmbeddingModel]] = {
        EmbeddingModelType.NOMIC_EMBED_TEXT.value: NomicEmbedModel,
        EmbeddingModelType.NOMIC_EMBED_CODE.value: NomicEmbedModel,
        EmbeddingModelType.QWEN3_EMBEDDING_06B.value: QwenEmbedModel,
        EmbeddingModelType.QWEN3_EMBEDDING_4B.value: QwenEmbedModel,
        EmbeddingModelType.QWEN3_EMBEDDING_8B.value: QwenEmbedModel,
        EmbeddingModelType.REMOTE.value: OllamaEmbedModel,
    }

    @classmethod
    def register_model(
            cls,
            model_type: str,
            model_class: Type[BaseEmbeddingModel]
    ) -> None:
        """
        Регистрация новой модели эмбеддингов в фабрике.

        Это позволяет добавлять кастомные модели без изменения фабрики.

        Args:
            model_type: Строковый идентификатор модели
            model_class: Класс модели, наследующий BaseEmbeddingModel

        Raises:
            ValueError: Если класс не наследует BaseEmbeddingModel
        """
        if not issubclass(model_class, BaseEmbeddingModel):
            raise ValueError(
                f"Класс {model_class.__name__} должен наследовать BaseEmbeddingModel"
            )
        cls._registry[model_type] = model_class
        logger.info(f"Модель '{model_type}' зарегистрирована в фабрике")

    @classmethod
    def create_model(
            cls,
            model_type: str,
            config: Optional[EmbeddingConfig] = None,
            **kwargs
    ) -> BaseEmbeddingModel:
        """
        Создание модели эмбеддингов по типу.

        Args:
            model_type: Тип модели (из EmbeddingModelType или кастомный)
            config: Конфигурация модели (если None, создается из kwargs)
            **kwargs: Параметры для создания конфигурации (если config=None)

        Returns:
            Экземпляр модели эмбеддингов

        Raises:
            ValueError: Если тип модели не зарегистрирован

        Examples:
            >>> # Создание с дефолтными параметрами
            >>> model = EmbeddingModelFactory.create_model("nomic-embed-text")

            >>> # Создание с кастомными параметрами
            >>> model = EmbeddingModelFactory.create_model(
            ...     "nomic-embed-text",
            ...     device="cuda",
            ...     batch_size=64
            ... )

            >>> # Создание Qwen модели
            >>> model = EmbeddingModelFactory.create_model(
            ...     "qwen3-embedding-0.6b",
            ...     device="cpu"
            ... )
        """
        if model_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Модель '{model_type}' не зарегистрирована. "
                f"Доступные модели: {available}"
            )

        model_class = cls._registry[model_type]

        # Если конфигурация не передана, создаем из kwargs
        if config is None:
            config = cls._create_default_config(model_type, **kwargs)

        return model_class(config)

    @classmethod
    def _create_default_config(cls, model_type: str, **kwargs) -> EmbeddingConfig:
        """
        Создание конфигурации по умолчанию для модели.

        Args:
            model_type: Тип модели
            **kwargs: Дополнительные параметры конфигурации

        Returns:
            EmbeddingConfig с параметрами по умолчанию
        """
        # Конфигурации по умолчанию для разных моделей
        defaults = {
            # Nomic Embed Text
            EmbeddingModelType.NOMIC_EMBED_TEXT.value: {
                "model_name": "nomic-embed-text",
                "dimension": 768,
                "max_tokens": 8192,
                "model_path": "nomic-ai/nomic-embed-text-v1.5",
                "device": "cpu",
                "batch_size": 32,
                "normalize": True
            },
            # Nomic Embed Code
            EmbeddingModelType.NOMIC_EMBED_CODE.value: {
                "model_name": "nomic-embed-code",
                "dimension": 768,
                "max_tokens": 8192,
                "model_path": "nomic-ai/nomic-embed-code",
                "device": "cpu",
                "batch_size": 32,
                "normalize": True
            },
            # Qwen3 Embedding 0.6B
            EmbeddingModelType.QWEN3_EMBEDDING_06B.value: {
                "model_name": "qwen3-embedding-0.6b",
                "dimension": 2000,
                "max_tokens": 4096,
                "model_path": "Qwen/Qwen3-Embedding-0.6B",
                "device": "cpu",
                "batch_size": 16,
                "normalize": True,
                "additional_params": {"model_size": "0.6B"}
            },
            # Qwen3 Embedding 4B
            EmbeddingModelType.QWEN3_EMBEDDING_4B.value: {
                "model_name": "qwen3-embedding-4b",
                "dimension": 2000,
                "max_tokens": 4096,
                "model_path": "Qwen/Qwen3-Embedding-4B",
                "device": "cpu",
                "batch_size": 16,
                "normalize": True,
                "additional_params": {"model_size": "4B"}
            },
            # Qwen3 Embedding 8B
            EmbeddingModelType.QWEN3_EMBEDDING_8B.value: {
                "model_name": "qwen3-embedding-8b",
                "dimension": 2000,
                "max_tokens": 4096,
                "model_path": "Qwen/Qwen3-Embedding-8B",
                "device": "cuda",  # 8B лучше на GPU
                "batch_size": 8,  # Меньше batch для большой модели
                "normalize": True,
                "additional_params": {"model_size": "8B"}
            },
            EmbeddingModelType.REMOTE.value: {
                "base_url": "https://ai.parma.ru/local/",
                "model_name": "nomic-embed-text:latest",
                "model_path": "https://ai.parma.ru/local/",
                "dimension": 768,
                "max_tokens": 8192,
                "batch_size": 32,
                "api_key": None,
                "timeout": 30,
                "normalize": True
            }
        }

        # Получаем дефолтные параметры для модели
        default_params = defaults.get(model_type, {})

        # Объединяем с переданными параметрами (kwargs имеет приоритет)
        config_params = {**default_params, **kwargs}

        return EmbeddingConfig(**config_params)

    @classmethod
    def get_available_models(cls) -> list:
        """
        Получение списка доступных моделей.

        Returns:
            Список строковых идентификаторов моделей
        """
        return list(cls._registry.keys())

    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, any]:
        """
        Получение информации о модели.

        Args:
            model_type: Тип модели

        Returns:
            Словарь с информацией о модели
        """
        config = cls._create_default_config(model_type)

        return {
            "model_name": config.model_name,
            "dimension": config.dimension,
            "max_tokens": config.max_tokens,
            "model_path": config.model_path,
            "recommended_device": config.device,
            "batch_size": config.batch_size
        }

    @classmethod
    def create_nomic_embed(
            cls,
            device: str = "cpu",
            matryoshka_dim: Optional[int] = None,
            **kwargs
    ) -> NomicEmbedModel:
        """
        Удобная функция для создания nomic-embed-text модели.

        Args:
            device: Устройство ("cpu" или "cuda")
            matryoshka_dim: Размерность для Matryoshka (None для полной)
            **kwargs: Дополнительные параметры

        Returns:
            Экземпляр NomicEmbedModel

        Examples:
            >>> # Простое создание
            >>> model = EmbeddingModelFactory.create_nomic_embed()

            >>> # С GPU и сжатием
            >>> model = EmbeddingModelFactory.create_nomic_embed(
            ...     device="cuda",
            ...     matryoshka_dim=256
            ... )
        """
        return create_nomic_embed_model(
            device=device,
            matryoshka_dim=matryoshka_dim,
            **kwargs
        )

    @classmethod
    def create_qwen_embed(
            cls,
            model_size: str = "0.6B",
            device: str = "cpu",
            **kwargs
    ) -> QwenEmbedModel:
        """
        Удобная функция для создания Qwen3-Embedding модели.

        Args:
            model_size: Размер модели ("0.6B", "4B", "8B")
            device: Устройство ("cpu" или "cuda")
            **kwargs: Дополнительные параметры

        Returns:
            Экземпляр QwenEmbedModel

        Examples:
            >>> # Малая модель для CPU
            >>> model = EmbeddingModelFactory.create_qwen_embed(
            ...     model_size="0.6B"
            ... )

            >>> # Большая модель для GPU
            >>> model = EmbeddingModelFactory.create_qwen_embed(
            ...     model_size="8B",
            ...     device="cuda"
            ... )
        """
        return create_qwen_embed_model(
            model_size=model_size,
            device=device,
            **kwargs
        )


# Пример использования
def example_usage():
    """Примеры использования фабрики."""

    print("=== Доступные модели ===")
    available = EmbeddingModelFactory.get_available_models()
    print(f"Всего моделей: {len(available)}")
    for model_type in available:
        info = EmbeddingModelFactory.get_model_info(model_type)
        print(f"\n{model_type}:")
        print(f"  Dimension: {info['dimension']}")
        print(f"  Max tokens: {info['max_tokens']}")
        print(f"  Device: {info['recommended_device']}")

    print("\n=== Создание моделей ===")

    # Способ 1: Nomic Embed через фабрику
    model1 = EmbeddingModelFactory.create_model("nomic-embed-text")
    print(f"✓ {model1}")

    # Способ 2: Nomic Embed через shortcut
    model2 = EmbeddingModelFactory.create_nomic_embed(
        device="cpu",
        matryoshka_dim=256
    )
    print(f"✓ {model2}")

    # Способ 3: Qwen через фабрику
    model3 = EmbeddingModelFactory.create_model(
        "qwen3-embedding-0.6b",
        device="cpu"
    )
    print(f"✓ {model3}")

    # Способ 4: Qwen через shortcut
    model4 = EmbeddingModelFactory.create_qwen_embed(
        model_size="0.6B",
        device="cpu"
    )
    print(f"✓ {model4}")

    print("\n=== Сравнение моделей ===")
    print("Nomic Embed: быстрая, 768 dim, 8192 tokens, ~550MB")
    print("Qwen 0.6B:   средняя, 2000 dim, 4096 tokens, ~2.4GB")
    print("Qwen 8B:     медленная, 2000 dim, 4096 tokens, ~32GB")


if __name__ == "__main__":
    example_usage()
