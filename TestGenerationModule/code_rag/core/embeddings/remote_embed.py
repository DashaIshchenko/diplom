"""
Реализация модели эмбеддингов через Ollama API.
"""

from typing import List, Optional
import numpy as np
import requests
import logging
from urllib.parse import urljoin

from .base import BaseEmbeddingModel, EmbeddingConfig

logger = logging.getLogger(__name__)


def join_url(base: str, path: str) -> str:
    """
    Безопасное объединение base URL и path.

    Args:
        base: Базовый URL (например, "https://ai.url.ru/local")
        path: Путь (например, "/api/tags" или "api/tags")

    Returns:
        Объединенный URL
    """
    base = base.rstrip('/')
    path = path.lstrip('/')
    return f"{base}/{path}"


class OllamaEmbedModel(BaseEmbeddingModel):
    """
    Реализация модели через Ollama API.

    Поддерживает работу с моделями эмбеддингов в Ollama:
    - nomic-embed-text:latest
    - mxbai-embed-large
    - и другие
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Инициализация Ollama модели.

        Args:
            config: Конфигурация модели
                - model_path: base URL Ollama (например, "http://192.168.1.100:11434")
                - model_name: имя модели в Ollama (например, "nomic-embed-text:latest")
        """
        super().__init__(config)
        self.base_url = config.base_url
        self.ollama_model = config.additional_params.get(
            'model_name') if config.additional_params else config.model_name
        self.session = requests.Session()

        # Настройки таймаутов
        self.timeout = config.additional_params.get('timeout', 60) if config.additional_params else 60

        # Дополнительные настройки
        self.request_timeout = config.additional_params.get('request_timeout', 30) if config.additional_params else 30
        self.retry_attempts = config.additional_params.get('retry_attempts', 3) if config.additional_params else 3
        self.backoff_factor = config.additional_params.get('backoff_factor', 1) if config.additional_params else 1

    def load_model(self) -> None:
        """Проверка доступности Ollama и модели."""
        if self._is_loaded:
            return

        logger.info(f"Подключение к Ollama: {self.base_url}")
        logger.info(f"Модель: {self.ollama_model}")

        try:
            # Проверяем доступность Ollama
            tags_url = join_url(self.base_url, '/api/tags')
            response = self.session.get(tags_url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                available_models = [model['name'] for model in data.get('models', [])]

                logger.info(f"✓ Ollama доступен")
                logger.info(f"Доступные модели: {', '.join(available_models)}")

                # Проверяем наличие нужной модели
                if self.ollama_model in available_models:
                    logger.info(f"✓ Модель {self.ollama_model} найдена")
                else:
                    logger.warning(
                        f"⚠ Модель {self.ollama_model} не найдена в Ollama. "
                        f"Попытка использования все равно будет выполнена."
                    )

                self._is_loaded = True
            else:
                logger.warning(f"Ollama вернул статус {response.status_code}")
                self._is_loaded = True

        except requests.RequestException as e:
            logger.error(f"Не удалось подключиться к Ollama: {e}")
            raise ConnectionError(f"Ollama недоступен по адресу {self.base_url}: {e}")

    def _make_request_with_retry(self, url: str, payload: dict) -> dict:
        """
        Выполнение запроса с повторными попытками.

        Args:
            url: URL для запроса
            payload: Параметры запроса

        Returns:
            JSON ответ
        """
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt == self.retry_attempts - 1:
                    raise
                logger.warning(f"Попытка {attempt + 1} не удалась: {e}. Повтор...")
                import time
                time.sleep(self.backoff_factor * (2 ** attempt))
        return {}

    def embed_text(self, texts: List[str], prefix: Optional[str] = None) -> np.ndarray:
        """
        Генерация эмбеддингов через Ollama API.

        Args:
            texts: Список текстов для эмбеддинга
            prefix: Опциональный префикс

        Returns:
            Numpy массив эмбеддингов
        """
        if not self._is_loaded:
            self.load_model()

        # Добавляем префикс если нужно
        if prefix:
            texts = [f"{prefix} {text}" for text in texts]

        embeddings = []
        embed_url = join_url(self.base_url, '/api/embeddings')

        logger.debug(f"Векторизация {len(texts)} текстов через Ollama")

        for i, text in enumerate(texts):
            try:
                payload = {
                    'model': self.ollama_model,
                    'prompt': text
                }

                data = self._make_request_with_retry(embed_url, payload)

                # Ollama возвращает формат: {"embedding": [...]}
                embedding = data['embedding']
                embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    logger.debug(f"Обработано {i + 1}/{len(texts)} текстов")

            except requests.RequestException as e:
                logger.error(f"Ошибка при запросе эмбеддинга для текста {i}: {e}")
                raise
            except (KeyError, ValueError) as e:
                logger.error(f"Неожиданный формат ответа от Ollama: {e}")
                raise

        # Конвертируем в numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Нормализация если требуется
        if self.config.normalize:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms

        logger.debug(f"Получены эмбеддинги: shape={embeddings_array.shape}")

        return embeddings_array

    def embed_code(self, code_snippets: List[str], prefix: Optional[str] = None) -> np.ndarray:
        """
        Генерация эмбеддингов для фрагментов кода.

        Args:
            code_snippets: Список фрагментов кода
            prefix: Опциональный префикс

        Returns:
            Numpy массив эмбеддингов
        """
        # Для кода можем добавить префикс
        if prefix is None and self.ollama_model.startswith('nomic-embed'):
            prefix = "search_document:"

        return self.embed_text(code_snippets, prefix)


def create_ollama_embed_model(
        base_url: str = "https://ai.parma.ru/local/",
        model_name: str = "nomic-embed-text:latest",
        dimension: int = 768,
        max_tokens: int = 8192,
        batch_size: int = 32,
        timeout: int = 60,
        request_timeout: int = 30,
        retry_attempts: int = 3,
        backoff_factor: int = 1,
        normalize: bool = True,
        api_key: Optional[str] = None
) -> OllamaEmbedModel:
    """
    Фабричная функция для создания OllamaEmbedModel.

    Args:
        base_url: Базовый URL Ollama (например, "http://192.168.1.100:11434")
        model_name: Имя модели в Ollama (например, "nomic-embed-text:latest")
        dimension: Размерность эмбеддингов
        max_tokens: Максимальное количество токенов
        batch_size: Размер батча (не используется, так как Ollama обрабатывает по одному)
        timeout: Таймаут общего запроса в секундах
        request_timeout: Таймаут отдельного HTTP запроса
        retry_attempts: Количество попыток при ошибках
        backoff_factor: Фактор экспоненциальной задержки для повторов
        normalize: Нормализовать эмбеддинги
        api_key: API ключ для авторизации (если требуется)

    Returns:
        Экземпляр OllamaEmbedModel

    Example:
        >>> model = create_ollama_embed_model(
        ...     base_url="http://192.168.1.100:11434",
        ...     model_name="nomic-embed-text:latest",
        ...     dimension=768
        ... )
        >>> embeddings = model.embed_text(["hello world"])
    """
    additional_params = {
        'ollama_model': model_name,
        'timeout': timeout,
        'request_timeout': request_timeout,
        'retry_attempts': retry_attempts,
        'backoff_factor': backoff_factor
    }

    if api_key:
        additional_params['api_key'] = api_key

    config = EmbeddingConfig(
        model_name=model_name,
        dimension=dimension,
        max_tokens=max_tokens,
        base_url=base_url,
        model_path=base_url,
        device="ollama",  # Специальное значение для Ollama
        batch_size=batch_size,
        normalize=normalize,
        timeout=timeout,
        additional_params=additional_params
    )

    return OllamaEmbedModel(config)
