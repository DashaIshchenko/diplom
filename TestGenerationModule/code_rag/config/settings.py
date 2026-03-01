"""
Настройки приложения Code RAG.
Использует Pydantic для валидации и загрузки из .env файла.
"""

from typing import Optional, List
from pathlib import Path
from pydantic_settings import BaseSettings
import logging
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Основные настройки приложения."""

    # Application
    app_name: str = "code-rag"
    app_env: str = "development"
    log_level: str = "INFO"

    # Directories
    data_dir: Path = Path("./data")
    repo_base_path: Path = Path("./data/repositories")
    cache_dir: Path = Path("./data/cache")

    # Git Settings - GitHub
    github_token: Optional[str] = None

    # Git Settings - GitLab
    gitlab_token: Optional[str] = None

    # Git Settings - Azure DevOps
    azure_devops_base_url: Optional[str] = None  # https://url_company/tfs
    azure_devops_collection: str = "DefaultCollection"
    azure_devops_pat: Optional[str] = None  # Personal Access Token
    azure_devops_organization: Optional[str] = None

    # SSH Settings
    ssh_key_path: Path = Path("~/.ssh/id_rsa")

    # Embedding Model Settings
    default_embedding_model: str = "nomic-embed-text"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # Nomic Embed Settings
    nomic_model_path: str = "nomic-ai/nomic-embed-text-v1.5"
    nomic_dimension: int = 768
    nomic_max_tokens: int = 8192
    nomic_matryoshka_dim: Optional[int] = None

    # Qwen Embed Settings
    qwen_model_size: str = "0.6B"  # 0.6B, 4B, 8B
    qwen_dimension: int = 2000
    qwen_max_tokens: int = 4096

    # Qdrant Settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "code_embeddings"
    qdrant_use_https: bool = False  # Для облачного Qdrant

    # Vector DB Configuration
    vector_size: int = 768
    distance_metric: str = "cosine"  # cosine, dot, euclidean

    # Repository Monitor Settings
    monitor_check_interval: int = 300  # секунды (5 минут)
    monitor_auto_start: bool = False

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_workers: int = 1
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # LLM Settings (для Qwen-coder)
    llm_model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    llm_device: str = "cuda"
    llm_max_new_tokens: int = 2048
    llm_temperature: float = 0.7
    llm_top_p: float = 0.95

    # Performance Settings
    max_workers: int = 4
    batch_size: int = 32
    chunk_size: int = 512

    # Cache Settings
    enable_cache: bool = True
    cache_ttl: int = 3600  # секунды

    # Logging
    log_file: Path = Path("./logs/code_rag.log")
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # RAG Settings
    rag_top_k: int = 5
    rag_score_threshold: float = 0.7
    rag_expand_query: bool = True
    rag_use_dual_embeddings: bool = True

    # Repository Providers
    # Можно добавить несколько Azure DevOps подключений через JSON
    azure_connections: Optional[str] = None  # JSON строка с конфигурациями

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Позволяет использовать вложенные настройки через __
        env_nested_delimiter = "__"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Создаем необходимые директории
        self._create_directories()

    def _create_directories(self) -> None:
        """Создание необходимых директорий."""
        directories = [
            self.data_dir,
            self.repo_base_path,
            self.cache_dir,
            self.log_file.parent
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_azure_connections(self) -> List[dict]:
        """
        Получение списка Azure DevOps подключений из конфигурации.

        Returns:
            Список словарей с конфигурациями Azure DevOps

        Example в .env:
            AZURE_CONNECTIONS='[{"name": "company1", "base_url": "https://dev.azure.com/org1", "collection": "DefaultCollection", "pat": "token1"}, {"name": "company2", "base_url": "https://company2.visualstudio.com", "collection": "MyCollection", "pat": "token2"}]'
        """
        if not self.azure_connections:
            # Если не задано через JSON, создаем из основных настроек
            if self.azure_devops_base_url and self.azure_devops_pat:
                return [{
                    "name": "default",
                    "base_url": self.azure_devops_base_url,
                    "collection": self.azure_devops_collection,
                    "pat": self.azure_devops_pat
                }]
            return []

        # Парсим JSON
        import json
        try:
            connections = json.loads(self.azure_connections)
            return connections if isinstance(connections, list) else []
        except json.JSONDecodeError:
            return []

    def get_qdrant_url(self) -> str:
        """
        Получение полного URL для подключения к Qdrant.

        Returns:
            URL для подключения
        """
        protocol = "https" if self.qdrant_use_https else "http"
        return f"{protocol}://{self.qdrant_host}:{self.qdrant_port}"

    def get_embedding_config(self) -> dict:
        """
        Получение конфигурации для модели эмбеддингов.

        Returns:
            Словарь с параметрами модели
        """
        if self.default_embedding_model == "nomic-embed-text":
            return {
                "model_name": "nomic-embed-text",
                "model_path": self.nomic_model_path,
                "dimension": self.nomic_dimension,
                "max_tokens": self.nomic_max_tokens,
                "device": self.embedding_device,
                "batch_size": self.embedding_batch_size,
                "matryoshka_dim": self.nomic_matryoshka_dim
            }
        elif self.default_embedding_model.startswith("qwen"):
            return {
                "model_name": f"qwen3-embedding-{self.qwen_model_size.lower()}",
                "model_size": self.qwen_model_size,
                "dimension": self.qwen_dimension,
                "max_tokens": self.qwen_max_tokens,
                "device": self.embedding_device,
                "batch_size": self.embedding_batch_size
            }
        else:
            raise ValueError(f"Неизвестная модель: {self.default_embedding_model}")

    def is_production(self) -> bool:
        """Проверка, запущено ли приложение в production."""
        return self.app_env.lower() == "production"

    def is_development(self) -> bool:
        """Проверка, запущено ли приложение в development."""
        return self.app_env.lower() == "development"

_settings_instance = None

def get_settings() -> Settings:
    """Singleton с ленивой инициализацией."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

# Добавить валидацию в __init__
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._validate_config()
    self._create_directories()

def _validate_config(self):
    """Валидация критических параметров."""
    if self.qdrant_port <= 0 or self.qdrant_port > 65535:
        raise ValueError(f"Невалидный qdrant_port: {self.qdrant_port}")
    if self.vector_size != self.nomic_dimension:
        logger.warning(f"vector_size ({self.vector_size}) != nomic_dimension ({self.nomic_dimension})")


# Пример использования
if __name__ == "__main__":
    settings = get_settings()

    print("=== Основные настройки ===")
    print(f"App name: {settings.app_name}")
    print(f"Environment: {settings.app_env}")
    print(f"Log level: {settings.log_level}")

    print("\n=== Git настройки ===")
    print(f"GitHub token: {'✓' if settings.github_token else '✗'}")
    print(f"GitLab token: {'✓' if settings.gitlab_token else '✗'}")
    print(f"Azure DevOps URL: {settings.azure_devops_base_url or 'не настроено'}")
    print(f"Azure DevOps PAT: {'✓' if settings.azure_devops_pat else '✗'}")

    print("\n=== Azure DevOps подключения ===")
    azure_conns = settings.get_azure_connections()
    if azure_conns:
        for conn in azure_conns:
            print(f"- {conn['name']}: {conn['base_url']}/{conn['collection']}")
    else:
        print("Нет настроенных подключений")

    print("\n=== Embedding модель ===")
    emb_config = settings.get_embedding_config()
    print(f"Модель: {emb_config['model_name']}")
    print(f"Размерность: {emb_config['dimension']}")
    print(f"Device: {emb_config['device']}")

    print("\n=== Qdrant ===")
    print(f"URL: {settings.get_qdrant_url()}")
    print(f"Collection: {settings.qdrant_collection_name}")

    print("\n=== RAG ===")
    print(f"Top K: {settings.rag_top_k}")
    print(f"Score threshold: {settings.rag_score_threshold}")
    print(f"Dual embeddings: {settings.rag_use_dual_embeddings}")
