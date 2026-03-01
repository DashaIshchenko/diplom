"""
Менеджер для управления git репозиториями.
Обеспечивает клонирование, обновление и отслеживание репозиториев.
Поддержка: GitHub, GitLab, Azure DevOps (TFS).
"""

from typing import Optional, Dict, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import requests
from urllib.parse import urlparse

from git import Repo, RemoteProgress
from git.exc import GitCommandError, InvalidGitRepositoryError

logger = logging.getLogger(__name__)


class RepositoryProvider:
    """Типы провайдеров репозиториев."""
    GITHUB = "github"
    GITLAB = "gitlab"
    AZURE_DEVOPS = "azure_devops"
    GENERIC = "generic"


@dataclass
class RepositoryInfo:
    """Информация о git репозитории."""
    url: str
    local_path: Path
    name: str
    branch: str = "main"
    last_commit: Optional[str] = None
    last_updated: Optional[datetime] = None
    is_private: bool = False
    ssh_key_path: Optional[str] = None
    access_token: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    provider: str = RepositoryProvider.GENERIC

    # Azure DevOps специфичные поля
    azure_organization: Optional[str] = None
    azure_project: Optional[str] = None
    azure_collection: Optional[str] = None

    def to_dict(self) -> Dict:
        """Сериализация в словарь."""
        return {
            "url": self.url,
            "local_path": str(self.local_path),
            "name": self.name,
            "branch": self.branch,
            "last_commit": self.last_commit,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "is_private": self.is_private,
            "tags": self.tags,
            "metadata": self.metadata,
            "provider": self.provider,
            "azure_organization": self.azure_organization,
            "azure_project": self.azure_project,
            "azure_collection": self.azure_collection
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RepositoryInfo":
        """Десериализация из словаря."""
        data = data.copy()
        data["local_path"] = Path(data["local_path"])
        if data.get("last_updated"):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        # Удаляем поля, которые не должны сохраняться
        data.pop("ssh_key_path", None)
        data.pop("access_token", None)
        return cls(**data)


class CloneProgress(RemoteProgress):
    """Прогресс бар для клонирования репозитория."""

    def __init__(self):
        super().__init__()
        self.current_op = ""

    def update(self, op_code, cur_count, max_count=None, message=''):
        """Обновление прогресса."""
        if message:
            self.current_op = message
        if max_count:
            percentage = (cur_count / max_count * 100)
            logger.info(f"Клонирование: {self.current_op} - {percentage:.1f}%")


class AzureDevOpsClient:
    """Клиент для работы с Azure DevOps REST API."""

    def __init__(self, base_url: str, collection: str, personal_access_token: str):
        """
        Инициализация клиента Azure DevOps.

        Args:
            base_url: Базовый URL (например: https://url_company/tfs)
            collection: Название коллекции
            personal_access_token: Personal Access Token для аутентификации
        """
        self.base_url = base_url.rstrip('/')
        self.collection = collection
        self.pat = personal_access_token
        self.session = requests.Session()
        self.session.auth = ('', personal_access_token)  # PAT в Basic Auth
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

        # Валидация подключения
        self._validate_connection()

    def _validate_connection(self):
        """Проверка валидности PAT и доступности сервера."""
        try:
            url = f"{self.base_url}/{self.collection}/_apis/projects?api-version=6.0"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise ValueError(f"Azure DevOps сервер недоступен: {self.base_url}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Невалидный Personal Access Token")
            elif e.response.status_code == 404:
                raise ValueError(f"Коллекция '{self.collection}' не найдена")
            raise ValueError(f"Ошибка подключения к Azure DevOps: {e}")

    def list_repositories(self, project: str, api_version: str = "6.0") -> List[Dict]:
        """
        Получение списка репозиториев из проекта.

        Args:
            project: Название проекта
            api_version: Версия API

        Returns:
            Список репозиториев

        Example:
            >>> repos = client.list_repositories("MyProject")
            >>> for repo in repos:
            ...     print(repo['name'], repo['remoteUrl'])
        """
        url = f"{self.base_url}/{self.collection}/{project}/_apis/git/repositories"
        params = {"api-version": api_version}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('value', [])
        except requests.RequestException as e:
            logger.error(f"Ошибка при получении списка репозиториев: {e}")
            raise

    def get_repository_info(self, project: str, repository_id: str, api_version: str = "6.0") -> Dict:
        """
        Получение информации о репозитории.

        Args:
            project: Название проекта
            repository_id: ID репозитория
            api_version: Версия API

        Returns:
            Информация о репозитории
        """
        url = f"{self.base_url}/{self.collection}/{project}/_apis/git/repositories/{repository_id}"
        params = {"api-version": api_version}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Ошибка при получении информации о репозитории: {e}")
            raise


class RepositoryManager:
    """
    Менеджер для управления множественными git репозиториями.

    Поддерживает:
    - GitHub, GitLab (через стандартный Git URL)
    - Azure DevOps (через REST API + Git)
    - Локальные репозитории
    """

    def __init__(self, base_path: Union[str, Path]):
        """
        Инициализация менеджера репозиториев.

        Args:
            base_path: Базовая директория для хранения репозиториев
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Директория для метаданных
        self.metadata_path = self.base_path / ".metadata"
        self.metadata_path.mkdir(exist_ok=True)

        # Реестр репозиториев
        self.repositories: Dict[str, RepositoryInfo] = {}
        self._load_registry()

        # Клиенты для разных провайдеров
        self._azure_clients: Dict[str, AzureDevOpsClient] = {}

        logger.info(f"RepositoryManager инициализирован: {self.base_path}")

    def register_azure_devops(
            self,
            name: str,
            base_url: str,
            collection: str,
            personal_access_token: str
    ) -> None:
        """
        Регистрация Azure DevOps organization для работы с API.

        Args:
            name: Уникальное имя для этого подключения
            base_url: Базовый URL (https://url_company/tfs)
            collection: Название коллекции
            personal_access_token: Personal Access Token

        Example:
            >>> manager.register_azure_devops(
            ...     name="company_azure",
            ...     base_url="https://url_company/tfs",
            ...     collection="DefaultCollection",
            ...     personal_access_token="your_pat_token"
            ... )
        """
        client = AzureDevOpsClient(base_url, collection, personal_access_token)
        self._azure_clients[name] = client
        logger.info(f"Azure DevOps клиент '{name}' зарегистрирован")

    def list_azure_repositories(
            self,
            azure_connection: str,
            project: str
    ) -> List[Dict]:
        """
        Получение списка репозиториев из Azure DevOps проекта.

        Args:
            azure_connection: Имя зарегистрированного Azure подключения
            project: Название проекта

        Returns:
            Список информации о репозиториях
        """
        if azure_connection not in self._azure_clients:
            raise ValueError(f"Azure подключение '{azure_connection}' не зарегистрировано")

        client = self._azure_clients[azure_connection]
        return client.list_repositories(project)

    def clone_repository(
            self,
            url: str,
            name: Optional[str] = None,
            branch: str = "main",
            is_private: bool = False,
            ssh_key_path: Optional[str] = None,
            access_token: Optional[str] = None,
            tags: Optional[List[str]] = None,
            provider: str = RepositoryProvider.GENERIC,
            # Azure DevOps специфичные параметры
            azure_organization: Optional[str] = None,
            azure_project: Optional[str] = None,
            azure_collection: Optional[str] = None
    ) -> RepositoryInfo:
        """
        Клонирование git репозитория.

        Args:
            url: URL репозитория
            name: Имя репозитория
            branch: Ветка для клонирования
            is_private: Флаг приватного репозитория
            ssh_key_path: Путь к SSH ключу
            access_token: Токен доступа (PAT для Azure DevOps)
            tags: Теги для категоризации
            provider: Провайдер (github, gitlab, azure_devops, generic)
            azure_organization: Azure DevOps organization
            azure_project: Azure DevOps project
            azure_collection: Azure DevOps collection

        Returns:
            RepositoryInfo с информацией о клонированном репозитории

        Examples:
            >>> # GitHub
            >>> repo = manager.clone_repository(
            ...     url="https://github.com/user/project.git",
            ...     provider="github"
            ... )

            >>> # Azure DevOps
            >>> repo = manager.clone_repository(
            ...     url="https://url_company/tfs/DefaultCollection/MyProject/_git/MyRepo",
            ...     is_private=True,
            ...     access_token="your_pat_token",
            ...     provider="azure_devops",
            ...     azure_collection="DefaultCollection",
            ...     azure_project="MyProject"
            ... )
        """
        # Определяем провайдер автоматически, если не указан
        if provider == RepositoryProvider.GENERIC:
            provider = self._detect_provider(url)

        # Извлекаем имя из URL, если не указано
        if name is None:
            name = self._extract_repo_name(url)

        local_path = self.base_path / name

        # Проверяем, не существует ли уже репозиторий
        if local_path.exists():
            logger.warning(f"Репозиторий {name} уже существует по пути {local_path}")
            if name in self.repositories:
                return self.repositories[name]
            else:
                return self._register_existing_repository(local_path, name, url, branch, provider)

        # Формируем URL с аутентификацией для приватных репозиториев
        clone_url = self._prepare_auth_url(url, access_token, provider) if is_private and access_token else url

        try:
            logger.info(f"Клонирование репозитория {name} из {url}...")

            # Настраиваем SSH, если используется
            git_env = None
            if is_private and ssh_key_path:
                git_ssh_cmd = f'ssh -i {ssh_key_path} -o StrictHostKeyChecking=no'
                git_env = {"GIT_SSH_COMMAND": git_ssh_cmd}

            # Клонируем репозиторий
            progress = CloneProgress()
            repo = Repo.clone_from(
                clone_url,
                local_path,
                branch=branch,
                progress=progress,
                env=git_env
            )

            # Получаем информацию о последнем коммите
            last_commit = repo.head.commit.hexsha

            # Создаем объект информации о репозитории
            repo_info = RepositoryInfo(
                url=url,
                local_path=local_path,
                name=name,
                branch=branch,
                last_commit=last_commit,
                last_updated=datetime.now(),
                is_private=is_private,
                ssh_key_path=ssh_key_path,
                access_token=access_token,
                tags=tags or [],
                provider=provider,
                azure_organization=azure_organization,
                azure_project=azure_project,
                azure_collection=azure_collection
            )

            # Сохраняем в реестр
            self.repositories[name] = repo_info
            self._save_registry()

            logger.info(f"Репозиторий {name} успешно клонирован ({provider})")
            return repo_info

        except GitCommandError as e:
            logger.error(f"Ошибка при клонировании репозитория {name}: {e}")
            raise

    def clone_azure_repository(
            self,
            azure_connection: str,
            project: str,
            repository_name: str,
            branch: str = "main",
            tags: Optional[List[str]] = None
    ) -> RepositoryInfo:
        """
        Удобный метод для клонирования репозитория из Azure DevOps.

        Args:
            azure_connection: Имя зарегистрированного Azure подключения
            project: Название проекта
            repository_name: Название репозитория
            branch: Ветка
            tags: Теги

        Returns:
            RepositoryInfo
        """

        # Проверка существования репозитория
        if repository_name in self.repositories:
            logger.warning(f"Репозиторий {repository_name} уже существует")
            return self.repositories[repository_name]

        if azure_connection not in self._azure_clients:
            raise ValueError(f"Azure подключение '{azure_connection}' не зарегистрировано")

        # Получаем информацию о репозитории через API
        client = self._azure_clients[azure_connection]
        # Проверка токена
        try:
            repos = client.list_repositories(project)
        except Exception as e:
            raise ValueError(f"Невалидный PAT токен или недоступный проект: {e}")

        repo_data = None
        for repo in repos:
            if repo['name'] == repository_name:
                repo_data = repo
                break

        if not repo_data:
            raise ValueError(f"Репозиторий '{repository_name}' не найден в проекте '{project}'")

        # Клонируем
        return self.clone_repository(
            url=repo_data['remoteUrl'],
            name=repository_name,
            branch=branch,
            is_private=True,
            access_token=client.pat,
            tags=tags,
            provider=RepositoryProvider.AZURE_DEVOPS,
            azure_organization=client.base_url.split('/')[-1],
            azure_project=project,
            azure_collection=client.collection
        )

    def update_repository(self, name: str, branch: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Обновление существующего репозитория (git pull).

        Args:
            name: Имя репозитория
            branch: Ветка для обновления

        Returns:
            True если были обновления
        """
        if name not in self.repositories:
            raise ValueError(f"Репозиторий {name} не найден в реестре")

        repo_info = self.repositories[name]

        try:
            repo = Repo(repo_info.local_path)

            # Сохраняем текущий коммит
            old_commit = repo.head.commit.hexsha

            # Переключаемся на нужную ветку, если указана
            if branch and branch != repo.active_branch.name:
                repo.git.checkout(branch)
                repo_info.branch = branch

            # Выполняем pull
            origin = repo.remotes.origin
            origin.pull()

            # Получаем новый коммит
            new_commit = repo.head.commit.hexsha

            if old_commit == new_commit:
                return {"changed": [], "deleted": []}

            # Получаем diff
            changed_files = []
            deleted_files = []

            for diff_item in repo.commit(old_commit).diff(new_commit):
                if diff_item.change_type == 'D':
                    deleted_files.append(diff_item.a_path)
                else:
                    changed_files.append(diff_item.b_path if diff_item.b_path else diff_item.a_path)

            # Обновляем информацию
            repo_info.last_commit = new_commit
            repo_info.last_updated = datetime.now()
            self._save_registry()

            has_changes = old_commit != new_commit

            if has_changes:
                logger.info(f"Репозиторий {name} обновлен: {old_commit[:7]} -> {new_commit[:7]}")
            else:
                logger.info(f"Репозиторий {name} уже актуален")

            return {"changed": changed_files, "deleted": deleted_files}

        except GitCommandError as e:
            logger.error(f"Ошибка при обновлении репозитория {name}: {e}")
            raise

    def _detect_provider(self, url: str) -> str:
        """Автоматическое определение провайдера по URL."""
        url_lower = url.lower()

        if 'github.com' in url_lower:
            return RepositoryProvider.GITHUB
        elif 'gitlab.com' in url_lower or 'gitlab' in url_lower:
            return RepositoryProvider.GITLAB
        elif '/tfs/' in url_lower or 'visualstudio.com' in url_lower or 'dev.azure.com' in url_lower:
            return RepositoryProvider.AZURE_DEVOPS

        return RepositoryProvider.GENERIC

    def _extract_repo_name(self, url: str) -> str:
        """Извлечение имени репозитория из URL."""
        # Azure DevOps: https://url_company/tfs/DefaultCollection/MyProject/_git/MyRepo
        if '/tfs/' in url or '/_git/' in url:
            parts = url.rstrip('/').split('/')
            if '_git' in parts:
                idx = parts.index('_git')
                if idx + 1 < len(parts):
                    return parts[idx + 1]

        # Стандартный Git URL
        return url.rstrip("/").split("/")[-1].replace(".git", "")

    def _prepare_auth_url(self, url: str, token: str, provider: str) -> str:
        """Подготовка URL с токеном аутентификации."""
        if provider == RepositoryProvider.AZURE_DEVOPS:
            # Azure DevOps PAT в Basic Auth
            if url.startswith("https://"):
                return url.replace("https://", f"https://:{token}@")
        elif provider in (RepositoryProvider.GITHUB, RepositoryProvider.GITLAB):
            # GitHub/GitLab token
            if url.startswith("https://"):
                return url.replace("https://", f"https://{token}@")
        return url

    def get_repository_info(self, name: str) -> RepositoryInfo:
        """Получение информации о репозитории."""
        if name not in self.repositories:
            raise ValueError(f"Репозиторий {name} не найден в реестре")
        return self.repositories[name]

    def list_repositories(self, tags: Optional[List[str]] = None, provider: Optional[str] = None) -> List[
        RepositoryInfo]:
        """
        Получение списка репозиториев.

        Args:
            tags: Фильтр по тегам
            provider: Фильтр по провайдеру
        """
        repos = list(self.repositories.values())

        if tags:
            repos = [r for r in repos if any(tag in r.tags for tag in tags)]

        if provider:
            repos = [r for r in repos if r.provider == provider]

        return repos

    def get_changed_files(self, name: str, since_commit: Optional[str] = None) -> List[str]:
        """Получение списка измененных файлов."""
        if name not in self.repositories:
            raise ValueError(f"Репозиторий {name} не найден в реестре")

        repo_info = self.repositories[name]
        repo = Repo(repo_info.local_path)

        if since_commit:
            diff = repo.git.diff(since_commit, repo.head.commit.hexsha, name_only=True)
            changed_files = diff.split('\n') if diff else []
        else:
            changed_files = [item.path for item in repo.tree().traverse() if item.type == 'blob']

        return changed_files

    def remove_repository(self, name: str, delete_files: bool = False) -> None:
        """Удаление репозитория из реестра."""
        if name not in self.repositories:
            raise ValueError(f"Репозиторий {name} не найден в реестре")

        repo_info = self.repositories.pop(name)

        if delete_files and repo_info.local_path.exists():
            import shutil
            shutil.rmtree(repo_info.local_path)
            logger.info(f"Файлы репозитория {name} удалены")

        self._save_registry()
        logger.info(f"Репозиторий {name} удален из реестра")

    def _register_existing_repository(
            self,
            local_path: Path,
            name: str,
            url: str,
            branch: str,
            provider: str
    ) -> RepositoryInfo:
        """Регистрация существующего репозитория."""
        try:
            repo = Repo(local_path)
            last_commit = repo.head.commit.hexsha

            repo_info = RepositoryInfo(
                url=url,
                local_path=local_path,
                name=name,
                branch=branch,
                last_commit=last_commit,
                last_updated=datetime.now(),
                provider=provider
            )


            self.repositories[name] = repo_info
            self._save_registry()

            logger.info(f"Существующий репозиторий {name} добавлен в реестр")
            return repo_info

        except InvalidGitRepositoryError:
            raise ValueError(f"Директория {local_path} не является git репозиторием")

    def _load_registry(self) -> None:
        """Загрузка реестра репозиториев."""
        registry_file = self.metadata_path / "registry.json"

        if registry_file.exists():
            try:
                with open(registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.repositories = {
                        name: RepositoryInfo.from_dict(info)
                        for name, info in data.items()
                    }
                logger.info(f"Загружено {len(self.repositories)} репозиториев из реестра")
            except Exception as e:
                logger.error(f"Ошибка загрузки реестра: {e}")
                self.repositories = {}

    def _save_registry(self) -> None:
        """Сохранение реестра репозиториев."""
        registry_file = self.metadata_path / "registry.json"

        data = {
            name: info.to_dict()
            for name, info in self.repositories.items()
        }

        try:
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Реестр сохранен: {len(self.repositories)} репозиториев")
        except Exception as e:
            logger.error(f"Ошибка сохранения реестра: {e}")

    def __repr__(self) -> str:
        return f"RepositoryManager(base_path={self.base_path}, repos={len(self.repositories)})"

def get_repository_manager() -> RepositoryManager:
    """
    Удобная функция для получения RepositoryManager.
    Используется в API для получения singleton instance.
    """
    from ...config.settings import get_settings
    settings = get_settings()
    return RepositoryManager(settings.repo_base_path)
