"""
Мониторинг изменений в Git репозиториях.
"""

from pathlib import Path
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time

try:
    from git import Repo

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    Repo = None

import logging
logger = logging.getLogger(__name__)


# ==================== Enums & Data Classes ====================

class ChangeType(Enum):
    """Типы изменений."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class CommitInfo:
    """Информация о коммите."""
    hash: str
    short_hash: str
    message: str
    author: str
    email: str
    date: datetime
    files_changed: List[str]

    @classmethod
    def from_git_commit(cls, commit) -> 'CommitInfo':
        """Создание из Git commit объекта."""
        return cls(
            hash=commit.hexsha,
            short_hash=commit.hexsha[:7],
            message=commit.message.strip(),
            author=commit.author.name,
            email=commit.author.email,
            date=datetime.fromtimestamp(commit.committed_date),
            files_changed=list(commit.stats.files.keys())
        )

    def to_dict(self) -> Dict:
        """Конвертация в словарь."""
        return {
            "hash": self.hash,
            "short_hash": self.short_hash,
            "message": self.message,
            "author": self.author,
            "email": self.email,
            "date": self.date.isoformat(),
            "files_changed": self.files_changed
        }


@dataclass
class ChangeEvent:
    """Событие изменения в репозитории."""
    repository_name: str
    change_type: ChangeType
    file_path: Path
    commit_info: Optional[CommitInfo] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """Конвертация в словарь."""
        return {
            "repository_name": self.repository_name,
            "change_type": self.change_type.value,
            "file_path": str(self.file_path),
            "commit_info": self.commit_info.to_dict() if self.commit_info else None,
            "timestamp": self.timestamp.isoformat()
        }


# ==================== RepositoryMonitor ====================

class RepositoryMonitor:
    """Мониторинг изменений в репозиториях."""

    def __init__(self, repo_manager=None):
        """
        Инициализация монитора.

        Args:
            repo_manager: RepositoryManager instance
        """
        if not GIT_AVAILABLE:
            raise RuntimeError("GitPython not installed")

        self.repo_manager = repo_manager
        self._last_commits: Dict[str, str] = {}  # repo_name -> last_commit_hash
        self._callbacks: List[Callable[[ChangeEvent], None]] = []
        self._running = False

    def add_callback(self, callback: Callable[[ChangeEvent], None]):
        """
        Добавление callback для событий изменений.

        Args:
            callback: Функция для вызова при изменениях
        """
        self._callbacks.append(callback)

    def start_monitoring(
            self,
            repository_path: Path,
            repository_name: str,
            interval: int = 60
    ):
        """
        Запуск мониторинга репозитория.

        Args:
            repository_path: Путь к репозиторию
            repository_name: Имя репозитория
            interval: Интервал проверки в секундах
        """
        if not repository_path.exists():
            raise ValueError(f"Repository path does not exist: {repository_path}")

        self._running = True
        logger.info(f"Starting monitoring for {repository_name}")

        try:
            repo = Repo(str(repository_path))

            # Сохраняем текущий коммит
            self._last_commits[repository_name] = repo.head.commit.hexsha

            while self._running:
                try:
                    # Проверяем изменения
                    changes = self.check_for_changes(repo, repository_name)

                    if changes:
                        logger.info(f"Detected {len(changes)} changes in {repository_name}")

                        # Вызываем callbacks
                        for change in changes:
                            self._notify_callbacks(change)

                    time.sleep(interval)

                except Exception as e:
                    logger.error(f"Error during monitoring: {e}")
                    time.sleep(interval)

        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            raise

    def stop_monitoring(self):
        """Остановка мониторинга."""
        self._running = False
        logger.info("Monitoring stopped")

    def check_for_changes(
            self,
            repo: Repo,
            repository_name: str
    ) -> List[ChangeEvent]:
        """
        Проверка изменений в репозитории.

        Args:
            repo: Git Repo объект
            repository_name: Имя репозитория

        Returns:
            Список событий изменений
        """
        changes = []

        try:
            # Получаем текущий коммит
            current_commit = repo.head.commit.hexsha
            last_commit = self._last_commits.get(repository_name)

            if last_commit and current_commit != last_commit:
                # Есть новые коммиты
                new_commits = list(repo.iter_commits(f'{last_commit}..{current_commit}'))

                for commit in reversed(new_commits):
                    commit_info = CommitInfo.from_git_commit(commit)

                    # Анализируем измененные файлы
                    if commit.parents:
                        # У коммита есть родители - сравниваем с первым родителем
                        parent = commit.parents[0]
                        diffs = parent.diff(commit)
                    else:
                        # Первый коммит в репозитории - все файлы новые
                        diffs = commit.diff(None)

                    for diff in diffs:
                        change_type = self._determine_change_type(diff)

                        # Определяем путь файла
                        file_path = diff.b_path if diff.b_path else diff.a_path

                        change_event = ChangeEvent(
                            repository_name=repository_name,
                            change_type=change_type,
                            file_path=Path(file_path),
                            commit_info=commit_info
                        )

                        changes.append(change_event)

                # Обновляем последний коммит
                self._last_commits[repository_name] = current_commit

            # Проверяем uncommitted изменения
            uncommitted = self._check_uncommitted_changes(repo, repository_name)
            changes.extend(uncommitted)

        except Exception as e:
            logger.error(f"Error checking changes: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return changes

    def _check_uncommitted_changes(
            self,
            repo: Repo,
            repository_name: str
    ) -> List[ChangeEvent]:
        """Проверка несохраненных изменений."""
        changes = []

        try:
            # Modified files
            for item in repo.index.diff(None):
                changes.append(ChangeEvent(
                    repository_name=repository_name,
                    change_type=ChangeType.MODIFIED,
                    file_path=Path(item.a_path)
                ))

            # Untracked files
            for file_path in repo.untracked_files:
                changes.append(ChangeEvent(
                    repository_name=repository_name,
                    change_type=ChangeType.ADDED,
                    file_path=Path(file_path)
                ))

        except Exception as e:
            logger.error(f"Error checking uncommitted changes: {e}")

        return changes

    def _determine_change_type(self, diff) -> ChangeType:
        """Определение типа изменения из diff."""
        if diff.new_file:
            return ChangeType.ADDED
        elif diff.deleted_file:
            return ChangeType.DELETED
        elif diff.renamed_file:
            return ChangeType.RENAMED
        else:
            return ChangeType.MODIFIED

    def _notify_callbacks(self, change_event: ChangeEvent):
        """Вызов всех зарегистрированных callbacks."""
        for callback in self._callbacks:
            try:
                callback(change_event)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

    def get_recent_commits(
            self,
            repository_path: Path,
            max_count: int = 10
    ) -> List[CommitInfo]:
        """
        Получение последних коммитов.

        Args:
            repository_path: Путь к репозиторию
            max_count: Максимальное количество коммитов

        Returns:
            Список CommitInfo
        """
        try:
            repo = Repo(str(repository_path))
            commits = []

            for commit in repo.iter_commits(max_count=max_count):
                commits.append(CommitInfo.from_git_commit(commit))

            return commits

        except Exception as e:
            logger.error(f"Error getting recent commits: {e}")
            return []

    # ==================== Дополнительные методы ====================

    def get_repository_statistics(self, repo_name: str) -> Dict[str, any]:
        """Расширенная статистика репозитория."""
        try:
            repo_info = self.repo_manager.get_repository_info(repo_name)
            repo = Repo(repo_info.local_path)

            # Основная статистика
            total_commits = len(list(repo.iter_commits()))
            branches = [b.name for b in repo.branches]
            tags = [t.name for t in repo.tags]

            # Статистика по типам файлов
            file_types = {}
            for item in repo.tree().traverse():
                if item.type == 'blob':
                    ext = Path(item.path).suffix
                    file_types[ext] = file_types.get(ext, 0) + 1

            # Статистика по авторам
            authors = {}
            for commit in repo.iter_commits(max_count=100):
                author = commit.author.name
                authors[author] = authors.get(author, 0) + 1

            return {
                "repository": repo_name,
                "provider": repo_info.provider,
                "total_commits": total_commits,
                "branches": branches,
                "tags": tags,
                "file_types": file_types,
                "top_contributors": dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]),
                "last_commit": repo_info.last_commit,
                "last_updated": repo_info.last_updated.isoformat() if repo_info.last_updated else None
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики для {repo_name}: {e}")
            return {}


"""
Опциональные улучшения для monitor.py
Добавьте эти методы в класс RepositoryMonitor
"""
from pathlib import Path
from typing import Optional, Dict, List

from git import Repo


def get_azure_devops_build_status(
        self,
        repo_name: str,
        azure_connection: str
) -> Optional[Dict]:
    """
    Получение статуса последнего build из Azure DevOps (опционально).

    Args:
        repo_name: Имя репозитория
        azure_connection: Имя Azure подключения

    Returns:
        Информация о build или None
    """
    try:
        repo_info = self.repo_manager.get_repository_info(repo_name)

        # Проверяем, что это Azure DevOps репозиторий
        if repo_info.provider != "azure_devops":
            logger.warning(f"{repo_name} не является Azure DevOps репозиторием")
            return None

        # Если у менеджера есть Azure клиент
        if hasattr(self.repo_manager, '_azure_clients') and azure_connection in self.repo_manager._azure_clients:
            client = self.repo_manager._azure_clients[azure_connection]

            # Запрос к Azure DevOps Pipelines API
            url = f"{client.base_url}/{client.collection}/{repo_info.azure_project}/_apis/build/builds"
            params = {
                "api-version": "6.0",
                "repositoryId": repo_name,
                "$top": 1
            }

            response = client.session.get(url, params=params)
            response.raise_for_status()

            builds = response.json().get('value', [])
            if builds:
                latest_build = builds[0]
                return {
                    "id": latest_build.get("id"),
                    "status": latest_build.get("status"),
                    "result": latest_build.get("result"),
                    "buildNumber": latest_build.get("buildNumber"),
                    "startTime": latest_build.get("startTime"),
                    "finishTime": latest_build.get("finishTime")
                }

        return None

    except Exception as e:
        logger.error(f"Ошибка получения build статуса для {repo_name}: {e}")
        return None


def check_azure_pull_requests(
        self,
        repo_name: str,
        azure_connection: str
) -> List[Dict]:
    """
    Проверка открытых Pull Requests в Azure DevOps (опционально).

    Args:
        repo_name: Имя репозитория
        azure_connection: Имя Azure подключения

    Returns:
        Список открытых PR
    """
    try:
        repo_info = self.repo_manager.get_repository_info(repo_name)

        if repo_info.provider != "azure_devops":
            return []

        if hasattr(self.repo_manager, '_azure_clients') and azure_connection in self.repo_manager._azure_clients:
            client = self.repo_manager._azure_clients[azure_connection]

            # Запрос к Azure DevOps Pull Requests API
            url = f"{client.base_url}/{client.collection}/{repo_info.azure_project}/_apis/git/repositories/{repo_name}/pullrequests"
            params = {
                "api-version": "6.0",
                "searchCriteria.status": "active"
            }

            response = client.session.get(url, params=params)
            response.raise_for_status()

            prs = response.json().get('value', [])
            return [{
                "id": pr.get("pullRequestId"),
                "title": pr.get("title"),
                "status": pr.get("status"),
                "createdBy": pr.get("createdBy", {}).get("displayName"),
                "creationDate": pr.get("creationDate"),
                "sourceRefName": pr.get("sourceRefName"),
                "targetRefName": pr.get("targetRefName")
            } for pr in prs]

        return []

    except Exception as e:
        logger.error(f"Ошибка получения PR для {repo_name}: {e}")
        return []


def get_repository_statistics(self, repo_name: str) -> Dict[str, any]:
    """
    Расширенная статистика репозитория.

    Returns:
        Словарь со статистикой
    """
    try:
        repo_info = self.repo_manager.get_repository_info(repo_name)
        repo = Repo(repo_info.local_path)

        # Основная статистика
        total_commits = len(list(repo.iter_commits()))
        branches = [b.name for b in repo.branches]
        tags = [t.name for t in repo.tags]

        # Статистика по типам файлов
        file_types = {}
        for item in repo.tree().traverse():
            if item.type == 'blob':
                ext = Path(item.path).suffix
                file_types[ext] = file_types.get(ext, 0) + 1

        # Статистика по авторам
        authors = {}
        for commit in repo.iter_commits(max_count=100):  # Последние 100 коммитов
            author = commit.author.name
            authors[author] = authors.get(author, 0) + 1

        return {
            "repository": repo_name,
            "provider": repo_info.provider,
            "total_commits": total_commits,
            "branches": branches,
            "tags": tags,
            "file_types": file_types,
            "top_contributors": dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]),
            "last_commit": repo_info.last_commit,
            "last_updated": repo_info.last_updated.isoformat() if repo_info.last_updated else None
        }

    except Exception as e:
        logger.error(f"Ошибка получения статистики для {repo_name}: {e}")
        return {}
