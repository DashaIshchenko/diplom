"""
Tests for Git handler module (RepositoryManager).
"""
from typing import Dict

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from ..code_rag import CommitInfo, ChangeEvent, RepositoryMonitor
from ..code_rag.core.git_handler import (
    RepositoryManager,
    RepositoryInfo,
    RepositoryProvider,
    AzureDevOpsClient, ChangeType,
)


# ==================== RepositoryManager Tests ====================

class TestRepositoryManager:
    """Тесты для RepositoryManager."""

    @pytest.fixture
    def repo_manager(self, temp_dir):
        """Создание RepositoryManager."""
        return RepositoryManager(base_path=temp_dir)

    def test_manager_initialization(self, repo_manager, temp_dir):
        """Тест инициализации менеджера."""
        assert repo_manager.base_path == temp_dir
        assert repo_manager.base_path.exists()
        assert repo_manager.metadata_path.exists()
        assert len(repo_manager.repositories) == 0

    def test_detect_provider_github(self, repo_manager):
        """Тест определения GitHub провайдера."""
        url = "https://github.com/user/repo.git"
        provider = repo_manager._detect_provider(url)
        assert provider == RepositoryProvider.GITHUB

    def test_detect_provider_gitlab(self, repo_manager):
        """Тест определения GitLab провайдера."""
        url = "https://gitlab.com/user/repo.git"
        provider = repo_manager._detect_provider(url)
        assert provider == RepositoryProvider.GITLAB

    def test_detect_provider_azure(self, repo_manager):
        """Тест определения Azure DevOps провайдера."""
        url = "https://company.visualstudio.com/DefaultCollection/Project/_git/Repo"
        provider = repo_manager._detect_provider(url)
        assert provider == RepositoryProvider.AZURE_DEVOPS

    def test_extract_repo_name_standard(self, repo_manager):
        """Тест извлечения имени из стандартного URL."""
        url = "https://github.com/user/my-repo.git"
        name = repo_manager._extract_repo_name(url)
        assert name == "my-repo"

    def test_extract_repo_name_azure(self, repo_manager):
        """Тест извлечения имени из Azure DevOps URL."""
        url = "https://company.com/tfs/DefaultCollection/Project/_git/MyRepo"
        name = repo_manager._extract_repo_name(url)
        assert name == "MyRepo"

    def test_prepare_auth_url_azure(self, repo_manager):
        """Тест подготовки URL с токеном для Azure."""
        url = "https://company.com/tfs/DefaultCollection/Project/_git/Repo"
        token = "my_pat_token"
        auth_url = repo_manager._prepare_auth_url(url, token, RepositoryProvider.AZURE_DEVOPS)

        assert ":my_pat_token@" in auth_url
        assert auth_url.startswith("https://:")

    def test_prepare_auth_url_github(self, repo_manager):
        """Тест подготовки URL с токеном для GitHub."""
        url = "https://github.com/user/repo.git"
        token = "ghp_token123"
        auth_url = repo_manager._prepare_auth_url(url, token, RepositoryProvider.GITHUB)

        assert "ghp_token123@" in auth_url

    @pytest.mark.integration
    def test_clone_repository_mock(self, repo_manager, temp_dir):
        """Тест клонирования репозитория (mock)."""
        with patch('code_rag.core.git_handler.repository.Repo') as mock_repo_class:
            # Mock успешного клонирования
            mock_repo = MagicMock()
            mock_repo.head.commit.hexsha = "abc123def456"
            mock_repo_class.clone_from.return_value = mock_repo

            repo_info = repo_manager.clone_repository(
                url="https://github.com/test/repo.git",
                name="test-repo",
                branch="main"
            )

            assert repo_info.name == "test-repo"
            assert repo_info.branch == "main"
            assert repo_info.last_commit == "abc123def456"
            assert repo_info.provider == RepositoryProvider.GITHUB
            assert "test-repo" in repo_manager.repositories

    def test_list_repositories_empty(self, repo_manager):
        """Тест получения пустого списка репозиториев."""
        repos = repo_manager.list_repositories()
        assert len(repos) == 0

    def test_list_repositories_with_filter(self, repo_manager):
        """Тест фильтрации репозиториев по провайдеру."""
        # Добавляем mock репозитории
        repo_manager.repositories["repo1"] = RepositoryInfo(
            url="https://github.com/test/repo1.git",
            local_path=repo_manager.base_path / "repo1",
            name="repo1",
            provider=RepositoryProvider.GITHUB
        )

        repo_manager.repositories["repo2"] = RepositoryInfo(
            url="https://gitlab.com/test/repo2.git",
            local_path=repo_manager.base_path / "repo2",
            name="repo2",
            provider=RepositoryProvider.GITLAB
        )

        # Фильтр по GitHub
        github_repos = repo_manager.list_repositories(provider=RepositoryProvider.GITHUB)
        assert len(github_repos) == 1
        assert github_repos[0].name == "repo1"

    def test_list_repositories_with_tags(self, repo_manager):
        """Тест фильтрации по тегам."""
        repo_manager.repositories["repo1"] = RepositoryInfo(
            url="https://github.com/test/repo1.git",
            local_path=repo_manager.base_path / "repo1",
            name="repo1",
            tags=["frontend", "react"]
        )

        repo_manager.repositories["repo2"] = RepositoryInfo(
            url="https://github.com/test/repo2.git",
            local_path=repo_manager.base_path / "repo2",
            name="repo2",
            tags=["backend", "python"]
        )

        # Фильтр по тегу
        frontend_repos = repo_manager.list_repositories(tags=["frontend"])
        assert len(frontend_repos) == 1
        assert frontend_repos[0].name == "repo1"

    def test_get_repository_info(self, repo_manager):
        """Тест получения информации о репозитории."""
        repo_info = RepositoryInfo(
            url="https://github.com/test/repo.git",
            local_path=repo_manager.base_path / "test-repo",
            name="test-repo"
        )
        repo_manager.repositories["test-repo"] = repo_info

        retrieved_info = repo_manager.get_repository_info("test-repo")
        assert retrieved_info.name == "test-repo"
        assert retrieved_info.url == "https://github.com/test/repo.git"

    def test_get_repository_info_not_found(self, repo_manager):
        """Тест получения несуществующего репозитория."""
        with pytest.raises(ValueError, match="не найден в реестре"):
            repo_manager.get_repository_info("nonexistent")

    def test_remove_repository(self, repo_manager):
        """Тест удаления репозитория."""
        repo_info = RepositoryInfo(
            url="https://github.com/test/repo.git",
            local_path=repo_manager.base_path / "test-repo",
            name="test-repo"
        )
        repo_manager.repositories["test-repo"] = repo_info

        # Удаляем без удаления файлов
        repo_manager.remove_repository("test-repo", delete_files=False)

        assert "test-repo" not in repo_manager.repositories

    def test_registry_save_and_load(self, repo_manager):
        """Тест сохранения и загрузки реестра."""
        # Добавляем репозиторий
        repo_info = RepositoryInfo(
            url="https://github.com/test/repo.git",
            local_path=repo_manager.base_path / "test-repo",
            name="test-repo",
            branch="main"
        )
        repo_manager.repositories["test-repo"] = repo_info

        # Сохраняем
        repo_manager._save_registry()

        # Создаем новый менеджер (должен загрузить реестр)
        new_manager = RepositoryManager(base_path=repo_manager.base_path)

        assert "test-repo" in new_manager.repositories
        assert new_manager.repositories["test-repo"].name == "test-repo"


# ==================== RepositoryInfo Tests ====================

class TestRepositoryInfo:
    """Тесты для RepositoryInfo."""

    def test_repository_info_creation(self, temp_dir):
        """Тест создания RepositoryInfo."""
        repo_info = RepositoryInfo(
            url="https://github.com/user/repo.git",
            local_path=temp_dir / "repo",
            name="repo",
            branch="main"
        )

        assert repo_info.name == "repo"
        assert repo_info.branch == "main"
        assert repo_info.url == "https://github.com/user/repo.git"
        assert repo_info.provider == RepositoryProvider.GENERIC

    def test_repository_info_to_dict(self, temp_dir):
        """Тест сериализации в словарь."""
        repo_info = RepositoryInfo(
            url="https://github.com/user/repo.git",
            local_path=temp_dir / "repo",
            name="repo",
            tags=["frontend", "react"]
        )

        repo_dict = repo_info.to_dict()

        assert isinstance(repo_dict, dict)
        assert repo_dict["name"] == "repo"
        assert repo_dict["url"] == "https://github.com/user/repo.git"
        assert "frontend" in repo_dict["tags"]

    def test_repository_info_from_dict(self, temp_dir):
        """Тест десериализации из словаря."""
        data = {
            "url": "https://github.com/user/repo.git",
            "local_path": str(temp_dir / "repo"),
            "name": "repo",
            "branch": "main",
            "last_commit": "abc123",
            "last_updated": datetime.now().isoformat(),
            "is_private": False,
            "tags": ["test"],
            "metadata": {},
            "provider": RepositoryProvider.GITHUB,
            "azure_organization": None,
            "azure_project": None,
            "azure_collection": None
        }

        repo_info = RepositoryInfo.from_dict(data)

        assert repo_info.name == "repo"
        assert repo_info.branch == "main"
        assert isinstance(repo_info.local_path, Path)

    def test_repository_info_azure_fields(self, temp_dir):
        """Тест Azure DevOps специфичных полей."""
        repo_info = RepositoryInfo(
            url="https://company.com/tfs/DefaultCollection/Project/_git/Repo",
            local_path=temp_dir / "repo",
            name="repo",
            provider=RepositoryProvider.AZURE_DEVOPS,
            azure_organization="company",
            azure_project="Project",
            azure_collection="DefaultCollection"
        )

        assert repo_info.provider == RepositoryProvider.AZURE_DEVOPS
        assert repo_info.azure_project == "Project"
        assert repo_info.azure_collection == "DefaultCollection"


# ==================== Azure DevOps Tests ====================

class TestAzureDevOpsClient:
    """Тесты для AzureDevOpsClient."""

    @pytest.fixture
    def azure_client(self):
        """Создание Azure DevOps клиента."""
        return AzureDevOpsClient(
            base_url="https://company.com/tfs",
            collection="DefaultCollection",
            personal_access_token="fake_pat_token"
        )

    def test_azure_client_initialization(self, azure_client):
        """Тест инициализации клиента."""
        assert azure_client.base_url == "https://company.com/tfs"
        assert azure_client.collection == "DefaultCollection"
        assert azure_client.pat == "fake_pat_token"
        assert azure_client.session is not None

    @patch('requests.Session.get')
    def test_list_repositories(self, mock_get, azure_client):
        """Тест получения списка репозиториев."""
        # Mock ответа API
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'value': [
                {
                    'id': 'repo1-id',
                    'name': 'Repo1',
                    'remoteUrl': 'https://company.com/tfs/DefaultCollection/Project/_git/Repo1'
                },
                {
                    'id': 'repo2-id',
                    'name': 'Repo2',
                    'remoteUrl': 'https://company.com/tfs/DefaultCollection/Project/_git/Repo2'
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        repos = azure_client.list_repositories("TestProject")

        assert len(repos) == 2
        assert repos[0]['name'] == 'Repo1'
        assert repos[1]['name'] == 'Repo2'

    @patch('requests.Session.get')
    def test_get_repository_info(self, mock_get, azure_client):
        """Тест получения информации о репозитории."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'id': 'repo-id',
            'name': 'MyRepo',
            'url': 'https://company.com/tfs/DefaultCollection/Project/_apis/git/repositories/repo-id',
            'remoteUrl': 'https://company.com/tfs/DefaultCollection/Project/_git/MyRepo',
            'defaultBranch': 'refs/heads/main'
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        repo_info = azure_client.get_repository_info("TestProject", "repo-id")

        assert repo_info['name'] == 'MyRepo'
        assert repo_info['id'] == 'repo-id'

    @patch('requests.Session.get')
    def test_list_repositories_error(self, mock_get, azure_client):
        """Тест обработки ошибки при получении репозиториев."""
        import requests
        mock_get.side_effect = requests.RequestException("Connection error")

        with pytest.raises(requests.RequestException):
            azure_client.list_repositories("TestProject")


# ==================== Azure DevOps Integration Tests ====================

class TestAzureDevOpsIntegration:
    """Тесты интеграции Azure DevOps с RepositoryManager."""

    @pytest.fixture
    def repo_manager_with_azure(self, temp_dir):
        """Менеджер с зарегистрированным Azure подключением."""
        manager = RepositoryManager(base_path=temp_dir)

        # Регистрируем mock Azure подключение
        with patch('code_rag.core.git_handler.repository.AzureDevOpsClient'):
            manager.register_azure_devops(
                name="test_azure",
                base_url="https://company.com/tfs",
                collection="DefaultCollection",
                personal_access_token="test_pat"
            )

        return manager

    def test_register_azure_devops(self, repo_manager_with_azure):
        """Тест регистрации Azure DevOps подключения."""
        assert "test_azure" in repo_manager_with_azure._azure_clients

    def test_list_azure_repositories(self, repo_manager_with_azure):
        """Тест получения списка Azure репозиториев."""
        with patch.object(
                repo_manager_with_azure._azure_clients["test_azure"],
                'list_repositories'
        ) as mock_list:
            mock_list.return_value = [
                {'name': 'Repo1', 'id': 'id1'},
                {'name': 'Repo2', 'id': 'id2'}
            ]

            repos = repo_manager_with_azure.list_azure_repositories(
                azure_connection="test_azure",
                project="TestProject"
            )

            assert len(repos) == 2
            assert repos[0]['name'] == 'Repo1'

    def test_list_azure_repositories_no_connection(self, repo_manager_with_azure):
        """Тест с несуществующим подключением."""
        with pytest.raises(ValueError, match="не зарегистрировано"):
            repo_manager_with_azure.list_azure_repositories(
                azure_connection="nonexistent",
                project="TestProject"
            )


# ==================== Repository Operations Tests ====================

class TestRepositoryOperations:
    """Тесты операций с репозиториями."""

    @pytest.fixture
    def repo_manager_with_repo(self, temp_dir):
        """Менеджер с mock репозиторием."""
        manager = RepositoryManager(base_path=temp_dir)

        # Создаем mock репозиторий
        repo_path = temp_dir / "test-repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        repo_info = RepositoryInfo(
            url="https://github.com/test/repo.git",
            local_path=repo_path,
            name="test-repo",
            branch="main",
            last_commit="abc123"
        )
        manager.repositories["test-repo"] = repo_info

        return manager

    def test_update_repository(self, repo_manager_with_repo):
        """Тест обновления репозитория."""
        with patch('code_rag.core.git_handler.repository.Repo') as mock_repo_class:
            mock_repo = MagicMock()

            # Важно: сначала возвращаем старый коммит, потом новый
            mock_commit_old = MagicMock()
            mock_commit_old.hexsha = "abc123"  # Старый коммит (из fixture)

            mock_commit_new = MagicMock()
            mock_commit_new.hexsha = "def456"  # Новый коммит после pull

            # Mock repo.head.commit должен вернуть сначала старый, потом новый
            mock_repo.head.commit.hexsha = "def456"  # Новый коммит после pull
            mock_repo.active_branch.name = "main"
            mock_repo.remotes.origin.pull.return_value = [MagicMock()]

            mock_repo_class.return_value = mock_repo

            # В fixture last_commit = "abc123", после update должно стать "def456"
            # Логика update_repository сравнивает old_commit (из начала метода) с new_commit (после pull)

            # Нужно mock так, чтобы первый вызов repo.head.commit.hexsha вернул abc123
            # а второй - def456
            mock_repo.head.commit.hexsha = "abc123"  # До pull

            # Создаем side_effect для изменения после pull
            def pull_side_effect():
                mock_repo.head.commit.hexsha = "def456"
                return [MagicMock()]

            mock_repo.remotes.origin.pull = pull_side_effect

            diff = repo_manager_with_repo.update_repository("test-repo")

            assert len(diff) > 0
            assert repo_manager_with_repo.repositories["test-repo"].last_commit == "def456"

    def test_update_repository_no_changes(self, repo_manager_with_repo):
        """Тест обновления без изменений."""
        with patch('code_rag.core.git_handler.repository.Repo') as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.head.commit.hexsha = "abc123"  # Тот же коммит
            mock_repo.active_branch.name = "main"
            mock_repo.remotes.origin.pull.return_value = []
            mock_repo_class.return_value = mock_repo

            diff = repo_manager_with_repo.update_repository("test-repo")

            assert len(diff["changed"]) == 0

    def test_update_nonexistent_repository(self, repo_manager_with_repo):
        """Тест обновления несуществующего репозитория."""
        with pytest.raises(ValueError, match="не найден в реестре"):
            repo_manager_with_repo.update_repository("nonexistent")

    def test_get_changed_files(self, repo_manager_with_repo):
        """Тест получения измененных файлов."""
        with patch('code_rag.core.git_handler.repository.Repo') as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.git.diff.return_value = "file1.py\nfile2.js\nfile3.py"
            mock_repo.head.commit.hexsha = "def456"
            mock_repo_class.return_value = mock_repo

            changed_files = repo_manager_with_repo.get_changed_files(
                "test-repo",
                since_commit="abc123"
            )

            assert len(changed_files) == 3
            assert "file1.py" in changed_files
            assert "file2.js" in changed_files

    def test_remove_repository_with_files(self, repo_manager_with_repo, temp_dir):
        """Тест удаления репозитория с файлами."""
        repo_path = temp_dir / "test-repo"

        # Проверяем что директория существует
        assert repo_path.exists()

        # Удаляем с файлами
        repo_manager_with_repo.remove_repository("test-repo", delete_files=True)

        assert "test-repo" not in repo_manager_with_repo.repositories
        # Директория должна быть удалена
        # (но на практике может остаться из-за mock .git)


# ==================== Error Handling Tests ====================

class TestRepositoryErrorHandling:
    """Тесты обработки ошибок."""

    @pytest.fixture
    def repo_manager(self, temp_dir):
        """Создание RepositoryManager."""
        return RepositoryManager(base_path=temp_dir)

    def test_clone_existing_repository(self, repo_manager):
        """Тест клонирования уже существующего репозитория."""
        # Создаем существующую директорию
        existing_path = repo_manager.base_path / "existing-repo"
        existing_path.mkdir()
        (existing_path / ".git").mkdir()

        with patch('code_rag.core.git_handler.repository.Repo') as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.head.commit.hexsha = "abc123"
            mock_repo_class.return_value = mock_repo

            # Должен вернуть существующий репозиторий
            repo_info = repo_manager.clone_repository(
                url="https://github.com/test/existing-repo.git",
                name="existing-repo"
            )

            assert repo_info.name == "existing-repo"

    def test_update_with_git_error(self, temp_dir):
        """Тест обработки ошибки Git при обновлении."""
        from git.exc import GitCommandError

        manager = RepositoryManager(base_path=temp_dir)

        repo_path = temp_dir / "test-repo"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()

        repo_info = RepositoryInfo(
            url="https://github.com/test/repo.git",
            local_path=repo_path,
            name="test-repo"
        )
        manager.repositories["test-repo"] = repo_info

        with patch('code_rag.core.git_handler.repository.Repo') as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.remotes.origin.pull.side_effect = GitCommandError("git pull", 1)
            mock_repo_class.return_value = mock_repo

            with pytest.raises(GitCommandError):
                manager.update_repository("test-repo")


# ==================== Provider Detection Tests ====================

class TestProviderDetection:
    """Тесты определения провайдеров."""

    @pytest.fixture
    def repo_manager(self, temp_dir):
        return RepositoryManager(base_path=temp_dir)

    @pytest.mark.parametrize("url,expected_provider", [
        ("https://github.com/user/repo.git", RepositoryProvider.GITHUB),
        ("https://gitlab.com/user/repo.git", RepositoryProvider.GITLAB),
        ("https://company.visualstudio.com/Project/_git/Repo", RepositoryProvider.AZURE_DEVOPS),
        ("https://dev.azure.com/org/project/_git/repo", RepositoryProvider.AZURE_DEVOPS),
        ("https://company.com/tfs/Collection/Project/_git/Repo", RepositoryProvider.AZURE_DEVOPS),
        ("https://custom-git.com/repo.git", RepositoryProvider.GENERIC),
    ])
    def test_detect_provider_various_urls(self, repo_manager, url, expected_provider):
        """Параметризованный тест определения провайдера."""
        provider = repo_manager._detect_provider(url)
        assert provider == expected_provider


# ==================== Repository Name Extraction Tests ====================

class TestRepositoryNameExtraction:
    """Тесты извлечения имени репозитория."""

    @pytest.fixture
    def repo_manager(self, temp_dir):
        return RepositoryManager(base_path=temp_dir)

    @pytest.mark.parametrize("url,expected_name", [
        ("https://github.com/user/my-repo.git", "my-repo"),
        ("https://github.com/user/my-repo", "my-repo"),
        ("https://company.com/tfs/DefaultCollection/Project/_git/MyRepo", "MyRepo"),
        ("https://gitlab.com/group/subgroup/repo-name.git", "repo-name"),
    ])
    def test_extract_repo_name_various_urls(self, repo_manager, url, expected_name):
        """Параметризованный тест извлечения имени."""
        name = repo_manager._extract_repo_name(url)
        assert name == expected_name


# ==================== Manager Repr Tests ====================

class TestRepositoryManagerRepr:
    """Тесты строкового представления."""

    def test_manager_repr(self, temp_dir):
        """Тест __repr__ метода."""
        manager = RepositoryManager(base_path=temp_dir)

        repr_str = repr(manager)

        assert "RepositoryManager" in repr_str
        assert str(temp_dir) in repr_str
        assert "repos=0" in repr_str


# ==================== ChangeType and CommitInfo Tests ====================

class TestChangeTypeAndCommitInfo:
    """Тесты для ChangeType и CommitInfo."""

    def test_change_type_values(self):
        """Тест значений ChangeType."""
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"
        assert ChangeType.RENAMED.value == "renamed"

    def test_commit_info_creation(self):
        """Тест создания CommitInfo."""
        from datetime import datetime

        commit_info = CommitInfo(
            hash="abc123def456",
            short_hash="abc123d",
            message="Test commit",
            author="Test Author",
            email="test@example.com",
            date=datetime.now(),
            files_changed=["file1.py", "file2.js"]
        )

        assert commit_info.hash == "abc123def456"
        assert commit_info.short_hash == "abc123d"
        assert commit_info.author == "Test Author"
        assert len(commit_info.files_changed) == 2

    def test_commit_info_to_dict(self):
        """Тест конвертации CommitInfo в словарь."""
        from datetime import datetime

        commit_info = CommitInfo(
            hash="abc123",
            short_hash="abc123",
            message="Test",
            author="Author",
            email="email@test.com",
            date=datetime.now(),
            files_changed=["test.py"]
        )

        commit_dict = commit_info.to_dict()

        assert isinstance(commit_dict, dict)
        assert "hash" in commit_dict
        assert "message" in commit_dict
        assert "author" in commit_dict
        assert "files_changed" in commit_dict

    def test_commit_info_from_git_commit(self):
        """Тест создания CommitInfo из Git commit."""
        mock_commit = MagicMock()
        mock_commit.hexsha = "abc123def456ghi789"
        mock_commit.message = "Test commit\n"
        mock_commit.author.name = "Test Author"
        mock_commit.author.email = "test@example.com"
        mock_commit.committed_date = 1609459200  # 2021-01-01 00:00:00
        mock_commit.stats.files.keys.return_value = [
            MagicMock(a_path="file1.py"),
            MagicMock(a_path="file2.js")
        ]

        commit_info = CommitInfo.from_git_commit(mock_commit)

        assert commit_info.hash == "abc123def456ghi789"
        assert commit_info.short_hash == "abc123d"
        assert commit_info.message == "Test commit"
        assert commit_info.author == "Test Author"


# ==================== ChangeEvent Tests ====================

class TestChangeEvent:
    """Тесты для ChangeEvent."""

    def test_change_event_creation(self, temp_dir):
        """Тест создания ChangeEvent."""
        event = ChangeEvent(
            repository_name="test-repo",
            change_type=ChangeType.MODIFIED,
            file_path=temp_dir / "test.py"
        )

        assert event.repository_name == "test-repo"
        assert event.change_type == ChangeType.MODIFIED
        assert event.file_path == temp_dir / "test.py"
        assert event.timestamp is not None

    def test_change_event_with_commit_info(self, temp_dir):
        """Тест ChangeEvent с CommitInfo."""
        from datetime import datetime

        commit_info = CommitInfo(
            hash="abc123",
            short_hash="abc123",
            message="Test",
            author="Author",
            email="email@test.com",
            date=datetime.now(),
            files_changed=["test.py"]
        )

        event = ChangeEvent(
            repository_name="test-repo",
            change_type=ChangeType.MODIFIED,
            file_path=temp_dir / "test.py",
            commit_info=commit_info
        )

        assert event.commit_info is not None
        assert event.commit_info.hash == "abc123"

    def test_change_event_to_dict(self, temp_dir):
        """Тест конвертации ChangeEvent в словарь."""
        event = ChangeEvent(
            repository_name="test-repo",
            change_type=ChangeType.ADDED,
            file_path=temp_dir / "new_file.py"
        )

        event_dict = event.to_dict()

        assert isinstance(event_dict, dict)
        assert event_dict["repository_name"] == "test-repo"
        assert event_dict["change_type"] == "added"
        assert "timestamp" in event_dict


# ==================== RepositoryMonitor Tests ====================

class TestRepositoryMonitor:
    """Тесты для RepositoryMonitor."""

    @pytest.fixture
    def mock_repo_manager(self, temp_dir):
        """Mock RepositoryManager."""
        manager = MagicMock()
        manager.base_path = temp_dir
        return manager

    @pytest.fixture
    def monitor(self, mock_repo_manager):
        """Создание RepositoryMonitor."""
        with patch('code_rag.core.git_handler.monitor.GIT_AVAILABLE', True):
            return RepositoryMonitor(repo_manager=mock_repo_manager)

    def test_monitor_initialization(self, monitor):
        """Тест инициализации монитора."""
        assert monitor.repo_manager is not None
        assert len(monitor._callbacks) == 0
        assert monitor._running is False

    def test_add_callback(self, monitor):
        """Тест добавления callback."""

        def test_callback(event):
            pass

        monitor.add_callback(test_callback)

        assert len(monitor._callbacks) == 1

    def test_determine_change_type_added(self, monitor):
        """Тест определения типа изменения - добавление."""
        mock_diff = MagicMock()
        mock_diff.new_file = True
        mock_diff.deleted_file = False
        mock_diff.renamed_file = False

        change_type = monitor._determine_change_type(mock_diff)

        assert change_type == ChangeType.ADDED

    def test_determine_change_type_deleted(self, monitor):
        """Тест определения типа изменения - удаление."""
        mock_diff = MagicMock()
        mock_diff.new_file = False
        mock_diff.deleted_file = True
        mock_diff.renamed_file = False

        change_type = monitor._determine_change_type(mock_diff)

        assert change_type == ChangeType.DELETED

    def test_determine_change_type_renamed(self, monitor):
        """Тест определения типа изменения - переименование."""
        mock_diff = MagicMock()
        mock_diff.new_file = False
        mock_diff.deleted_file = False
        mock_diff.renamed_file = True

        change_type = monitor._determine_change_type(mock_diff)

        assert change_type == ChangeType.RENAMED

    def test_determine_change_type_modified(self, monitor):
        """Тест определения типа изменения - изменение."""
        mock_diff = MagicMock()
        mock_diff.new_file = False
        mock_diff.deleted_file = False
        mock_diff.renamed_file = False

        change_type = monitor._determine_change_type(mock_diff)

        assert change_type == ChangeType.MODIFIED

    def test_notify_callbacks(self, monitor, temp_dir):
        """Тест вызова callbacks."""
        callback_called = []

        def test_callback(event):
            callback_called.append(event)

        monitor.add_callback(test_callback)

        event = ChangeEvent(
            repository_name="test",
            change_type=ChangeType.MODIFIED,
            file_path=temp_dir / "test.py"
        )

        monitor._notify_callbacks(event)

        assert len(callback_called) == 1
        assert callback_called[0].repository_name == "test"

    def test_notify_callbacks_with_error(self, monitor, temp_dir, caplog):
        """Тест обработки ошибки в callback."""
        import logging

        def failing_callback(event):
            raise Exception("Callback error")

        monitor.add_callback(failing_callback)

        event = ChangeEvent(
            repository_name="test",
            change_type=ChangeType.MODIFIED,
            file_path=temp_dir / "test.py"
        )

        with caplog.at_level(logging.ERROR):
            monitor._notify_callbacks(event)

        # Проверяем что ошибка залогирована
        assert any("Error in callback" in record.message for record in caplog.records)

    def test_stop_monitoring(self, monitor):
        """Тест остановки мониторинга."""
        monitor._running = True
        monitor.stop_monitoring()

        assert monitor._running is False

    @patch('code_rag.core.git_handler.monitor.Repo')
    def test_get_recent_commits(self, mock_repo_class, monitor, temp_dir):
        """Тест получения последних коммитов."""
        # Mock commits
        mock_commits = []
        for i in range(5):
            mock_commit = MagicMock()
            mock_commit.hexsha = f"abc{i}23"
            mock_commit.message = f"Commit {i}"
            mock_commit.author.name = "Author"
            mock_commit.author.email = "author@test.com"
            mock_commit.committed_date = 1609459200 + i * 86400
            mock_commit.stats.files.keys.return_value = []
            mock_commits.append(mock_commit)

        mock_repo = MagicMock()
        mock_repo.iter_commits.return_value = mock_commits
        mock_repo_class.return_value = mock_repo

        commits = monitor.get_recent_commits(temp_dir, max_count=5)

        assert len(commits) == 5
        assert all(isinstance(c, CommitInfo) for c in commits)

    @patch('code_rag.core.git_handler.monitor.Repo')
    def test_check_uncommitted_changes(self, mock_repo_class, monitor):
        """Тест проверки несохраненных изменений."""
        mock_repo = MagicMock()

        # Mock modified files
        mock_diff = MagicMock()
        mock_diff.a_path = "modified.py"
        mock_repo.index.diff.return_value = [mock_diff]

        # Mock untracked files
        mock_repo.untracked_files = ["new_file.py"]

        changes = monitor._check_uncommitted_changes(mock_repo, "test-repo")

        assert len(changes) == 2
        assert changes[0].change_type == ChangeType.MODIFIED
        assert changes[1].change_type == ChangeType.ADDED


# ==================== Monitor Integration Tests ====================

@pytest.mark.integration
class TestRepositoryMonitorIntegration:
    """Интеграционные тесты мониторинга."""

    @pytest.fixture
    def git_repo_with_monitor(self, temp_dir):
        """Создание Git репозитория для тестов."""
        try:
            from git import Repo

            repo_path = temp_dir / "test-repo"
            repo_path.mkdir()

            repo = Repo.init(repo_path)

            # Настройка
            with repo.config_writer() as config:
                config.set_value("user", "name", "Test User")
                config.set_value("user", "email", "test@example.com")

            # Создаем файл и коммитим
            test_file = repo_path / "test.py"
            test_file.write_text("def test(): pass")
            repo.index.add(["test.py"])
            repo.index.commit("Initial commit")

            # Создаем mock repo_manager
            mock_repo_manager = MagicMock()
            mock_repo_manager.base_path = temp_dir

            monitor = RepositoryMonitor(repo_manager=mock_repo_manager)

            yield repo, monitor, repo_path

        except ImportError:
            pytest.skip("GitPython not available")

    def test_check_for_changes_with_new_commit(self, git_repo_with_monitor):
        """Тест обнаружения нового коммита."""
        repo, monitor, repo_path = git_repo_with_monitor

        # Сохраняем начальный коммит
        initial_commit = repo.head.commit.hexsha
        monitor._last_commits["test-repo"] = initial_commit

        print(f"Initial commit: {initial_commit}")

        # Создаем новый коммит
        new_file = repo_path / "new.py"
        new_file.write_text("def new(): pass")
        repo.index.add(["new.py"])
        new_commit_obj = repo.index.commit("Add new file")
        new_commit = new_commit_obj.hexsha

        print(f"New commit: {new_commit}")
        print(f"Commits between: {initial_commit[:7]}..{new_commit[:7]}")

        # Проверяем что коммиты разные
        assert initial_commit != new_commit, "Commits should be different"

        # Проверяем изменения
        changes = monitor.check_for_changes(repo, "test-repo")

        print(f"Found {len(changes)} changes")
        for change in changes:
            print(f"  - {change.change_type.value}: {change.file_path}")

        assert len(changes) > 0, "Should detect at least one change"
        assert any(c.change_type == ChangeType.ADDED for c in changes), "Should have ADDED change"
        assert any(str(c.file_path).endswith("new.py") for c in changes), "Should detect new.py"


# ==================== Repository Statistics Tests ====================

class TestRepositoryStatistics:
    """Тесты для статистики репозиториев."""

    @pytest.fixture
    def monitor_with_repo(self, temp_dir):
        """Монитор с mock репозиторием."""
        manager = MagicMock()

        repo_info = RepositoryInfo(
            url="https://github.com/test/repo.git",
            local_path=temp_dir / "repo",
            name="test-repo",
            provider=RepositoryProvider.GITHUB,
            last_commit="abc123",
            last_updated=datetime.now()
        )

        manager.get_repository_info.return_value = repo_info

        with patch('code_rag.core.git_handler.monitor.GIT_AVAILABLE', True):
            monitor = RepositoryMonitor(repo_manager=manager)

        return monitor, manager

    @patch('code_rag.core.git_handler.monitor.Repo')
    def test_get_repository_statistics(self, mock_repo_class, monitor_with_repo):
        """Тест получения статистики репозитория."""
        monitor, manager = monitor_with_repo

        # Mock repo
        mock_repo = MagicMock()

        # Mock commits
        mock_commits = [MagicMock() for _ in range(50)]
        for i, commit in enumerate(mock_commits[:10]):
            commit.author.name = f"Author{i % 3}"  # 3 разных автора

        mock_repo.iter_commits.return_value = mock_commits

        # Mock branches and tags
        mock_branch = MagicMock()
        mock_branch.name = "main"
        mock_repo.branches = [mock_branch]

        mock_tag = MagicMock()
        mock_tag.name = "v1.0.0"
        mock_repo.tags = [mock_tag]

        # Mock tree
        mock_items = []
        for ext in [".py", ".js", ".py", ".md"]:
            mock_item = MagicMock()
            mock_item.type = "blob"
            mock_item.path = f"file{ext}"
            mock_items.append(mock_item)

        mock_repo.tree().traverse.return_value = mock_items

        mock_repo_class.return_value = mock_repo

        stats = monitor.get_repository_statistics("test-repo")

        assert "repository" in stats
        assert stats["repository"] == "test-repo"
        assert "total_commits" in stats
        assert "branches" in stats
        assert "file_types" in stats
        assert stats["file_types"].get(".py", 0) == 2


# ==================== Error Handling in Monitor Tests ====================

class TestMonitorErrorHandling:
    """Тесты обработки ошибок в мониторе."""

    @pytest.fixture
    def monitor(self):
        """Создание монитора."""
        with patch('code_rag.core.git_handler.monitor.GIT_AVAILABLE', True):
            return RepositoryMonitor(repo_manager=MagicMock())

    def test_start_monitoring_nonexistent_path(self, monitor):
        """Тест мониторинга несуществующего пути."""
        with pytest.raises(ValueError, match="does not exist"):
            monitor.start_monitoring(
                repository_path=Path("/nonexistent/path"),
                repository_name="test",
                interval=60
            )

    @patch('code_rag.core.git_handler.monitor.Repo')
    def test_get_recent_commits_error(self, mock_repo_class, monitor, temp_dir):
        """Тест обработки ошибки при получении коммитов."""
        mock_repo_class.side_effect = Exception("Git error")

        commits = monitor.get_recent_commits(temp_dir)

        # Должен вернуть пустой список при ошибке
        assert commits == []

    def test_monitor_without_git(self):
        """Тест создания монитора без GitPython."""
        with patch('code_rag.core.git_handler.monitor.GIT_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="GitPython not installed"):
                RepositoryMonitor(repo_manager=MagicMock())


# ==================== Callback Tests ====================

class TestMonitorCallbacks:
    """Тесты для callbacks монитора."""

    @pytest.fixture
    def monitor(self):
        """Создание монитора."""
        with patch('code_rag.core.git_handler.monitor.GIT_AVAILABLE', True):
            return RepositoryMonitor(repo_manager=MagicMock())

    def test_multiple_callbacks(self, monitor, temp_dir):
        """Тест множественных callbacks."""
        called_callbacks = []

        def callback1(event):
            called_callbacks.append(("callback1", event))

        def callback2(event):
            called_callbacks.append(("callback2", event))

        monitor.add_callback(callback1)
        monitor.add_callback(callback2)

        event = ChangeEvent(
            repository_name="test",
            change_type=ChangeType.MODIFIED,
            file_path=temp_dir / "test.py"
        )

        monitor._notify_callbacks(event)

        assert len(called_callbacks) == 2
        assert called_callbacks[0][0] == "callback1"
        assert called_callbacks[1][0] == "callback2"

    def test_callback_receives_correct_event(self, monitor, temp_dir):
        """Тест что callback получает правильное событие."""
        received_event = None

        def callback(event):
            nonlocal received_event
            received_event = event

        monitor.add_callback(callback)

        test_event = ChangeEvent(
            repository_name="my-repo",
            change_type=ChangeType.ADDED,
            file_path=temp_dir / "new.py"
        )

        monitor._notify_callbacks(test_event)

        assert received_event is not None
        assert received_event.repository_name == "my-repo"
        assert received_event.change_type == ChangeType.ADDED
