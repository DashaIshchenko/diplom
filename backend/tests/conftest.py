"""
Pytest configuration and fixtures for Code RAG tests.
"""
import sys

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import numpy as np

from code_rag.core import (
    CodeEmbedder,
    QdrantClient,
    VectorizationPipeline,
    RAGRetriever,
    QwenIntegration,
    ParserFactory,
    ProgrammingLanguage,
    CodeElementType,
    OllamaEmbedModel,
)

from code_rag.core.vector_db import CollectionSchema
import logging
logger = logging.getLogger(__name__)

# ==================== Temporary Directories ====================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Временная директория для тестов."""
    # Создаем временную директорию
    if sys.platform == 'win32':
        # На Windows используем системную TEMP папку явно
        import os
        temp_base = os.environ.get('TEMP', tempfile.gettempdir())
        # Создаем с ASCII именем
        temp_path = Path(tempfile.mkdtemp(prefix='code_rag_test_', dir=temp_base))
    else:
        temp_path = Path(tempfile.mkdtemp(prefix='code_rag_test_'))

    try:
        yield temp_path
    finally:
        # Более надежное удаление
        try:
            if temp_path.exists():
                # На Windows иногда нужно подождать
                import time
                time.sleep(0.1)

                # Сначала делаем все файлы записываемыми
                if sys.platform == 'win32':
                    import stat
                    for root, dirs, files in os.walk(temp_path):
                        for name in files:
                            filepath = os.path.join(root, name)
                            os.chmod(filepath, stat.S_IWRITE)

                shutil.rmtree(temp_path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir {temp_path}: {e}")


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Временный файл."""
    file_path = temp_dir / "test_file.py"
    file_path.touch()
    return file_path


# ==================== Sample Code ====================

@pytest.fixture
def sample_python_code() -> str:
    """Примерный Python код для тестов."""
    return '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b

def calculate_product(a: int, b: int) -> int:
    """Calculate product of two numbers."""
    return a * b

class Calculator:
    """Simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        result = a + b
        self.history.append(f"add({a}, {b}) = {result}")
        return result

    def subtract(self, a: int, b: int) -> int:
        """Subtract two numbers."""
        result = a - b
        self.history.append(f"subtract({a}, {b}) = {result}")
        return result
'''


@pytest.fixture
def sample_javascript_code() -> str:
    """Примерный JavaScript код."""
    return '''
function authenticateUser(username, password) {
    // Check credentials
    if (!username || !password) {
        throw new Error('Invalid credentials');
    }

    return verifyPassword(username, password);
}

function verifyPassword(username, password) {
    const hashedPassword = hash(password);
    return hashedPassword === getStoredPassword(username);
}

class UserManager {
    constructor() {
        this.users = new Map();
    }

    addUser(username, password) {
        this.users.set(username, hash(password));
    }

    removeUser(username) {
        return this.users.delete(username);
    }
}
'''


@pytest.fixture
def sample_files(temp_dir: Path, sample_python_code: str, sample_javascript_code: str) -> Dict[str, Path]:
    """Создание примерных файлов для тестов."""
    files = {}

    # Python файл
    python_file = temp_dir / "calculator.py"
    python_file.write_text(sample_python_code)
    files["python"] = python_file

    # JavaScript файл
    js_file = temp_dir / "auth.js"
    js_file.write_text(sample_javascript_code)
    files["javascript"] = js_file

    return files


# ==================== Embedder Fixtures ====================

@pytest.fixture(scope="session")
def embedder() -> CodeEmbedder:
    """Code embedder для тестов (session scope для скорости)."""
    return CodeEmbedder()


@pytest.fixture
def mock_embedder(monkeypatch):
    """Mock embedder для быстрых тестов без реальной модели."""

    class MockEmbedder:
        @property
        def embedding_dim(self) -> int:
            return 768

        def encode_text(self, text: str) -> np.ndarray:
            # Возвращаем случайный вектор
            np.random.seed(hash(text) % (2 ** 32))
            return np.random.rand(768).astype(np.float32)

        def encode_batch(self, texts: list, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
            return np.array([self.encode_text(t) for t in texts])

        def encode_query(self, query: str) -> np.ndarray:
            return self.encode_text(query)

    return MockEmbedder()


# ==================== Qdrant Fixtures ====================

@pytest.fixture(scope="session")
def qdrant_url() -> str:
    """URL Qdrant для тестов."""
    return "http://localhost:6333"


@pytest.fixture
def test_collection_name() -> str:
    """Имя тестовой коллекции."""
    import uuid
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def qdrant_client(qdrant_url: str) -> Generator[QdrantClient, None, None]:
    """Qdrant client для тестов."""

    try:

        client = QdrantClient(location=":memory:")
        #client = QdrantClient(url="http://localhost:6333")

        # Проверка доступности
        if not client.health_check():
            pytest.skip("Qdrant не доступен. Запустите: docker run -p 6333:6333 qdrant/qdrant")

        yield client
    except Exception as e:
        pytest.skip(f"Qdrant не доступен: {e}")


@pytest.fixture
def test_collection(
        qdrant_client: QdrantClient,
        test_collection_name: str,
        embedder: CodeEmbedder
) -> Generator[str, None, None]:
    """Создание и очистка тестовой коллекции."""
    # Создание коллекции
    schema = CollectionSchema(
        collection_name=test_collection_name,
        vector_size=embedder.embedding_dim
    )

    qdrant_client.create_collection(schema, recreate=True)

    yield test_collection_name

    # Очистка после теста
    try:
        qdrant_client.delete_collection(test_collection_name)
    except:
        pass


# ==================== Pipeline Fixtures ====================

@pytest.fixture
def vectorization_pipeline(
        test_collection: str,
        embedder: CodeEmbedder,
        qdrant_client: QdrantClient
) -> VectorizationPipeline:
    """Vectorization pipeline для тестов."""
    return VectorizationPipeline(
        collection_name=test_collection,
        embedder=embedder,
        qdrant_client=qdrant_client
    )


@pytest.fixture
def rag_retriever(
        test_collection: str,
        embedder: CodeEmbedder,
        qdrant_client: QdrantClient
) -> RAGRetriever:
    """RAG retriever для тестов."""
    return RAGRetriever(
        collection_name=test_collection,
        embedder=embedder,
        qdrant_client=qdrant_client,
        use_reranking=False  # Отключаем для быстроты тестов
    )


# ==================== Parser Fixtures ====================

@pytest.fixture
def python_parser():
    """Python parser для тестов."""
    return ParserFactory.create_parser(ProgrammingLanguage.PYTHON)


@pytest.fixture
def javascript_parser():
    """JavaScript parser для тестов."""
    return ParserFactory.create_parser(ProgrammingLanguage.JAVASCRIPT)


# ==================== Qwen Fixtures ====================

@pytest.fixture
def qwen_api_key() -> str:
    """Qwen API ключ из переменной окружения."""
    import os
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        pytest.skip("QWEN_API_KEY не установлен")
    return api_key


@pytest.fixture
def qwen_integration():
    """Qwen integration для тестов."""
    return QwenIntegration(
        base_url="https://ai.parma.ru/cloud/v1",
        api_key="",
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        max_tokens=8096
    )


# ==================== Mock Data ====================

@pytest.fixture
def mock_code_elements():
    """Mock code elements для тестов."""
    from code_rag.core.parser import CodeElement, CodeLocation

    elements = []

    # Function 1
    elements.append(CodeElement(
        name="authenticate_user",
        qualified_name="auth.authenticate_user",
        type=CodeElementType.FUNCTION,
        language=ProgrammingLanguage.PYTHON,
        source_code="def authenticate_user(username, password):\n    return True",
        signature="authenticate_user(username: str, password: str) -> bool",
        docstring="Authenticate user credentials.",
        location=CodeLocation(
            file_path=Path("auth.py"),
            start_line=10,
            end_line=15
        ),
        complexity=3
    ))

    # Function 2
    elements.append(CodeElement(
        name="verify_password",
        qualified_name="auth.verify_password",
        type=CodeElementType.FUNCTION,
        language=ProgrammingLanguage.PYTHON,
        source_code="def verify_password(password, hash):\n    return check_hash(password, hash)",
        signature="verify_password(password: str, hash: str) -> bool",
        docstring="Verify password against hash.",
        location=CodeLocation(
            file_path=Path("auth.py"),
            start_line=17,
            end_line=20
        ),
        complexity=2
    ))

    # Class
    elements.append(CodeElement(
        name="Database",
        qualified_name="db.Database",
        type=CodeElementType.CLASS,
        language=ProgrammingLanguage.PYTHON,
        source_code="class Database:\n    def connect(self):\n        pass",
        signature="Database",
        docstring="Database connection manager.",
        location=CodeLocation(
            file_path=Path("db.py"),
            start_line=5,
            end_line=10
        ),
        complexity=5
    ))

    return elements


@pytest.fixture
def mock_search_results(mock_code_elements):
    """Mock search results для тестов."""
    from code_rag.core.rag import SearchResult

    results = []
    for i, element in enumerate(mock_code_elements):
        results.append(SearchResult(
            element=element,
            score=0.9 - (i * 0.1),  # Decreasing scores
            rank=i+1,
            metadata={}
        ))

    return results


# ==================== Indexed Collection ====================

@pytest.fixture
def indexed_collection(
        vectorization_pipeline: VectorizationPipeline,
        sample_files: Dict[str, Path],
        test_collection: str
) -> str:
    """Коллекция с проиндексированными файлами."""
    # Индексируем все sample файлы
    for file_path in sample_files.values():
        vectorization_pipeline.process_file(
            file_path,
            repository_info={"repository_name": "test-repo"}
        )

    return test_collection


# ==================== Markers ====================

def pytest_configure(config):
    """Настройка pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_qdrant: marks tests that require Qdrant"
    )
    config.addinivalue_line(
        "markers", "requires_qwen: marks tests that require Qwen API"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )


# ==================== Pytest Hooks ====================

def pytest_collection_modifyitems(config, items):
    """Автоматическое добавление markers."""
    for item in items:
        # Автоматически добавляем marker для integration тестов
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Автоматически skip тесты требующие Qdrant если не доступен
        if "qdrant" in item.fixturenames:
            item.add_marker(pytest.mark.requires_qdrant)

        # Автоматически skip тесты требующие Qwen если нет API key
        if "qwen" in item.fixturenames or "qwen_integration" in item.fixturenames:
            item.add_marker(pytest.mark.requires_qwen)


# ==================== Utility Functions ====================

@pytest.fixture
def assert_vectors_similar():
    """Helper для сравнения векторов."""

    def _assert_similar(vec1: np.ndarray, vec2: np.ndarray, threshold: float = 0.01):
        """Проверка что векторы похожи."""
        distance = np.linalg.norm(vec1 - vec2)
        assert distance < threshold, f"Vectors too different: distance={distance}"

    return _assert_similar


@pytest.fixture
def create_temp_code_file(temp_dir: Path):
    """Helper для создания временных файлов с кодом."""

    def _create_file(filename: str, content: str, language: str = "python") -> Path:
        """Создать временный файл с кодом."""
        ext_map = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "csharp": ".cs",
            "kotlin": ".kt"
        }

        ext = ext_map.get(language, ".txt")
        file_path = temp_dir / f"{filename}{ext}"
        file_path.write_text(content)

        return file_path

    return _create_file


# ==================== Performance Fixtures ====================

@pytest.fixture
def benchmark_timer():
    """Helper для измерения производительности."""
    import time

    class Timer:
        def __init__(self):
            self.times = {}

        def start(self, name: str):
            self.times[name] = {"start": time.time()}

        def stop(self, name: str):
            if name in self.times:
                self.times[name]["end"] = time.time()
                self.times[name]["duration"] = (
                        self.times[name]["end"] - self.times[name]["start"]
                )

        def get_duration(self, name: str) -> float:
            return self.times.get(name, {}).get("duration", 0.0)

        def assert_faster_than(self, name: str, max_seconds: float):
            duration = self.get_duration(name)
            assert duration < max_seconds, (
                f"{name} took {duration:.2f}s, expected < {max_seconds}s"
            )

    return Timer()


# ==================== Cleanup ====================

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_collections(qdrant_url: str):
    """Очистка всех тестовых коллекций после сессии."""
    yield

    # После всех тестов удаляем тестовые коллекции
    try:
        client = QdrantClient(url=qdrant_url)
        collections = client.list_collections()

        for collection_name in collections:
            if collection_name.startswith("test_collection_"):
                try:
                    client.delete_collection(collection_name)
                except:
                    pass
    except:
        pass


# ==================== Skip Conditions ====================

@pytest.fixture
def skip_if_no_gpu():
    """Skip тест если нет GPU."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def skip_if_slow(request):
    """Skip медленные тесты если не указано --runslow."""
    if request.config.getoption("--runslow") is False:
        pytest.skip("need --runslow option to run")


def pytest_addoption(parser):
    """Добавление опций командной строки."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests"
    )


@pytest.fixture
def mock_repo_manager(temp_dir):
    """Mock RepositoryManager для тестов."""
    from unittest.mock import MagicMock

    manager = MagicMock()
    manager.base_path = temp_dir
    manager.metadata_path = temp_dir / ".metadata"
    manager.repositories = {}

    return manager

# Исправленный __init__ метод
def _validate_config(self):
    """Валидация критических параметров."""
    if self.qdrant_port <= 0 or self.qdrant_port > 65535:
        raise ValueError(f"Невалидный qdrant_port: {self.qdrant_port}")
    if self.vector_size != self.nomic_dimension:
        logger.warning(f"vector_size ({self.vector_size}) != nomic_dimension ({self.nomic_dimension})")

# Добавляем в конец файла
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._validate_config()
    self._create_directories()
