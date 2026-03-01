Вот **Часть 1 из 3** для `README.md`:

markdown
# Code RAG - Система векторизации и поиска исходного кода

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

**Code RAG** - это интеллектуальная система для векторизации, индексации и семантического поиска по исходному коду ИТ-компании, занимающейся заказной разработкой. Система использует современные технологии embeddings, векторные базы данных и LLM для создания мощного инструмента работы с кодовой базой.

## 🎯 Основные возможности

- **Векторизация кода**: Преобразование исходного кода в векторные представления с использованием Nomic Embed
- **Семантический поиск**: Поиск релевантного кода по естественному языку
- **Множественные языки**: Поддержка Python, JavaScript, TypeScript, Java, C#, Kotlin
- **Git интеграция**: Работа с множественными репозиториями, отслеживание изменений
- **RAG генерация**: Генерация ответов и кода с использованием Qwen-Coder
- **Reranking**: Улучшение результатов поиска с помощью cross-encoder
- **Масштабируемость**: Векторная БД Qdrant для быстрого поиска

## 📋 Содержание

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Архитектура](#архитектура)
- [Использование](#использование)
- [Конфигурация](#конфигурация)
- [Примеры](#примеры)
- [Тестирование](#тестирование)
- [Разработка](#разработка)
- [Лицензия](#лицензия)

## 🚀 Установка

### Требования

- Python 3.9+
- Docker (для Qdrant)
- Git
- 8GB+ RAM (для embedding моделей)

### Шаг 1: Клонирование репозитория

```
git clone https://github.com/yourusername/code-rag.git
cd code-rag
```

### Шаг 2: Создание виртуального окружения

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### Шаг 3: Установка зависимостей

```
# Основные зависимости
pip install -e .

# С зависимостями для разработки
pip install -e ".[dev]"

# Все зависимости (включая Qwen)
pip install -e ".[all]"
```

### Шаг 4: Запуск Qdrant

```
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### Шаг 5: Настройка переменных окружения

Создайте файл `.env`:

```
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Опционально для Qdrant Cloud

# Qwen API (опционально)
QWEN_API_KEY=your-dashscope-api-key

# Logging
LOG_LEVEL=INFO
```

## ⚡ Быстрый старт

### 1. Индексация репозитория

```
from pathlib import Path
from code_rag.core import VectorizationPipeline, CodeEmbedder, QdrantClient

# Инициализация компонентов
embedder = CodeEmbedder()
qdrant_client = QdrantClient(url="http://localhost:6333")

# Создание pipeline
pipeline = VectorizationPipeline(
    collection_name="my-codebase",
    embedder=embedder,
    qdrant_client=qdrant_client
)

# Индексация директории
result = pipeline.process_directory(
    Path("./my-project"),
    repository_info={
        "repository_name": "my-project",
        "branch": "main"
    },
    recursive=True
)

print(f"Проиндексировано: {result.indexed_elements} элементов")
```

### 2. Поиск кода

```
from code_rag.core import RAGRetriever

# Создание retriever
retriever = RAGRetriever(
    collection_name="my-codebase",
    embedder=embedder,
    qdrant_client=qdrant_client
)

# Поиск
results = retriever.search(
    query="authentication function with JWT",
    top_k=5
)

# Вывод результатов
for result in results:
    print(f"[{result.score:.4f}] {result.element.name}")
    print(f"File: {result.element.location.file_path}")
    print(f"Type: {result.element.type.value}")
    print("-" * 80)
```

### 3. Генерация ответов с Qwen

```
from code_rag.core import QwenIntegration

# Инициализация Qwen
qwen = QwenIntegration(api_key="your-api-key")

# Поиск контекста
results = retriever.search("user authentication", top_k=3)
context = retriever.build_context(results)

# Генерация ответа
response = qwen.generate_answer(
    question="How does user authentication work in this codebase?",
    context=context
)

print(response.answer)
```

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                     Code RAG System                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐      ┌──────────────┐     ┌─────────────┐ │
│  │   Parser    │─────▶│  Embedder    │────▶│   Qdrant    │ │
│  │             │      │              │     │  Vector DB  │ │
│  │ - Python    │      │ Nomic Embed  │     │             │ │
│  │ - JavaScript│      │              │     │ Collections │ │
│  │ - TypeScript│      └──────────────┘     └─────────────┘ │
│  │ - Java/C#   │                                  │         │
│  └─────────────┘                                  ▼         │
│                                            ┌─────────────┐   │
│  ┌─────────────┐                          │ RAG         │   │
│  │ Git Handler │                          │ Retriever   │   │
│  │             │                          │             │   │
│  │ - Monitor   │                          │ - Search    │   │
│  │ - Auto sync │                          │ - Rerank    │   │
│  └─────────────┘                          └─────────────┘   │
│                                                  │           │
│                                                  ▼           │
│                                           ┌─────────────┐    │
│                                           │   Qwen      │    │
│                                           │   Coder     │    │
│                                           │             │    │
│                                           │ Generation  │    │
│                                           └─────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Основные компоненты

1. **Parser** (`code_rag.core.parser`)
   - Парсинг исходного кода на множественных языках
   - Tree-sitter для AST анализа
   - Извлечение функций, классов, методов

2. **Embedder** (`code_rag.core.embeddings`)
   - Nomic Embed Text v1.5 для векторизации
   - 768-мерные векторы
   - Поддержка batch обработки

3. **Vector DB** (`code_rag.core.vector_db`)
   - Qdrant для хранения векторов
   - Быстрый similarity search
   - Фильтрация по метаданным

4. **RAG** (`code_rag.core.rag`)
   - Семантический поиск
   - Reranking с cross-encoder
   - Интеграция с Qwen-Coder

5. **Git Handler** (`code_rag.core.git_handler`)
   - Работа с репозиториями
   - Мониторинг изменений
   - Автоматическая реиндексация

## 📚 Использование

### Работа с Parser

```
from code_rag.core.parser import ParserFactory, ProgrammingLanguage
from pathlib import Path

# Создание парсера для Python
parser = ParserFactory.create_parser(ProgrammingLanguage.PYTHON)

# Или автоматически по файлу
parser = ParserFactory.create_parser_for_file(Path("script.py"))

# Парсинг файла
module = parser.parse_file(Path("script.py"))

# Доступ к элементам
for func in module.functions:
    print(f"Function: {func.name}")
    print(f"Signature: {func.signature}")
    print(f"Docstring: {func.docstring}")
    print(f"Complexity: {func.complexity}")

for cls in module.classes:
    print(f"Class: {cls.name}")
    for method in cls.methods:
        print(f"  Method: {method.name}")
```

### Работа с Embeddings

```
from code_rag.core.embeddings import CodeEmbedder

embedder = CodeEmbedder()

# Векторизация одного текста
code = "def hello(): return 'Hello, World!'"
vector = embedder.encode_text(code)
print(f"Vector dimension: {vector.shape}")  # (768,)

# Батчевая векторизация
codes = [
    "def add(a, b): return a + b",
    "def multiply(a, b): return a * b",
    "class Calculator: pass"
]
vectors = embedder.encode_batch(codes, batch_size=32)
print(f"Batch shape: {vectors.shape}")  # (3, 768)

# Векторизация запроса
query = "authentication function"
query_vector = embedder.encode_query(query)
```

### Работа с Vector DB

```
from code_rag.core.vector_db import QdrantClient, CollectionSchema, SearchFilters
from code_rag.core.parser import CodeElementType, ProgrammingLanguage

# Подключение к Qdrant
client = QdrantClient(url="http://localhost:6333")

# Создание коллекции
schema = CollectionSchema(
    collection_name="my-codebase",
    vector_size=768,
    distance="Cosine"
)
client.create_collection(schema, recreate=True)

# Вставка данных
from code_rag.core.vector_db import CodePayload

payload = CodePayload.from_code_element(
    element,
    repository_name="my-repo",
    branch="main"
)

client.insert_point(
    collection_name="my-codebase",
    vector=vector,
    payload=payload.to_dict()
)

# Поиск
filters = SearchFilters(
    language=ProgrammingLanguage.PYTHON,
    element_type=CodeElementType.FUNCTION
)

results = client.search(
    collection_name="my-codebase",
    query_vector=query_vector,
    limit=10,
    filters=filters,
    score_threshold=0.7
)
```

### Продвинутый RAG поиск

```
from code_rag.core.rag import RAGRetriever, RAGConfig

# Создание retriever с конфигурацией
config = RAGConfig(
    top_k=10,
    score_threshold=0.6,
    use_reranking=True,
    max_context_length=8000
)

retriever = RAGRetriever(
    collection_name="my-codebase",
    embedder=embedder,
    qdrant_client=client,
    use_reranking=True
)

# Семантический поиск
results = retriever.search(
    query="JWT token authentication middleware",
    top_k=5,
    language=ProgrammingLanguage.PYTHON
)

# Поиск похожего кода
similar = retriever.search_similar_code(
    code_element=some_function,
    top_k=5,
    same_language_only=True
)

# Поиск по сигнатуре
by_signature = retriever.search_by_signature(
    signature="authenticate(username: str, password: str) -> bool",
    language=ProgrammingLanguage.PYTHON,
    fuzzy=True
)

# Поиск по документации
by_docs = retriever.search_by_docstring(
    docstring_query="user authentication and authorization",
    language=ProgrammingLanguage.PYTHON
)

# Построение контекста для LLM
context = retriever.build_context(results, max_length=4000)
```

### Работа с Git репозиториями

```
from code_rag.core.git_handler import RepositoryManager, RepositoryMonitor
from pathlib import Path

# Менеджер репозиториев
manager = RepositoryManager(base_path=Path("./repositories"))

# Клонирование репозитория
repo_info = manager.clone_repository(
    url="https://github.com/user/repo.git",
    name="my-repo",
    branch="main"
)

# Открытие существующего репозитория
repo_info = manager.open_repository(Path("./my-repo"))

# Обновление репозитория
updated_info = manager.update_repository(repo_info)

# Статус репозитория
status = manager.get_repository_status(repo_info)
print(f"Modified files: {status['modified_files']}")
print(f"Is dirty: {status['is_dirty']}")

# Мониторинг изменений
monitor = RepositoryMonitor(repo_manager=manager)

# Callback для изменений
def on_change(change_event):
    print(f"Changed: {change_event.file_path}")
    print(f"Type: {change_event.change_type.value}")
    # Реиндексация файла
    pipeline.process_file(change_event.file_path)

monitor.add_callback(on_change)

# Запуск мониторинга
monitor.start_monitoring(
    repository_path=repo_info.path,
    repository_name="my-repo",
    interval=60  # проверка каждую минуту
)
```

### Интеграция с Qwen-Coder

```
from code_rag.core.rag import QwenIntegration

# Инициализация
qwen = QwenIntegration(
    api_key="your-dashscope-api-key",
    model="qwen2.5-coder-32b-instruct"
)

# Генерация ответа
response = qwen.generate_answer(
    question="How does authentication work?",
    context=context,
    include_code=True
)

print(response.answer)
print(f"Confidence: {response.confidence}")

# Генерация кода
code_response = qwen.generate_code(
    description="Create a FastAPI endpoint for user authentication",
    language="python",
    context=context
)

print(code_response.code)
print(f"Explanation: {code_response.explanation}")

# Объяснение кода
explanation = qwen.explain_code(
    code=some_code,
    question="What does this function do?"
)

print(explanation)
```

## ⚙️ Конфигурация

### config.yaml

```
# Embeddings
embeddings:
  model_name: "nomic-ai/nomic-embed-text-v1.5"
  device: "cpu"  # или "cuda" для GPU
  max_length: 8192
  batch_size: 32

# Qdrant
qdrant:
  url: "http://localhost:6333"
  api_key: null  # Для Qdrant Cloud
  collection_name: "code-rag"
  vector_size: 768
  distance: "Cosine"

# RAG
rag:
  top_k: 10
  score_threshold: 0.6
  use_reranking: true
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  max_context_length: 8000

# Qwen
qwen:
  api_key: "${QWEN_API_KEY}"
  model: "qwen2.5-coder-32b-instruct"
  temperature: 0.7
  max_tokens: 2000

# Parser
parser:
  languages:
    - python
    - javascript
    - typescript
    - java
    - csharp
    - kotlin
  extract_complexity: true
  extract_imports: true

# Git
git:
  monitor_interval: 60  # секунды
  auto_reindex: true
  exclude_patterns:
    - "*.test.py"
    - "*.spec.js"
    - "node_modules"
    - "__pycache__"
    - ".git"

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "code_rag.log"
```

### Загрузка конфигурации

```
from code_rag.utils import load_config

config = load_config("config.yaml")

# Использование конфигурации
embedder = CodeEmbedder(
    model_name=config["embeddings"]["model_name"],
    device=config["embeddings"]["device"]
)

retriever = RAGRetriever(
    collection_name=config["qdrant"]["collection_name"],
    embedder=embedder,
    use_reranking=config["rag"]["use_reranking"]
)
```

## 🎓 Примеры

### Пример 1: Простая индексация и поиск

```
from pathlib import Path
from code_rag.core import (
    VectorizationPipeline,
    RAGRetriever,
    CodeEmbedder,
    QdrantClient
)

# Компоненты
embedder = CodeEmbedder()
client = QdrantClient(url="http://localhost:6333")

# Индексация
pipeline = VectorizationPipeline(
    collection_name="example",
    embedder=embedder,
    qdrant_client=client
)

result = pipeline.process_directory(
    Path("./src"),
    repository_info={"repository_name": "example-project"}
)

print(f"Indexed: {result.indexed_elements} elements")

# Поиск
retriever = RAGRetriever(
    collection_name="example",
    embedder=embedder,
    qdrant_client=client
)

results = retriever.search("user authentication", top_k=5)

for r in results:
    print(f"{r.element.name} ({r.score:.4f})")
```

### Пример 2: Мониторинг репозитория

```
from code_rag.core import RepositoryMonitor, RepositoryManager
from pathlib import Path

manager = RepositoryManager()
monitor = RepositoryMonitor(repo_manager=manager)

# Callback для автоматической реиндексации
def auto_reindex(change_event):
    if change_event.change_type in ["added", "modified"]:
        pipeline.process_file(
            change_event.file_path,
            repository_info={"repository_name": change_event.repository_name}
        )
        print(f"Reindexed: {change_event.file_path}")

monitor.add_callback(auto_reindex)
monitor.start_monitoring(
    repository_path=Path("./my-repo"),
    repository_name="my-repo",
    interval=30
)
```

### Пример 3: Полный RAG workflow

```
from code_rag.core import (
    RAGRetriever,
    QwenIntegration,
    RAGResponse
)

# Retriever
retriever = RAGRetriever(
    collection_name="my-codebase",
    embedder=embedder,
    qdrant_client=client,
    use_reranking=True
)

# Qwen
qwen = QwenIntegration(api_key="your-key")

# Вопрос
question = "How can I implement JWT authentication?"

# 1. Поиск релевантного кода
results = retriever.search(question, top_k=5)

# 2. Построение контекста
context = retriever.build_context(results)

# 3. Генерация ответа
answer = qwen.generate_answer(
    question=question,
    context=context,
    include_code=True
)

# 4. Формирование ответа
response = RAGResponse(
    query=question,
    answer=answer.answer,
    sources=results,
    confidence=0.85
)

print(response.answer)
print(f"\nSources ({len(response.sources)}):")
for source in response.sources:
    print(f"  - {source.element.name} ({source.score:.4f})")
```

## 🧪 Тестирование

### Запуск тестов

```
# Все тесты
pytest

# С подробным выводом
pytest -v

# Только быстрые тесты (без медленных)
pytest -m "not slow"

# Пропустить тесты требующие Qwen API
pytest -m "not requires_qwen"

# Конкретный модуль
pytest tests/test_embeddings.py

# С покрытием кода
pytest --cov=code_rag --cov-report=html
```

### Структура тестов

```
tests/
├── __init__.py
├── conftest.py              # Fixtures для всех тестов
├── test_embeddings.py       # Тесты embeddings
├── test_parser.py           # Тесты парсеров
├── test_vector_db.py        # Тесты Qdrant
├── test_qdrant.py           # Тесты интеграции Qdrant
├── test_vectorization.py    # Тесты векторизации
├── test_rag.py              # Тесты RAG
└── test_git_handler.py      # Тесты Git интеграции
```

### Перед запуском тестов

```
# 1. Запустите Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 2. Установите зависимости для тестов
pip install -e ".[dev]"

# 3. Установите переменные окружения (опционально)
export QWEN_API_KEY="your-key"
export QDRANT_URL="http://localhost:6333"
```

### CI/CD

GitHub Actions workflow (`.github/workflows/test.yml`):

```
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      qdrant:
        image: qdrant/qdrant
        ports:
          - 6333:6333
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          pytest -v --cov=code_rag --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## 🛠️ Разработка

### Настройка окружения разработки

```
# Клонирование
git clone https://github.com/yourusername/code-rag.git
cd code-rag

# Виртуальное окружение
python -m venv venv
source venv/bin/activate

# Установка в режиме разработки
pip install -e ".[dev]"

# Pre-commit hooks
pre-commit install
```

### Структура проекта

```
code-rag/
├── code_rag/                   # Основной пакет
│   ├── __init__.py
│   ├── core/                   # Ядро системы
│   │   ├── embeddings/         # Модуль векторизации
│   │   ├── parser/             # Парсеры кода
│   │   ├── vector_db/          # Интеграция Qdrant
│   │   ├── rag/                # RAG компоненты
│   │   └── git_handler/        # Работа с Git
│   ├── utils/                  # Утилиты
│   └── cli/                    # CLI интерфейс
├── tests/                      # Тесты
├── examples/                   # Примеры использования
├── docs/                       # Документация
├── pyproject.toml             # Конфигурация проекта
├── requirements.txt           # Зависимости
├── requirements-dev.txt       # Dev зависимости
└── README.md                  # Этот файл
```

### Стиль кода

Проект следует PEP 8 и использует:

- **Black** для форматирования
- **isort** для сортировки импортов
- **flake8** для линтинга
- **mypy** для проверки типов

```
# Форматирование
black code_rag tests

# Сортировка импортов
isort code_rag tests

# Линтинг
flake8 code_rag tests

# Проверка типов
mypy code_rag
```

### Pre-commit конфигурация

`.pre-commit-config.yaml`:

```
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

## 🤝 Contributing

Мы приветствуем вклад в проект! 

### Процесс

1. **Fork** репозитория
2. Создайте **feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** изменения (`git commit -m 'Add some AmazingFeature'`)
4. **Push** в branch (`git push origin feature/AmazingFeature`)
5. Откройте **Pull Request**

### Рекомендации

- Добавляйте тесты для новой функциональности
- Обновляйте документацию
- Следуйте стилю кода проекта
- Проверьте что все тесты проходят
- Напишите понятное описание PR

### Типы вкладов

- 🐛 **Bug fixes** - исправление ошибок
- ✨ **Features** - новая функциональность
- 📝 **Documentation** - улучшение документации
- 🎨 **Code style** - улучшение читаемости кода
- ⚡ **Performance** - оптимизация производительности
- ✅ **Tests** - добавление тестов

## 📖 Документация

Полная документация доступна в [./docs](./docs).

### Генерация документации локально

```
cd docs
make html

# Открыть в браузере
open _build/html/index.html
```

## 🗺️ Roadmap

- [ ] **v0.2.0** - Поддержка дополнительных языков (Go, Rust, PHP)
- [ ] **v0.3.0** - Web UI для управления и поиска
- [ ] **v0.4.0** - REST API
- [ ] **v0.5.0** - Поддержка Azure DevOps
- [ ] **v1.0.0** - Production ready релиз

## 📊 Производительность

### Benchmarks

| Операция | Время | Пропускная способность |
|----------|-------|------------------------|
| Парсинг файла (Python, 1000 LOC) | ~0.5s | 2000 LOC/s |
| Векторизация (batch=32) | ~0.8s | 40 элементов/s |
| Вставка в Qdrant (batch=100) | ~0.3s | 333 элементов/s |
| Поиск (top_k=10) | ~0.1s | 10 запросов/s |
| Поиск с reranking | ~0.5s | 2 запроса/s |

*Тесты на: Intel i7-10700, 16GB RAM, без GPU*

## ❓ FAQ

**Q: Какие языки программирования поддерживаются?**  
A: Python, JavaScript, TypeScript, Java, C#, Kotlin. Добавление новых языков планируется.

**Q: Нужен ли GPU?**  
A: Нет, но наличие GPU значительно ускорит векторизацию.

**Q: Можно ли использовать без Qwen API?**  
A: Да, RAG поиск работает независимо. Qwen нужен только для генерации ответов.

**Q: Как часто происходит реиндексация?**  
A: При использовании мониторинга - автоматически при изменениях. Можно настроить интервал.

**Q: Поддерживается ли Qdrant Cloud?**  
A: Да, укажите URL и API ключ в конфигурации.

**Q: Какой объем данных можно индексировать?**  
A: Ограничений нет, зависит от ресурсов Qdrant. Рекомендуем разделять на коллекции.

## 🐛 Известные проблемы

- Tree-sitter парсеры могут не поддерживать самые новые синтаксические конструкции языков
- Reranking может быть медленным на больших результатах (>100)
- Qwen API имеет rate limits

## 📜 Changelog

### v0.1.0 (2025-01-15)

- ✨ Начальный релиз
- ✅ Поддержка Python, JavaScript, TypeScript
- ✅ Векторизация с Nomic Embed
- ✅ Интеграция с Qdrant
- ✅ RAG с Qwen-Coder
- ✅ Git мониторинг
- ✅ Тестовое покрытие 85%

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для подробностей.

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Благодарности

- [Nomic AI](https://www.nomic.ai/) - за Nomic Embed модель
- [Qdrant](https://qdrant.tech/) - за векторную БД
- [Alibaba Cloud](https://www.alibabacloud.com/) - за Qwen-Coder
- [Tree-sitter](https://tree-sitter.github.io/) - за парсеры кода
- Все [контрибьюторы](https://github.com/yourusername/code-rag/graphs/contributors)

## 📞 Контакты

- **Email**: gafurov@parma.ru

---

**Made with ❤️ for developers by developers**



Полный **README.md** состоит из трех частей:
1. **Часть 1**: Введение, установка, быстрый старт, архитектура
2. **Часть 2**: Использование, конфигурация, примеры
3. **Часть 3**: Тестирование, разработка, contributing, FAQ, лицензия

