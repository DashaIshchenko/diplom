
# Architecture Documentation

## Обзор системы

Code RAG - это система для индексации, поиска и генерации ответов на основе кодовой базы с использованием технологий RAG (Retrieval-Augmented Generation).

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   FastAPI REST API                                    │   │
│  │   - /embeddings  - /search  - /generation             │   │
│  │   - /indexing    - /repositories  - /health           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Core Layer                            │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Parser   │  │Embeddings│  │Vector DB │  │    RAG    │  │
│  │           │  │          │  │          │  │           │  │
│  │ -  Python  │  │ -  Nomic  │  │ -  Qdrant │  │ -  Retrie- │  │
│  │ -  Java    │  │   Embed  │  │ -  Schema │  │   ver     │  │
│  │ -  JS/TS   │  │ -  Custom │  │ -  Index  │  │ -  Qwen    │  │
│  │ -  C#      │  │   Models │  │ -  Vector │  │   Integr. │  │
│  │ -  Kotlin  │  │          │  │   Search │  │           │  │
│  │ -  HTML    │  │          │  │          │  │           │  │
│  │ -  CSS     │  │          │  │          │  │           │  │
│  └───────────┘  └──────────┘  └──────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Qdrant     │  │   Qwen API   │  │   Storage    │      │
│  │   Vector DB  │  │   (LLM)      │  │   (Logs)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Компоненты

### 1. Parser Module (`code_rag/core/parser`)

**Назначение:** Парсинг исходного кода на различных языках программирования.

**Структура:**
- `base.py` - Базовый класс `BaseParser`
- `structures.py` - Структуры данных (`CodeElement`, `ModuleInfo`, `ParseResult`)
- `factory.py` - Фабрика для создания парсеров
- `parsers/` - Реализации для каждого языка

**Поддерживаемые языки:**
- Python (tree-sitter)
- Java
- JavaScript
- TypeScript
- C#
- Kotlin
- HTML
- CSS

**Workflow:**
```
parser = ParserFactory.create_parser_for_file(file_path)
module = parser.parse_file(file_path)
elements = module.all_elements  # Классы, функции, методы
```

### 2. Embeddings Module (`code_rag/core/embeddings`)

**Назначение:** Векторизация кода в embedding vectors.

**Компоненты:**
- `base.py` - Базовая абстракция `BaseEmbeddingModel`
- `nomic_embed.py` - Реализация Nomic Embed
- `factory.py` - Фабрика моделей
- `code_embedder.py` - Обертка для pipeline

**Характеристики:**
- Модель: Nomic Embed Text v1.5
- Размерность: 768
- Батчевая обработка
- Нормализация векторов

**Использование:**
```
embedder = CodeEmbedder()
embedding = embedder.encode_text("def hello(): pass")
batch = embedder.encode_batch(["code1", "code2"])
```

### 3. Vector DB Module (`code_rag/core/vector_db`)

**Назначение:** Работа с Qdrant векторной базой данных.

**Компоненты:**
- `qdrant_client.py` - Клиент для Qdrant
- `schemas.py` - Схемы коллекций и данных
- `vectorization_pipeline.py` - Pipeline индексации

**Схема данных:**
```
CodePayload:
  - name: str
  - qualified_name: str
  - type: CodeElementType
  - language: ProgrammingLanguage
  - source_code: str
  - signature: Optional[str]
  - docstring: Optional[str]
  - file_path: str
  - start_line: int
  - end_line: int
  - complexity: int
  - repository_name: str
  - branch: str
  - indexed_at: str
```

**Workflow индексации:**
```
File → Parser → CodeElements → Embedder → Vectors → Qdrant
```

### 4. RAG Module (`code_rag/core/rag`)

**Назначение:** Retrieval и генерация ответов.

**Компоненты:**
- `rag_retriever.py` - Поиск и retrieval
- `qwen_integration.py` - Интеграция с Qwen Coder

**RAG Pipeline:**
```
1. Query → Embedder → Query Vector
2. Query Vector → Qdrant Search → Top-K Results
3. Results + Query → Qwen → Generated Answer
```

**Особенности:**
- Семантический поиск
- Фильтрация (язык, тип, репозиторий)
- Reranking (опционально)
- Context window management
- Prompt engineering для кода

### 5. API Module (`code_rag/api`)

**Назначение:** REST API для взаимодействия с системой.

**Endpoints:**

**Health & Info:**
- `GET /health` - Health check
- `GET /info` - Информация о системе
- `GET /version` - Версия API

**Embeddings:**
- `POST /embeddings/embed` - Векторизация текста
- `POST /embeddings/embed/batch` - Батч векторизация
- `POST /embeddings/similarity` - Вычисление схожести

**Indexing:**
- `POST /indexing/file` - Индексация файла
- `POST /indexing/directory` - Индексация директории
- `POST /indexing/upload` - Загрузка и индексация
- `POST /indexing/collection/create` - Создание коллекции
- `GET /indexing/collection/status` - Статус коллекции

**Search:**
- `POST /search/` - Поиск кода
- `GET /search/similar/{id}` - Похожие элементы
- `GET /search/by-file` - Поиск по файлу

**Generation:**
- `POST /generation/generate` - Генерация с RAG
- `POST /generation/explain` - Объяснение кода
- `POST /generation/improve` - Улучшение кода
- `POST /generation/chat` - Чат с контекстом

**Repositories:**
- `POST /repositories/index` - Индексация репозитория
- `GET /repositories/list` - Список репозиториев
- `GET /repositories/{name}/info` - Информация
- `GET /repositories/{name}/statistics` - Статистика
- `DELETE /repositories/{name}` - Удаление

## Data Flow

### Индексация
```
1. Repository Path
   ↓
2. File Discovery (recursive walk)
   ↓
3. For each file:
   a. Parser.parse_file() → CodeElements
   b. Embedder.encode() → Vectors
   c. QdrantClient.insert() → Storage
   ↓
4. VectorizationResult (stats)
```

### Поиск
```
1. User Query
   ↓
2. Embedder.encode_query() → Query Vector
   ↓
3. QdrantClient.search(vector, filters) → Results
   ↓
4. [Optional] Reranking → Top Results
   ↓
5. SearchResult[] (with scores)
```

### RAG Generation
```
1. User Question
   ↓
2. Retriever.search() → Context (Code Elements)
   ↓
3. Build Prompt:
   - System instruction
   - Context (code snippets)
   - User question
   ↓
4. Qwen.generate() → Answer
   ↓
5. QwenResponse (content, tokens)
```

## Deployment

### Docker Compose
```
services:
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    
  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [qdrant]
    environment:
      - QDRANT_URL=http://qdrant:6333
      - QWEN_API_KEY=${QWEN_API_KEY}
```

### Environment Variables
```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=optional
COLLECTION_NAME=code_collection
QWEN_API_KEY=your-api-key
QWEN_MODEL=qwen2.5-coder-32b-instruct
LOG_LEVEL=INFO
```

## Performance

### Характеристики:
- **Парсинг:** ~100-500 файлов/сек (зависит от размера)
- **Векторизация:** ~1000 элементов/мин (batch=32)
- **Поиск:** <100ms (top_k=10)
- **Генерация:** 2-5 сек (зависит от контекста)

### Оптимизации:
- Батчевая обработка embeddings
- Кэширование векторов
- Асинхронная индексация
- HNSW индекс в Qdrant
- Parallel processing файлов

## Scalability

### Horizontal Scaling:
- Multiple API instances (load balancer)
- Qdrant cluster (sharding)
- Distributed embedder workers

### Vertical Scaling:
- GPU для embeddings
- Больше RAM для Qdrant
- SSD для faster I/O

## Security

- API key authentication
- Rate limiting
- Input validation
- Sanitization кода
- CORS настройки

## Monitoring

- Health checks
- Prometheus metrics (optional)
- Logging (structured)
- Error tracking

## Future Enhancements

1. **Больше языков:** Go, Rust, PHP, Ruby
2. **Hybrid search:** BM25 + Semantic
3. **Fine-tuned models:** Специализация на код
4. **Code generation:** Генерация кода с нуля
5. **Test generation:** Автогенерация тестов
6. **Documentation generation:** Автодокументация
7. **Code review:** AI code review
8. **Refactoring suggestions:** Предложения рефакторинга


Эта архитектурная документация описывает полную структуру системы Code RAG, включая все модули, компоненты, data flow, deployment и характеристики производительности.