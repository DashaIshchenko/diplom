# API Reference

## Обзор

Code RAG API предоставляет RESTful эндпоинты для индексации, поиска и генерации ответов на основе кодовой базы.

**Base URL:** `http://localhost:8000`

**Authentication:** API key (опционально)

**Content-Type:** `application/json`

---

## Health & Status

### GET /health

Проверка здоровья всех компонентов системы.

**Response:**
```
{
  "status": "healthy",
  "timestamp": "2025-10-15T12:00:00",
  "version": "0.1.0",
  "components": {
    "embedder": {
      "status": "healthy",
      "model": "NomicEmbedModel",
      "dimension": 768
    },
    "qdrant": {
      "status": "healthy",
      "url": "http://localhost:6333"
    },
    "collection": {
      "status": "healthy",
      "name": "code_collection",
      "points_count": 1234
    }
  }
}
```

### GET /info

Информация о системе.

**Response:**
```
{
  "version": "0.1.0",
  "embedder_model": "NomicEmbedModel",
  "embedding_dimension": 768,
  "qdrant_url": "http://localhost:6333",
  "collection_name": "code_collection",
  "components_status": {
    "embedder": "healthy",
    "qdrant": "healthy",
    "collection": "exists"
  }
}
```

### GET /version

Версия API.

**Response:**
```
{
  "version": "0.1.0",
  "title": "Code RAG API"
}
```

---

## Embeddings

### POST /embeddings/embed

Векторизация текста кода.

**Request:**
```
{
  "text": "def calculate_sum(a, b): return a + b",
  "normalize": true
}
```

**Response:**
```
{
  "embedding": [0.123, -0.456, 0.789, ...],
  "dimension": 768,
  "model": "NomicEmbedModel"
}
```

### POST /embeddings/embed/batch

Батчевая векторизация.

**Request:**
```
{
  "texts": [
    "def add(a, b): return a + b",
    "def subtract(a, b): return a - b"
  ],
  "batch_size": 32,
  "normalize": true
}
```

**Response:**
```
{
  "embeddings": [
    [0.123, -0.456, ...],
    [0.321, 0.654, ...]
  ],
  "count": 2,
  "dimension": 768,
  "model": "NomicEmbedModel"
}
```

### POST /embeddings/similarity

Вычисление схожести между двумя текстами.

**Request:**
```
{
  "text1": "def add(a, b): return a + b",
  "text2": "def sum_numbers(x, y): return x + y"
}
```

**Response:**
```
{
  "similarity": 0.92,
  "text1_length": 32,
  "text2_length": 38
}
```

### GET /embeddings/model/info

Информация о модели эмбеддингов.

**Response:**
```
{
  "model_name": "NomicEmbedModel",
  "embedding_dimension": 768,
  "max_sequence_length": 8192,
  "device": "cpu"
}
```

---

## Indexing

### POST /indexing/file

Индексация одного файла.

**Request:**
```
{
  "file_path": "/path/to/file.py",
  "repository_name": "my-project"
}
```

**Response:**
```
{
  "status": "completed",
  "total_files": 1,
  "parsed_files": 1,
  "failed_files": 0,
  "total_elements": 15,
  "indexed_elements": 15,
  "success_rate": 100.0,
  "errors": []
}
```

### POST /indexing/directory

Индексация директории.

**Request:**
```
{
  "directory_path": "/path/to/project",
  "repository_name": "my-project",
  "recursive": true
}
```

**Response:**
```
{
  "status": "completed",
  "total_files": 50,
  "parsed_files": 48,
  "failed_files": 2,
  "total_elements": 532,
  "indexed_elements": 520,
  "success_rate": 96.0,
  "errors": [
    {"file": "broken.py", "error": "SyntaxError"}
  ]
}
```

### POST /indexing/upload

Загрузка и индексация файла.

**Request:**
```
Content-Type: multipart/form-data

file: <file_content>
repository_name: my-project (optional)
```

**Response:**
```
{
  "status": "completed",
  "total_files": 1,
  "parsed_files": 1,
  "failed_files": 0,
  "total_elements": 10,
  "indexed_elements": 10,
  "success_rate": 100.0,
  "errors": []
}
```

### GET /indexing/collection/status

Статус коллекции.

**Response:**
```
{
  "name": "code_collection",
  "exists": true,
  "points_count": 1234,
  "vectors_count": 1234,
  "vector_size": 768
}
```

### POST /indexing/collection/create

Создание коллекции.

**Query Parameters:**
- `recreate` (bool) - Пересоздать если существует (default: false)

**Response:**
```
{
  "status": "created",
  "collection_name": "code_collection"
}
```

### DELETE /indexing/collection

Удаление коллекции.

**Response:**
```
{
  "status": "deleted",
  "collection_name": "code_collection"
}
```

---

## Search

### POST /search/

Поиск кода по запросу.

**Request:**
```
{
  "query": "authentication function",
  "top_k": 10,
  "language": "python",
  "element_type": "function",
  "repository_name": "my-project",
  "min_score": 0.7
}
```

**Response:**
```
{
  "results": [
    {
      "score": 0.92,
      "element": {
        "name": "authenticate_user",
        "qualified_name": "auth.authenticate_user",
        "type": "function",
        "language": "python",
        "source_code": "def authenticate_user(username, password):\n    ...",
        "signature": "authenticate_user(username: str, password: str) -> bool",
        "docstring": "Authenticate user credentials.",
        "file_path": "auth.py",
        "start_line": 10,
        "end_line": 25,
        "complexity": 5,
        "repository_name": "my-project"
      }
    }
  ],
  "total": 1,
  "query": "authentication function"
}
```

### GET /search/similar/{element_id}

Поиск похожих элементов кода.

**Query Parameters:**
- `top_k` (int) - Количество результатов (default: 10)

**Response:**
```
{
  "results": [...],
  "total": 5,
  "query": "similar_to:abc123"
}
```

### POST /search/semantic

Семантический поиск (алиас для /search/).

### GET /search/by-file

Поиск элементов из конкретного файла.

**Query Parameters:**
- `file_path` (str) - Путь к файлу
- `repository_name` (str, optional) - Имя репозитория

**Response:**
```
{
  "results": [...],
  "total": 15,
  "query": "file:auth.py"
}
```

---

## Generation

### POST /generation/generate

Генерация ответа с RAG.

**Request:**
```
{
  "query": "How is authentication implemented?",
  "use_context": true,
  "top_k": 5,
  "language": "python",
  "stream": false
}
```

**Response:**
```
{
  "content": "Authentication is implemented using...",
  "model": "qwen2.5-coder-32b-instruct",
  "tokens_used": 256,
  "context_used": true,
  "context_count": 5
}
```

### POST /generation/generate/stream

Потоковая генерация (Server-Sent Events).

**Request:** Same as `/generation/generate` with `stream: true`

**Response:**
```
data: {"chunk": "Authentication"}
data: {"chunk": " is"}
data: {"chunk": " implemented"}
...
data: [DONE]
```

### POST /generation/explain

Объяснение кода.

**Request:**
```
{
  "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "language": "python",
  "detailed": true
}
```

**Response:**
```
{
  "content": "This is a recursive Fibonacci function...",
  "model": "qwen2.5-coder-32b-instruct",
  "tokens_used": 180,
  "context_used": false,
  "context_count": 0
}
```

### POST /generation/improve

Улучшение кода.

**Request:**
```
{
  "code": "def add(a, b):\n    return a + b",
  "language": "python",
  "focus": "performance"
}
```

**Response:**
```
{
  "content": "Here are suggestions to improve the code...",
  "model": "qwen2.5-coder-32b-instruct",
  "tokens_used": 200,
  "context_used": false,
  "context_count": 0
}
```

### POST /generation/chat

Чат с контекстом кода.

**Request:**
```
{
  "messages": [
    {"role": "user", "content": "Show me authentication code"},
    {"role": "assistant", "content": "Here is the auth code..."},
    {"role": "user", "content": "How to improve it?"}
  ],
  "use_context": true,
  "top_k": 5
}
```

**Response:**
```
{
  "content": "To improve the authentication code...",
  "model": "qwen2.5-coder-32b-instruct",
  "tokens_used": 300,
  "context_used": true,
  "context_count": 5
}
```

### GET /generation/model/info

Информация о модели Qwen.

**Response:**
```
{
  "model": "qwen2.5-coder-32b-instruct",
  "max_tokens": 2048,
  "temperature": 0.7
}
```

---

## Repositories

### POST /repositories/index

Индексация репозитория (фоновая задача).

**Request:**
```
{
  "repository_path": "/path/to/repo",
  "repository_name": "my-project",
  "branch": "main",
  "commit_hash": "abc123",
  "provider": "local",
  "exclude_dirs": [".git", "node_modules"]
}
```

**Response:**
```
{
  "status": "queued",
  "repository_name": "my-project"
}
```

### GET /repositories/list

Список проиндексированных репозиториев.

**Response:**
```
["project-a", "project-b", "project-c"]
```

### GET /repositories/{repository_name}/info

Информация о репозитории.

**Response:**
```
{
  "repository_name": "my-project",
  "branch": "main",
  "total_files": 150,
  "total_elements": 1234,
  "indexed_at": "2025-10-15T12:00:00",
  "languages": ["python", "javascript", "typescript"]
}
```

### GET /repositories/{repository_name}/statistics

Статистика по репозиторию.

**Response:**
```
{
  "total_elements": 1234,
  "by_language": {
    "python": 800,
    "javascript": 400,
    "typescript": 34
  },
  "by_type": {
    "function": 600,
    "class": 200,
    "method": 434
  },
  "by_file": {
    "auth.py": 50,
    "database.py": 80,
    ...
  },
  "total_lines": 50000,
  "avg_complexity": 3.5
}
```

### DELETE /repositories/{repository_name}

Удаление репозитория из индекса.

**Response:**
```
{
  "status": "deleted",
  "repository_name": "my-project"
}
```

---

## Error Responses

Все эндпоинты могут вернуть следующие ошибки:

### 400 Bad Request
```
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found
```
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```
{
  "detail": "Internal server error: <error_message>"
}
```

### 503 Service Unavailable
```
{
  "detail": "Service temporarily unavailable"
}
```

---

## Rate Limiting

- Default: 100 requests/minute per IP
- Burst: 10 requests/second

Headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1634567890
```

---

## Examples

### Python Client

```
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Index file
response = requests.post(
    f"{BASE_URL}/indexing/file",
    json={
        "file_path": "/path/to/file.py",
        "repository_name": "my-project"
    }
)
print(response.json())

# Search
response = requests.post(
    f"{BASE_URL}/search/",
    json={
        "query": "authentication function",
        "top_k": 5,
        "language": "python"
    }
)
results = response.json()

# Generate answer
response = requests.post(
    f"{BASE_URL}/generation/generate",
    json={
        "query": "How is auth implemented?",
        "use_context": True,
        "top_k": 5
    }
)
answer = response.json()
print(answer["content"])
```

### cURL Examples

```
# Health check
curl http://localhost:8000/health

# Index directory
curl -X POST http://localhost:8000/indexing/directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/path/to/project",
    "repository_name": "my-project",
    "recursive": true
  }'

# Search
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database connection",
    "top_k": 10,
    "language": "python"
  }'

# Generate answer
curl -X POST http://localhost:8000/generation/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to connect to database?",
    "use_context": true,
    "top_k": 5
  }'
```

---

## Interactive Documentation

OpenAPI/Swagger документация доступна по адресу:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`


Это полный API Reference с описанием всех эндпоинтов, примерами запросов и ответов, обработкой ошибок и примерами использования.