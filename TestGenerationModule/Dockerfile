# Dockerfile для Code RAG API

FROM python:3.10-slim

# Метаданные
LABEL maintainer="your-email@example.com"
LABEL description="Code RAG - векторный поиск и RAG для кода"
LABEL version="0.1.0"

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY code_rag/ ./code_rag/
COPY examples/ ./examples/
COPY .env.example .env

# Создаем директории для данных
RUN mkdir -p /app/logs /app/data

# Переменные окружения по умолчанию
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Qdrant настройки (по умолчанию локальный)
ENV QDRANT_URL=http://qdrant:6333
ENV COLLECTION_NAME=code_collection

# Expose порт
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск приложения
CMD ["python", "-m", "uvicorn", "code_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
