"""
Базовый пример использования Code RAG.

Демонстрирует основные возможности:
1. Парсинг кода
2. Создание коллекции
3. Индексация
4. Поиск
5. Генерация ответов с RAG
"""
import logging
from pathlib import Path
import sys

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..code_rag.core import (
    # Parser
    ParserFactory,
    list_supported_languages,

    # Embeddings
    CodeEmbedder,

    # Vector DB
    QdrantClient,
    VectorizationPipeline,
    DEFAULT_CODE_SCHEMA,

    # RAG
    RAGRetriever,
    QwenIntegration,
)

from ..code_rag.utils import setup_logger, format_duration
import time


def main():
    """Главная функция примера."""

    # Настройка логирования
    logger = logging.getLogger(__name__)
    logger.info("🚀 Запуск примера Code RAG")

    # ========== 1. Информация о поддерживаемых языках ==========
    logger.info(f"\n📚 Поддерживаемые языки: {', '.join(list_supported_languages())}")

    # ========== 2. Парсинг файла ==========
    logger.info("\n🔍 Парсинг файла...")

    # Пример файла для парсинга
    example_code = '''
def calculate_fibonacci(n: int) -> int:
    """Вычисление числа Фибоначчи."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class Calculator:
    """Простой калькулятор."""

    def add(self, a: int, b: int) -> int:
        """Сложение двух чисел."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Вычитание двух чисел."""
        return a - b
'''

    # Создаем временный файл
    temp_file = Path("temp_example.py")
    temp_file.write_text(example_code)

    try:
        # Парсим файл
        parser = ParserFactory.create_parser_for_file(temp_file)
        module = parser.parse_file(temp_file)

        logger.info(f"✓ Найдено элементов: {len(module.all_elements)}")

        # Выводим элементы
        for element in module.all_elements:
            logger.info(f"  - {element.type.value}: {element.qualified_name}")

    finally:
        temp_file.unlink(missing_ok=True)

    # ========== 3. Инициализация компонентов ==========
    logger.info("\n⚙️ Инициализация компонентов...")

    # Embedder
    logger.info("  Загрузка embedder модели...")
    embedder = CodeEmbedder()
    logger.info(f"  ✓ Embedder готов (dim={embedder.embedding_dim})")

    # Qdrant Client
    logger.info("  Подключение к Qdrant...")
    qdrant_client = QdrantClient(url="http://localhost:6333")

    if not qdrant_client.health_check():
        logger.error("❌ Qdrant недоступен. Запустите: docker run -p 6333:6333 qdrant/qdrant")
        return

    logger.info("  ✓ Qdrant подключен")

    # ========== 4. Создание коллекции ==========
    collection_name = "example_collection"
    logger.info(f"\n📦 Создание коллекции: {collection_name}")

    # Проверяем существование
    if qdrant_client.collection_exists(collection_name):
        logger.info("  Коллекция существует, пересоздаём...")
        qdrant_client.delete_collection(collection_name)

    # Создаем коллекцию
    from ..code_rag.core.vector_db import CollectionSchema
    schema = CollectionSchema(
        collection_name=collection_name,
        vector_size=embedder.embedding_dim
    )

    qdrant_client.create_collection(schema)
    logger.info("  ✓ Коллекция создана")

    # ========== 5. Индексация кода ==========
    logger.info("\n📥 Индексация примеров кода...")

    # Создаем pipeline
    pipeline = VectorizationPipeline(
        collection_name=collection_name,
        embedder=embedder,
        qdrant_client=qdrant_client
    )

    # Примеры кода для индексации
    examples = [
        ("auth.py", '''
def authenticate_user(username: str, password: str) -> bool:
    """Аутентификация пользователя."""
    # Проверка credentials
    return verify_password(password, get_password_hash(username))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверка пароля."""
    return hash(plain_password) == hashed_password
'''),
        ("database.py", '''
class DatabaseConnection:
    """Подключение к базе данных."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def connect(self):
        """Установка соединения."""
        pass

    def execute_query(self, query: str):
        """Выполнение SQL запроса."""
        pass
'''),
        ("api.py", '''
def create_user(user_data: dict) -> dict:
    """Создание нового пользователя."""
    user = User(**user_data)
    db.save(user)
    return user.to_dict()

def get_user(user_id: int) -> dict:
    """Получение пользователя по ID."""
    user = db.query(User).filter_by(id=user_id).first()
    return user.to_dict() if user else None
'''),
    ]

    # Индексируем примеры
    start_time = time.time()
    total_indexed = 0

    for filename, code in examples:
        temp_file = Path(filename)
        temp_file.write_text(code)

        try:
            result = pipeline.process_file(
                temp_file,
                repository_info={"repository_name": "example-project"}
            )
            total_indexed += result.indexed_elements
            logger.info(f"  ✓ {filename}: {result.indexed_elements} элементов")
        finally:
            temp_file.unlink(missing_ok=True)

    duration = time.time() - start_time
    logger.info(f"\n✓ Проиндексировано {total_indexed} элементов за {format_duration(duration)}")

    # ========== 6. Поиск кода ==========
    logger.info("\n🔎 Поиск кода...")

    retriever = RAGRetriever(
        collection_name=collection_name,
        embedder=embedder,
        qdrant_client=qdrant_client
    )

    # Примеры поисковых запросов
    queries = [
        "authentication function",
        "database connection",
        "create user API",
    ]

    for query in queries:
        logger.info(f"\n  Запрос: '{query}'")
        results = retriever.search(query=query, top_k=2)

        for i, result in enumerate(results, 1):
            logger.info(f"    {i}. {result.element.qualified_name} (score: {result.score:.3f})")
            logger.info(f"       {result.element.type.value} в {result.element.location.file_path}")

    # ========== 7. RAG с Qwen (опционально) ==========
    logger.info("\n🤖 Генерация ответов с RAG...")

    # Проверяем наличие API ключа
    import os
    qwen_api_key = os.getenv("QWEN_API_KEY")

    if qwen_api_key:
        try:
            qwen = QwenIntegration(
                api_key=qwen_api_key,
                model="qwen2.5-coder-32b-instruct"
            )

            question = "How is user authentication implemented?"
            logger.info(f"\n  Вопрос: {question}")

            # Получаем контекст
            context_results = retriever.search(query=question, top_k=3)

            # Генерируем ответ
            response = qwen.answer_question_with_rag(
                question=question,
                context_results=context_results
            )

            logger.info(f"\n  Ответ:\n{response.content[:500]}...")
            logger.info(f"\n  Использовано токенов: {response.tokens_used}")

        except Exception as e:
            logger.warning(f"  ⚠️ Qwen недоступен: {e}")
    else:
        logger.info("  ⚠️ QWEN_API_KEY не установлен, пропускаем генерацию")

    # ========== 8. Статистика ==========
    logger.info("\n📊 Статистика коллекции:")
    info = qdrant_client.get_collection_info(collection_name)
    logger.info(f"  Точек: {info.get('points_count', 0)}")
    logger.info(f"  Размер вектора: {info.get('vector_size', 0)}")

    # ========== Очистка ==========
    logger.info("\n🧹 Очистка...")
    qdrant_client.delete_collection(collection_name)
    logger.info("  ✓ Коллекция удалена")

    logger.info("\n✅ Пример завершён!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
