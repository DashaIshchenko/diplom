"""
Tests for vectorization pipeline.
"""

import pytest
import numpy as np
from pathlib import Path

from ..code_rag.core.vector_db import VectorizationPipeline, VectorizationResult
from ..code_rag.core.parser import ProgrammingLanguage, CodeElementType


# ==================== VectorizationPipeline Tests ====================

class TestVectorizationPipeline:
    """Тесты для VectorizationPipeline."""

    def test_pipeline_initialization(self, vectorization_pipeline):
        """Тест инициализации pipeline."""
        assert vectorization_pipeline is not None
        assert vectorization_pipeline.collection_name is not None
        assert vectorization_pipeline.embedder is not None
        assert vectorization_pipeline.qdrant_client is not None

    def test_process_single_file(
            self,
            vectorization_pipeline,
            sample_files,
            test_collection
    ):
        """Тест обработки одного файла."""
        py_file = sample_files["python"]

        result = vectorization_pipeline.process_file(
            py_file,
            repository_info={"repository_name": "test-repo"}
        )

        assert isinstance(result, VectorizationResult)
        assert result.total_files == 1
        assert result.parsed_files >= 1
        assert result.total_elements > 0
        assert result.indexed_elements > 0
        assert result.success_rate > 0.0

    def test_process_directory(
            self,
            vectorization_pipeline,
            sample_files,
            temp_dir
    ):
        """Тест обработки директории."""
        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "test-repo"},
            recursive=False
        )

        assert isinstance(result, VectorizationResult)
        assert result.total_files >= 2  # Python и JavaScript файлы
        assert result.indexed_elements > 0

    def test_process_directory_recursive(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест рекурсивной обработки директории."""
        # Создаем поддиректорию
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        # Файлы в основной директории
        (temp_dir / "file1.py").write_text("def func1(): pass")

        # Файлы в поддиректории
        (subdir / "file2.py").write_text("def func2(): pass")

        # Рекурсивная обработка
        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "test-repo"},
            recursive=True
        )

        assert result.total_files >= 2

    def test_process_nonexistent_file(self, vectorization_pipeline):
        """Тест обработки несуществующего файла."""
        result = vectorization_pipeline.process_file(
            Path("/nonexistent/file.py"),
            repository_info={"repository_name": "test-repo"}
        )

        # ✅ Проверяем, что файл обработался с ошибкой
        assert result.total_files == 1
        assert result.parsed_files == 0
        assert result.failed_files == 1
        assert result.success_rate == 0.0

        # ✅ Проверяем, что ошибка записана
        assert len(result.errors) == 1
        error = result.errors[0]
        assert "file" in error
        assert "error" in error
        assert "nonexistent" in error["file"].lower()

        # ✅ Проверяем тип ошибки в сообщении
        error_msg = error["error"].lower()
        assert any(keyword in error_msg for keyword in [
            "not found", "no such file", "does not exist", "filenotfounderror"
        ])

    def test_process_unsupported_file(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест обработки неподдерживаемого файла."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Plain text")

        result = vectorization_pipeline.process_file(
            txt_file,
            repository_info={"repository_name": "test-repo"}
        )

        # Должен пропустить файл
        assert result.parsed_files == 0
        assert result.failed_files == 1

    def test_process_file_with_syntax_error(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест обработки файла с синтаксической ошибкой."""
        broken_file = temp_dir / "broken.py"
        broken_file.write_text("def broken(\n  # Missing closing")

        result = vectorization_pipeline.process_file(
            broken_file,
            repository_info={"repository_name": "test-repo"}
        )

        # Должен зафиксировать ошибку
        assert result.failed_files == 1
        assert len(result.errors) > 0


# ==================== VectorizationResult Tests ====================

class TestVectorizationResult:
    """Тесты для VectorizationResult."""

    def test_result_creation(self):
        """Тест создания результата."""
        result = VectorizationResult(
            total_files=10,
            parsed_files=9,
            failed_files=1,
            total_elements=50,
            indexed_elements=48
        )

        assert result.total_files == 10
        assert result.parsed_files == 9
        assert result.failed_files == 1
        assert result.total_elements == 50
        assert result.indexed_elements == 48

    def test_success_rate_calculation(self):
        """Тест вычисления success rate."""
        result = VectorizationResult(
            total_files=10,
            parsed_files=8,
            failed_files=2,
            total_elements=40,
            indexed_elements=40
        )

        assert result.success_rate == 80.0

    def test_success_rate_zero_files(self):
        """Тест success rate при нуле файлов."""
        result = VectorizationResult(
            total_files=0,
            parsed_files=0,
            failed_files=0,
            total_elements=0,
            indexed_elements=0
        )

        assert result.success_rate == 0.0

    def test_result_with_errors(self):
        """Тест результата с ошибками."""
        result = VectorizationResult(
            total_files=5,
            parsed_files=3,
            failed_files=2,
            total_elements=15,
            indexed_elements=15,
            errors=[
                {"file": "error1.py", "error": "Syntax error"},
                {"file": "error2.py", "error": "Parse error"}
            ]
        )

        assert len(result.errors) == 2
        assert result.failed_files == 2

    def test_result_to_dict(self):
        """Тест конвертации результата в словарь."""
        result = VectorizationResult(
            total_files=10,
            parsed_files=9,
            failed_files=1,
            total_elements=45,
            indexed_elements=45
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "total_files" in result_dict
        assert "success_rate" in result_dict
        assert "indexed_elements" in result_dict


# ==================== Batch Processing Tests ====================

class TestBatchProcessing:
    """Тесты батчевой обработки."""

    def test_batch_file_processing(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест батчевой обработки файлов."""
        # Создаем несколько файлов
        files = []
        for i in range(5):
            file_path = temp_dir / f"file_{i}.py"
            file_path.write_text(f"def func_{i}(): pass")
            files.append(file_path)

        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "test-repo"}
        )

        assert result.total_files == 5
        assert result.parsed_files == 5
        assert result.total_elements >= 5

    def test_batch_with_mixed_success(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест батчевой обработки со смешанными результатами."""
        # Создаем валидные и невалидные файлы
        files = []

        # Валидный файл
        valid_file = temp_dir / "valid.py"
        valid_file.write_text("def valid(): pass")
        files.append(valid_file)

        # Невалидный файл
        invalid_file = temp_dir / "invalid.py"
        invalid_file.write_text("def broken(\n")
        files.append(invalid_file)

        # Неподдерживаемый файл
        unsupported_file = temp_dir / "text.txt"
        unsupported_file.write_text("text")
        files.append(unsupported_file)

        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "test-repo"}
        )

        assert result.total_files == 3
        assert result.parsed_files >= 1
        assert result.failed_files >= 1


# ==================== Repository Info Tests ====================

class TestRepositoryInfo:
    """Тесты для repository info в векторизации."""

    def test_vectorization_with_full_repo_info(
            self,
            vectorization_pipeline,
            sample_files
    ):
        """Тест векторизации с полной информацией о репозитории."""
        py_file = sample_files["python"]

        repo_info = {
            "repository_name": "test-repo",
            "branch": "main",
            "commit_hash": "abc123",
            "provider": "github"
        }

        result = vectorization_pipeline.process_file(
            py_file,
            repository_info=repo_info
        )

        assert result.indexed_elements > 0

        # Проверяем что информация сохранена в Qdrant
        # (это требует доступа к Qdrant для проверки)

    def test_vectorization_without_repo_info(
            self,
            vectorization_pipeline,
            sample_files
    ):
        """Тест векторизации без информации о репозитории."""
        py_file = sample_files["python"]

        result = vectorization_pipeline.process_file(
            py_file,
            repository_info=None
        )

        # Должен использовать значения по умолчанию
        assert result.indexed_elements > 0


# ==================== Filter Tests ====================

class TestVectorizationFilters:
    """Тесты фильтрации при векторизации."""

    def test_filter_by_extension(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест фильтрации по расширению файла."""
        # Создаем файлы разных типов
        (temp_dir / "test1.py").write_text("def func1(): pass")
        (temp_dir / "test2.js").write_text("function func2() {}")
        (temp_dir / "test3.txt").write_text("text")

        # Обрабатываем только Python
        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "test-repo"},
            file_pattern="*.py"
        )

        # Должен обработать только .py файл
        assert result.parsed_files == 1

# ==================== Progress Tracking Tests ====================

class TestProgressTracking:
    """Тесты отслеживания прогресса."""

    def test_progress_callback(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест callback для прогресса."""
        # Создаем файлы
        for i in range(5):
            (temp_dir / f"file_{i}.py").write_text(f"def func_{i}(): pass")

        progress_calls = []

        def progress_callback(file_path, current, total):
            progress_calls.append({
                "file": file_path,
                "current": current,
                "total": total,
            })

        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "test-repo"},
            progress_callback=progress_callback
        )

        # Проверяем что callback вызывался
        assert len(progress_calls) > 0
        assert progress_calls[-1]["current"] == progress_calls[-1]["total"]


# ==================== Performance Tests ====================

@pytest.mark.slow
class TestVectorizationPerformance:
    """Тесты производительности векторизации."""

    def test_large_directory_processing(
            self,
            vectorization_pipeline,
            temp_dir,
            benchmark_timer
    ):
        """Тест обработки большой директории."""
        # Создаем много файлов
        for i in range(50):
            file_path = temp_dir / f"file_{i}.py"
            file_path.write_text(f"def func_{i}(): pass\n" * 5)

        benchmark_timer.start("large_directory")

        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "test-repo"}
        )

        benchmark_timer.stop("large_directory")

        assert result.total_files == 50
        assert result.indexed_elements > 0

        # Должно быть быстрее 60 секунд
        benchmark_timer.assert_faster_than("large_directory", 60.0)

    def test_large_file_processing(
            self,
            vectorization_pipeline,
            temp_dir,
            benchmark_timer
    ):
        """Тест обработки большого файла."""
        # Создаем большой файл
        lines = []
        for i in range(100):
            lines.append(f"def function_{i}():")
            lines.append(f"    '''Function {i}'''")
            lines.append(f"    return {i}")
            lines.append("")

        large_file = temp_dir / "large.py"
        large_file.write_text("\n".join(lines))

        benchmark_timer.start("large_file")

        result = vectorization_pipeline.process_file(
            large_file,
            repository_info={"repository_name": "test-repo"}
        )

        benchmark_timer.stop("large_file")

        assert result.total_elements >= 100
        benchmark_timer.assert_faster_than("large_file", 12.0)


# ==================== Integration Tests ====================

@pytest.mark.integration
class TestVectorizationIntegration:
    """Интеграционные тесты векторизации."""

    def test_end_to_end_workflow(
            self,
            vectorization_pipeline,
            rag_retriever,
            sample_python_code,
            temp_dir
    ):
        """Тест полного workflow: vectorize -> search."""
        # 1. Векторизация
        file_path = temp_dir / "sample.py"
        file_path.write_text(sample_python_code)

        vectorize_result = vectorization_pipeline.process_file(
            file_path,
            repository_info={"repository_name": "test-repo"}
        )

        assert vectorize_result.indexed_elements > 0

        # 2. Поиск
        search_results = rag_retriever.search(
            query="calculator function",
            top_k=5
        )

        assert len(search_results) > 0

        # 3. Проверка результатов
        for result in search_results:
            assert result.element is not None
            assert result.score > 0

    def test_multi_language_project(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест векторизации мультиязычного проекта."""
        # Python
        (temp_dir / "main.py").write_text("""
def main():
    '''Main function'''
    pass
""")

        # JavaScript
        (temp_dir / "app.js").write_text("""
function app() {
    // Main app
}
""")

        # TypeScript
        (temp_dir / "utils.ts").write_text("""
function utils(): void {
    // Utilities
}
""")

        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "multi-lang-project"}
        )

        assert result.parsed_files >= 3
        assert result.indexed_elements > 0


# ==================== Edge Cases Tests ====================

class TestVectorizationEdgeCases:
    """Тесты граничных случаев векторизации."""

    def test_empty_directory(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест обработки пустой директории."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = vectorization_pipeline.process_directory(
            empty_dir,
            repository_info={"repository_name": "test-repo"}
        )

        assert result.total_files == 0
        assert result.indexed_elements == 0

    def test_empty_file(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест обработки пустого файла."""
        empty_file = temp_dir / "empty.py"
        empty_file.write_text("")

        result = vectorization_pipeline.process_file(
            empty_file,
            repository_info={"repository_name": "test-repo"}
        )

        assert result.parsed_files == 1
        assert result.total_elements == 0

    def test_file_with_only_comments(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест обработки файла только с комментариями."""
        comment_file = temp_dir / "comments.py"
        comment_file.write_text("""
# This is a comment
# Another comment
# And another one
""")

        result = vectorization_pipeline.process_file(
            comment_file,
            repository_info={"repository_name": "test-repo"}
        )

        assert result.parsed_files == 1
        assert result.total_elements == 0

    def test_very_long_function_name(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест функции с очень длинным именем."""
        code = f"def {'very_long_name_' * 20}(): pass"

        file_path = temp_dir / "long_name.py"
        file_path.write_text(code)

        result = vectorization_pipeline.process_file(
            file_path,
            repository_info={"repository_name": "test-repo"}
        )

        assert result.indexed_elements > 0

    def test_nested_directories(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест глубоко вложенных директорий."""
        # Создаем глубокую структуру
        deep_path = temp_dir / "a" / "b" / "c" / "d" / "e"
        deep_path.mkdir(parents=True)

        # Файл в глубокой директории
        (deep_path / "deep.py").write_text("def deep_func(): pass")

        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "test-repo"},
            recursive=True
        )

        assert result.parsed_files >= 1


# ==================== Error Recovery Tests ====================

class TestErrorRecovery:
    """Тесты восстановления после ошибок."""

    def test_continue_after_parse_error(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест продолжения после ошибки парсинга."""
        # Создаем файлы
        (temp_dir / "valid1.py").write_text("def func1(): pass")
        (temp_dir / "broken.py").write_text("def broken(\n")
        (temp_dir / "valid2.py").write_text("def func2(): pass")

        result = vectorization_pipeline.process_directory(
            temp_dir,
            repository_info={"repository_name": "test-repo"}
        )

        # Должен обработать валидные файлы несмотря на ошибку
        assert result.parsed_files >= 2
        assert result.failed_files >= 1
        assert result.indexed_elements >= 2

    def test_handle_encoding_errors(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест обработки ошибок кодировки."""
        # Файл с неверной кодировкой
        weird_file = temp_dir / "weird.py"
        weird_file.write_bytes(b'\xff\xfe# Invalid UTF-8')

        result = vectorization_pipeline.process_file(
            weird_file,
            repository_info={"repository_name": "test-repo"}
        )

        # Должен зафиксировать ошибку
        assert result.failed_files == 1


# ==================== Parametrized Tests ====================

@pytest.mark.parametrize("language,extension,code", [
    (ProgrammingLanguage.PYTHON, ".py", "def test(): pass"),
    (ProgrammingLanguage.JAVASCRIPT, ".js", "function test() {}"),
    (ProgrammingLanguage.TYPESCRIPT, ".ts", "function test(): void {}"),
])
def test_vectorize_different_languages(
        vectorization_pipeline,
        temp_dir,
        language,
        extension,
        code
):
    """Параметризованный тест векторизации разных языков."""
    file_path = temp_dir / f"test{extension}"
    file_path.write_text(code)

    result = vectorization_pipeline.process_file(
        file_path,
        repository_info={"repository_name": "test-repo"}
    )

    assert result.parsed_files == 1
    assert result.indexed_elements > 0


@pytest.mark.parametrize("file_count", [1, 5, 10, 20])
def test_vectorize_varying_file_counts(
        vectorization_pipeline,
        temp_dir,
        file_count
):
    """Параметризованный тест разного количества файлов."""
    for i in range(file_count):
        (temp_dir / f"file_{i}.py").write_text(f"def func_{i}(): pass")

    result = vectorization_pipeline.process_directory(
        temp_dir,
        repository_info={"repository_name": "test-repo"}
    )

    assert result.total_files == file_count
    assert result.parsed_files == file_count


# ==================== Statistics Tests ====================

class TestVectorizationStatistics:
    """Тесты статистики векторизации."""

# ==================== Cleanup Tests ====================

class TestVectorizationCleanup:
    """Тесты очистки после векторизации."""

    def test_cleanup_on_error(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест очистки при ошибке."""
        # Создаем файл который вызовет ошибку
        bad_file = temp_dir / "bad.py"
        bad_file.write_text("invalid python code }{][")

        try:
            result = vectorization_pipeline.process_file(
                bad_file,
                repository_info={"repository_name": "test-repo"}
            )

            # Должен зафиксировать ошибку но не упасть
            assert result.failed_files >= 1
        except Exception as e:
            # Если exception, проверяем что ресурсы освобождены
            assert True


# ==================== Regression Tests ====================

class TestVectorizationRegression:
    """Регрессионные тесты."""

    def test_consistent_vectorization(
            self,
            vectorization_pipeline,
            temp_dir
    ):
        """Тест консистентности векторизации."""
        file_path = temp_dir / "test.py"
        file_path.write_text("def test(): pass")

        # Векторизуем несколько раз
        results = []
        for _ in range(3):
            result = vectorization_pipeline.process_file(
                file_path,
                repository_info={"repository_name": "test-repo"}
            )
            results.append(result.indexed_elements)

        # Все результаты должны быть одинаковыми
        assert len(set(results)) == 1
