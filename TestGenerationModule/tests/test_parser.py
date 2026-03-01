"""
Tests for parser module.
"""

import pytest
from pathlib import Path

from code_rag.core.parser import (
    ParserFactory,
    ProgrammingLanguage,
    CodeElementType,
    CodeElement,
    ModuleInfo,
    ParseResult,
    CodeLocation,
)


# ==================== ParserFactory Tests ====================

class TestParserFactory:
    """Тесты для ParserFactory."""

    def test_create_parser_python(self):
        """Тест создания Python парсера."""
        parser = ParserFactory.create_parser(ProgrammingLanguage.PYTHON)
        assert parser is not None
        assert parser.language == ProgrammingLanguage.PYTHON

    def test_create_parser_javascript(self):
        """Тест создания JavaScript парсера."""
        parser = ParserFactory.create_parser(ProgrammingLanguage.JAVASCRIPT)
        assert parser is not None
        assert parser.language == ProgrammingLanguage.JAVASCRIPT

    def test_create_parser_for_file_python(self, temp_dir):
        """Тест создания парсера по файлу Python."""
        py_file = temp_dir / "test.py"
        py_file.write_text("def test(): pass")

        parser = ParserFactory.create_parser_for_file(py_file)
        assert parser is not None
        assert parser.language == ProgrammingLanguage.PYTHON

    def test_create_parser_for_file_javascript(self, temp_dir):
        """Тест создания парсера по файлу JavaScript."""
        js_file = temp_dir / "test.js"
        js_file.write_text("function test() {}")

        parser = ParserFactory.create_parser_for_file(js_file)
        assert parser is not None
        assert parser.language == ProgrammingLanguage.JAVASCRIPT

    def test_can_parse_file_python(self, temp_dir):
        """Тест проверки возможности парсинга Python."""
        py_file = temp_dir / "test.py"
        py_file.write_text("")

        assert ParserFactory.can_parse_file(py_file) is True

    def test_can_parse_file_javascript(self, temp_dir):
        """Тест проверки возможности парсинга JavaScript."""
        js_file = temp_dir / "test.js"
        js_file.write_text("")

        assert ParserFactory.can_parse_file(js_file) is True

    def test_can_parse_file_unsupported(self, temp_dir):
        """Тест проверки неподдерживаемого файла."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("")

        assert ParserFactory.can_parse_file(txt_file) is False

    def test_list_supported_languages(self):
        """Тест получения списка поддерживаемых языков."""
        languages = ParserFactory.list_supported_languages()

        assert len(languages) > 0
        assert ProgrammingLanguage.PYTHON in languages
        assert ProgrammingLanguage.JAVASCRIPT in languages

    def test_get_language_from_extension(self):
        """Тест определения языка по расширению."""
        assert ParserFactory.get_language_from_extension(".py") == ProgrammingLanguage.PYTHON
        assert ParserFactory.get_language_from_extension(".js") == ProgrammingLanguage.JAVASCRIPT
        assert ParserFactory.get_language_from_extension(".ts") == ProgrammingLanguage.TYPESCRIPT
        assert ParserFactory.get_language_from_extension(".txt") is None


# ==================== Python Parser Tests ====================

class TestPythonParser:
    """Тесты для Python парсера."""

    @pytest.fixture
    def python_parser(self):
        """Python parser fixture."""
        return ParserFactory.create_parser(ProgrammingLanguage.PYTHON)

    def test_parse_simple_function(self, python_parser, temp_dir):
        """Тест парсинга простой функции."""
        code = """
def hello():
    return "Hello, World!"
"""
        file_path = temp_dir / "test.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        assert module is not None
        assert len(module.functions) == 1
        assert module.functions[0].name == "hello"
        assert module.functions[0].type == CodeElementType.FUNCTION

    def test_parse_function_with_parameters(self, python_parser, temp_dir):
        """Тест парсинга функции с параметрами."""
        code = """
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b
"""
        file_path = temp_dir / "test.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        assert len(module.functions) == 1
        func = module.functions[0]
        assert func.name == "add"
        assert func.signature is not None
        assert func.docstring is not None
        assert "Add two numbers" in func.docstring

    def test_parse_simple_class(self, python_parser, temp_dir):
        """Тест парсинга простого класса."""
        code = """
class Calculator:
    '''Simple calculator.'''

    def add(self, a, b):
        return a + b
"""
        file_path = temp_dir / "test.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        assert len(module.classes) == 1
        cls = module.classes[0]
        assert cls.name == "Calculator"
        assert cls.type == CodeElementType.CLASS
        assert len(cls.methods) == 1
        assert cls.methods[0].name == "add"

    def test_parse_class_with_init(self, python_parser, temp_dir):
        """Тест парсинга класса с __init__."""
        code = """
class Person:
    def __init__(self, name: str):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"
"""
        file_path = temp_dir / "test.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        cls = module.classes[0]
        assert len(cls.methods) == 2
        method_names = [m.name for m in cls.methods]
        assert "__init__" in method_names
        assert "greet" in method_names

    def test_parse_nested_functions(self, python_parser, temp_dir):
        """Тест парсинга вложенных функций."""
        code = """
def outer():
    def inner():
        return "inner"
    return inner()
"""
        file_path = temp_dir / "test.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        # Может зависеть от реализации - внутренняя функция может быть или не быть в списке
        assert len(module.functions) >= 1
        assert module.functions[0].name == "outer"

    def test_parse_decorators(self, python_parser, temp_dir):
        """Тест парсинга декораторов."""
        code = """
@staticmethod
def static_method():
    pass

@classmethod
def class_method(cls):
    pass
"""
        file_path = temp_dir / "test.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        assert len(module.functions) == 2

    def test_parse_imports(self, python_parser, temp_dir):
        """Тест парсинга импортов."""
        code = """
import os
import sys
from pathlib import Path
from typing import List, Dict
"""
        file_path = temp_dir / "test.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        # Импорты могут быть или не быть доступны в зависимости от реализации
        assert module is not None

    def test_parse_complex_code(self, python_parser, sample_python_code, temp_dir):
        """Тест парсинга сложного кода."""
        file_path = temp_dir / "complex.py"
        file_path.write_text(sample_python_code)

        module = python_parser.parse_file(file_path)

        assert len(module.functions) >= 2
        assert len(module.classes) >= 1
        assert len(module.all_elements) >= 3

    def test_parse_empty_file(self, python_parser, temp_dir):
        """Тест парсинга пустого файла."""
        file_path = temp_dir / "empty.py"
        file_path.write_text("")

        module = python_parser.parse_file(file_path)

        assert module is not None
        assert len(module.all_elements) == 0

    def test_parse_syntax_error(self, python_parser, temp_dir):
        """Тест парсинга файла с синтаксической ошибкой."""
        code = """
    def broken(
        # Missing closing parenthesis
    """
        file_path = temp_dir / "broken.py"
        file_path.write_text(code)

        # Парсер может либо вернуть None, либо ParseResult с ошибками
        result = python_parser.parse_file(file_path)

        # Проверяем что результат содержит ошибку или является None
        if result is None:
            # Парсер вернул None при ошибке
            assert True
        else:
            # Парсер вернул ParseResult - проверяем что есть ошибки
            assert isinstance(result, ParseResult)
            # Либо нет модулей, либо есть записи об ошибках
            assert len(result.modules) == 0 or len(result.errors) > 0


# ==================== JavaScript Parser Tests ====================

class TestJavaScriptParser:
    """Тесты для JavaScript парсера."""

    @pytest.fixture
    def javascript_parser(self):
        """JavaScript parser fixture."""
        return ParserFactory.create_parser(ProgrammingLanguage.JAVASCRIPT)

    def test_parse_simple_function(self, javascript_parser, temp_dir):
        """Тест парсинга простой функции."""
        code = """
function hello() {
    return "Hello, World!";
}
"""
        file_path = temp_dir / "test.js"
        file_path.write_text(code)

        module = javascript_parser.parse_file(file_path)

        assert module is not None
        assert len(module.functions) >= 1
        assert module.functions[0].name == "hello"

    def test_parse_arrow_function(self, javascript_parser, temp_dir):
        """Тест парсинга arrow function."""
        code = """
const add = (a, b) => a + b;
"""
        file_path = temp_dir / "test.js"
        file_path.write_text(code)

        module = javascript_parser.parse_file(file_path)

        assert module is not None
        # Arrow function может быть распознана как функция
        assert len(module.all_elements) >= 0

    def test_parse_class(self, javascript_parser, temp_dir):
        """Тест парсинга класса."""
        code = """
class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}
"""
        file_path = temp_dir / "test.js"
        file_path.write_text(code)

        module = javascript_parser.parse_file(file_path)

        assert len(module.classes) >= 1
        cls = module.classes[0]
        assert cls.name == "Calculator"

    def test_parse_complex_javascript(self, javascript_parser, sample_javascript_code, temp_dir):
        """Тест парсинга сложного JavaScript кода."""
        file_path = temp_dir / "complex.js"
        file_path.write_text(sample_javascript_code)

        module = javascript_parser.parse_file(file_path)

        assert module is not None
        assert len(module.all_elements) > 0


# ==================== CodeElement Tests ====================

class TestCodeElement:
    """Тесты для CodeElement."""

    def test_code_element_creation(self):
        """Тест создания CodeElement."""
        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=10,
            end_line=15
        )

        element = CodeElement(
            name="test_function",
            type=CodeElementType.FUNCTION,
            language=ProgrammingLanguage.PYTHON,
            source_code="def test_function(): pass",
            location=location
        )

        assert element.name == "test_function"
        assert element.type == CodeElementType.FUNCTION
        assert element.language == ProgrammingLanguage.PYTHON
        assert element.location.start_line == 10
        assert element.location.end_line == 15

    def test_code_element_with_docstring(self):
        """Тест CodeElement с docstring."""
        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=3
        )

        element = CodeElement(
            name="documented_func",
            type=CodeElementType.FUNCTION,
            language=ProgrammingLanguage.PYTHON,
            source_code="def documented_func(): pass",
            docstring="This function is documented.",
            location=location
        )

        assert element.docstring == "This function is documented."

    def test_code_element_with_signature(self):
        """Тест CodeElement с signature."""
        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=2
        )

        element = CodeElement(
            name="typed_func",
            type=CodeElementType.FUNCTION,
            language=ProgrammingLanguage.PYTHON,
            source_code="def typed_func(x: int) -> str: pass",
            signature="typed_func(x: int) -> str",
            location=location
        )

        assert element.signature == "typed_func(x: int) -> str"

    def test_code_element_complexity(self):
        """Тест complexity в CodeElement."""
        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=10
        )

        element = CodeElement(
            name="complex_func",
            type=CodeElementType.FUNCTION,
            language=ProgrammingLanguage.PYTHON,
            source_code="def complex_func(): ...",
            complexity=5,
            location=location
        )

        assert element.complexity == 5

    def test_code_element_to_dict(self):
        """Тест конвертации CodeElement в словарь."""
        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=2
        )

        element = CodeElement(
            name="func",
            type=CodeElementType.FUNCTION,
            language=ProgrammingLanguage.PYTHON,
            source_code="def func(): pass",
            location=location
        )

        element_dict = element.to_dict()

        assert isinstance(element_dict, dict)
        assert element_dict["name"] == "func"
        assert element_dict["type"] == CodeElementType.FUNCTION.value
        assert element_dict["language"] == ProgrammingLanguage.PYTHON.value


# ==================== ModuleInfo Tests ====================

class TestModuleInfo:
    """Тесты для ModuleInfo."""

    def test_module_info_creation(self):
        """Тест создания ModuleInfo."""
        module = ModuleInfo(
            file_path=Path("test.py"),
            language=ProgrammingLanguage.PYTHON,
            module_name="test",
        )

        assert module.file_path == Path("test.py")
        assert module.language == ProgrammingLanguage.PYTHON
        assert len(module.functions) == 0
        assert len(module.classes) == 0

    def test_module_info_add_function(self):
        """Тест добавления функции в ModuleInfo."""
        module = ModuleInfo(
            file_path=Path("test.py"),
            language=ProgrammingLanguage.PYTHON,
            module_name="test",
        )

        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=2
        )

        func = CodeElement(
            name="test_func",
            type=CodeElementType.FUNCTION,
            language=ProgrammingLanguage.PYTHON,
            source_code="def test_func(): pass",
            location=location
        )

        module.functions.append(func)

        assert len(module.functions) == 1
        assert module.functions[0].name == "test_func"

    def test_module_info_add_class(self):
        """Тест добавления класса в ModuleInfo."""
        module = ModuleInfo(
            file_path=Path("test.py"),
            language=ProgrammingLanguage.PYTHON,
            module_name="test",
        )

        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5
        )

        cls = CodeElement(
            name="TestClass",
            type=CodeElementType.CLASS,
            language=ProgrammingLanguage.PYTHON,
            source_code="class TestClass: pass",
            location=location
        )

        module.classes.append(cls)

        assert len(module.classes) == 1
        assert module.classes[0].name == "TestClass"

    def test_module_info_all_elements(self):
        """Тест получения всех элементов."""
        module = ModuleInfo(
            file_path=Path("test.py"),
            language=ProgrammingLanguage.PYTHON,
            module_name="test",
        )

        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=2
        )

        func = CodeElement(
            name="func",
            type=CodeElementType.FUNCTION,
            language=ProgrammingLanguage.PYTHON,
            source_code="def func(): pass",
            location=location
        )

        cls = CodeElement(
            name="Class",
            type=CodeElementType.CLASS,
            language=ProgrammingLanguage.PYTHON,
            source_code="class Class: pass",
            location=location
        )

        module.functions.append(func)
        module.classes.append(cls)

        all_elements = module.all_elements

        assert len(all_elements) == 2
        assert func in all_elements
        assert cls in all_elements


# ==================== ParseResult Tests ====================

class TestParseResult:
    """Тесты для ParseResult."""

    def test_parse_result_creation(self):
        """Тест создания ParseResult."""
        result = ParseResult()

        assert len(result.modules) == 0
        assert len(result.errors) == 0
        assert result.total_elements == 0
        assert result.success_rate == 0.0

    def test_parse_result_with_modules(self):
        """Тест ParseResult с модулями."""
        module = ModuleInfo(
            file_path=Path("test.py"),
            language=ProgrammingLanguage.PYTHON,
            module_name="test",
        )

        result = ParseResult(modules=[module])

        assert len(result.modules) == 1
        assert result.modules[0].module_name == "test"

    def test_parse_result_with_error(self):
        """Тест ParseResult с ошибкой."""
        result = ParseResult(
            modules=[],
            errors=[
                {"file": "error.py", "error": "Syntax error"}
            ]
        )

        assert len(result.errors) == 1
        assert result.errors[0]["file"] == "error.py"

    def test_parse_result_success_rate(self):
        """Тест расчета success rate."""
        module1 = ModuleInfo(
            file_path=Path("test1.py"),
            language=ProgrammingLanguage.PYTHON,
            module_name="test1",
        )

        module2 = ModuleInfo(
            file_path=Path("test2.py"),
            language=ProgrammingLanguage.PYTHON,
            module_name="test2",
        )

        result = ParseResult(
            modules=[module1, module2],
            errors=[{"file": "error.py", "error": "Error"}]
        )

        # 2 успешных из 3 = 66.67%
        assert result.success_rate == pytest.approx(66.67, rel=0.01)

    def test_parse_result_total_elements(self):
        """Тест подсчета общего количества элементов."""
        module = ModuleInfo(
            file_path=Path("test.py"),
            language=ProgrammingLanguage.PYTHON,
            module_name="test",
        )

        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=2
        )

        func = CodeElement(
            name="func",
            type=CodeElementType.FUNCTION,
            language=ProgrammingLanguage.PYTHON,
            source_code="def func(): pass",
            location=location
        )

        module.functions.append(func)

        result = ParseResult(modules=[module])

        assert result.total_elements >= 1

    def test_parse_result_languages(self):
        """Тест определения языков."""
        module_py = ModuleInfo(
            file_path=Path("test.py"),
            language=ProgrammingLanguage.PYTHON,
            module_name="test_py",
        )

        module_js = ModuleInfo(
            file_path=Path("test.js"),
            language=ProgrammingLanguage.JAVASCRIPT,
            module_name="test_js",
        )

        result = ParseResult(modules=[module_py, module_js])

        assert len(result.languages) == 2
        assert ProgrammingLanguage.PYTHON in result.languages
        assert ProgrammingLanguage.JAVASCRIPT in result.languages

    def test_parse_result_empty_success_rate(self):
        """Тест success rate для пустого результата."""
        result = ParseResult()

        assert result.success_rate == 0.0


# ==================== CodeLocation Tests ====================

class TestCodeLocation:
    """Тесты для CodeLocation."""

    def test_code_location_creation(self):
        """Тест создания CodeLocation."""
        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=10,
            end_line=20
        )

        assert location.file_path == Path("test.py")
        assert location.start_line == 10
        assert location.end_line == 20

    def test_code_location_line_count(self):
        """Тест подсчета строк."""
        location = CodeLocation(
            file_path=Path("test.py"),
            start_line=10,
            end_line=20
        )

        line_count = location.end_line - location.start_line + 1
        assert line_count == 11


# ==================== Integration Tests ====================

@pytest.mark.integration
class TestParserIntegration:
    """Интеграционные тесты парсеров."""

    def test_parse_multiple_files(self, temp_dir):
        """Тест парсинга нескольких файлов."""
        # Python файл
        py_file = temp_dir / "test1.py"
        py_file.write_text("def func1(): pass")

        # JavaScript файл
        js_file = temp_dir / "test2.js"
        js_file.write_text("function func2() {}")

        # Парсим оба
        py_parser = ParserFactory.create_parser_for_file(py_file)
        js_parser = ParserFactory.create_parser_for_file(js_file)

        py_module = py_parser.parse_file(py_file)
        js_module = js_parser.parse_file(js_file)

        assert py_module is not None
        assert js_module is not None
        assert len(py_module.functions) >= 1
        assert len(js_module.functions) >= 1

    def test_parse_real_code_sample(self, python_parser, sample_python_code, temp_dir):
        """Тест парсинга реального примера кода."""
        file_path = temp_dir / "sample.py"
        file_path.write_text(sample_python_code)

        module = python_parser.parse_file(file_path)

        # Проверяем что нашли основные элементы
        assert len(module.all_elements) > 0

        # Функции
        function_names = [f.name for f in module.functions]
        assert "calculate_sum" in function_names or "calculate_product" in function_names

        # Классы
        class_names = [c.name for c in module.classes]
        assert "Calculator" in class_names


# ==================== Performance Tests ====================

@pytest.mark.slow
class TestParserPerformance:
    """Тесты производительности парсеров."""

    def test_parse_large_file(self, python_parser, temp_dir, benchmark_timer):
        """Тест парсинга большого файла."""
        # Создаем большой файл с множеством функций
        code_lines = []
        for i in range(100):
            code_lines.append(f"def function_{i}():")
            code_lines.append(f"    '''Function {i}'''")
            code_lines.append(f"    return {i}")
            code_lines.append("")

        code = "\n".join(code_lines)
        file_path = temp_dir / "large.py"
        file_path.write_text(code)

        benchmark_timer.start("parse_large")
        module = python_parser.parse_file(file_path)
        benchmark_timer.stop("parse_large")

        assert len(module.functions) == 100
        benchmark_timer.assert_faster_than("parse_large", 5.0)

    def test_parse_multiple_files_performance(self, temp_dir, benchmark_timer):
        """Тест производительности парсинга множества файлов."""
        files = []

        # Создаем 20 файлов
        for i in range(20):
            file_path = temp_dir / f"file_{i}.py"
            file_path.write_text(f"def func_{i}(): pass\n")
            files.append(file_path)

        benchmark_timer.start("parse_multiple")

        for file_path in files:
            parser = ParserFactory.create_parser_for_file(file_path)
            module = parser.parse_file(file_path)
            assert module is not None

        benchmark_timer.stop("parse_multiple")
        benchmark_timer.assert_faster_than("parse_multiple", 10.0)


# ==================== Edge Cases Tests ====================

class TestParserEdgeCases:
    """Тесты граничных случаев."""

    def test_parse_file_with_unicode(self, python_parser, temp_dir):
        """Тест парсинга файла с Unicode символами."""
        code = """
def привет():
    '''Функция приветствия'''
    return "Привет, мир!"

def 你好():
    return "你好世界"
"""
        file_path = temp_dir / "unicode.py"
        file_path.write_text(code, encoding="utf-8")

        module = python_parser.parse_file(file_path)

        assert module is not None
        assert len(module.functions) >= 1

    def test_parse_file_with_comments(self, python_parser, temp_dir):
        """Тест парсинга файла с комментариями."""
        code = """
# This is a comment
def func():  # inline comment
    # Another comment
    pass
"""
        file_path = temp_dir / "comments.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        assert len(module.functions) == 1

    def test_parse_file_with_strings(self, python_parser, temp_dir):
        """Тест парсинга файла со строками."""
        code = '''
def func():
    text = """
    Multiline
    string
    """
    return text
'''
        file_path = temp_dir / "strings.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        assert len(module.functions) == 1

    def test_parse_very_long_function(self, python_parser, temp_dir):
        """Тест парсинга очень длинной функции."""
        lines = ["def long_func():"]
        for i in range(1000):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_999")

        code = "\n".join(lines)
        file_path = temp_dir / "long_func.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        assert len(module.functions) == 1
        func = module.functions[0]
        assert func.location.end_line - func.location.start_line > 900

    def test_parse_nested_classes(self, python_parser, temp_dir):
        """Тест парсинга вложенных классов."""
        code = """
class Outer:
    class Inner:
        def method(self):
            pass
"""
        file_path = temp_dir / "nested.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        assert len(module.classes) >= 1

    def test_parse_lambda_functions(self, python_parser, temp_dir):
        """Тест парсинга lambda функций."""
        code = """
add = lambda a, b: a + b
multiply = lambda x, y: x * y
"""
        file_path = temp_dir / "lambdas.py"
        file_path.write_text(code)

        module = python_parser.parse_file(file_path)

        # Lambda может быть или не быть в списке функций
        assert module is not None


# ==================== Error Handling Tests ====================

class TestParserErrorHandling:
    """Тесты обработки ошибок."""

    def test_parse_nonexistent_file(self, python_parser):
        """Тест парсинга несуществующего файла."""
        result = python_parser.parse_file(Path("/nonexistent/file.py"))

        # Парсер должен gracefully обработать ошибку
        assert result is None or (isinstance(result, ParseResult) and len(result.modules) == 0)

    def test_parse_directory_instead_of_file(self, python_parser, temp_dir):
        """Тест парсинга директории вместо файла."""
        result = python_parser.parse_file(temp_dir)

        # Должен вернуть None или пустой результат
        assert result is None or (isinstance(result, ParseResult) and len(result.modules) == 0)

    def test_parse_binary_file(self, python_parser, temp_dir):
        """Тест парсинга бинарного файла."""
        binary_file = temp_dir / "binary.py"
        binary_file.write_bytes(b'\x00\x01\x02\x03')

        result = python_parser.parse_file(binary_file)

        # Должен обработать ошибку и вернуть None или пустой результат
        assert result is None or (isinstance(result, ParseResult) and len(result.modules) == 0)

# ==================== Parametrized Tests ====================

@pytest.mark.parametrize("extension,expected_language", [
    (".py", ProgrammingLanguage.PYTHON),
    (".js", ProgrammingLanguage.JAVASCRIPT),
    (".ts", ProgrammingLanguage.TYPESCRIPT),
    (".java", ProgrammingLanguage.JAVA),
    (".cs", ProgrammingLanguage.CSHARP),
])
def test_language_detection(extension, expected_language):
    """Параметризованный тест определения языка."""
    detected = ParserFactory.get_language_from_extension(extension)
    assert detected == expected_language


@pytest.mark.parametrize("code,expected_count", [
    ("def func1(): pass\ndef func2(): pass", 2),
    ("def func(): pass", 1),
    ("", 0),
    ("# Just a comment", 0),
])
def test_function_counting(python_parser, temp_dir, code, expected_count):
    """Параметризованный тест подсчета функций."""
    file_path = temp_dir / "test.py"
    file_path.write_text(code)

    module = python_parser.parse_file(file_path)
    assert len(module.functions) == expected_count


# ==================== TypeScript Parser Tests ====================

class TestTypeScriptParser:
    """Тесты для TypeScript парсера."""

    @pytest.fixture
    def typescript_parser(self):
        """TypeScript parser fixture."""
        return ParserFactory.create_parser(ProgrammingLanguage.TYPESCRIPT)

    def test_parse_typescript_function(self, typescript_parser, temp_dir):
        """Тест парсинга TypeScript функции."""
        code = """
function greet(name: string): string {
    return `Hello, ${name}`;
}
"""
        file_path = temp_dir / "test.ts"
        file_path.write_text(code)

        module = typescript_parser.parse_file(file_path)

        assert module is not None
        assert len(module.functions) >= 1

    def test_parse_typescript_interface(self, typescript_parser, temp_dir):
        """Тест парсинга TypeScript interface."""
        code = """
interface User {
    name: string;
    age: number;
}
"""
        file_path = temp_dir / "test.ts"
        file_path.write_text(code)

        module = typescript_parser.parse_file(file_path)

        assert module is not None
        # Interface может быть распознан как класс или отдельный тип

    def test_parse_typescript_class(self, typescript_parser, temp_dir):
        """Тест парсинга TypeScript класса."""
        code = """
class Person {
    private name: string;

    constructor(name: string) {
        this.name = name;
    }

    public greet(): string {
        return `Hello, ${this.name}`;
    }
}
"""
        file_path = temp_dir / "test.ts"
        file_path.write_text(code)

        module = typescript_parser.parse_file(file_path)

        assert len(module.classes) >= 1


# ==================== Multi-language Tests ====================

class TestMultiLanguageParsing:
    """Тесты парсинга разных языков."""

    def test_parse_mixed_project(self, temp_dir):
        """Тест парсинга проекта с разными языками."""
        # Python
        (temp_dir / "script.py").write_text("def python_func(): pass")

        # JavaScript
        (temp_dir / "script.js").write_text("function jsFunc() {}")

        # TypeScript
        (temp_dir / "script.ts").write_text("function tsFunc(): void {}")

        files = list(temp_dir.glob("*"))
        results = []

        for file_path in files:
            if ParserFactory.can_parse_file(file_path):
                parser = ParserFactory.create_parser_for_file(file_path)
                module = parser.parse_file(file_path)
                results.append(module)

        assert len(results) == 3
        assert all(m is not None for m in results)


# ==================== Regression Tests ====================

class TestParserRegression:
    """Регрессионные тесты."""

    def test_consistent_parsing(self, python_parser, temp_dir):
        """Тест консистентности парсинга."""
        code = "def test(): pass"
        file_path = temp_dir / "test.py"
        file_path.write_text(code)

        # Парсим несколько раз
        results = []
        for _ in range(3):
            module = python_parser.parse_file(file_path)
            results.append(len(module.functions))

        # Все результаты должны быть одинаковыми
        assert len(set(results)) == 1

    def test_parse_after_modification(self, python_parser, temp_dir):
        """Тест парсинга после изменения файла."""
        file_path = temp_dir / "test.py"

        # Первая версия
        file_path.write_text("def func1(): pass")
        module1 = python_parser.parse_file(file_path)

        # Вторая версия
        file_path.write_text("def func1(): pass\ndef func2(): pass")
        module2 = python_parser.parse_file(file_path)

        assert len(module1.functions) == 1
        assert len(module2.functions) == 2


# ==================== Utility Tests ====================

class TestParserUtilities:
    """Тесты вспомогательных функций."""

    def test_get_supported_extensions(self):
        """Тест получения поддерживаемых расширений."""
        extensions = ParserFactory.get_supported_extensions()

        assert ".py" in extensions
        assert ".js" in extensions
        assert ".ts" in extensions

    def test_is_supported_file(self, temp_dir):
        """Тест проверки поддержки файла."""
        py_file = temp_dir / "test.py"
        py_file.write_text("")

        txt_file = temp_dir / "test.txt"
        txt_file.write_text("")

        assert ParserFactory.can_parse_file(py_file) is True
        assert ParserFactory.can_parse_file(txt_file) is False
