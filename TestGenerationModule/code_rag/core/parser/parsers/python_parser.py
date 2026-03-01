"""
Python парсер на основе AST (Abstract Syntax Tree).
Извлекает функции, классы, методы и их метаданные.
"""

import ast
from typing import List, Optional, Set, Any, Dict
from pathlib import Path
import logging

from ..base_parser import BaseParser
from ..code_structure import (
    CodeElement, CodeElementType, CodeLocation,
    ModuleInfo, ProgrammingLanguage
)

logger = logging.getLogger(__name__)


class PythonParser(BaseParser):
    """
    Парсер Python кода на основе встроенного модуля ast.

    Извлекает:
    - Функции и их сигнатуры
    - Классы и их методы
    - Docstrings
    - Декораторы
    - Параметры и типы (type hints)
    - Зависимости (imports)
    """

    @property
    def language(self) -> ProgrammingLanguage:
        """Язык программирования."""
        return ProgrammingLanguage.PYTHON

    @property
    def file_extensions(self) -> List[str]:
        """Поддерживаемые расширения файлов."""
        return ['.py', '.pyi']

    def parse_file(
            self,
            file_path: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None
    ) -> Optional[ModuleInfo]:
        """
        Парсинг Python файла.

        Args:
            file_path: Путь к файлу
            repository_name: Имя репозитория
            branch: Ветка
            commit_hash: Хэш коммита
            provider: Провайдер (github, gitlab, azure_devops)

        Returns:
            ModuleInfo с извлеченными элементами или None при ошибке
        """
        try:
            # Читаем файл
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Парсим AST
            tree = ast.parse(source_code, filename=str(file_path))

            # Извлекаем имя модуля
            module_name = self._get_module_name(file_path)

            # Извлекаем docstring модуля
            module_docstring = ast.get_docstring(tree)

            # Извлекаем импорты
            imports = self._extract_imports(tree)

            # Создаем объект модуля
            module_info = ModuleInfo(
                file_path=file_path,
                module_name=module_name,
                language=self.language,
                docstring=module_docstring,
                imports=imports,
                total_lines=len(source_code.split('\n')),
                repository_name=repository_name,
                branch=branch,
                commit_hash=commit_hash,
                provider=provider
            )

            # Извлекаем функции и классы
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Функция на уровне модуля
                    func_element = self._parse_function(
                        node, source_code, file_path, parent=None
                    )
                    if func_element:
                        func_element.imports = imports
                        module_info.functions.append(func_element)

                elif isinstance(node, ast.ClassDef):
                    # Класс
                    class_element = self._parse_class(
                        node, source_code, file_path
                    )
                    if class_element:
                        class_element.imports = imports
                        module_info.classes.append(class_element)

            logger.info(
                f"Распарсен Python файл {file_path.name}: "
                f"{len(module_info.functions)} функций, "
                f"{len(module_info.classes)} классов"
            )

            return module_info

        except SyntaxError as e:
            logger.error(f"Синтаксическая ошибка в {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка парсинга {file_path}: {e}")
            return None

    def _parse_function(
            self,
            node: ast.FunctionDef,
            source_code: str,
            file_path: Path,
            parent: Optional[str] = None
    ) -> Optional[CodeElement]:
        """
        Парсинг функции или метода.

        Args:
            node: AST узел функции
            source_code: Исходный код файла
            file_path: Путь к файлу
            parent: Имя родительского класса (для методов)

        Returns:
            CodeElement с информацией о функции
        """
        try:
            # Определяем тип элемента
            is_async = isinstance(node, ast.AsyncFunctionDef)

            if parent:
                # Это метод класса
                element_type = self._determine_method_type(node, is_async)

                # Определяем модификаторы
                is_static = any(
                    self._get_decorator_name(dec) == "staticmethod"
                    for dec in node.decorator_list
                )
            else:
                # Это функция на уровне модуля
                element_type = (CodeElementType.ASYNC_FUNCTION if is_async
                                else CodeElementType.FUNCTION)
                is_static = False

            # Извлекаем исходный код функции
            func_source = ast.get_source_segment(source_code, node)
            if not func_source:
                func_source = self._extract_source_lines(
                    source_code, node.lineno, node.end_lineno
                )

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                start_col=node.col_offset,
                end_col=node.end_col_offset or 0
            )

            # Docstring
            docstring = ast.get_docstring(node)

            # Сигнатура
            signature = self._build_function_signature(node)

            # Декораторы
            decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]

            # Параметры
            parameters = self._extract_parameters(node.args)

            # Тип возвращаемого значения
            return_type = None
            if node.returns:
                return_type = ast.unparse(node.returns)

            # Вызываемые функции
            called_functions = self._extract_function_calls(node)

            # Вычисляем сложность
            complexity = self._calculate_complexity(node)

            # Модификатор доступа (для методов)
            access_modifier = None
            if parent:
                if node.name.startswith('__') and not node.name.endswith('__'):
                    access_modifier = "private"
                elif node.name.startswith('_'):
                    access_modifier = "protected"
                else:
                    access_modifier = "public"

            element = CodeElement(
                name=node.name,
                type=element_type,
                language=self.language,
                location=location,
                source_code=func_source,
                docstring=docstring,
                signature=signature,
                decorators=decorators,
                parent=parent,
                parameters=parameters,
                return_type=return_type,
                is_async=is_async,
                is_static=is_static,
                access_modifier=access_modifier,
                called_functions=called_functions,
                complexity=complexity
            )

            return element

        except Exception as e:
            logger.error(f"Ошибка парсинга функции {node.name}: {e}")
            return None

    def _parse_class(
            self,
            node: ast.ClassDef,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """
        Парсинг класса.

        Args:
            node: AST узел класса
            source_code: Исходный код файла
            file_path: Путь к файлу

        Returns:
            CodeElement с информацией о классе
        """
        try:
            # Извлекаем исходный код класса
            class_source = ast.get_source_segment(source_code, node)
            if not class_source:
                class_source = self._extract_source_lines(
                    source_code, node.lineno, node.end_lineno
                )

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                start_col=node.col_offset,
                end_col=node.end_col_offset or 0
            )

            # Docstring
            docstring = ast.get_docstring(node)

            # Базовые классы
            base_classes = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_classes.append(ast.unparse(base))

            # Декораторы
            decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]

            # Атрибуты класса
            attributes = self._extract_class_attributes(node)

            # Создаем элемент класса
            class_element = CodeElement(
                name=node.name,
                type=CodeElementType.CLASS,
                language=self.language,
                location=location,
                source_code=class_source,
                docstring=docstring,
                signature=f"class {node.name}({', '.join(base_classes)})" if base_classes else f"class {node.name}",
                decorators=decorators,
                base_classes=base_classes,
                attributes=attributes,
                methods=[],
                access_modifier="public"  # Python классы всегда public
            )

            # Парсим методы класса
            for child_node in ast.iter_child_nodes(node):
                if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method = self._parse_function(
                        child_node, source_code, file_path, parent=node.name
                    )
                    if method:
                        class_element.methods.append(method)

            return class_element

        except Exception as e:
            logger.error(f"Ошибка парсинга класса {node.name}: {e}")
            return None

    def _extract_imports(self, tree: ast.Module) -> List[str]:
        """Извлечение всех импортов из модуля."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        return list(set(imports))  # Убираем дубликаты

    def _extract_parameters(self, args: ast.arguments) -> List[Dict[str, Any]]:
        """Извлечение параметров функции."""
        parameters = []

        # Обычные аргументы
        for arg in args.args:
            param_type = None
            if arg.annotation:
                param_type = ast.unparse(arg.annotation)

            parameters.append({
                "name": arg.arg,
                "type": param_type,
                "kind": "positional"
            })

        # *args
        if args.vararg:
            param_type = None
            if args.vararg.annotation:
                param_type = ast.unparse(args.vararg.annotation)

            parameters.append({
                "name": f"*{args.vararg.arg}",
                "type": param_type,
                "kind": "var_positional"
            })

        # **kwargs
        if args.kwarg:
            param_type = None
            if args.kwarg.annotation:
                param_type = ast.unparse(args.kwarg.annotation)

            parameters.append({
                "name": f"**{args.kwarg.arg}",
                "type": param_type,
                "kind": "var_keyword"
            })

        return parameters

    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Извлечение вызовов функций внутри функции."""
        calls = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(ast.unparse(child.func))

        return list(set(calls))

    def _extract_class_attributes(self, node: ast.ClassDef) -> List[str]:
        """Извлечение атрибутов класса."""
        attributes = []

        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
            elif isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Name):
                    attributes.append(child.target.id)

        return list(set(attributes))

    def _build_function_signature(self, node: ast.FunctionDef) -> str:
        """Построение сигнатуры функции."""
        try:
            params = []

            # Обычные аргументы
            for arg in node.args.args:
                if arg.annotation:
                    params.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
                else:
                    params.append(arg.arg)

            # *args
            if node.args.vararg:
                if node.args.vararg.annotation:
                    params.append(f"*{node.args.vararg.arg}: {ast.unparse(node.args.vararg.annotation)}")
                else:
                    params.append(f"*{node.args.vararg.arg}")

            # **kwargs
            if node.args.kwarg:
                if node.args.kwarg.annotation:
                    params.append(f"**{node.args.kwarg.arg}: {ast.unparse(node.args.kwarg.annotation)}")
                else:
                    params.append(f"**{node.args.kwarg.arg}")

            # Формируем сигнатуру
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            signature = f"{prefix} {node.name}({', '.join(params)})"

            # Добавляем return type
            if node.returns:
                signature += f" -> {ast.unparse(node.returns)}"

            return signature

        except Exception:
            return f"def {node.name}(...)"

    def _determine_method_type(
            self,
            node: ast.FunctionDef,
            is_async: bool
    ) -> CodeElementType:
        """Определение типа метода (обычный, статический, классовый, property)."""
        for decorator in node.decorator_list:
            dec_name = self._get_decorator_name(decorator)

            if dec_name == "staticmethod":
                return CodeElementType.STATIC_METHOD
            elif dec_name == "classmethod":
                return CodeElementType.CLASS_METHOD
            elif dec_name == "property":
                return CodeElementType.PROPERTY

        # Обычный метод
        return CodeElementType.ASYNC_METHOD if is_async else CodeElementType.METHOD

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Получение имени декоратора."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            return ast.unparse(decorator.func)
        return ast.unparse(decorator)

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """
        Упрощенный расчет цикломатической сложности.
        Считает количество путей выполнения.
        """
        complexity = 1  # Базовая сложность

        for child in ast.walk(node):
            # Условные операторы
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            # Логические операторы
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            # Comprehensions
            elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1

        return complexity

    def _extract_source_lines(
            self,
            source_code: str,
            start_line: int,
            end_line: Optional[int]
    ) -> str:
        """Извлечение строк кода по номерам."""
        lines = source_code.split('\n')
        if end_line is None:
            end_line = start_line

        # Корректируем индексы (линии в AST начинаются с 1)
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        return '\n'.join(lines[start_idx:end_idx])


# Удобная функция для быстрого парсинга
def parse_python_file(file_path: str, **kwargs) -> Optional[ModuleInfo]:
    """
    Быстрый парсинг Python файла.

    Args:
        file_path: Путь к Python файлу
        **kwargs: Дополнительные параметры

    Returns:
        ModuleInfo или None
    """
    parser = PythonParser()
    return parser.parse_file(Path(file_path), **kwargs)
