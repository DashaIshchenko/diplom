"""
JavaScript парсер на основе tree-sitter.
Извлекает функции, классы, методы и их метаданные.
Поддерживает ES6+ синтаксис, включая arrow functions, async/await, и JSX.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

try:
    from tree_sitter import Language, Parser, Node

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("tree-sitter не установлен. JavaScript парсер будет недоступен.")

from ..base_parser import BaseParser
from ..code_structure import (
    CodeElement, CodeElementType, CodeLocation,
    ModuleInfo, ProgrammingLanguage
)
from ..utils.tree_sitter_helper import TreeSitterHelper
from ..utils.docstring_extractor import extract_jsdoc
from ..utils.complexity_calculator import calculate_complexity

logger = logging.getLogger(__name__)


class JavaScriptParser(BaseParser):
    """
    Парсер JavaScript кода на основе tree-sitter.

    Извлекает:
    - Функции (обычные и arrow functions)
    - Классы и методы
    - Async/await функции
    - JSDoc комментарии
    - Exports/Imports (ES6 modules)
    - JSX компоненты (React)
    """

    def __init__(self, max_chunk_size: int = 8192):
        """
        Инициализация JavaScript парсера.

        Args:
            max_chunk_size: Максимальный размер чанка
        """
        super().__init__(max_chunk_size)

        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter не установлен. Установите: pip install tree-sitter"
            )

        # Инициализация tree-sitter для JavaScript
        self.ts_helper = TreeSitterHelper()
        self.parser = self.ts_helper.get_parser('javascript')

        if not self.parser:
            raise RuntimeError("Не удалось инициализировать JavaScript парсер")

    @property
    def language(self) -> ProgrammingLanguage:
        """Язык программирования."""
        return ProgrammingLanguage.JAVASCRIPT

    @property
    def file_extensions(self) -> List[str]:
        """Поддерживаемые расширения файлов."""
        return ['.js', '.jsx', '.mjs', '.cjs']

    def parse_file(
            self,
            file_path: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None
    ) -> Optional[ModuleInfo]:
        """
        Парсинг JavaScript файла.

        Args:
            file_path: Путь к файлу
            repository_name: Имя репозитория
            branch: Ветка
            commit_hash: Хэш коммита
            provider: Провайдер

        Returns:
            ModuleInfo с извлеченными элементами или None при ошибке
        """
        try:
            # Читаем файл
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Парсим через tree-sitter
            tree = self.parser.parse(bytes(source_code, 'utf-8'))
            root_node = tree.root_node

            # Извлекаем имя модуля
            module_name = self._get_module_name(file_path)

            # Извлекаем imports и exports
            imports = self._extract_imports(root_node, source_code)

            # Создаем объект модуля
            module_info = ModuleInfo(
                file_path=file_path,
                module_name=module_name,
                language=self.language,
                docstring=None,  # JavaScript не имеет module-level docstrings
                imports=imports,
                total_lines=len(source_code.split('\n')),
                repository_name=repository_name,
                branch=branch,
                commit_hash=commit_hash,
                provider=provider
            )

            # Извлекаем функции и классы
            self._extract_elements(root_node, source_code, file_path, module_info)

            logger.info(
                f"Распарсен JavaScript файл {file_path.name}: "
                f"{len(module_info.functions)} функций, "
                f"{len(module_info.classes)} классов"
            )

            return module_info

        except Exception as e:
            logger.error(f"Ошибка парсинга JavaScript файла {file_path}: {e}")
            return None

    def _extract_elements(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            module_info: ModuleInfo,
            parent_class: Optional[str] = None
    ) -> None:
        """Рекурсивное извлечение элементов из AST."""
        for child in node.children:
            # Функции
            if child.type in ['function_declaration', 'function']:
                func_element = self._parse_function(child, source_code, file_path, parent_class)
                if func_element:
                    func_element.imports = module_info.imports
                    if parent_class:
                        # Это метод класса - будет добавлен в класс
                        pass
                    else:
                        module_info.functions.append(func_element)

            # Arrow functions (только если они экспортируются или присваиваются)
            elif child.type == 'lexical_declaration' or child.type == 'variable_declaration':
                arrow_func = self._parse_arrow_function(child, source_code, file_path, parent_class)
                if arrow_func:
                    arrow_func.imports = module_info.imports
                    if not parent_class:
                        module_info.functions.append(arrow_func)

            # Классы
            elif child.type == 'class_declaration':
                class_element = self._parse_class(child, source_code, file_path)
                if class_element:
                    class_element.imports = module_info.imports
                    module_info.classes.append(class_element)

            # Export statements
            elif child.type in ['export_statement', 'export_default_declaration']:
                self._handle_export(child, source_code, file_path, module_info)

            # Рекурсивно обрабатываем вложенные узлы
            else:
                self._extract_elements(child, source_code, file_path, module_info, parent_class)

    def _parse_function(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: Optional[str] = None
    ) -> Optional[CodeElement]:
        """Парсинг обычной функции."""
        try:
            # Имя функции
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            func_name = self._get_node_text(name_node, source_code)

            # Исходный код
            func_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # Проверяем async
            is_async = self._is_async_function(node, source_code)

            # JSDoc
            jsdoc = self._extract_jsdoc(node, source_code)

            # Параметры
            parameters = self._extract_function_parameters(node, source_code)

            # Определяем тип функции
            if parent_class:
                # Это метод
                if func_name == 'constructor':
                    element_type = CodeElementType.CONSTRUCTOR
                else:
                    element_type = CodeElementType.ASYNC_METHOD if is_async else CodeElementType.METHOD
            else:
                # Это функция на уровне модуля
                element_type = CodeElementType.ASYNC_FUNCTION if is_async else CodeElementType.FUNCTION

            # Сигнатура
            signature = self._build_function_signature(func_name, parameters, is_async)

            # Вычисляем сложность
            complexity = calculate_complexity(func_source, self.language)

            func_element = CodeElement(
                name=func_name,
                type=element_type,
                language=self.language,
                location=location,
                source_code=func_source,
                docstring=jsdoc,
                signature=signature,
                parent=parent_class,
                parameters=parameters,
                is_async=is_async,
                complexity=complexity,
                access_modifier='public'  # JavaScript не имеет модификаторов (до приватных полей)
            )

            return func_element

        except Exception as e:
            logger.error(f"Ошибка парсинга JavaScript функции: {e}")
            return None

    def _parse_arrow_function(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: Optional[str] = None
    ) -> Optional[CodeElement]:
        """Парсинг arrow function."""
        try:
            # Ищем arrow function в объявлении переменной
            # const myFunc = (x) => { ... }

            declarator = None
            for child in node.children:
                if child.type == 'variable_declarator':
                    declarator = child
                    break

            if not declarator:
                return None

            # Имя функции (имя переменной)
            name_node = declarator.child_by_field_name('name')
            if not name_node:
                return None
            func_name = self._get_node_text(name_node, source_code)

            # Проверяем, что значение - это arrow function
            value_node = declarator.child_by_field_name('value')
            if not value_node or value_node.type != 'arrow_function':
                return None

            # Исходный код
            func_source = self._get_node_text(value_node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=value_node.start_point[0] + 1,
                end_line=value_node.end_point[0] + 1,
                start_col=value_node.start_point[1],
                end_col=value_node.end_point[1]
            )

            # Проверяем async
            is_async = self._is_async_function(value_node, source_code)

            # JSDoc (для всего объявления переменной)
            jsdoc = self._extract_jsdoc(node, source_code)

            # Параметры
            parameters = self._extract_arrow_function_parameters(value_node, source_code)

            # Тип элемента
            element_type = CodeElementType.ASYNC_FUNCTION if is_async else CodeElementType.FUNCTION

            # Сигнатура
            signature = self._build_function_signature(func_name, parameters, is_async, arrow=True)

            # Сложность
            complexity = calculate_complexity(func_source, self.language)

            func_element = CodeElement(
                name=func_name,
                type=element_type,
                language=self.language,
                location=location,
                source_code=func_source,
                docstring=jsdoc,
                signature=signature,
                parent=parent_class,
                parameters=parameters,
                is_async=is_async,
                complexity=complexity,
                access_modifier='public'
            )

            return func_element

        except Exception as e:
            logger.debug(f"Не удалось распарсить arrow function: {e}")
            return None

    def _parse_class(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг JavaScript класса."""
        try:
            # Имя класса
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            class_name = self._get_node_text(name_node, source_code)

            # Исходный код
            class_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # JSDoc
            jsdoc = self._extract_jsdoc(node, source_code)

            # Базовый класс (extends)
            base_classes = []
            heritage_node = node.child_by_field_name('heritage')
            if heritage_node:
                base_class = self._get_node_text(heritage_node, source_code)
                base_class = base_class.replace('extends', '').strip()
                base_classes.append(base_class)

            # Сигнатура
            signature = f"class {class_name}"
            if base_classes:
                signature += f" extends {base_classes[0]}"

            # Создаем элемент класса
            class_element = CodeElement(
                name=class_name,
                type=CodeElementType.CLASS,
                language=self.language,
                location=location,
                source_code=class_source,
                docstring=jsdoc,
                signature=signature,
                base_classes=base_classes,
                methods=[],
                access_modifier='public'
            )

            # Парсим методы класса
            body_node = node.child_by_field_name('body')
            if body_node:
                for child in body_node.children:
                    if child.type == 'method_definition':
                        method = self._parse_method(child, source_code, file_path, class_name)
                        if method:
                            class_element.methods.append(method)

                    # Приватные поля (ES2022+)
                    elif child.type == 'field_definition':
                        field_name = self._extract_field_name(child, source_code)
                        if field_name:
                            if not class_element.attributes:
                                class_element.attributes = []
                            class_element.attributes.append(field_name)

            return class_element

        except Exception as e:
            logger.error(f"Ошибка парсинга JavaScript класса: {e}")
            return None

    def _parse_method(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: str
    ) -> Optional[CodeElement]:
        """Парсинг метода класса."""
        try:
            # Имя метода
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            method_name = self._get_node_text(name_node, source_code)

            # Исходный код
            method_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # Проверяем модификаторы
            is_static = self._is_static_method(node, source_code)
            is_async = self._is_async_function(node, source_code)
            is_private = method_name.startswith('#')  # Приватные методы (ES2022+)

            # JSDoc
            jsdoc = self._extract_jsdoc(node, source_code)

            # Параметры
            parameters = self._extract_function_parameters(node, source_code)

            # Тип метода
            if method_name == 'constructor':
                element_type = CodeElementType.CONSTRUCTOR
            elif is_static:
                element_type = CodeElementType.STATIC_METHOD
            else:
                element_type = CodeElementType.ASYNC_METHOD if is_async else CodeElementType.METHOD

            # Сигнатура
            signature = self._build_method_signature(
                method_name, parameters, is_async, is_static
            )

            # Сложность
            complexity = calculate_complexity(method_source, self.language)

            method_element = CodeElement(
                name=method_name,
                type=element_type,
                language=self.language,
                location=location,
                source_code=method_source,
                docstring=jsdoc,
                signature=signature,
                parent=parent_class,
                parameters=parameters,
                is_async=is_async,
                is_static=is_static,
                complexity=complexity,
                access_modifier='private' if is_private else 'public'
            )

            return method_element

        except Exception as e:
            logger.error(f"Ошибка парсинга JavaScript метода: {e}")
            return None

    def _extract_imports(self, node: Node, source_code: str) -> List[str]:
        """Извлечение import statements."""
        imports = []

        for child in node.children:
            if child.type == 'import_statement':
                import_text = self._get_node_text(child, source_code)
                # Упрощаем: просто берем имена импортов
                # import { x, y } from 'module' -> module.x, module.y
                imports.append(import_text)

            # require() для CommonJS
            elif child.type == 'variable_declaration':
                declarator = child.child_by_field_name('declarator')
                if declarator:
                    value = declarator.child_by_field_name('value')
                    if value and 'require' in self._get_node_text(value, source_code):
                        require_text = self._get_node_text(value, source_code)
                        imports.append(require_text)

        return imports

    def _extract_function_parameters(self, node: Node, source_code: str) -> List[Dict[str, Any]]:
        """Извлечение параметров функции."""
        parameters = []

        params_node = node.child_by_field_name('parameters')
        if params_node:
            for child in params_node.children:
                if child.type in ['identifier', 'required_parameter', 'optional_parameter']:
                    param_name = self._get_node_text(child, source_code)
                    parameters.append({
                        'name': param_name,
                        'type': None,  # JavaScript не имеет обязательных типов
                        'kind': 'positional'
                    })

                # Rest параметр (...args)
                elif child.type == 'rest_pattern':
                    param_name = self._get_node_text(child, source_code)
                    parameters.append({
                        'name': param_name,
                        'type': None,
                        'kind': 'rest'
                    })

        return parameters

    def _extract_arrow_function_parameters(self, node: Node, source_code: str) -> List[Dict[str, Any]]:
        """Извлечение параметров arrow function."""
        parameters = []

        # Arrow function может иметь параметры в разных форматах
        # (x) => { ... }
        # x => { ... }
        # (x, y) => { ... }

        for child in node.children:
            if child.type == 'formal_parameters':
                for param_child in child.children:
                    if param_child.type in ['identifier', 'required_parameter']:
                        param_name = self._get_node_text(param_child, source_code)
                        parameters.append({
                            'name': param_name,
                            'type': None,
                            'kind': 'positional'
                        })

            # Одиночный параметр без скобок
            elif child.type == 'identifier' and not parameters:
                param_name = self._get_node_text(child, source_code)
                parameters.append({
                    'name': param_name,
                    'type': None,
                    'kind': 'positional'
                })
                break

        return parameters

    def _extract_field_name(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение имени поля класса."""
        property_node = node.child_by_field_name('property')
        if property_node:
            return self._get_node_text(property_node, source_code)
        return None

    def _is_async_function(self, node: Node, source_code: str) -> bool:
        """Проверка, является ли функция async."""
        # Проверяем наличие async ключевого слова перед функцией
        text = self._get_node_text(node, source_code)
        return text.strip().startswith('async ')

    def _is_static_method(self, node: Node, source_code: str) -> bool:
        """Проверка, является ли метод статическим."""
        text = self._get_node_text(node, source_code)
        return text.strip().startswith('static ')

    def _extract_jsdoc(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение JSDoc комментария."""
        return extract_jsdoc(node, source_code)

    def _build_function_signature(
            self,
            name: str,
            parameters: List[Dict],
            is_async: bool,
            arrow: bool = False
    ) -> str:
        """Построение сигнатуры функции."""
        params = ", ".join([p['name'] for p in parameters])

        if arrow:
            prefix = "async " if is_async else ""
            return f"{prefix}const {name} = ({params}) => {{ ... }}"
        else:
            prefix = "async function" if is_async else "function"
            return f"{prefix} {name}({params})"

    def _build_method_signature(
            self,
            name: str,
            parameters: List[Dict],
            is_async: bool,
            is_static: bool
    ) -> str:
        """Построение сигнатуры метода."""
        params = ", ".join([p['name'] for p in parameters])

        parts = []
        if is_static:
            parts.append("static")
        if is_async:
            parts.append("async")
        parts.append(f"{name}({params})")

        return " ".join(parts)

    def _handle_export(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            module_info: ModuleInfo
    ) -> None:
        """Обработка export statements."""
        # Извлекаем экспортируемые функции/классы
        declaration = node.child_by_field_name('declaration')
        if declaration:
            if declaration.type == 'function_declaration':
                func = self._parse_function(declaration, source_code, file_path)
                if func:
                    func.imports = module_info.imports
                    module_info.functions.append(func)

            elif declaration.type == 'class_declaration':
                cls = self._parse_class(declaration, source_code, file_path)
                if cls:
                    cls.imports = module_info.imports
                    module_info.classes.append(cls)

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Получение текста узла."""
        return source_code[node.start_byte:node.end_byte]


# Удобная функция для быстрого парсинга
def parse_javascript_file(file_path: str, **kwargs) -> Optional[ModuleInfo]:
    """
    Быстрый парсинг JavaScript файла.

    Args:
        file_path: Путь к JavaScript файлу
        **kwargs: Дополнительные параметры

    Returns:
        ModuleInfo или None
    """
    parser = JavaScriptParser()
    return parser.parse_file(Path(file_path), **kwargs)
