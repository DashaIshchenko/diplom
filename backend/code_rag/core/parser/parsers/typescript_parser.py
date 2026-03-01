"""
TypeScript парсер на основе tree-sitter.
Извлекает функции, классы, интерфейсы, типы и их метаданные.
Поддерживает TypeScript и TSX (React) синтаксис.
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
    logger.warning("tree-sitter не установлен. TypeScript парсер будет недоступен.")

from ..base_parser import BaseParser
from ..code_structure import (
    CodeElement, CodeElementType, CodeLocation,
    ModuleInfo, ProgrammingLanguage
)
from ..utils.tree_sitter_helper import TreeSitterHelper
from ..utils.docstring_extractor import extract_jsdoc
from ..utils.complexity_calculator import calculate_complexity

logger = logging.getLogger(__name__)


class TypeScriptParser(BaseParser):
    """
    Парсер TypeScript кода на основе tree-sitter.

    Извлекает:
    - Функции (обычные, arrow, async)
    - Классы и методы
    - Интерфейсы
    - Type aliases
    - Enums
    - Generics
    - Decorators (@Component, @Injectable и т.д.)
    - JSDoc/TSDoc комментарии
    - TSX компоненты (React)
    """

    def __init__(self, max_chunk_size: int = 8192):
        """
        Инициализация TypeScript парсера.

        Args:
            max_chunk_size: Максимальный размер чанка
        """
        super().__init__(max_chunk_size)

        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter не установлен. Установите: pip install tree-sitter"
            )

        # Инициализация tree-sitter для TypeScript
        self.ts_helper = TreeSitterHelper()

        # TypeScript имеет два парсера: typescript и tsx
        self.parser_ts = self.ts_helper.get_parser('typescript')
        self.parser_tsx = self.ts_helper.get_parser('tsx')

        if not self.parser_ts:
            raise RuntimeError("Не удалось инициализировать TypeScript парсер")

    @property
    def language(self) -> ProgrammingLanguage:
        """Язык программирования."""
        return ProgrammingLanguage.TYPESCRIPT

    @property
    def file_extensions(self) -> List[str]:
        """Поддерживаемые расширения файлов."""
        return ['.ts', '.tsx', '.mts', '.cts']

    def parse_file(
            self,
            file_path: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None
    ) -> Optional[ModuleInfo]:
        """
        Парсинг TypeScript файла.

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

            # Выбираем парсер в зависимости от расширения
            if file_path.suffix == '.tsx':
                parser = self.parser_tsx
            else:
                parser = self.parser_ts

            # Парсим через tree-sitter
            tree = parser.parse(bytes(source_code, 'utf-8'))
            root_node = tree.root_node

            # Извлекаем имя модуля
            module_name = self._get_module_name(file_path)

            # Извлекаем imports
            imports = self._extract_imports(root_node, source_code)

            # Создаем объект модуля
            module_info = ModuleInfo(
                file_path=file_path,
                module_name=module_name,
                language=self.language,
                docstring=None,
                imports=imports,
                total_lines=len(source_code.split('\n')),
                repository_name=repository_name,
                branch=branch,
                commit_hash=commit_hash,
                provider=provider
            )

            # Извлекаем элементы
            self._extract_elements(root_node, source_code, file_path, module_info)

            logger.info(
                f"Распарсен TypeScript файл {file_path.name}: "
                f"{len(module_info.functions)} функций, "
                f"{len(module_info.classes)} классов, "
                f"{len(module_info.interfaces)} интерфейсов"
            )

            return module_info

        except Exception as e:
            logger.error(f"Ошибка парсинга TypeScript файла {file_path}: {e}")
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
            if child.type in ['function_declaration', 'function_signature']:
                func_element = self._parse_function(child, source_code, file_path, parent_class)
                if func_element:
                    func_element.imports = module_info.imports
                    if not parent_class:
                        module_info.functions.append(func_element)

            # Arrow functions
            elif child.type in ['lexical_declaration', 'variable_declaration']:
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

            # Интерфейсы
            elif child.type == 'interface_declaration':
                interface_element = self._parse_interface(child, source_code, file_path)
                if interface_element:
                    interface_element.imports = module_info.imports
                    module_info.interfaces.append(interface_element)

            # Enums
            elif child.type == 'enum_declaration':
                enum_element = self._parse_enum(child, source_code, file_path)
                if enum_element:
                    enum_element.imports = module_info.imports
                    module_info.enums.append(enum_element)

            # Type aliases
            elif child.type == 'type_alias_declaration':
                # Можно добавить в будущем как отдельный тип
                pass

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

            # TSDoc/JSDoc
            tsdoc = self._extract_jsdoc(node, source_code)

            # Decorators (для Angular, NestJS и т.д.)
            decorators = self._extract_decorators(node, source_code)

            # Параметры с типами
            parameters = self._extract_function_parameters(node, source_code)

            # Return type
            return_type = self._extract_return_type(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Определяем тип функции
            if parent_class:
                if func_name == 'constructor':
                    element_type = CodeElementType.CONSTRUCTOR
                else:
                    element_type = CodeElementType.ASYNC_METHOD if is_async else CodeElementType.METHOD
            else:
                element_type = CodeElementType.ASYNC_FUNCTION if is_async else CodeElementType.FUNCTION

            # Сигнатура
            signature = self._build_function_signature(
                func_name, parameters, return_type, generics, is_async
            )

            # Сложность
            complexity = calculate_complexity(func_source, self.language)

            func_element = CodeElement(
                name=func_name,
                type=element_type,
                language=self.language,
                location=location,
                source_code=func_source,
                docstring=tsdoc,
                signature=signature,
                decorators=decorators,
                parent=parent_class,
                parameters=parameters,
                return_type=return_type,
                generics=generics,
                is_async=is_async,
                complexity=complexity,
                access_modifier='public'
            )

            return func_element

        except Exception as e:
            logger.error(f"Ошибка парсинга TypeScript функции: {e}")
            return None

    def _parse_arrow_function(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: Optional[str] = None
    ) -> Optional[CodeElement]:
        """Парсинг arrow function (аналогично JavaScript)."""
        try:
            declarator = None
            for child in node.children:
                if child.type == 'variable_declarator':
                    declarator = child
                    break

            if not declarator:
                return None

            # Имя функции
            name_node = declarator.child_by_field_name('name')
            if not name_node:
                return None
            func_name = self._get_node_text(name_node, source_code)

            # Проверяем, что значение - arrow function
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

            # Async
            is_async = self._is_async_function(value_node, source_code)

            # TSDoc
            tsdoc = self._extract_jsdoc(node, source_code)

            # Параметры
            parameters = self._extract_arrow_function_parameters(value_node, source_code)

            # Return type (для arrow functions может быть указан)
            return_type = self._extract_arrow_return_type(value_node, source_code)

            # Сигнатура
            signature = self._build_function_signature(
                func_name, parameters, return_type, [], is_async, arrow=True
            )

            # Сложность
            complexity = calculate_complexity(func_source, self.language)

            func_element = CodeElement(
                name=func_name,
                type=CodeElementType.ASYNC_FUNCTION if is_async else CodeElementType.FUNCTION,
                language=self.language,
                location=location,
                source_code=func_source,
                docstring=tsdoc,
                signature=signature,
                parent=parent_class,
                parameters=parameters,
                return_type=return_type,
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
        """Парсинг TypeScript класса."""
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

            # TSDoc
            tsdoc = self._extract_jsdoc(node, source_code)

            # Decorators (для Angular @Component, NestJS @Injectable и т.д.)
            decorators = self._extract_decorators(node, source_code)

            # Модификаторы (abstract, readonly)
            modifiers = self._extract_class_modifiers(node, source_code)
            is_abstract = 'abstract' in modifiers

            # Базовый класс (extends)
            base_classes = []
            heritage_node = node.child_by_field_name('heritage')
            if heritage_node:
                extends = self._extract_extends_clause(heritage_node, source_code)
                if extends:
                    base_classes.append(extends)

            # Интерфейсы (implements)
            interfaces = []
            if heritage_node:
                implements = self._extract_implements_clause(heritage_node, source_code)
                interfaces.extend(implements)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Сигнатура
            signature = self._build_class_signature(
                class_name, base_classes, interfaces, generics, is_abstract
            )

            # Создаем элемент класса
            class_element = CodeElement(
                name=class_name,
                type=CodeElementType.CLASS,
                language=self.language,
                location=location,
                source_code=class_source,
                docstring=tsdoc,
                signature=signature,
                decorators=decorators,
                base_classes=base_classes,
                interfaces=interfaces,
                generics=generics,
                methods=[],
                is_abstract=is_abstract,
                access_modifier='public'
            )

            # Парсим методы и свойства
            body_node = node.child_by_field_name('body')
            if body_node:
                for child in body_node.children:
                    if child.type in ['method_definition', 'method_signature']:
                        method = self._parse_method(child, source_code, file_path, class_name)
                        if method:
                            class_element.methods.append(method)

                    # Свойства класса
                    elif child.type in ['public_field_definition', 'property_declaration']:
                        field_name = self._extract_field_name(child, source_code)
                        if field_name:
                            if not class_element.attributes:
                                class_element.attributes = []
                            class_element.attributes.append(field_name)

            return class_element

        except Exception as e:
            logger.error(f"Ошибка парсинга TypeScript класса: {e}")
            return None

    def _parse_interface(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг TypeScript интерфейса."""
        try:
            # Имя интерфейса
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            interface_name = self._get_node_text(name_node, source_code)

            # Исходный код
            interface_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # TSDoc
            tsdoc = self._extract_jsdoc(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Extends (интерфейсы могут наследовать другие интерфейсы)
            base_interfaces = []
            heritage_node = node.child_by_field_name('heritage')
            if heritage_node:
                extends = self._extract_interface_extends(heritage_node, source_code)
                base_interfaces.extend(extends)

            # Сигнатура
            signature = f"interface {interface_name}"
            if generics:
                signature += f"<{', '.join(generics)}>"
            if base_interfaces:
                signature += f" extends {', '.join(base_interfaces)}"

            interface_element = CodeElement(
                name=interface_name,
                type=CodeElementType.INTERFACE,
                language=self.language,
                location=location,
                source_code=interface_source,
                docstring=tsdoc,
                signature=signature,
                generics=generics,
                interfaces=base_interfaces,
                methods=[],
                is_abstract=True,
                access_modifier='public'
            )

            # Парсим методы интерфейса
            body_node = node.child_by_field_name('body')
            if body_node:
                for child in body_node.children:
                    if child.type in ['method_signature', 'property_signature']:
                        method = self._parse_interface_method(child, source_code, file_path, interface_name)
                        if method:
                            interface_element.methods.append(method)

            return interface_element

        except Exception as e:
            logger.error(f"Ошибка парсинга TypeScript интерфейса: {e}")
            return None

    def _parse_enum(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг TypeScript enum."""
        try:
            # Имя enum
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            enum_name = self._get_node_text(name_node, source_code)

            # Исходный код
            enum_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # TSDoc
            tsdoc = self._extract_jsdoc(node, source_code)

            # Модификаторы (const enum)
            modifiers = self._extract_class_modifiers(node, source_code)
            is_const = 'const' in modifiers

            signature = f"{'const ' if is_const else ''}enum {enum_name}"

            enum_element = CodeElement(
                name=enum_name,
                type=CodeElementType.ENUM,
                language=self.language,
                location=location,
                source_code=enum_source,
                docstring=tsdoc,
                signature=signature,
                is_final=is_const,
                access_modifier='public'
            )

            return enum_element

        except Exception as e:
            logger.error(f"Ошибка парсинга TypeScript enum: {e}")
            return None

    def _parse_method(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: str
    ) -> Optional[CodeElement]:
        """Парсинг метода класса (аналогично JavaScript с добавлением типов)."""
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

            # Модификаторы
            modifiers = self._extract_method_modifiers(node, source_code)
            is_static = 'static' in modifiers
            is_async = 'async' in modifiers
            is_private = 'private' in modifiers
            is_protected = 'protected' in modifiers
            is_readonly = 'readonly' in modifiers
            is_abstract = 'abstract' in modifiers

            # Access modifier
            if is_private:
                access_modifier = 'private'
            elif is_protected:
                access_modifier = 'protected'
            else:
                access_modifier = 'public'

            # TSDoc
            tsdoc = self._extract_jsdoc(node, source_code)

            # Decorators
            decorators = self._extract_decorators(node, source_code)

            # Параметры
            parameters = self._extract_function_parameters(node, source_code)

            # Return type
            return_type = self._extract_return_type(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Тип метода
            if method_name == 'constructor':
                element_type = CodeElementType.CONSTRUCTOR
            elif is_static:
                element_type = CodeElementType.STATIC_METHOD
            else:
                element_type = CodeElementType.ASYNC_METHOD if is_async else CodeElementType.METHOD

            # Сигнатура
            signature = self._build_method_signature(
                method_name, parameters, return_type, generics, modifiers
            )

            # Сложность
            complexity = calculate_complexity(method_source, self.language)

            method_element = CodeElement(
                name=method_name,
                type=element_type,
                language=self.language,
                location=location,
                source_code=method_source,
                docstring=tsdoc,
                signature=signature,
                decorators=decorators,
                parent=parent_class,
                parameters=parameters,
                return_type=return_type,
                generics=generics,
                is_async=is_async,
                is_static=is_static,
                is_abstract=is_abstract,
                is_final=is_readonly,
                access_modifier=access_modifier,
                complexity=complexity
            )

            return method_element

        except Exception as e:
            logger.error(f"Ошибка парсинга TypeScript метода: {e}")
            return None

    def _parse_interface_method(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_interface: str
    ) -> Optional[CodeElement]:
        """Парсинг метода интерфейса (только сигнатура)."""
        # Упрощенная версия метода без тела
        try:
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None

            method_name = self._get_node_text(name_node, source_code)
            method_source = self._get_node_text(node, source_code)

            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            parameters = self._extract_function_parameters(node, source_code)
            return_type = self._extract_return_type(node, source_code)

            # Формируем сигнатуру
            param_parts = []
            for p in parameters:
                param_type = p.get('type', 'any')
                param_parts.append(f"{p['name']}: {param_type}")

            signature = f"{method_name}({', '.join(param_parts)})"
            if return_type:
                signature += f": {return_type}"

            return CodeElement(
                name=method_name,
                type=CodeElementType.METHOD,
                language=self.language,
                location=location,
                source_code=method_source,
                signature=signature,
                parent=parent_interface,
                parameters=parameters,
                return_type=return_type,
                is_abstract=True,
                access_modifier='public'
            )

        except Exception as e:
            logger.debug(f"Не удалось распарсить метод интерфейса: {e}")
            return None

    def _extract_imports(self, node: Node, source_code: str) -> List[str]:
        """Извлечение import statements (аналогично JavaScript)."""
        imports = []

        for child in node.children:
            if child.type == 'import_statement':
                import_text = self._get_node_text(child, source_code)
                imports.append(import_text)

        return imports

    def _extract_function_parameters(self, node: Node, source_code: str) -> List[Dict[str, Any]]:
        """Извлечение параметров функции с типами."""
        parameters = []

        params_node = node.child_by_field_name('parameters')
        if params_node:
            for child in params_node.children:
                if child.type in ['required_parameter', 'optional_parameter']:
                    param_name_node = child.child_by_field_name('pattern')
                    param_type_node = child.child_by_field_name('type')

                    if param_name_node:
                        param_name = self._get_node_text(param_name_node, source_code)
                        param_type = None
                        if param_type_node:
                            param_type = self._get_node_text(param_type_node, source_code)
                            param_type = param_type.lstrip(':').strip()

                        parameters.append({
                            'name': param_name,
                            'type': param_type,
                            'kind': 'optional' if child.type == 'optional_parameter' else 'positional'
                        })

                # Rest параметр (...args)
                elif child.type == 'rest_pattern':
                    param_name = self._get_node_text(child, source_code)
                    parameters.append({
                        'name': param_name,
                        'type': 'any[]',
                        'kind': 'rest'
                    })

        return parameters

    def _extract_arrow_function_parameters(self, node: Node, source_code: str) -> List[Dict[str, Any]]:
        """Извлечение параметров arrow function с типами."""
        # Аналогично _extract_function_parameters
        return self._extract_function_parameters(node, source_code)

    def _extract_return_type(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение типа возвращаемого значения."""
        type_node = node.child_by_field_name('return_type')
        if type_node:
            type_text = self._get_node_text(type_node, source_code)
            return type_text.lstrip(':').strip()
        return None

    def _extract_arrow_return_type(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение return type для arrow function."""
        # Arrow function может иметь тип между ) и =>
        # const func = (x: number): string => { ... }
        return self._extract_return_type(node, source_code)

    def _extract_generics(self, node: Node, source_code: str) -> List[str]:
        """Извлечение generic параметров."""
        generics = []

        type_params_node = node.child_by_field_name('type_parameters')
        if type_params_node:
            for child in type_params_node.children:
                if child.type == 'type_parameter':
                    generic_name = self._get_node_text(child, source_code)
                    generics.append(generic_name)

        return generics

    def _extract_decorators(self, node: Node, source_code: str) -> List[str]:
        """Извлечение decorators (@Component, @Injectable и т.д.)."""
        decorators = []

        # Decorators обычно находятся перед узлом
        prev_sibling = node.prev_sibling
        while prev_sibling and prev_sibling.type == 'decorator':
            decorator_text = self._get_node_text(prev_sibling, source_code)
            decorators.insert(0, decorator_text)
            prev_sibling = prev_sibling.prev_sibling

        return decorators

    def _extract_class_modifiers(self, node: Node, source_code: str) -> List[str]:
        """Извлечение модификаторов класса."""
        modifiers = []
        text = self._get_node_text(node, source_code)

        if 'abstract' in text.split()[0:3]:
            modifiers.append('abstract')
        if 'const' in text.split()[0:3]:
            modifiers.append('const')

        return modifiers

    def _extract_method_modifiers(self, node: Node, source_code: str) -> List[str]:
        """Извлечение модификаторов метода."""
        modifiers = []
        text = self._get_node_text(node, source_code).strip()

        # Проверяем первые слова
        words = text.split()
        for word in words[:5]:  # Проверяем первые 5 слов
            if word in ['public', 'private', 'protected', 'static', 'async', 'readonly', 'abstract']:
                modifiers.append(word)

        return modifiers

    def _extract_extends_clause(self, heritage_node: Node, source_code: str) -> Optional[str]:
        """Извлечение extends класса."""
        for child in heritage_node.children:
            if child.type == 'extends_clause':
                # Берем первый тип после extends
                for subchild in child.children:
                    if subchild.type in ['type_identifier', 'generic_type']:
                        return self._get_node_text(subchild, source_code)
        return None

    def _extract_implements_clause(self, heritage_node: Node, source_code: str) -> List[str]:
        """Извлечение implements интерфейсов."""
        interfaces = []

        for child in heritage_node.children:
            if child.type == 'implements_clause':
                for subchild in child.children:
                    if subchild.type in ['type_identifier', 'generic_type']:
                        interfaces.append(self._get_node_text(subchild, source_code))

        return interfaces

    def _extract_interface_extends(self, heritage_node: Node, source_code: str) -> List[str]:
        """Извлечение extends для интерфейсов."""
        extends = []

        for child in heritage_node.children:
            if child.type == 'extends_type_clause':
                for subchild in child.children:
                    if subchild.type in ['type_identifier', 'generic_type']:
                        extends.append(self._get_node_text(subchild, source_code))

        return extends

    def _extract_field_name(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение имени поля класса."""
        property_node = node.child_by_field_name('name')
        if property_node:
            return self._get_node_text(property_node, source_code)
        return None

    def _is_async_function(self, node: Node, source_code: str) -> bool:
        """Проверка async функции."""
        text = self._get_node_text(node, source_code)
        return 'async ' in text.split()[0:2]

    def _extract_jsdoc(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение JSDoc/TSDoc комментария."""
        return extract_jsdoc(node, source_code)

    def _build_function_signature(
            self,
            name: str,
            parameters: List[Dict],
            return_type: Optional[str],
            generics: List[str],
            is_async: bool,
            arrow: bool = False
    ) -> str:
        """Построение сигнатуры функции."""
        params = ", ".join([
            f"{p['name']}: {p.get('type', 'any')}"
            for p in parameters
        ])

        generic_str = f"<{', '.join(generics)}>" if generics else ""
        return_str = f": {return_type}" if return_type else ""

        if arrow:
            prefix = "async " if is_async else ""
            return f"{prefix}const {name}{generic_str} = ({params}){return_str} => {{ ... }}"
        else:
            prefix = "async function" if is_async else "function"
            return f"{prefix} {name}{generic_str}({params}){return_str}"

    def _build_method_signature(
            self,
            name: str,
            parameters: List[Dict],
            return_type: Optional[str],
            generics: List[str],
            modifiers: List[str]
    ) -> str:
        """Построение сигнатуры метода."""
        # Формируем строку параметров
        param_parts = []
        for p in parameters:
            param_type = p.get('type', 'any')
            param_parts.append(f"{p['name']}: {param_type}")
        params = ", ".join(param_parts)

        generic_str = f"<{', '.join(generics)}>" if generics else ""
        return_str = f": {return_type}" if return_type else ""

        modifier_str = " ".join(modifiers) + " " if modifiers else ""

        return f"{modifier_str}{name}{generic_str}({params}){return_str}"

    def _build_class_signature(
            self,
            name: str,
            base_classes: List[str],
            interfaces: List[str],
            generics: List[str],
            is_abstract: bool
    ) -> str:
        """Построение сигнатуры класса."""
        parts = []

        if is_abstract:
            parts.append("abstract")

        parts.append("class")

        class_name = name
        if generics:
            class_name += f"<{', '.join(generics)}>"
        parts.append(class_name)

        if base_classes:
            parts.append(f"extends {base_classes[0]}")

        if interfaces:
            parts.append(f"implements {', '.join(interfaces)}")

        return " ".join(parts)

    def _handle_export(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            module_info: ModuleInfo
    ) -> None:
        """Обработка export statements."""
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

            elif declaration.type == 'interface_declaration':
                interface = self._parse_interface(declaration, source_code, file_path)
                if interface:
                    interface.imports = module_info.imports
                    module_info.interfaces.append(interface)

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Получение текста узла."""
        return source_code[node.start_byte:node.end_byte]


# Удобная функция для быстрого парсинга
def parse_typescript_file(file_path: str, **kwargs) -> Optional[ModuleInfo]:
    """
    Быстрый парсинг TypeScript файла.

    Args:
        file_path: Путь к TypeScript файлу
        **kwargs: Дополнительные параметры

    Returns:
        ModuleInfo или None
    """
    parser = TypeScriptParser()
    return parser.parse_file(Path(file_path), **kwargs)
