"""
Kotlin парсер на основе tree-sitter.
Извлекает классы, функции, объекты, интерфейсы и их метаданные.
Поддерживает современный Kotlin синтаксис включая data classes, sealed classes, и coroutines.
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
    logger.warning("tree-sitter не установлен. Kotlin парсер будет недоступен.")

from ..base_parser import BaseParser
from ..code_structure import (
    CodeElement, CodeElementType, CodeLocation,
    ModuleInfo, ProgrammingLanguage
)
from ..utils.tree_sitter_helper import TreeSitterHelper
from ..utils.docstring_extractor import extract_javadoc
from ..utils.complexity_calculator import calculate_complexity

logger = logging.getLogger(__name__)


class KotlinParser(BaseParser):
    """
    Парсер Kotlin кода на основе tree-sitter.

    Извлекает:
    - Классы (включая data classes, sealed classes, inner classes)
    - Функции (обычные и suspend)
    - Object declarations (singleton)
    - Companion objects
    - Интерфейсы
    - Extension functions
    - KDoc комментарии
    - Аннотации (@JvmStatic, @Deprecated и т.д.)
    - Coroutines (suspend functions)
    """

    def __init__(self, max_chunk_size: int = 8192):
        """
        Инициализация Kotlin парсера.

        Args:
            max_chunk_size: Максимальный размер чанка
        """
        super().__init__(max_chunk_size)

        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter не установлен. Установите: pip install tree-sitter"
            )

        # Инициализация tree-sitter для Kotlin
        self.ts_helper = TreeSitterHelper()
        self.parser = self.ts_helper.get_parser('kotlin')

        if not self.parser:
            raise RuntimeError("Не удалось инициализировать Kotlin парсер")

    @property
    def language(self) -> ProgrammingLanguage:
        """Язык программирования."""
        return ProgrammingLanguage.KOTLIN

    @property
    def file_extensions(self) -> List[str]:
        """Поддерживаемые расширения файлов."""
        return ['.kt', '.kts']

    def parse_file(
            self,
            file_path: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None
    ) -> Optional[ModuleInfo]:
        """
        Парсинг Kotlin файла.

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

            # Извлекаем package
            package_name = self._extract_package(root_node, source_code)

            # Извлекаем imports
            imports = self._extract_imports(root_node, source_code)

            # Создаем объект модуля
            module_info = ModuleInfo(
                file_path=file_path,
                module_name=module_name,
                language=self.language,
                docstring=package_name,  # Используем package как docstring
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
                f"Распарсен Kotlin файл {file_path.name}: "
                f"{len(module_info.functions)} функций, "
                f"{len(module_info.classes)} классов"
            )

            return module_info

        except Exception as e:
            logger.error(f"Ошибка парсинга Kotlin файла {file_path}: {e}")
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
            # Функции на уровне файла
            if child.type == 'function_declaration':
                func_element = self._parse_function(child, source_code, file_path, parent_class)
                if func_element:
                    func_element.imports = module_info.imports
                    if not parent_class:
                        module_info.functions.append(func_element)

            # Классы
            elif child.type == 'class_declaration':
                class_element = self._parse_class(child, source_code, file_path)
                if class_element:
                    class_element.imports = module_info.imports
                    module_info.classes.append(class_element)

            # Object declarations (singleton)
            elif child.type == 'object_declaration':
                object_element = self._parse_object(child, source_code, file_path)
                if object_element:
                    object_element.imports = module_info.imports
                    module_info.classes.append(object_element)

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

            # Property declarations на уровне файла
            elif child.type == 'property_declaration':
                # Можно добавить в будущем
                pass

            # Рекурсивно обрабатываем другие узлы
            else:
                self._extract_elements(child, source_code, file_path, module_info, parent_class)

    def _parse_function(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: Optional[str] = None
    ) -> Optional[CodeElement]:
        """Парсинг Kotlin функции."""
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

            # Модификаторы
            modifiers = self._extract_modifiers(node, source_code)
            is_suspend = modifiers.get('suspend', False)
            is_inline = modifiers.get('inline', False)
            is_infix = modifiers.get('infix', False)
            is_operator = modifiers.get('operator', False)
            access_modifier = modifiers.get('access', 'public')

            # KDoc
            kdoc = self._extract_kdoc(node, source_code)

            # Аннотации
            annotations = self._extract_annotations(node, source_code)

            # Параметры
            parameters = self._extract_function_parameters(node, source_code)

            # Return type
            return_type = self._extract_return_type(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Определяем тип функции
            if parent_class:
                element_type = CodeElementType.ASYNC_METHOD if is_suspend else CodeElementType.METHOD
            else:
                element_type = CodeElementType.ASYNC_FUNCTION if is_suspend else CodeElementType.FUNCTION

            # Сигнатура
            signature = self._build_function_signature(
                func_name, parameters, return_type, generics, modifiers
            )

            # Сложность
            complexity = calculate_complexity(func_source, self.language)

            func_element = CodeElement(
                name=func_name,
                type=element_type,
                language=self.language,
                location=location,
                source_code=func_source,
                docstring=kdoc,
                signature=signature,
                decorators=annotations,
                parent=parent_class,
                parameters=parameters,
                return_type=return_type,
                generics=generics,
                is_async=is_suspend,
                access_modifier=access_modifier,
                complexity=complexity,
                metadata={
                    'inline': is_inline,
                    'infix': is_infix,
                    'operator': is_operator
                }
            )

            return func_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Kotlin функции: {e}")
            return None

    def _parse_class(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг Kotlin класса."""
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

            # Модификаторы
            modifiers = self._extract_modifiers(node, source_code)
            is_data = modifiers.get('data', False)
            is_sealed = modifiers.get('sealed', False)
            is_abstract = modifiers.get('abstract', False)
            is_open = modifiers.get('open', False)
            is_inner = modifiers.get('inner', False)
            access_modifier = modifiers.get('access', 'public')

            # KDoc
            kdoc = self._extract_kdoc(node, source_code)

            # Аннотации
            annotations = self._extract_annotations(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Supertype (наследование и интерфейсы)
            base_classes = []
            interfaces = []
            delegation_specifiers = self._extract_delegation_specifiers(node, source_code)

            # Разделяем на классы и интерфейсы
            for spec in delegation_specifiers:
                # Простая эвристика: если начинается с заглавной буквы и не содержит 'I' в начале
                if spec and not spec.startswith('I'):
                    base_classes.append(spec)
                else:
                    interfaces.append(spec)

            # Сигнатура
            signature = self._build_class_signature(
                class_name, modifiers, base_classes, interfaces, generics
            )

            # Создаем элемент класса
            class_element = CodeElement(
                name=class_name,
                type=CodeElementType.CLASS,
                language=self.language,
                location=location,
                source_code=class_source,
                docstring=kdoc,
                signature=signature,
                decorators=annotations,
                base_classes=base_classes,
                interfaces=interfaces,
                generics=generics,
                methods=[],
                is_abstract=is_abstract,
                is_final=is_sealed,
                access_modifier=access_modifier,
                metadata={
                    'data': is_data,
                    'sealed': is_sealed,
                    'open': is_open,
                    'inner': is_inner
                }
            )

            # Парсим тело класса
            body_node = node.child_by_field_name('body')
            if body_node:
                for child in body_node.children:
                    if child.type == 'function_declaration':
                        method = self._parse_function(child, source_code, file_path, class_name)
                        if method:
                            class_element.methods.append(method)

                    elif child.type == 'property_declaration':
                        property_name = self._extract_property_name(child, source_code)
                        if property_name:
                            if not class_element.attributes:
                                class_element.attributes = []
                            class_element.attributes.append(property_name)

                    # Companion object
                    elif child.type == 'companion_object':
                        companion = self._parse_companion_object(child, source_code, file_path, class_name)
                        if companion:
                            class_element.methods.append(companion)

            return class_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Kotlin класса: {e}")
            return None

    def _parse_object(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг object declaration (singleton)."""
        try:
            # Имя object
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            object_name = self._get_node_text(name_node, source_code)

            # Исходный код
            object_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # KDoc
            kdoc = self._extract_kdoc(node, source_code)

            # Аннотации
            annotations = self._extract_annotations(node, source_code)

            # Модификаторы
            modifiers = self._extract_modifiers(node, source_code)
            access_modifier = modifiers.get('access', 'public')

            # Interfaces
            interfaces = self._extract_delegation_specifiers(node, source_code)

            object_element = CodeElement(
                name=object_name,
                type=CodeElementType.CLASS,
                language=self.language,
                location=location,
                source_code=object_source,
                docstring=kdoc,
                signature=f"object {object_name}",
                decorators=annotations,
                interfaces=interfaces,
                methods=[],
                is_static=True,  # Object это singleton
                is_final=True,  # Objects всегда final
                access_modifier=access_modifier,
                metadata={'is_object': True}
            )

            # Парсим тело object
            body_node = node.child_by_field_name('body')
            if body_node:
                for child in body_node.children:
                    if child.type == 'function_declaration':
                        method = self._parse_function(child, source_code, file_path, object_name)
                        if method:
                            object_element.methods.append(method)

            return object_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Kotlin object: {e}")
            return None

    def _parse_companion_object(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: str
    ) -> Optional[CodeElement]:
        """Парсинг companion object."""
        try:
            companion_source = self._get_node_text(node, source_code)

            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            companion_element = CodeElement(
                name="Companion",
                type=CodeElementType.STATIC_METHOD,
                language=self.language,
                location=location,
                source_code=companion_source,
                parent=parent_class,
                is_static=True,
                access_modifier='public',
                metadata={'is_companion': True}
            )

            return companion_element

        except Exception as e:
            logger.debug(f"Не удалось распарсить companion object: {e}")
            return None

    def _parse_interface(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг Kotlin интерфейса."""
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

            # KDoc
            kdoc = self._extract_kdoc(node, source_code)

            # Аннотации
            annotations = self._extract_annotations(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Supertype interfaces
            base_interfaces = self._extract_delegation_specifiers(node, source_code)

            # Сигнатура
            signature = f"interface {interface_name}"
            if generics:
                signature += f"<{', '.join(generics)}>"
            if base_interfaces:
                signature += f" : {', '.join(base_interfaces)}"

            interface_element = CodeElement(
                name=interface_name,
                type=CodeElementType.INTERFACE,
                language=self.language,
                location=location,
                source_code=interface_source,
                docstring=kdoc,
                signature=signature,
                decorators=annotations,
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
                    if child.type == 'function_declaration':
                        method = self._parse_function(child, source_code, file_path, interface_name)
                        if method:
                            interface_element.methods.append(method)

            return interface_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Kotlin интерфейса: {e}")
            return None

    def _parse_enum(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг Kotlin enum."""
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

            # KDoc
            kdoc = self._extract_kdoc(node, source_code)

            # Аннотации
            annotations = self._extract_annotations(node, source_code)

            enum_element = CodeElement(
                name=enum_name,
                type=CodeElementType.ENUM,
                language=self.language,
                location=location,
                source_code=enum_source,
                docstring=kdoc,
                signature=f"enum class {enum_name}",
                decorators=annotations,
                is_final=True,
                access_modifier='public'
            )

            return enum_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Kotlin enum: {e}")
            return None

    def _extract_package(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение package declaration."""
        for child in node.children:
            if child.type == 'package_header':
                # Извлекаем имя пакета
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        return self._get_node_text(subchild, source_code)
        return None

    def _extract_imports(self, node: Node, source_code: str) -> List[str]:
        """Извлечение import statements."""
        imports = []

        for child in node.children:
            if child.type == 'import_header':
                import_text = self._get_node_text(child, source_code)
                import_text = import_text.replace('import', '').strip()
                imports.append(import_text)

        return imports

    def _extract_modifiers(self, node: Node, source_code: str) -> Dict[str, Any]:
        """Извлечение модификаторов."""
        modifiers = {}

        for child in node.children:
            if child.type == 'modifiers':
                modifier_text = self._get_node_text(child, source_code).lower()

                # Access modifiers
                if 'public' in modifier_text:
                    modifiers['access'] = 'public'
                elif 'private' in modifier_text:
                    modifiers['access'] = 'private'
                elif 'protected' in modifier_text:
                    modifiers['access'] = 'protected'
                elif 'internal' in modifier_text:
                    modifiers['access'] = 'internal'

                # Function modifiers
                modifiers['suspend'] = 'suspend' in modifier_text
                modifiers['inline'] = 'inline' in modifier_text
                modifiers['infix'] = 'infix' in modifier_text
                modifiers['operator'] = 'operator' in modifier_text

                # Class modifiers
                modifiers['data'] = 'data' in modifier_text
                modifiers['sealed'] = 'sealed' in modifier_text
                modifiers['abstract'] = 'abstract' in modifier_text
                modifiers['open'] = 'open' in modifier_text
                modifiers['inner'] = 'inner' in modifier_text

                break

        return modifiers

    def _extract_annotations(self, node: Node, source_code: str) -> List[str]:
        """Извлечение аннотаций."""
        annotations = []

        for child in node.children:
            if child.type == 'annotation':
                annotation_text = self._get_node_text(child, source_code)
                annotations.append(annotation_text)

        return annotations

    def _extract_generics(self, node: Node, source_code: str) -> List[str]:
        """Извлечение generic параметров."""
        generics = []

        for child in node.children:
            if child.type == 'type_parameters':
                for subchild in child.children:
                    if subchild.type == 'type_parameter':
                        generic_name = self._get_node_text(subchild, source_code)
                        generics.append(generic_name)

        return generics

    def _extract_delegation_specifiers(self, node: Node, source_code: str) -> List[str]:
        """Извлечение delegation specifiers (наследование и интерфейсы)."""
        specifiers = []

        for child in node.children:
            if child.type == 'delegation_specifiers':
                for subchild in child.children:
                    if subchild.type in ['type_reference', 'user_type']:
                        spec_name = self._get_node_text(subchild, source_code)
                        # Убираем параметры конструктора если есть
                        spec_name = spec_name.split('(')[0].strip()
                        specifiers.append(spec_name)

        return specifiers

    def _extract_function_parameters(self, node: Node, source_code: str) -> List[Dict[str, Any]]:
        """Извлечение параметров функции."""
        parameters = []

        for child in node.children:
            if child.type == 'function_value_parameters':
                for subchild in child.children:
                    if subchild.type == 'parameter':
                        param_name_node = subchild.child_by_field_name('name')
                        param_type_node = subchild.child_by_field_name('type')

                        if param_name_node:
                            param_name = self._get_node_text(param_name_node, source_code)
                            param_type = None

                            if param_type_node:
                                param_type = self._get_node_text(param_type_node, source_code)
                                param_type = param_type.lstrip(':').strip()

                            parameters.append({
                                'name': param_name,
                                'type': param_type,
                                'kind': 'positional'
                            })

        return parameters

    def _extract_return_type(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение типа возвращаемого значения."""
        for child in node.children:
            if child.type == 'type':
                return self._get_node_text(child, source_code).lstrip(':').strip()
        return None

    def _extract_property_name(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение имени property."""
        for child in node.children:
            if child.type == 'variable_declaration':
                name_node = child.child_by_field_name('name')
                if name_node:
                    return self._get_node_text(name_node, source_code)
        return None

    def _extract_kdoc(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение KDoc комментария (аналогично JavaDoc)."""
        return extract_javadoc(node, source_code)

    def _build_function_signature(
            self,
            name: str,
            parameters: List[Dict],
            return_type: Optional[str],
            generics: List[str],
            modifiers: Dict
    ) -> str:
        """Построение сигнатуры функции."""
        parts = []

        # Модификаторы
        if modifiers.get('suspend'):
            parts.append('suspend')
        if modifiers.get('inline'):
            parts.append('inline')
        if modifiers.get('infix'):
            parts.append('infix')
        if modifiers.get('operator'):
            parts.append('operator')

        parts.append('fun')

        # Generics
        if generics:
            parts.append(f"<{', '.join(generics)}>")

        # Имя и параметры
        param_parts = []
        for p in parameters:
            param_type = p.get('type', 'Any')
            param_parts.append(f"{p['name']}: {param_type}")

        parts.append(f"{name}({', '.join(param_parts)})")

        # Return type
        if return_type:
            parts.append(f": {return_type}")

        return ' '.join(parts)

    def _build_class_signature(
            self,
            name: str,
            modifiers: Dict,
            base_classes: List[str],
            interfaces: List[str],
            generics: List[str]
    ) -> str:
        """Построение сигнатуры класса."""
        parts = []

        # Модификаторы
        if modifiers.get('data'):
            parts.append('data')
        if modifiers.get('sealed'):
            parts.append('sealed')
        if modifiers.get('abstract'):
            parts.append('abstract')
        if modifiers.get('open'):
            parts.append('open')
        if modifiers.get('inner'):
            parts.append('inner')

        parts.append('class')

        # Имя класса
        class_name = name
        if generics:
            class_name += f"<{', '.join(generics)}>"
        parts.append(class_name)

        # Supertype
        supertypes = []
        if base_classes:
            supertypes.extend(base_classes)
        if interfaces:
            supertypes.extend(interfaces)

        if supertypes:
            parts.append(f": {', '.join(supertypes)}")

        return ' '.join(parts)

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Получение текста узла."""
        return source_code[node.start_byte:node.end_byte]


# Удобная функция для быстрого парсинга
def parse_kotlin_file(file_path: str, **kwargs) -> Optional[ModuleInfo]:
    """
    Быстрый парсинг Kotlin файла.

    Args:
        file_path: Путь к Kotlin файлу
        **kwargs: Дополнительные параметры

    Returns:
        ModuleInfo или None
    """
    parser = KotlinParser()
    return parser.parse_file(Path(file_path), **kwargs)
