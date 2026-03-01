"""
C# парсер на основе tree-sitter.
Извлекает классы, методы, интерфейсы, свойства и их метаданные.
Поддерживает современный C# синтаксис включая records, pattern matching и nullable reference types.
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
    logger.warning("tree-sitter не установлен. C# парсер будет недоступен.")

from ..base_parser import BaseParser
from ..code_structure import (
    CodeElement, CodeElementType, CodeLocation,
    ModuleInfo, ProgrammingLanguage
)
from ..utils.tree_sitter_helper import TreeSitterHelper
from ..utils.docstring_extractor import extract_xmldoc
from ..utils.complexity_calculator import calculate_complexity

logger = logging.getLogger(__name__)


class CSharpParser(BaseParser):
    """
    Парсер C# кода на основе tree-sitter.

    Извлекает:
    - Классы (включая records)
    - Методы и свойства
    - Интерфейсы
    - Enums и structs
    - XML Documentation комментарии
    - Атрибуты ([Serializable], [Obsolete] и т.д.)
    - Модификаторы доступа (public, private, protected, internal)
    - Async/await методы
    - Generics
    """

    def __init__(self, max_chunk_size: int = 8192):
        """
        Инициализация C# парсера.

        Args:
            max_chunk_size: Максимальный размер чанка
        """
        super().__init__(max_chunk_size)

        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter не установлен. Установите: pip install tree-sitter"
            )

        # Инициализация tree-sitter для C#
        self.ts_helper = TreeSitterHelper()
        self.parser = self.ts_helper.get_parser('c_sharp')

        if not self.parser:
            raise RuntimeError("Не удалось инициализировать C# парсер")

    @property
    def language(self) -> ProgrammingLanguage:
        """Язык программирования."""
        return ProgrammingLanguage.CSHARP

    @property
    def file_extensions(self) -> List[str]:
        """Поддерживаемые расширения файлов."""
        return ['.cs']

    def parse_file(
            self,
            file_path: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None
    ) -> Optional[ModuleInfo]:
        """
        Парсинг C# файла.

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

            # Извлекаем namespace
            namespace = self._extract_namespace(root_node, source_code)

            # Извлекаем using directives
            imports = self._extract_using_directives(root_node, source_code)

            # Создаем объект модуля
            module_info = ModuleInfo(
                file_path=file_path,
                module_name=module_name,
                language=self.language,
                docstring=namespace,  # Используем namespace как docstring
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
                f"Распарсен C# файл {file_path.name}: "
                f"{len(module_info.classes)} классов, "
                f"{len(module_info.interfaces)} интерфейсов"
            )

            return module_info

        except Exception as e:
            logger.error(f"Ошибка парсинга C# файла {file_path}: {e}")
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
            # Классы
            if child.type == 'class_declaration':
                class_element = self._parse_class(child, source_code, file_path)
                if class_element:
                    class_element.imports = module_info.imports
                    module_info.classes.append(class_element)

            # Records (C# 9+)
            elif child.type == 'record_declaration':
                record_element = self._parse_record(child, source_code, file_path)
                if record_element:
                    record_element.imports = module_info.imports
                    module_info.classes.append(record_element)

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

            # Structs
            elif child.type == 'struct_declaration':
                struct_element = self._parse_struct(child, source_code, file_path)
                if struct_element:
                    struct_element.imports = module_info.imports
                    module_info.classes.append(struct_element)

            # Namespace - обрабатываем вложенные элементы
            elif child.type == 'namespace_declaration':
                self._extract_elements(child, source_code, file_path, module_info, parent_class)

            # Рекурсивно обрабатываем другие узлы
            else:
                self._extract_elements(child, source_code, file_path, module_info, parent_class)

    def _parse_class(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг C# класса."""
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
            access_modifier = modifiers.get('access', 'internal')
            is_abstract = modifiers.get('abstract', False)
            is_static = modifiers.get('static', False)
            is_sealed = modifiers.get('sealed', False)
            is_partial = modifiers.get('partial', False)

            # XML Documentation
            xmldoc = self._extract_xmldoc(node, source_code)

            # Атрибуты
            attributes = self._extract_attributes(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Base class
            base_classes = []
            base_list_node = node.child_by_field_name('bases')
            if base_list_node:
                base_class = self._extract_base_class(base_list_node, source_code)
                if base_class:
                    base_classes.append(base_class)

            # Interfaces
            interfaces = []
            if base_list_node:
                interface_list = self._extract_interfaces(base_list_node, source_code)
                interfaces.extend(interface_list)

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
                docstring=xmldoc,
                signature=signature,
                decorators=attributes,
                base_classes=base_classes,
                interfaces=interfaces,
                generics=generics,
                methods=[],
                is_abstract=is_abstract,
                is_static=is_static,
                is_final=is_sealed,
                access_modifier=access_modifier,
                metadata={'partial': is_partial}
            )

            # Парсим методы, свойства и конструкторы
            body_node = node.child_by_field_name('body')
            if body_node:
                for child in body_node.children:
                    if child.type == 'method_declaration':
                        method = self._parse_method(child, source_code, file_path, class_name)
                        if method:
                            class_element.methods.append(method)

                    elif child.type == 'constructor_declaration':
                        constructor = self._parse_constructor(child, source_code, file_path, class_name)
                        if constructor:
                            class_element.methods.append(constructor)

                    elif child.type == 'property_declaration':
                        property_elem = self._parse_property(child, source_code, file_path, class_name)
                        if property_elem:
                            class_element.methods.append(property_elem)

                    elif child.type == 'field_declaration':
                        field_names = self._extract_field_names(child, source_code)
                        if not class_element.attributes:
                            class_element.attributes = []
                        class_element.attributes.extend(field_names)

            return class_element

        except Exception as e:
            logger.error(f"Ошибка парсинга C# класса: {e}")
            return None

    def _parse_record(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг C# record (C# 9+)."""
        # Record похож на класс, используем ту же логику
        element = self._parse_class(node, source_code, file_path)
        if element:
            # Помечаем как record в метаданных
            element.metadata['is_record'] = True
            signature_parts = element.signature.split()
            # Заменяем 'class' на 'record' в сигнатуре
            if 'class' in signature_parts:
                idx = signature_parts.index('class')
                signature_parts[idx] = 'record'
                element.signature = ' '.join(signature_parts)
        return element

    def _parse_interface(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг C# интерфейса."""
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

            # Модификаторы
            modifiers = self._extract_modifiers(node, source_code)
            access_modifier = modifiers.get('access', 'internal')

            # XML Doc
            xmldoc = self._extract_xmldoc(node, source_code)

            # Атрибуты
            attributes = self._extract_attributes(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Base interfaces
            base_interfaces = []
            base_list_node = node.child_by_field_name('bases')
            if base_list_node:
                base_interfaces = self._extract_interfaces(base_list_node, source_code)

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
                docstring=xmldoc,
                signature=signature,
                decorators=attributes,
                generics=generics,
                interfaces=base_interfaces,
                methods=[],
                is_abstract=True,
                access_modifier=access_modifier
            )

            # Парсим методы интерфейса
            body_node = node.child_by_field_name('body')
            if body_node:
                for child in body_node.children:
                    if child.type in ['method_declaration', 'property_declaration']:
                        method = self._parse_interface_member(child, source_code, file_path, interface_name)
                        if method:
                            interface_element.methods.append(method)

            return interface_element

        except Exception as e:
            logger.error(f"Ошибка парсинга C# интерфейса: {e}")
            return None

    def _parse_enum(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг C# enum."""
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

            # Модификаторы
            modifiers = self._extract_modifiers(node, source_code)
            access_modifier = modifiers.get('access', 'internal')

            # XML Doc
            xmldoc = self._extract_xmldoc(node, source_code)

            # Атрибуты
            attributes = self._extract_attributes(node, source_code)

            enum_element = CodeElement(
                name=enum_name,
                type=CodeElementType.ENUM,
                language=self.language,
                location=location,
                source_code=enum_source,
                docstring=xmldoc,
                signature=f"enum {enum_name}",
                decorators=attributes,
                access_modifier=access_modifier,
                is_final=True
            )

            return enum_element

        except Exception as e:
            logger.error(f"Ошибка парсинга C# enum: {e}")
            return None

    def _parse_struct(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг C# struct."""
        # Struct похож на класс
        element = self._parse_class(node, source_code, file_path)
        if element:
            element.type = CodeElementType.STRUCT
            element.metadata['is_struct'] = True
        return element

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

            # Модификаторы
            modifiers = self._extract_modifiers(node, source_code)
            access_modifier = modifiers.get('access', 'private')
            is_static = modifiers.get('static', False)
            is_async = modifiers.get('async', False)
            is_virtual = modifiers.get('virtual', False)
            is_override = modifiers.get('override', False)
            is_abstract = modifiers.get('abstract', False)
            is_sealed = modifiers.get('sealed', False)

            # XML Doc
            xmldoc = self._extract_xmldoc(node, source_code)

            # Атрибуты
            attributes = self._extract_attributes(node, source_code)

            # Параметры
            parameters = self._extract_method_parameters(node, source_code)

            # Return type
            return_type = self._extract_return_type(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Тип метода
            if is_static:
                element_type = CodeElementType.STATIC_METHOD
            elif is_async:
                element_type = CodeElementType.ASYNC_METHOD
            else:
                element_type = CodeElementType.METHOD

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
                docstring=xmldoc,
                signature=signature,
                decorators=attributes,
                parent=parent_class,
                parameters=parameters,
                return_type=return_type,
                generics=generics,
                is_async=is_async,
                is_static=is_static,
                is_abstract=is_abstract,
                is_final=is_sealed,
                access_modifier=access_modifier,
                complexity=complexity,
                metadata={
                    'virtual': is_virtual,
                    'override': is_override
                }
            )

            return method_element

        except Exception as e:
            logger.error(f"Ошибка парсинга C# метода: {e}")
            return None

    def _parse_constructor(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: str
    ) -> Optional[CodeElement]:
        """Парсинг конструктора."""
        try:
            # Имя конструктора
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            constructor_name = self._get_node_text(name_node, source_code)

            # Исходный код
            constructor_source = self._get_node_text(node, source_code)

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
            access_modifier = modifiers.get('access', 'private')
            is_static = modifiers.get('static', False)

            # XML Doc
            xmldoc = self._extract_xmldoc(node, source_code)

            # Атрибуты
            attributes = self._extract_attributes(node, source_code)

            # Параметры
            parameters = self._extract_method_parameters(node, source_code)

            # Сигнатура
            param_parts = []
            for p in parameters:
                param_type = p.get('type', 'object')
                param_parts.append(f"{param_type} {p['name']}")
            signature = f"{constructor_name}({', '.join(param_parts)})"

            constructor_element = CodeElement(
                name=constructor_name,
                type=CodeElementType.CONSTRUCTOR,
                language=self.language,
                location=location,
                source_code=constructor_source,
                docstring=xmldoc,
                signature=signature,
                decorators=attributes,
                parent=parent_class,
                parameters=parameters,
                is_static=is_static,
                access_modifier=access_modifier
            )

            return constructor_element

        except Exception as e:
            logger.error(f"Ошибка парсинга C# конструктора: {e}")
            return None

    def _parse_property(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: str
    ) -> Optional[CodeElement]:
        """Парсинг свойства (property)."""
        try:
            # Имя свойства
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            property_name = self._get_node_text(name_node, source_code)

            # Исходный код
            property_source = self._get_node_text(node, source_code)

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
            access_modifier = modifiers.get('access', 'private')
            is_static = modifiers.get('static', False)

            # XML Doc
            xmldoc = self._extract_xmldoc(node, source_code)

            # Атрибуты
            attributes = self._extract_attributes(node, source_code)

            # Type
            property_type = self._extract_property_type(node, source_code)

            # Сигнатура
            signature = f"{property_type} {property_name} {{ get; set; }}"

            property_element = CodeElement(
                name=property_name,
                type=CodeElementType.PROPERTY,
                language=self.language,
                location=location,
                source_code=property_source,
                docstring=xmldoc,
                signature=signature,
                decorators=attributes,
                parent=parent_class,
                return_type=property_type,
                is_static=is_static,
                access_modifier=access_modifier
            )

            return property_element

        except Exception as e:
            logger.debug(f"Не удалось распарсить property: {e}")
            return None

    def _parse_interface_member(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_interface: str
    ) -> Optional[CodeElement]:
        """Парсинг члена интерфейса."""
        # Используем парсинг метода, но помечаем как абстрактный
        if node.type == 'method_declaration':
            member = self._parse_method(node, source_code, file_path, parent_interface)
        elif node.type == 'property_declaration':
            member = self._parse_property(node, source_code, file_path, parent_interface)
        else:
            return None

        if member:
            member.is_abstract = True
            member.access_modifier = 'public'

        return member

    def _extract_namespace(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение namespace."""
        for child in node.children:
            if child.type == 'namespace_declaration':
                name_node = child.child_by_field_name('name')
                if name_node:
                    return self._get_node_text(name_node, source_code)
        return None

    def _extract_using_directives(self, node: Node, source_code: str) -> List[str]:
        """Извлечение using directives."""
        usings = []

        for child in node.children:
            if child.type == 'using_directive':
                using_text = self._get_node_text(child, source_code)
                using_text = using_text.replace('using', '').replace(';', '').strip()
                usings.append(using_text)

        return usings

    def _extract_modifiers(self, node: Node, source_code: str) -> Dict[str, Any]:
        """Извлечение модификаторов."""
        modifiers = {}

        for child in node.children:
            if child.type == 'modifier':
                modifier_text = self._get_node_text(child, source_code).strip()

                # Access modifiers
                if modifier_text == 'public':
                    modifiers['access'] = 'public'
                elif modifier_text == 'private':
                    modifiers['access'] = 'private'
                elif modifier_text == 'protected':
                    modifiers['access'] = 'protected'
                elif modifier_text == 'internal':
                    modifiers['access'] = 'internal'

                # Other modifiers
                elif modifier_text == 'static':
                    modifiers['static'] = True
                elif modifier_text == 'abstract':
                    modifiers['abstract'] = True
                elif modifier_text == 'sealed':
                    modifiers['sealed'] = True
                elif modifier_text == 'virtual':
                    modifiers['virtual'] = True
                elif modifier_text == 'override':
                    modifiers['override'] = True
                elif modifier_text == 'async':
                    modifiers['async'] = True
                elif modifier_text == 'partial':
                    modifiers['partial'] = True
                elif modifier_text == 'readonly':
                    modifiers['readonly'] = True

        return modifiers

    def _extract_attributes(self, node: Node, source_code: str) -> List[str]:
        """Извлечение атрибутов ([Serializable], [Obsolete] и т.д.)."""
        attributes = []

        # Атрибуты обычно находятся перед узлом
        prev_sibling = node.prev_sibling
        while prev_sibling and prev_sibling.type == 'attribute_list':
            attr_text = self._get_node_text(prev_sibling, source_code)
            attributes.insert(0, attr_text)
            prev_sibling = prev_sibling.prev_sibling

        return attributes

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

    def _extract_base_class(self, base_list_node: Node, source_code: str) -> Optional[str]:
        """Извлечение базового класса."""
        # В C# первый элемент в base list - это либо класс, либо интерфейс
        # Предполагаем, что если начинается с 'I' - это интерфейс
        for child in base_list_node.children:
            if child.type in ['simple_base_type', 'base_type']:
                base_name = self._get_node_text(child, source_code).strip()
                # Если не начинается с I и не известный интерфейс - это базовый класс
                if not base_name.startswith('I') or len(base_name) < 2 or not base_name[1].isupper():
                    return base_name

        return None

    def _extract_interfaces(self, base_list_node: Node, source_code: str) -> List[str]:
        """Извлечение интерфейсов."""
        interfaces = []

        for child in base_list_node.children:
            if child.type in ['simple_base_type', 'base_type']:
                interface_name = self._get_node_text(child, source_code).strip()
                # Простая эвристика: интерфейсы обычно начинаются с I
                if interface_name.startswith('I') and len(interface_name) > 1 and interface_name[1].isupper():
                    interfaces.append(interface_name)

        return interfaces

    def _extract_method_parameters(self, node: Node, source_code: str) -> List[Dict[str, Any]]:
        """Извлечение параметров метода."""
        parameters = []

        params_node = node.child_by_field_name('parameters')
        if params_node:
            for child in params_node.children:
                if child.type == 'parameter':
                    param_type_node = child.child_by_field_name('type')
                    param_name_node = child.child_by_field_name('name')

                    if param_name_node:
                        param_name = self._get_node_text(param_name_node, source_code)
                        param_type = 'object'

                        if param_type_node:
                            param_type = self._get_node_text(param_type_node, source_code)

                        parameters.append({
                            'name': param_name,
                            'type': param_type,
                            'kind': 'positional'
                        })

        return parameters

    def _extract_return_type(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение типа возвращаемого значения."""
        type_node = node.child_by_field_name('type')
        if type_node:
            return self._get_node_text(type_node, source_code)
        return None

    def _extract_property_type(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение типа свойства."""
        return self._extract_return_type(node, source_code)

    def _extract_field_names(self, node: Node, source_code: str) -> List[str]:
        """Извлечение имен полей."""
        field_names = []

        for child in node.children:
            if child.type == 'variable_declaration':
                for subchild in child.children:
                    if subchild.type == 'variable_declarator':
                        name_node = subchild.child_by_field_name('name')
                        if name_node:
                            field_names.append(self._get_node_text(name_node, source_code))

        return field_names

    def _extract_xmldoc(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение XML Documentation комментария."""
        return extract_xmldoc(node, source_code)

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
        if modifiers.get('access'):
            parts.append(modifiers['access'])
        if modifiers.get('static'):
            parts.append('static')
        if modifiers.get('abstract'):
            parts.append('abstract')
        if modifiers.get('sealed'):
            parts.append('sealed')
        if modifiers.get('partial'):
            parts.append('partial')

        # Class name
        parts.append('class')
        class_name = name
        if generics:
            class_name += f"<{', '.join(generics)}>"
        parts.append(class_name)

        # Base class and interfaces
        inheritance = []
        if base_classes:
            inheritance.extend(base_classes)
        if interfaces:
            inheritance.extend(interfaces)

        if inheritance:
            parts.append(f": {', '.join(inheritance)}")

        return ' '.join(parts)

    def _build_method_signature(
            self,
            name: str,
            parameters: List[Dict],
            return_type: Optional[str],
            generics: List[str],
            modifiers: Dict
    ) -> str:
        """Построение сигнатуры метода."""
        parts = []

        # Модификаторы
        if modifiers.get('access'):
            parts.append(modifiers['access'])
        if modifiers.get('static'):
            parts.append('static')
        if modifiers.get('virtual'):
            parts.append('virtual')
        if modifiers.get('override'):
            parts.append('override')
        if modifiers.get('async'):
            parts.append('async')
        if modifiers.get('abstract'):
            parts.append('abstract')
        if modifiers.get('sealed'):
            parts.append('sealed')

        # Return type
        if return_type:
            parts.append(return_type)
        else:
            parts.append('void')

        # Method name and generics
        method_name = name
        if generics:
            method_name += f"<{', '.join(generics)}>"

        # Parameters
        param_parts = []
        for p in parameters:
            param_type = p.get('type', 'object')
            param_parts.append(f"{param_type} {p['name']}")

        parts.append(f"{method_name}({', '.join(param_parts)})")

        return ' '.join(parts)

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Получение текста узла."""
        return source_code[node.start_byte:node.end_byte]


# Удобная функция для быстрого парсинга
def parse_csharp_file(file_path: str, **kwargs) -> Optional[ModuleInfo]:
    """
    Быстрый парсинг C# файла.

    Args:
        file_path: Путь к C# файлу
        **kwargs: Дополнительные параметры

    Returns:
        ModuleInfo или None
    """
    parser = CSharpParser()
    return parser.parse_file(Path(file_path), **kwargs)
