"""
Java парсер на основе tree-sitter.
Извлекает классы, методы, интерфейсы и их метаданные.
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
    logger.warning("tree-sitter не установлен. Java парсер будет недоступен.")

from ..base_parser import BaseParser
from ..code_structure import (
    CodeElement, CodeElementType, CodeLocation,
    ModuleInfo, ProgrammingLanguage
)
from ..utils.tree_sitter_helper import TreeSitterHelper
from ..utils.docstring_extractor import extract_javadoc
from ..utils.complexity_calculator import calculate_complexity

logger = logging.getLogger(__name__)


class JavaParser(BaseParser):
    """
    Парсер Java кода на основе tree-sitter.

    Извлекает:
    - Классы и интерфейсы
    - Методы и конструкторы
    - Поля (атрибуты)
    - JavaDoc комментарии
    - Модификаторы доступа (public, private, protected)
    - Аннотации (@Override, @Deprecated и т.д.)
    - Generics
    - Implements/Extends
    """

    def __init__(self, max_chunk_size: int = 8192):
        """
        Инициализация Java парсера.

        Args:
            max_chunk_size: Максимальный размер чанка
        """
        super().__init__(max_chunk_size)

        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter не установлен. Установите: pip install tree-sitter"
            )

        # Инициализация tree-sitter для Java
        self.ts_helper = TreeSitterHelper()
        self.parser = self.ts_helper.get_parser('java')

        if not self.parser:
            raise RuntimeError("Не удалось инициализировать Java парсер")

    @property
    def language(self) -> ProgrammingLanguage:
        """Язык программирования."""
        return ProgrammingLanguage.JAVA

    @property
    def file_extensions(self) -> List[str]:
        """Поддерживаемые расширения файлов."""
        return ['.java']

    def parse_file(
            self,
            file_path: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None
    ) -> Optional[ModuleInfo]:
        """
        Парсинг Java файла.

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

            # Извлекаем имя модуля (имя файла)
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

            # Извлекаем классы и интерфейсы
            for node in root_node.children:
                if node.type == 'class_declaration':
                    class_element = self._parse_class(node, source_code, file_path)
                    if class_element:
                        class_element.imports = imports
                        module_info.classes.append(class_element)

                elif node.type == 'interface_declaration':
                    interface_element = self._parse_interface(node, source_code, file_path)
                    if interface_element:
                        interface_element.imports = imports
                        module_info.interfaces.append(interface_element)

                elif node.type == 'enum_declaration':
                    enum_element = self._parse_enum(node, source_code, file_path)
                    if enum_element:
                        enum_element.imports = imports
                        module_info.enums.append(enum_element)

            logger.info(
                f"Распарсен Java файл {file_path.name}: "
                f"{len(module_info.classes)} классов, "
                f"{len(module_info.interfaces)} интерфейсов"
            )

            return module_info

        except Exception as e:
            logger.error(f"Ошибка парсинга Java файла {file_path}: {e}")
            return None

    def _parse_class(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг Java класса."""
        try:
            # Извлекаем имя класса
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            class_name = self._get_node_text(name_node, source_code)

            # Извлекаем исходный код класса
            class_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # Извлекаем модификаторы (public, private, abstract, final и т.д.)
            modifiers = self._extract_modifiers(node, source_code)
            access_modifier = modifiers.get('access', 'package-private')
            is_abstract = modifiers.get('abstract', False)
            is_final = modifiers.get('final', False)
            is_static = modifiers.get('static', False)

            # Извлекаем JavaDoc
            javadoc = self._extract_javadoc(node, source_code)

            # Извлекаем аннотации
            annotations = self._extract_annotations(node, source_code)

            # Извлекаем generics
            generics = self._extract_generics(node, source_code)

            # Извлекаем базовый класс (extends)
            base_classes = []
            superclass_node = node.child_by_field_name('superclass')
            if superclass_node:
                base_class = self._get_node_text(superclass_node, source_code)
                base_classes.append(base_class.replace('extends', '').strip())

            # Извлекаем интерфейсы (implements)
            interfaces = []
            interfaces_node = node.child_by_field_name('interfaces')
            if interfaces_node:
                interface_list = self._get_node_text(interfaces_node, source_code)
                interface_list = interface_list.replace('implements', '').strip()
                interfaces = [i.strip() for i in interface_list.split(',')]

            # Извлекаем атрибуты (поля)
            attributes = self._extract_class_fields(node, source_code)

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
                docstring=javadoc,
                signature=signature,
                decorators=annotations,
                base_classes=base_classes,
                interfaces=interfaces,
                attributes=attributes,
                generics=generics,
                methods=[],
                access_modifier=access_modifier,
                is_abstract=is_abstract,
                is_final=is_final,
                is_static=is_static
            )

            # Парсим методы и конструкторы
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

            return class_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Java класса: {e}")
            return None

    def _parse_interface(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг Java интерфейса."""
        try:
            # Извлекаем имя интерфейса
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            interface_name = self._get_node_text(name_node, source_code)

            # Извлекаем исходный код
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
            access_modifier = modifiers.get('access', 'public')

            # JavaDoc
            javadoc = self._extract_javadoc(node, source_code)

            # Аннотации
            annotations = self._extract_annotations(node, source_code)

            # Generics
            generics = self._extract_generics(node, source_code)

            # Extends (интерфейсы могут наследовать другие интерфейсы)
            base_interfaces = []
            extends_node = node.child_by_field_name('interfaces')
            if extends_node:
                extends_list = self._get_node_text(extends_node, source_code)
                extends_list = extends_list.replace('extends', '').strip()
                base_interfaces = [i.strip() for i in extends_list.split(',')]

            # Сигнатура
            signature = f"interface {interface_name}"
            if generics:
                signature += f"<{', '.join(generics)}>"
            if base_interfaces:
                signature += f" extends {', '.join(base_interfaces)}"

            # Создаем элемент интерфейса
            interface_element = CodeElement(
                name=interface_name,
                type=CodeElementType.INTERFACE,
                language=self.language,
                location=location,
                source_code=interface_source,
                docstring=javadoc,
                signature=signature,
                decorators=annotations,
                interfaces=base_interfaces,
                generics=generics,
                methods=[],
                access_modifier=access_modifier,
                is_abstract=True  # Интерфейсы всегда абстрактные
            )

            # Парсим методы интерфейса
            body_node = node.child_by_field_name('body')
            if body_node:
                for child in body_node.children:
                    if child.type == 'method_declaration':
                        method = self._parse_method(child, source_code, file_path, interface_name)
                        if method:
                            interface_element.methods.append(method)

            return interface_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Java интерфейса: {e}")
            return None

    def _parse_enum(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг Java enum."""
        try:
            # Извлекаем имя enum
            name_node = node.child_by_field_name('name')
            if not name_node:
                return None
            enum_name = self._get_node_text(name_node, source_code)

            # Извлекаем исходный код
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
            access_modifier = modifiers.get('access', 'package-private')

            # JavaDoc
            javadoc = self._extract_javadoc(node, source_code)

            # Аннотации
            annotations = self._extract_annotations(node, source_code)

            # Создаем элемент enum
            enum_element = CodeElement(
                name=enum_name,
                type=CodeElementType.ENUM,
                language=self.language,
                location=location,
                source_code=enum_source,
                docstring=javadoc,
                signature=f"enum {enum_name}",
                decorators=annotations,
                access_modifier=access_modifier,
                is_final=True  # Enums всегда final
            )

            return enum_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Java enum: {e}")
            return None

    def _parse_method(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: str
    ) -> Optional[CodeElement]:
        """Парсинг Java метода."""
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
            access_modifier = modifiers.get('access', 'package-private')
            is_static = modifiers.get('static', False)
            is_abstract = modifiers.get('abstract', False)
            is_final = modifiers.get('final', False)

            # JavaDoc
            javadoc = self._extract_javadoc(node, source_code)

            # Аннотации
            annotations = self._extract_annotations(node, source_code)

            # Параметры
            parameters = self._extract_method_parameters(node, source_code)

            # Return type
            return_type = None
            type_node = node.child_by_field_name('type')
            if type_node:
                return_type = self._get_node_text(type_node, source_code)

            # Сигнатура
            signature = self._build_method_signature(
                method_name, parameters, return_type, modifiers
            )

            # Вычисляем сложность
            complexity = calculate_complexity(method_source, self.language)

            # Определяем тип метода
            element_type = CodeElementType.STATIC_METHOD if is_static else CodeElementType.METHOD

            method_element = CodeElement(
                name=method_name,
                type=element_type,
                language=self.language,
                location=location,
                source_code=method_source,
                docstring=javadoc,
                signature=signature,
                decorators=annotations,
                parent=parent_class,
                parameters=parameters,
                return_type=return_type,
                access_modifier=access_modifier,
                is_static=is_static,
                is_abstract=is_abstract,
                is_final=is_final,
                complexity=complexity
            )

            return method_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Java метода: {e}")
            return None

    def _parse_constructor(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            parent_class: str
    ) -> Optional[CodeElement]:
        """Парсинг Java конструктора."""
        try:
            # Имя конструктора (совпадает с именем класса)
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
            access_modifier = modifiers.get('access', 'public')

            # JavaDoc
            javadoc = self._extract_javadoc(node, source_code)

            # Аннотации
            annotations = self._extract_annotations(node, source_code)

            # Параметры
            parameters = self._extract_method_parameters(node, source_code)

            # Сигнатура
            param_str = ", ".join([f"{p['type']} {p['name']}" for p in parameters])
            signature = f"{constructor_name}({param_str})"

            constructor_element = CodeElement(
                name=constructor_name,
                type=CodeElementType.CONSTRUCTOR,
                language=self.language,
                location=location,
                source_code=constructor_source,
                docstring=javadoc,
                signature=signature,
                decorators=annotations,
                parent=parent_class,
                parameters=parameters,
                access_modifier=access_modifier
            )

            return constructor_element

        except Exception as e:
            logger.error(f"Ошибка парсинга Java конструктора: {e}")
            return None

    def _extract_package(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение package declaration."""
        for child in node.children:
            if child.type == 'package_declaration':
                return self._get_node_text(child, source_code).replace('package', '').replace(';', '').strip()
        return None

    def _extract_imports(self, node: Node, source_code: str) -> List[str]:
        """Извлечение imports."""
        imports = []
        for child in node.children:
            if child.type == 'import_declaration':
                import_text = self._get_node_text(child, source_code)
                import_text = import_text.replace('import', '').replace('static', '').replace(';', '').strip()
                imports.append(import_text)
        return imports

    def _extract_modifiers(self, node: Node, source_code: str) -> Dict[str, Any]:
        """Извлечение модификаторов (public, private, static, final и т.д.)."""
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

                # Other modifiers
                modifiers['static'] = 'static' in modifier_text
                modifiers['final'] = 'final' in modifier_text
                modifiers['abstract'] = 'abstract' in modifier_text
                modifiers['synchronized'] = 'synchronized' in modifier_text

                break

        return modifiers

    def _extract_annotations(self, node: Node, source_code: str) -> List[str]:
        """Извлечение аннотаций (@Override, @Deprecated и т.д.)."""
        annotations = []

        for child in node.children:
            if child.type == 'modifiers':
                for modifier_child in child.children:
                    if modifier_child.type == 'marker_annotation' or modifier_child.type == 'annotation':
                        annotation_text = self._get_node_text(modifier_child, source_code)
                        annotations.append(annotation_text)

        return annotations

    def _extract_generics(self, node: Node, source_code: str) -> List[str]:
        """Извлечение generic типов (<T>, <K, V> и т.д.)."""
        generics = []

        type_params_node = node.child_by_field_name('type_parameters')
        if type_params_node:
            generic_text = self._get_node_text(type_params_node, source_code)
            generic_text = generic_text.strip('<>').strip()
            generics = [g.strip() for g in generic_text.split(',')]

        return generics

    def _extract_class_fields(self, node: Node, source_code: str) -> List[str]:
        """Извлечение полей класса."""
        fields = []

        body_node = node.child_by_field_name('body')
        if body_node:
            for child in body_node.children:
                if child.type == 'field_declaration':
                    field_text = self._get_node_text(child, source_code)
                    # Простое извлечение имени поля
                    for declarator in child.children:
                        if declarator.type == 'variable_declarator':
                            name_node = declarator.child_by_field_name('name')
                            if name_node:
                                fields.append(self._get_node_text(name_node, source_code))

        return fields

    def _extract_method_parameters(self, node: Node, source_code: str) -> List[Dict[str, Any]]:
        """Извлечение параметров метода."""
        parameters = []

        params_node = node.child_by_field_name('parameters')
        if params_node:
            for child in params_node.children:
                if child.type == 'formal_parameter':
                    param_type_node = child.child_by_field_name('type')
                    param_name_node = child.child_by_field_name('name')

                    if param_type_node and param_name_node:
                        parameters.append({
                            'name': self._get_node_text(param_name_node, source_code),
                            'type': self._get_node_text(param_type_node, source_code),
                            'kind': 'positional'
                        })

        return parameters

    def _extract_javadoc(self, node: Node, source_code: str) -> Optional[str]:
        """Извлечение JavaDoc комментария."""
        return extract_javadoc(node, source_code)

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
        if modifiers.get('abstract'):
            parts.append('abstract')
        if modifiers.get('final'):
            parts.append('final')
        if modifiers.get('static'):
            parts.append('static')

        # Class name
        parts.append('class')
        class_name = name
        if generics:
            class_name += f"<{', '.join(generics)}>"
        parts.append(class_name)

        # Extends
        if base_classes:
            parts.append(f"extends {', '.join(base_classes)}")

        # Implements
        if interfaces:
            parts.append(f"implements {', '.join(interfaces)}")

        return ' '.join(parts)

    def _build_method_signature(
            self,
            name: str,
            parameters: List[Dict],
            return_type: Optional[str],
            modifiers: Dict
    ) -> str:
        """Построение сигнатуры метода."""
        parts = []

        # Модификаторы
        if modifiers.get('access'):
            parts.append(modifiers['access'])
        if modifiers.get('static'):
            parts.append('static')
        if modifiers.get('final'):
            parts.append('final')
        if modifiers.get('abstract'):
            parts.append('abstract')

        # Return type
        if return_type:
            parts.append(return_type)

        # Method name and parameters
        param_str = ", ".join([f"{p['type']} {p['name']}" for p in parameters])
        parts.append(f"{name}({param_str})")

        return ' '.join(parts)

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Получение текста узла."""
        return source_code[node.start_byte:node.end_byte]


# Удобная функция для быстрого парсинга
def parse_java_file(file_path: str, **kwargs) -> Optional[ModuleInfo]:
    """
    Быстрый парсинг Java файла.

    Args:
        file_path: Путь к Java файлу
        **kwargs: Дополнительные параметры

    Returns:
        ModuleInfo или None
    """
    parser = JavaParser()
    return parser.parse_file(Path(file_path), **kwargs)
