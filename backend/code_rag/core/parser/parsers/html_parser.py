"""
HTML парсер на основе tree-sitter.
Извлекает теги, компоненты, семантические элементы и их метаданные.
Поддерживает стандартный HTML, а также Vue, Svelte и другие template форматы.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import re

try:
    from tree_sitter import Language, Parser, Node

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("tree-sitter не установлен. HTML парсер будет недоступен.")

from ..base_parser import BaseParser
from ..code_structure import (
    CodeElement, CodeElementType, CodeLocation,
    ModuleInfo, ProgrammingLanguage
)
from ..utils.tree_sitter_helper import TreeSitterHelper

logger = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    """
    Парсер HTML кода на основе tree-sitter.

    Извлекает:
    - Семантические теги (header, main, section, article, nav, footer)
    - Форм-элементы (form, input, button, select, textarea)
    - Компоненты (custom elements, Web Components)
    - Template блоки (для Vue, Svelte, Angular)
    - Встроенные script и style теги
    - Атрибуты (id, class, data-*, aria-*)
    - Структурный анализ (вложенность, семантика)
    """

    # Семантические HTML5 теги
    SEMANTIC_TAGS = {
        'header', 'footer', 'main', 'nav', 'section', 'article',
        'aside', 'figure', 'figcaption', 'time', 'mark'
    }

    # Форм-элементы
    FORM_ELEMENTS = {
        'form', 'input', 'button', 'select', 'textarea',
        'label', 'fieldset', 'legend', 'datalist', 'output'
    }

    # Интерактивные элементы
    INTERACTIVE_ELEMENTS = {
        'button', 'a', 'input', 'select', 'textarea', 'details', 'summary'
    }

    def __init__(self, max_chunk_size: int = 8192):
        """
        Инициализация HTML парсера.

        Args:
            max_chunk_size: Максимальный размер чанка
        """
        super().__init__(max_chunk_size)

        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter не установлен. Установите: pip install tree-sitter"
            )

        # Инициализация tree-sitter для HTML
        self.ts_helper = TreeSitterHelper()
        self.parser = self.ts_helper.get_parser('html')

        if not self.parser:
            raise RuntimeError("Не удалось инициализировать HTML парсер")

    @property
    def language(self) -> ProgrammingLanguage:
        """Язык программирования."""
        return ProgrammingLanguage.HTML

    @property
    def file_extensions(self) -> List[str]:
        """Поддерживаемые расширения файлов."""
        return ['.html', '.htm', '.xhtml', '.vue', '.svelte']

    def parse_file(
            self,
            file_path: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None
    ) -> Optional[ModuleInfo]:
        """
        Парсинг HTML файла.

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

            # Определяем тип файла
            file_type = self._detect_file_type(file_path, source_code)

            # Создаем объект модуля
            module_info = ModuleInfo(
                file_path=file_path,
                module_name=module_name,
                language=self.language,
                docstring=f"HTML file type: {file_type}",
                imports=[],  # HTML не имеет импортов в классическом смысле
                total_lines=len(source_code.split('\n')),
                repository_name=repository_name,
                branch=branch,
                commit_hash=commit_hash,
                provider=provider
            )

            # Извлекаем элементы
            self._extract_elements(root_node, source_code, file_path, module_info)

            # Анализируем структуру
            structure_info = self._analyze_structure(root_node, source_code)
            module_info.metadata = {
                'file_type': file_type,
                'max_depth': structure_info['max_depth'],
                'total_tags': structure_info['total_tags'],
                'semantic_tags': structure_info['semantic_tags'],
                'form_elements': structure_info['form_elements'],
                'accessibility_score': structure_info['accessibility_score']
            }

            logger.info(
                f"Распарсен HTML файл {file_path.name}: "
                f"{len(module_info.classes)} компонентов, "
                f"глубина: {structure_info['max_depth']}"
            )

            return module_info

        except Exception as e:
            logger.error(f"Ошибка парсинга HTML файла {file_path}: {e}")
            return None

    def _extract_elements(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            module_info: ModuleInfo,
            depth: int = 0
    ) -> None:
        """Рекурсивное извлечение элементов из HTML."""
        for child in node.children:
            # HTML элементы (теги)
            if child.type == 'element':
                element = self._parse_html_element(child, source_code, file_path, depth)
                if element:
                    # Определяем куда добавить элемент
                    if element.type == CodeElementType.HTML_COMPONENT:
                        module_info.classes.append(element)
                    else:
                        module_info.functions.append(element)

            # Script теги (встроенный JavaScript)
            elif child.type == 'script_element':
                script_element = self._parse_script_element(child, source_code, file_path)
                if script_element:
                    module_info.functions.append(script_element)

            # Style теги (встроенный CSS)
            elif child.type == 'style_element':
                style_element = self._parse_style_element(child, source_code, file_path)
                if style_element:
                    module_info.functions.append(style_element)

            # Рекурсивно обрабатываем дочерние узлы
            else:
                self._extract_elements(child, source_code, file_path, module_info, depth)

    def _parse_html_element(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            depth: int
    ) -> Optional[CodeElement]:
        """Парсинг HTML элемента (тега)."""
        try:
            # Извлекаем тег
            start_tag = node.child_by_field_name('start_tag')
            if not start_tag:
                return None

            # Имя тега
            tag_name_node = start_tag.child_by_field_name('name')
            if not tag_name_node:
                return None
            tag_name = self._get_node_text(tag_name_node, source_code)

            # Исходный код элемента
            element_source = self._get_node_text(node, source_code)

            # Ограничиваем размер для больших элементов
            if len(element_source) > self.max_chunk_size:
                element_source = element_source[:self.max_chunk_size] + "\n... (truncated)"

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # Извлекаем атрибуты
            attributes = self._extract_attributes(start_tag, source_code)

            # Определяем тип элемента
            element_type = self._determine_element_type(tag_name, attributes)

            # Анализируем семантику
            is_semantic = tag_name.lower() in self.SEMANTIC_TAGS
            is_form_element = tag_name.lower() in self.FORM_ELEMENTS
            is_interactive = tag_name.lower() in self.INTERACTIVE_ELEMENTS
            is_custom_element = '-' in tag_name  # Web Components содержат дефис

            # Извлекаем классы и id
            classes = attributes.get('class', '').split() if 'class' in attributes else []
            element_id = attributes.get('id', '')

            # Анализируем accessibility
            aria_attributes = {k: v for k, v in attributes.items() if k.startswith('aria-')}
            has_role = 'role' in attributes

            # Подсчитываем дочерние элементы
            children_count = self._count_children(node)

            # Формируем сигнатуру
            signature = self._build_html_signature(tag_name, attributes, classes, element_id)

            # Формируем текстовое представление для эмбеддинга
            text_content = self._extract_text_content(node, source_code)
            docstring = self._build_html_docstring(
                tag_name, attributes, text_content, is_semantic, aria_attributes
            )

            # Создаем элемент
            html_element = CodeElement(
                name=element_id or f"{tag_name}_{location.start_line}",
                type=element_type,
                language=self.language,
                location=location,
                source_code=element_source,
                docstring=docstring,
                signature=signature,
                decorators=classes,  # Используем классы как "декораторы"
                attributes=list(attributes.keys()),
                complexity=depth,  # Используем глубину как меру сложности
                metadata={
                    'tag_name': tag_name,
                    'attributes': attributes,
                    'classes': classes,
                    'id': element_id,
                    'is_semantic': is_semantic,
                    'is_form_element': is_form_element,
                    'is_interactive': is_interactive,
                    'is_custom_element': is_custom_element,
                    'has_accessibility': bool(aria_attributes) or has_role,
                    'aria_attributes': aria_attributes,
                    'role': attributes.get('role'),
                    'children_count': children_count,
                    'depth': depth,
                    'text_content': text_content[:100] if text_content else None
                }
            )

            return html_element

        except Exception as e:
            logger.debug(f"Не удалось распарсить HTML элемент: {e}")
            return None

    def _parse_script_element(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг script тега (встроенный JavaScript)."""
        try:
            script_source = self._get_node_text(node, source_code)

            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # Извлекаем атрибуты script тега
            start_tag = node.child_by_field_name('start_tag')
            attributes = {}
            if start_tag:
                attributes = self._extract_attributes(start_tag, source_code)

            script_type = attributes.get('type', 'text/javascript')

            return CodeElement(
                name=f"script_{location.start_line}",
                type=CodeElementType.FUNCTION,
                language=self.language,
                location=location,
                source_code=script_source,
                signature=f"<script type='{script_type}'>",
                metadata={
                    'tag_name': 'script',
                    'type': script_type,
                    'attributes': attributes,
                    'is_inline': True
                }
            )

        except Exception as e:
            logger.debug(f"Не удалось распарсить script элемент: {e}")
            return None

    def _parse_style_element(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг style тега (встроенный CSS)."""
        try:
            style_source = self._get_node_text(node, source_code)

            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            return CodeElement(
                name=f"style_{location.start_line}",
                type=CodeElementType.FUNCTION,
                language=self.language,
                location=location,
                source_code=style_source,
                signature="<style>",
                metadata={
                    'tag_name': 'style',
                    'is_inline': True
                }
            )

        except Exception as e:
            logger.debug(f"Не удалось распарсить style элемент: {e}")
            return None

    def _extract_attributes(self, start_tag_node: Node, source_code: str) -> Dict[str, str]:
        """Извлечение атрибутов тега."""
        attributes = {}

        for child in start_tag_node.children:
            if child.type == 'attribute':
                # Имя атрибута
                attr_name_node = child.child_by_field_name('name')
                if not attr_name_node:
                    continue
                attr_name = self._get_node_text(attr_name_node, source_code)

                # Значение атрибута
                attr_value_node = child.child_by_field_name('value')
                if attr_value_node:
                    attr_value = self._get_node_text(attr_value_node, source_code)
                    # Убираем кавычки
                    attr_value = attr_value.strip('"\'')
                else:
                    # Boolean атрибут (без значения)
                    attr_value = 'true'

                attributes[attr_name] = attr_value

        return attributes

    def _determine_element_type(self, tag_name: str, attributes: Dict[str, str]) -> CodeElementType:
        """Определение типа HTML элемента."""
        tag_lower = tag_name.lower()

        # Web Component (custom element с дефисом)
        if '-' in tag_name:
            return CodeElementType.HTML_COMPONENT

        # Vue/Angular/Svelte компоненты (template, slot и т.д.)
        if tag_lower in ['template', 'slot', 'component']:
            return CodeElementType.HTML_TEMPLATE

        # Семантические элементы
        if tag_lower in self.SEMANTIC_TAGS:
            return CodeElementType.HTML_TAG

        # Обычный тег
        return CodeElementType.HTML_TAG

    def _count_children(self, node: Node) -> int:
        """Подсчет дочерних элементов."""
        count = 0
        for child in node.children:
            if child.type == 'element':
                count += 1
        return count

    def _extract_text_content(self, node: Node, source_code: str) -> str:
        """Извлечение текстового содержимого элемента."""
        text_parts = []

        for child in node.children:
            if child.type == 'text':
                text = self._get_node_text(child, source_code).strip()
                if text:
                    text_parts.append(text)

        return ' '.join(text_parts)

    def _build_html_signature(
            self,
            tag_name: str,
            attributes: Dict[str, str],
            classes: List[str],
            element_id: str
    ) -> str:
        """Построение сигнатуры HTML элемента."""
        parts = [f"<{tag_name}"]

        if element_id:
            parts.append(f"id='{element_id}'")

        if classes:
            parts.append(f"class='{' '.join(classes)}'")

        # Добавляем важные атрибуты
        important_attrs = ['type', 'name', 'role', 'href', 'src', 'alt']
        for attr in important_attrs:
            if attr in attributes:
                parts.append(f"{attr}='{attributes[attr]}'")

        return ' '.join(parts) + '>'

    def _build_html_docstring(
            self,
            tag_name: str,
            attributes: Dict[str, str],
            text_content: str,
            is_semantic: bool,
            aria_attributes: Dict[str, str]
    ) -> str:
        """Построение docstring для HTML элемента."""
        parts = []

        parts.append(f"HTML {tag_name} element")

        if is_semantic:
            parts.append(f"Semantic HTML5 tag: {tag_name}")

        if 'role' in attributes:
            parts.append(f"ARIA role: {attributes['role']}")

        if aria_attributes:
            aria_list = [f"{k}={v}" for k, v in aria_attributes.items()]
            parts.append(f"Accessibility attributes: {', '.join(aria_list)}")

        if text_content:
            content_preview = text_content[:100] + '...' if len(text_content) > 100 else text_content
            parts.append(f"Content: {content_preview}")

        # Добавляем важные атрибуты в описание
        if 'href' in attributes:
            parts.append(f"Link to: {attributes['href']}")

        if 'src' in attributes:
            parts.append(f"Source: {attributes['src']}")

        if 'alt' in attributes:
            parts.append(f"Alt text: {attributes['alt']}")

        return '\n'.join(parts)

    def _analyze_structure(self, node: Node, source_code: str) -> Dict[str, Any]:
        """Анализ структуры HTML документа."""
        stats = {
            'max_depth': 0,
            'total_tags': 0,
            'semantic_tags': 0,
            'form_elements': 0,
            'accessibility_score': 0
        }

        def traverse(n: Node, depth: int):
            stats['max_depth'] = max(stats['max_depth'], depth)

            if n.type == 'element':
                stats['total_tags'] += 1

                # Извлекаем имя тега
                start_tag = n.child_by_field_name('start_tag')
                if start_tag:
                    tag_name_node = start_tag.child_by_field_name('name')
                    if tag_name_node:
                        tag_name = self._get_node_text(tag_name_node, source_code).lower()

                        if tag_name in self.SEMANTIC_TAGS:
                            stats['semantic_tags'] += 1

                        if tag_name in self.FORM_ELEMENTS:
                            stats['form_elements'] += 1

                        # Проверяем accessibility
                        attributes = self._extract_attributes(start_tag, source_code)
                        if any(k.startswith('aria-') for k in attributes.keys()) or 'role' in attributes:
                            stats['accessibility_score'] += 1

            for child in n.children:
                traverse(child, depth + 1)

        traverse(node, 0)

        # Вычисляем процент accessibility
        if stats['total_tags'] > 0:
            stats['accessibility_score'] = (stats['accessibility_score'] / stats['total_tags']) * 100

        return stats

    def _detect_file_type(self, file_path: Path, source_code: str) -> str:
        """Определение типа HTML файла."""
        extension = file_path.suffix.lower()

        if extension == '.vue':
            return 'Vue Component'
        elif extension == '.svelte':
            return 'Svelte Component'
        elif '<template' in source_code and '</template>' in source_code:
            return 'Template-based Component'
        elif '<!DOCTYPE html>' in source_code or '<html' in source_code:
            return 'Full HTML Document'
        else:
            return 'HTML Fragment'

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Получение текста узла."""
        return source_code[node.start_byte:node.end_byte]


# Удобная функция для быстрого парсинга
def parse_html_file(file_path: str, **kwargs) -> Optional[ModuleInfo]:
    """
    Быстрый парсинг HTML файла.

    Args:
        file_path: Путь к HTML файлу
        **kwargs: Дополнительные параметры

    Returns:
        ModuleInfo или None
    """
    parser = HTMLParser()
    return parser.parse_file(Path(file_path), **kwargs)
