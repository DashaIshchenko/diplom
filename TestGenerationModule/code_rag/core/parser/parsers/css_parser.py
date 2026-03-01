"""
CSS парсер на основе tree-sitter.
Извлекает селекторы, правила, media queries, keyframes и их метаданные.
Поддерживает CSS, SCSS, SASS и LESS синтаксис.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import re

try:
    from tree_sitter import Language, Parser, Node

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("tree-sitter не установлен. CSS парсер будет недоступен.")

from ..base_parser import BaseParser
from ..code_structure import (
    CodeElement, CodeElementType, CodeLocation,
    ModuleInfo, ProgrammingLanguage
)
from ..utils.tree_sitter_helper import TreeSitterHelper

logger = logging.getLogger(__name__)


class CSSParser(BaseParser):
    """
    Парсер CSS кода на основе tree-sitter.

    Извлекает:
    - Селекторы (.class, #id, element, :pseudo, [attribute])
    - CSS Rules (наборы свойств)
    - Media queries (@media)
    - Keyframes (@keyframes)
    - CSS Variables (--custom-property)
    - Import statements (@import)
    - Font-face declarations (@font-face)
    - Специфичность селекторов
    - Vendor prefixes
    """

    # Vendor prefixes
    VENDOR_PREFIXES = {'-webkit-', '-moz-', '-ms-', '-o-'}

    # Важные CSS свойства для категоризации
    LAYOUT_PROPERTIES = {
        'display', 'position', 'float', 'clear', 'flex', 'grid',
        'width', 'height', 'margin', 'padding', 'box-sizing'
    }

    VISUAL_PROPERTIES = {
        'color', 'background', 'border', 'box-shadow', 'text-shadow',
        'opacity', 'visibility', 'filter'
    }

    TYPOGRAPHY_PROPERTIES = {
        'font', 'font-size', 'font-weight', 'font-family', 'line-height',
        'text-align', 'text-decoration', 'letter-spacing', 'word-spacing'
    }

    ANIMATION_PROPERTIES = {
        'animation', 'transition', 'transform', 'keyframes'
    }

    def __init__(self, max_chunk_size: int = 8192):
        """
        Инициализация CSS парсера.

        Args:
            max_chunk_size: Максимальный размер чанка
        """
        super().__init__(max_chunk_size)

        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter не установлен. Установите: pip install tree-sitter"
            )

        # Инициализация tree-sitter для CSS
        self.ts_helper = TreeSitterHelper()
        self.parser = self.ts_helper.get_parser('css')

        if not self.parser:
            raise RuntimeError("Не удалось инициализировать CSS парсер")

    @property
    def language(self) -> ProgrammingLanguage:
        """Язык программирования."""
        return ProgrammingLanguage.CSS

    @property
    def file_extensions(self) -> List[str]:
        """Поддерживаемые расширения файлов."""
        return ['.css', '.scss', '.sass', '.less']

    def parse_file(
            self,
            file_path: Path,
            repository_name: Optional[str] = None,
            branch: Optional[str] = None,
            commit_hash: Optional[str] = None,
            provider: Optional[str] = None
    ) -> Optional[ModuleInfo]:
        """
        Парсинг CSS файла.

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

            # Извлекаем imports
            imports = self._extract_imports(root_node, source_code)

            # Создаем объект модуля
            module_info = ModuleInfo(
                file_path=file_path,
                module_name=module_name,
                language=self.language,
                docstring=f"CSS file type: {file_type}",
                imports=imports,
                total_lines=len(source_code.split('\n')),
                repository_name=repository_name,
                branch=branch,
                commit_hash=commit_hash,
                provider=provider
            )

            # Извлекаем элементы
            self._extract_elements(root_node, source_code, file_path, module_info)

            # Анализируем статистику
            stats = self._analyze_css_stats(module_info)
            module_info.metadata = {
                'file_type': file_type,
                'total_rules': stats['total_rules'],
                'total_selectors': stats['total_selectors'],
                'media_queries': stats['media_queries'],
                'keyframes': stats['keyframes'],
                'css_variables': stats['css_variables'],
                'vendor_prefixes': stats['vendor_prefixes'],
                'max_specificity': stats['max_specificity']
            }

            logger.info(
                f"Распарсен CSS файл {file_path.name}: "
                f"{stats['total_rules']} правил, "
                f"{stats['total_selectors']} селекторов"
            )

            return module_info

        except Exception as e:
            logger.error(f"Ошибка парсинга CSS файла {file_path}: {e}")
            return None

    def _extract_elements(
            self,
            node: Node,
            source_code: str,
            file_path: Path,
            module_info: ModuleInfo
    ) -> None:
        """Извлечение элементов из CSS."""
        for child in node.children:
            # CSS Rules (селектор + блок свойств)
            if child.type == 'rule_set':
                rule_element = self._parse_rule_set(child, source_code, file_path)
                if rule_element:
                    module_info.functions.append(rule_element)

            # Media queries
            elif child.type == 'media_statement':
                media_element = self._parse_media_query(child, source_code, file_path)
                if media_element:
                    module_info.classes.append(media_element)

                    # Извлекаем правила внутри media query
                    for subchild in child.children:
                        if subchild.type == 'block':
                            self._extract_elements(subchild, source_code, file_path, module_info)

            # Keyframes
            elif child.type == 'keyframes_statement':
                keyframe_element = self._parse_keyframes(child, source_code, file_path)
                if keyframe_element:
                    module_info.classes.append(keyframe_element)

            # Font-face
            elif child.type == 'font_face_statement':
                fontface_element = self._parse_font_face(child, source_code, file_path)
                if fontface_element:
                    module_info.functions.append(fontface_element)

            # Import statements
            elif child.type == 'import_statement':
                # Уже обработано в _extract_imports
                pass

            # Рекурсивно обрабатываем другие узлы
            else:
                self._extract_elements(child, source_code, file_path, module_info)

    def _parse_rule_set(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг CSS rule (селектор + свойства)."""
        try:
            # Извлекаем селекторы
            selectors_node = node.child_by_field_name('selectors')
            if not selectors_node:
                return None

            selectors = self._extract_selectors(selectors_node, source_code)
            if not selectors:
                return None

            # Исходный код правила
            rule_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # Извлекаем блок свойств
            block_node = node.child_by_field_name('block')
            properties = {}
            if block_node:
                properties = self._extract_properties(block_node, source_code)

            # Анализируем селекторы
            primary_selector = selectors[0]
            selector_type = self._determine_selector_type(primary_selector)
            specificity = self._calculate_specificity(primary_selector)

            # Категоризируем свойства
            property_categories = self._categorize_properties(properties)

            # Проверяем vendor prefixes
            vendor_prefixes = self._find_vendor_prefixes(properties)

            # Извлекаем CSS переменные
            css_variables = self._find_css_variables(properties)

            # Формируем сигнатуру
            signature = f"{primary_selector} {{ {len(properties)} properties }}"

            # Формируем docstring
            docstring = self._build_css_docstring(
                selectors, properties, property_categories, specificity
            )

            # Определяем имя элемента
            element_name = self._get_selector_name(primary_selector)

            rule_element = CodeElement(
                name=element_name,
                type=selector_type,
                language=self.language,
                location=location,
                source_code=rule_source,
                docstring=docstring,
                signature=signature,
                decorators=selectors[1:] if len(selectors) > 1 else [],  # Дополнительные селекторы
                complexity=specificity[0] + specificity[1] + specificity[2],  # Сумма специфичности
                metadata={
                    'selectors': selectors,
                    'primary_selector': primary_selector,
                    'specificity': specificity,
                    'properties': properties,
                    'property_count': len(properties),
                    'property_categories': property_categories,
                    'vendor_prefixes': vendor_prefixes,
                    'css_variables': css_variables,
                    'is_responsive': False,  # Будет установлен в media query
                }
            )

            return rule_element

        except Exception as e:
            logger.debug(f"Не удалось распарсить CSS rule: {e}")
            return None

    def _parse_media_query(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг media query."""
        try:
            # Исходный код
            media_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # Извлекаем условие media query
            query_node = node.child_by_field_name('query')
            if query_node:
                query_text = self._get_node_text(query_node, source_code)
            else:
                query_text = "unknown"

            # Подсчитываем вложенные правила
            nested_rules = 0
            block_node = node.child_by_field_name('block')
            if block_node:
                for child in block_node.children:
                    if child.type == 'rule_set':
                        nested_rules += 1

            media_element = CodeElement(
                name=f"media_{location.start_line}",
                type=CodeElementType.CSS_MEDIA_QUERY,
                language=self.language,
                location=location,
                source_code=media_source,
                docstring=f"Media query: {query_text}",
                signature=f"@media {query_text}",
                complexity=nested_rules,
                metadata={
                    'query': query_text,
                    'nested_rules': nested_rules,
                    'breakpoint': self._extract_breakpoint(query_text)
                }
            )

            return media_element

        except Exception as e:
            logger.debug(f"Не удалось распарсить media query: {e}")
            return None

    def _parse_keyframes(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг keyframes анимации."""
        try:
            # Исходный код
            keyframes_source = self._get_node_text(node, source_code)

            # Местоположение
            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # Извлекаем имя анимации
            name_node = node.child_by_field_name('name')
            if name_node:
                animation_name = self._get_node_text(name_node, source_code)
            else:
                animation_name = f"animation_{location.start_line}"

            # Подсчитываем keyframe селекторы (0%, 50%, 100% и т.д.)
            keyframe_count = 0
            for child in node.children:
                if child.type == 'keyframe_block_list':
                    for subchild in child.children:
                        if subchild.type == 'keyframe_block':
                            keyframe_count += 1

            keyframe_element = CodeElement(
                name=animation_name,
                type=CodeElementType.CSS_KEYFRAME,
                language=self.language,
                location=location,
                source_code=keyframes_source,
                docstring=f"CSS Animation: {animation_name}",
                signature=f"@keyframes {animation_name}",
                complexity=keyframe_count,
                metadata={
                    'animation_name': animation_name,
                    'keyframe_count': keyframe_count
                }
            )

            return keyframe_element

        except Exception as e:
            logger.debug(f"Не удалось распарсить keyframes: {e}")
            return None

    def _parse_font_face(
            self,
            node: Node,
            source_code: str,
            file_path: Path
    ) -> Optional[CodeElement]:
        """Парсинг @font-face."""
        try:
            fontface_source = self._get_node_text(node, source_code)

            location = CodeLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_col=node.start_point[1],
                end_col=node.end_point[1]
            )

            # Извлекаем свойства
            properties = {}
            for child in node.children:
                if child.type == 'block':
                    properties = self._extract_properties(child, source_code)

            font_family = properties.get('font-family', 'unknown')

            return CodeElement(
                name=f"font_face_{location.start_line}",
                type=CodeElementType.FUNCTION,
                language=self.language,
                location=location,
                source_code=fontface_source,
                signature=f"@font-face {{ font-family: {font_family} }}",
                metadata={
                    'properties': properties,
                    'font_family': font_family
                }
            )

        except Exception as e:
            logger.debug(f"Не удалось распарсить @font-face: {e}")
            return None

    def _extract_selectors(self, selectors_node: Node, source_code: str) -> List[str]:
        """Извлечение селекторов."""
        selectors = []

        for child in selectors_node.children:
            if child.type in ['class_selector', 'id_selector', 'tag_name',
                              'universal_selector', 'attribute_selector',
                              'pseudo_class_selector', 'pseudo_element_selector']:
                selector_text = self._get_node_text(child, source_code).strip()
                if selector_text and selector_text != ',':
                    selectors.append(selector_text)

        # Если селекторы не найдены, берем весь текст
        if not selectors:
            full_text = self._get_node_text(selectors_node, source_code).strip()
            selectors = [s.strip() for s in full_text.split(',') if s.strip()]

        return selectors

    def _extract_properties(self, block_node: Node, source_code: str) -> Dict[str, str]:
        """Извлечение CSS свойств."""
        properties = {}

        for child in block_node.children:
            if child.type == 'declaration':
                # Имя свойства
                property_name_node = child.child_by_field_name('property')
                if not property_name_node:
                    continue
                property_name = self._get_node_text(property_name_node, source_code).strip()

                # Значение свойства
                value_node = child.child_by_field_name('value')
                if value_node:
                    property_value = self._get_node_text(value_node, source_code).strip()
                else:
                    property_value = ''

                properties[property_name] = property_value

        return properties

    def _extract_imports(self, node: Node, source_code: str) -> List[str]:
        """Извлечение @import statements."""
        imports = []

        for child in node.children:
            if child.type == 'import_statement':
                import_text = self._get_node_text(child, source_code)
                # Извлекаем URL из @import
                url_match = re.search(r'["\']([^"\']+)["\']', import_text)
                if url_match:
                    imports.append(url_match.group(1))
                else:
                    imports.append(import_text)

        return imports

    def _determine_selector_type(self, selector: str) -> CodeElementType:
        """Определение типа селектора."""
        selector = selector.strip()

        if selector.startswith('.'):
            return CodeElementType.CSS_CLASS
        elif selector.startswith('#'):
            return CodeElementType.CSS_ID
        elif selector.startswith('['):
            return CodeElementType.CSS_SELECTOR
        elif selector.startswith(':'):
            return CodeElementType.CSS_SELECTOR
        else:
            return CodeElementType.CSS_RULE

    def _calculate_specificity(self, selector: str) -> Tuple[int, int, int]:
        """
        Расчет специфичности CSS селектора.

        Возвращает (ids, classes, elements)
        Например: #header .nav li -> (1, 1, 1)
        """
        ids = selector.count('#')
        classes = selector.count('.') + selector.count('[')

        # Псевдоклассы тоже считаются как классы
        pseudo_classes = len(re.findall(r':[a-zA-Z-]+', selector))
        classes += pseudo_classes

        # Элементы (примерная оценка)
        # Убираем специальные символы и считаем слова
        clean_selector = re.sub(r'[#.\[\]:>+~]', ' ', selector)
        words = [w for w in clean_selector.split() if w]
        elements = len(words) - ids - classes
        elements = max(0, elements)

        return (ids, classes, elements)

    def _categorize_properties(self, properties: Dict[str, str]) -> Dict[str, List[str]]:
        """Категоризация CSS свойств."""
        categories = {
            'layout': [],
            'visual': [],
            'typography': [],
            'animation': [],
            'other': []
        }

        for prop in properties.keys():
            prop_lower = prop.lower()

            if any(p in prop_lower for p in self.LAYOUT_PROPERTIES):
                categories['layout'].append(prop)
            elif any(p in prop_lower for p in self.VISUAL_PROPERTIES):
                categories['visual'].append(prop)
            elif any(p in prop_lower for p in self.TYPOGRAPHY_PROPERTIES):
                categories['typography'].append(prop)
            elif any(p in prop_lower for p in self.ANIMATION_PROPERTIES):
                categories['animation'].append(prop)
            else:
                categories['other'].append(prop)

        return categories

    def _find_vendor_prefixes(self, properties: Dict[str, str]) -> List[str]:
        """Поиск vendor prefixes в свойствах."""
        prefixes = []

        for prop in properties.keys():
            for prefix in self.VENDOR_PREFIXES:
                if prop.startswith(prefix):
                    prefixes.append(prefix.strip('-'))
                    break

        return list(set(prefixes))

    def _find_css_variables(self, properties: Dict[str, str]) -> List[str]:
        """Поиск CSS переменных (--custom-property)."""
        variables = []

        # Переменные в ключах (объявления)
        for prop in properties.keys():
            if prop.startswith('--'):
                variables.append(prop)

        # Переменные в значениях (использование)
        for value in properties.values():
            var_matches = re.findall(r'var\((--[a-zA-Z0-9-]+)\)', value)
            variables.extend(var_matches)

        return list(set(variables))

    def _extract_breakpoint(self, query_text: str) -> Optional[str]:
        """Извлечение breakpoint из media query."""
        # Ищем min-width или max-width
        width_match = re.search(r'(min|max)-width:\s*(\d+)(px|em|rem)', query_text)
        if width_match:
            return f"{width_match.group(1)}-width: {width_match.group(2)}{width_match.group(3)}"
        return None

    def _get_selector_name(self, selector: str) -> str:
        """Получение имени из селектора для использования в name."""
        selector = selector.strip()

        # Удаляем специальные символы
        name = re.sub(r'[#.:>\[\]+~\s]', '_', selector)
        name = name.strip('_')

        # Ограничиваем длину
        if len(name) > 50:
            name = name[:50]

        return name or 'unnamed_rule'

    def _build_css_docstring(
            self,
            selectors: List[str],
            properties: Dict[str, str],
            categories: Dict[str, List[str]],
            specificity: Tuple[int, int, int]
    ) -> str:
        """Построение docstring для CSS rule."""
        parts = []

        # Селекторы
        if len(selectors) == 1:
            parts.append(f"CSS rule for: {selectors[0]}")
        else:
            parts.append(f"CSS rule for {len(selectors)} selectors: {', '.join(selectors[:3])}")

        # Специфичность
        spec_sum = sum(specificity)
        parts.append(f"Specificity: [{specificity[0]},{specificity[1]},{specificity[2]}] (total: {spec_sum})")

        # Категории свойств
        active_categories = [cat for cat, props in categories.items() if props]
        if active_categories:
            parts.append(f"Property categories: {', '.join(active_categories)}")

        # Важные свойства
        important_props = ['display', 'position', 'color', 'background', 'font-size']
        found_props = [prop for prop in important_props if prop in properties]
        if found_props:
            prop_values = [f"{prop}: {properties[prop]}" for prop in found_props[:3]]
            parts.append(f"Key properties: {', '.join(prop_values)}")

        return '\n'.join(parts)

    def _analyze_css_stats(self, module_info: ModuleInfo) -> Dict[str, Any]:
        """Анализ статистики CSS."""
        stats = {
            'total_rules': 0,
            'total_selectors': 0,
            'media_queries': 0,
            'keyframes': 0,
            'css_variables': set(),
            'vendor_prefixes': set(),
            'max_specificity': (0, 0, 0)
        }

        for element in module_info.all_elements:
            if element.type in [CodeElementType.CSS_RULE, CodeElementType.CSS_CLASS,
                                CodeElementType.CSS_ID, CodeElementType.CSS_SELECTOR]:
                stats['total_rules'] += 1

                if 'selectors' in element.metadata:
                    stats['total_selectors'] += len(element.metadata['selectors'])

                if 'specificity' in element.metadata:
                    spec = element.metadata['specificity']
                    if sum(spec) > sum(stats['max_specificity']):
                        stats['max_specificity'] = spec

                if 'css_variables' in element.metadata:
                    stats['css_variables'].update(element.metadata['css_variables'])

                if 'vendor_prefixes' in element.metadata:
                    stats['vendor_prefixes'].update(element.metadata['vendor_prefixes'])

            elif element.type == CodeElementType.CSS_MEDIA_QUERY:
                stats['media_queries'] += 1

            elif element.type == CodeElementType.CSS_KEYFRAME:
                stats['keyframes'] += 1

        # Конвертируем sets в списки для JSON
        stats['css_variables'] = list(stats['css_variables'])
        stats['vendor_prefixes'] = list(stats['vendor_prefixes'])

        return stats

    def _detect_file_type(self, file_path: Path, source_code: str) -> str:
        """Определение типа CSS файла."""
        extension = file_path.suffix.lower()

        if extension == '.scss':
            return 'SCSS (Sass)'
        elif extension == '.sass':
            return 'SASS'
        elif extension == '.less':
            return 'LESS'
        elif '@import' in source_code and 'node_modules' in source_code:
            return 'CSS with imports'
        elif ':root' in source_code or '--' in source_code:
            return 'CSS with variables'
        else:
            return 'Standard CSS'

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Получение текста узла."""
        return source_code[node.start_byte:node.end_byte]


# Удобная функция для быстрого парсинга
def parse_css_file(file_path: str, **kwargs) -> Optional[ModuleInfo]:
    """
    Быстрый парсинг CSS файла.

    Args:
        file_path: Путь к CSS файлу
        **kwargs: Дополнительные параметры

    Returns:
        ModuleInfo или None
    """
    parser = CSSParser()
    return parser.parse_file(Path(file_path), **kwargs)
