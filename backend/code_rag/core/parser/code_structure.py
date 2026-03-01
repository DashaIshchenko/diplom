"""
Структуры данных для представления элементов кода (мультиязычные).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from enum import Enum
from pathlib import Path
import json


class ProgrammingLanguage(Enum):
    """Поддерживаемые языки программирования."""
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CSHARP = "csharp"
    KOTLIN = "kotlin"
    HTML = "html"
    CSS = "css"

    @classmethod
    def from_extension(cls, extension: str) -> Optional['ProgrammingLanguage']:
        """
        Определение языка по расширению файла.

        Args:
            extension: Расширение файла (с точкой)

        Returns:
            ProgrammingLanguage или None
        """
        extension_map = {
            '.py': cls.PYTHON,
            '.pyi': cls.PYTHON,
            '.java': cls.JAVA,
            '.js': cls.JAVASCRIPT,
            '.jsx': cls.JAVASCRIPT,
            '.mjs': cls.JAVASCRIPT,
            '.cjs': cls.JAVASCRIPT,
            '.ts': cls.TYPESCRIPT,
            '.tsx': cls.TYPESCRIPT,
            '.cs': cls.CSHARP,
            '.kt': cls.KOTLIN,
            '.kts': cls.KOTLIN,
            '.html': cls.HTML,
            '.htm': cls.HTML,
            '.css': cls.CSS,
            '.scss': cls.CSS,
            '.sass': cls.CSS,
        }
        return extension_map.get(extension.lower())

    @classmethod
    def is_programming_language(cls, language: 'ProgrammingLanguage') -> bool:
        """Проверка, является ли язык программирования (не разметки)."""
        markup_languages = {cls.HTML, cls.CSS}
        return language not in markup_languages


class CodeElementType(Enum):
    """Типы элементов кода."""
    MODULE = "module"
    CLASS = "class"
    INTERFACE = "interface"
    FUNCTION = "function"
    METHOD = "method"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"
    PROPERTY = "property"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"
    CONSTRUCTOR = "constructor"
    ENUM = "enum"
    NAMESPACE = "namespace"
    STRUCT = "struct"

    # HTML специфичные
    HTML_TAG = "html_tag"
    HTML_COMPONENT = "html_component"
    HTML_TEMPLATE = "html_template"

    # CSS специфичные
    CSS_RULE = "css_rule"
    CSS_CLASS = "css_class"
    CSS_ID = "css_id"
    CSS_SELECTOR = "css_selector"
    CSS_MEDIA_QUERY = "css_media_query"
    CSS_KEYFRAME = "css_keyframe"
    CSS_VARIABLE = "css_variable"

    def is_callable(self) -> bool:
        """Проверка, является ли элемент вызываемым (функция/метод)."""
        callable_types = {
            self.FUNCTION, self.METHOD, self.ASYNC_FUNCTION,
            self.ASYNC_METHOD, self.STATIC_METHOD, self.CLASS_METHOD,
            self.CONSTRUCTOR
        }
        return self in callable_types

    def is_class_like(self) -> bool:
        """Проверка, является ли элемент классом или интерфейсом."""
        class_types = {self.CLASS, self.INTERFACE, self.ENUM, self.STRUCT}
        return self in class_types


@dataclass
class CodeLocation:
    """Информация о местоположении кода в файле."""
    file_path: Path
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0

    @property
    def line_count(self) -> int:
        """Количество строк кода."""
        return self.end_line - self.start_line + 1

    @property
    def is_single_line(self) -> bool:
        """Проверка, является ли элемент однострочным."""
        return self.start_line == self.end_line

    def contains_line(self, line: int) -> bool:
        """Проверка, содержит ли локация указанную строку."""
        return self.start_line <= line <= self.end_line

    def overlaps_with(self, other: 'CodeLocation') -> bool:
        """Проверка пересечения с другой локацией."""
        return not (self.end_line < other.start_line or self.start_line > other.end_line)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "file_path": str(self.file_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_col": self.start_col,
            "end_col": self.end_col,
            "line_count": self.line_count
        }

    def __repr__(self) -> str:
        return f"{self.file_path.name}:{self.start_line}-{self.end_line}"


@dataclass
class CodeElement:
    """
    Универсальная структура для элемента кода (любого языка).
    """
    name: str
    type: CodeElementType
    language: ProgrammingLanguage
    location: CodeLocation
    source_code: str
    qualified_name: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)  # Python decorators / Java annotations
    parent: Optional[str] = None
    complexity: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    repository_name: Optional[str] = None

    # Функции/методы
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    access_modifier: Optional[str] = None  # public, private, protected, internal
    is_static: bool = False
    is_abstract: bool = False
    is_final: bool = False  # final (Java), sealed (C#)

    # Классы/интерфейсы
    base_classes: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)  # Implemented interfaces
    methods: List['CodeElement'] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    generics: List[str] = field(default_factory=list)  # Generic types

    # Зависимости
    imports: List[str] = field(default_factory=list)
    called_functions: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Если поле qualified_name не передано — генерируем автоматически
        if not self.qualified_name:
            if self.parent:
                full = f"{self.parent}.{self.name}"
            else:
                full = self.name
            self.qualified_name = f"{self.language.value}::{full}"

    @property
    def full_name(self) -> str:
        """Полное имя элемента (включая родителя)."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    @property
    def text_representation(self) -> str:
        """Текстовое представление для эмбеддинга."""
        parts = []

        # Язык и тип
        parts.append(f"Language: {self.language.value}")
        parts.append(f"Type: {self.type.value}")
        parts.append(f"Name: {self.full_name}")

        # Модификаторы доступа
        modifiers = []
        if self.access_modifier:
            modifiers.append(self.access_modifier)
        if self.is_static:
            modifiers.append("static")
        if self.is_abstract:
            modifiers.append("abstract")
        if self.is_final:
            modifiers.append("final")
        if modifiers:
            parts.append(f"Modifiers: {' '.join(modifiers)}")

        # Сигнатура
        if self.signature:
            parts.append(f"Signature: {self.signature}")

        # Docstring / JavaDoc / XML Doc
        if self.docstring:
            doc = self.docstring[:500] + "..." if len(self.docstring) > 500 else self.docstring
            parts.append(f"Documentation: {doc}")

        # Параметры
        if self.parameters:
            param_info = ", ".join([
                f"{p['name']}: {p.get('type', 'Any')}"
                for p in self.parameters
            ])
            parts.append(f"Parameters: {param_info}")

        # Return type
        if self.return_type:
            parts.append(f"Returns: {self.return_type}")

        # Родитель
        if self.parent:
            parts.append(f"Parent: {self.parent}")

        # Наследование
        if self.base_classes:
            parts.append(f"Extends: {', '.join(self.base_classes)}")

        if self.interfaces:
            parts.append(f"Implements: {', '.join(self.interfaces)}")

        # Декораторы/аннотации
        if self.decorators:
            parts.append(f"Annotations: {', '.join(self.decorators)}")

        # Generics
        if self.generics:
            parts.append(f"Generics: {', '.join(self.generics)}")

        return "\n".join(parts)

    @property
    def char_count(self) -> int:
        """Количество символов в коде (без пробелов)."""
        return len(self.source_code.replace(" ", "").replace("\n", "").replace("\t", ""))

    @property
    def token_estimate(self) -> int:
        """Примерная оценка количества токенов."""
        return len(self.source_code) // 4

    @property
    def is_public(self) -> bool:
        """Проверка, является ли элемент публичным."""
        return self.access_modifier == "public" or self.access_modifier is None

    @property
    def is_private(self) -> bool:
        """Проверка, является ли элемент приватным."""
        return self.access_modifier == "private"

    @property
    def has_documentation(self) -> bool:
        """Проверка наличия документации."""
        return self.docstring is not None and len(self.docstring.strip()) > 0

    def get_parameter_names(self) -> List[str]:
        """Получение списка имен параметров."""
        return [p['name'] for p in self.parameters]

    def get_all_methods(self) -> List['CodeElement']:
        """Получение всех методов (включая вложенные)."""
        all_methods = list(self.methods)
        for method in self.methods:
            all_methods.extend(method.get_all_methods())
        return all_methods

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "type": self.type.value,
            "element_type": self.type.value,  # ✅ Дублируем для совместимости
            "language": self.language.value,

            # ✅ Сохраняем location и как объект, и как отдельные поля
            "location": self.location.to_dict(),
            "file_path": str(self.location.file_path),
            "start_line": self.location.start_line,
            "end_line": self.location.end_line,
            "line_count": self.location.end_line - self.location.start_line + 1,

            "source_code": self.source_code,
            "docstring": self.docstring,
            "signature": self.signature,
            "decorators": self.decorators,
            "parent": self.parent,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "is_async": self.is_async,
            "access_modifier": self.access_modifier,
            "is_static": self.is_static,
            "is_abstract": self.is_abstract,
            "is_final": self.is_final,
            "base_classes": self.base_classes,
            "interfaces": self.interfaces,
            "attributes": self.attributes,
            "generics": self.generics,
            "imports": self.imports,
            "called_functions": self.called_functions,
            "complexity": self.complexity,
            "metadata": self.metadata,
            "repository_name": self.repository_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeElement":
        """Десериализация из словаря."""
        from pathlib import Path

        # Восстанавливаем Location
        if "location" in data and isinstance(data["location"], dict):
            loc_data = data["location"]
            location = CodeLocation(
                file_path=Path(loc_data.get("file_path", "")),
                start_line=loc_data.get("start_line", 0),
                end_line=loc_data.get("end_line", 0),
                start_col=loc_data.get("start_col", 0),
                end_col=loc_data.get("end_col", 0),
            )
        else:
            location = CodeLocation(
                file_path=Path(data.get("file_path", "")),
                start_line=data.get("start_line", 0),
                end_line=data.get("end_line", 0),
                start_col=data.get("start_col", 0),
                end_col=data.get("end_col", 0),
            )

        # Восстанавливаем Enums
        element_type = CodeElementType(
            data.get("type") or data.get("element_type", "unknown")
        )
        language = ProgrammingLanguage(data.get("language", "unknown"))

        # ✅ Явно создаём объект со всеми нужными полями
        return cls(
            name=data.get("name", ""),
            type=element_type,
            language=language,
            location=location,
            source_code=data.get("source_code", ""),
            qualified_name=data.get("qualified_name"),
            docstring=data.get("docstring"),
            signature=data.get("signature"),
            decorators=data.get("decorators", []),
            parent=data.get("parent"),
            complexity=data.get("complexity", 0),
            metadata=data.get("metadata", {}),
            parameters=data.get("parameters", []),
            return_type=data.get("return_type"),
            is_async=data.get("is_async", False),
            access_modifier=data.get("access_modifier"),
            is_static=data.get("is_static", False),
            is_abstract=data.get("is_abstract", False),
            is_final=data.get("is_final", False),
            base_classes=data.get("base_classes", []),
            interfaces=data.get("interfaces", []),
            methods=data.get("methods", []),
            attributes=data.get("attributes", []),
            generics=data.get("generics", []),
            imports=data.get("imports", []),
            called_functions=data.get("called_functions", []),
            repository_name=data.get("repository_name", ""),
        )

    def __repr__(self) -> str:
        return f"CodeElement({self.language.value}/{self.type.value}: {self.full_name}, lines={self.location.line_count})"


@dataclass
class ModuleInfo:
    """Информация о модуле/файле кода."""
    file_path: Path
    module_name: str
    language: ProgrammingLanguage
    docstring: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    classes: List[CodeElement] = field(default_factory=list)
    functions: List[CodeElement] = field(default_factory=list)
    interfaces: List[CodeElement] = field(default_factory=list)
    enums: List[CodeElement] = field(default_factory=list)
    total_lines: int = 0

    # Git метаданные
    repository_name: Optional[str] = None
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    provider: Optional[str] = None

    @property
    def all_elements(self) -> List[CodeElement]:
        """Все элементы кода в модуле."""
        elements = []
        elements.extend(self.functions)
        elements.extend(self.classes)
        elements.extend(self.interfaces)
        elements.extend(self.enums)

        # Добавляем методы классов
        for cls in self.classes:
            elements.extend(cls.methods)

        # Добавляем методы интерфейсов
        for interface in self.interfaces:
            elements.extend(interface.methods)

        return elements

    @property
    def element_count(self) -> int:
        """Общее количество элементов."""
        return len(self.all_elements)

    @property
    def total_complexity(self) -> int:
        """Суммарная сложность всех элементов."""
        return sum(element.complexity for element in self.all_elements)

    @property
    def average_complexity(self) -> float:
        """Средняя сложность элементов."""
        elements = self.all_elements
        if not elements:
            return 0.0
        return self.total_complexity / len(elements)

    @property
    def documented_elements_count(self) -> int:
        """Количество элементов с документацией."""
        return sum(1 for e in self.all_elements if e.has_documentation)

    @property
    def documentation_coverage(self) -> float:
        """Процент элементов с документацией."""
        total = len(self.all_elements)
        if total == 0:
            return 0.0
        return (self.documented_elements_count / total) * 100

    def get_elements_by_type(self, element_type: CodeElementType) -> List[CodeElement]:
        """Получение элементов по типу."""
        return [e for e in self.all_elements if e.type == element_type]

    def get_public_elements(self) -> List[CodeElement]:
        """Получение публичных элементов."""
        return [e for e in self.all_elements if e.is_public]

    def get_complex_elements(self, threshold: int = 10) -> List[CodeElement]:
        """Получение сложных элементов (сложность > threshold)."""
        return [e for e in self.all_elements if e.complexity > threshold]

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация модуля."""
        return {
            "file_path": str(self.file_path),
            "module_name": self.module_name,
            "language": self.language.value,
            "docstring": self.docstring,
            "imports": self.imports,
            "classes": [cls.to_dict() for cls in self.classes],
            "functions": [func.to_dict() for func in self.functions],
            "interfaces": [iface.to_dict() for iface in self.interfaces],
            "enums": [enum.to_dict() for enum in self.enums],
            "total_lines": self.total_lines,
            "element_count": self.element_count,
            "total_complexity": self.total_complexity,
            "average_complexity": self.average_complexity,
            "documentation_coverage": self.documentation_coverage,
            "repository_name": self.repository_name,
            "branch": self.branch,
            "commit_hash": self.commit_hash,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleInfo":
        """Десериализация модуля."""
        # Восстанавливаем Path
        data["file_path"] = Path(data["file_path"])

        # Восстанавливаем язык
        data["language"] = ProgrammingLanguage(data["language"])

        # Восстанавливаем элементы
        classes = [CodeElement.from_dict(c) for c in data.pop("classes", [])]
        functions = [CodeElement.from_dict(f) for f in data.pop("functions", [])]
        interfaces = [CodeElement.from_dict(i) for i in data.pop("interfaces", [])]
        enums = [CodeElement.from_dict(e) for e in data.pop("enums", [])]

        # Удаляем computed свойства
        data.pop("element_count", None)
        data.pop("total_complexity", None)
        data.pop("average_complexity", None)
        data.pop("documentation_coverage", None)

        return cls(
            classes=classes,
            functions=functions,
            interfaces=interfaces,
            enums=enums,
            **data
        )

    def __repr__(self) -> str:
        return (f"ModuleInfo({self.language.value}/{self.module_name}, "
                f"classes={len(self.classes)}, "
                f"functions={len(self.functions)}, "
                f"elements={self.element_count})")


@dataclass
class ParseResult:
    """Результат парсинга файла или репозитория."""
    modules: List[ModuleInfo] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    @property
    def total_elements(self) -> int:
        """Общее количество элементов во всех модулях."""
        return sum(module.element_count for module in self.modules)

    @property
    def total_lines(self) -> int:
        """Общее количество строк кода."""
        return sum(module.total_lines for module in self.modules)

    @property
    def success_rate(self) -> float:
        """Процент успешно распарсенных файлов."""
        total = len(self.modules) + len(self.errors)
        if total == 0:
            return 0.0
        return len(self.modules) / total * 100

    @property
    def total_complexity(self) -> int:
        """Суммарная сложность всего кода."""
        return sum(module.total_complexity for module in self.modules)

    @property
    def languages(self) -> Set[ProgrammingLanguage]:
        """Множество обнаруженных языков."""
        return set(module.language for module in self.modules)

    @property
    def average_documentation_coverage(self) -> float:
        """Средний процент документирования."""
        if not self.modules:
            return 0.0
        return sum(m.documentation_coverage for m in self.modules) / len(self.modules)

    def get_all_elements(self) -> List[CodeElement]:
        """Получить все элементы из всех модулей."""
        elements = []
        for module in self.modules:
            elements.extend(module.all_elements)
        return elements

    def get_elements_by_type(self, element_type: CodeElementType) -> List[CodeElement]:
        """Получение элементов по типу из всех модулей."""
        elements = []
        for module in self.modules:
            elements.extend(module.get_elements_by_type(element_type))
        return elements

    def get_modules_by_language(self, language: ProgrammingLanguage) -> List[ModuleInfo]:
        """Получение модулей по языку."""
        return [m for m in self.modules if m.language == language]

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики парсинга."""
        all_elements = self.get_all_elements()

        # Статистика по типам
        type_counts = {}
        for element in all_elements:
            type_counts[element.type.value] = type_counts.get(element.type.value, 0) + 1

        # Статистика по языкам
        language_counts = {}
        for module in self.modules:
            lang = module.language.value
            language_counts[lang] = language_counts.get(lang, 0) + 1

        # Статистика по сложности
        complexities = [e.complexity for e in all_elements if e.type.is_callable()]

        return {
            "total_modules": len(self.modules),
            "total_elements": self.total_elements,
            "total_lines": self.total_lines,
            "total_errors": len(self.errors),
            "success_rate": self.success_rate,
            "total_complexity": self.total_complexity,
            "average_complexity": sum(complexities) / len(complexities) if complexities else 0,
            "max_complexity": max(complexities) if complexities else 0,
            "min_complexity": min(complexities) if complexities else 0,
            "documentation_coverage": self.average_documentation_coverage,
            "by_type": type_counts,
            "by_language": language_counts,
            "languages": [lang.value for lang in self.languages]
        }

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация результата."""
        return {
            "modules": [module.to_dict() for module in self.modules],
            "errors": self.errors,
            "statistics": self.get_statistics(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Сериализация в JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __repr__(self) -> str:
        return (f"ParseResult(modules={len(self.modules)}, "
                f"elements={self.total_elements}, "
                f"errors={len(self.errors)}, "
                f"success_rate={self.success_rate:.1f}%)")
