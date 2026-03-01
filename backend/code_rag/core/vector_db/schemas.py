"""
Схемы данных для векторной базы данных Qdrant.
Определяет структуру коллекций и payload для хранения кода.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

import qdrant_client.http
from qdrant_client.conversions.common_types import HnswConfigDiff
from qdrant_client.http.models import StrictModeConfig, OptimizersConfigDiff
from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        PayloadSchemaType
    )
from sympy.physics.vector.printing import params

from core.parser import ProgrammingLanguage, CodeElementType


class VectorDistance(Enum):
    """Метрики расстояния для векторов."""
    COSINE = "Cosine"
    EUCLIDEAN = "Euclidean"
    DOT = "Dot"

    def to_qdrant(self) -> Distance:
        """Конвертация в Qdrant Distance."""
        mapping = {
            self.COSINE: Distance.COSINE,
            self.EUCLIDEAN: Distance.EUCLID,
            self.DOT: Distance.DOT
        }
        return mapping[self]


@dataclass
class CodePayload:
    """
    Payload для хранения элемента кода в Qdrant.

    Содержит всю метаинформацию о коде для фильтрации и отображения.
    """
    # Основная информация
    name: str
    qualified_name: str
    element_type: str
    language: str

    # Код и документация
    source_code: str
    signature: Optional[str] = None
    docstring: Optional[str] = None

    # Местоположение
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    line_count: int = 0

    # Git метаданные
    repository_name: Optional[str] = None
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    provider: Optional[str] = None

    # Метрики
    complexity: int = 0
    char_count: int = 0
    token_estimate: int = 0

    # Иерархия
    parent: Optional[str] = None
    module_name: Optional[str] = None

    # Типы и параметры (для функций/методов)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None

    # Модификаторы
    access_modifier: Optional[str] = None
    is_async: bool = False
    is_static: bool = False
    is_abstract: bool = False
    is_final: bool = False

    # Наследование (для классов)
    base_classes: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    generics: List[str] = field(default_factory=list)

    # Декораторы/аннотации
    decorators: List[str] = field(default_factory=list)

    # Дополнительные атрибуты
    attributes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    # Метаданные
    has_documentation: bool = False
    is_public: bool = True

    # Временные метки
    indexed_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Дополнительная метаинформация
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь для Qdrant."""
        data = asdict(self)

        # Добавляем timestamp если не установлен
        if not data.get("indexed_at"):
            data["indexed_at"] = datetime.utcnow().isoformat()

        return data

    @classmethod
    def from_code_element(cls,
                          element,
                          repository_info: Optional[Union[str, Dict[str, Any]]] = None,
                          repository_name: Optional[str] = None,
                          branch: Optional[str] = None,
                          commit_hash: Optional[str] = None,
                          provider: Optional[str] = None
                          ):
        """
        Создание payload из CodeElement.

        Args:
            element: CodeElement из парсера
            repository_info: Информация о репозитории
            repository_name: Имя репозитория (переопределяет repository_info)
            branch: Ветка
            commit_hash: Хэш коммита
            provider: Провайдер репозитория

        Returns:
            CodePayload
        """
        # Базовый repo_info из repository_info
        if isinstance(repository_info, str):
            repo_info: Dict[str, Any] = {"repository_name": repository_info}
        else:
            repo_info = repository_info.copy() if isinstance(repository_info, dict) else {}

        # Приоритет именованных аргументов
        if repository_name is not None:
            repo_info["repository_name"] = repository_name
        if branch is not None:
            repo_info["branch"] = branch
        if commit_hash is not None:
            repo_info["commit_hash"] = commit_hash
        if provider is not None:
            repo_info["provider"] = provider

        return cls(
            name=element.name,
            qualified_name=element.qualified_name,
            element_type=element.type.value,
            language=element.language.value,
            source_code=element.source_code,
            signature=element.signature,
            docstring=element.docstring,
            file_path=str(element.location.file_path),
            start_line=element.location.start_line,
            end_line=element.location.end_line,
            line_count=element.location.line_count,
            repository_name=repo_info.get("repository_name") or getattr(element, "repository_name", None),
            branch=repo_info.get("branch") or getattr(element, "branch", None),
            commit_hash=repo_info.get("commit_hash") or getattr(element, "commit_hash", None),
            provider=repo_info.get("provider") or getattr(element, "provider", None),
            complexity=element.complexity,
            char_count=element.char_count,
            token_estimate=element.token_estimate,
            parent=element.parent,
            parameters=element.parameters,
            return_type=element.return_type,
            access_modifier=element.access_modifier,
            is_async=element.is_async,
            is_static=element.is_static,
            is_abstract=element.is_abstract,
            is_final=element.is_final,
            base_classes=element.base_classes,
            interfaces=element.interfaces,
            generics=element.generics,
            decorators=element.decorators,
            attributes=element.attributes,
            imports=element.imports,
            has_documentation=element.has_documentation,
            is_public=element.is_public,
            metadata=element.metadata
        )


@dataclass
class CollectionSchema:
    """
    Схема коллекции в Qdrant.

    Определяет конфигурацию векторов и индексов.
    """
    collection_name: str
    vector_size: int
    distance: VectorDistance = VectorDistance.COSINE

    # Настройки оптимизации
    on_disk: bool = False
    hnsw_config: Optional[Dict[str, Any]] = None

    # Настройки WAL (Write-Ahead Log)
    wal_capacity_mb: int = 32

    # Настройки quantization (для экономии памяти)
    use_quantization: bool = False
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Имя коллекции не может быть пустым
        if not self.collection_name:
            raise ValueError("collection_name must be non-empty")
        # Размер вектора должен быть положительным
        if self.vector_size <= 0:
            raise ValueError("vector_size must be positive")

    def to_vector_params(self) -> VectorParams:
        """Конвертация в VectorParams для Qdrant."""
        params = {
            "size": self.vector_size,
            "distance": self.distance.to_qdrant(),
        }

        if self.on_disk:
            params["on_disk"] = True

        if self.hnsw_config:
            params["hnsw_config"] = self.hnsw_config
        else:
            params["hnsw_config"] = self.get_default_hnsw_config()

        return VectorParams(**params)

    def get_default_hnsw_config(self) -> Dict[str, Any]:
        """Получение дефолтной HNSW конфигурации."""
        return {
            "m": 48,  # Количество соседей в графе
            "ef_construct": 200,  # Размер динамического списка при индексации
            "full_scan_threshold": 10000,  # Порог для полного сканирования
            "on_disk": self.on_disk,
        }

    def get_scalar_quantization_config(self) -> Dict[str, Any]:
        """Конфигурация scalar quantization."""
        return {
            "type": "scalar",
            "quantile": 0.99,
            "always_ram": True
        }

    def to_hnsw_config_params(self) -> HnswConfigDiff:
        params = self.get_default_hnsw_config()
        return qdrant_client.http.models.HnswConfigDiff(**params)

    def to_strict_config_params(self, unindexed_filtering_retrieve: bool = False) -> StrictModeConfig:
        params = {
        "enabled": True,
        "unindexed_filtering_retrieve": unindexed_filtering_retrieve
        }
        return StrictModeConfig(**params)

    def to_optimizers_config(self, indexing_threshold: int = 10000) -> OptimizersConfigDiff:
        params = {
            "indexing_threshold" : indexing_threshold,
        }
        return OptimizersConfigDiff(**params)


@dataclass
class IndexConfig:
    """
    Конфигурация индексов для payload полей.

    Индексы ускоряют фильтрацию по полям.
    """
    field_name: str
    schema_type: PayloadSchemaType

    @classmethod
    def get_default_indexes(cls) -> List['IndexConfig']:
        """
        Получение списка рекомендуемых индексов для кода.

        Returns:
            Список конфигураций индексов
        """
        return [
            # Основные поля для фильтрации
            cls(field_name="language", schema_type=PayloadSchemaType.KEYWORD),
            cls(field_name="element_type", schema_type=PayloadSchemaType.KEYWORD),
            cls(field_name="repository_name", schema_type=PayloadSchemaType.KEYWORD),
            cls(field_name="branch", schema_type=PayloadSchemaType.KEYWORD),
            cls(field_name="file_path", schema_type=PayloadSchemaType.KEYWORD),
            cls(field_name="module_name", schema_type=PayloadSchemaType.KEYWORD),

            # Модификаторы
            cls(field_name="access_modifier", schema_type=PayloadSchemaType.KEYWORD),
            cls(field_name="is_async", schema_type=PayloadSchemaType.BOOL),
            cls(field_name="is_static", schema_type=PayloadSchemaType.BOOL),
            cls(field_name="is_abstract", schema_type=PayloadSchemaType.BOOL),
            cls(field_name="is_public", schema_type=PayloadSchemaType.BOOL),
            cls(field_name="has_documentation", schema_type=PayloadSchemaType.BOOL),

            # Числовые поля для range queries
            cls(field_name="complexity", schema_type=PayloadSchemaType.INTEGER),
            cls(field_name="line_count", schema_type=PayloadSchemaType.INTEGER),
            cls(field_name="start_line", schema_type=PayloadSchemaType.INTEGER),

            # Полнотекстовый поиск
            cls(field_name="name", schema_type=PayloadSchemaType.TEXT),
            cls(field_name="qualified_name", schema_type=PayloadSchemaType.TEXT),
            cls(field_name="signature", schema_type=PayloadSchemaType.TEXT),
        ]


@dataclass
class PointData:
    """
    Данные для создания точки в Qdrant.

    Комбинирует вектор и payload.
    """
    id: str  # UUID или уникальный идентификатор
    vector: List[float]
    payload: CodePayload

    def to_point_struct(self) -> PointStruct:
        """Конвертация в PointStruct для Qdrant."""
        return PointStruct(
            id=self.id,
            vector=self.vector,
            payload=self.payload.to_dict()
        )


@dataclass
class BatchInsertData:
    """
    Данные для батч-вставки точек в Qdrant.
    """
    points: List[PointData]

    def to_point_structs(self) -> List[PointStruct]:
        """Конвертация всех точек в PointStruct."""
        return [point.to_point_struct() for point in self.points]

    def get_ids(self) -> List[str]:
        """Получение всех ID."""
        return [point.id for point in self.points]

    def get_vectors(self) -> List[List[float]]:
        """Получение всех векторов."""
        return [point.vector for point in self.points]

    def get_payloads(self) -> List[Dict[str, Any]]:
        """Получение всех payload."""
        return [point.payload.to_dict() for point in self.points]

    @property
    def size(self) -> int:
        """Количество точек в батче."""
        return len(self.points)


@dataclass
class SearchFilters:
    """
    Фильтры для поиска в Qdrant.

    Упрощает создание Filter объектов для Qdrant.
    """
    language: Optional[ProgrammingLanguage] = None
    element_type: Optional[CodeElementType] = None
    repository_name: Optional[str] = None
    branch: Optional[str] = None
    file_path: Optional[str] = None

    # Range фильтры
    min_complexity: Optional[int] = None
    max_complexity: Optional[int] = None
    min_lines: Optional[int] = None
    max_lines: Optional[int] = None

    # Boolean фильтры
    is_async: Optional[bool] = None
    is_static: Optional[bool] = None
    is_public: Optional[bool] = None
    has_documentation: Optional[bool] = None

    # Текстовые фильтры
    name_contains: Optional[str] = None
    signature: Optional[str] = None

    def to_qdrant_filter(self):
        """
        Конвертация в Qdrant Filter.

        Returns:
            Filter object для Qdrant или None
        """
        from qdrant_client.models import (Filter, FieldCondition, MatchValue, Range)

        conditions = []

        # Exact match фильтры
        if self.language:
            conditions.append(
                FieldCondition(key="language", match=MatchValue(value=self.language.value))
            )

        if self.element_type:
            conditions.append(
                FieldCondition(key="element_type", match=MatchValue(value=self.element_type.value))
            )

        if self.repository_name:
            conditions.append(
                FieldCondition(key="repository_name", match=MatchValue(value=self.repository_name))
            )

        if self.branch:
            conditions.append(
                FieldCondition(key="branch", match=MatchValue(value=self.branch))
            )

        if self.file_path:
            conditions.append(
                FieldCondition(key="file_path", match=MatchValue(value=self.file_path))
            )

        # Range фильтры
        if self.min_complexity is not None or self.max_complexity is not None:
            range_params = {}
            if self.min_complexity is not None:
                range_params["gte"] = self.min_complexity
            if self.max_complexity is not None:
                range_params["lte"] = self.max_complexity

            conditions.append(
                FieldCondition(key="complexity", range=Range(**range_params))
            )

        if self.min_lines is not None or self.max_lines is not None:
            range_params = {}
            if self.min_lines is not None:
                range_params["gte"] = self.min_lines
            if self.max_lines is not None:
                range_params["lte"] = self.max_lines

            conditions.append(
                FieldCondition(key="line_count", range=Range(**range_params))
            )

        # Boolean фильтры
        if self.is_async is not None:
            conditions.append(
                FieldCondition(key="is_async", match=MatchValue(value=self.is_async))
            )

        if self.is_static is not None:
            conditions.append(
                FieldCondition(key="is_static", match=MatchValue(value=self.is_static))
            )

        if self.is_public is not None:
            conditions.append(
                FieldCondition(key="is_public", match=MatchValue(value=self.is_public))
            )

        if self.has_documentation is not None:
            conditions.append(
                FieldCondition(key="has_documentation", match=MatchValue(value=self.has_documentation))
            )

        # Текстовый поиск (contains)
        if self.name_contains:
            conditions.append(
                FieldCondition(key="name", match=MatchValue(value=self.name_contains))
            )

        if self.signature:
            conditions.append(
                FieldCondition(key="signature", match=MatchValue(value=self.signature))
            )

        if not conditions:
            return None

        return Filter(must=conditions)



# Предопределенные схемы для разных use cases
DEFAULT_CODE_SCHEMA = CollectionSchema(
    collection_name="code_collection",
    vector_size=768,  # Для sentence-transformers
    distance=VectorDistance.COSINE,
    on_disk=False
)

LARGE_CODE_SCHEMA = CollectionSchema(
    collection_name="large_code_collection",
    vector_size=1536,  # Для OpenAI ada-002 или similar
    distance=VectorDistance.COSINE,
    on_disk=True,  # Хранить на диске для экономии RAM
    use_quantization=True
)

FAST_CODE_SCHEMA = CollectionSchema(
    collection_name="fast_code_collection",
    vector_size=384,  # Меньший размер для скорости
    distance=VectorDistance.DOT,  # Dot product быстрее
    hnsw_config={
        "m": 8,  # Меньше связей для скорости
        "ef_construct": 64
    }
)


def create_collection_schema(
        collection_name: str,
        vector_size: int,
        **kwargs
) -> CollectionSchema:
    """
    Создание кастомной схемы коллекции.

    Args:
        collection_name: Имя коллекции
        vector_size: Размер вектора
        **kwargs: Дополнительные параметры

    Returns:
        CollectionSchema
    """
    return CollectionSchema(
        collection_name=collection_name,
        vector_size=vector_size,
        **kwargs
    )
