"""
Помощник для работы с tree-sitter.
Управляет загрузкой и инициализацией парсеров для разных языков.
"""

import logging
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Проверяем доступность tree-sitter
try:
    from tree_sitter import Parser, Language

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Parser = None
    Language = None


class TreeSitterHelper:
    """
    Помощник для работы с tree-sitter парсерами.

    Использует предустановленные языковые пакеты:
    - tree-sitter-python
    - tree-sitter-javascript
    - tree-sitter-typescript
    - tree-sitter-java
    - tree-sitter-c-sharp
    """

    def __init__(self):
        """Инициализация tree-sitter helper."""
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter не установлен. Установите: pip install tree-sitter"
            )

        # Кэш парсеров и языков
        self._parsers: Dict[str, Parser] = {}
        self._languages: Dict[str, Language] = {}

        # Инициализируем языки
        self._initialize_languages()

    def _initialize_languages(self) -> None:
        """Инициализация всех поддерживаемых языков."""
        from tree_sitter import Language

        # Python
        try:
            import tree_sitter_python as ts_python
            # Оборачиваем PyCapsule в Language
            self._languages['python'] = Language(ts_python.language())
            logger.info("✓ Python parser loaded")
        except ImportError:
            logger.warning("Python parser not available. Install: pip install tree-sitter-python")
        except Exception as e:
            logger.error(f"Failed to load Python parser: {e}")

        # JavaScript
        try:
            import tree_sitter_javascript as ts_javascript
            self._languages['javascript'] = Language(ts_javascript.language())
            logger.info("✓ JavaScript parser loaded")
        except ImportError:
            logger.warning("JavaScript parser not available. Install: pip install tree-sitter-javascript")
        except Exception as e:
            logger.error(f"Failed to load JavaScript parser: {e}")

        # TypeScript
        try:
            import tree_sitter_typescript as ts_typescript
            self._languages['typescript'] = Language(ts_typescript.language_typescript())
            self._languages['tsx'] = Language(ts_typescript.language_tsx())
            logger.info("✓ TypeScript parser loaded")
        except ImportError:
            if 'javascript' in self._languages:
                self._languages['typescript'] = self._languages['javascript']
                logger.warning("TypeScript parser not found, using JavaScript as fallback")
        except Exception as e:
            logger.error(f"Failed to load TypeScript parser: {e}")

        # Java
        try:
            import tree_sitter_java as ts_java
            self._languages['java'] = Language(ts_java.language())
            logger.info("✓ Java parser loaded")
        except ImportError:
            logger.debug("Java parser not available. Install: pip install tree-sitter-java")
        except Exception as e:
            logger.error(f"Failed to load Java parser: {e}")

        # C#
        try:
            import tree_sitter_c_sharp as ts_csharp
            self._languages['c_sharp'] = Language(ts_csharp.language())
            self._languages['csharp'] = self._languages['c_sharp']  # Алиас
            logger.info("✓ C# parser loaded")
        except ImportError:
            logger.debug("C# parser not available. Install: pip install tree-sitter-c-sharp")
        except Exception as e:
            logger.error(f"Failed to load C# parser: {e}")

        # Kotlin
        try:
            import tree_sitter_kotlin as ts_kotlin
            self._languages['kotlin'] = Language(ts_kotlin.language())
            logger.info("✓ Kotlin parser loaded")
        except ImportError:
            logger.debug("Kotlin parser not available. Install: pip install tree-sitter-kotlin")
        except Exception as e:
            logger.error(f"Failed to load Kotlin parser: {e}")

        # HTML
        try:
            import tree_sitter_html as ts_html
            self._languages['html'] = Language(ts_html.language())
            logger.info("✓ HTML parser loaded")
        except ImportError:
            logger.debug("HTML parser not available. Install: pip install tree-sitter-html")
        except Exception as e:
            logger.error(f"Failed to load HTML parser: {e}")

        # CSS
        try:
            import tree_sitter_css as ts_css
            self._languages['css'] = Language(ts_css.language())
            logger.info("✓ CSS parser loaded")
        except ImportError:
            logger.debug("CSS parser not available. Install: pip install tree-sitter-css")
        except Exception as e:
            logger.error(f"Failed to load CSS parser: {e}")

        logger.info(f"Loaded {len(self._languages)} language parsers")

    def get_parser(self, language_name: str) -> Optional[Parser]:
        """
        Получение парсера для языка.

        Args:
            language_name: Имя языка

        Returns:
            Parser или None если язык не поддерживается
        """
        language_name = language_name.lower().replace('-', '_')

        # Проверяем кэш
        if language_name in self._parsers:
            return self._parsers[language_name]

        # Проверяем наличие языка
        if language_name not in self._languages:
            logger.error(f"Language {language_name} not loaded")
            return None

        try:
            # Создаем парсер с языком
            parser = Parser()
            parser.language = self._languages[language_name]

            # Кэшируем
            self._parsers[language_name] = parser

            return parser

        except Exception as e:
            logger.error(f"Error creating parser for {language_name}: {e}")
            return None

    def get_language(self, language_name: str) -> Optional[Language]:
        """
        Получение объекта Language.

        Args:
            language_name: Имя языка

        Returns:
            Language или None
        """
        language_name = language_name.lower().replace('-', '_')
        return self._languages.get(language_name)

    def is_language_available(self, language_name: str) -> bool:
        """
        Проверка доступности языка.

        Args:
            language_name: Имя языка

        Returns:
            True если язык доступен
        """
        language_name = language_name.lower().replace('-', '_')
        return language_name in self._languages

    def get_available_languages(self) -> list:
        """
        Получение списка доступных языков.

        Returns:
            Список имен языков
        """
        return list(self._languages.keys())

    def get_node_text(self, node, source_code: str) -> str:
        """
        Получение текста узла AST.

        Args:
            node: Tree-sitter узел
            source_code: Исходный код

        Returns:
            Текст узла
        """
        return source_code[node.start_byte:node.end_byte]

    def traverse_tree(self, node, callback, source_code: str = None):
        """
        Обход дерева с callback функцией.

        Args:
            node: Корневой узел
            callback: Функция callback(node, text)
            source_code: Исходный код (опционально)
        """
        text = None
        if source_code:
            text = self.get_node_text(node, source_code)

        callback(node, text)

        for child in node.children:
            self.traverse_tree(child, callback, source_code)

    def find_nodes_by_type(self, root_node, node_type: str) -> list:
        """
        Поиск всех узлов определенного типа.

        Args:
            root_node: Корневой узел
            node_type: Тип искомых узлов

        Returns:
            Список найденных узлов
        """
        nodes = []

        def collect_nodes(node, text):
            if node.type == node_type:
                nodes.append(node)

        self.traverse_tree(root_node, collect_nodes)

        return nodes

    def get_node_by_position(self, root_node, line: int, column: int):
        """
        Получение узла по позиции в коде.

        Args:
            root_node: Корневой узел
            line: Номер строки (начиная с 0)
            column: Номер столбца (начиная с 0)

        Returns:
            Узел в указанной позиции или None
        """

        def find_node_at_position(node):
            start_point = node.start_point
            end_point = node.end_point

            # Проверяем, находится ли позиция в этом узле
            if (start_point[0] <= line <= end_point[0]):
                if start_point[0] == line and start_point[1] > column:
                    return None
                if end_point[0] == line and end_point[1] < column:
                    return None

                # Ищем в дочерних узлах
                for child in node.children:
                    result = find_node_at_position(child)
                    if result:
                        return result

                return node

            return None

        return find_node_at_position(root_node)

    def __repr__(self) -> str:
        available = len(self._languages)
        return f"TreeSitterHelper(available={available} languages)"


# Глобальный экземпляр
_global_helper: Optional[TreeSitterHelper] = None


def get_tree_sitter_helper() -> TreeSitterHelper:
    """
    Получение глобального экземпляра TreeSitterHelper.

    Returns:
        TreeSitterHelper singleton
    """
    global _global_helper

    if _global_helper is None:
        _global_helper = TreeSitterHelper()

    return _global_helper


def parse_code(code: str, language: str):
    """
    Быстрый парсинг кода.

    Args:
        code: Исходный код
        language: Язык (python, java, javascript, typescript и т.д.)

    Returns:
        Tree или None
    """
    helper = get_tree_sitter_helper()
    parser = helper.get_parser(language)

    if parser:
        return parser.parse(bytes(code, 'utf-8'))

    return None


def is_tree_sitter_available() -> bool:
    """
    Проверка доступности tree-sitter.

    Returns:
        True если tree-sitter установлен
    """
    return TREE_SITTER_AVAILABLE


def get_supported_languages() -> list:
    """
    Получение списка потенциально поддерживаемых языков.

    Returns:
        Список имен языков
    """
    return [
        'python',
        'javascript',
        'typescript',
        'tsx',
        'java',
        'c_sharp',
        'csharp',
        'kotlin',
        'html',
        'css',
    ]


# Пример использования
if __name__ == "__main__":
    print("=== Tree-Sitter Helper ===")

    if not is_tree_sitter_available():
        print("❌ tree-sitter не установлен")
        print("Установите: pip install tree-sitter")
    else:
        print("✓ tree-sitter доступен")

        try:
            helper = TreeSitterHelper()
            print(f"\n{helper}")

            print("\nДоступные языки:")
            for lang in helper.get_available_languages():
                print(f"  ✓ {lang}")

            # Пример парсинга Python кода
            python_code = """
def hello_world():
    print("Hello, World!")
    return True
"""

            print("\n--- Пример парсинга Python ---")
            parser = helper.get_parser('python')
            if parser:
                tree = parser.parse(bytes(python_code, 'utf-8'))
                print(f"Root node type: {tree.root_node.type}")
                print(f"Children count: {len(tree.root_node.children)}")

                # Найдем все функции
                functions = helper.find_nodes_by_type(tree.root_node, 'function_definition')
                print(f"Found {len(functions)} function(s)")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback

            traceback.print_exc()
