"""
Извлечение документации из различных языков программирования.
Поддерживает: Python docstrings, JavaDoc, JSDoc, XML Documentation (C#).
"""

from typing import Optional, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


def extract_docstring(node, source_code: str, language: str) -> Optional[str]:
    """
    Универсальная функция извлечения документации.

    Автоматически определяет тип документации по языку.

    Args:
        node: AST узел (tree-sitter или ast)
        source_code: Исходный код
        language: Язык программирования (python, java, javascript, typescript, csharp)

    Returns:
        Текст документации или None
    """
    language = language.lower()

    if language == 'python':
        return extract_python_docstring(node, source_code)
    elif language == 'java':
        return extract_javadoc(node, source_code)
    elif language in ['javascript', 'typescript']:
        return extract_jsdoc(node, source_code)
    elif language in ['csharp', 'c#']:
        return extract_xmldoc(node, source_code)

    return None


# ============================================================================
# Python Docstrings
# ============================================================================

def extract_python_docstring(node, source_code: str) -> Optional[str]:
    """
    Извлечение Python docstring.

    Поддерживает:
    - Одинарные кавычки: 'docstring'
    - Двойные кавычки: "docstring"
    - Тройные кавычки: '''docstring''' или \"\"\"docstring\"\"\"

    Args:
        node: AST узел
        source_code: Исходный код

    Returns:
        Docstring или None
    """
    try:
        # Для Python AST
        import ast
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            return ast.get_docstring(node)

        # Для tree-sitter
        if hasattr(node, 'children'):
            for child in node.children:
                if child.type == 'expression_statement':
                    for expr_child in child.children:
                        if expr_child.type == 'string':
                            docstring = source_code[expr_child.start_byte:expr_child.end_byte]
                            # Убираем кавычки
                            docstring = _clean_python_string(docstring)
                            return docstring

        return None

    except Exception as e:
        logger.debug(f"Ошибка извлечения Python docstring: {e}")
        return None


def _clean_python_string(s: str) -> str:
    """Очистка Python строки от кавычек."""
    # Убираем тройные кавычки
    s = re.sub(r'^["\']{{3}}|["\']{{3}}$', '', s)
    # Убираем одинарные/двойные кавычки
    s = re.sub(r'^["\']|["\']$', '', s)
    return s.strip()


# ============================================================================
# JavaDoc
# ============================================================================

def extract_javadoc(node, source_code: str) -> Optional[str]:
    """
    Извлечение JavaDoc комментариев.

    Формат JavaDoc:
    /**
     * Description
     * @param name description
     * @return description
     */

    Args:
        node: Tree-sitter узел
        source_code: Исходный код

    Returns:
        JavaDoc текст или None
    """
    try:
        # Ищем комментарий перед узлом
        start_line = node.start_point[0]

        # Получаем строки до узла
        lines = source_code.split('\n')

        # Ищем JavaDoc комментарий (начинается с /**)
        javadoc_lines = []
        in_javadoc = False

        for i in range(start_line - 1, -1, -1):
            line = lines[i].strip()

            if line.endswith('*/'):
                in_javadoc = True
                javadoc_lines.insert(0, line)
            elif in_javadoc:
                javadoc_lines.insert(0, line)
                if line.startswith('/**'):
                    break
            elif line and not line.startswith('//'):
                # Встретили не-комментарий
                break

        if javadoc_lines and javadoc_lines[0].startswith('/**'):
            return _clean_javadoc('\n'.join(javadoc_lines))

        return None

    except Exception as e:
        logger.debug(f"Ошибка извлечения JavaDoc: {e}")
        return None


def _clean_javadoc(javadoc: str) -> str:
    """
    Очистка JavaDoc от символов форматирования.

    Убирает:
    - /** и */
    - * в начале строк
    - Лишние пробелы
    """
    lines = javadoc.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Убираем /** и */
        line = re.sub(r'^/\*\*|^\*/|\*/$', '', line)
        # Убираем * в начале
        line = re.sub(r'^\*\s?', '', line)

        if line:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def parse_javadoc(javadoc: str) -> Dict[str, Any]:
    """
    Парсинг JavaDoc в структурированный формат.

    Извлекает:
    - description: основное описание
    - params: параметры (@param)
    - returns: возвращаемое значение (@return)
    - throws: исключения (@throws)
    - author, since, deprecated и другие теги

    Args:
        javadoc: Текст JavaDoc

    Returns:
        Словарь с распарсенными элементами
    """
    result = {
        'description': '',
        'params': {},
        'returns': None,
        'throws': {},
        'tags': {}
    }

    lines = javadoc.split('\n')
    current_section = 'description'
    description_lines = []

    for line in lines:
        line = line.strip()

        # Проверяем теги
        if line.startswith('@'):
            parts = line.split(None, 2)
            tag = parts[0][1:]  # Убираем @

            if tag == 'param' and len(parts) >= 3:
                param_name = parts[1]
                param_desc = parts[2] if len(parts) > 2 else ''
                result['params'][param_name] = param_desc

            elif tag == 'return' and len(parts) >= 2:
                result['returns'] = ' '.join(parts[1:])

            elif tag == 'throws' and len(parts) >= 3:
                exception_name = parts[1]
                exception_desc = parts[2]
                result['throws'][exception_name] = exception_desc

            else:
                # Другие теги
                tag_value = ' '.join(parts[1:]) if len(parts) > 1 else ''
                result['tags'][tag] = tag_value

        elif current_section == 'description' and line:
            description_lines.append(line)

    result['description'] = ' '.join(description_lines)

    return result


# ============================================================================
# JSDoc (JavaScript/TypeScript)
# ============================================================================

def extract_jsdoc(node, source_code: str) -> Optional[str]:
    """
    Извлечение JSDoc комментариев.

    Формат JSDoc:
    /**
     * Description
     * @param {string} name - Parameter description
     * @returns {number} Return description
     */

    Args:
        node: Tree-sitter узел
        source_code: Исходный код

    Returns:
        JSDoc текст или None
    """
    try:
        # JSDoc имеет похожий формат с JavaDoc
        # Используем ту же логику
        start_line = node.start_point[0]
        lines = source_code.split('\n')

        jsdoc_lines = []
        in_jsdoc = False

        for i in range(start_line - 1, -1, -1):
            line = lines[i].strip()

            if line.endswith('*/'):
                in_jsdoc = True
                jsdoc_lines.insert(0, line)
            elif in_jsdoc:
                jsdoc_lines.insert(0, line)
                if line.startswith('/**'):
                    break
            elif line and not line.startswith('//'):
                break

        if jsdoc_lines and jsdoc_lines[0].startswith('/**'):
            return _clean_jsdoc('\n'.join(jsdoc_lines))

        return None

    except Exception as e:
        logger.debug(f"Ошибка извлечения JSDoc: {e}")
        return None


def _clean_jsdoc(jsdoc: str) -> str:
    """Очистка JSDoc (аналогично JavaDoc)."""
    return _clean_javadoc(jsdoc)


def parse_jsdoc(jsdoc: str) -> Dict[str, Any]:
    """
    Парсинг JSDoc в структурированный формат.

    JSDoc теги:
    - @param {type} name - description
    - @returns {type} description
    - @throws {Error} description
    - @example
    - @deprecated

    Args:
        jsdoc: Текст JSDoc

    Returns:
        Словарь с распарсенными элементами
    """
    result = {
        'description': '',
        'params': {},
        'returns': None,
        'throws': {},
        'examples': [],
        'tags': {}
    }

    lines = jsdoc.split('\n')
    description_lines = []
    in_example = False
    example_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith('@'):
            # Сохраняем пример, если был
            if in_example:
                result['examples'].append('\n'.join(example_lines))
                example_lines = []
                in_example = False

            # Парсим тег
            match = re.match(r'@(\w+)\s*(.*)', line)
            if match:
                tag = match.group(1)
                content = match.group(2)

                if tag == 'param':
                    # @param {type} name - description
                    param_match = re.match(r'\{([^}]+)\}\s+(\w+)\s*-?\s*(.*)', content)
                    if param_match:
                        param_type = param_match.group(1)
                        param_name = param_match.group(2)
                        param_desc = param_match.group(3)
                        result['params'][param_name] = {
                            'type': param_type,
                            'description': param_desc
                        }

                elif tag in ['returns', 'return']:
                    # @returns {type} description
                    return_match = re.match(r'\{([^}]+)\}\s*(.*)', content)
                    if return_match:
                        result['returns'] = {
                            'type': return_match.group(1),
                            'description': return_match.group(2)
                        }

                elif tag == 'throws':
                    # @throws {Error} description
                    throws_match = re.match(r'\{([^}]+)\}\s*(.*)', content)
                    if throws_match:
                        error_type = throws_match.group(1)
                        error_desc = throws_match.group(2)
                        result['throws'][error_type] = error_desc

                elif tag == 'example':
                    in_example = True

                else:
                    result['tags'][tag] = content

        elif in_example:
            example_lines.append(line)

        elif line:
            description_lines.append(line)

    # Последний пример
    if example_lines:
        result['examples'].append('\n'.join(example_lines))

    result['description'] = ' '.join(description_lines)

    return result


# ============================================================================
# XML Documentation (C#)
# ============================================================================

def extract_xmldoc(node, source_code: str) -> Optional[str]:
    """
    Извлечение XML Documentation комментариев (C#).

    Формат XML Doc:
    /// <summary>
    /// Description
    /// </summary>
    /// <param name="x">Parameter description</param>
    /// <returns>Return description</returns>

    Args:
        node: Tree-sitter узел
        source_code: Исходный код

    Returns:
        XML Doc текст или None
    """
    try:
        start_line = node.start_point[0]
        lines = source_code.split('\n')

        xmldoc_lines = []

        # Ищем строки с /// выше узла
        for i in range(start_line - 1, -1, -1):
            line = lines[i].strip()

            if line.startswith('///'):
                xmldoc_lines.insert(0, line)
            elif line:
                # Встретили не-комментарий
                break

        if xmldoc_lines:
            return _clean_xmldoc('\n'.join(xmldoc_lines))

        return None

    except Exception as e:
        logger.debug(f"Ошибка извлечения XML Doc: {e}")
        return None


def _clean_xmldoc(xmldoc: str) -> str:
    """Очистка XML Documentation от /// символов."""
    lines = xmldoc.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Убираем ///
        line = re.sub(r'^///', '', line).strip()
        if line:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def parse_xmldoc(xmldoc: str) -> Dict[str, Any]:
    """
    Парсинг XML Documentation в структурированный формат.

    XML теги:
    - <summary>: описание
    - <param name="x">: параметр
    - <returns>: возвращаемое значение
    - <exception cref="Type">: исключение
    - <example>: пример

    Args:
        xmldoc: Текст XML Doc

    Returns:
        Словарь с распарсенными элементами
    """
    result = {
        'description': '',
        'params': {},
        'returns': None,
        'exceptions': {},
        'examples': [],
        'tags': {}
    }

    # Извлекаем summary
    summary_match = re.search(r'<summary>(.*?)</summary>', xmldoc, re.DOTALL)
    if summary_match:
        result['description'] = summary_match.group(1).strip()

    # Извлекаем параметры
    for param_match in re.finditer(r'<param name="(\w+)">(.*?)</param>', xmldoc, re.DOTALL):
        param_name = param_match.group(1)
        param_desc = param_match.group(2).strip()
        result['params'][param_name] = param_desc

    # Извлекаем returns
    returns_match = re.search(r'<returns>(.*?)</returns>', xmldoc, re.DOTALL)
    if returns_match:
        result['returns'] = returns_match.group(1).strip()

    # Извлекаем исключения
    for exception_match in re.finditer(r'<exception cref="(\w+)">(.*?)</exception>', xmldoc, re.DOTALL):
        exception_type = exception_match.group(1)
        exception_desc = exception_match.group(2).strip()
        result['exceptions'][exception_type] = exception_desc

    # Извлекаем примеры
    for example_match in re.finditer(r'<example>(.*?)</example>', xmldoc, re.DOTALL):
        result['examples'].append(example_match.group(1).strip())

    # Другие теги
    for tag_match in re.finditer(r'<(\w+)>(.*?)</\1>', xmldoc, re.DOTALL):
        tag_name = tag_match.group(1)
        if tag_name not in ['summary', 'param', 'returns', 'exception', 'example']:
            result['tags'][tag_name] = tag_match.group(2).strip()

    return result


# ============================================================================
# Вспомогательные функции
# ============================================================================

def format_docstring_for_embedding(doc: str, max_length: int = 500) -> str:
    """
    Форматирование docstring для эмбеддинга.

    Убирает лишние пробелы, переносы строк и ограничивает длину.

    Args:
        doc: Исходная документация
        max_length: Максимальная длина

    Returns:
        Отформатированная документация
    """
    if not doc:
        return ""

    # Убираем лишние пробелы
    doc = re.sub(r'\s+', ' ', doc)

    # Ограничиваем длину
    if len(doc) > max_length:
        doc = doc[:max_length] + "..."

    return doc.strip()


def extract_code_examples_from_docstring(doc: str) -> list:
    """
    Извлечение примеров кода из документации.

    Ищет блоки кода в markdown формате (```

    Args:
        doc: Текст документации

    Returns:
        Список примеров кода
    """
    examples = []

    # Ищем блоки кода в markdown формате
    for match in re.finditer(r'```\w*\n(.*?)```', doc, re.DOTALL):
        examples.append(match.group(1).strip())

    # Ищем блоки с отступами (Python docstring examples)
    for match in re.finditer(r'>>>(.*?)(?:\n\S|\Z)', doc, re.DOTALL):
        examples.append(match.group(1).strip())

    return examples


# Пример использования
if __name__ == "__main__":
    # JavaDoc пример
    javadoc = """
    /**
     * Calculates the sum of two numbers.
     * 
     * @param a the first number
     * @param b the second number
     * @return the sum of a and b
     * @throws IllegalArgumentException if numbers are negative
     */
    """

    cleaned = _clean_javadoc(javadoc)
    print("=== Cleaned JavaDoc ===")
    print(cleaned)

    parsed = parse_javadoc(cleaned)
    print("\n=== Parsed JavaDoc ===")
    print(f"Description: {parsed['description']}")
    print(f"Params: {parsed['params']}")
    print(f"Returns: {parsed['returns']}")
    print(f"Throws: {parsed['throws']}")

    # JSDoc пример
    jsdoc = """
    /**
     * Adds two numbers together.
     * @param {number} a - The first number
     * @param {number} b - The second number
     * @returns {number} The sum of a and b
     * @example
     * add(1, 2); // returns 3
     */
    """

    parsed_js = parse_jsdoc(_clean_jsdoc(jsdoc))
    print("\n=== Parsed JSDoc ===")
    print(f"Description: {parsed_js['description']}")
    print(f"Params: {parsed_js['params']}")
    print(f"Returns: {parsed_js['returns']}")
    print(f"Examples: {parsed_js['examples']}")
