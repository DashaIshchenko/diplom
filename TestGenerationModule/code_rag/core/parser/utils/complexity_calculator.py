"""
Расчет циклометрической сложности кода для разных языков.
Универсальный калькулятор, поддерживающий множество языков программирования.
"""

from typing import Optional, Dict, Any
import re
import logging

from ..code_structure import ProgrammingLanguage

logger = logging.getLogger(__name__)


def calculate_complexity(
        source_code: str,
        language: ProgrammingLanguage,
        detailed: bool = False
) -> int:
    """
    Расчет циклометрической сложности кода.

    Циклометрическая сложность (McCabe) = количество независимых путей выполнения.
    Базовая формула: 1 + количество точек ветвления

    Args:
        source_code: Исходный код функции/метода
        language: Язык программирования
        detailed: Вернуть детальную информацию

    Returns:
        Значение сложности (int) или словарь с деталями

    Examples:
        >>> code = "if x > 0: return 1\\nelse: return 0"
        >>> calculate_complexity(code, ProgrammingLanguage.PYTHON)
        2
    """
    if language == ProgrammingLanguage.PYTHON:
        return calculate_python_complexity(source_code, detailed)
    elif language == ProgrammingLanguage.JAVA:
        return calculate_java_complexity(source_code, detailed)
    elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
        return calculate_javascript_complexity(source_code, detailed)
    elif language == ProgrammingLanguage.CSHARP:
        return calculate_csharp_complexity(source_code, detailed)
    elif language == ProgrammingLanguage.KOTLIN:
        return calculate_kotlin_complexity(source_code, detailed)
    else:
        # Для неподдерживаемых языков используем упрощенный подсчет
        return calculate_generic_complexity(source_code, detailed)


# ============================================================================
# Python
# ============================================================================

def calculate_python_complexity(source_code: str, detailed: bool = False) -> int:
    """Расчет сложности для Python кода."""
    complexity = 1  # Базовая сложность
    details = {'if': 0, 'for': 0, 'while': 0, 'and': 0, 'or': 0, 'except': 0, 'with': 0}

    # Условные операторы
    details['if'] = len(re.findall(r'\bif\b', source_code))
    details['elif'] = len(re.findall(r'\belif\b', source_code))
    complexity += details['if'] + details['elif']

    # Циклы
    details['for'] = len(re.findall(r'\bfor\b', source_code))
    details['while'] = len(re.findall(r'\bwhile\b', source_code))
    complexity += details['for'] + details['while']

    # Логические операторы
    details['and'] = len(re.findall(r'\band\b', source_code))
    details['or'] = len(re.findall(r'\bor\b', source_code))
    complexity += details['and'] + details['or']

    # Обработка исключений
    details['except'] = len(re.findall(r'\bexcept\b', source_code))
    complexity += details['except']

    # Контекстные менеджеры with (каждый может вызвать исключение)
    details['with'] = len(re.findall(r'\bwith\b', source_code))
    complexity += details['with']

    # List/dict/set comprehensions (каждый добавляет путь)
    comprehensions = len(re.findall(r'\bfor\b.*\bin\b.*[\[\{\(]', source_code))
    details['comprehensions'] = comprehensions
    complexity += comprehensions

    # Lambda функции
    lambdas = len(re.findall(r'\blambda\b', source_code))
    details['lambdas'] = lambdas
    complexity += lambdas

    if detailed:
        return {
            'complexity': complexity,
            'details': details,
            'rating': get_complexity_rating(complexity)
        }

    return complexity


# ============================================================================
# Java
# ============================================================================

def calculate_java_complexity(source_code: str, detailed: bool = False) -> int:
    """Расчет сложности для Java кода."""
    complexity = 1
    details = {'if': 0, 'for': 0, 'while': 0, 'case': 0, 'catch': 0, 'ternary': 0}

    # if statements
    details['if'] = len(re.findall(r'\bif\s*\(', source_code))
    complexity += details['if']

    # Циклы
    details['for'] = len(re.findall(r'\bfor\s*\(', source_code))
    details['while'] = len(re.findall(r'\bwhile\s*\(', source_code))
    details['do_while'] = len(re.findall(r'\bdo\s*\{', source_code))
    complexity += details['for'] + details['while'] + details['do_while']

    # switch/case
    details['case'] = len(re.findall(r'\bcase\b', source_code))
    complexity += details['case']

    # Логические операторы
    details['and'] = len(re.findall(r'&&', source_code))
    details['or'] = len(re.findall(r'\|\|', source_code))
    complexity += details['and'] + details['or']

    # Exception handling
    details['catch'] = len(re.findall(r'\bcatch\s*\(', source_code))
    complexity += details['catch']

    # Тернарный оператор
    details['ternary'] = len(re.findall(r'\?.*:', source_code))
    complexity += details['ternary']

    # Stream API (forEach, filter, map и т.д.)
    stream_operations = len(re.findall(r'\.(forEach|filter|map|flatMap|anyMatch|allMatch)\s*\(', source_code))
    details['stream_operations'] = stream_operations
    complexity += stream_operations

    if detailed:
        return {
            'complexity': complexity,
            'details': details,
            'rating': get_complexity_rating(complexity)
        }

    return complexity


# ============================================================================
# JavaScript / TypeScript
# ============================================================================

def calculate_javascript_complexity(source_code: str, detailed: bool = False) -> int:
    """Расчет сложности для JavaScript/TypeScript кода."""
    complexity = 1
    details = {'if': 0, 'for': 0, 'while': 0, 'case': 0, 'catch': 0, 'ternary': 0}

    # if statements
    details['if'] = len(re.findall(r'\bif\s*\(', source_code))
    complexity += details['if']

    # Циклы
    details['for'] = len(re.findall(r'\bfor\s*\(', source_code))
    details['while'] = len(re.findall(r'\bwhile\s*\(', source_code))
    details['do_while'] = len(re.findall(r'\bdo\s*\{', source_code))
    complexity += details['for'] + details['while'] + details['do_while']

    # for...of и for...in
    details['for_of'] = len(re.findall(r'\bfor\s*\(.*\bof\b', source_code))
    details['for_in'] = len(re.findall(r'\bfor\s*\(.*\bin\b', source_code))
    complexity += details['for_of'] + details['for_in']

    # switch/case
    details['case'] = len(re.findall(r'\bcase\b', source_code))
    complexity += details['case']

    # Логические операторы
    details['and'] = len(re.findall(r'&&', source_code))
    details['or'] = len(re.findall(r'\|\|', source_code))
    details['nullish'] = len(re.findall(r'\?\?', source_code))  # Nullish coalescing
    complexity += details['and'] + details['or'] + details['nullish']

    # Exception handling
    details['catch'] = len(re.findall(r'\bcatch\s*\(', source_code))
    complexity += details['catch']

    # Тернарный оператор
    details['ternary'] = len(re.findall(r'\?[^?]*:', source_code))
    complexity += details['ternary']

    # Array methods (forEach, filter, map, reduce)
    array_methods = len(re.findall(r'\.(forEach|filter|map|reduce|some|every|find)\s*\(', source_code))
    details['array_methods'] = array_methods
    complexity += array_methods

    # Promise chains (.then, .catch)
    promise_chains = len(re.findall(r'\.(then|catch)\s*\(', source_code))
    details['promise_chains'] = promise_chains
    complexity += promise_chains

    if detailed:
        return {
            'complexity': complexity,
            'details': details,
            'rating': get_complexity_rating(complexity)
        }

    return complexity


# ============================================================================
# C#
# ============================================================================

def calculate_csharp_complexity(source_code: str, detailed: bool = False) -> int:
    """Расчет сложности для C# кода."""
    complexity = 1
    details = {'if': 0, 'for': 0, 'while': 0, 'case': 0, 'catch': 0}

    # if statements
    details['if'] = len(re.findall(r'\bif\s*\(', source_code))
    complexity += details['if']

    # Циклы
    details['for'] = len(re.findall(r'\bfor\s*\(', source_code))
    details['foreach'] = len(re.findall(r'\bforeach\s*\(', source_code))
    details['while'] = len(re.findall(r'\bwhile\s*\(', source_code))
    details['do_while'] = len(re.findall(r'\bdo\s*\{', source_code))
    complexity += details['for'] + details['foreach'] + details['while'] + details['do_while']

    # switch/case
    details['case'] = len(re.findall(r'\bcase\b', source_code))
    complexity += details['case']

    # Логические операторы
    details['and'] = len(re.findall(r'&&', source_code))
    details['or'] = len(re.findall(r'\|\|', source_code))
    complexity += details['and'] + details['or']

    # Exception handling
    details['catch'] = len(re.findall(r'\bcatch\s*\(', source_code))
    complexity += details['catch']

    # Тернарный оператор
    details['ternary'] = len(re.findall(r'\?[^?]*:', source_code))
    complexity += details['ternary']

    # Null coalescing operator (??)
    details['null_coalescing'] = len(re.findall(r'\?\?', source_code))
    complexity += details['null_coalescing']

    # LINQ queries (where, select, any, all)
    linq_operations = len(re.findall(r'\.(Where|Select|Any|All|First|Single)\s*\(', source_code))
    details['linq_operations'] = linq_operations
    complexity += linq_operations

    # Pattern matching (switch expressions in C# 8+)
    switch_expressions = len(re.findall(r'=>', source_code))
    details['switch_expressions'] = switch_expressions
    complexity += switch_expressions

    if detailed:
        return {
            'complexity': complexity,
            'details': details,
            'rating': get_complexity_rating(complexity)
        }

    return complexity


# ============================================================================
# Kotlin
# ============================================================================

def calculate_kotlin_complexity(source_code: str, detailed: bool = False) -> int:
    """Расчет сложности для Kotlin кода."""
    complexity = 1
    details = {'if': 0, 'for': 0, 'while': 0, 'when': 0, 'catch': 0}

    # if statements (if может быть выражением в Kotlin)
    details['if'] = len(re.findall(r'\bif\s*\(', source_code))
    complexity += details['if']

    # Циклы
    details['for'] = len(re.findall(r'\bfor\s*\(', source_code))
    details['while'] = len(re.findall(r'\bwhile\s*\(', source_code))
    details['do_while'] = len(re.findall(r'\bdo\s*\{', source_code))
    complexity += details['for'] + details['while'] + details['do_while']

    # when (замена switch в Kotlin)
    details['when'] = len(re.findall(r'\bwhen\s*[\(\{]', source_code))
    # when branches
    when_branches = len(re.findall(r'->', source_code))
    details['when_branches'] = when_branches
    complexity += details['when'] + when_branches

    # Логические операторы
    details['and'] = len(re.findall(r'&&', source_code))
    details['or'] = len(re.findall(r'\|\|', source_code))
    complexity += details['and'] + details['or']

    # Exception handling
    details['catch'] = len(re.findall(r'\bcatch\s*\(', source_code))
    complexity += details['catch']

    # Elvis operator (?:)
    details['elvis'] = len(re.findall(r'\?:', source_code))
    complexity += details['elvis']

    # Collection operations (filter, map, forEach)
    collection_ops = len(re.findall(r'\.(filter|map|forEach|any|all|find)\s*\{', source_code))
    details['collection_ops'] = collection_ops
    complexity += collection_ops

    # Safe calls (?.)
    safe_calls = len(re.findall(r'\?\.', source_code))
    details['safe_calls'] = safe_calls
    # Не добавляем к сложности, так как это не ветвление

    if detailed:
        return {
            'complexity': complexity,
            'details': details,
            'rating': get_complexity_rating(complexity)
        }

    return complexity


# ============================================================================
# Generic (для неподдерживаемых языков)
# ============================================================================

def calculate_generic_complexity(source_code: str, detailed: bool = False) -> int:
    """
    Упрощенный расчет сложности для любого языка.
    Использует общие паттерны.
    """
    complexity = 1
    details = {}

    # Условные операторы (if, elif, else if)
    if_count = len(re.findall(r'\bif\b', source_code, re.IGNORECASE))
    details['if'] = if_count
    complexity += if_count

    # Циклы (for, while, do)
    for_count = len(re.findall(r'\bfor\b', source_code, re.IGNORECASE))
    while_count = len(re.findall(r'\bwhile\b', source_code, re.IGNORECASE))
    details['loops'] = for_count + while_count
    complexity += for_count + while_count

    # Case/when statements
    case_count = len(re.findall(r'\bcase\b', source_code, re.IGNORECASE))
    details['case'] = case_count
    complexity += case_count

    # Exception handling
    catch_count = len(re.findall(r'\bcatch\b', source_code, re.IGNORECASE))
    except_count = len(re.findall(r'\bexcept\b', source_code, re.IGNORECASE))
    details['exceptions'] = catch_count + except_count
    complexity += catch_count + except_count

    # Логические операторы
    and_count = len(re.findall(r'&&|\band\b', source_code, re.IGNORECASE))
    or_count = len(re.findall(r'\|\||\bor\b', source_code, re.IGNORECASE))
    details['logical'] = and_count + or_count
    complexity += and_count + or_count

    if detailed:
        return {
            'complexity': complexity,
            'details': details,
            'rating': _get_complexity_rating(complexity),
            'note': 'Generic complexity calculation (language-agnostic)'
        }

    return complexity


# ============================================================================
# Вспомогательные функции
# ============================================================================

def get_complexity_rating(complexity: int) -> str:
    """
    Получение рейтинга сложности кода.

    Рейтинги:
    - A: 1-5 (простой)
    - B: 6-10 (средний)
    - C: 11-20 (сложный)
    - D: 21-40 (очень сложный)
    - F: 41+ (критически сложный, требует рефакторинга)

    Args:
        complexity: Значение циклометрической сложности

    Returns:
        Строка с рейтингом и описанием

    Examples:
        >>> get_complexity_rating(3)
        'A - Simple'

        >>> get_complexity_rating(15)
        'C - Complex'

        >>> get_complexity_rating(45)
        'F - Critically Complex (needs refactoring)'
    """
    if complexity <= 5:
        return 'A - Simple'
    elif complexity <= 10:
        return 'B - Medium'
    elif complexity <= 20:
        return 'C - Complex'
    elif complexity <= 40:
        return 'D - Very Complex'
    else:
        return 'F - Critically Complex (needs refactoring)'


def get_complexity_recommendation(complexity: int) -> str:
    """
    Получение рекомендации по упрощению кода.

    Args:
        complexity: Значение сложности

    Returns:
        Текст рекомендации
    """
    if complexity <= 5:
        return "Код простой и легко поддерживается."
    elif complexity <= 10:
        return "Код имеет приемлемую сложность."
    elif complexity <= 20:
        return "Рассмотрите возможность разбиения функции на более мелкие части."
    elif complexity <= 40:
        return "Функция слишком сложная. Рекомендуется рефакторинг."
    else:
        return "КРИТИЧЕСКАЯ СЛОЖНОСТЬ! Необходим срочный рефакторинг. Разделите на несколько функций."


def calculate_maintainability_index(
        complexity: int,
        lines_of_code: int,
        halstead_volume: Optional[float] = None
) -> float:
    """
    Расчет индекса поддерживаемости (Maintainability Index).

    MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)

    Где:
    - V: Halstead Volume (опционально)
    - G: Cyclomatic Complexity
    - LOC: Lines of Code

    Args:
        complexity: Циклометрическая сложность
        lines_of_code: Количество строк кода
        halstead_volume: Halstead Volume (опционально)

    Returns:
        Индекс поддерживаемости (0-100)
    """
    import math

    if halstead_volume is None:
        # Упрощенная формула без Halstead Volume
        if lines_of_code == 0:
            return 100.0

        mi = 171 - 0.23 * complexity - 16.2 * math.log(lines_of_code)
    else:
        mi = 171 - 5.2 * math.log(halstead_volume) - 0.23 * complexity - 16.2 * math.log(lines_of_code)

    # Нормализуем к диапазону 0-100
    mi = max(0, min(100, mi))

    return round(mi, 2)


# Пример использования
if __name__ == "__main__":
    # Python пример
    python_code = """
    def process_data(data):
        if data is None:
            return None

        result = []
        for item in data:
            if item > 0:
                result.append(item * 2)
            elif item < 0:
                result.append(abs(item))

        return result if result else None
    """

    print("=== Python Code Complexity ===")
    complexity = calculate_complexity(python_code, ProgrammingLanguage.PYTHON, detailed=True)
    print(f"Complexity: {complexity['complexity']}")
    print(f"Rating: {complexity['rating']}")
    print(f"Details: {complexity['details']}")
    print(f"Recommendation: {get_complexity_recommendation(complexity['complexity'])}")

    # Java пример
    java_code = """
    public int calculate(int x) {
        if (x > 0) {
            for (int i = 0; i < x; i++) {
                if (i % 2 == 0) {
                    return i;
                }
            }
        } else if (x < 0) {
            return -1;
        }
        return 0;
    }
    """

    print("\n=== Java Code Complexity ===")
    java_complexity = calculate_complexity(java_code, ProgrammingLanguage.JAVA, detailed=True)
    print(f"Complexity: {java_complexity['complexity']}")
    print(f"Rating: {java_complexity['rating']}")

    # Maintainability Index
    mi = calculate_maintainability_index(
        complexity=java_complexity['complexity'],
        lines_of_code=len(java_code.split('\n'))
    )
    print(f"Maintainability Index: {mi}/100")
