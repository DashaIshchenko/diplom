# RAG Strategies for Code

## Введение

RAG (Retrieval-Augmented Generation) для кода имеет свои особенности и требует специальных стратегий для достижения высокого качества ответов. В этом документе описаны различные стратегии и техники оптимизации RAG для работы с кодовой базой.

## Базовый RAG Pipeline

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Query Embedding     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Vector Search       │
│ (Top-K Results)     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Context Selection   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Prompt Construction │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ LLM Generation      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Response            │
└─────────────────────┘
```

## 1. Стратегии поиска

### 1.1 Semantic Search

Базовый семантический поиск через векторные эмбеддинги.

```
def semantic_search(query: str, top_k: int = 5) -> List[SearchResult]:
    """Семантический поиск по векторным эмбеддингам."""
    query_vector = embedder.encode_query(query)
    
    results = qdrant_client.search(
        collection_name="code_collection",
        query_vector=query_vector,
        limit=top_k
    )
    
    return results
```

**Преимущества:**
- Находит семантически похожий код
- Не зависит от точного совпадения слов
- Работает с разными формулировками

**Недостатки:**
- Может пропустить точные matches
- Зависит от качества embeddings

### 1.2 Hybrid Search

Комбинация семантического и keyword-based поиска.

```
def hybrid_search(
    query: str, 
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    top_k: int = 10
) -> List[SearchResult]:
    """Гибридный поиск."""
    semantic_results = semantic_search(query, top_k=top_k)
    keyword_results = keyword_search(query, top_k=top_k)
    
    combined = combine_results(
        semantic_results, 
        keyword_results,
        semantic_weight,
        keyword_weight
    )
    
    return combined[:top_k]
```

### 1.3 Multi-Stage Retrieval

```
def multi_stage_retrieval(query: str) -> List[SearchResult]:
    """Многоэтапный retrieval."""
    # Stage 1: Широкий поиск
    candidates = semantic_search(query, top_k=100)
    
    # Stage 2: Reranking
    reranked = rerank_with_cross_encoder(query, candidates, top_k=20)
    
    # Stage 3: Фильтрация
    filtered = context_filter(reranked, top_k=10)
    
    # Stage 4: Diversity
    diverse = select_diverse_results(filtered, top_k=5)
    
    return diverse
```

## 2. Контекст и Prompt Engineering

### 2.1 Context Window Management

Управление размером контекстного окна.

```
class ContextWindowManager:
    """Управление контекстным окном."""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
    
    def build_context(
        self, 
        results: List[SearchResult],
        strategy: str = "top_score"
    ) -> str:
        """Построение контекста."""
        if strategy == "top_score":
            return self._build_by_score(results)
        elif strategy == "diversity":
            return self._build_diverse(results)
    
    def _build_by_score(self, results: List[SearchResult]) -> str:
        """Выбор по score до заполнения окна."""
        context_parts = []
        total_tokens = 0
        
        for result in sorted(results, key=lambda x: x.score, reverse=True):
            code = result.element.source_code
            tokens = estimate_tokens(code)
            
            if total_tokens + tokens > self.max_tokens:
                break
            
            context_parts.append(self._format_code(result))
            total_tokens += tokens
        
        return "\n\n".join(context_parts)
```

### 2.2 Prompt Templates

**Базовый Q&A:**
```
BASE_QA_TEMPLATE = """You are an expert code assistant.

Context:
{context}

Question: {question}

Instructions:
- Provide clear, concise answer
- Reference specific code
- Include examples if helpful

Answer:"""
```

**Code Explanation:**
```
EXPLAIN_TEMPLATE = """Explain this code in detail.

Code:
{language}
{code}


Provide:
1. What this code does
2. Key components
3. Patterns used
4. Potential improvements

Explanation:"""
```

**Code Generation:**
```
GENERATION_TEMPLATE = """Generate code based on request.

Request: {request}

Similar Examples:
{examples}

Instructions:
- Follow example patterns
- Include error handling
- Add comments

Generated Code:"""
```

**Code Review:**
```
REVIEW_TEMPLATE = """Review this code.

Code:
{language}
{code}


Review for:
1. Correctness
2. Performance
3. Security
4. Best practices

Context: {context}

Review:"""
```

### 2.3 Chain of Thought

```
def generate_with_cot(question: str, context: List[SearchResult]) -> str:
    """Генерация с Chain of Thought."""
    
    # Шаг 1: Анализ
    analysis_prompt = f"""Analyze this question:

Question: {question}

Break down:
1. What is being asked?
2. What information needed?
3. What context relevant?

Analysis:"""
    
    analysis = llm.generate(analysis_prompt)
    
    # Шаг 2: Ответ
    answer_prompt = f"""Based on analysis and context:

Analysis: {analysis}
Context: {format_context(context)}
Question: {question}

Think step by step:
1. Identify relevant code
2. Understand implementation
3. Formulate answer

Answer:"""
    
    return llm.generate(answer_prompt)
```

## 3. Reranking Strategies

### 3.1 Score-based Reranking

```
def rerank_results(
    query: str,
    results: List[SearchResult],
    weights: Dict[str, float] = None
) -> List[SearchResult]:
    """Переранжирование результатов."""
    
    if weights is None:
        weights = {
            "semantic_score": 0.4,
            "recency": 0.2,
            "complexity": 0.1,
            "popularity": 0.1,
            "documentation": 0.2
        }
    
    for result in results:
        scores = {
            "semantic_score": result.score,
            "recency": calculate_recency_score(result),
            "complexity": calculate_complexity_score(result),
            "popularity": calculate_popularity_score(result),
            "documentation": calculate_doc_score(result)
        }
        
        result.final_score = sum(
            scores[k] * weights[k] for k in weights.keys()
        )
    
    return sorted(results, key=lambda x: x.final_score, reverse=True)
```

### 3.2 Cross-Encoder Reranking

```
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Reranker с cross-encoder."""
    
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def rerank(
        self, 
        query: str, 
        results: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        """Переранжирование."""
        pairs = [(query, r.element.source_code) for r in results]
        scores = self.model.predict(pairs)
        
        for result, score in zip(results, scores):
            result.rerank_score = score
        
        return sorted(
            results, 
            key=lambda x: x.rerank_score, 
            reverse=True
        )[:top_k]
```

## 4. Context Enhancement

### 4.1 Hierarchical Context

Включение иерархического контекста.

```
def build_hierarchical_context(result: SearchResult) -> str:
    """Построение иерархического контекста."""
    element = result.element
    parts = []
    
    # Родительский класс
    if element.parent_class:
        parts.append(f"Parent Class: {element.parent_class}")
    
    # Модуль
    parts.append(f"Module: {element.location.file_path}")
    
    # Imports
    if hasattr(element, 'imports'):
        parts.append(f"Imports: {', '.join(element.imports)}")
    
    # Сам элемент
    parts.append(f"""
{element.type.value}: {element.qualified_name}
{element.language.value}
{element.source_code}

    
    # Связанные вызовы
    if hasattr(element, 'calls'):
        parts.append(f"Calls: {', '.join(element.calls)}")
    
    return "\n\n".join(parts)
```

### 4.2 Dependency Graph Context

```
def add_dependency_context(
    result: SearchResult,
    max_depth: int = 2
) -> str:
    """Добавление контекста зависимостей."""
    element = result.element
    
    dependencies = find_dependencies(element, max_depth=max_depth)
    
    context_parts = [result.element.source_code]
    
    for dep in dependencies:
        context_parts.append(f"""
// Dependency: {dep.qualified_name}
{dep.source_code}
""")
    
    return "\n\n".join(context_parts)
```

### 4.3 Documentation Enrichment

```
def enrich_with_documentation(result: SearchResult) -> str:
    """Обогащение документацией."""
    element = result.element
    parts = []
    
    if element.docstring:
        parts.append(f"Documentation:\n{element.docstring}")
    
    if element.signature:
        parts.append(f"Signature: {element.signature}")
    
    parts.append(f"Implementation:\n```{element.language.value}\n{element.source_code}\n```
    
    examples = find_usage_examples(element)
    if examples:
        parts.append(f"Usage Examples:\n{examples}")
    
    return "\n\n".join(parts)
```

## 5. Query Enhancement

### 5.1 Query Expansion

```
def expand_query(query: str) -> List[str]:
    """Расширение запроса."""
    expanded = [query]
    
    # Синонимы
    synonyms = get_synonyms(query)
    expanded.extend(synonyms)
    
    # Связанные термины
    related = get_related_terms(query)
    expanded.extend(related)
    
    # Технические вариации
    if "function" in query:
        expanded.append(query.replace("function", "method"))
    if "auth" in query:
        expanded.extend(["authentication", "authorization"])
    
    return list(set(expanded))
```

### 5.2 Query Decomposition

```
def decompose_query(complex_query: str) -> List[str]:
    """Разложение сложного запроса на подзапросы."""
    
    prompt = f"""Decompose this complex query into simpler sub-queries:

Query: {complex_query}

Sub-queries:
1."""
    
    response = llm.generate(prompt)
    sub_queries = parse_sub_queries(response)
    
    return sub_queries

# Пример использования
query = "How is user authentication and authorization implemented?"
sub_queries = decompose_query(query)
# ["How is user authentication implemented?",
#  "How is authorization handled?",
#  "What are the security measures?"]

# Поиск по каждому подзапросу
all_results = []
for sub_query in sub_queries:
    results = semantic_search(sub_query, top_k=5)
    all_results.extend(results)

# Объединение и deduplicate
final_results = deduplicate_results(all_results)
```

### 5.3 Query Reformulation

```
def reformulate_query(query: str, context: str = "code") -> str:
    """Переформулирование запроса для лучшего поиска."""
    
    prompt = f"""Reformulate this query for better code search:

Original: {query}
Context: {context}

Reformulated (technical, specific):"""
    
    reformulated = llm.generate(prompt, max_tokens=100)
    
    return reformulated.strip()

# Пример
original = "how to login"
reformulated = reformulate_query(original)
# "authentication login function implementation"
```

## 6. Self-Reflection and Verification

### 6.1 Answer Verification

```
def verify_answer(question: str, answer: str, context: List[SearchResult]) -> bool:
    """Проверка корректности ответа."""
    
    prompt = f"""Verify if this answer is correct based on context.

Question: {question}

Answer: {answer}

Context: {format_context(context)}

Is the answer:
1. Accurate? (yes/no)
2. Complete? (yes/no)
3. Based on context? (yes/no)

Verification:"""
    
    verification = llm.generate(prompt)
    
    return parse_verification(verification)
```

### 6.2 Self-Reflection

```
def self_reflect_and_improve(
    question: str,
    initial_answer: str,
    context: List[SearchResult]
) -> str:
    """Самопроверка и улучшение ответа."""
    
    # Рефлексия
    reflection_prompt = f"""Review this answer:

Question: {question}
Answer: {initial_answer}

Issues to check:
1. Accuracy
2. Completeness
3. Clarity
4. Code quality

Critique:"""
    
    critique = llm.generate(reflection_prompt)
    
    # Если есть проблемы - улучшаем
    if has_issues(critique):
        improvement_prompt = f"""Improve the answer:

Original: {initial_answer}
Critique: {critique}
Context: {format_context(context)}

Improved Answer:"""
        
        improved = llm.generate(improvement_prompt)
        return improved
    
    return initial_answer
```

### 6.3 Confidence Scoring

```
def calculate_confidence(
    question: str,
    answer: str,
    context: List[SearchResult]
) -> float:
    """Вычисление уверенности в ответе."""
    
    factors = {
        "context_relevance": calculate_context_relevance(question, context),
        "answer_specificity": calculate_specificity(answer),
        "code_coverage": calculate_code_coverage(answer, context),
        "consistency": check_consistency(answer, context)
    }
    
    weights = {
        "context_relevance": 0.4,
        "answer_specificity": 0.2,
        "code_coverage": 0.3,
        "consistency": 0.1
    }
    
    confidence = sum(
        factors[k] * weights[k] for k in factors.keys()
    )
    
    return confidence
```

## 7. Advanced Techniques

### 7.1 Iterative Refinement

```
def iterative_refinement(
    question: str,
    max_iterations: int = 3
) -> str:
    """Итеративное улучшение ответа."""
    
    answer = None
    context = []
    
    for iteration in range(max_iterations):
        # Поиск контекста
        if iteration == 0:
            context = semantic_search(question, top_k=5)
        else:
            # Расширенный поиск на основе предыдущего ответа
            expanded_query = extract_key_terms(answer)
            context = semantic_search(expanded_query, top_k=5)
        
        # Генерация ответа
        answer = generate_answer(question, context)
        
        # Проверка качества
        confidence = calculate_confidence(question, answer, context)
        
        if confidence > 0.85:
            break
    
    return answer
```

### 7.2 Multi-Modal Context

```
def build_multimodal_context(result: SearchResult) -> Dict[str, Any]:
    """Мультимодальный контекст."""
    
    return {
        "code": result.element.source_code,
        "documentation": result.element.docstring,
        "signature": result.element.signature,
        "type_hints": extract_type_hints(result.element),
        "comments": extract_comments(result.element),
        "test_cases": find_test_cases(result.element),
        "usage_patterns": find_usage_patterns(result.element),
        "dependencies": get_dependencies(result.element)
    }
```

### 7.3 Feedback Loop

```
class RAGWithFeedback:
    """RAG с обратной связью."""
    
    def __init__(self):
        self.feedback_store = []
    
    def generate_with_feedback(
        self,
        question: str,
        user_feedback: Optional[str] = None
    ) -> str:
        """Генерация с учетом feedback."""
        
        # Получаем контекст
        context = self.search_with_feedback(question)
        
        # Генерируем ответ
        answer = generate_answer(question, context)
        
        # Если есть feedback - улучшаем
        if user_feedback:
            answer = self.improve_with_feedback(
                question, 
                answer, 
                user_feedback
            )
        
        return answer
    
    def improve_with_feedback(
        self,
        question: str,
        answer: str,
        feedback: str
    ) -> str:
        """Улучшение на основе feedback."""
        
        prompt = f"""Improve answer based on user feedback:

Question: {question}
Original Answer: {answer}
User Feedback: {feedback}

Improved Answer:"""
        
        improved = llm.generate(prompt)
        
        # Сохраняем feedback для обучения
        self.feedback_store.append({
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "improved": improved
        })
        
        return improved
```

## 8. Best Practices

### Выбор стратегии

| Scenario | Strategy | Reasoning |
|----------|----------|-----------|
| Точный поиск API | Hybrid Search | Нужны keyword matches |
| Концептуальный поиск | Semantic Search | Понимание контекста |
| Сложные запросы | Multi-Stage + CoT | Точность важнее скорости |
| Быстрый поиск | Semantic only | Скорость важнее |
| Code generation | With examples + CoT | Нужен паттерн |
| Code review | Hierarchical + Deps | Полный контекст |

### Оптимизация производительности

```
# 1. Кэширование embeddings
@lru_cache(maxsize=10000)
def get_cached_embedding(text: str) -> np.ndarray:
    return embedder.encode_text(text)

# 2. Батчевая обработка
results = []
for batch in chunk_list(queries, batch_size=32):
    batch_results = search_batch(batch)
    results.extend(batch_results)

# 3. Параллельный поиск
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(semantic_search, query) 
        for query in sub_queries
    ]
    results = [f.result() for f in futures]

# 4. Ограничение контекста
context = limit_context_tokens(results, max_tokens=6000)
```

### Метрики качества

```
def evaluate_rag_quality(
    questions: List[str],
    ground_truth: List[str],
    system_answers: List[str]
) -> Dict[str, float]:
    """Оценка качества RAG."""
    
    metrics = {
        "exact_match": calculate_exact_match(ground_truth, system_answers),
        "f1_score": calculate_f1(ground_truth, system_answers),
        "bleu": calculate_bleu(ground_truth, system_answers),
        "rouge": calculate_rouge(ground_truth, system_answers),
        "code_execution": test_code_execution(system_answers)
    }
    
    return metrics
```

## 9. Common Pitfalls

### Проблема 1: Context Overflow

**Симптом:** Превышение токен лимита

**Решение:**
```
def smart_context_selection(results: List[SearchResult], max_tokens: int):
    """Умный выбор контекста."""
    # Приоритизация
    scored = score_by_relevance_and_size(results)
    
    # Жадный выбор до лимита
    selected = []
    total_tokens = 0
    
    for result in scored:
        tokens = estimate_tokens(result.element.source_code)
        if total_tokens + tokens <= max_tokens:
            selected.append(result)
            total_tokens += tokens
    
    return selected
```

### Проблема 2: Irrelevant Context

**Симптом:** LLM галлюцинирует или игнорирует контекст

**Решение:**
```
# Установить минимальный threshold
results = [r for r in results if r.score > 0.7]

# Добавить инструкцию в prompt
prompt = f"""IMPORTANT: Only use information from the provided context.
If context doesn't contain answer, say "I don't have enough information."

Context: {context}
Question: {question}
"""
```

### Проблема 3: Outdated Code

**Симптом:** Рекомендации устаревшего кода

**Решение:**
```
# Фильтрация по времени
recent_results = filter_by_recency(results, days=90)

# Приоритет недавнему коду
results = sorted(results, key=lambda x: (x.score, x.indexed_at), reverse=True)
```

## 10. Example Implementation

Полный пример продвинутого RAG:

```
class AdvancedCodeRAG:
    """Продвинутая RAG система."""
    
    def __init__(self, config: RAGConfig):
        self.embedder = CodeEmbedder()
        self.retriever = RAGRetriever(...)
        self.llm = QwenIntegration(...)
        self.config = config
    
    def answer_question(self, question: str) -> RAGResponse:
        """Главный метод."""
        
        # 1. Query enhancement
        enhanced_query = self.enhance_query(question)
        
        # 2. Hybrid search
        results = self.hybrid_search(enhanced_query)
        
        # 3. Reranking
        results = self.rerank(question, results)
        
        # 4. Context building
        context = self.build_context(results)
        
        # 5. Generation with CoT
        answer = self.generate_with_cot(question, context)
        
        # 6. Self-reflection
        if self.config.use_reflection:
            answer = self.self_reflect(question, answer, context)
        
        # 7. Confidence
        confidence = self.calculate_confidence(question, answer, results)
        
        return RAGResponse(
            answer=answer,
            confidence=confidence,
            sources=results,
            strategy_used="advanced"
        )
```

## Заключение

Выбор стратегии RAG зависит от:
- **Типа запроса:** простой vs сложный
- **Требований к качеству:** точность vs скорость
- **Доступных ресурсов:** CPU/GPU, memory
- **Размера кодовой базы:** маленькая vs большая

Рекомендуется начинать с базового semantic search, затем добавлять более сложные техники по мере необходимости.

## Дополнительные ресурсы

- [RAG Papers](https://github.com/RUC-NLPIR/FlashRAG)
- [Advanced RAG Techniques](https://arxiv.org/abs/2312.10997)
- [Code Understanding with LLMs](https://arxiv.org/abs/2308.12950)

