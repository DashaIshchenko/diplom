"""API роуты для генерации."""
from typing import Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from ...core.rag import QwenIntegration, RAGRetriever
from ..dependencies import get_qwen, get_rag_retriever

router = APIRouter(prefix="/generation", tags=["generation"])


class GenerateRequest(BaseModel):
    query: str
    use_context: bool = True
    top_k: int = 5


class GenerationResponse(BaseModel):
    content: str
    model: str
    tokens_used: int


@router.post("/generate", response_model=GenerationResponse)
async def generate_answer(
        request: GenerateRequest,
        qwen: QwenIntegration = Depends(get_qwen),
        retriever: RAGRetriever = Depends(get_rag_retriever)
):
    context_results = []
    if request.use_context:
        context_results = retriever.search(query=request.query, top_k=request.top_k)

    response = qwen.generate(
        query=request.query,
        context_results=context_results
    )

    return GenerationResponse(
        content=response.content,
        model=response.model,
        tokens_used=response.tokens_used
    )


@router.post("/explain", response_model=GenerationResponse)
async def explain_code(
        code: str = Query(..., description="Код для объяснения"),
        language: str = Query(..., description="Язык программирования"),
        detailed: bool = Query(False, description="Детальное объяснение"),
        qwen: QwenIntegration = Depends(get_qwen)
):
    """Объяснение кода."""
    query = f"Explain this {language} code:\n\n{code}"

    if detailed:
        query += "\n\nProvide detailed line-by-line explanation."

    response = qwen.generate(query)

    return GenerationResponse(
        content=response.content,
        model=response.model,
        tokens_used=response.tokens_used
    )


from typing import List


class ConversationMessage(BaseModel):
    role: str = Field(..., description="user или assistant")
    content: str


class ChatRequest(BaseModel):
    messages: List[ConversationMessage]
    use_context: bool = True
    top_k: int = 5


@router.post("/chat", response_model=GenerationResponse)
async def chat(
        request: ChatRequest,
        qwen: QwenIntegration = Depends(get_qwen),
        retriever: RAGRetriever = Depends(get_rag_retriever)
):
    """Чат с контекстом кода."""
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages[:-1]
    ]

    current_query = request.messages[-1].content

    context_results = []
    if request.use_context:
        context_results = retriever.search(query=current_query, top_k=request.top_k)

    response = qwen.generate(
        query=current_query,
        context_results=context_results,
        conversation_history=conversation_history
    )

    return GenerationResponse(
        content=response.content,
        model=response.model,
        tokens_used=response.tokens_used
    )


class ImproveCodeRequest(BaseModel):
    code: str = Field(..., description="Код для улучшения")
    language: str = Field(..., description="Язык программирования")
    focus: Optional[str] = Field(None, description="performance, readability, security")


@router.post("/improve", response_model=GenerationResponse)
async def improve_code(
        request: ImproveCodeRequest,
        qwen: QwenIntegration = Depends(get_qwen)
):
    """Улучшение кода."""
    focus_text = f"Focus on {request.focus}" if request.focus else "General improvements"

    query = f"Suggest improvements for this {request.language} code. {focus_text}:\n\n``````"

    response = qwen.generate(query)

    return GenerationResponse(
        content=response.content,
        model=response.model,
        tokens_used=response.tokens_used
    )


@router.get("/model/info")
async def get_model_info(qwen: QwenIntegration = Depends(get_qwen)):
    return qwen.get_model_info()
