"""
Интеграция с Qwen (Alibaba Cloud) для генерации ответов на основе кода.
Поддерживает Qwen2.5-Coder, Qwen3-Coder и другие модели семейства Qwen.
"""

from typing import List, Optional, Dict, Any, Generator
import logging
from dataclasses import dataclass
import json

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("openai библиотека не установлена. Установите: pip install openai")

from .rag_retriever import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class QwenResponse:
    """Ответ от Qwen модели."""
    content: str
    model: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "content": self.content,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata
        }


class QwenIntegration:
    """
    Интеграция с Qwen для RAG.

    Поддерживает:
    - Qwen3-Coder-30B (новейшая модель для кода)
    - Qwen2.5-Coder (различные размеры)
    - Qwen-Turbo, Qwen-Plus, Qwen-Max
    - Локальные модели через OpenAI-совместимый API
    - Streaming ответов
    - Контекст из retrieved code
    """

    MODELS = {
        "qwen3-coder-30b-a3b-instruct": {
            "max_tokens": 131072, "context_window": 131072,
            "description": "Qwen3 Coder 30B - Latest generation",
            "provider": "openai", "release": "2025"
        },
        "qwen2.5-coder-32b-instruct": {
            "max_tokens": 131072, "context_window": 131072,
            "description": "Qwen2.5 Coder 32B",
            "provider": "alibaba", "release": "2024"
        },
        "qwen2.5-coder:1.5b": {
            "max_tokens": 32768, "context_window": 32768,
            "description": "Qwen2.5 Coder 1.5B - Lightweight",
            "provider": "openai", "release": "2024"
        },
        "qwen2.5-coder:7b": {
            "max_tokens": 65536, "context_window": 65536,
            "description": "Qwen2.5 Coder 7B",
            "provider": "openai", "release": "2024"
        }
    }

    PROVIDERS = {
        "alibaba": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        "openai": {"base_url": "https://ai.parma.ru/cloud/v1"},
        "local": {"base_url": "http://localhost:11434/v1"}
    }

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            provider: str = "auto",
            base_url: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            top_p: float = 0.9,
            system_prompt: Optional[str] = None
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai библиотека не установлена")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        if provider == "auto":
            provider = self.MODELS.get(model, {}).get("provider", "openai")

        self.provider = provider

        if base_url:
            final_base_url = base_url
        else:
            final_base_url = self.PROVIDERS.get(provider, {}).get("base_url")

        client_params = {"api_key": api_key}
        if final_base_url:
            client_params["base_url"] = final_base_url

        self.client = OpenAI(**client_params)
        self.system_prompt = system_prompt or self._default_system_prompt()

        logger.info(f"Qwen: model={model}, provider={provider}")

    def _default_system_prompt(self) -> str:
        return """You are an expert programming assistant. Help users understand and work with code."""

    def generate(
            self,
            query: str,
            context_results: Optional[List[SearchResult]] = None,
            conversation_history: Optional[List[Dict[str, str]]] = None,
            stream: bool = False
    ) -> QwenResponse:
        try:
            messages = self._build_messages(query, context_results, conversation_history)

            if stream:
                return self._generate_stream(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )

            return self._parse_response(response)
        except Exception as e:
            logger.error(f"Error: {e}")
            return QwenResponse(content=f"Error: {e}", model=self.model)

    def generate_stream(
            self,
            query: str,
            context_results: Optional[List[SearchResult]] = None,
            conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        try:
            messages = self._build_messages(query, context_results, conversation_history)

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {e}"

    def _build_messages(
            self,
            query: str,
            context_results: Optional[List[SearchResult]] = None,
            conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        user_message = self._format_user_message(query, context_results)
        messages.append({"role": "user", "content": user_message})

        return messages

    def _format_user_message(
            self,
            query: str,
            context_results: Optional[List[SearchResult]] = None
    ) -> str:
        parts = []

        if context_results and len(context_results) > 0:
            parts.append("# Relevant Code Context\n")

            for idx, result in enumerate(context_results[:5], 1):
                element = result.element
                parts.append(f"\n## Context {idx}: {element.name}")
                parts.append(f"**Type**: {element.type.value}")
                parts.append(f"**Language**: {element.language.value}")

                if element.signature:
                    parts.append(f"**Signature**: `{element.signature}`")

                if element.docstring:
                    parts.append(f"**Documentation**:\n{element.docstring}")

                parts.append(f"\n**Code**:\n```")
                parts.append(element.source_code)
                parts.append("```\n")

                parts.append(f"\n# User Question\n\n{query}")
        return "\n".join(parts)

    def _parse_response(self, response) -> QwenResponse:
        choice = response.choices[0]

        return QwenResponse(
            content=choice.message.content,
            model=response.model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=choice.finish_reason,
            metadata={"response_id": response.id, "provider": self.provider}
        )

    def _generate_stream(self, messages: List[Dict[str, str]]) -> QwenResponse:
        full_content = ""

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        return QwenResponse(
            content=full_content,
            model=self.model,
            metadata={"streaming": True}
        )

    def explain_code(self, code_element: SearchResult, detailed: bool = False) -> QwenResponse:
        element = code_element.element
        query = f"Explain this {element.type.value}:\n\n``````"
        return self.generate(query, context_results=[code_element])

    def suggest_improvements(self, code_element: SearchResult) -> QwenResponse:
        element = code_element.element
        query = f"Suggest improvements:\n\n``````"
        return self.generate(query, context_results=[code_element])

    def get_model_info(self) -> Dict[str, Any]:
        model_info = self.MODELS.get(self.model, {})
        return {
            "model": self.model,
            "provider": self.provider,
            "max_tokens": model_info.get("max_tokens", "unknown"),
            "temperature": self.temperature
        }

    @staticmethod
    def list_available_models() -> Dict[str, Dict[str, Any]]:
        return QwenIntegration.MODELS


def create_qwen_integration(api_key: str, model: str = "qwen2.5-coder-32b-instruct", **kwargs) -> QwenIntegration:
    return QwenIntegration(api_key=api_key, model=model, **kwargs)
