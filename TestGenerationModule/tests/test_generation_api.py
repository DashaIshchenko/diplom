# tests/test_generation_api.py
import pytest
from fastapi.testclient import TestClient
from code_rag.api.app import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_generate_answer():
    """Тест генерации ответа"""
    with patch('code_rag.api.dependencies.get_qwen') as mock_qwen, \
            patch('code_rag.api.dependencies.get_rag_retriever') as mock_retriever:

        mock_qwen_instance = MagicMock()
        mock_qwen_instance.generate.return_value = MagicMock(
            content="Test response",
            model="test-model",
            tokens_used=100
        )
        mock_qwen.return_value = mock_qwen_instance

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.search.return_value = []
        mock_retriever.return_value = mock_retriever_instance

        request_data = {
            "query": "test query",
            "use_context": True,
            "top_k": 5
        }

        response = client.post("/generation/generate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "model" in data
        assert "tokens_used" in data

def test_explain_code():
    """Тест объяснения кода"""
    with patch('code_rag.api.dependencies.get_qwen') as mock_qwen:
        mock_qwen_instance = MagicMock()
        mock_qwen_instance.generate.return_value = MagicMock(
            content="Explanation of code",
            model="test-model",
            tokens_used=150
        )
        mock_qwen.return_value = mock_qwen_instance

        request_data = {
            "code": "def test(): pass",
            "language": "python",
            "detailed": True
        }

        response = client.post("/generation/explain", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "model" in data
        assert "tokens_used" in data

def test_chat():
    """Тест чата с контекстом"""
    with patch('code_rag.api.dependencies.get_qwen') as mock_qwen, \
            patch('code_rag.api.dependencies.get_rag_retriever') as mock_retriever:

        mock_qwen_instance = MagicMock()
        mock_qwen_instance.generate.return_value = MagicMock(
            content="Chat response",
            model="test-model",
            tokens_used=120
        )
        mock_qwen.return_value = mock_qwen_instance

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.search.return_value = []
        mock_retriever.return_value = mock_retriever_instance

        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "use_context": True,
            "top_k": 5
        }

        response = client.post("/generation/chat", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "model" in data
        assert "tokens_used" in data

def test_improve_code():
    """Тест улучшения кода"""
    with patch('code_rag.api.dependencies.get_qwen') as mock_qwen:
        mock_qwen_instance = MagicMock()
        mock_qwen_instance.generate.return_value = MagicMock(
            content="Improved code suggestions",
            model="test-model",
            tokens_used=200
        )
        mock_qwen.return_value = mock_qwen_instance

        request_data = {
            "code": "def test(): pass",
            "language": "python",
            "focus": "readability"
        }

        response = client.post("/generation/improve", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "model" in data
        assert "tokens_used" in data

def test_get_model_info():
    """Тест получения информации о модели"""
    with patch('code_rag.api.dependencies.get_qwen') as mock_qwen:
        mock_qwen_instance = MagicMock()
        mock_qwen_instance.get_model_info.return_value = {
            "model_name": "Qwen2.5-Coder-7B-Instruct",
            "context_length": 4096,
            "max_tokens": 2048
        }
        mock_qwen.return_value = mock_qwen_instance

        response = client.get("/generation/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data