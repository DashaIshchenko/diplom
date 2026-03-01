import pytest
from fastapi.testclient import TestClient
from code_rag.api.app import app
from code_rag.core.rag import SearchResult
from code_rag.core.parser import CodeElement, CodeLocation, ProgrammingLanguage, CodeElementType
from code_rag.api.routes.search import SearchRequest, SearchResponse, SearchResultResponse, CodeElementResponse
import json

client = TestClient(app)

def test_search_code_success(mock_embedder, mock_code_elements, mock_search_results):
    """Тест успешного поиска кода"""
    # Мокаем RAG retriever
    from unittest.mock import patch, MagicMock

    with patch('code_rag.api.dependencies.get_rag_retriever') as mock_retriever:
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.search.return_value = mock_search_results
        mock_retriever.return_value = mock_retriever_instance

        # Тест POST /search
        request_data = {
            "query": "authentication function",
            "top_k": 10,
            "language": "python",
            "element_type": "function"
        }

        response = client.post("/search/", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert "query" in data

def test_search_code_validation_error():
    """Тест валидации запроса"""
    # Пустой query
    response = client.post("/search/", json={"query": "", "top_k": 10})
    assert response.status_code == 422

    # Недопустимый top_k
    response = client.post("/search/", json={"query": "test", "top_k": 0})
    assert response.status_code == 422

def test_search_similar_success(mock_embedder):
    """Тест поиска похожего кода"""
    from unittest.mock import patch, MagicMock

    with patch('code_rag.api.dependencies.get_rag_retriever') as mock_retriever:
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.search_similar_code.return_value = []
        mock_retriever.return_value = mock_retriever_instance

        response = client.get("/search/similar/123?source_code=def test(): pass&top_k=5")
        assert response.status_code == 200

def test_semantic_search_alias():
    """Тест семантического поиска (алиас)"""
    from unittest.mock import patch, MagicMock

    with patch('code_rag.api.dependencies.get_rag_retriever') as mock_retriever:
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.search.return_value = []
        mock_retriever.return_value = mock_retriever_instance

        request_data = {
            "query": "authentication function",
            "top_k": 10
        }

        response = client.post("/search/semantic", json=request_data)
        assert response.status_code == 200

def test_search_by_file_success():
    """Тест поиска по файлу"""
    from unittest.mock import patch, MagicMock

    with patch('code_rag.api.dependencies.get_rag_retriever') as mock_retriever:
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.search_by_file.return_value = []
        mock_retriever.return_value = mock_retriever_instance

        response = client.get("/search/by-file?file_path=/test/file.py")
        assert response.status_code == 200