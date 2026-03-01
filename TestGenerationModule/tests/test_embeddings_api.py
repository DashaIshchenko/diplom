# tests/test_embeddings_api.py
import pytest
from fastapi.testclient import TestClient
from code_rag.api.app import app
import numpy as np
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_embed_text():
    """Тест векторизации текста"""
    with patch('code_rag.api.dependencies.get_embedder') as mock_embedder:
        # Мокаем embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode_text.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_embedder_instance.embedding_dim = 3
        mock_embedder_instance.model.__class__.__name__ = "TestEmbedder"
        mock_embedder.return_value = mock_embedder_instance

        request_data = {
            "text": "def test_function(): pass",
            "normalize": True
        }

        response = client.post("/embeddings/embed", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "dimension" in data
        assert "model" in data
        assert data["dimension"] == 3
        assert data["model"] == "TestEmbedder"

def test_embed_batch():
    """Тест векторизации батча текстов"""
    with patch('code_rag.api.dependencies.get_embedder') as mock_embedder:
        # Мокаем embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode_batch.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)
        mock_embedder_instance.embedding_dim = 3
        mock_embedder_instance.model.__class__.__name__ = "TestEmbedder"
        mock_embedder.return_value = mock_embedder_instance

        request_data = {
            "texts": [
                "def test1(): pass",
                "def test2(): pass"
            ],
            "batch_size": 32,
            "normalize": True
        }

        response = client.post("/embeddings/embed/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert "count" in data
        assert "dimension" in data
        assert "model" in data
        assert data["count"] == 2
        assert data["dimension"] == 3
        assert data["model"] == "TestEmbedder"

def test_calculate_similarity():
    """Тест вычисления схожести"""
    with patch('code_rag.api.dependencies.get_embedder') as mock_embedder:
        # Мокаем embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode_batch.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)
        mock_embedder_instance.embedding_dim = 3
        mock_embedder_instance.model.__class__.__name__ = "TestEmbedder"
        mock_embedder.return_value = mock_embedder_instance

        request_data = {
            "text1": "def test1(): pass",
            "text2": "def test2(): pass"
        }

        response = client.post("/embeddings/similarity", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "similarity" in data
        assert "text1_length" in data
        assert "text2_length" in data
        assert isinstance(data["similarity"], (int, float))
        assert 0.0 <= data["similarity"] <= 1.0
        assert data["text1_length"] == 17  # длина "def test1(): pass"
        assert data["text2_length"] == 17  # длина "def test2(): pass"

def test_get_model_info():
    """Тест получения информации о модели"""
    with patch('code_rag.api.dependencies.get_embedder') as mock_embedder:
        # Мокаем embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.model.__class__.__name__ = "TestEmbedder"
        mock_embedder_instance.embedding_dim = 768
        mock_embedder_instance.model.max_seq_length = 8192
        mock_embedder_instance.model.device = "cpu"
        mock_embedder.return_value = mock_embedder_instance

        response = client.get("/embeddings/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "embedding_dimension" in data
        assert "max_sequence_length" in data
        assert "device" in data
        assert data["model_name"] == "TestEmbedder"
        assert data["embedding_dimension"] == 768
        assert data["max_sequence_length"] == 8192
        assert data["device"] == "cpu"

def test_embed_code():
    """Тест векторизации кода"""
    with patch('code_rag.api.dependencies.get_embedder') as mock_embedder:
        # Мокаем embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode_text.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_embedder_instance.embedding_dim = 3
        mock_embedder_instance.model.__class__.__name__ = "TestEmbedder"
        mock_embedder.return_value = mock_embedder_instance

        request_data = {
            "text": "def test_function(): pass",
            "normalize": True
        }

        response = client.post("/embeddings/embed/code", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "dimension" in data
        assert "model" in data
        assert data["dimension"] == 3
        assert data["model"] == "TestEmbedder"
        assert len(data["embedding"]) == 3

def test_embed_query():
    """Тест векторизации запроса"""
    with patch('code_rag.api.dependencies.get_embedder') as mock_embedder:
        # Мокаем embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode_query.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_embedder_instance.embedding_dim = 3
        mock_embedder_instance.model.__class__.__name__ = "TestEmbedder"
        mock_embedder.return_value = mock_embedder_instance

        request_data = {
            "text": "find authentication functions",
            "normalize": True
        }

        response = client.post("/embeddings/embed/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "dimension" in data
        assert "model" in data
        assert data["dimension"] == 3
        assert data["model"] == "TestEmbedder"
        assert len(data["embedding"]) == 3

def test_health_check():
    """Тест проверки здоровья"""
    with patch('code_rag.api.dependencies.get_embedder') as mock_embedder:
        # Мокаем embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.encode_text.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_embedder_instance.embedding_dim = 3
        mock_embedder_instance.model.__class__.__name__ = "TestEmbedder"
        mock_embedder.return_value = mock_embedder_instance

        response = client.get("/embeddings/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "embedding_dim" in data
        assert data["status"] == "healthy"
        assert data["model"] == "TestEmbedder"
        assert data["embedding_dim"] == 3

def test_embed_text_validation():
    """Тест валидации ввода для embed_text"""
    # Пустой текст
    response = client.post("/embeddings/embed", json={"text": "", "normalize": True})
    assert response.status_code == 422  # Validation error

    # Недопустимый normalize
    response = client.post("/embeddings/embed", json={"text": "test", "normalize": "invalid"})
    assert response.status_code == 422  # Validation error

def test_embed_batch_validation():
    """Тест валидации ввода для embed_batch"""
    # Пустой список текстов
    response = client.post("/embeddings/embed/batch", json={"texts": [], "batch_size": 32})
    assert response.status_code == 422  # Validation error

    # Слишком много текстов
    many_texts = ["text"] * 101  # больше 100
    response = client.post("/embeddings/embed/batch", json={"texts": many_texts, "batch_size": 32})
    assert response.status_code == 422  # Validation error

    # Недопустимый batch_size
    response = client.post("/embeddings/embed/batch", json={"texts": ["test"], "batch_size": 0})
    assert response.status_code == 422  # Validation error

def test_similarity_validation():
    """Тест валидации ввода для similarity"""
    # Пустой text1
    response = client.post("/embeddings/similarity", json={"text1": "", "text2": "test"})
    assert response.status_code == 422  # Validation error

    # Пустой text2
    response = client.post("/embeddings/similarity", json={"text1": "test", "text2": ""})
    assert response.status_code == 422  # Validation error