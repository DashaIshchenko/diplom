# tests/test_indexing_api.py
import pytest
from fastapi.testclient import TestClient
from ..code_rag.api.app import app
from ..code_rag.core.vector_db import VectorizationResult
import json

client = TestClient(app)

def test_index_file_success():
    """Тест индексации файла"""
    request_data = {
        "file_path": "/tmp/test_file.py",
        "repository_name": "test-repo"
    }

    response = client.post("/indexing/file", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "total_files" in data

def test_index_directory_success():
    """Тест индексации директории"""
    request_data = {
        "directory_path": "/tmp/test_dir",
        "repository_name": "test-repo",
        "recursive": True
    }

    response = client.post("/indexing/directory", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "total_files" in data

def test_upload_and_index():
    """Тест загрузки и индексации файла"""
    # В реальном тесте нужно будет отправить actual file
    # Но мы можем проверить структуру запроса
    response = client.post("/indexing/upload")
    # Ожидаем 400, так как файл не передан
    assert response.status_code == 400

def test_get_collection_status():
    """Тест получения статуса коллекции"""
    response = client.get("/indexing/collection/status")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "exists" in data

def test_create_collection():
    """Тест создания коллекции"""
    response = client.post("/indexing/collection/create")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_delete_collection():
    """Тест удаления коллекции"""
    response = client.delete("/indexing/collection")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data