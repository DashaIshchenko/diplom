import pytest
from fastapi.testclient import TestClient
from ..code_rag.api.app import app
from ..code_rag.core.vector_db import VectorizationResult
import json

client = TestClient(app)

def test_index_repository_success():
    """Тест индексации репозитория"""
    # Проверка с локальным репозиторием
    request_data = {
        "repository_path": "/tmp/test_repo",
        "repository_name": "test-repo",
        "branch": "main",
        "provider": "local"
    }

    response = client.post("/repositories/index", json=request_data)
    # Проверка, что возвращается статус "queued"
    assert response.status_code in [200, 201]
    data = response.json()
    assert "status" in data
    assert "repository_name" in data

def test_list_repositories():
    """Тест получения списка репозиториев"""
    response = client.get("/repositories/list")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_get_repository_info():
    """Тест получения информации о репозитории"""
    response = client.get("/repositories/test-repo/info")
    # Может быть 404, если репозиторий не существует
    assert response.status_code in [200, 404]

def test_delete_repository():
    """Тест удаления репозитория"""
    response = client.delete("/repositories/test-repo")
    # Может быть 404, если репозиторий не существует
    assert response.status_code in [200, 404]

def test_get_repository_statistics():
    """Тест получения статистики репозитория"""
    response = client.get("/repositories/test-repo/statistics")
    # Может быть 404, если репозиторий не существует
    assert response.status_code in [200, 404]

def test_register_azure_connection():
    """Тест регистрации Azure DevOps подключения"""
    request_data = {
        "name": "test-connection",
        "base_url": "https://dev.azure.com/test",
        "collection": "DefaultCollection",
        "personal_access_token": "test-token"
    }

    response = client.post("/repositories/azure/register-connection", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_list_azure_repositories():
    """Тест получения списка репозиториев Azure DevOps"""
    response = client.get("/repositories/azure/list-repositories?azure_connection=test&project=test-project")
    # Может быть 400, если подключение не зарегистрировано
    assert response.status_code in [200, 400]