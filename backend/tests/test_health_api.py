# tests/test_health_api.py
import pytest
from fastapi.testclient import TestClient
from code_rag.api.app import app
import json

client = TestClient(app)

def test_health_check():
    """Тест health check"""
    response = client.get("/health/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data
    assert "components" in data

def test_root_endpoint():
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert "docs" in data

def test_system_info():
    """Тест информации о системе"""
    response = client.get("/health/info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "embedder_model" in data
    assert "qdrant_url" in data
    assert "components_status" in data

def test_ping():
    """Тест ping"""
    response = client.get("/health/ping")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

def test_version():
    """Тест версии API"""
    response = client.get("/health/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "title" in data

def test_detailed_health():
    """Тест детального health check"""
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data