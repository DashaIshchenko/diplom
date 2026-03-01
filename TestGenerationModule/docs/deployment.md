# Deployment Guide

## Обзор

Это руководство описывает различные способы развертывания Code RAG в production окружении.

## Содержание

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Performance Tuning](#performance-tuning)
6. [Security](#security)
7. [Monitoring](#monitoring)

---

## Local Development

### Требования

**Минимальные:**
- Python 3.9+
- 4GB RAM
- 2 CPU cores
- 10GB disk space

**Рекомендуемые:**
- Python 3.10+
- 16GB RAM
- 4+ CPU cores
- 50GB SSD
- GPU (опционально)

### Установка

```
# 1. Клонирование репозитория
git clone https://github.com/your-org/code-rag.git
cd code-rag

# 2. Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Установка для разработки
pip install -e .
```

### Конфигурация

Создайте `.env` файл:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_VERSION=0.1.0

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=code_collection

# Embeddings
EMBEDDING_MODEL=nomic-embed-code
EMBEDDING_DEVICE=cpu

# Qwen
QWEN_API_KEY=your-api-key-here
QWEN_MODEL=qwen2.5-coder-32b-instruct
QWEN_TEMPERATURE=0.7
QWEN_MAX_TOKENS=2048

# Logging
LOG_LEVEL=INFO
```

### Запуск Qdrant

```
# Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# Или Docker Compose
docker-compose up -d qdrant
```

### Запуск API

```
# Development сервер с hot reload
uvicorn code_rag.api.app:app --reload --host 0.0.0.0 --port 8000

# Или через Python
python -m code_rag.api.main

# С workers
uvicorn code_rag.api.app:app --workers 4 --host 0.0.0.0 --port 8000
```

### Проверка

```
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs

# Qdrant UI
open http://localhost:6333/dashboard
```

---

## Docker Deployment

### Single Container

**Dockerfile:**
```
FROM python:3.10-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY code_rag/ ./code_rag/
COPY .env .

# Данные
RUN mkdir -p /app/logs /app/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "code_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Сборка и запуск:**
```
# Сборка
docker build -t code-rag:latest .

# Запуск
docker run -d \
  --name code-rag-api \
  -p 8000:8000 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e QWEN_API_KEY=your-key \
  -v $(pwd)/logs:/app/logs \
  code-rag:latest
```

### Docker Compose

**docker-compose.yml:**
```
version: '3.8'

services:
  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: code_rag_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Code RAG API
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: code_rag_api
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - QDRANT_URL=http://qdrant:6333
      - COLLECTION_NAME=code_collection
      - LOG_LEVEL=INFO
      - QWEN_API_KEY=${QWEN_API_KEY}
    depends_on:
      qdrant:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Nginx (опционально)
  nginx:
    image: nginx:alpine
    container_name: code_rag_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped

volumes:
  qdrant_storage:
    driver: local

networks:
  default:
    name: code_rag_network
```

**Запуск:**
```
# Создайте .env файл с переменными
echo "QWEN_API_KEY=your-key" > .env

# Запуск всех сервисов
docker-compose up -d

# Логи
docker-compose logs -f

# Остановка
docker-compose down

# Полная очистка (включая volumes)
docker-compose down -v
```

### Nginx Configuration

**nginx.conf:**
```
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL certificates
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Client max body size
        client_max_body_size 100M;

        # Timeouts
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;

        # API endpoints
        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket support (для streaming)
        location /generation/stream {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Health check
        location /health {
            proxy_pass http://api/health;
            access_log off;
        }
    }
}
```

### Multi-Stage Build (Оптимизация)

```
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Копируем wheels из builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

# Код приложения
COPY code_rag/ ./code_rag/
COPY .env .

RUN mkdir -p /app/logs /app/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "code_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## Kubernetes Deployment

### Architecture

```
┌─────────────────────────────────────────┐
│           Ingress Controller            │
│              (NGINX/Traefik)            │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────────┐   ┌───────▼──────────┐
│  API Service   │   │  Qdrant Service  │
│  (LoadBalancer)│   │  (ClusterIP)     │
└───┬────────────┘   └───────┬──────────┘
    │                        │
┌───▼──────────────────┐  ┌──▼─────────────┐
│   API Deployment     │  │ Qdrant StatefulSet│
│   (3 replicas)       │  │   (1 replica)     │
└──────────────────────┘  └───────────────────┘
```

### Namespace

**namespace.yaml:**
```
apiVersion: v1
kind: Namespace
metadata:
  name: code-rag
```

### ConfigMap

**configmap.yaml:**
```
apiVersion: v1
kind: ConfigMap
metadata:
  name: code-rag-config
  namespace: code-rag
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  QDRANT_URL: "http://qdrant-service:6333"
  COLLECTION_NAME: "code_collection"
  LOG_LEVEL: "INFO"
  EMBEDDING_MODEL: "nomic-embed-code"
```

### Secrets

**secrets.yaml:**
```
apiVersion: v1
kind: Secret
metadata:
  name: code-rag-secrets
  namespace: code-rag
type: Opaque
stringData:
  QWEN_API_KEY: "your-api-key-here"
  QDRANT_API_KEY: ""
```

```
# Создание из командной строки
kubectl create secret generic code-rag-secrets \
  --namespace=code-rag \
  --from-literal=QWEN_API_KEY=your-key
```

### Qdrant StatefulSet

**qdrant-statefulset.yaml:**
```
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
  namespace: code-rag
spec:
  selector:
    app: qdrant
  ports:
    - name: http
      port: 6333
      targetPort: 6333
    - name: grpc
      port: 6334
      targetPort: 6334
  clusterIP: None
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: code-rag
spec:
  serviceName: qdrant-service
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
          name: http
        - containerPort: 6334
          name: grpc
        env:
        - name: QDRANT__SERVICE__GRPC_PORT
          value: "6334"
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /
            port: 6333
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 6333
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "standard"
      resources:
        requests:
          storage: 20Gi
```

### API Deployment

**api-deployment.yaml:**
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-rag-api
  namespace: code-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: code-rag-api
  template:
    metadata:
      labels:
        app: code-rag-api
    spec:
      containers:
      - name: api
        image: your-registry/code-rag:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: code-rag-config
        - secretRef:
            name: code-rag-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
***
apiVersion: v1
kind: Service
metadata:
  name: code-rag-api-service
  namespace: code-rag
spec:
  type: LoadBalancer
  selector:
    app: code-rag-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
```

### Ingress

**ingress.yaml:**
```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: code-rag-ingress
  namespace: code-rag
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
spec:
  tls:
  - hosts:
    - code-rag.your-domain.com
    secretName: code-rag-tls
  rules:
  - host: code-rag.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: code-rag-api-service
            port:
              number: 80
```

### HorizontalPodAutoscaler

**hpa.yaml:**
```
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: code-rag-api-hpa
  namespace: code-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: code-rag-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

### Deployment Commands

```
# 1. Создать namespace
kubectl apply -f namespace.yaml

# 2. Создать ConfigMap и Secrets
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml

# 3. Deploy Qdrant
kubectl apply -f qdrant-statefulset.yaml

# 4. Дождаться готовности Qdrant
kubectl wait --namespace=code-rag \
  --for=condition=ready pod \
  --selector=app=qdrant \
  --timeout=300s

# 5. Deploy API
kubectl apply -f api-deployment.yaml

# 6. Apply Ingress
kubectl apply -f ingress.yaml

# 7. Apply HPA
kubectl apply -f hpa.yaml

# 8. Проверка
kubectl get all -n code-rag
kubectl logs -n code-rag -l app=code-rag-api -f
```

### Helm Chart

**Chart.yaml:**
```
apiVersion: v2
name: code-rag
description: Code RAG Helm Chart
type: application
version: 0.1.0
appVersion: "0.1.0"
```

**values.yaml:**
```
replicaCount: 3

image:
  repository: your-registry/code-rag
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: LoadBalancer
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: code-rag.your-domain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: code-rag-tls
      hosts:
        - code-rag.your-domain.com

resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

qdrant:
  enabled: true
  replicas: 1
  storage:
    size: 20Gi
    storageClass: standard
  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"
    limits:
      memory: "4Gi"
      cpu: "2000m"

config:
  apiHost: "0.0.0.0"
  apiPort: 8000
  collectionName: "code_collection"
  logLevel: "INFO"

secrets:
  qwenApiKey: ""
```

**Установка через Helm:**
```
# Установка
helm install code-rag ./helm/code-rag \
  --namespace code-rag \
  --create-namespace \
  --set secrets.qwenApiKey=your-key

# Обновление
helm upgrade code-rag ./helm/code-rag \
  --namespace code-rag

# Удаление
helm uninstall code-rag --namespace code-rag
```

## Cloud Deployment

### AWS (Elastic Kubernetes Service)

**Infrastructure as Code (Terraform):**

```
# eks-cluster.tf
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "code-rag-cluster"
  cluster_version = "1.27"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    api_nodes = {
      min_size     = 3
      max_size     = 10
      desired_size = 3

      instance_types = ["t3.xlarge"]
      capacity_type  = "ON_DEMAND"

      labels = {
        role = "api"
      }
    }

    qdrant_nodes = {
      min_size     = 1
      max_size     = 3
      desired_size = 1

      instance_types = ["r6i.xlarge"]
      capacity_type  = "ON_DEMAND"

      labels = {
        role = "qdrant"
      }
    }
  }
}

# s3-bucket.tf
resource "aws_s3_bucket" "code_rag_storage" {
  bucket = "code-rag-storage"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    enabled = true

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }
}
```

**Deployment:**
```
# 1. Создать кластер
terraform apply

# 2. Настроить kubectl
aws eks update-kubeconfig --name code-rag-cluster --region us-east-1

# 3. Deploy приложение
kubectl apply -f k8s/

# 4. Проверка
kubectl get nodes
kubectl get pods -n code-rag
```

### Google Cloud (GKE)

```
# Создать кластер
gcloud container clusters create code-rag-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-4 \
  --region=us-central1 \
  --enable-autoscaling \
  --min-nodes=3 \
  --max-nodes=10

# Получить credentials
gcloud container clusters get-credentials code-rag-cluster \
  --region=us-central1

# Deploy
kubectl apply -f k8s/
```

### Azure (AKS)

```
# Создать resource group
az group create --name code-rag-rg --location eastus

# Создать кластер
az aks create \
  --resource-group code-rag-rg \
  --name code-rag-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10 \
  --generate-ssh-keys

# Получить credentials
az aks get-credentials \
  --resource-group code-rag-rg \
  --name code-rag-cluster

# Deploy
kubectl apply -f k8s/
```

---

## Performance Tuning

### API Optimization

**uvicorn с workers:**
```
uvicorn code_rag.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --backlog 2048 \
  --timeout-keep-alive 30
```

**gunicorn configuration:**
```
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
timeout = 300
keepalive = 30
preload_app = True
```

### Embedder Optimization

```
# config.py
EMBEDDING_BATCH_SIZE = 32  # CPU
EMBEDDING_BATCH_SIZE = 128  # GPU

# Используйте GPU если доступен
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Кэширование
EMBEDDING_CACHE_SIZE = 10000
```

### Qdrant Optimization

```
# qdrant-config.yaml
service:
  max_request_size_mb: 100
  max_workers: 4
  
storage:
  performance:
    max_optimization_threads: 4
  
collections:
  default_segment_number: 4
  
hnsw_config:
  m: 16
  ef_construct: 200
  full_scan_threshold: 10000
```

### Database Indexing

```
# Создание индексов
from qdrant_client.models import PayloadSchemaType

qdrant_client.create_payload_index(
    collection_name="code_collection",
    field_name="repository_name",
    field_schema=PayloadSchemaType.KEYWORD
)

qdrant_client.create_payload_index(
    collection_name="code_collection",
    field_name="language",
    field_schema=PayloadSchemaType.KEYWORD
)
```

### Caching Strategy

```
from functools import lru_cache
from redis import Redis

# In-memory cache
@lru_cache(maxsize=10000)
def get_embedding_cached(text: str):
    return embedder.encode_text(text)

# Redis cache
redis_client = Redis(host='localhost', port=6379, db=0)

def get_search_results_cached(query: str, top_k: int):
    cache_key = f"search:{query}:{top_k}"
    
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    results = retriever.search(query, top_k=top_k)
    redis_client.setex(cache_key, 3600, json.dumps(results))
    
    return results
```

---

## Security

### Authentication

**API Key Authentication:**
```
# middleware.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# В routes
@router.post("/search/")
async def search(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key)
):
    ...
```

### Rate Limiting

```
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/search/")
@limiter.limit("100/minute")
async def search_endpoint(request: Request):
    ...
```

### HTTPS/TLS

```
# Let's Encrypt с cert-manager
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

### Network Policies

```
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: code-rag-network-policy
  namespace: code-rag
spec:
  podSelector:
    matchLabels:
      app: code-rag-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - protocol: TCP
      port: 6333
```

---

## Monitoring

### Prometheus Metrics

```
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Метрики
request_count = Counter(
    'code_rag_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'code_rag_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'code_rag_active_connections',
    'Active connections'
)

# В middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response
```

### Grafana Dashboard

```
{
  "dashboard": {
    "title": "Code RAG Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(code_rag_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(code_rag_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(code_rag_requests_total{status=~\"5..\"}[5m])"
          }
        ]
      }
    ]
  }
}
```

### Logging

```
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": "/app/logs/app.log",
            "maxBytes": 10485760,
            "backupCount": 5
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

### Health Checks

```
# health.py
@router.get("/health/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness(
    embedder: CodeEmbedder = Depends(get_embedder),
    qdrant: QdrantClient = Depends(get_qdrant_client)
):
    """Kubernetes readiness probe."""
    checks = {
        "embedder": False,
        "qdrant": False
    }
    
    try:
        embedder.encode_text("test")
        checks["embedder"] = True
    except:
        pass
    
    try:
        checks["qdrant"] = qdrant.health_check()
    except:
        pass
    
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail="Not ready")
```

---

## Backup and Restore

### Qdrant Backup

```
# Создание snapshot
curl -X POST "http://localhost:6333/collections/code_collection/snapshots"

# Список snapshots
curl "http://localhost:6333/collections/code_collection/snapshots"

# Восстановление
curl -X PUT "http://localhost:6333/collections/code_collection/snapshots/upload" \
  --data-binary @snapshot.tar
```

### Automated Backups (Kubernetes CronJob)

```
apiVersion: batch/v1
kind: CronJob
metadata:
  name: qdrant-backup
  namespace: code-rag
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: curlimages/curl:latest
            command:
            - /bin/sh
            - -c
            - |
              curl -X POST http://qdrant-service:6333/collections/code_collection/snapshots
              # Upload to S3 или другое хранилище
          restartPolicy: OnFailure
```

---

## Troubleshooting

### Common Issues

**Проблема: OOM (Out of Memory)**
```
# Проверка использования памяти
kubectl top pods -n code-rag

# Увеличение лимитов
kubectl set resources deployment code-rag-api \
  --limits=memory=8Gi,cpu=4 \
  --requests=memory=4Gi,cpu=2
```

**Проблема: Медленный поиск**
```
# Проверка Qdrant
curl http://localhost:6333/collections/code_collection

# Оптимизация индексов
# Увеличение ef_construct и m параметров HNSW
```

**Проблема: API timeout**
```
# Увеличение таймаутов в Nginx/Ingress
kubectl annotate ingress code-rag-ingress \
  nginx.ingress.kubernetes.io/proxy-read-timeout="600"
```

---

## Best Practices

1. **Use separate namespaces** для разных окружений (dev, staging, prod)
2. **Implement proper monitoring** с alerts
3. **Regular backups** Qdrant данных
4. **Use secrets management** (Vault, AWS Secrets Manager)
5. **Implement CI/CD** pipeline
6. **Use resource limits** для всех pods
7. **Enable autoscaling** для API
8. **Use persistent volumes** для Qdrant
9. **Implement proper logging** с централизованным хранением
10. **Regular security updates** для Docker images

---

## CI/CD Pipeline

**GitHub Actions Example:**

```
# .github/workflows/deploy.yml
name: Deploy to Kubernetes

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t ${{ secrets.REGISTRY }}/code-rag:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USER }} --password-stdin
        docker push ${{ secrets.REGISTRY }}/code-rag:${{ github.sha }}
    
    - name: Deploy to K8s
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/deployment.yaml
        images: |
          ${{ secrets.REGISTRY }}/code-rag:${{ github.sha }}
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
```

---