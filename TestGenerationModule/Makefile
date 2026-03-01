.PHONY: help build up down logs restart clean

help:
	@echo "Code RAG - Docker команды"
	@echo ""
	@echo "  make build    - Собрать Docker образы"
	@echo "  make up       - Запустить сервисы"
	@echo "  make down     - Остановить сервисы"
	@echo "  make logs     - Показать логи"
	@echo "  make restart  - Перезапустить сервисы"
	@echo "  make clean    - Очистить всё (включая volumes)"
	@echo ""

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "✅ Сервисы запущены!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "Qdrant: http://localhost:6333/dashboard"

down:
	docker-compose down

logs:
	docker-compose logs -f

restart:
	docker-compose restart

clean:
	docker-compose down -v
	docker system prune -f
	@echo "✅ Очистка завершена"
