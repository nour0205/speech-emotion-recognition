.PHONY: install install-dev backend frontend run docker docker-build docker-down docker-dev docker-test clean help fixtures predict

# Default target
help:
	@echo "Speech Emotion Recognition - Available commands:"
	@echo ""
	@echo "  Local development:"
	@echo "    make run          Start both backend and frontend locally"
	@echo "    make backend      Start the FastAPI backend only (port 8000)"
	@echo "    make frontend     Start the Streamlit frontend only (port 8501)"
	@echo "    make install      Install production dependencies"
	@echo "    make install-dev  Install development dependencies"
	@echo "    make test         Run tests locally"
	@echo "    make fixtures     Generate test audio fixtures"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker       Build and run with Docker Compose"
	@echo "    make docker-build Build Docker images without cache"
	@echo "    make docker-down  Stop and remove containers"
	@echo "    make docker-dev   Run dev container interactively"
	@echo "    make docker-test  Run tests in Docker"
	@echo ""
	@echo "  Scripts:"
	@echo "    make predict FILE=path/to/audio.wav    Predict emotion from file"
	@echo ""
	@echo "  Utilities:"
	@echo "    make clean        Remove cache and temporary files"
	@echo ""

# Run both services together (local)
run:
	@chmod +x run.sh
	@./run.sh

# Docker commands
docker:
	docker compose up --build

docker-build:
	docker compose build --no-cache

docker-down:
	docker compose down -v

# Install production dependencies
install:
	pip install -r requirements/backend.txt
	pip install -r requirements/frontend.txt

# Install development dependencies
install-dev:
	pip install -r requirements/dev.txt

# Start backend API server
backend:
	@echo "Starting FastAPI backend on http://localhost:8000"
	@echo "API docs available at http://localhost:8000/docs"
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend Streamlit app
frontend:
	@echo "Starting Streamlit frontend on http://localhost:8501"
	cd frontend && streamlit run app.py --server.port 8501

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cleaned up cache files"

# Run tests
test:
	pytest -v

# Format code
format:
	black backend/ frontend/
	ruff check --fix backend/ frontend/

# Lint code
lint:
	ruff check backend/ frontend/

# Docker dev container
docker-dev:
	docker compose run --rm dev bash

# Run tests in Docker
docker-test:
	docker compose run --rm dev pytest -v

# Run tests in Docker with integration tests
docker-test-integration:
	docker compose run --rm -e RUN_INTEGRATION_TESTS=1 dev pytest -v

# Generate test fixtures
fixtures:
	python scripts/generate_fixtures.py

# Generate fixtures in Docker
docker-fixtures:
	docker compose run --rm dev python scripts/generate_fixtures.py

# Predict emotion from a file (local)
predict:
	@if [ -z "$(FILE)" ]; then echo "Usage: make predict FILE=path/to/audio.wav"; exit 1; fi
	python scripts/predict_file.py --input "$(FILE)" --pretty

# Predict emotion from a file (Docker)
docker-predict:
	@if [ -z "$(FILE)" ]; then echo "Usage: make docker-predict FILE=path/to/audio.wav"; exit 1; fi
	docker compose run --rm dev python scripts/predict_file.py --input "$(FILE)" --pretty
