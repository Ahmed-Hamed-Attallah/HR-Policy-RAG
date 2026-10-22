.PHONY: help install lint test test-unit test-api test-integration ingest build-index inspect eval run-backend run-frontend docker-up docker-down clean

PYTHON ?= python
PIP ?= pip
UVICORN ?= uvicorn
STREAMLIT ?= streamlit

help:
	@echo "Available commands:"
	@echo "  make install           Install Python dependencies"
	@echo "  make lint              Run ruff and black code checks"
	@echo "  make test              Execute full test suite with pytest"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-api          Run API route tests"
	@echo "  make ingest            Ingest raw documents from data/raw/"
	@echo "  make build-index       Chunk documents and build FAISS vector index"
	@echo "  make inspect           Inspect FAISS index contents and run sample query"
	@echo "  make eval              Run quantitative RAG evaluation benchmark"
	@echo "  make run-backend       Launch FastAPI backend service"
	@echo "  make run-frontend      Launch Streamlit frontend service"
	@echo "  make docker-up         Build and run services with docker-compose"
	@echo "  make docker-down       Stop and tear down container services"
	@echo "  make clean             Remove temporary artifacts and caches"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	ruff check .
	black --check .

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-api:
	pytest tests/api/ -v

test-integration:
	pytest tests/integration/ -v

ingest:
	$(PYTHON) scripts/ingest_documents.py

build-index:
	$(PYTHON) scripts/build_index.py --rebuild

inspect:
	$(PYTHON) scripts/inspect_index.py --query "What is the probation period?"

eval:
	$(PYTHON) scripts/evaluate_rag.py --top-k 5

run-backend:
	$(UVICORN) app.backend.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend:
	$(STREAMLIT) run app/frontend/Home.py --server.port 8501

docker-up:
	docker compose -f deployment/docker-compose.yml up --build -d

docker-down:
	docker compose -f deployment/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov