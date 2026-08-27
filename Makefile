VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup db-up db-down ingest inspect index reindex search test clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## create venv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "ready. next: make ingest"

db-up: ## start Postgres (pgvector + pg_search)
	docker compose up -d db
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' rag-db 2>/dev/null)" = healthy ]; do sleep 2; done
	@echo "db healthy on localhost:5432"

db-down: ## stop the database (keeps data)
	docker compose down

index: ## embed chunks and load them into Postgres
	$(PY) scripts/index.py

reindex: ## drop the table and rebuild from scratch
	$(PY) scripts/index.py --recreate

search: ## probe both indexes:  make search Q="how many days to submit expenses"
	$(PY) scripts/index.py --probe "$(Q)"

ingest: ## parse + chunk corpus/ -> data/chunks.jsonl
	$(PY) scripts/ingest.py

inspect: ## ingest and print sample chunks for eyeballing
	$(PY) scripts/ingest.py --inspect 5

test: ## run the test suite
	$(PY) -m pytest tests/ -q

clean:
	rm -rf data/*.jsonl .pytest_cache __pycache__ app/__pycache__ app/ingest/__pycache__
