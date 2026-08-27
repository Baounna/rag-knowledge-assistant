VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup db-up db-down ollama-up ollama-down ingest inspect index reindex search demo serve admin eval eval-full test clean

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

ollama-up: ## start the local LLM and pull its model (free, no API key)
	docker compose --profile ollama up -d ollama
	@until curl -sf localhost:11434/api/tags >/dev/null; do sleep 2; done
	docker exec rag-ollama ollama pull $${OLLAMA_MODEL:-qwen2.5:7b-instruct}
	@echo "local LLM ready -- set LLM_PROVIDER=ollama in .env"

ollama-down: ## stop the local LLM
	docker compose --profile ollama down

db-down: ## stop the database (keeps data)
	docker compose down

index: ## embed chunks and load them into Postgres
	$(PY) scripts/index.py

reindex: ## drop the table and rebuild from scratch
	$(PY) scripts/index.py --recreate

demo: ## interactive retrieval demo (type questions, see both indexes)
	$(PY) scripts/demo.py

search: ## probe both indexes:  make search Q="how many days to submit expenses"
	$(PY) scripts/index.py --probe "$(Q)"

ingest: ## parse + chunk corpus/ -> data/chunks.jsonl
	$(PY) scripts/ingest.py

inspect: ## ingest and print sample chunks for eyeballing
	$(PY) scripts/ingest.py --inspect 5

serve: ## run the app at http://localhost:8000
	$(PY) -m uvicorn app.api:app --reload --port 8000

admin: ## promote a user to admin:  make admin EMAIL=you@company.com
	$(PY) scripts/admin.py --email "$(EMAIL)"

eval: ## retrieval metrics across all configurations (no API key needed)
	$(PY) scripts/eval.py

eval-full: ## + answers, citations, refusal, LLM judge (needs ANTHROPIC_API_KEY)
	$(PY) scripts/eval.py --generate --judge

test: ## run the test suite
	$(PY) -m pytest tests/ -q

clean:
	rm -rf data/*.jsonl .pytest_cache __pycache__ app/__pycache__ app/ingest/__pycache__
