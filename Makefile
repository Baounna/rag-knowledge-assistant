VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup ingest inspect test clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## create venv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "ready. next: make ingest"

ingest: ## parse + chunk corpus/ -> data/chunks.jsonl
	$(PY) scripts/ingest.py

inspect: ## ingest and print sample chunks for eyeballing
	$(PY) scripts/ingest.py --inspect 5

test: ## run the test suite
	$(PY) -m pytest tests/ -q

clean:
	rm -rf data/*.jsonl .pytest_cache __pycache__ app/__pycache__ app/ingest/__pycache__
