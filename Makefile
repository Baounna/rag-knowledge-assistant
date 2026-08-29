VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
DC   := docker compose
RUN  := $(DC) run --rm app python

.PHONY: help up up-local down logs ps rebuild ingest index reindex eval eval-full test \
        demo search shell admin db-up db-down ollama-up ollama-down clean nuke \
        local-setup local-serve local-test

help:
	@echo "Everything runs in Docker. Nothing is installed on your machine."
	@echo ""
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

# ---- run --------------------------------------------------------------

up: ## start database + app          -> http://localhost:8000
	$(DC) up -d --build app
	@echo "ready: http://localhost:8000"

up-local: ## start database + app + local LLM (free, no API key)
	$(DC) --profile ollama up -d ollama
	@until curl -sf localhost:11434/api/tags >/dev/null 2>&1; do sleep 2; done
	$(DC) exec ollama ollama pull $${OLLAMA_MODEL:-qwen2.5:7b-instruct}
	$(DC) up -d --build app
	@echo "ready: http://localhost:8000  (set LLM_PROVIDER=ollama in .env)"

down: ## stop everything (data is kept)
	$(DC) --profile ollama down

logs: ## follow the app log
	$(DC) logs -f app

ps: ## what is running, and how much it is using
	@docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' \
	 | grep -E 'NAME|rag-' || echo "nothing running"

rebuild: ## rebuild the app image after changing dependencies
	$(DC) build --no-cache app

# ---- pipeline ---------------------------------------------------------

ingest: ## parse + chunk corpus/ -> data/chunks.jsonl
	$(RUN) scripts/ingest.py

index: ## embed chunks and load them into Postgres
	$(RUN) scripts/index.py

reindex: ## drop the table and rebuild from scratch
	$(RUN) scripts/index.py --recreate

eval: ## retrieval metrics across all configurations (no API key needed)
	$(RUN) scripts/eval.py

eval-full: ## + answers, citations, refusal, LLM judge
	$(RUN) scripts/eval.py --generate --judge

test: ## run the test suite
	# tests are mounted rather than baked into the image: the deployable
	# artifact should not carry the test suite or pytest.
	$(DC) run --rm -v $(PWD)/tests:/app/tests:ro \
	  -e DATABASE_URL=postgresql://rag:rag@db:5432/rag app \
	  sh -c "pip install -q pytest && python -m pytest tests/ -q"

demo: ## interactive retrieval demo in the terminal
	$(DC) run --rm -it app python scripts/demo.py

search: ## probe both indexes:  make search Q="expense deadline"
	$(RUN) scripts/index.py --probe "$(Q)"

admin: ## grant admin rights:  make admin EMAIL=you@company.com
	$(RUN) scripts/admin.py --email "$(EMAIL)"

shell: ## a shell inside the app container
	$(DC) run --rm -it app bash

# ---- pieces -----------------------------------------------------------

db-up: ## start only the database
	$(DC) up -d db
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' rag-db 2>/dev/null)" = healthy ]; do sleep 2; done

db-down: ## stop only the database
	$(DC) stop db

ollama-up: ## start only the local LLM and pull its model
	$(DC) --profile ollama up -d ollama
	@until curl -sf localhost:11434/api/tags >/dev/null 2>&1; do sleep 2; done
	$(DC) exec ollama ollama pull $${OLLAMA_MODEL:-qwen2.5:7b-instruct}

ollama-down: ## stop only the local LLM
	$(DC) stop ollama

# ---- cleanup ----------------------------------------------------------

clean: ## stop containers and delete generated data (keeps images + models)
	$(DC) --profile ollama down
	rm -rf data/*.jsonl .pytest_cache
	@echo "containers stopped, generated data removed"

nuke: ## delete EVERYTHING this project created: containers, volumes, images
	$(DC) --profile ollama down -v
	-docker rmi 1317stage-app paradedb/paradedb ollama/ollama
	rm -rf .venv data/*.jsonl .pytest_cache
	@echo "removed. reclaims roughly 15GB."

# ---- optional: run on the host instead of in Docker -------------------

local-setup: ## OPTIONAL: python venv on your machine (not needed if using Docker)
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@test -f .env || cp .env.example .env

local-serve: ## OPTIONAL: run the app from the host venv
	$(PY) -m uvicorn app.api:app --reload --port 8000

local-test: ## OPTIONAL: run tests from the host venv
	$(PY) -m pytest tests/ -q
