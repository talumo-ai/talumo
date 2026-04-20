# Zora — MVP Local-First LLM Orchestrator

A minimal orchestration layer that routes LLM requests through a local-first tiered pipeline with validation and escalation.

## Architecture

```
Client → FastAPI Orchestrator → LiteLLM Gateway → llama-server (local tier)
                                               → cheap remote model
                                               → frontier remote model
```

The orchestrator:
1. Classifies the request to pick an initial tier
2. Calls LiteLLM with the tier's model alias (`local`, `cheap`, or `frontier`)
3. Validates the output (length, refusal detection, JSON/code checks)
4. Retries once on the same tier if repairable
5. Escalates to the next tier if still failing
6. Returns the result with a full routing trace

## Project structure

```
src/zora/
  schemas.py       # Request/response models and enums
  settings.py      # Environment-based configuration
  classifier.py    # Tier selection logic
  backend.py       # HTTP client for OpenAI-compatible endpoints
  validators.py    # Output validation pipeline
  orchestrator.py  # Core routing/retry/escalation loop
  app.py           # FastAPI application
config/
  litellm_config.yaml  # LiteLLM model routing config
tests/                 # 33 tests covering all four MVP claims
```

## Quick start

```bash
# Create venv and install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Start the orchestrator (assumes backends are running)
uvicorn zora.app:app --port 8000
```

## Running with Docker Compose

Copy `.env.example` to `.env` and adjust model location values if needed:

```bash
cp .env.example .env

# Optional for local-only runs; required if a request uses remote tiers.
export OPENAI_API_KEY=sk-...
docker compose up
```

This starts llama-server (port 8080), LiteLLM (port 4000), and the orchestrator (port 8000).

For GPU inference in llama-server:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up
```

Model path variables used by Compose (`.env`):

- `LLAMA_MODEL_HOST_DIR` (default `./models`)
- `LLAMA_MODEL_CONTAINER_DIR` (default `/models`)
- `LLAMA_MODEL_FILE` (default `model.gguf`)

## Example request

```bash
curl -X POST http://localhost:8000/v1/orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "analysis",
    "finality": "draft",
    "sensitivity": "private",
    "messages": [{"role": "user", "content": "Summarize the key risks of this approach."}]
  }'
```

## Configuration

All settings are configurable via environment variables with `ZORA_` prefix:

| Variable | Default | Description |
|---|---|---|
| `ZORA_LITELLM_BASE_URL` | `http://localhost:4000/v1` | LiteLLM gateway endpoint |
| `ZORA_LITELLM_API_KEY` | `sk-litellm` | LiteLLM master key |
| `ZORA_LITELLM_LOCAL_MODEL` | `local` | LiteLLM model name for local tier |
| `ZORA_LITELLM_CHEAP_MODEL` | `cheap` | LiteLLM model name for cheap tier |
| `ZORA_LITELLM_FRONTIER_MODEL` | `frontier` | LiteLLM model name for frontier tier |
| `ZORA_MAX_TOKENS` | `2048` | Max tokens per completion |
| `ZORA_TEMPERATURE` | `0.2` | Sampling temperature |
| `ZORA_REQUEST_TIMEOUT` | `120.0` | Backend request timeout in seconds |
| `ZORA_MIN_OUTPUT_CHARS` | `20` | Minimum acceptable output length |
| `ZORA_MAX_OUTPUT_CHARS` | `50000` | Maximum acceptable output length |

## Tier selection rules

| Condition | Starting tier |
|---|---|
| `finality: final` | `remote_frontier` |
| `code`/`analysis` + `needs_schema`/`can_test` | `remote_cheap` |
| `sensitivity: public` | `remote_cheap` |
| Everything else | `local` |
