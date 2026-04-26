# Talumo — MVP Local-First LLM Orchestrator

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
src/talumo/
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
uvicorn talumo.app:app --port 8000
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

llama-server runs in router mode and can serve both a general local model and a code local model from one process.

For GPU inference in llama-server:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up
```

Model path variables used by Compose (`.env`):

- `LLAMA_MODEL_HOST_DIR` (default `./models`)
- `LLAMA_MODELS_MAX` (default `2`)

Model IDs and files are configured in `config/llama_models.ini`:

- `local-general` -> `/models/model.gguf`
- `local-code` -> `/models/code.gguf`

Keep `.env` and `config/llama_models.ini` in sync if you change model filenames.

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

All settings are configurable via environment variables with `TALUMO_` prefix:

| Variable | Default | Description |
|---|---|---|
| `TALUMO_LITELLM_BASE_URL` | `http://localhost:4000/v1` | LiteLLM gateway endpoint |
| `TALUMO_LITELLM_API_KEY` | `sk-litellm` | LiteLLM master key |
| `TALUMO_LITELLM_LOCAL_MODEL` | `local` | LiteLLM model name for local tier |
| `TALUMO_LITELLM_LOCAL_CODE_MODEL` | `local_code` | LiteLLM model name for local code requests |
| `TALUMO_LITELLM_CHEAP_MODEL` | `cheap` | LiteLLM model name for cheap tier |
| `TALUMO_LITELLM_FRONTIER_MODEL` | `frontier` | LiteLLM model name for frontier tier |
| `TALUMO_MAX_TOKENS` | `2048` | Max tokens per completion |
| `TALUMO_TEMPERATURE` | `0.2` | Sampling temperature |
| `TALUMO_REQUEST_TIMEOUT` | `120.0` | Backend request timeout in seconds |
| `TALUMO_MIN_OUTPUT_CHARS` | `20` | Minimum acceptable output length |
| `TALUMO_MAX_OUTPUT_CHARS` | `50000` | Maximum acceptable output length |

## Tier selection rules

| Condition | Starting tier |
|---|---|
| `finality: final` | `remote_frontier` |
| `code`/`analysis` + `needs_schema`/`can_test` | `remote_cheap` |
| `sensitivity: public` | `remote_cheap` |
| Everything else | `local` |

Within the local tier, code requests automatically use the local code model alias.
You can override local model selection per request by adding:

```json
{
  "local_model_hint": "general"
}
```

Allowed values are `general` and `code`.
