# CUWALID-LLM-Web

CUWALID-LLM-Web is a FastAPI web application that provides a chat interface for two use cases: deterministic seasonal forecast lookup and CUWALID modelling/deployment guidance. Forecast answers combine an LLM-powered intent extractor with strict deterministic lookup from bundled CSV data, while modelling answers are generated via OpenRouter using curated prompt instructions. The frontend is a single-page UI served by FastAPI templates and static assets. This document explains architecture, runtime flow, operations, and each major file so a new maintainer can take over quickly.

## Architecture Overview

The app is organized into four runtime layers:

1. Web/API layer
- `app/main.py` receives requests, validates payloads, initializes services, serves UI assets, and exposes `/`, `/api/chat`, and `/status`.

2. Orchestration and domain logic
- `app/core/chat_service.py` routes by mode (`forecast` or `modelling`), applies guardrails, manages slot-filling/clarification loops, and formats responses.
- `app/core/forecast_store.py` is the deterministic source of truth for data loading, place resolution, forecast retrieval, and status interpretation.
- `app/core/session_store.py` keeps in-memory conversation state (intent + recent history).
- `app/core/guardrails.py` blocks out-of-scope and policy-prescriptive requests.

3. Extraction/integration layer
- `app/integrations/openrouter_extractor.py` extracts forecast intent fields from user text via OpenRouter.
- `app/integrations/extractor_stub.py` provides deterministic regex fallback extraction when LLM extraction is sparse or unavailable.

4. Prompt and UI layer
- `app/prompts/*.prompt` define system behavior for modelling and forecast-help interactions.
- `app/templates/index.html` and `app/static/*` implement the browser UI and client logic.

## Runtime Flow

1. Startup
- FastAPI starts in `app/main.py`.
- `ForecastStore.load_all()` reads CSVs from `data/` into memory.

2. User request
- Frontend posts to `/api/chat` with `message`, `mode`, `tier`, `conversation_id`.

3. Forecast mode
- Guardrails run first.
- Intent score decides whether to route to capability/help response or field extraction.
- Extracted intent is merged with session intent, normalized, and validated.
- If complete and valid, deterministic lookup returns status + map path metadata.
- If incomplete/ambiguous, clarification response is returned.

4. Modelling mode
- `expert.prompt` + user message are sent to OpenRouter.
- If provider is unavailable/slow, a fallback message with docs guidance is returned.

## Docker And Local Operations

### Run with Docker Compose

```bash
docker compose up --build
```

Open: `http://localhost:8000`

Stop:

```bash
docker compose down
```

### Direct Docker build/run

```bash
docker build -t cuwalid-web .
docker run --rm -p 8000:8000 --env-file .env cuwalid-web
```

### Local Python run (without Docker)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Required Configuration

Create `.env` from `.env.example` and set at minimum:

- `CHAT_PROVIDER=openrouter`
- `OPENROUTER_API_KEY=<your_key>`
- `LLM_MODEL=<model_for_forecast_help_and_extraction>`
- `MODELLING_LLM_MODEL=<model_for_modelling_assistant>`
- `OPENROUTER_TIMEOUT_SECONDS=18`
- `MODELLING_RETRY_COUNT=1`
- `CUWALID_DATA_ROOT=data`

## File-By-File Takeover Notes

### Top level

- `Dockerfile`: Builds the production image from `python:3.11-slim`, installs dependencies, copies `app/` and `data/`, and runs Uvicorn.
- `docker-compose.yml`: Defines the single `cuwalid-web` service, exposes port `8000`, loads `.env`, and sets default runtime env vars.
- `DOCKER.md`: Short Docker operator notes for partners and direct command examples.
- `requirements.txt`: Python dependency lock list for API server, templating, and OpenRouter client.
- `.env.example`: Baseline environment template for new deployments.

### App entry and web layer

- `app/main.py`: Main application entry point. Initializes logging, mounts static/template directories, wires `ChatService`, and defines API endpoints and middleware.
- `app/templates/index.html`: Main UI shell including landing view, chat screen, mode switch, audience chips, and script/style loading.
- `app/static/app.js`: Client application logic for chats, session tabs, request handling, rendering markdown, mode/audience selection, and localStorage persistence.
- `app/static/styles.css`: Theme, layout, responsive behavior, and component styling for both landing and chat views.

### Core logic

- `app/core/chat_service.py`: Central business logic. Handles routing, guardrails, intent extraction/merging, normalization/autocorrect, deterministic lookup, response formatting, and fallback behavior.
- `app/core/forecast_store.py`: In-memory deterministic data engine for loading forecast CSVs, searching/resolving locations, retrieving status codes, and interpreting domain labels (including flood semantics).
- `app/core/session_store.py`: In-memory conversation state manager with intent/history updates, reset, and stale-session pruning.
- `app/core/guardrails.py`: Rule-based checks and standard refusal/clarification payloads for unsafe, vague, or out-of-domain prompts.
- `app/core/config.py`: Environment-backed settings dataclass used by chat and integration modules.

### Integrations and prompts

- `app/integrations/openrouter_extractor.py`: OpenRouter JSON extractor that converts free text into the forecast slot schema.
- `app/integrations/extractor_stub.py`: Regex-based backup extractor for robust slot backfilling.
- `app/prompts/expert.prompt`: System prompt for modelling assistant behavior and knowledge boundaries.
- `app/prompts/forecast.prompt`: System prompt for forecast capability/help replies.

### Data

- `data/2026/OND/kenya.csv`: Example deterministic seasonal forecast input; additional country/season/year CSVs follow the same loader pattern.

## Operational Checks

Use these after deployment:

1. `GET /status` should show loaded countries, seasons, and variables.
2. `GET /` should render UI and load static assets.
3. A forecast query with complete fields should return a deterministic `success` response.
4. A modelling query should return LLM output or fallback guidance if provider is unavailable.

## Known Implementation Notes

- Session state is in-memory only; restarting the service clears all conversations.
- Horizontal scaling would require shared session storage (for example Redis).
- Forecast responses are deterministic from CSV data; no forecast values are fabricated by the LLM.
- Some legacy MCP-related env keys may still exist in `.env.example` but are not currently required for the active OpenRouter flow.
