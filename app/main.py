"""
By: Arsalaan Ahmad (12-02-2026) last update: 04/03/2026

This file handles:
1. Serving the frontend
2. Exposing an HTTP API endpoint for chat requests
3. Acting as an adapter layer between the frontend and whichever backend provider we use (Mock, MCP, etc.)

Note to self:
- This file does NOT contain scientific logic.
- It does NOT modify CUWALID outputs.
- It ONLY adapts requests and responses between UI and backend.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os

# Deterministic truth layer + extractors
from app.core.forecast_store import ForecastStore
from app.core.extractor_stub import extract_intent_stub
# from app.core.mcp_extractor import extract_intent_via_mcp

# OpeNRouter extractor (currently active)
from app.core.openrouter_extractor import extract_intent_via_openrouter

# -------------------------------------------------
# FastAPI App Setup
# -------------------------------------------------

app = FastAPI(title="CUWALID-GPT Web Client")

# Mount frontend static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Optional: serve forecast assets (maps / voice) if present
if os.path.isdir("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

templates = Jinja2Templates(directory="app/templates")

DOCS_URL = "https://cuwalid.github.io/"

# Provider mode:
# - mock = uses local extractor stub
# - mcp  = uses MCP/OpenRouter extraction
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "mock").strip().lower()

# Forecast data root
DATA_ROOT = os.getenv("CUWALID_DATA_ROOT", "data")

# In-memory deterministic forecast store
store = ForecastStore(data_root=DATA_ROOT)


@app.on_event("startup")
def startup():
    """
    Load all seasonal forecast CSVs into memory on startup.
    This makes forecast retrieval deterministic and fast.
    """
    try:
        store.load_all()
    except Exception as e:
        # In dev we don't want startup to hard-fail immediately.
        # In production, you may want to fail loudly instead.
        print(f"[startup] ForecastStore load failed: {e}")


# -------------------------------------------------
# Helper: slot requirements + deterministic orchestration
# -------------------------------------------------

REQUIRED_FIELDS = ["country", "location", "variable", "season", "year", "language"]


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _clarify_question(intent: dict) -> str:
    """
    Ask one targeted clarification question based on missing fields.
    """
    if _is_missing(intent.get("country")):
        return "Which country is this for? (Kenya / Ethiopia / Somalia)"
    if _is_missing(intent.get("location")):
        return "Which region/county/district should I use? (e.g., Marsabit, Garissa)"
    if _is_missing(intent.get("variable")):
        return "Which forecast type do you want? (crop, pasture, surface_water, groundwater, flood)"
    if _is_missing(intent.get("season")) or _is_missing(intent.get("year")):
        return "Which season and year? (e.g., OND 2026, MAM 2026, JJAS 2026)"
    if _is_missing(intent.get("language")):
        return "Which language should I respond in? (en/sw/so/am/or)"
    return "Could you clarify your request?"


def _handle_intent(intent: dict) -> dict:
    """
    Deterministic orchestration pipeline:

    intent JSON (from extractor) ->
    validate required fields ->
    resolve place -> location_id ->
    lookup forecast ->
    build response contract for frontend
    """
    # 1) Clarify if required fields are missing
    for f in REQUIRED_FIELDS:
        if _is_missing(intent.get(f)):
            return {
                "kind": "clarify",
                "reply": _clarify_question(intent)
            }

    try:
        country = intent["country"].strip().lower()
        location = intent["location"].strip()
        variable = intent["variable"].strip().lower()
        season = intent["season"].strip().upper()
        year = int(intent["year"])
        language = intent["language"].strip().lower()
    except Exception:
        return {
            "kind": "error",
            "reply": "I couldn’t parse the extracted request fields correctly."
        }

    # 2) Resolve place -> location_id
    location_id = store.resolve_location_id(country, location)

    # If exact match fails, offer suggestions
    if not location_id:
        suggestions = store.search_locations(country=country, q=location, limit=5)
        if suggestions:
            opts = ", ".join([f"{p} ({loc_id})" for (p, loc_id) in suggestions])
            return {
                "kind": "clarify",
                "reply": f"I couldn’t find an exact match for '{location}'. Did you mean: {opts} ? Reply with the correct place name."
            }

        return {
            "kind": "clarify",
            "reply": f"I couldn’t find '{location}' in {country.title()}. Try a nearby district/county name or a different spelling."
        }

    # 3) Deterministic prediction lookup
    status_code = store.get_prediction(country, season, year, location_id, variable)
    if status_code is None:
        return {
            "kind": "error",
            "reply": "I couldn’t find a forecast entry for that combination (season/year/location/variable)."
        }

    # 4) Human-readable label + asset path
    status_label = store.label_status(variable, status_code)
    map_url = store.build_map_path(year, season, location_id, variable, language)

    # 5) Return stable contract for frontend
    return {
        "kind": "success",
        "reply": (
            f"{location} ({country.title()}) — {variable.replace('_', ' ').title()} "
            f"for {season} {year}: {status_label} (code {status_code})."
        ),
        "attachments": [
            {"type": "map", "url": map_url}
        ],
        "meta": {
            "country": country,
            "season": season,
            "year": year,
            "location": location,
            "location_id": location_id,
            "variable": variable,
            "language": language,
            "flood_reversal_applies": (variable == "flood")
        }
    }


# -------------------------------------------------
# Provider abstraction
# -------------------------------------------------

async def mock_chat(message: str) -> dict:
    """
    Development mode:
    - uses local extractor_stub
    - then routes into the deterministic truth layer
    """
    lower = message.lower()

    # Example hard refusal at interface level
    if "what should we do" in lower or "tell me what to do" in lower:
        return {
            "kind": "refuse",
            "reply": "I can’t make operational or policy decisions. If you share location, season/year, and forecast type, I can summarise the forecast implications and show the relevant map."
        }

    intent = extract_intent_stub(message)
    return _handle_intent(intent)

async def openrouter_chat(message: str) -> dict:
    """
    Real OpenRouter-backed extraction:
    - sends user text to Qwen
    - receives structured JSON
    - passes it into the deterministic truth layer
    """
    lower = message.lower()

    if "what should we do" in lower or "tell me what to do" in lower:
        return {
            "kind": "refuse",
            "reply": "I can’t make operational or policy decisions. If you share location, season/year, and forecast type, I can summarise the forecast implications and show the relevant map."
        }

    intent = await extract_intent_via_openrouter(message)
    return _handle_intent(intent)

# Below Section currently INACTIVE.
""""" 
async def mcp_chat(message: str) -> dict:

    
    # Real MCP-backed extraction:
   # - sends user text to MCP/OpenRouter
   # - receives structured JSON
   # - passes it into the deterministic truth layer
    
    lower = message.lower()

    # Keep the same hard refusal logic here too
    if "what should we do" in lower or "tell me what to do" in lower:
        return {
            "kind": "refuse",
            "reply": "I can’t make operational or policy decisions. If you share location, season/year, and forecast type, I can summarise the forecast implications and show the relevant map."
        }

    intent = await extract_intent_via_mcp(message)
    return _handle_intent(intent)
"""

async def chat_provider(message: str) -> dict:
    if CHAT_PROVIDER == "openrouter":
        return await openrouter_chat(message)
    return await mock_chat(message)

# -------------------------------------------------
# Frontend Route
# -------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Serves the landing page / chat UI.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "docs_url": DOCS_URL
        }
    )


# -------------------------------------------------
# API Contract
# -------------------------------------------------

class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(payload: ChatRequest):
    """
    Main frontend chat endpoint.

    Flow:
    1. Receive free-text user prompt
    2. Route to provider (mock or MCP)
    3. Extract structured intent
    4. Perform deterministic forecast lookup
    5. Return stable JSON contract to frontend
    """
    message = payload.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = await chat_provider(message)
        return result
    except Exception as e:
        print(f"[chat] Error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "kind": "error",
                "reply": "Backend unavailable or misconfigured."
            }
        )


# -------------------------------------------------
# Debug / Health Endpoints
# -------------------------------------------------

@app.get("/status")
def status():
    """
    Lightweight health/debug endpoint.
    Useful for confirming:
    - current provider mode
    - whether data loaded
    - which countries/seasons/variables are available
    """
    return {
        "chat_provider": CHAT_PROVIDER,
        "mcp_ready": CHAT_PROVIDER == "mcp",
        "data_root": DATA_ROOT,
        "countries_loaded": sorted(store.supported_countries),
        "variables_loaded": sorted(store.supported_variables),
        "seasons_loaded": sorted([f"{s}-{y}" for (s, y) in store.supported_seasons]),
    }

