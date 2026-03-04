"""
By: Arsalaan Ahmad (12-02-2026) last update: 03/03/2026
This file handles:
1. Serving the frontend
2. Exposing an HTTP API endpoint for chat requests
3. Acting as an adapter layer between the frontend and whichever backend provider we use (Mock, MCP, etc.)

note to self:
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

# NEW: deterministic truth layer + temporary extractor
from app.core.forecast_store import ForecastStore
from app.core.extractor_stub import extract_intent_stub


# -------------------------------------------------
# FastAPI App Setup
# -------------------------------------------------

app = FastAPI(title="CUWALID-GPT Web Client")

# Mount static folder (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# OPTIONAL: serve forecast assets (maps/voice) if you have an assets/ folder
# Adjust path to match repo structure.
if os.path.isdir("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

templates = Jinja2Templates(directory="app/templates")

DOCS_URL = "https://cuwalid.github.io/"
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "mock")

# NEW: Forecast store (CSV cache). Adjust data_root to your repo structure.
DATA_ROOT = os.getenv("CUWALID_DATA_ROOT", "data")
store = ForecastStore(data_root=DATA_ROOT)


@app.on_event("startup")
def startup():
    """
    Load all seasonal CSVs into memory so retrieval is fast and deterministic.
    """
    try:
        store.load_all()
    except Exception as e:
        # Do not crash hard in dev; but log so you notice.
        # In production you'd likely want to fail startup if data isn't available.
        print(f"[startup] ForecastStore load failed: {e}")


# -------------------------------------------------
# Helper: slot requirements + simple orchestrator
# -------------------------------------------------

REQUIRED_FIELDS = ["country", "location", "variable", "season", "year", "language"]


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _clarify_question(intent: dict) -> str:
    """
    Ask 1 targeted question based on what's missing.
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
    Deterministic orchestration:
    intent (from extractor/LLM) -> resolve -> lookup -> response contract.
    """
    # 1) Clarify if missing critical slots
    for f in REQUIRED_FIELDS:
        if _is_missing(intent.get(f)):
            return {"kind": "clarify", "reply": _clarify_question(intent)}

    country = intent["country"].strip().lower()
    location = intent["location"].strip()
    variable = intent["variable"].strip().lower()
    season = intent["season"].strip().upper()
    year = int(intent["year"])
    language = intent["language"].strip().lower()

    # 2) Resolve place -> location_id
    location_id = store.resolve_location_id(country, location)

    # If exact match fails, try search suggestions
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

    # 3) Deterministic lookup
    status_code = store.get_prediction(country, season, year, location_id, variable)
    if status_code is None:
        return {
            "kind": "error",
            "reply": "I couldn’t find a forecast entry for that combination (season/year/location/variable)."
        }

    status_label = store.label_status(variable, status_code)
    map_url = store.build_map_path(year, season, location_id, variable, language)

    # 4) Return stable contract (UI can later show attachments)
    # NOTE: flood reversal meaning is handled in label/notes; refine messaging later.
    return {
        "kind": "success",
        "reply": (
            f"{location} ({country.title()}) — {variable.replace('_',' ').title()} for {season} {year}: "
            f"{status_label} (code {status_code})."
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
# Provider Abstraction
# -------------------------------------------------

async def mock_chat(message: str) -> dict:
    """
    Mock provider used for development/testing.

    UPDATED BEHAVIOR (04/03/2026):
    - Instead of random mock replies, we now:
      1) extract structured intent, stub until MCP
      2) deterministically fetch forecast from ForecastStore in core/forecast_store.py
    """
    lower = message.lower()

    # UI-level guardrail example: refuse operational decisions
    if "what should we do" in lower or "tell me what to do" in lower:
        return {
            "kind": "refuse",
            "reply": "I can’t make operational or policy decisions. If you share location + season/year + forecast type, I can summarise the forecast implications and show the relevant map."
        }

    # Temporary extraction (replace later with MCP LLM extractor)
    intent = extract_intent_stub(message)

    # NOTE: extractor_stub may not fill location/season/year yet; clarifications will trigger.
    return _handle_intent(intent)


async def mcp_chat(message: str) -> dict:
    """
    Placeholder for MCP integration. 

    Future behavior:
    - Call MCP/LLM extractor -> structured intent JSON
    - Run _handle_intent(intent) to fetch deterministic results
    - Optionally call MCP again to generate nicer NLG response using fetched meta
    """
    raise RuntimeError("MCP provider not yet configured.")


async def chat_provider(message: str) -> dict:
    """
    Unified entrypoint used by the API.
    Frontend always calls this route.
    """
    if CHAT_PROVIDER == "mcp":
        return await mcp_chat(message)
    return await mock_chat(message)


# -------------------------------------------------
# Frontend Route
# -------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "docs_url": DOCS_URL})


# -------------------------------------------------
# API Contract
# -------------------------------------------------

class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(payload: ChatRequest):
    """
    Main API endpoint called by frontend JS.
    """
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = await chat_provider(message)
        return result
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"kind": "error", "reply": "Backend unavailable or misconfigured."}
        )


# -------------------------------------------------
# Debug / Health Endpoints
# -------------------------------------------------

@app.get("/status")
def status():
    return {
        "chat_provider": CHAT_PROVIDER,
        "mcp_ready": CHAT_PROVIDER == "mcp",
        "data_root": DATA_ROOT,
        "countries_loaded": sorted(store.supported_countries),
        "variables_loaded": sorted(store.supported_variables),
        "seasons_loaded": sorted([f"{s}-{y}" for (s, y) in store.supported_seasons]),
    }