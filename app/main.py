"""
FastAPI app wiring for CUWALID-GPT.

This module intentionally stays thin:
- app/bootstrap setup
- request middleware
- route handlers
- dependency wiring for chat service
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.core.chat_service import ChatService, SUPPORTED_MODES, SUPPORTED_TIERS
from app.core.forecast_store import ForecastStore
from app.core.session_store import SessionStore

# Load environment variables from .env file
load_dotenv()


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger = logging.getLogger("cuwalid.main")


app = FastAPI(title="CUWALID-GPT Web Client")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

if os.path.isdir("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

templates = Jinja2Templates(directory="app/templates")

DOCS_URL = "https://cuwalid.github.io/"
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "mock").strip().lower()
DATA_ROOT = os.getenv("CUWALID_DATA_ROOT", "data")

store = ForecastStore(data_root=DATA_ROOT)
session_store = SessionStore()
chat_service = ChatService(
    store=store,
    session_store=session_store,
    docs_url=DOCS_URL,
    chat_provider=CHAT_PROVIDER,
)


@app.on_event("startup")
def startup() -> None:
    """Load forecast CSV data at startup for deterministic lookups."""
    try:
        store.load_all()
        logger.info(
            "startup_complete provider=%s data_root=%s countries=%s variables=%s seasons=%s",
            CHAT_PROVIDER,
            DATA_ROOT,
            len(store.supported_countries),
            len(store.supported_variables),
            len(store.supported_seasons),
        )
    except Exception as exc:
        logger.exception("startup_forecaststore_load_failed error=%s", exc)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Add request correlation/timing logs and return X-Request-ID header."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "http_request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve the landing/chat UI."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "docs_url": DOCS_URL,
        },
    )


class ChatRequest(BaseModel):
    """Input contract for frontend chat requests."""

    message: str
    mode: Literal["forecast", "modelling"] = "forecast"
    tier: Literal["general-public", "practitioners", "policy-makers"] = "general-public"
    conversation_id: str = "default"


@app.post("/api/chat")
async def chat(payload: ChatRequest):
    """Main chat endpoint with mode/tier routing and conversation context."""
    message = payload.message.strip()
    mode = payload.mode.strip().lower()
    tier = payload.tier.strip().lower()
    conversation_id = payload.conversation_id.strip()

    logger.info(
        "chat_request_received conversation_id=%s mode=%s tier=%s message_len=%s",
        conversation_id,
        mode,
        tier,
        len(message),
    )

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if mode not in SUPPORTED_MODES:
        raise HTTPException(status_code=400, detail="Unsupported mode.")
    if tier not in SUPPORTED_TIERS:
        raise HTTPException(status_code=400, detail="Unsupported tier.")
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id required.")

    try:
        result = await chat_service.route_chat(
            message=message,
            mode=mode,
            tier=tier,
            conversation_id=conversation_id,
        )
        logger.info(
            "chat_request_complete conversation_id=%s mode=%s tier=%s kind=%s",
            conversation_id,
            mode,
            tier,
            result.get("kind"),
        )
        return result
    except Exception as exc:
        logger.exception(
            "chat_request_failed conversation_id=%s mode=%s tier=%s error=%s",
            conversation_id,
            mode,
            tier,
            exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "kind": "error",
                "reply": "Backend unavailable or misconfigured.",
            },
        )


@app.get("/status")
def status():
    """Health/debug endpoint for provider and loaded deterministic forecast metadata."""
    return {
        "chat_provider": CHAT_PROVIDER,
        "mcp_ready": CHAT_PROVIDER == "mcp",
        "data_root": DATA_ROOT,
        "countries_loaded": sorted(store.supported_countries),
        "variables_loaded": sorted(store.supported_variables),
        "seasons_loaded": sorted([f"{season}-{year}" for (season, year) in store.supported_seasons]),
    }
