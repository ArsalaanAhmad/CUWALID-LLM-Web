from __future__ import annotations

import logging
import time
from typing import Any

from app.core.extractor_stub import extract_intent_stub
from app.core.forecast_store import ForecastStore
from app.core.guardrails import (
    is_nonsense_or_empty,
    is_out_of_scope,
    is_policy_request,
    nonsense_response,
    out_of_scope_response,
    refusal_response,
)
from app.core.openrouter_extractor import extract_intent_via_openrouter
from app.core.session_store import SessionStore

REQUIRED_FIELDS = ["country", "location", "variable", "season", "year", "language"]
SUPPORTED_MODES = {"forecast", "modelling"}
SUPPORTED_TIERS = {"general-public", "practitioners", "policy-makers"}

logger = logging.getLogger("cuwalid.chat_service")


def merge_intents(old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """
    Merge old and new extracted intent fields, preferring non-empty values from `new`.
    """
    merged = dict(old or {})
    for key, value in (new or {}).items():
        if value is not None and not (isinstance(value, str) and not value.strip()):
            merged[key] = value
    return merged


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def get_missing_field(intent: dict) -> str | None:
    """
    Return the first missing required field from the standard forecast intent schema.
    """
    for field in REQUIRED_FIELDS:
        if _is_missing(intent.get(field)):
            return field
    return None


def clarification_response(intent: dict) -> dict:
    """
    Return a structured clarification response for missing required fields.
    """
    missing_field = get_missing_field(intent)

    if missing_field == "country":
        reply = "Which country is this for? (Kenya / Ethiopia / Somalia)"
    elif missing_field == "location":
        reply = "Which region or district should I use? (e.g., Marsabit, Garissa)"
    elif missing_field == "variable":
        reply = "Which forecast type do you want? (crop, pasture, surface_water, groundwater, flood)"
    elif missing_field in {"season", "year"}:
        reply = "Which season and year? (e.g., OND 2026, MAM 2026)"
    elif missing_field == "language":
        reply = "Which language should I respond in? (en/sw/so/am/or)"
    else:
        reply = "Please provide more details such as location, forecast type, and time period."

    reason_suffix = missing_field if missing_field else "details"
    return {
        "kind": "clarify",
        "reply": reply,
        "meta": {
            "reason": f"missing_{reason_suffix}",
        },
    }


def format_forecast_response(
    *,
    tier: str,
    country: str,
    location: str,
    variable: str,
    season: str,
    year: int,
    status_label: str,
    status_code: int,
) -> str:
    """
    Format deterministic forecast outputs by audience tier.
    """
    variable_title = variable.replace("_", " ")
    place = f"{location} ({country.title()})"

    if tier == "policy-makers":
        return (
            f"{place}: {variable_title.title()} outlook for {season} {year} is {status_label}. "
            f"This indicates elevated likelihood relative to climatology, not certainty. "
            f"Use with other evidence as conditions may evolve."
        )

    if tier == "practitioners":
        return (
            f"{place} - {variable_title.title()} for {season} {year}: {status_label} (code {status_code}). "
            f"This is a probabilistic seasonal signal and should be combined with local monitoring."
        )

    return (
        f"For {place}, the {variable_title} forecast for {season} {year} is {status_label}. "
        f"This is a seasonal outlook, so real conditions can still vary."
    )


class ChatService:
    """
    Keeps chat orchestration logic out of FastAPI route wiring.
    """

    def __init__(
        self,
        *,
        store: ForecastStore,
        session_store: SessionStore,
        docs_url: str,
        chat_provider: str,
    ) -> None:
        self.store = store
        self.session_store = session_store
        self.docs_url = docs_url
        self.chat_provider = chat_provider

    async def route_chat(self, *, message: str, mode: str, tier: str, conversation_id: str) -> dict[str, Any]:
        if mode == "forecast":
            return await self.handle_forecast_chat(message=message, tier=tier, conversation_id=conversation_id)
        return await self.handle_modelling_chat(message=message, tier=tier, conversation_id=conversation_id)

    async def _extract_intent(self, message: str, conversation_id: str) -> dict[str, Any]:
        if self.chat_provider == "openrouter":
            start = time.perf_counter()
            intent = await extract_intent_via_openrouter(message)
            logger.debug(
                "extractor_openrouter_complete conversation_id=%s duration_ms=%.2f",
                conversation_id,
                (time.perf_counter() - start) * 1000,
            )
            return intent

        start = time.perf_counter()
        intent = extract_intent_stub(message)
        logger.debug(
            "extractor_mock_complete conversation_id=%s duration_ms=%.2f",
            conversation_id,
            (time.perf_counter() - start) * 1000,
        )
        return intent

    def _handle_intent(self, intent: dict[str, Any], conversation_id: str) -> dict[str, Any]:
        missing_field = get_missing_field(intent)
        if missing_field:
            logger.info(
                "forecast_clarify_missing_field conversation_id=%s field=%s",
                conversation_id,
                missing_field,
            )
            return clarification_response(intent)

        try:
            country = intent["country"].strip().lower()
            location = intent["location"].strip()
            variable = intent["variable"].strip().lower()
            season = intent["season"].strip().upper()
            year = int(intent["year"])
            language = intent["language"].strip().lower()
        except Exception:
            logger.warning("forecast_intent_parse_failed conversation_id=%s intent=%s", conversation_id, intent)
            return {
                "kind": "error",
                "reply": "I couldn’t parse the extracted request fields correctly.",
            }

        location_id = self.store.resolve_location_id(country, location)
        if not location_id:
            suggestions = self.store.search_locations(country=country, q=location, limit=5)
            if suggestions:
                options = ", ".join([f"{name} ({loc_id})" for (name, loc_id) in suggestions])
                logger.info(
                    "forecast_unknown_location_with_suggestions conversation_id=%s country=%s location=%s suggestions=%s",
                    conversation_id,
                    country,
                    location,
                    len(suggestions),
                )
                return {
                    "kind": "clarify",
                    "reply": (
                        f"I couldn’t find an exact match for '{location}'. Did you mean: {options} ? "
                        "Reply with the correct place name."
                    ),
                }

            logger.info(
                "forecast_unknown_location conversation_id=%s country=%s location=%s",
                conversation_id,
                country,
                location,
            )
            return {
                "kind": "clarify",
                "reply": (
                    f"I couldn’t find '{location}' in {country.title()}. "
                    "Try a nearby district/county name or a different spelling."
                ),
                "meta": {
                    "reason": "unknown_location",
                    "country": country,
                    "location": location,
                },
            }

        status_code = self.store.get_prediction(country, season, year, location_id, variable)
        if status_code is None:
            logger.info(
                "forecast_missing_entry conversation_id=%s country=%s location_id=%s variable=%s season=%s year=%s",
                conversation_id,
                country,
                location_id,
                variable,
                season,
                year,
            )
            return {
                "kind": "clarify",
                "reply": (
                    "I found the location, but there is no forecast entry for that variable, season, "
                    "and year combination. Please try a different variable or season/year."
                ),
                "meta": {
                    "reason": "known_location_missing_forecast_entry",
                    "country": country,
                    "location": location,
                    "location_id": location_id,
                    "variable": variable,
                    "season": season,
                    "year": year,
                },
            }

        status_label = self.store.interpret_status(variable, status_code)
        map_url = self.store.build_map_path(year, season, location_id, variable, language)
        logger.info(
            "forecast_lookup_success conversation_id=%s country=%s location_id=%s variable=%s season=%s year=%s status_code=%s",
            conversation_id,
            country,
            location_id,
            variable,
            season,
            year,
            status_code,
        )

        return {
            "kind": "success",
            "reply": (
                f"{location} ({country.title()}) - {variable.replace('_', ' ').title()} "
                f"for {season} {year}: {status_label} (code {status_code})."
            ),
            "attachments": [{"type": "map", "url": map_url}],
            "meta": {
                "country": country,
                "season": season,
                "year": year,
                "location": location,
                "location_id": location_id,
                "variable": variable,
                "language": language,
                "flood_reversal_applies": (variable == "flood"),
            },
        }

    async def handle_forecast_chat(self, *, message: str, tier: str, conversation_id: str) -> dict[str, Any]:
        stage_start = time.perf_counter()
        state = self.session_store.get(conversation_id)
        logger.info(
            "forecast_chat_start conversation_id=%s tier=%s history_len=%s",
            conversation_id,
            tier,
            len(state.history),
        )

        self.session_store.update(conversation_id, mode="forecast", tier=tier)
        self.session_store.update(conversation_id, append_history={"role": "user", "text": message})

        if is_nonsense_or_empty(message):
            logger.info("forecast_insufficient_input conversation_id=%s", conversation_id)
            result = nonsense_response()
            result.setdefault("meta", {})
            result["meta"]["mode"] = "forecast"
            result["meta"]["tier"] = tier
            self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})
            return result

        if is_policy_request(message):
            logger.info("forecast_policy_refusal conversation_id=%s", conversation_id)
            result = refusal_response()
            result.setdefault("meta", {})
            result["meta"]["mode"] = "forecast"
            result["meta"]["tier"] = tier
            self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})
            return result

        if is_out_of_scope(message):
            logger.info("forecast_out_of_scope conversation_id=%s", conversation_id)
            result = out_of_scope_response()
            result.setdefault("meta", {})
            result["meta"]["mode"] = "forecast"
            result["meta"]["tier"] = tier
            self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})
            return result

        new_intent = await self._extract_intent(message, conversation_id)
        merged_intent = merge_intents(state.intent, new_intent)
        logger.debug(
            "forecast_intent_merged conversation_id=%s old_keys=%s new_keys=%s merged_keys=%s",
            conversation_id,
            sorted(list((state.intent or {}).keys())),
            sorted(list((new_intent or {}).keys())),
            sorted(list((merged_intent or {}).keys())),
        )
        self.session_store.update(conversation_id, intent=merged_intent)

        result = self._handle_intent(merged_intent, conversation_id)
        self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})

        if result.get("kind") == "success":
            meta = result.get("meta") or {}
            location = str(meta.get("location", "")).strip()
            country = str(meta.get("country", "")).strip()
            variable = str(meta.get("variable", "")).strip()
            season = str(meta.get("season", "")).strip()
            year = int(meta.get("year"))
            location_id = str(meta.get("location_id", "")).strip()
            language = str(meta.get("language", "")).strip().lower()

            status_code = self.store.get_prediction(country, season, year, location_id, variable)
            if status_code is None:
                result = {
                    "kind": "clarify",
                    "reply": (
                        "I found the location, but there is no forecast entry for that variable, season, "
                        "and year combination. Please try a different variable or season/year."
                    ),
                    "meta": {
                        "reason": "known_location_missing_forecast_entry",
                        "country": country,
                        "location": location,
                        "location_id": location_id,
                        "variable": variable,
                        "season": season,
                        "year": year,
                        "mode": "forecast",
                        "tier": tier,
                    },
                }
                self.session_store.update(
                    conversation_id,
                    append_history={"role": "assistant", "text": result["reply"]},
                )
                return result

            status_label = self.store.interpret_status(variable, status_code)
            result["reply"] = format_forecast_response(
                tier=tier,
                country=country,
                location=location,
                variable=variable,
                season=season,
                year=year,
                status_label=status_label,
                status_code=status_code,
            )

            result.setdefault("meta", {})
            result["meta"]["mode"] = "forecast"
            result["meta"]["tier"] = tier
            result["meta"]["language"] = language
            self.session_store.clear_intent(conversation_id)

        elif result.get("kind") == "clarify":
            result.setdefault("meta", {})
            result["meta"].setdefault("reason", "missing_required_fields")
            result["meta"]["mode"] = "forecast"
            result["meta"]["tier"] = tier

        logger.info(
            "forecast_chat_complete conversation_id=%s kind=%s duration_ms=%.2f",
            conversation_id,
            result.get("kind"),
            (time.perf_counter() - stage_start) * 1000,
        )
        return result

    async def handle_modelling_chat(self, *, message: str, tier: str, conversation_id: str) -> dict[str, Any]:
        logger.info(
            "modelling_chat_stub conversation_id=%s tier=%s message_len=%s",
            conversation_id,
            tier,
            len((message or "").strip()),
        )
        return {
            "kind": "success",
            "reply": (
                "Modelling mode is available as a placeholder. "
                "Grounded modelling documents and workflow execution are not yet connected in this build."
            ),
            "attachments": [{"type": "link", "url": self.docs_url, "label": "CUWALID documentation"}],
            "meta": {
                "mode": "modelling",
                "tier": tier,
                "status": "stub",
                "message_preview": (message or "").strip()[:120],
            },
        }
