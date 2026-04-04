import logging
import time
import re
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.core.config import settings
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

# Load environment variables
load_dotenv()

REQUIRED_FIELDS = ["country", "location", "variable", "season", "year", "language"]
SUPPORTED_MODES = {"forecast", "modelling"}
SUPPORTED_TIERS = {"general-public", "practitioners", "policy-makers"}

LANGUAGE_SYNONYMS = {
    "en": "en",
    "english": "en",
    "eng": "en",
    "sw": "sw",
    "swahili": "sw",
    "kiswahili": "sw",
    "so": "so",
    "somali": "so",
    "am": "am",
    "amharic": "am",
    "or": "or",
    "oromo": "or",
}

VARIABLE_SYNONYMS = {
    "surface water": "surface_water",
    "surface-water": "surface_water",
    "surfacewater": "surface_water",
    "surface_water": "surface_water",
    "ground water": "groundwater",
    "ground-water": "groundwater",
    "groundwater": "groundwater",
    "crop": "crop",
    "crops": "crop",
    "pasture": "pasture",
    "pastures": "pasture",
    "flood": "flood",
    "flooding": "flood",
}

SEASON_SYNONYMS = {
    "ond": "OND",
    "mam": "MAM",
    "jjas": "JJAS",
    "jja": "JJA",
    "son": "SON",
    "djf": "DJF",
}

logger = logging.getLogger("cuwalid.chat_service")

# Load expert.prompt for modelling chat
EXPERT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "expert.prompt"
EXPERT_SYSTEM_PROMPT = ""
if EXPERT_PROMPT_PATH.exists():
    try:
        EXPERT_SYSTEM_PROMPT = EXPERT_PROMPT_PATH.read_text()
        logger.info("expert_prompt_loaded path=%s len=%s", EXPERT_PROMPT_PATH, len(EXPERT_SYSTEM_PROMPT))
    except Exception as exc:
        logger.warning("expert_prompt_load_failed path=%s error=%s", EXPERT_PROMPT_PATH, exc)
else:
    logger.warning("expert_prompt_not_found path=%s", EXPERT_PROMPT_PATH)


def merge_intents(old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """
    Merge old and new extracted intent fields, preferring non-empty values from `new`.
    """
    merged = dict(old or {})
    for key, value in (new or {}).items():
        if value is not None and not (isinstance(value, str) and not value.strip()):
            merged[key] = value
    return merged


async def _call_openrouter_modelling(message: str) -> str:
    """
    Call OpenRouter with modelling/help prompt and return text response.
    Returns empty string if no response or on error.
    """
    if not settings.openrouter_api_key:
        logger.error("openrouter_api_key_not_set")
        return ""

    if not EXPERT_SYSTEM_PROMPT:
        logger.error("expert_system_prompt_empty")
        return ""

    try:
        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.exception("openrouter_modelling_failed error=%s", exc)
        return ""


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _normalize_spaces_hyphens_underscores(value: str) -> str:
    cleaned = re.sub(r"[\s_-]+", " ", value.strip())
    return " ".join(cleaned.split())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _best_close_matches(value: str, choices: list[str], cutoff: float = 0.78) -> list[str]:
    if not value or not choices:
        return []
    return get_close_matches(value, choices, n=2, cutoff=cutoff)


def _safe_autocorrect(value: str, choices: list[str], *, auto_cutoff: float = 0.9, suggest_cutoff: float = 0.78) -> tuple[str, str | None, bool]:
    """
    Return (resolved_value, suggestion_if_any, was_auto_corrected).
    Auto-correct only for high-confidence and non-ambiguous matches.
    """
    normalized = value.strip()
    if not normalized:
        return normalized, None, False

    lowered_choices = {c.lower(): c for c in choices}
    if normalized.lower() in lowered_choices:
        return lowered_choices[normalized.lower()], None, False

    matches = _best_close_matches(normalized.lower(), list(lowered_choices.keys()), cutoff=suggest_cutoff)
    if not matches:
        return normalized, None, False

    best = matches[0]
    best_ratio = _similarity(normalized.lower(), best)
    second_ratio = _similarity(normalized.lower(), matches[1]) if len(matches) > 1 else 0.0
    best_value = lowered_choices[best]

    if best_ratio >= auto_cutoff and (best_ratio - second_ratio) >= 0.06:
        return best_value, None, True

    return normalized, best_value, False


def _normalize_language(value: Any) -> tuple[str | None, str | None, bool]:
    if _is_missing(value):
        return None, None, False
    raw = _normalize_spaces_hyphens_underscores(str(value)).lower()
    canonical = LANGUAGE_SYNONYMS.get(raw)
    if canonical:
        return canonical, None, canonical != raw

    known_languages = sorted(set(LANGUAGE_SYNONYMS.values()))
    return _safe_autocorrect(raw, known_languages, auto_cutoff=0.92, suggest_cutoff=0.7)


def _normalize_variable(value: Any) -> tuple[str | None, str | None, bool]:
    if _is_missing(value):
        return None, None, False
    raw = _normalize_spaces_hyphens_underscores(str(value)).lower()
    canonical = VARIABLE_SYNONYMS.get(raw)
    if canonical:
        return canonical, None, canonical != raw

    known_variables = sorted(set(VARIABLE_SYNONYMS.values()))
    return _safe_autocorrect(raw.replace(" ", "_"), known_variables, auto_cutoff=0.9, suggest_cutoff=0.74)


def _normalize_season(value: Any) -> tuple[str | None, str | None, bool]:
    if _is_missing(value):
        return None, None, False
    raw = _normalize_spaces_hyphens_underscores(str(value)).lower().replace(" ", "")
    canonical = SEASON_SYNONYMS.get(raw)
    if canonical:
        return canonical, None, canonical != raw

    known_seasons = sorted(SEASON_SYNONYMS.values())
    return _safe_autocorrect(raw.upper(), known_seasons, auto_cutoff=0.95, suggest_cutoff=0.7)


def _normalize_country(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return _normalize_spaces_hyphens_underscores(str(value)).lower()


def _normalize_location(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return _normalize_spaces_hyphens_underscores(str(value))


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
        normalized_intent = dict(intent or {})

        normalized_intent["country"] = _normalize_country(normalized_intent.get("country"))
        normalized_intent["location"] = _normalize_location(normalized_intent.get("location"))

        variable, variable_suggestion, variable_autocorrected = _normalize_variable(normalized_intent.get("variable"))
        normalized_intent["variable"] = variable

        season, season_suggestion, season_autocorrected = _normalize_season(normalized_intent.get("season"))
        normalized_intent["season"] = season

        language, language_suggestion, language_autocorrected = _normalize_language(normalized_intent.get("language"))
        normalized_intent["language"] = language

        country = normalized_intent.get("country")
        if country and self.store.supported_countries:
            resolved_country, country_suggestion, country_autocorrected = _safe_autocorrect(
                country,
                sorted(self.store.supported_countries),
                auto_cutoff=0.92,
                suggest_cutoff=0.78,
            )
            normalized_intent["country"] = resolved_country
        else:
            country_suggestion = None
            country_autocorrected = False

        missing_field = get_missing_field(normalized_intent)
        if missing_field:
            logger.info(
                "forecast_clarify_missing_field conversation_id=%s field=%s",
                conversation_id,
                missing_field,
            )
            suggestion_bits: list[str] = []
            if variable_suggestion:
                suggestion_bits.append(f"variable '{normalized_intent.get('variable')}' maybe '{variable_suggestion}'")
            if season_suggestion:
                suggestion_bits.append(f"season '{normalized_intent.get('season')}' maybe '{season_suggestion}'")
            if language_suggestion:
                suggestion_bits.append(f"language '{normalized_intent.get('language')}' maybe '{language_suggestion}'")
            if country_suggestion:
                suggestion_bits.append(f"country '{normalized_intent.get('country')}' maybe '{country_suggestion}'")

            response = clarification_response(normalized_intent)
            if suggestion_bits:
                response["reply"] = f"{response['reply']} Did you mean: {', '.join(suggestion_bits)}?"
                response.setdefault("meta", {})
                response["meta"]["did_you_mean"] = suggestion_bits
            return response

        try:
            country = normalized_intent["country"].strip().lower()
            location = normalized_intent["location"].strip()
            variable = normalized_intent["variable"].strip().lower()
            season = normalized_intent["season"].strip().upper()
            year = int(normalized_intent["year"])
            language = normalized_intent["language"].strip().lower()
        except Exception:
            logger.warning("forecast_intent_parse_failed conversation_id=%s intent=%s", conversation_id, intent)
            return {
                "kind": "error",
                "reply": "I couldn’t parse the extracted request fields correctly.",
            }

        did_you_mean: list[str] = []
        if country_suggestion and not country_autocorrected:
            did_you_mean.append(f"country '{country_suggestion}'")
        if variable_suggestion and not variable_autocorrected:
            did_you_mean.append(f"variable '{variable_suggestion}'")
        if season_suggestion and not season_autocorrected:
            did_you_mean.append(f"season '{season_suggestion}'")
        if language_suggestion and not language_autocorrected:
            did_you_mean.append(f"language '{language_suggestion}'")

        if did_you_mean:
            return {
                "kind": "clarify",
                "reply": (
                    "I found close matches, but I want to avoid guessing. "
                    f"Did you mean: {', '.join(did_you_mean)}?"
                ),
                "meta": {
                    "reason": "ambiguous_close_match",
                    "did_you_mean": did_you_mean,
                },
            }

        meta_updates: dict[str, Any] = {}
        if variable_autocorrected:
            meta_updates["normalized_variable"] = variable
        if season_autocorrected:
            meta_updates["normalized_season"] = season
        if language_autocorrected:
            meta_updates["normalized_language"] = language
        if country_autocorrected:
            meta_updates["normalized_country"] = country

        location_id = self.store.resolve_location_id(country, location)
        if not location_id:
            country_places = list(self.store.place_to_id.get(country, {}).keys())
            close_locations = _best_close_matches(location.lower(), country_places, cutoff=0.82)
            if close_locations:
                candidate_norm = close_locations[0]
                candidate_id = self.store.place_to_id.get(country, {}).get(candidate_norm)
                if candidate_id:
                    score_best = _similarity(location.lower(), candidate_norm)
                    score_second = _similarity(location.lower(), close_locations[1]) if len(close_locations) > 1 else 0.0
                    if score_best >= 0.93 and (score_best - score_second) >= 0.06:
                        location_id = candidate_id
                        location = self.store.id_to_place.get(country, {}).get(candidate_id, location)
                        meta_updates["normalized_location"] = location

        if not location_id:
            suggestions = self.store.search_locations(country=country, q=location, limit=5)
            fuzzy_suggestions = _best_close_matches(location.lower(), list(self.store.place_to_id.get(country, {}).keys()), cutoff=0.76)
            for place_norm in fuzzy_suggestions:
                loc_id = self.store.place_to_id.get(country, {}).get(place_norm)
                if not loc_id:
                    continue
                human_place = self.store.id_to_place.get(country, {}).get(loc_id, place_norm)
                candidate = (human_place, loc_id)
                if candidate not in suggestions:
                    suggestions.append(candidate)
                if len(suggestions) >= 5:
                    break

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
                **meta_updates,
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
        stage_start = time.perf_counter()
        state = self.session_store.get(conversation_id)
        logger.info(
            "modelling_chat_start conversation_id=%s tier=%s history_len=%s",
            conversation_id,
            tier,
            len(state.history),
        )

        self.session_store.update(conversation_id, mode="modelling", tier=tier)
        self.session_store.update(conversation_id, append_history={"role": "user", "text": message})

        if is_nonsense_or_empty(message):
            logger.info("modelling_insufficient_input conversation_id=%s", conversation_id)
            result = nonsense_response()
            result.setdefault("meta", {})
            result["meta"]["mode"] = "modelling"
            result["meta"]["tier"] = tier
            self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})
            return result

        if is_policy_request(message):
            logger.info("modelling_policy_refusal conversation_id=%s", conversation_id)
            result = refusal_response()
            result.setdefault("meta", {})
            result["meta"]["mode"] = "modelling"
            result["meta"]["tier"] = tier
            self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})
            return result

        if is_out_of_scope(message):
            logger.info("modelling_out_of_scope conversation_id=%s", conversation_id)
            result = out_of_scope_response()
            result.setdefault("meta", {})
            result["meta"]["mode"] = "modelling"
            result["meta"]["tier"] = tier
            self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})
            return result

        if self.chat_provider != "openrouter":
            logger.error(
                "modelling_provider_not_configured conversation_id=%s provider=%s",
                conversation_id,
                self.chat_provider,
            )
            result = {
                "kind": "error",
                "reply": "Modelling chat requires CHAT_PROVIDER=openrouter to be configured.",
                "meta": {
                    "mode": "modelling",
                    "tier": tier,
                    "reason": "provider_not_configured",
                    "provider": self.chat_provider,
                },
            }
            self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})
            return result

        reply_text = await _call_openrouter_modelling(message)
        if not reply_text:
            logger.error("modelling_openrouter_failed conversation_id=%s", conversation_id)
            result = {
                "kind": "error",
                "reply": "I encountered an error calling the OpenRouter service. Please try again.",
                "meta": {
                    "mode": "modelling",
                    "tier": tier,
                    "reason": "openrouter_call_failed",
                },
            }
            self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})
            return result

        logger.info("modelling_chat_success conversation_id=%s reply_len=%s", conversation_id, len(reply_text))
        result = {
            "kind": "success",
            "reply": reply_text,
            "attachments": [{"type": "link", "url": self.docs_url, "label": "CUWALID documentation"}],
            "meta": {
                "mode": "modelling",
                "tier": tier,
                "provider": "openrouter",
                "model": settings.llm_model,
            },
        }
        self.session_store.update(conversation_id, append_history={"role": "assistant", "text": result["reply"]})

        logger.info(
            "modelling_chat_complete conversation_id=%s kind=%s duration_ms=%.2f",
            conversation_id,
            result.get("kind"),
            (time.perf_counter() - stage_start) * 1000,
        )
        return result
