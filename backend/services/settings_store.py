"""User settings: API keys and default model. Stored in data/settings.json.
Env vars (.env) take precedence for API keys if set; settings file overrides when env is empty."""
import json
from pathlib import Path

import config
from backend.services.web_search_mode import (
    WEB_SEARCH_MODE_OFF,
    normalize_web_search_mode,
)

SETTINGS_PATH = config.DATA_DIR / "settings.json"

_PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "tavily": "TAVILY_API_KEY",
}

_PROVIDER_ENV_FALLBACK = {
    "google": "GEMINI_API_KEY",
}


def _load_raw():
    """Load settings dict from file. Returns {} if missing."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_raw(data):
    """Write settings dict to file."""
    config.ensure_data_dirs()
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_api_key(provider):
    """Return effective API key for provider. Checks env first, then settings file."""
    import os
    env_key = _PROVIDER_ENV_KEYS.get(provider)
    fallback_key = _PROVIDER_ENV_FALLBACK.get(provider)
    # Env takes precedence
    val = os.environ.get(env_key, "") if env_key else ""
    if not val and fallback_key:
        val = os.environ.get(fallback_key, "") or ""
    if val and val.strip():
        return val.strip()
    # Fall back to settings file
    data = _load_raw()
    keys = data.get("api_keys") or {}
    return (keys.get(provider) or "").strip()


def mask_key(key):
    """Return masked representation (••••••••last4). Never expose full keys."""
    if not key or not isinstance(key, str) or len(key) < 4:
        return "••••••••" if key else ""
    return "••••••••" + key[-4:]


def get_ai_memory_enabled():
    """When True, chat uses RAG/fallback memory in the system prompt and runs post-reply memory extraction. Default off."""
    data = _load_raw()
    val = data.get("ai_memory_enabled")
    if val is True:
        return True
    if isinstance(val, str) and val.strip().lower() in ("1", "true", "yes"):
        return True
    return False


def get_settings_for_api():
    """Return settings safe for API response: masked API key status. default_model comes from models.yaml via API layer."""
    effective = {}
    for p in ("openai", "anthropic", "google", "tavily"):
        k = get_api_key(p)
        effective[p] = {
            "set": bool(k),
            "masked": mask_key(k),
        }
    return {
        "api_keys": effective,
        "default_web_search_mode": get_default_web_search_mode(),
        "ai_memory_enabled": get_ai_memory_enabled(),
    }


def get_default_web_search_mode():
    """Return persisted default web search mode for new chats."""
    data = _load_raw()
    return normalize_web_search_mode(
        data.get("default_web_search_mode"),
        default=WEB_SEARCH_MODE_OFF,
    )


def update_settings(updates):
    """Update settings. updates: { api_keys? }. default_model is managed in models.yaml via API layer."""
    data = _load_raw()
    if "api_keys" in updates:
        new_keys = updates["api_keys"]
        if isinstance(new_keys, dict):
            current = data.get("api_keys") or {}
            for k, v in new_keys.items():
                if k in ("openai", "anthropic", "google", "tavily") and v is not None:
                    v = str(v).strip()
                    if v:
                        current[k] = v
                    elif k in current:
                        del current[k]
            data["api_keys"] = current
    if "default_web_search_mode" in updates:
        data["default_web_search_mode"] = normalize_web_search_mode(
            updates.get("default_web_search_mode"),
            default=WEB_SEARCH_MODE_OFF,
        )
    if "ai_memory_enabled" in updates:
        v = updates["ai_memory_enabled"]
        data["ai_memory_enabled"] = bool(v)
    _save_raw(data)


def invalidate_provider_clients():
    """Clear cached provider clients/model catalog so they pick up new API keys."""
    try:
        from backend.providers import openai_provider
        openai_provider._client = None  # noqa
    except Exception:
        pass
    try:
        from backend.providers import anthropic_provider
        anthropic_provider._client = None  # noqa
    except Exception:
        pass
    try:
        from backend.providers import google_provider
        google_provider._client = None  # noqa
    except Exception:
        pass
    try:
        from backend.services import models_config
        models_config.invalidate_models_cache()
    except Exception:
        pass
