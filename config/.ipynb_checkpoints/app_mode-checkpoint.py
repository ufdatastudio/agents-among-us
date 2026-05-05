"""Central APP_MODE configuration for Agents Among Us.

Reads the APP_MODE environment variable and exposes helpers that
gate dotenv loading, GPU/torch imports, and provider visibility.

Modes:
    full      - All features enabled (local GPU models + all APIs). Default.
    api       - All API providers, no local GPU models.
    navigator - Navigator API only, no dotenv, no GPU.
"""

import os

VALID_MODES = {"full", "api", "navigator"}


def get_app_mode():
    """Return the current APP_MODE, defaulting to 'full'."""
    mode = os.environ.get("APP_MODE", "full").lower()
    if mode not in VALID_MODES:
        mode = "full"
    return mode


def should_load_dotenv():
    """Return True only in full mode (dotenv provides local secrets)."""
    return get_app_mode() == "full"


def should_load_gpu():
    """Return True only in full mode (GPU/torch required)."""
    return get_app_mode() == "full"


def get_allowed_providers():
    """Return the set of allowed API provider names, or None for all.

    Returns:
        None if all providers are allowed (full mode),
        otherwise a set of lowercase provider strings.
    """
    mode = get_app_mode()
    if mode == "full":
        return None
    if mode == "api":
        return {"navigator", "anthropic", "openai"}
    if mode == "navigator":
        return {"navigator"}
    return None
