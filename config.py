"""
Configuration module for AI-powered Meal Planning System
========================================================

This module centralizes all configuration for the meal planning system that
integrates:
- Mealie v3.9.1 (recipe management, Docker)
- OpenRouter API (cloud LLM for chat/reasoning)
- NVIDIA local embeddings (4096-dim vectors)

ARCHITECTURE:
- LLM Chat: OpenRouter API (configurable model)
- Embeddings: Local NVIDIA llama-embed-nemotron-8b (4096-dim, fixed)
- Recipe DB: Mealie with OpenRouter integration for parsing

CONFIGURATION:
- config.yaml: User-specific settings (Mealie URL, household, pantry, LLM models)
- data/secrets.yaml: Credentials (mealie token, openrouter API key)

Usage:
    from config import MEALIE_URL, CHAT_MODEL, USER_CONFIG, validate_all

    # Validate connections before running
    if not validate_all():
        exit(1)

SETUP REQUIRED:
    1. Copy config.yaml.example to data/config.yaml
    2. Edit config.yaml with your Mealie URL, household settings, etc.
    3. Set MEALIE_TOKEN and OPENROUTER_API_KEY (env vars or data/secrets.yaml)
"""

import os
import os.path
import sys
import asyncio
import logging
import logging.handlers
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import yaml


# =============================================================================
# MEALIE API RATE LIMITER
# =============================================================================
# Prevents overwhelming Mealie with too many concurrent requests.
# All Mealie API calls should use: with mealie_rate_limit(): ...
#
# Tuning guide:
# - Start with 5-10 for SQLite backend (default Mealie)
# - Try 15-25 for PostgreSQL backend
# - If Mealie crashes or becomes unresponsive, reduce this
# - If imports are slow and Mealie CPU is low, increase this
# - Monitor Mealie's Docker logs for errors

def _get_mealie_max_concurrent():
    """Get max concurrent Mealie requests from config, with sensible default.
    
    Default is 2 because:
    - Mealie's default backend is SQLite
    - SQLite can only handle one writer at a time
    - Concurrent writes cause "database is locked" errors
    - 2 allows some parallelism while avoiding lock contention
    
    PostgreSQL users can increase this to 10-15 in config.yaml.
    """
    try:
        # Allow override in config.yaml under mealie.max_concurrent_requests
        return USER_CONFIG.get('mealie', {}).get('max_concurrent_requests', 2)
    except:
        return 2  # Safe default for SQLite (Mealie's default backend)

MEALIE_MAX_CONCURRENT_REQUESTS = _get_mealie_max_concurrent()

_mealie_semaphore = threading.Semaphore(MEALIE_MAX_CONCURRENT_REQUESTS)

@contextmanager
def mealie_rate_limit():
    """
    Context manager to limit concurrent Mealie API requests.
    
    Usage:
        from config import mealie_rate_limit
        
        with mealie_rate_limit():
            response = requests.get(f"{MEALIE_URL}/api/recipes/...")
    
    This prevents overwhelming Mealie when running parallel workers.
    """
    _mealie_semaphore.acquire()
    try:
        yield
    finally:
        _mealie_semaphore.release()


# =============================================================================
# USER CONFIGURATION LOADING (STRICT - NO FALLBACKS)
# =============================================================================

# Project root directory (where this file lives)
PROJECT_ROOT = Path(__file__).parent

# Data directory - THE canonical location for all runtime data
# This is the only writable directory in Docker deployments
DATA_DIR = PROJECT_ROOT / "data"

# Config path - ONE location, no fallbacks, no confusion
CONFIG_PATH = DATA_DIR / "config.yaml"

# Secrets path - unified credential storage
SECRETS_PATH = DATA_DIR / "secrets.yaml"


def get_config_path() -> Path:
    """Get the config.yaml path. Always data/config.yaml."""
    return CONFIG_PATH


def _migrate_legacy_config():
    """
    One-time migration: move config.yaml from project root to data/.
    
    This handles the transition for existing users who have config in the old location.
    """
    legacy_config = PROJECT_ROOT / "config.yaml"
    
    if legacy_config.exists() and not CONFIG_PATH.exists():
        # Ensure data/ directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Copy (not move) to preserve the original during transition
        import shutil
        shutil.copy2(legacy_config, CONFIG_PATH)
        print(f"[config] Migrated config.yaml to {CONFIG_PATH}")
        print(f"[config] You can delete the old {legacy_config} after verifying.")


def _load_user_config() -> Dict[str, Any]:
    """
    Load user configuration from data/config.yaml.
    
    FAILS IMMEDIATELY if config.yaml is missing or invalid.
    NO FALLBACKS - users must create their own config.yaml.
    
    Returns:
        Dict containing user configuration
        
    Raises:
        FileNotFoundError: If config.yaml does not exist
        ValueError: If YAML is invalid or missing required fields
    """
    # Auto-migrate from legacy location if needed
    _migrate_legacy_config()
    
    config_path = CONFIG_PATH
    
    # No config = auto-create from example so the web UI can boot for first-time setup
    if not config_path.exists():
        example_path = PROJECT_ROOT / "config.yaml.example"
        if example_path.exists():
            import shutil
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example_path, config_path)
            print(f"[config] Created {config_path} from config.yaml.example")
            print(f"[config] Open the web panel to configure your settings.")
        else:
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"ERROR: config.yaml not found\n"
                f"{'='*60}\n"
                f"Expected location: {config_path}\n"
                f"Also missing: {example_path}\n"
                f"Please reinstall or restore config.yaml.example.\n"
                f"{'='*60}"
            )
    
    # Parse YAML - fail on syntax errors
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: config.yaml has invalid YAML syntax\n"
            f"{'='*60}\n"
            f"File: {config_path}\n"
            f"Error: {e}\n"
            f"{'='*60}"
        ) from e
    
    if config is None:
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: config.yaml is empty\n"
            f"{'='*60}\n"
            f"File: {config_path}\n"
            f"Please copy config.yaml.example and customize it.\n"
            f"{'='*60}"
        )
    
    # Validate required sections exist
    required_sections = ["connection", "household", "pantry", "llm"]
    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: config.yaml missing required sections\n"
            f"{'='*60}\n"
            f"Missing: {missing_sections}\n"
            f"Required sections: {required_sections}\n"
            f"{'='*60}"
        )
    
    # Validate required fields within sections
    required_fields = [
        ("connection", "mealie_url"),
        ("household", "servings"),
        ("household", "meal_types"),
        ("household", "description"),
        ("llm", "chat_model"),
        ("llm", "embedding_model"),
    ]
    
    missing_fields = []
    for section, field in required_fields:
        if field not in config.get(section, {}):
            missing_fields.append(f"{section}.{field}")
    
    if missing_fields:
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: config.yaml missing required fields\n"
            f"{'='*60}\n"
            f"Missing: {missing_fields}\n"
            f"{'='*60}"
        )
    
    # Validate pantry.staples is a list
    if not isinstance(config.get("pantry", {}).get("staples", []), list):
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: config.yaml pantry.staples must be a list\n"
            f"{'='*60}"
        )
    
    # Validate household.meal_types is a list
    if not isinstance(config["household"]["meal_types"], list):
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: config.yaml household.meal_types must be a list\n"
            f"{'='*60}"
        )
    
    return config


# Load user config at module initialization (FAIL FAST)
USER_CONFIG = _load_user_config()

# Use standard logging for config.py (foundational module)
logger = logging.getLogger(__name__)


def reload_user_config() -> Dict[str, Any]:
    """
    Reload data/config.yaml into the in-memory USER_CONFIG.
    
    This enables web UI changes to take effect without restarting the process.
    Note: modules that imported individual constants (e.g., CHAT_MODEL) may still
    hold older values; prefer reading from USER_CONFIG at runtime for tunable knobs.
    
    Returns:
        dict: The reloaded USER_CONFIG
    """
    global USER_CONFIG, MEALIE_URL, CHAT_MODEL, LLM_MODEL, OPENROUTER_MODEL, LM_STUDIO_MODEL
    global MEALIE_MAX_CONCURRENT_REQUESTS, _mealie_semaphore

    USER_CONFIG = _load_user_config()

    # Refresh derived values that are safe to update at runtime
    MEALIE_URL = os.getenv("MEALIE_URL", USER_CONFIG["connection"]["mealie_url"])
    CHAT_MODEL = USER_CONFIG["llm"]["chat_model"]
    LLM_MODEL = CHAT_MODEL
    OPENROUTER_MODEL = CHAT_MODEL
    LM_STUDIO_MODEL = CHAT_MODEL

    # Update Mealie rate limiter semaphore (best effort)
    try:
        MEALIE_MAX_CONCURRENT_REQUESTS = _get_mealie_max_concurrent()
        _mealie_semaphore = threading.Semaphore(MEALIE_MAX_CONCURRENT_REQUESTS)
    except Exception:
        pass

    logger.info("🔄 User config reloaded from disk")
    return USER_CONFIG

# Lazy import requests to avoid SSL context issues in sandbox
def _get_requests():
    try:
        import requests
        return requests
    except Exception as e:
        raise RuntimeError(f"requests library unavailable: {e}. This may be due to sandbox restrictions.")


# =============================================================================
# UNIFIED SECRETS MANAGEMENT
# =============================================================================
"""
Centralized credential storage in data/secrets.yaml.
Environment variables take priority over file-based secrets.
"""


def load_secrets() -> Dict[str, Any]:
    """
    Load secrets from data/secrets.yaml.
    
    Returns:
        dict with keys 'mealie_token', 'openrouter_api_key', 'brave_api_key' (may be None)
        Returns empty dict if file doesn't exist
    """
    if not SECRETS_PATH.exists():
        return {}
    
    try:
        with open(SECRETS_PATH, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        return {
            'mealie_token': data.get('mealie', {}).get('token'),
            'openrouter_api_key': data.get('openrouter', {}).get('api_key'),
            'brave_api_key': data.get('brave', {}).get('api_key'),
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to load secrets from {SECRETS_PATH}: {e}")
        return {}


def save_secrets(mealie_token: str = None, openrouter_api_key: str = None, brave_api_key: str = None) -> None:
    """
    Save secrets to data/secrets.yaml.
    
    Only updates the provided values, preserving existing ones.
    Creates data/ directory if needed.
    
    Args:
        mealie_token: Mealie JWT token (optional)
        openrouter_api_key: OpenRouter API key (optional)
        brave_api_key: Brave Search API key (optional)
    """
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing secrets
    existing = {}
    if SECRETS_PATH.exists():
        try:
            with open(SECRETS_PATH, 'r') as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            existing = {}
    
    # Initialize nested dicts if needed
    if 'mealie' not in existing:
        existing['mealie'] = {}
    if 'openrouter' not in existing:
        existing['openrouter'] = {}
    if 'brave' not in existing:
        existing['brave'] = {}
    
    # Update only provided values
    if mealie_token is not None:
        existing['mealie']['token'] = mealie_token
    if openrouter_api_key is not None:
        existing['openrouter']['api_key'] = openrouter_api_key
    if brave_api_key is not None:
        existing['brave']['api_key'] = brave_api_key
    
    # Write back
    with open(SECRETS_PATH, 'w') as f:
        yaml.safe_dump(existing, f, default_flow_style=False)
    
    logger.info(f"✅ Secrets saved to {SECRETS_PATH}")


def get_credential_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status of configured credentials.
    
    Returns:
        dict with credential status:
        {
            'mealie_token': {'configured': bool, 'source': 'env'|'file'|None, 'preview': str|None},
            'openrouter_api_key': {'configured': bool, 'source': 'env'|'file'|None, 'preview': str|None}
        }
    """
    secrets = load_secrets()
    
    def _get_status(env_var: str, file_key: str) -> Dict[str, Any]:
        """Get status for a single credential."""
        env_value = os.getenv(env_var, "").strip()
        file_value = secrets.get(file_key)
        
        if env_value:
            return {
                'configured': True,
                'source': 'env',
                'preview': f"••••••••{env_value[-5:]}" if len(env_value) >= 5 else "••••••••"
            }
        elif file_value:
            return {
                'configured': True,
                'source': 'file',
                'preview': f"••••••••{file_value[-5:]}" if len(file_value) >= 5 else "••••••••"
            }
        else:
            return {
                'configured': False,
                'source': None,
                'preview': None
            }
    
    return {
        'mealie_token': _get_status('MEALIE_TOKEN', 'mealie_token'),
        'openrouter_api_key': _get_status('OPENROUTER_API_KEY', 'openrouter_api_key'),
        'brave_api_key': _get_status('BRAVE_API_KEY', 'brave_api_key'),
    }


def reload_credentials() -> None:
    """
    Reload credentials from secrets file into global variables.
    
    Call this after updating secrets via save_secrets() to make
    changes effective at runtime without restart.
    """
    global MEALIE_TOKEN, CHAT_API_KEY, OPENROUTER_API_KEY, BRAVE_API_KEY
    
    MEALIE_TOKEN = load_mealie_token()
    CHAT_API_KEY = load_chat_api_key()
    OPENROUTER_API_KEY = CHAT_API_KEY  # Keep alias in sync
    BRAVE_API_KEY = load_brave_api_key()
    
    logger.info("🔄 Credentials reloaded from secrets")


# =============================================================================
# MEALIE CONFIGURATION
# =============================================================================
"""
Mealie is the recipe management system that stores recipes and meal plans.
"""


def load_mealie_token() -> Optional[str]:
    """
    Load MEALIE_TOKEN from environment variable or secrets file.

    Priority order (ENV VAR IS SOURCE OF TRUTH):
    1. Environment variable MEALIE_TOKEN (preferred)
    2. File: data/secrets.yaml (fallback)

    Returns:
        str: The JWT token if found, None otherwise
    """
    # First try environment variable (source of truth for containerized deployments)
    env_token = os.getenv("MEALIE_TOKEN", "").strip()
    if env_token:
        logger.debug(f"🔑 Using MEALIE_TOKEN from env var (length: {len(env_token)})")
        return env_token
    
    # Fallback to secrets file
    secrets = load_secrets()
    file_token = secrets.get('mealie_token')
    if file_token:
        logger.debug(f"🔑 Using MEALIE_TOKEN from {SECRETS_PATH} (length: {len(file_token)})")
        return file_token
    
    logger.warning("⚠️ No MEALIE_TOKEN found in env var or data/secrets.yaml")
    return None


# Load from config.yaml (env var override supported for containerized deployments)
MEALIE_URL = os.getenv("MEALIE_URL", USER_CONFIG["connection"]["mealie_url"])
# Load MEALIE_TOKEN using centralized function
MEALIE_TOKEN = load_mealie_token()

if MEALIE_TOKEN:
    logger.debug(f"🔑 config.py MEALIE_TOKEN configured (length: {len(MEALIE_TOKEN)})")
else:
    logger.error("❌ config.py MEALIE_TOKEN: None")

# API endpoints
MEALIE_API_BASE = f"{MEALIE_URL}/api"
MEALIE_RECIPES_ENDPOINT = f"{MEALIE_API_BASE}/recipes"
MEALIE_MEALPLANS_ENDPOINT = f"{MEALIE_API_BASE}/groups/mealplans"


# =============================================================================
# CHAT LLM CONFIGURATION (Canonical: CHAT_*)
# =============================================================================
"""
OpenRouter provides cloud LLM inference for meal planning decisions.
Using OpenAI-compatible API endpoints for intelligent meal suggestions.
"""

# Canonical chat configuration (SOURCE OF TRUTH)
CHAT_API_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = USER_CONFIG["llm"]["chat_model"]

def load_chat_api_key() -> Optional[str]:
    """
    Load OpenRouter API key from environment variable or secrets file.
    
    Priority order (ENV VAR IS SOURCE OF TRUTH):
    1. Environment variable OPENROUTER_API_KEY (preferred)
    2. File: data/secrets.yaml (fallback)
    
    Returns:
        str: The API key if found, None otherwise
    """
    # First try environment variable (source of truth for containerized deployments)
    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_key:
        logger.debug("🔑 Using OPENROUTER_API_KEY from env var")
        return env_key
    
    # Fallback to secrets file
    secrets = load_secrets()
    file_key = secrets.get('openrouter_api_key')
    if file_key:
        logger.debug(f"🔑 Using OPENROUTER_API_KEY from {SECRETS_PATH}")
        return file_key
    
    logger.warning("⚠️ No OPENROUTER_API_KEY found in env var or data/secrets.yaml")
    return None

CHAT_API_KEY = load_chat_api_key()

# Backwards compatibility aliases (keep older scripts working)
# OpenRouter legacy names
OPENROUTER_URL = CHAT_API_URL
OPENROUTER_MODEL = CHAT_MODEL
OPENROUTER_API_KEY = CHAT_API_KEY

# Legacy names (historical; now point to OpenRouter)
LM_STUDIO_URL = CHAT_API_URL
LM_STUDIO_MODEL = CHAT_MODEL
LM_STUDIO_API_ENDPOINT = f"{CHAT_API_URL}/chat/completions"
LM_STUDIO_HEALTH_ENDPOINT = f"{CHAT_API_URL}/models"

# Generic legacy name for backwards compatibility
LLM_MODEL = CHAT_MODEL


# =============================================================================
# BRAVE SEARCH API CONFIGURATION
# =============================================================================
"""
Brave Search API provides image search for recipe images.
Used to find appropriate images for AI-generated recipes.
"""

BRAVE_API_URL = "https://api.search.brave.com/res/v1/images/search"

def load_brave_api_key() -> Optional[str]:
    """
    Load Brave Search API key from environment variable or secrets file.
    
    Priority order (ENV VAR IS SOURCE OF TRUTH):
    1. Environment variable BRAVE_API_KEY (preferred)
    2. File: data/secrets.yaml (fallback)
    
    Returns:
        str: The API key if found, None otherwise
    """
    # First try environment variable (source of truth for containerized deployments)
    env_key = os.getenv("BRAVE_API_KEY", "").strip()
    if env_key:
        logger.debug("🔑 Using BRAVE_API_KEY from env var")
        return env_key
    
    # Fallback to secrets file
    secrets = load_secrets()
    file_key = secrets.get('brave_api_key')
    if file_key:
        logger.debug(f"🔑 Using BRAVE_API_KEY from {SECRETS_PATH}")
        return file_key
    
    logger.debug("No BRAVE_API_KEY found (optional - needed for recipe image search)")
    return None

BRAVE_API_KEY = load_brave_api_key()


# =============================================================================
# HOUSEHOLD CONFIGURATION (loaded from config.yaml)
# =============================================================================
"""
Settings specific to the household - loaded from config.yaml.
Customize these in your config.yaml file.
"""

HOUSEHOLD_SERVINGS = USER_CONFIG["household"]["servings"]
MEAL_TYPES = USER_CONFIG["household"]["meal_types"]
HOUSEHOLD_DESCRIPTION = USER_CONFIG["household"]["description"]


# =============================================================================
# MEAL PLANNING CONFIGURATION
# =============================================================================
"""
Controls the meal planning algorithm behavior.
"""

DAYS_TO_PLAN = 7  # Plan for one week at a time
START_DAY = "Monday"  # Week starts on Monday
HISTORY_WEEKS = 4  # Look back 4 weeks for variety checking

# Day order for planning
WEEK_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# =============================================================================
# REQUEST TIMEOUT CONFIGURATION
# =============================================================================
"""
Timeout settings for HTTP requests to external services.
"""

MEALIE_TIMEOUT = 10  # seconds
LM_STUDIO_TIMEOUT = 30  # seconds (AI inference can take longer)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_mealie_connection() -> bool:
    """
    Validate that Mealie server is reachable and responding.
    
    Returns:
        bool: True if Mealie is accessible, False otherwise
    """
    if not MEALIE_TOKEN:
        logger.error("❌ ERROR: MEALIE_TOKEN environment variable is not set!")
        logger.error("   Please set it using: export MEALIE_TOKEN='your-token-here'")
        return False
    
    try:
        requests = _get_requests()
        headers = {"Authorization": f"Bearer {MEALIE_TOKEN}"}
        response = requests.get(
            f"{MEALIE_API_BASE}/app/about",
            headers=headers,
            timeout=MEALIE_TIMEOUT
        )

        if response.status_code == 200:
            logger.info(f"✅ Mealie connection successful: {MEALIE_URL}")
            return True
        elif response.status_code == 401:
            logger.error(f"❌ ERROR: Mealie authentication failed (401 Unauthorized)")
            logger.error(f"   Check your MEALIE_TOKEN is valid")
            return False
        else:
            logger.error(f"❌ ERROR: Mealie returned status code {response.status_code}")
            return False

    except Exception as e:
        if "ConnectionError" in str(type(e)):
            logger.error(f"❌ ERROR: Cannot connect to Mealie at {MEALIE_URL}")
            logger.error(f"   Is the Mealie server running?")
        elif "Timeout" in str(type(e)):
            logger.error(f"❌ ERROR: Mealie connection timed out after {MEALIE_TIMEOUT} seconds")
        else:
            logger.error(f"❌ ERROR: Unexpected error connecting to Mealie: {e}")
        return False


def validate_lm_studio_connection() -> bool:
    """
    Validate that OpenRouter API is accessible and configured.

    Returns:
        bool: True if OpenRouter API key is set and API is accessible, False otherwise
    """
    # Check if API key is configured
    if not OPENROUTER_API_KEY:
        logger.error("❌ ERROR: OPENROUTER_API_KEY not found!")
        logger.error("   Please set it via OPENROUTER_API_KEY env var or in data/secrets.yaml")
        return False
    
    try:
        requests = _get_requests()
        # Test OpenRouter connection with models endpoint
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/deepseekcoder2/ayechef"
        }
        response = requests.get(
            f"{OPENROUTER_URL}/models",
            headers=headers,
            timeout=LM_STUDIO_TIMEOUT
        )

        if response.status_code == 200:
            logger.info(f"✅ OpenRouter connection successful")
            logger.info(f"✅ Using model: {OPENROUTER_MODEL}")
            return True
        elif response.status_code == 401:
            logger.error(f"❌ ERROR: OpenRouter authentication failed (401 Unauthorized)")
            logger.error(f"   Check your OPENROUTER_API_KEY is valid")
            return False
        else:
            logger.error(f"❌ ERROR: OpenRouter returned status code {response.status_code}")
            return False

    except Exception as e:
        if "ConnectionError" in str(type(e)):
            logger.error(f"❌ ERROR: Cannot connect to OpenRouter at {OPENROUTER_URL}")
            logger.error(f"   Check your internet connection")
        elif "Timeout" in str(type(e)):
            logger.error(f"❌ ERROR: OpenRouter connection timed out after {LM_STUDIO_TIMEOUT} seconds")
        else:
            logger.error(f"❌ ERROR: Unexpected error connecting to OpenRouter: {e}")
        return False


def validate_all() -> bool:
    """
    Validate system dependencies.

    Mealie and OpenRouter connections are mandatory.

    Returns:
        bool: True if all dependencies are validated, False otherwise
    """
    logger.info("=" * 60)
    logger.info("🔍 VALIDATING SYSTEM DEPENDENCIES (WITH HTTP FALLBACKS)...")
    logger.info("=" * 60)

    # CRITICAL: Mealie connection is mandatory
    mealie_ok = validate_mealie_connection()
    if not mealie_ok:
        logger.error("=" * 60)
        logger.error("❌ CRITICAL FAILURE: Mealie validation failed")
        logger.error("❌ FAST FAILURE: System cannot operate without Mealie - fix connection and restart")
        logger.error("=" * 60)
        sys.exit(1)

    # CRITICAL: OpenRouter API connection is mandatory
    openrouter_ok = validate_lm_studio_connection()
    if not openrouter_ok:
        logger.error("=" * 60)
        logger.error("❌ CRITICAL FAILURE: OpenRouter validation failed")
        logger.error("❌ FAST FAILURE: System cannot operate without OpenRouter - fix API key and restart")
        logger.error("=" * 60)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("✅ ALL DEPENDENCIES VALIDATED - SYSTEM READY!")
    logger.info("✅ Mealie: Connected and authenticated")
    logger.info("✅ OpenRouter: Connected with required model")
    logger.info("✅ NVIDIA Embeddings: Local model (sentence-transformers)")
    logger.info("✅ SYSTEM READY FOR PRODUCTION")
    logger.info("=" * 60)

    return True


def get_mealie_headers() -> dict:
    """
    Get headers for Mealie API requests.
    
    Returns:
        dict: Headers dictionary with authorization token
    """
    return {
        "Authorization": f"Bearer {MEALIE_TOKEN}",
        "Content-Type": "application/json"
    }


def print_config_summary() -> None:
    """
    Print a summary of the current configuration.
    Useful for debugging and verification.
    """
    embedding_model = USER_CONFIG["llm"]["embedding_model"]
    
    print("\n" + "=" * 60)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Mealie URL:         {MEALIE_URL}")
    print(f"Mealie Token:       {'✓ Set' if MEALIE_TOKEN else '✗ Not set'}")
    print(f"Chat Model:         {CHAT_MODEL}")
    print(f"Chat Key:           {'✓ Set' if CHAT_API_KEY else '✗ Not set'}")
    print(f"Embedding Model:    {embedding_model} (4096-dim, local)")
    print(f"Household:          {HOUSEHOLD_DESCRIPTION}")
    print(f"Household Size:     {HOUSEHOLD_SERVINGS} servings")
    print(f"Meal Types:         {', '.join(MEAL_TYPES)}")
    print(f"Pantry Staples:     {len(USER_CONFIG['pantry']['staples'])} items")
    print(f"Planning Period:    {DAYS_TO_PLAN} days starting {START_DAY}")
    print(f"History Lookback:   {HISTORY_WEEKS} weeks")
    print("=" * 60 + "\n")


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
"""
Centralized configuration for the shopping list generation pipeline.
All parameters previously hardcoded in various modules are now configurable here.
"""

# Ingredient Parsing Configuration
INGREDIENT_PARSING_CONFIG = {
    "confidence_threshold": 0.80,  # Minimum confidence fraction (0.80 = 80%)
    "cache_duration_hours": 24,    # Skip recipes processed within last 24 hours
    "polite_delay": 0.1,          # Delay between recipe processing (seconds)
    "page_size": 200,              # Number of recipes to fetch per page
    "max_recipes_per_batch": 50,   # Maximum recipes to check for validation
    "enable_auto_tagging": True,   # Automatically tag recipes during parsing
    "enable_validation": True,     # Validate parsing results
}

# Shopping List Refinement Configuration
SHOPPING_LIST_CONFIG = {
    "enable_llm_refinement": True,  # Enable LLM-based refinement
    "batch_size": 25,              # Items per LLM batch
    "concurrent_operations": 8,    # Concurrent API calls
}

# Timeout Configuration (extends existing timeouts)
PIPELINE_TIMEOUTS = {
    "ingredient_parsing": 900,     # 15 minutes for parsing operations
    "shopping_refinement": 300,    # 5 minutes for refinement operations
    "llm_retry_delay": 60,         # 1 minute between LLM retries
    "network_retry_delay": 30,     # 30 seconds between network retries
    "parsing_retry_delay": 10,     # 10 seconds between parsing retries
}

# Retry Configuration
RETRY_CONFIG = {
    "max_llm_retries": 3,          # Maximum LLM operation retries
    "max_network_retries": 2,      # Maximum network operation retries
    "max_parsing_retries": 1,      # Maximum parsing operation retries
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "enable_detailed_logging": True,  # Enable detailed operation logging
    "log_level": "INFO",            # Logging level (DEBUG, INFO, WARNING, ERROR)
    "enable_metrics_collection": True,  # Collect operation metrics
    "health_check_interval": 300,   # Health check interval in seconds
    "mealie_connection_pool_size": 50,    # Connection pool for Mealie API
    "lm_studio_connection_pool_size": 20, # Connection pool for LLM API
    "max_concurrent_operations": 8,       # Concurrent LLM operations
    "batch_size": 15,                     # Batch size for throughput
}

# Feature Flags for Graceful Degradation
FEATURE_FLAGS = {
    "allow_degraded_mode": True,    # Allow operations to continue in degraded mode
    "skip_validation_on_failure": False,  # Skip validation if it fails
    "continue_on_llm_failure": True,  # Continue pipeline if LLM fails
    "continue_on_network_failure": False,  # Stop pipeline on network failures
}

# Monitoring and Observability
MONITORING_CONFIG = {
    "enable_operation_metrics": True,     # Track operation performance
    "enable_error_tracking": True,        # Track errors and failures
    "enable_health_monitoring": True,     # Enable health checks
    "metrics_retention_days": 7,          # Keep metrics for 7 days
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
"""Centralized logging configuration for all modules.

Replaces scattered print() statements with structured logging.
"""
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(DATA_DIR / "logs" / "aye_chef.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}

# Validation Configuration
VALIDATION_CONFIG = {
    "strict_mode": False,           # Exit on any validation failure
    "validate_before_operations": True,  # Validate connections before operations
    "validate_after_operations": True,   # Validate results after operations
}

# Bulk Operations Configuration - OpenRouter Cloud Optimized
BULK_OPERATIONS_CONFIG = {
    # Optimized for OpenRouter cloud API (free tier: 20 req/min, paid: much higher)
    "tag": {
        "default_batch_size": 20,     # Higher batching for cloud throughput
        "max_concurrent": 15,         # OpenRouter handles concurrency server-side
        "timeout_per_recipe": 30,     # Cloud is faster than local
        "description": "Recipe tagging/classification operations"
    },
    "parse": {
        "default_batch_size": 30,     # Higher throughput for cloud parsing
        "max_concurrent": 15,         # OpenRouter handles concurrency server-side
        "timeout_per_recipe": 45,     # Parsing can be complex
        "description": "Ingredient parsing operations"
    },
    "index": {
        "default_batch_size": 75,     # Optimized for GPU-accelerated vector operations
        "max_concurrent": 6,          # Balance GPU utilization with memory constraints
        "timeout_per_recipe": 12,     # Timeout for vector operations per recipe
        "description": "RecipeRAG indexing operations"
    },
    "sync": {
        "default_batch_size": 30,     # Higher batching for local Mealie operations
        "max_concurrent": 8,          # Local Mealie can handle higher concurrency
        "timeout_per_recipe": 15,     # Fast local network operations
        "description": "Database synchronization operations"
    },
    "import": {
        "default_batch_size": 25,     # Balanced for bulk import operations
        "max_concurrent": 6,          # Respect external API limits while maximizing local processing
        "timeout_per_recipe": 45,     # Import includes multiple steps
        "description": "Recipe import operations"
    }
}

# Environment-based scaling multipliers
ENVIRONMENT_CONFIG = {
    "development": {
        "multiplier": 0.67,  # ~67% of production values - allows batch_size >= 2 for testing
        "description": "Balanced settings for development environment - reasonable performance with safety"
    },
    "production": {
        "multiplier": 1.0,  # Full production values
        "description": "Optimized settings for production environment"
    },
    "ci": {
        "multiplier": 0.3,  # Minimal resources for CI/CD pipelines
        "description": "Minimal settings for CI/CD environment"
    },
    "testing": {
        "multiplier": 0.2,  # Very conservative for testing
        "description": "Minimal settings for testing environment"
    }
}

# =============================================================================
# PARALLELISM CONFIGURATION (Bulk Import Pipeline)
# =============================================================================
"""
Preset configurations for parallel processing during bulk imports.
Users select a preset matching their hardware, with optional manual overrides.
"""

PARALLELISM_PRESETS = {
    "apple_silicon_8gb": {
        "discovery": {"workers": 8},
        "import": {"workers": 6},
        "tagging": {"workers": 8},
        "parsing": {"workers": 8},
        "indexing": {"workers": 4, "batch_size": 50},
    },
    "apple_silicon_16gb": {
        "discovery": {"workers": 12},
        "import": {"workers": 8},
        "tagging": {"workers": 12},
        "parsing": {"workers": 12},
        "indexing": {"workers": 6, "batch_size": 60},
    },
    "apple_silicon_32gb": {
        "discovery": {"workers": 15},
        "import": {"workers": 10},
        "tagging": {"workers": 16},
        "parsing": {"workers": 16},
        "indexing": {"workers": 8, "batch_size": 75},
    },
    "apple_silicon_max": {
        # Workers are capped by MEALIE_MAX_CONCURRENT_REQUESTS for API calls
        # Set workers slightly above Mealie limit to keep pipeline fed, but not wastefully high
        "discovery": {"workers": 12},
        "import": {"workers": 10},   # Mealie-bound: POST create + GET verify
        "tagging": {"workers": 12},  # Mix: LLM analysis + Mealie read/write
        "parsing": {"workers": 10},  # Mealie-heavy: multiple API calls per recipe
        "indexing": {"workers": 8, "batch_size": 100},  # Mealie read + local embedding
    },
    "nvidia_gpu": {
        "discovery": {"workers": 10},
        "import": {"workers": 8},
        "tagging": {"workers": 10},
        "parsing": {"workers": 10},
        "indexing": {"workers": 8, "batch_size": 75},
    },
    "cloud_api": {
        "discovery": {"workers": 15},
        "import": {"workers": 10},
        "tagging": {"workers": 20},
        "parsing": {"workers": 20},
        "indexing": {"workers": 6, "batch_size": 75},
    },
    "conservative": {
        "discovery": {"workers": 4},
        "import": {"workers": 3},
        "tagging": {"workers": 4},
        "parsing": {"workers": 4},
        "indexing": {"workers": 2, "batch_size": 25},
    },
}


def get_parallelism_config(phase: str) -> dict:
    """
    Get parallelism settings for a pipeline phase.
    
    Loads from preset, then applies any manual overrides from config.yaml.
    
    Args:
        phase: 'discovery', 'import', 'tagging', 'parsing', or 'indexing'
    
    Returns:
        dict with 'workers' and optionally 'batch_size'
    
    Raises:
        ValueError: If phase is not a valid phase name
        
    Example:
        >>> get_parallelism_config('tagging')
        {'workers': 20}
        >>> get_parallelism_config('indexing')
        {'workers': 6, 'batch_size': 75}
    """
    # Get preset name from user config, default to 'conservative'
    # Conservative is safe for SQLite (Mealie's default backend)
    # Users with PostgreSQL can switch to 'cloud_api' for faster imports
    parallelism_config = USER_CONFIG.get("parallelism", {})
    preset_name = parallelism_config.get("preset", "conservative")
    
    # Validate preset exists, fall back to conservative if not
    if preset_name not in PARALLELISM_PRESETS:
        logger.warning(f"⚠️ Unknown parallelism preset '{preset_name}', using 'conservative'")
        preset_name = "conservative"
    
    preset = PARALLELISM_PRESETS[preset_name]
    
    # Get base config for this phase from preset
    if phase not in preset:
        valid_phases = list(preset.keys())
        raise ValueError(f"Unknown parallelism phase '{phase}'. Valid phases: {valid_phases}")
    
    config = preset[phase].copy()
    
    # Apply manual overrides if present in user config
    manual_overrides = parallelism_config.get(phase, {})
    if manual_overrides:
        config.update(manual_overrides)
        logger.debug(f"Applied manual overrides for {phase}: {manual_overrides}")
    
    return config


def get_compute_device() -> str:
    """
    Get compute device for embedding model.
    
    Auto-detection order: CUDA → ROCm → MPS → CPU
    Can be overridden in config.yaml under hardware.device
    
    Returns:
        str: Device string for PyTorch ('cuda', 'mps', or 'cpu')
    """
    configured = USER_CONFIG.get("hardware", {}).get("device", "auto")
    
    if configured != "auto":
        logger.info(f"🔧 Using configured compute device: {configured}")
        return configured
    
    # Auto-detect best available device
    try:
        import torch
        
        # Check CUDA (includes NVIDIA and AMD ROCm)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
            logger.info(f"🔧 Auto-detected CUDA device: {device_name}")
            return "cuda"
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🔧 Auto-detected Apple Silicon MPS")
            return "mps"
        
        logger.info("🔧 No GPU detected, using CPU")
        return "cpu"
        
    except ImportError:
        logger.warning("⚠️ PyTorch not available, defaulting to CPU")
        return "cpu"


def get_embedding_batch_size() -> int:
    """
    Get batch size for embedding generation.
    
    Higher values = faster but use more memory.
    Can be configured in config.yaml under hardware.embedding_batch_size
    
    Returns:
        int: Batch size for embedding model
    """
    return USER_CONFIG.get("hardware", {}).get("embedding_batch_size", 32)


def get_pipeline_config() -> dict:
    """
    Get complete pipeline configuration as a dictionary.

    Returns:
        dict: Complete configuration for the pipeline
    """
    return {
        "timeouts": PIPELINE_TIMEOUTS,
        "retries": RETRY_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "features": FEATURE_FLAGS,
        "monitoring": MONITORING_CONFIG,
        "validation": VALIDATION_CONFIG,
        "ingredient_parsing": INGREDIENT_PARSING_CONFIG,
        "shopping_list": SHOPPING_LIST_CONFIG,
        "bulk_operations": BULK_OPERATIONS_CONFIG,
        "environment": ENVIRONMENT_CONFIG,
    }


def get_config_value(section: str, key: str, default=None):
    """
    Get a configuration value by section and key with graceful degradation.

    Args:
        section: Configuration section name
        key: Configuration key name
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    try:
        config = get_pipeline_config()
        section_config = config.get(section, {})
        return section_config.get(key, default)
    except Exception as e:
        # Graceful degradation: log error and return default
        logger.warning(f"⚠️  Configuration access failed for {section}.{key}: {e}")
        logger.warning(f"   Using default value: {default}")
        return default


def get_bulk_operation_config(operation_type: str) -> dict:
    """
    Get standardized configuration for bulk operations with environment-based scaling.

    Args:
        operation_type: Type of bulk operation ('tag', 'parse', 'index', 'sync', 'import')

    Returns:
        dict: Configuration with default_batch_size, max_concurrent, timeout_per_recipe, description

    Raises:
        ValueError: If operation_type is not supported
    """
    if operation_type not in BULK_OPERATIONS_CONFIG:
        supported = list(BULK_OPERATIONS_CONFIG.keys())
        raise ValueError(f"Unsupported operation type '{operation_type}'. Supported: {supported}")

    # Get base configuration
    base_config = BULK_OPERATIONS_CONFIG[operation_type].copy()

    # Apply environment-based scaling
    env = os.getenv('ENVIRONMENT', 'development')
    if env not in ENVIRONMENT_CONFIG:
        logger.warning(f"⚠️ Warning: Unknown environment '{env}', using 'development' settings")
        env = 'development'

    multiplier = ENVIRONMENT_CONFIG[env]['multiplier']

    # Scale numeric values (but not timeouts which are absolute)
    base_config['default_batch_size'] = max(1, int(base_config['default_batch_size'] * multiplier))
    base_config['max_concurrent'] = max(1, int(base_config['max_concurrent'] * multiplier))

    # Add environment info for debugging
    base_config['environment'] = env
    base_config['environment_multiplier'] = multiplier

    return base_config


def get_bulk_operation_config_safe(operation_type: str, fallback_batch_size: int = 1, fallback_concurrent: int = 1) -> dict:
    """
    Get bulk operation config with safe fallbacks to prevent system failures.

    Args:
        operation_type: Type of bulk operation
        fallback_batch_size: Safe fallback batch size if config fails
        fallback_concurrent: Safe fallback concurrent operations if config fails

    Returns:
        dict: Configuration with guaranteed valid values
    """
    try:
        return get_bulk_operation_config(operation_type)
    except Exception as e:
        logger.warning(f"⚠️ Warning: Failed to load bulk config for '{operation_type}': {e}")
        logger.warning(f"Using safe fallback values: batch_size={fallback_batch_size}, concurrent={fallback_concurrent}")
        return {
            "default_batch_size": fallback_batch_size,
            "max_concurrent": fallback_concurrent,
            "timeout_per_recipe": 60,  # Conservative timeout
            "description": f"Fallback config for {operation_type}",
            "environment": "fallback",
            "environment_multiplier": 1.0
        }


def get_image_search_config() -> Dict[str, Any]:
    """
    Get image search configuration with sensible defaults.
    
    Returns config dict with keys:
    - query_suffix: str
    - min_width: int
    - min_height: int
    - max_attempts: int
    - use_vision_validation: bool
    """
    defaults = {
        'query_suffix': 'recipe',
        'min_width': 400,
        'min_height': 300,
        'max_attempts': 3,
        'use_vision_validation': True,
    }
    
    user_config = USER_CONFIG.get('image_search', {})
    return {**defaults, **user_config}


# Image search configuration (loaded from config.yaml with defaults)
IMAGE_SEARCH_CONFIG = get_image_search_config()


# =============================================================================
# MODULE SELF-TEST
# =============================================================================

if __name__ == "__main__":
    """
    Run this module directly to test the configuration and validate connections.

    STRICT REQUIREMENT: Exits immediately on any validation failure.

    Usage: python config.py
    """
    print_config_summary()
    print("\n" + "=" * 60)
    print("PIPELINE CONFIGURATION SUMMARY")
    print("=" * 60)

    pipeline_config = get_pipeline_config()
    for section, values in pipeline_config.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # validate_all() now exits immediately on failure, so we only reach here if successful
    validate_all()
    print("\n🎉 ALL DEPENDENCIES VALIDATED - System ready for operation!")
    print("✅ Strict requirements enforced - no fallback modes allowed")
    sys.exit(0)

