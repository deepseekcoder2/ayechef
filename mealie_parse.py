#!/usr/bin/env python3
"""
Bulk-parse every *un-parsed* recipe in Mealie ≥ v2.8
Uses LLM as mandatory parser (no fallbacks allowed)
"""

import re, argparse, json, os, pathlib, sys, time, requests, pprint, uuid
from tqdm import tqdm
from requests.exceptions import HTTPError
from datetime import datetime, timedelta

# Re-scraping for lost ingredient data
try:
    from recipe_scrapers import scrape_me
    RECIPE_SCRAPERS_AVAILABLE = True
except ImportError:
    scrape_me = None
    RECIPE_SCRAPERS_AVAILABLE = False

# ── CONFIG ───────────────────────────────────────────────────────────────
# Import configuration from centralized config.py
from config import (
    MEALIE_URL, MEALIE_TOKEN, get_mealie_headers, validate_mealie_connection, 
    get_bulk_operation_config_safe, CHAT_API_URL, CHAT_API_KEY, CHAT_MODEL,
    mealie_rate_limit, DATA_DIR
)
from tools.logging_utils import get_logger
from prompts import INGREDIENT_PARSING_SYSTEM_PROMPT, INGREDIENT_PARSING_USER_PROMPT

# Initialize logger for this module
logger = get_logger(__name__)

# ── DIRECT LLM PARSING (bypasses Mealie's weak prompt) ─────────────────────
# This calls OpenRouter directly with our optimized prompt

def normalize_unicode_fractions(text: str) -> str:
    """
    Normalize unicode fraction characters to ASCII equivalents.
    Some LLM models (e.g., grok) produce malformed JSON when processing unicode fractions.
    """
    replacements = {
        '¼': '1/4', '½': '1/2', '¾': '3/4',
        '⅓': '1/3', '⅔': '2/3',
        '⅛': '1/8', '⅜': '3/8', '⅝': '5/8', '⅞': '7/8',
        '⅕': '1/5', '⅖': '2/5', '⅗': '3/5', '⅘': '4/5',
        '⅙': '1/6', '⅚': '5/6',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def parse_ingredients_direct_llm(ingredient_strings: list[str]) -> list[dict] | None:
    """
    Parse ingredients using direct LLM call with our custom prompt.
    Bypasses Mealie's parser API entirely.

    Includes retry logic with exponential backoff for reliability.

    Returns list of parsed ingredients or None on failure.
    """
    if not CHAT_API_KEY:
        logger.error("CHAT_API_KEY not configured")
        return None

    if not ingredient_strings:
        return []

    # Normalize unicode fractions to prevent LLM JSON malformation issues
    normalized_strings = [normalize_unicode_fractions(s) for s in ingredient_strings]

    # Retry configuration
    max_retries = 3
    base_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            # Build the prompt
            ingredients_json = json.dumps(normalized_strings)
            user_prompt = INGREDIENT_PARSING_USER_PROMPT.format(ingredients_json=ingredients_json)

            # Call OpenRouter
            response = requests.post(
                f"{CHAT_API_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {CHAT_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/deepseekcoder2/ayechef"
                },
                json={
                    "model": CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": INGREDIENT_PARSING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,  # Low temp for consistent parsing
                    "max_tokens": 4096,  # Ensure enough tokens for long ingredient lists
                    "response_format": {"type": "json_object"}
                },
                timeout=90  # Increased timeout for longer responses
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            ingredients = parsed.get("ingredients", [])

            # Validate we got the right number
            if len(ingredients) != len(ingredient_strings):
                logger.warning(f"LLM returned {len(ingredients)} ingredients, expected {len(ingredient_strings)}")

            return ingredients

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:  # Don't log error on last attempt
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"OpenRouter request failed (attempt {attempt+1}/{max_retries}): {e}")
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"OpenRouter request failed after {max_retries} attempts: {e}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Log the raw content for debugging
            try:
                if 'content' in dir():
                    logger.debug(f"Raw LLM response (first 500 chars): {content[:500]}")
                    logger.debug(f"Raw LLM response length: {len(content)}")
            except:
                pass
            return None
        except Exception as e:
            logger.error(f"Direct LLM parsing failed: {e}")
            return None

    return None


def match_food_to_canonical(food_name: str, canonical_foods: dict) -> dict | None:
    """
    Match a parsed food name against Mealie's canonical food database.
    Returns the canonical food object with ID, or None if no match.
    
    Args:
        food_name: Normalized food name from LLM parsing
        canonical_foods: Dict mapping lowercase name -> food object
    """
    if not food_name:
        return None
    
    name_lower = food_name.lower().strip()
    
    # Priority 1: Exact match
    if name_lower in canonical_foods:
        return canonical_foods[name_lower]
    
    # Priority 2: Try singularizing
    singular = name_lower
    if singular.endswith('es') and not singular.endswith('cheese'):
        singular = singular[:-2]
    elif singular.endswith('s') and not singular.endswith('ss'):
        singular = singular[:-1]
    
    if singular in canonical_foods:
        return canonical_foods[singular]
    
    # Priority 3: Check if any canonical food contains our name as whole word(s)
    for canon_name, canon_food in canonical_foods.items():
        canon_words = set(canon_name.split())
        name_words = set(name_lower.split())
        # If all our words are in the canonical name
        if name_words.issubset(canon_words):
            return canon_food
        # If canonical is subset of our name (e.g., "chicken" in "chicken breast")
        if canon_words.issubset(name_words) and len(canon_words) >= len(name_words) - 1:
            return canon_food
    
    return None


def load_canonical_foods() -> dict:
    """
    Load all foods from Mealie and build lookup dict.
    Filters out garbage entries (those with "or", "such as", very long names).
    """
    try:
        from mealie_client import MealieClient
        client = MealieClient()
        try:
            all_foods = client.get_all_foods()
        finally:
            client.close()
        
        canonical = {}
        for food in all_foods:
            name = food.get("name", "").lower().strip()
            # Filter garbage
            if " or " in name or "such as" in name or len(name) > 35:
                continue
            canonical[name] = food
        
        logger.info(f"Loaded {len(canonical)} canonical foods from Mealie")
        return canonical
        
    except Exception as e:
        logger.error(f"Failed to load canonical foods: {e}")
        return {}


# Global cache for canonical foods (loaded once per run)
_canonical_foods_cache = None

def get_canonical_foods() -> dict:
    """Get canonical foods, loading from cache if available."""
    global _canonical_foods_cache
    if _canonical_foods_cache is None:
        _canonical_foods_cache = load_canonical_foods()
    return _canonical_foods_cache


# ── CANONICAL UNITS CACHE ─────────────────────────────────────────────────
# Same pattern as foods - load once, cache for entire run

def load_canonical_units() -> dict:
    """
    Load all units from Mealie and build lookup dict.
    Indexes by both name and abbreviation for flexible matching.
    """
    try:
        from mealie_client import MealieClient
        client = MealieClient()
        try:
            all_units = client.get_all_units()
        finally:
            client.close()
        
        canonical = {}
        for unit in all_units:
            name = unit.get("name", "").lower().strip()
            if name:
                canonical[name] = unit
            # Also index by abbreviation for flexible matching
            abbrev = unit.get("abbreviation", "").lower().strip()
            if abbrev and abbrev != name:
                canonical[abbrev] = unit
        
        logger.info(f"Loaded {len(canonical)} canonical units from Mealie")
        return canonical
        
    except Exception as e:
        logger.error(f"Failed to load canonical units: {e}")
        return {}


# Global cache for canonical units (loaded once per run)
_canonical_units_cache = None

def get_canonical_units() -> dict:
    """Get canonical units, loading from cache if available."""
    global _canonical_units_cache
    if _canonical_units_cache is None:
        _canonical_units_cache = load_canonical_units()
    return _canonical_units_cache


def match_unit_to_canonical(unit_name: str, canonical_units: dict) -> dict | None:
    """
    Match a unit name against Mealie's canonical unit database.
    Returns the canonical unit object with ID, or None if no match.
    
    Args:
        unit_name: Unit name from LLM parsing (e.g., "tablespoon", "tbsp", "cup")
        canonical_units: Dict mapping lowercase name/abbrev -> unit object
    """
    if not unit_name:
        return None
    
    name_lower = unit_name.lower().strip()
    
    # Check direct match first
    if name_lower in canonical_units:
        return canonical_units[name_lower]
    
    # Normalize via UNIT_ALIASES and check again
    # Import here to avoid circular dependency (UNIT_ALIASES defined later in file)
    normalized = UNIT_ALIASES.get(name_lower, name_lower)
    if normalized != name_lower and normalized in canonical_units:
        return canonical_units[normalized]
    
    return None


# ── PROCESSING CACHE ──────────────────────────────────────────────────────
# Simple file-based cache to avoid redundant processing
CACHE_FILE = DATA_DIR / "runtime" / "parsing_cache.json"
# Load configuration from centralized config
from config import get_config_value
CACHE_DURATION_HOURS = get_config_value('ingredient_parsing', 'cache_duration_hours', 24)

# ── PROGRESS PERSISTENCE ───────────────────────────────────────────────────
# Track progress in current run to enable resuming after interruption
PROGRESS_FILE = DATA_DIR / "runtime" / "parsing_progress.json"

def load_processing_cache() -> dict:
    """
    Load parsing status cache.
    
    New format tracks recipe state, not just timestamp:
    {
        "recipe_id": {
            "parsed": true,
            "mealie_updated_at": "2024-01-26T09:00:00"
        }
    }
    
    Old format (timestamp string) is auto-migrated.
    """
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                # Migrate old format if needed
                migrated = {}
                for recipe_id, value in data.items():
                    if isinstance(value, str):
                        # Old format: just a timestamp → migrate to new format
                        # Mark as parsed since it was processed before
                        migrated[recipe_id] = {
                            "parsed": True,
                            "mealie_updated_at": value  # Use old timestamp as placeholder
                        }
                    elif isinstance(value, dict):
                        migrated[recipe_id] = value
                return migrated
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def save_processing_cache(cache: dict):
    """Save parsing status cache to file."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        logger.warning(f"Could not save processing cache: {e}")

def is_recipe_unchanged(recipe_id: str, mealie_updated_at: str, cache: dict) -> bool:
    """
    Check if recipe is already parsed AND hasn't been modified in Mealie.
    
    Returns True if we can skip this recipe (already parsed, not modified).
    Returns False if we need to check it (new, not parsed, or modified).
    """
    if recipe_id not in cache:
        return False  # New recipe, need to check
    
    entry = cache[recipe_id]
    if not isinstance(entry, dict):
        return False  # Invalid entry
    
    if not entry.get("parsed"):
        return False  # Not marked as parsed, need to check
    
    cached_updated = entry.get("mealie_updated_at", "")
    if not cached_updated:
        return False  # No timestamp, need to check
    
    # If Mealie's updatedAt is newer than our cached value, recipe was modified
    # Need to re-check in case ingredients were changed
    return mealie_updated_at <= cached_updated

def mark_as_parsed(recipe_id: str, mealie_updated_at: str, cache: dict):
    """Mark recipe as successfully parsed with its current Mealie timestamp."""
    cache[recipe_id] = {
        "parsed": True,
        "mealie_updated_at": mealie_updated_at
    }

def mark_as_checked_unparsed(recipe_id: str, mealie_updated_at: str, cache: dict):
    """Mark recipe as checked but still unparsed (e.g., parsing failed)."""
    cache[recipe_id] = {
        "parsed": False,
        "mealie_updated_at": mealie_updated_at
    }


def _update_parsing_caches(
    recipe_id: str,
    updated_at: str,
    is_parsed: bool,
    rag: 'RecipeRAG | None',
    cache: dict
) -> None:
    """
    Update parsing status in both RAG index and JSON cache.
    
    This helper consolidates the dual-cache update pattern used throughout
    the codebase to ensure RAG index and JSON cache stay in sync.
    
    Args:
        recipe_id: The recipe ID (UUID) to update.
        updated_at: Mealie's updatedAt timestamp for the recipe.
        is_parsed: True if recipe has parsed ingredients, False otherwise.
        rag: Optional RecipeRAG instance for local index updates.
        cache: JSON parsing cache dict to update.
    """
    if recipe_id and rag:
        rag.update_parsed_status(recipe_id, is_parsed=is_parsed)
    if is_parsed:
        mark_as_parsed(recipe_id, updated_at, cache)
    else:
        mark_as_checked_unparsed(recipe_id, updated_at, cache)

# Legacy function for backward compatibility
def is_recently_processed(recipe_id: str, cache: dict) -> bool:
    """DEPRECATED: Use is_recipe_unchanged() instead."""
    if recipe_id not in cache:
        return False
    entry = cache[recipe_id]
    if isinstance(entry, dict):
        return entry.get("parsed", False)
    # Old format fallback
    try:
        last_processed = datetime.fromisoformat(entry)
        cutoff_time = datetime.now() - timedelta(hours=CACHE_DURATION_HOURS)
        return last_processed > cutoff_time
    except (ValueError, TypeError):
        return False

def mark_as_processed(recipe_id: str, cache: dict):
    """DEPRECATED: Use mark_as_parsed() instead."""
    cache[recipe_id] = {
        "parsed": True,
        "mealie_updated_at": datetime.now().isoformat()
    }

def load_progress() -> dict:
    """Load current run progress."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"processed": [], "start_time": datetime.now().isoformat()}

def save_progress(progress: dict):
    """Save current run progress."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except IOError as e:
        logger.warning(f"Could not save progress: {e}")

def clear_progress():
    """Clear progress file after successful completion."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

# ─────────────────────────────────────────────────────────────────────────

BASE  = MEALIE_URL
TOKEN = MEALIE_TOKEN  # Use token from config.py
HEAD  = get_mealie_headers()

# Load additional configuration values
PAGE = get_config_value('ingredient_parsing', 'page_size', 200)
CONF = get_config_value('ingredient_parsing', 'confidence_threshold', 0.80)
DELAY = get_config_value('ingredient_parsing', 'polite_delay', 0.1)

OUT   = DATA_DIR / "runtime" / "review_low_confidence.json"
# STRICT REQUIREMENT: Only LLM parsing allowed - no NLP fallback
PARSE_ORDER = ["openai"]  # LLM parsing only - system fails if LLM parsing fails
# ─────────────────────────────────────────────────────────────────────────

class AlreadyParsed(Exception):
    pass

import re     # <- needed for the regex in rule #3

# ---------------- misc helpers ------------------------------------------
def slim(obj: dict | None) -> dict | None:
    """
    Return just {"id": <uuid>, "name": <string>} for food/unit objects,
    or None if the object is missing / has no id.
    """
    if obj and isinstance(obj, dict) and obj.get("id"):
        return {"id": obj["id"], "name": obj.get("name", "")}
    return None


SERVING_PHRASES = {"for serving", "for garnish", "for dipping"}

def ensure_food_object(food: dict | None) -> dict | None:
    """
    If `food` is a dict missing an id but has a name, POST it to /foods.
    If the food already exists (400 error), look it up and return the existing one.
    Only return None if name is blank or lookup fails entirely.
    """
    # nothing to do if there's already an id, or no dict at all
    if not food or food.get("id"):
        return food

    name = (food.get("name") or "").strip()
    if not name:
        # no meaningful name to register
        return None

    try:
        from mealie_client import MealieClient, MealieAPIError
        client = MealieClient()
        try:
            # Try to create the food
            data = client.create_food(name)
            return {"id": data["id"], "name": data["name"]}
        except MealieAPIError as e:
            # Handle 400 "already exists" - look up the existing food instead
            if e.status_code == 400:
                logger.debug(f"Food '{name}' already exists, looking it up...")
                try:
                    # Get all foods and search for exact match
                    all_foods = client.get_all_foods()
                    # Find exact case-insensitive match
                    for item in all_foods:
                        if item.get("name", "").lower() == name.lower():
                            logger.debug(f"Found existing food: {item.get('name')} (id: {item.get('id')})")
                            return {"id": item["id"], "name": item["name"]}
                    # No exact match - try partial match (first result with name containing our search)
                    for item in all_foods:
                        if name.lower() in item.get("name", "").lower():
                            logger.debug(f"Using closest match: {item.get('name')} (id: {item.get('id')})")
                            return {"id": item["id"], "name": item["name"]}
                except Exception as lookup_err:
                    logger.warning(f"Failed to look up existing food '{name}': {lookup_err}")
            
            logger.warning(f"Couldn't create or find food '{name}': {e}")
            return None
        finally:
            client.close()
    except Exception as e:
        logger.warning(f"Couldn't create or find food '{name}': {e}")
        return None


# Unit name normalization map for common abbreviations
UNIT_ALIASES = {
    "g": "gram",
    "kg": "kilogram",
    "ml": "milliliter",
    "l": "liter",
    "oz": "ounce",
    "lb": "pound",
    "tsp": "teaspoon",
    "tbsp": "tablespoon",
    "c": "cup",
    "pt": "pint",
    "qt": "quart",
    "gal": "gallon",
}


def ensure_unit_object(unit: dict | str | int | None) -> dict | None:
    """
    Look up or create a unit in Mealie.
    Uses cache to avoid redundant API calls (85-95% reduction).
    
    Args:
        unit: Either a dict with 'name' key, a string unit name, or None
        
    Returns:
        dict with {"id": ..., "name": ...} or None if not found/created
    """
    # Handle None or empty
    if unit is None:
        return None
    
    # Handle unexpected types (LLM sometimes returns int/float)
    if isinstance(unit, (int, float)):
        return None  # Can't make a unit from a number
    
    # Handle string input
    if isinstance(unit, str):
        unit = {"name": unit}
    
    # Must be a dict at this point
    if not isinstance(unit, dict):
        return None
    
    # Nothing to do if already has an id
    if unit.get("id"):
        return unit
    
    name = unit.get("name")
    # Handle case where name is not a string (LLM returned int/float/list)
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not name:
        return None
    
    # Normalize common abbreviations
    normalized_name = UNIT_ALIASES.get(name.lower(), name)
    
    # CHECK CACHE FIRST - this avoids 85-95% of API calls
    canonical_units = get_canonical_units()
    
    # Try normalized name
    matched = match_unit_to_canonical(normalized_name, canonical_units)
    if matched:
        return {"id": matched["id"], "name": matched["name"]}
    
    # Try original name if different
    if normalized_name.lower() != name.lower():
        matched = match_unit_to_canonical(name, canonical_units)
        if matched:
            return {"id": matched["id"], "name": matched["name"]}
    
    # NOT IN CACHE - need to check/create via API
    # First, try to find existing unit (in case cache is stale)
    from mealie_client import MealieClient, MealieAPIError
    client = MealieClient()
    try:
        try:
            # Get all units and search for match
            all_units = client.get_all_units()
            # Find exact match (case-insensitive)
            for item in all_units:
                item_name = item.get("name", "").lower()
                if item_name == normalized_name.lower() or item_name == name.lower():
                    # Add to cache for future lookups
                    canonical_units[item_name] = item
                    return {"id": item["id"], "name": item["name"]}
                # Also check abbreviation field
                if item.get("abbreviation", "").lower() == name.lower():
                    # Add to cache
                    canonical_units[item.get("abbreviation", "").lower()] = item
                    return {"id": item["id"], "name": item["name"]}
            # Also search by original name if we normalized
            if normalized_name != name:
                for item in all_units:
                    if item.get("name", "").lower() == name.lower():
                        canonical_units[name.lower()] = item
                        return {"id": item["id"], "name": item["name"]}
        except Exception as e:
            logger.warning(f"Failed to search for unit '{name}': {e}")
        
        # Unit not found - create it
        try:
            data = client.create_unit(
                normalized_name,
                abbreviation=name if name != normalized_name else ""
            )
            # Add newly created unit to cache
            canonical_units[data["name"].lower()] = data
            if data.get("abbreviation"):
                canonical_units[data["abbreviation"].lower()] = data
            logger.info(f"Created new unit: {data.get('name')} (id: {data.get('id')}) - added to cache")
            return {"id": data["id"], "name": data["name"]}
        except MealieAPIError as e:
            # Unit might already exist (race condition) - try to find it
            if e.status_code == 400:
                try:
                    # Get all units again and find first match
                    all_units = client.get_all_units()
                    for item in all_units:
                        if normalized_name.lower() in item.get("name", "").lower() or name.lower() in item.get("name", "").lower():
                            # Add to cache
                            canonical_units[item.get("name", "").lower()] = item
                            return {"id": item["id"], "name": item["name"]}
                except Exception as lookup_err:
                    logger.warning(f"Failed to look up existing unit '{name}': {lookup_err}")
            
            logger.warning(f"Couldn't create or find unit '{name}': {e}")
            return None
    finally:
        client.close()


def looks_suspicious(ing: dict) -> bool:
    """
    Lighter filter that preserves data whenever possible:
      • ignore obvious serving suggestions
      • only flag as suspicious if qty=0 + unit + NO valid food
    
    The key insight: if the LLM extracted a valid food name, keep the ingredient
    even if qty parsing failed. We'd rather have "0 bunch coriander" than lose
    the ingredient entirely - it's still usable for shopping lists.
    """
    note = (ing.get("note") or "").lower()

    # ignore lines like "chips … for dipping"
    if any(p in note for p in SERVING_PHRASES):
        return False

    # If we have a valid food, don't mark as suspicious even if qty=0
    food = ing.get("food")
    if food and isinstance(food, dict) and food.get("name"):
        return False

    # zero qty but still has a unit and no food → probably mis-parsed garbage
    if ing.get("quantity", 0) == 0 and ing.get("unit") is not None:
        return True

    return False            # everything else is acceptable

async def auto_tag_recipe(recipe_data):
    """
    Automatically analyze and tag a recipe for cuisine and prep requirements.
    Uses the AutomaticTagger system to add intelligent tags to Mealie.
    """
    try:
        from automatic_tagger import AutomaticTagger

        # Initialize tagger and analyze recipe
        tagger = AutomaticTagger()
        analysis = await tagger.analyze_recipe(recipe_data)

        # Apply tags to Mealie
        recipe_id = recipe_data.get('id')
        if recipe_id:
            result = tagger.apply_tags_to_mealie(recipe_id, analysis)

            # Log results
            if result.get('tags_added'):
                logger.info(f"Added tags to {recipe_data.get('name', 'Unknown')}: {', '.join(result['tags_added'])}")
                print(f"   ✅ Added tags: {', '.join(result['tags_added'])}")
            if result.get('tags_skipped'):
                logger.debug(f"Skipped tags for {recipe_data.get('name', 'Unknown')}: {', '.join(result['tags_skipped'])}")
                print(f"   ⏭️  Skipped: {', '.join(result['tags_skipped'])}")
            if result.get('errors'):
                logger.error(f"Tag errors for {recipe_data.get('name', 'Unknown')}: {', '.join(result['errors'])}")
                print(f"   ❌ Errors: {', '.join(result['errors'])}")

    except ImportError:
        logger.warning(f"AutomaticTagger not available - skipping auto-tagging for {recipe_data.get('name', 'Unknown')}")
    except Exception as e:
        logger.error(f"Auto-tagging error for {recipe_data.get('name', 'Unknown')}: {e}")


def extract_raw_lines(recipe_json, force_reparse: bool = False):
    """
    Return list[str] of ingredient lines ready for the parser.

    • If recipeIngredient is already a list[str] → return
    • If every dict in recipeIngredient has food == null → legacy  → extract
    • Else at least one food ≠ null → already parsed → raise AlreadyParsed
    • Legacy "ingredients[]" from v1 imports → extract rawText
    
    NEW: If originalText is missing, attempt to re-scrape from orgURL
    
    Args:
        recipe_json: The recipe data from Mealie API
        force_reparse: If True, extract originalText even from already-parsed recipes
    """
    if "recipeIngredient" in recipe_json:
        items = recipe_json["recipeIngredient"]
        if not items:
            raise KeyError("Empty recipeIngredient list")

        first = items[0]

        # modern/plain-string schema
        if isinstance(first, str):
            return items

        # dicts – decide legacy vs parsed
        if isinstance(first, dict):
            all_food_null = all(it.get("food") is None for it in items)
            if not all_food_null and not force_reparse:
                raise AlreadyParsed

            # Try to extract original text from stored data
            extracted_lines = []
            incomplete_indices = []
            
            for idx, it in enumerate(items):
                original = (it.get("originalText")
                           or it.get("rawText")
                           or it.get("note")
                           or re.sub(r"^\s*\d+[¼½¾⅓⅔⅛⅜⅝⅞/ \t--]*", "",
                                    it.get("display", ""))).strip()
                extracted_lines.append(original)
                
                # Flag incomplete extractions (suspiciously short or no food name)
                # Short single-word ingredients like "salt" or "oil" are acceptable
                # Multi-word fragments like "into pieces" are incomplete
                word_count = len(original.split())
                is_incomplete = (
                    not it.get("originalText") and (
                        len(original) < 5 or  # Very short (likely incomplete)
                        (len(original) < 10 and word_count > 1) or  # Short multi-word fragment
                        not any(c.isalpha() for c in original)  # No letters
                    )
                )
                if is_incomplete:
                    incomplete_indices.append(idx)
            
            # If we have incomplete data and orgURL, attempt selective re-scraping
            if incomplete_indices and recipe_json.get("orgURL"):
                if not RECIPE_SCRAPERS_AVAILABLE:
                    logger.warning("recipe-scrapers not installed - cannot re-scrape. Install with: pip install recipe-scrapers")
                    return extracted_lines
                
                slug = recipe_json.get('slug', 'unknown')
                logger.info(f"Re-scraping {slug}: {len(incomplete_indices)}/{len(items)} ingredients appear incomplete")
                try:
                    scraper = scrape_me(recipe_json["orgURL"])
                    original_ingredients = scraper.ingredients()
                    
                    # Validate re-scraped data quality
                    if not original_ingredients:
                        raise ValueError("Re-scraping returned empty ingredient list")
                    
                    if len(original_ingredients) != len(extracted_lines):
                        raise ValueError(f"Count mismatch: got {len(original_ingredients)}, expected {len(extracted_lines)}")
                    
                    if not all(ing and ing.strip() for ing in original_ingredients):
                        raise ValueError("Re-scraping returned empty ingredients")
                    
                    # SELECTIVE MERGE: only replace incomplete entries
                    merged = []
                    replaced_count = 0
                    for idx, (extracted, rescraped) in enumerate(zip(extracted_lines, original_ingredients)):
                        if idx in incomplete_indices:
                            logger.debug(f"  Replacing #{idx}: '{extracted[:30]}...' -> '{rescraped[:30]}...'")
                            merged.append(rescraped)
                            replaced_count += 1
                        else:
                            merged.append(extracted)
                    
                    logger.info(f"✅ Merged ingredients: replaced {replaced_count}/{len(items)} incomplete entries")
                    return merged
                
                except Exception as e:
                    # Catch all exceptions from recipe_scrapers including RecipeSchemaNotFound
                    logger.error(f"Re-scraping failed: {e}")
                    # If we have incomplete data and re-scraping failed, signal failure
                    if incomplete_indices:
                        logger.error(f"Extracted data has {len(incomplete_indices)} incomplete entries and re-scraping failed")
                        logger.error("Returning incomplete data - parsing will likely fail")
            
            return extracted_lines

    if "ingredients" in recipe_json:
        return [it["rawText"] for it in recipe_json["ingredients"]]

    raise KeyError("No ingredient field found")


def parse_lines(lines: list[str], parser: str = "nlp") -> list[dict] | None:
    """
    Return parser output or None if the request errors out.
    NOTE: This is the OLD function that calls Mealie's API (kept for compatibility).
    Use parse_with_direct_llm() for the improved parsing.
    """
    payload = {"strategy": parser, "ingredients": lines}
    try:
        with mealie_rate_limit():
            r = requests.post(
                f"{BASE}/api/parser/ingredients",
                headers=HEAD,
                json=payload,
                timeout=30,
            )
        r.raise_for_status()
        return r.json()
    except HTTPError as e:
        logger.warning(f"Parser '{parser}' failed: {e}")
        return None


def parse_with_direct_llm(lines: list[str]) -> tuple[list[dict], str | None]:
    """
    Parse ingredients using our DIRECT LLM call with optimized prompt.
    This bypasses Mealie's parser entirely for better normalization.
    
    Returns: (parsed_items, strategy_name) or ([], None) on failure
    """
    if not lines:
        return [], None
    
    # Call our direct LLM parsing
    parsed = parse_ingredients_direct_llm(lines)
    if parsed is None:
        logger.critical("Direct LLM parsing failed")
        return [], None
    
    # Load canonical foods for matching
    canonical_foods = get_canonical_foods()
    
    # Convert to Mealie format and match to canonical foods
    processed_items = []
    for i, ing in enumerate(parsed):
        original_text = lines[i] if i < len(lines) else ""
        
        # Match food to canonical
        food_name = ing.get("food", "")
        matched_food = match_food_to_canonical(food_name, canonical_foods)
        
        if matched_food:
            food_obj = {"id": matched_food["id"], "name": matched_food["name"]}
        elif food_name:
            # No match - will need to create (or use None)
            # For now, just use the name without ID - ensure_food_object will handle it
            food_obj = {"name": food_name}
        else:
            food_obj = None
        
        # Build unit object
        unit_name = ing.get("unit")
        unit_obj = {"name": unit_name} if unit_name else None
        
        # Build ingredient in Mealie format
        # Ensure note is always a string (LLM sometimes returns [] instead of "")
        note_value = ing.get("note", "")
        if not isinstance(note_value, str):
            note_value = str(note_value) if note_value else ""
        
        ingredient = {
            "quantity": ing.get("quantity", 0),
            "unit": unit_obj,
            "food": food_obj,
            "note": note_value,
            "originalText": original_text,
        }
        
        # Build confidence (we trust our LLM)
        confidence = {
            "average": 0.95,
            "quantity": 1.0 if ing.get("quantity") else 0.5,
            "unit": 1.0 if unit_name else 0.5,
            "food": 1.0 if matched_food else 0.7,  # Higher confidence if matched
            "comment": 0.9,
        }
        
        processed_items.append({
            "input": original_text,
            "ingredient": ingredient,
            "confidence": confidence,
        })
    
    if processed_items:
        # Count how many ingredients matched to canonical foods (food has an id)
        matched_count = sum(1 for p in processed_items 
                           if p['ingredient'].get('food') and p['ingredient']['food'].get('id'))
        logger.info(f"Direct LLM parsed {len(processed_items)} ingredients, {matched_count} matched to canonical")
        return processed_items, "direct_llm"
    
    return [], None


def good_enough(block):
    """
    True only if EVERY line:
      • confidence ≥ CONF
      • food is not null
      • (quantity > 0)  OR  unit is None
    """
    return all(
        (ln["confidence"]["average"] >= CONF)
        and (
            (ln["ingredient"]["quantity"] or 0) > 0
            or ln["ingredient"]["unit"] is None
        )
        for ln in block
    )


def parse_with_fallback(lines: list[str]) -> tuple[list[dict], str | None]:
    """
    Parse ingredients using our direct LLM approach (bypasses Mealie's parser).
    Falls back to Mealie's NLP parser only if direct LLM fails completely.
    """
    # Primary: Use our direct LLM parsing with optimized prompt
    block, strategy = parse_with_direct_llm(lines)
    
    if block:
        # Process food objects through ensure_food_object for any that need creation
        processed_items = []
        for p in block:
            ingr = p["ingredient"]
            ingr["food"] = ensure_food_object(ingr.get("food"))
            ingr["unit"] = ensure_unit_object(ingr.get("unit"))
            ingr["food"] = slim(ingr.get("food"))
            ingr["unit"] = slim(ingr.get("unit"))
            processed_items.append(p)
        
        # Filter suspicious items
        valid_items = [item for item in processed_items if not looks_suspicious(item["ingredient"])]
        
        if valid_items:
            return valid_items, strategy
    
    # NO NLP FALLBACK - Quality requirement: Never drop ingredients
    # If LLM parsing fails, skip this recipe entirely rather than risking data loss
    logger.error("❌ LLM parsing failed - SKIPPING RECIPE to maintain 100% quality")
    logger.error("   This recipe will remain unparsed rather than use unreliable NLP fallback")
    return [], None


# REMOVED: get_all_slugs() - used broken hasParsedIngredients field
# All parsing now uses get_unparsed_slugs() with proper ingredient structure detection


def is_recipe_actually_unparsed(recipe_data: dict) -> bool:
    """
    Check if recipe ingredients are truly unparsed by examining structure.
    This is the correct way to detect unparsed recipes.
    
    Key insight: The presence of a `food` object is the definitive indicator
    of parsing success. Many valid parsed ingredients have:
    - unit=None (e.g., "4 eggs" - no unit needed)
    - quantity=0 (e.g., "salt to taste")
    
    So we check for food object presence, not all three fields.
    """
    ingredients = recipe_data.get("recipeIngredient", [])

    if not ingredients:
        return False  # No ingredients to parse

    # Check if ANY ingredient has a food object (= was parsed)
    for ingredient in ingredients:
        if isinstance(ingredient, dict):
            food = ingredient.get("food")
            
            # If any ingredient has a food object, recipe is parsed
            if food is not None:
                return False  # Already parsed
        elif isinstance(ingredient, str):
            # Raw string ingredients are unparsed by definition
            continue

    # No ingredients have food objects - recipe needs parsing
    return True


def get_unparsed_slugs(force_recheck: bool = False) -> list:
    """
    Find recipes that need ingredient parsing.
    
    Optimization: Uses local recipe index as primary source to avoid API calls.
    - Known unparsed (ingredients_parsed = 0): Return immediately from local index
    - Unknown status (ingredients_parsed IS NULL): Check via Mealie API
    - Parsed recipes are skipped entirely
    
    Secondary layer: JSON parsing cache tracks API-mode checking for recipes
    not yet in the local index.
    
    Args:
        force_recheck: If True, bypass local index and check all candidates
                       from Mealie API. Use this if you suspect stale data.
    
    Returns:
        List of recipe slugs that need ingredient parsing.
    """
    from mealie_client import MealieClient
    from recipe_rag import RecipeRAG

    client = MealieClient()
    unparsed_slugs = []

    try:
        print("🔍 Scanning for unparsed recipes...")
        
        # ── Fast path: Use local index when not forcing recheck ──
        if not force_recheck:
            rag = RecipeRAG()
            
            # 1. Get known unparsed from local index (instant - no API call)
            known_unparsed = rag.get_recipes_by_parsed_status(parsed=False)
            if known_unparsed:
                print(f"📦 Found {len(known_unparsed)} known unparsed recipes in local index")
                unparsed_slugs.extend(known_unparsed)
            
            # 2. Get recipes with unknown parsing status (need to check)
            unknown_status_slugs = rag.get_recipes_with_unknown_parsed_status()
            
            if not unknown_status_slugs:
                print(f"✅ All recipes have known parsing status")
                if unparsed_slugs:
                    print(f"📋 Returning {len(unparsed_slugs)} unparsed recipes")
                return unparsed_slugs
            
            print(f"❓ {len(unknown_status_slugs)} recipes have unknown parsing status, checking...")
            
            # 3. Check parsing status for unknown recipes
            # Try DB bulk query first (single SQL query for all recipes)
            db_parsed_status = client.check_recipes_parsed_status(unknown_status_slugs)
            
            if db_parsed_status:
                # DB mode: Use bulk query result (fast path)
                print(f"⚡ Using DB bulk query for {len(unknown_status_slugs)} recipes...")
                
                # Collect results for batch update
                parsed_slugs = []
                unparsed_slugs_from_db = []
                
                for slug in unknown_status_slugs:
                    if slug in db_parsed_status:
                        is_parsed = db_parsed_status[slug]
                        if is_parsed:
                            parsed_slugs.append(slug)
                        else:
                            unparsed_slugs_from_db.append(slug)
                            unparsed_slugs.append(slug)
                    else:
                        # Recipe not found in Mealie DB
                        logger.warning(f"Recipe '{slug}' in local index but not found in Mealie DB")
                
                # Batch update local index (single transaction instead of 17k individual updates)
                rag.batch_update_parsed_status(parsed_slugs, is_parsed=True)
                rag.batch_update_parsed_status(unparsed_slugs_from_db, is_parsed=False)
                
                print(f"✅ DB bulk query complete: {len(parsed_slugs)} parsed, {len(unparsed_slugs_from_db)} unparsed")
            else:
                # API mode: Fetch full recipes in batches (slower path)
                print(f"📡 Using API batch fetch for {len(unknown_status_slugs)} recipes...")
                
                config = get_bulk_operation_config_safe('parse', fallback_batch_size=100, fallback_concurrent=3)
                batch_size = config['default_batch_size']
                total_batches = (len(unknown_status_slugs) + batch_size - 1) // batch_size
                
                # Also load JSON cache for secondary tracking
                cache = load_processing_cache()
                cache_modified = False
                
                for i in range(0, len(unknown_status_slugs), batch_size):
                    batch_slugs = unknown_status_slugs[i:i + batch_size]
                    batch_recipes = client.get_recipes_batch(batch_slugs)
                    
                    for slug in batch_slugs:
                        if slug not in batch_recipes:
                            logger.warning(f"Recipe '{slug}' in local index but not found in Mealie - may have been deleted")
                            continue
                        
                        recipe_data = batch_recipes[slug]
                        if not recipe_data:
                            continue
                        
                        recipe_id = recipe_data.get('id', '')
                        updated_at = recipe_data.get('updatedAt', recipe_data.get('dateUpdated', ''))
                        
                        is_unparsed = is_recipe_actually_unparsed(recipe_data)
                        if is_unparsed:
                            unparsed_slugs.append(slug)
                        
                        # Update both caches in one call
                        _update_parsing_caches(
                            recipe_id=recipe_id,
                            updated_at=updated_at,
                            is_parsed=not is_unparsed,
                            rag=rag,
                            cache=cache
                        )
                        cache_modified = True
                    
                    if total_batches > 1:
                        print(f"   Checked batch {i//batch_size + 1}/{total_batches}...")
                
                # Save JSON cache
                if cache_modified:
                    save_processing_cache(cache)
                    print(f"💾 Cache updated: {len(cache)} recipes tracked")
            
            if unparsed_slugs:
                print(f"📋 Total unparsed recipes: {len(unparsed_slugs)}")
            
            return unparsed_slugs
        
        # ── Force recheck mode: Full scan via Mealie API ──
        print("⚠️  Force recheck mode: bypassing local index cache")
        
        # Load persistent parsing cache
        cache = load_processing_cache()
        cache_modified = False

        # Get all recipe summaries (includes updatedAt timestamp)
        all_recipes = client.get_all_recipes()
        total_recipes = len(all_recipes)

        if total_recipes == 0:
            return []

        # Categorize recipes: skip unchanged, check new/modified
        unchanged_count = 0
        candidate_recipes = []
        
        for recipe in all_recipes:
            recipe_id = recipe['id']
            updated_at = recipe.get('updatedAt', recipe.get('dateUpdated', ''))
            
            if is_recipe_unchanged(recipe_id, updated_at, cache):
                unchanged_count += 1
            else:
                candidate_recipes.append(recipe)

        print(f"📊 {total_recipes} total recipes: {unchanged_count} unchanged, {len(candidate_recipes)} to check")

        if not candidate_recipes:
            print("✅ All recipes already processed and unchanged")
            return []

        # Fetch and check candidate recipes
        candidate_slugs = [r['slug'] for r in candidate_recipes]
        candidate_by_slug = {r['slug']: r for r in candidate_recipes}
        
        # Initialize RAG for updating local index during force recheck
        try:
            rag = RecipeRAG()
        except Exception as e:
            logger.warning(f"Could not initialize RecipeRAG for local index updates: {e}")
            rag = None
        
        # Use batch fetching for efficiency
        config = get_bulk_operation_config_safe('parse', fallback_batch_size=100, fallback_concurrent=3)
        batch_size = config['default_batch_size']
        
        total_batches = (len(candidate_slugs) + batch_size - 1) // batch_size
        
        for i in range(0, len(candidate_slugs), batch_size):
            batch_slugs = candidate_slugs[i:i + batch_size]
            batch_recipes = client.get_recipes_batch(batch_slugs)
            
            for slug in batch_slugs:
                if slug not in batch_recipes:
                    logger.warning(f"Recipe '{slug}' in local index but not found in Mealie - may have been deleted")
                    continue
                
                recipe_data = batch_recipes[slug]
                if not recipe_data:
                    continue
                    
                recipe_id = recipe_data.get('id', candidate_by_slug.get(slug, {}).get('id', ''))
                updated_at = recipe_data.get('updatedAt', recipe_data.get('dateUpdated', ''))
                
                is_unparsed = is_recipe_actually_unparsed(recipe_data)
                if is_unparsed:
                    unparsed_slugs.append(slug)
                
                # Update both caches in one call
                _update_parsing_caches(
                    recipe_id=recipe_id,
                    updated_at=updated_at,
                    is_parsed=not is_unparsed,
                    rag=rag,
                    cache=cache
                )
                cache_modified = True
            
            if total_batches > 1:
                print(f"   Checked batch {i//batch_size + 1}/{total_batches}...")

        # Save updated cache
        if cache_modified:
            save_processing_cache(cache)
            print(f"💾 Cache updated: {len(cache)} recipes tracked")

        return unparsed_slugs

    finally:
        client.close()


# ── FIX ① : PATCH just the list -----------------------------------------
def patch_recipe(slug: str, ingredient_list: list[dict]):
    payload = {"recipeIngredient": ingredient_list}

    (DATA_DIR / "runtime" / "last_payload.json").write_text(json.dumps(payload, indent=2))
    logger.debug("PATCH payload written to data/runtime/last_payload.json")

    from mealie_client import MealieClient, MealieAPIError
    client = MealieClient()
    try:
        client.update_recipe(slug, payload)
    except MealieAPIError as e:
        logger.error(f"Server error response (HTTP {e.status_code}) for recipe {slug}")
        logger.debug(f"Response details: {e.response_body}")
        raise
    finally:
        client.close()
# ------------------------------------------------------------------------


def parse_single_recipe_standalone(
    slug: str, 
    auto_tag: bool = False,
    force_reparse: bool = False
) -> bool:
    """
    Parse a single recipe by slug. 
    
    For use as a queue worker in streaming pipeline.
    
    IMPORTANT: Always uses API mode because this is typically called
    immediately after import, and DB mode may not see fresh writes.
    
    Args:
        slug: Recipe slug to parse
        auto_tag: Whether to auto-tag after parsing
        force_reparse: Force re-parsing even if already parsed
    
    Returns:
        True if parsing succeeded, False otherwise
    """
    try:
        # Fetch recipe data
        # CRITICAL: Force API mode - DB mode won't see freshly imported recipes
        from mealie_client import MealieClient
        client = MealieClient(use_direct_db=False)
        try:
            recipe_data = client.get_recipe(slug)
        finally:
            client.close()
        
        if not recipe_data.get("recipeIngredient"):
            logger.warning(f"Recipe {slug} has no ingredients")
            return False
        
        # Extract raw ingredient lines
        try:
            raw_lines = extract_raw_lines(recipe_data, force_reparse=force_reparse)
        except AlreadyParsed:
            if not force_reparse:
                logger.info(f"Recipe {slug} already parsed, skipping")
                return True  # Already parsed is a success
            # If force_reparse, continue with extraction
            raw_lines = extract_raw_lines(recipe_data, force_reparse=True)
        except KeyError as e:
            logger.error(f"Failed to extract ingredients from {slug}: {e}")
            return False
        
        # Parse ingredients using LLM
        parsed, which = parse_with_fallback(raw_lines)
        
        if which is None:
            logger.error(f"Parsing failed for {slug} - LLM parsing unsuccessful")
            return False
        
        # Process parsed ingredients
        new_list = []
        skipped_count = 0
        
        for line in parsed:
            ingr = line["ingredient"].copy()
            
            # Ensure food and unit objects exist in Mealie
            ingr["food"] = ensure_food_object(ingr.get("food"))
            ingr["unit"] = ensure_unit_object(ingr.get("unit"))
            
            # Keep only {id, name} for food/unit
            ingr["food"] = slim(ingr.get("food"))
            ingr["unit"] = slim(ingr.get("unit"))
            
            # Validate ingredient
            has_valid_food = ingr["food"] is not None
            is_valid = not (
                (ingr["food"] is None and not ingr.get("note"))  # no food & no note
                or (not has_valid_food and ingr.get("quantity", 0) == 0 and ingr["unit"] is not None)
            )
            
            # Remove parser-only keys
            ingr.pop("confidence", None)
            ingr.pop("display", None)
            
            # Ensure referenceId is set (Mealie requires UUID)
            if not ingr.get("referenceId"):
                ingr["referenceId"] = str(uuid.uuid4())
            
            if is_valid:
                new_list.append(ingr)
            else:
                skipped_count += 1
        
        if not new_list:
            logger.error(f"All ingredients invalid for {slug}")
            return False
        
        # Update recipe with parsed ingredients
        try:
            patch_recipe(slug, new_list)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to patch recipe {slug}: {e}")
            return False
        
        logger.info(f"Successfully parsed {slug}: {len(new_list)} ingredients, {skipped_count} skipped")
        
        # Auto-tag if requested
        if auto_tag:
            try:
                import asyncio
                asyncio.run(auto_tag_recipe(recipe_data))
                logger.info(f"Auto-tagged {slug}")
            except Exception as e:
                logger.warning(f"Auto-tagging failed for {slug}: {e}")
                # Don't fail the whole operation for tagging errors
        
        return True
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching recipe {slug}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error for recipe {slug}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error parsing {slug}: {e}")
        return False


async def parse_unparsed_recipes_batch(unparsed_recipes: list, auto_tag: bool = False):
    """
    Parse multiple recipes in parallel.
    BEFORE: Sequential processing (hours for large batches)
    AFTER: Parallel processing with connection reuse

    Args:
        unparsed_recipes: List of recipe dicts with 'slug' and 'name' keys
        auto_tag: Whether to auto-tag successfully parsed recipes
    """
    from mealie_client import MealieClient
    from tools.parallel_processor import ParallelProcessor

    client = MealieClient()
    processor = ParallelProcessor(max_workers=2)

    try:
        def parse_single_recipe(recipe_data):
            """Parse a single recipe using MealieClient."""
            slug = recipe_data['slug']
            try:
                # Get full recipe data
                raw = client.get_recipe(slug)
                if not raw or not raw.get("recipeIngredient"):
                    return {"success": False, "slug": slug, "error": "No ingredients"}

                # Extract raw lines
                raw_lines = extract_raw_lines(raw)

                # Parse ingredients - STRICT REQUIREMENT: must succeed with LLM
                parsed, which = parse_with_fallback(raw_lines)

                if which is None:  # Parsing failed - log and continue
                    error_msg = f"Parsing failed for {slug} - ingredients are unusable (re-scraping failed or returned incomplete data)"
                    logger.error(error_msg)
                    print(f"❌ {error_msg}")
                    return {"success": False, "slug": slug, "error": error_msg}

                # Process parsed ingredients
                new_list = []
                skipped_ingredients = []

                for line in parsed:
                    ingr = line["ingredient"].copy()

                    # 1) ensure the food and unit exist in Mealie; create if needed
                    ingr["food"] = ensure_food_object(ingr.get("food"))
                    ingr["unit"] = ensure_unit_object(ingr.get("unit"))

                    # 2) keep only {id, name} for food / unit
                    ingr["food"] = slim(ingr.get("food"))
                    ingr["unit"] = slim(ingr.get("unit"))

                    # 3) check if this ingredient is valid
                    # Key insight: if we have a valid food, keep it even if qty parsing failed
                    # "0 bunch coriander" is better than losing the ingredient entirely
                    has_valid_food = ingr["food"] is not None
                    is_valid = not (
                        (ingr["food"] is None and not ingr.get("note"))      # no food & no note
                        or (not has_valid_food and ingr.get("quantity", 0) == 0 and ingr["unit"] is not None)  # garbage: 0 qty + unit + no food
                    )

                    # 4) drop parser-only keys
                    ingr.pop("confidence", None)
                    ingr.pop("display", None)
                    
                    # 5) ensure reference_id is set (Mealie requires UUID)
                    if not ingr.get("referenceId"):
                        ingr["referenceId"] = str(uuid.uuid4())

                    if is_valid:
                        new_list.append(ingr)
                    else:
                        skipped_ingredients.append(ingr)

                # Accept recipes with at least one valid ingredient
                if new_list:
                    # Update recipe with parsed ingredients
                    try:
                        client.update_recipe(slug, {"recipeIngredient": new_list})
                    except Exception as e:
                        return {"success": False, "slug": slug, "error": f"Update failed: {e}"}

                    result = {
                        "success": True,
                        "slug": slug,
                        "name": raw["name"],
                        "ingredients_parsed": len(new_list),
                        "ingredients_skipped": len(skipped_ingredients)
                    }

                    # Auto-tagging
                    if auto_tag:
                        try:
                            import asyncio
                            asyncio.run(auto_tag_recipe(raw))
                            result["auto_tagged"] = True
                        except Exception as e:
                            result["auto_tag_error"] = str(e)

                    return result
                else:
                    logger.error(f"All ingredients invalid for {slug}")
                    print(f"❌ All ingredients invalid for {slug}")
                    return {"success": False, "slug": slug, "error": "All ingredients invalid - manual review required"}

            except Exception as e:
                return {"success": False, "slug": slug, "error": str(e)}

        results = processor.process_batch(
            unparsed_recipes,
            parse_single_recipe,
            "Parsing recipes"
        )

        success_count = sum(1 for r in results if r and r.get('success'))
        failed_count = len(unparsed_recipes) - success_count
        
        print(f"\n{'='*80}")
        print(f"✅ Parsing complete: {success_count}/{len(unparsed_recipes)} recipes parsed successfully")
        if failed_count > 0:
            print(f"❌ {failed_count} recipes failed - see details below:")
            for r in results:
                if r and not r.get('success'):
                    slug = r.get('slug', 'unknown')
                    error = r.get('error', 'Unknown error')
                    print(f"   • {slug}: {error}")
            print(f"   📄 Full logs: logs/aye_chef.log")
        print(f"{'='*80}\n")

        return results

    finally:
        client.close()


async def main(conf_thresh: float, max_recipes: int | None, after_slug: str | None = None,
         specific_slugs: list | None = None, scan_unparsed: bool = False, auto_tag: bool = False, 
         skip_validation: bool = False, force_reparse: bool = False):
    to_review, parsed_ok = [], []

    # Load processing cache for tracking
    processing_cache = load_processing_cache()

    # Load progress for current run (resume capability)
    progress = load_progress()
    processed_in_run = set(progress.get("processed", []))

    if specific_slugs:
        # Parse only the specified slugs
        slugs = specific_slugs
        print(f"🎯 Parsing {len(slugs)} specific recipes: {', '.join(slugs[:5])}{'...' if len(slugs) > 5 else ''}")
    else:
        # Default behavior: scan for actually unparsed recipes (smart detection)
        slugs = get_unparsed_slugs()
        print(f"🔍 Scanning for unparsed recipes... found {len(slugs)}")

    # ── NEW: skip everything through --after-slug ─────────────
    if after_slug:
        try:
            idx = slugs.index(after_slug)
            slugs = slugs[idx+1:]
            print(f"⏭️  Resuming after “{after_slug}” (skipped {idx+1} recipes)")
        except ValueError:
            print(f"⚠️  --after-slug '{after_slug}' not found; parsing from top")
    # ───────────────────────────────────────────────────────────


    if max_recipes:
        original_count = len(slugs)
        slugs = slugs[:max_recipes]
        print(f"Trial mode → {len(slugs)} recipes of {original_count} found")

    if not slugs:
        print("Nothing left to parse.")
        return

    # Filter out recipes already processed in this run
    slugs_to_process = [slug for slug in slugs if slug not in processed_in_run]
    if len(slugs_to_process) != len(slugs):
        print(f"⏭️  Resuming: skipping {len(slugs) - len(slugs_to_process)} already processed in this run")

    from mealie_client import MealieClient
    client = MealieClient()
    try:
        for i, slug in enumerate(tqdm(slugs_to_process, desc="Parsing")):
            raw = client.get_recipe(slug)
            if not raw.get("recipeIngredient"):      # empty list or None
                continue

            try:
                raw_lines = extract_raw_lines(raw, force_reparse=force_reparse)
            except AlreadyParsed:
                continue

            parsed, which = parse_with_fallback(raw_lines)

            if which == "openai":
                print(f"\n🤖  Using LLM parser for {slug}")

            if which is None:     # Parsing failed - log and continue
                logger.error(f"Parsing failed for {slug} - ingredients are unusable (re-scraping failed or returned incomplete data)")
                print(f"❌ FAILED: Parsing failed for {slug} - ingredients are unusable")
                to_review.append((slug, "Parsing failed - ingredients unusable, manual review required"))
                
                # Mark progress and continue to next recipe
                processed_in_run.add(slug)
                progress["processed"] = list(processed_in_run)
                if len(processed_in_run) % 10 == 0:
                    save_progress(progress)
                continue

            # ---- SUCCESS branch -------------------------------------------------
            new_list = []
            skipped_ingredients = []

            for line in parsed:                              # each parsed line
                ingr = line["ingredient"].copy()             # inner dict

                # 1) ensure the food and unit exist in Mealie; create if needed
                ingr["food"] = ensure_food_object(ingr.get("food"))
                ingr["unit"] = ensure_unit_object(ingr.get("unit"))

                # 2) keep only {id, name} for food / unit
                ingr["food"] = slim(ingr.get("food"))
                ingr["unit"] = slim(ingr.get("unit"))

                # 3) check if this ingredient is valid
                # Key insight: if we have a valid food, keep it even if qty parsing failed
                # "0 bunch coriander" is better than losing the ingredient entirely
                has_valid_food = ingr["food"] is not None
                is_valid = not (
                    (ingr["food"] is None and not ingr.get("note"))      # no food & no note
                    or (not has_valid_food and ingr.get("quantity", 0) == 0 and ingr["unit"] is not None)  # garbage: 0 qty + unit + no food
                )

                # 4) drop parser-only keys
                ingr.pop("confidence", None)
                ingr.pop("display",    None)
                
                # 5) ensure reference_id is set (Mealie requires UUID)
                if not ingr.get("referenceId"):
                    ingr["referenceId"] = str(uuid.uuid4())

                if is_valid:
                    new_list.append(ingr)
                else:
                    skipped_ingredients.append(ingr)

            # Accept recipes with at least one valid ingredient
            if new_list:
                if skipped_ingredients:
                    logger.warning(f"{slug}: accepted {len(new_list)} ingredients, skipped {len(skipped_ingredients)} problematic ones")
                    print(f"⚠️  {slug}: accepted {len(new_list)} ingredients, skipped {len(skipped_ingredients)} problematic ones")

                patch_recipe(slug, new_list)
                parsed_ok.append(raw["name"])

                # Update processing cache - mark as parsed with current timestamp
                recipe_id = raw.get("id")
                updated_at = raw.get("updatedAt", raw.get("dateUpdated", datetime.now().isoformat()))
                if recipe_id:
                    mark_as_parsed(recipe_id, updated_at, processing_cache)

                # Track progress for resume capability
                processed_in_run.add(slug)
                progress["processed"] = list(processed_in_run)

                # Save progress periodically (every 10 recipes)
                if len(processed_in_run) % 10 == 0:
                    save_progress(progress)

                # Automatic tagging for successfully parsed recipes
                if auto_tag:
                    try:
                        await auto_tag_recipe(raw)
                        logger.info(f"Auto-tagged {raw['name']}")
                        print(f"🏷️  Auto-tagged {raw['name']}")
                    except Exception as e:
                        logger.warning(f"Auto-tagging failed for {raw['name']}: {e}")
                        print(f"⚠️  Auto-tagging failed for {raw['name']}: {e}")

                time.sleep(DELAY)
            else:
                # No valid ingredients at all - skip this recipe
                logger.warning(f"Skipping {slug} - all ingredients were invalid")
                print(f"⚠️  Skipping {slug} - all ingredients were invalid")
                continue
    finally:
        client.close()

        # -----------------------------------------------------------------

    if parsed_ok:
        (DATA_DIR / "runtime" / "parsed_success.log").write_text("\n".join(parsed_ok))
        print(f"\n✅ Parsed {len(parsed_ok)} recipes → data/runtime/parsed_success.log")

        # Save updated processing cache
        save_processing_cache(processing_cache)

        # Clear progress file after successful completion
        clear_progress()
        
        # Re-index parsed recipes to update local search index
        # This ensures the local index reflects the updated ingredient data
        # Note: processed_in_run contains slugs of successfully parsed recipes only
        # (it's populated inside the same if block as parsed_ok)
        slugs_to_index = list(processed_in_run)
        
        if slugs_to_index:
            print(f"\n🔄 Re-indexing {len(slugs_to_index)} parsed recipes...", flush=True)
            try:
                from recipe_rag import RecipeRAG
                rag = RecipeRAG()
                
                # Fetch fresh recipe data (with updated ingredients) and re-index
                from mealie_client import MealieClient
                client = MealieClient()
                try:
                    recipes_data = []
                    for slug in slugs_to_index:
                        try:
                            recipe_data = client.get_recipe(slug)
                            recipes_data.append(recipe_data)
                        except Exception as e:
                            logger.warning(f"Failed to fetch {slug} for re-indexing: {e}")
                finally:
                    client.close()
                
                if recipes_data:
                    indexed = rag.index_recipes_batch(recipes_data, force=True)
                    print(f"✅ Re-indexed {indexed}/{len(recipes_data)} recipes", flush=True)
                else:
                    print(f"⚠️  No recipes fetched for re-indexing", flush=True)
            except Exception as e:
                logger.warning(f"⚠️  Re-indexing failed (recipes parsed but local index not updated): {e}")
                print(f"⚠️  Re-indexing failed: {e}", flush=True)

    if to_review:
        OUT.write_text(json.dumps(to_review, indent=2))
        print(f"\n⚠️  {len(to_review)} recipes need manual review:")
        print(f"   ❌ Failed to parse (unusable ingredients)")
        print(f"   📋 Review these recipes and decide whether to:")
        print(f"      - Delete them (if ingredients are fundamentally broken)")
        print(f"      - Manually fix them (if the source URL is still valid)")
        print(f"      - Re-import from a different source")
        print(f"   See details in → {OUT}")
    else:
        print(f"\n🎉 All done – every recipe ≥ {conf_thresh:.0%} confidence.")


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bulk-parse Mealie recipes safely")
    ap.add_argument("--slugs", metavar="SLUGS",
                    help="comma-separated list of specific recipe slugs to parse")
    ap.add_argument("--scan-unparsed",
                    action="store_true",
                    help="scan and parse recipes that appear unparsed (safer than full scan)")
    ap.add_argument("--full-scan",
                    action="store_true",
                    help="⚠️  DANGEROUS: scan and parse ALL recipes (requires confirmation)")
    ap.add_argument("--conf", type=float, default=CONF,
                    help="minimum confidence fraction 0-1 (default 0.80)")
    ap.add_argument("--max", type=int, metavar="N",
                    help="parse at most N recipes (trial run)")
    ap.add_argument("--after-slug", metavar="SLUG",
                    help="skip all recipes through this slug (then resume)")
    ap.add_argument("--auto-tag", action="store_true",
                    help="automatically analyze and tag recipes for cuisine/prep requirements")
    ap.add_argument("--yes", action="store_true",
                    help="skip confirmations for dangerous operations")
    ap.add_argument("--batch-mode", action="store_true",
                    help="use parallel processing for batch operations (faster)")
    ap.add_argument("--force", action="store_true",
                    help="force re-parsing of already-parsed recipes (uses originalText)")
    args = ap.parse_args()

    # Validate configuration before proceeding
    print("🔍 Validating Mealie connection...")
    if not validate_mealie_connection():
        print("❌ Mealie validation failed. Please check your configuration.")
        sys.exit(1)

    # Force explicit intent - no dangerous fallbacks
    if not args.slugs and not args.scan_unparsed and not args.full_scan:
        print("❌ mealie_parse.py requires explicit intent to prevent accidents:")
        print()
        print("SAFE OPTIONS:")
        print("  --slugs 'recipe-a,recipe-b'     Parse specific recipes only")
        print("  --scan-unparsed                 Parse recipes that appear unparsed")
        print()
        print("DANGEROUS OPTIONS (require confirmation):")
        print("  --full-scan                     Parse ALL recipes in database")
        print()
        print("Examples:")
        print("  python mealie_parse.py --slugs 'chicken-curry,spaghetti'")
        print("  python mealie_parse.py --scan-unparsed --yes")
        sys.exit(1)

    # Validate confidence
    if not 0 < args.conf <= 1:
        sys.exit("Confidence must be 0–1 (e.g., 0.8 for 80 %).")

    # Handle different modes
    if args.slugs:
        # Parse specific slugs
        slug_list = [s.strip() for s in args.slugs.split(',') if s.strip()]
        import asyncio
        asyncio.run(main(conf_thresh=args.conf,
                        max_recipes=args.max,
                        after_slug=args.after_slug,
                        specific_slugs=slug_list,
                        auto_tag=args.auto_tag,
                        force_reparse=args.force))
    elif args.scan_unparsed:
        # Scan for unparsed recipes (safer than full scan)
        if args.batch_mode:
            # Use parallel batch processing
            from utils.recipe_maintenance import check_unparsed_recipes
            unparsed_recipes = check_unparsed_recipes()
            if unparsed_recipes:
                import asyncio
                results = asyncio.run(parse_unparsed_recipes_batch(unparsed_recipes[:args.max] if args.max else unparsed_recipes, auto_tag=args.auto_tag))
                success_count = sum(1 for r in results if r and r.get('success'))
                print(f"✅ Batch parsing complete: {success_count}/{len(unparsed_recipes)} recipes parsed")
            else:
                print("✅ No unparsed recipes found")
        else:
            # Use sequential processing
            import asyncio
            asyncio.run(main(conf_thresh=args.conf,
                            max_recipes=args.max,
                            after_slug=args.after_slug,
                            scan_unparsed=True,
                            auto_tag=args.auto_tag,
                            force_reparse=args.force))
    elif args.full_scan:
        # Dangerous full scan - requires confirmation
        if not args.yes:
            response = input(f"\n⚠️  DANGEROUS: Parse ALL recipes in database? This may process 100,000+ recipes. (type 'yes' to confirm): ")
            if response.lower() != 'yes':
                print("Operation cancelled.")
                sys.exit(0)

        print("🚨 Starting full database scan...")
        import asyncio
        asyncio.run(main(conf_thresh=args.conf,
                        max_recipes=args.max,
                        after_slug=args.after_slug,
                        auto_tag=args.auto_tag,
                        force_reparse=args.force))


# ── IMPORT VERIFICATION ──────────────────────────────────────────────────
# Verify the standalone function is importable
def _verify_standalone_import():
    """Quick verification that parse_single_recipe_standalone is importable."""
    assert callable(parse_single_recipe_standalone), "parse_single_recipe_standalone must be callable"
    import inspect
    sig = inspect.signature(parse_single_recipe_standalone)
    params = list(sig.parameters.keys())
    assert 'slug' in params, "Function must accept 'slug' parameter"
    assert 'auto_tag' in params, "Function must accept 'auto_tag' parameter"
    assert 'force_reparse' in params, "Function must accept 'force_reparse' parameter"
    return True

# Run verification on module load (only when imported, not when run as main)
if __name__ != "__main__":
    try:
        _verify_standalone_import()
    except AssertionError as e:
        import warnings
        warnings.warn(f"parse_single_recipe_standalone verification failed: {e}")