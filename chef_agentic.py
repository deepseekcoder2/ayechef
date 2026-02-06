#!/usr/bin/env python3
"""
Agentic Chef (Dry Run) - Tool-Using Weekly Planner
==================================================
This script implements an *agentic* chef loop that plans one meal slot at a time.

Core principles:
- Source of truth: Mealie (recipes + tags + meal history live there)
- Retrieval/index: local RecipeRAG (SQLite + ANN index) is a rebuildable accelerator
- Agent loop: choose cuisine -> weighted random sample from user's recipes -> LLM picks from sample
- Components are enforced contextually: pick -> validate role fit vs primary -> repick within a bounded budget
- Fallback generation: if no suitable side found in corpus, LLM generates a simple recipe (writes to Mealie)
- Fail fast: invalid JSON, invalid picks, tool errors, or no viable plan within bounded retries

Mode:
- LIVE MODE (default): writes meal plan entries to Mealie after planning.
- DRY RUN (--dry-run): prints planned recipes (IDs + names) and does NOT write meal plans to Mealie (diagnostic only).
- Note: May create NEW recipes in Mealie (for generated fallback sides) regardless of mode.

Usage:
    python chef_agentic.py                                    # plan + write to Mealie
    python chef_agentic.py --dry-run                         # diagnostic (no Mealie writes)
    python chef_agentic.py --start-date YYYY-MM-DD           # custom start date + write
    python chef_agentic.py --candidate-k 100 --max-refines 2  # tuning params + write
"""

import json
import re
import sys
import random
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Set, Tuple

from config import MEAL_TYPES, HISTORY_WEEKS, CHAT_MODEL, DATA_DIR
from mealie_client import MealieClient
from recipe_rag import RecipeRAG
from batch_llm_processor import get_llm_cache
from tools.history_cache import HistoryCache
from tools.logging_utils import get_logger

# Import ingredient parsing infrastructure from mealie_parse (no hardcoded rules)
from mealie_parse import (
    parse_ingredients_direct_llm,
    match_food_to_canonical,
    get_canonical_foods,
    ensure_unit_object,
)

logger = get_logger(__name__)


WEEK_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

def calculate_next_monday(start_date_str: Optional[str] = None) -> datetime.date:
    """
    Calculate next Monday or use provided start date (YYYY-MM-DD).
    """
    if start_date_str:
        try:
            return datetime.strptime(start_date_str, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(f"Invalid date format: {start_date_str}. Use YYYY-MM-DD") from e

    today = datetime.now().date()
    days_ahead = (7 - today.weekday()) % 7 or 7  # Monday=0
    return today + timedelta(days=days_ahead)


def _date_range_for_history(weeks: int, planning_start: date = None) -> Tuple[str, str]:
    """
    Get date range for history query.
    
    Args:
        weeks: How many weeks to look back
        planning_start: Start date of the week being planned. If provided,
                       history extends up to this date to include any 
                       already-planned meals between now and planning_start.
    
    Returns:
        (start_date, end_date) as ISO strings
    """
    today = datetime.now().date()
    
    # End date: either planning_start (to see already-planned meals) or today
    if planning_start and planning_start > today:
        end_date = planning_start - timedelta(days=1)  # Day before planning starts
    else:
        end_date = today
    
    start_date = today - timedelta(weeks=weeks)
    return start_date.isoformat(), end_date.isoformat()


def fetch_meal_history_processed(rag: RecipeRAG, cache: HistoryCache, client: MealieClient, planning_start: date = None) -> Dict[str, Any]:
    """
    Fetch meal history from Mealie and process into variety constraints.
    Uses HistoryCache to avoid redundant API calls.
    
    Args:
        rag: RecipeRAG instance
        cache: HistoryCache instance
        client: MealieClient instance
        planning_start: Start date of week being planned. History extends to this
                       date to include already-planned meals (prevents repetition
                       across multiple planning sessions).
    """
    start_date, end_date = _date_range_for_history(HISTORY_WEEKS, planning_start)

    cached = cache.get(start_date, end_date)
    if cached:
        return cached

    logger.info(f"Fetching fresh history data ({start_date} to {end_date})")
    data = client.get_meal_plans(start_date, end_date)
    logger.debug(f"get_meal_plans returned type={type(data).__name__}, len={len(data) if isinstance(data, list) else 'N/A'}")
    
    # MealieClient returns list directly or wrapped in 'items'
    if isinstance(data, list):
        items = data
    else:
        items = data.get("items", []) if data else []
    logger.debug(f"Extracted {len(items)} meal plan items")

    history_recipes: List[str] = []
    for item in items:
        recipe = item.get("recipe")
        if recipe and recipe.get("name"):
            history_recipes.append(recipe["name"])

    variety_analysis = rag.analyze_menu_history(history_recipes)
    variety_constraints = {
        "recent_recipes": history_recipes[-50:],  # keep small for prompt
        "recent_cuisines": variety_analysis.get("recent_cuisines", []),
        "recent_proteins": variety_analysis.get("recent_proteins", []),
    }

    processed = {
        "date_range": {"start": start_date, "end": end_date},
        "variety_constraints": variety_constraints,
        "meal_history_summary": {
            "total_meals": len(history_recipes),
            "unique_recipes": len(set(history_recipes)),
            "history_weeks": HISTORY_WEEKS,
        },
    }

    cache.put(start_date, end_date, processed)
    print(f"✅ Processed {len(history_recipes)} meals into variety constraints")
    return processed


def _fetch_mealie_mealplan_items(client: MealieClient, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetch Mealie meal plan entries for a date range.
    FAIL FAST on API errors.
    """
    data = client.get_meal_plans(start_date, end_date)
    if isinstance(data, list):
        return data
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    return items


def build_cook_counts(
    client: MealieClient,
    weeks_recent: int = 12,
    years_lifetime: int = 5,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build cook counts from Mealie mealplan history.

    - recent: last N weeks (rolling window)
    - lifetime: last N years (practical "all-time" without needing an unbounded query)
    """
    today = datetime.now().date()
    recent_start = (today - timedelta(weeks=weeks_recent)).isoformat()
    lifetime_start = (today - timedelta(days=365 * years_lifetime)).isoformat()
    end = today.isoformat()

    recent_items = _fetch_mealie_mealplan_items(client, recent_start, end)
    lifetime_items = _fetch_mealie_mealplan_items(client, lifetime_start, end)

    def _count(items: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in items:
            recipe = item.get("recipe")
            rid = item.get("recipeId") or (recipe.get("id") if recipe else None)
            if isinstance(rid, str) and rid:
                counts[rid] = counts.get(rid, 0) + 1
        return counts

    return _count(recent_items), _count(lifetime_items)


@dataclass
class Candidate:
    recipe_id: str
    name: str
    cuisine_primary: Optional[str] = None
    tag_names: List[str] = field(default_factory=list)
    category_names: List[str] = field(default_factory=list)

    def to_llm_brief(self) -> Dict[str, Any]:
        return {
            "id": self.recipe_id,
            "name": self.name,
            "cuisine_primary": self.cuisine_primary or "",
            "tags": self.tag_names[:12],
            "categories": self.category_names[:8],
        }


@dataclass
class PlannedDish:
    candidate: Candidate


@dataclass
class NoteItem:
    """Non-recipe meal plan item (prep/buy)."""
    title: str       # The item name (e.g., "Pappardelle")
    text: str        # Instructions or note (e.g., "boil per package directions")
    item_type: str   # "prep" or "buy"
    ingredients: List[str] = field(default_factory=list)  # Specific items to purchase
    
    def text_with_ingredients(self) -> str:
        """Encode text with ingredients for storage in Mealie (which only has title/text)."""
        if not self.ingredients:
            return self.text
        # Format: "note text ||INGREDIENTS:item1,item2,item3"
        # Using || as delimiter since it's unlikely in natural text
        ingredients_str = ",".join(self.ingredients)
        return f"{self.text} ||INGREDIENTS:{ingredients_str}"
    
    @staticmethod
    def parse_text_with_ingredients(text: str) -> tuple:
        """Parse text field to extract note and ingredients."""
        if "||INGREDIENTS:" not in text:
            return text, []
        parts = text.split("||INGREDIENTS:", 1)
        note_text = parts[0].strip()
        ingredients = [i.strip() for i in parts[1].split(",") if i.strip()]
        return note_text, ingredients


@dataclass
class PlannedMeal:
    dishes: List[PlannedDish] = field(default_factory=list)
    notes: List[NoteItem] = field(default_factory=list)

    def summary(self) -> str:
        parts = [d.candidate.name for d in self.dishes]
        parts += [f"[{n.item_type.upper()}] {n.title}" for n in self.notes]
        return "; ".join(parts)


@dataclass
class AgentState:
    week_start: datetime.date
    # processed variety constraints dict (not wrapped)
    history: Dict[str, Any]
    planned: Dict[str, Dict[str, PlannedMeal]] = field(default_factory=dict)
    used_recipe_ids: Set[str] = field(default_factory=set)
    used_recipe_names: Set[str] = field(default_factory=set)
    # Track accompaniment usage for variety (normalized name -> count)
    used_accompaniments: Dict[str, int] = field(default_factory=dict)
    # Track newly created recipes during this planning session (for image fetching)
    created_recipes: List[Dict[str, str]] = field(default_factory=list)

    def add_meal(self, day: str, meal_type: str, meal: PlannedMeal) -> None:
        if day not in self.planned:
            self.planned[day] = {}
        self.planned[day][meal_type] = meal
        for dish in meal.dishes:
            self.used_recipe_ids.add(dish.candidate.recipe_id)
            self.used_recipe_names.add(dish.candidate.name.lower().strip())

    def add_dish_to_meal(self, day: str, meal_type: str, candidate: Candidate) -> None:
        if day not in self.planned or meal_type not in self.planned[day]:
            raise RuntimeError(f"Internal error: meal container missing for {day} {meal_type}")
        self.planned[day][meal_type].dishes.append(PlannedDish(candidate=candidate))
        self.used_recipe_ids.add(candidate.recipe_id)
        self.used_recipe_names.add(candidate.name.lower().strip())

    def add_note_to_meal(self, day: str, meal_type: str, note: NoteItem) -> None:
        """Add a non-recipe note (prep/buy item) to a meal."""
        if day not in self.planned or meal_type not in self.planned[day]:
            raise RuntimeError(f"Internal error: meal container missing for {day} {meal_type}")
        self.planned[day][meal_type].notes.append(note)

    def track_accompaniment(self, item_name: str) -> None:
        """Track an accompaniment for variety enforcement."""
        normalized = item_name.lower().strip()
        self.used_accompaniments[normalized] = self.used_accompaniments.get(normalized, 0) + 1

    def get_accompaniment_summary(self, min_count: int = 2) -> str:
        """Get summary of frequently used accompaniments for variety prompting."""
        frequent = [(name, count) for name, count in self.used_accompaniments.items() if count >= min_count]
        if not frequent:
            return ""
        frequent.sort(key=lambda x: -x[1])
        return ", ".join(f"{name} ({count}x)" for name, count in frequent)

    def track_created_recipe(self, slug: str, name: str) -> None:
        """Track a newly created recipe for image fetching later."""
        self.created_recipes.append({"slug": slug, "name": name})


def _format_cuisine_options(cuisines: List[str], counts: Optional[Dict[str, int]]) -> str:
    """Format cuisine options with counts for LLM prompt."""
    if not counts:
        return str(cuisines[:80])
    
    # Show cuisines with counts, sorted by count descending
    formatted = []
    for cuisine in cuisines[:60]:
        count = counts.get(cuisine, 0)
        formatted.append(f"{cuisine} ({count})")
    return ", ".join(formatted)


from prompts import (
    ACCOMPANIMENT_SYSTEM_PROMPT,
    build_accompaniment_prompt,
    build_agentic_recipe_selection_prompt,
    build_cuisine_selection_prompt,
    build_accompaniment_pick_prompt,
    SIMPLE_RECIPE_GENERATION_SYSTEM_PROMPT,
    build_simple_recipe_generation_prompt,
)

# Re-export for backward compatibility with other parts of this file
AGENT_SYSTEM_PROMPT = ACCOMPANIMENT_SYSTEM_PROMPT

def _weighted_sample_without_replacement(
    rng: random.Random,
    items: List[Candidate],
    weights: List[float],
    k: int,
) -> List[Candidate]:
    """
    Weighted sampling without replacement using Efraimidis–Spirakis.
    """
    if k <= 0:
        return []
    if len(items) != len(weights):
        raise ValueError("items and weights must be same length")
    if not items:
        return []

    scored: List[Tuple[float, Candidate]] = []
    for item, w in zip(items, weights):
        w = float(w)
        if w <= 0:
            continue
        u = rng.random() or 1e-12
        key = -math.log(u) / w
        scored.append((key, item))

    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[: min(k, len(scored))]]


def _schema_pick_from_sample() -> dict:
    """
    The LLM must pick a recipe_id from the provided sample list.
    No refine/search action here: sampling is handled outside the LLM.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_pick_from_sample",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "recipe_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["recipe_id", "reason", "confidence"],
                "additionalProperties": False,
            },
        },
    }


async def agent_pick_from_sample_for_role(
    state: AgentState,
    day: str,
    meal_type: str,
    role: str,
    slot_cuisine: str,
    candidates: List[Candidate],
    household_context: str = "",
) -> Candidate:
    """
    Ask the LLM to pick a candidate from a provided sample.
    The role parameter is kept for logging purposes only.
    FAIL FAST on invalid JSON or picking an id not in the sample list.
    
    Args:
        household_context: Household description and dietary restrictions
    """
    # Show what candidates the LLM will see
    print(f"    📋 Showing LLM {len(candidates)} candidates for {role}:")
    for i, c in enumerate(candidates[:8]):  # Show first 8
        tags_preview = ", ".join(c.tag_names[:4]) if c.tag_names else "(no tags)"
        print(f"       {i+1}. {c.name} [{tags_preview}]")
    if len(candidates) > 8:
        print(f"       ... and {len(candidates) - 8} more")
    
    sample_briefs = [c.to_llm_brief() for c in candidates]
    primary_ctx = _get_primary_context(state, day, meal_type)
    already = []
    meal = state.planned.get(day, {}).get(meal_type)
    if meal:
        for d in meal.dishes:
            already.append({"id": d.candidate.recipe_id, "name": d.candidate.name})

    user_prompt = build_agentic_recipe_selection_prompt(
        day=day,
        meal_type=meal_type,
        slot_cuisine=slot_cuisine,
        already_json=json.dumps(already, ensure_ascii=False),
        primary_ctx_json=json.dumps(primary_ctx, ensure_ascii=False),
        sample_briefs_json=json.dumps(sample_briefs, ensure_ascii=False),
        household_context=household_context,
    )

    cache_llm = await get_llm_cache()
    resp = await cache_llm.call_llm(
        prompt=user_prompt,
        system_prompt=AGENT_SYSTEM_PROMPT,
        model=CHAT_MODEL,
        temperature=0.4,
        max_tokens=350,
        response_format=_schema_pick_from_sample(),
        reasoning={"effort": "none"}  # Disable reasoning for structured output
    )
    pick = json.loads(resp.strip())
    recipe_id = (pick.get("recipe_id") or "").strip()
    if not recipe_id:
        raise RuntimeError("Agent returned empty recipe_id for pick")
    pick_map = {c.recipe_id: c for c in candidates}
    if recipe_id not in pick_map:
        raise RuntimeError(f"Agent picked recipe_id not in sample: {recipe_id!r}")
    chosen = pick_map[recipe_id]
    reason = (pick.get('reason') or '').strip()
    confidence = pick.get('confidence', 'N/A')
    
    print(f"    🧠 LLM picked: {chosen.name}")
    print(f"    💭 Reasoning: {reason}")
    print(f"    📊 Confidence: {confidence}")
    
    return chosen


def _get_primary_context(state: AgentState, day: str, meal_type: str) -> Dict[str, Any]:
    """Get context about the primary dish (first dish) for a meal slot."""
    meal = state.planned.get(day, {}).get(meal_type)
    if not meal or not meal.dishes:
        return {}
    # The first dish is considered the primary
    c = meal.dishes[0].candidate
    return {
        "primary_name": c.name,
        "primary_cuisine_primary": c.cuisine_primary or "",
        "primary_tags": c.tag_names,
        "primary_categories": c.category_names,
    }


async def agent_choose_cuisine_for_slot(
    state: AgentState,
    day: str,
    meal_type: str,
    available_cuisines: List[str],
    cuisine_counts: Optional[Dict[str, int]] = None,
    household_context: str = "",
    temp_prompt: str = "",
) -> str:
    """
    Ask the LLM to choose a cuisine/country label for this slot from available cuisines.
    FAIL FAST: must choose one of the provided values.
    
    Args:
        household_context: Household description and dietary restrictions
        temp_prompt: One-shot instructions for this planning session
    """
    slot_date = state.week_start + timedelta(days=WEEK_DAYS.index(day))
    history = state.history if isinstance(state.history, dict) else {}

    planned = []
    cuisine_counts: Dict[str, int] = {}
    for d in WEEK_DAYS:
        for mt in MEAL_TYPES:
            meal = state.planned.get(d, {}).get(mt)
            if not meal:
                continue
            planned.append(f"{d} {mt}: {meal.summary()}")
            # count cuisines already used (based on selected dishes' cuisine_primary where present)
            for dish in meal.dishes:
                c = (dish.candidate.cuisine_primary or "").strip()
                if c:
                    cuisine_counts[c] = cuisine_counts.get(c, 0) + 1

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_choose_cuisine",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "cuisine": {"type": "string", "enum": available_cuisines[:200]},
                    "reason": {"type": "string"},
                },
                "required": ["cuisine", "reason"],
                "additionalProperties": False,
            },
        },
    }

    # Track proteins used this week
    protein_counts: Dict[str, int] = {}
    for d in WEEK_DAYS:
        for mt in MEAL_TYPES:
            meal = state.planned.get(d, {}).get(mt)
            if not meal:
                continue
            for dish in meal.dishes:
                # Extract protein from tags or name
                tags = dish.candidate.tags if hasattr(dish.candidate, 'tags') else []
                for tag in tags:
                    tag_lower = tag.lower()
                    if any(p in tag_lower for p in ['pork', 'beef', 'chicken', 'lamb', 'fish', 'salmon', 'turkey', 'duck']):
                        protein_counts[tag] = protein_counts.get(tag, 0) + 1

    # Determine meal complexity preference based on day of week
    day_index = WEEK_DAYS.index(day)
    is_weekend = day_index >= 5  # Saturday, Sunday
    complexity_guidance = "Weekday meal: prefer simpler, quicker dishes" if not is_weekend else "Weekend meal: can be more elaborate"

    # AI tuning knobs (optional; defaults applied if missing)
    from config import USER_CONFIG
    variety_cfg = USER_CONFIG.get("meal_planning", {}).get("variety", {})
    max_protein_repeats = int(variety_cfg.get("max_protein_repetitions_per_week", 2) or 2)
    max_cuisine_streak = int(variety_cfg.get("max_consecutive_same_cuisine", 3) or 3)
    max_protein_repeats = max(1, min(14, max_protein_repeats))
    max_cuisine_streak = max(1, min(14, max_cuisine_streak))

    user_prompt = build_cuisine_selection_prompt(
        day=day,
        meal_type=meal_type,
        slot_date_iso=slot_date.isoformat(),
        complexity_guidance=complexity_guidance,
        recent_cuisines=history.get('recent_cuisines', []),
        recent_proteins=history.get('recent_proteins', []),
        planned_summary=chr(10).join(planned) if planned else "",
        cuisine_counts_json=json.dumps(cuisine_counts, ensure_ascii=False),
        protein_counts_json=json.dumps(protein_counts, ensure_ascii=False),
        available_cuisines_formatted=_format_cuisine_options(available_cuisines, cuisine_counts),
        household_context=household_context,
        temp_prompt=temp_prompt,
        max_protein_repetitions_per_week=max_protein_repeats,
        max_consecutive_same_cuisine=max_cuisine_streak,
    )

    cache = await get_llm_cache()
    resp = await cache.call_llm(
        prompt=user_prompt,
        system_prompt=AGENT_SYSTEM_PROMPT,
        model=CHAT_MODEL,
        temperature=0.4,
        max_tokens=250,
        response_format=schema,
        reasoning={"effort": "none"}  # Disable reasoning for structured output
    )
    data = json.loads(resp.strip())
    cuisine = (data.get("cuisine") or "").strip()
    reason = (data.get("reason") or "").strip()
    
    print(f"    🧠 LLM chose: {cuisine}")
    print(f"    💭 Reasoning: {reason}")
    
    if cuisine not in available_cuisines:
        raise RuntimeError(f"Agent chose cuisine not in available list: {cuisine!r}")
    return cuisine


def _schema_accompaniments() -> dict:
    """Schema for LLM to return classified meal accompaniments."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "meal_accompaniments",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "accompaniments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item": {
                                    "type": "string",
                                    "description": "The accompaniment name (e.g., 'steamed jasmine rice', 'simple green salad')"
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["recipe", "prep", "buy"],
                                    "description": "Classification: recipe (needs cooking instructions), prep (trivial preparation), buy (purchase ready-made)"
                                },
                                "note": {
                                    "type": "string",
                                    "description": "Prep instructions or buying note (e.g., 'boil per package directions', 'from bakery')"
                                },
                                "ingredients": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific ingredients to purchase for PREP/BUY items (e.g., ['mixed greens', 'cherry tomatoes', 'balsamic vinaigrette'] for a salad, ['crusty baguette'] for bread). Required for prep/buy, empty for recipe."
                                }
                            },
                            "required": ["item", "type", "note", "ingredients"],
                            "additionalProperties": False
                        },
                        "description": "List of classified accompaniments"
                    },
                    "condiments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item": {
                                    "type": "string",
                                    "description": "Optional table condiment/pantry item (does not count as a dish-level accompaniment)"
                                },
                                "note": {
                                    "type": "string",
                                    "description": "Brief optional note (e.g., 'serve at table', 'optional for heat')"
                                },
                                "ingredients": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Must be empty for condiments (do not add pantry staples to shopping list)."
                                },
                            },
                            "required": ["item", "note", "ingredients"],
                            "additionalProperties": False,
                        },
                        "description": "Optional condiments/pantry items that do NOT consume accompaniment slots",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why these accompaniments complete the meal"
                    },
                },
                # NOTE: Some providers (e.g., Azure via OpenRouter) require that every
                # key in properties also appears in required when strict mode is enabled.
                # "condiments" can be an empty list when there are none.
                "required": ["accompaniments", "condiments", "reasoning"],
                "additionalProperties": False,
            },
        },
    }


async def determine_meal_accompaniments(
    primary: Candidate,
    cuisine: str,
    meal_type: str,
    day: str,
    state: AgentState,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Ask LLM for classic/traditional accompaniments that complement this dish.
    
    Returns:
        - dish_accompaniments: list of {"item": str, "type": "recipe|prep|buy", "note": str, "ingredients": [...]}
        - condiments: list of {"item": str, "note": str, "ingredients": []}
    
    The prompt focuses on CULINARY TRADITION first (what classic sides go with this dish?),
    then classification for how to fulfill them (recipe from DB, simple prep, or buy).
    """
    from prompts import get_household_context
    household_context = get_household_context()
    
    # Build variety constraint if we've used accompaniments frequently
    variety_constraint = ""
    frequent_accompaniments = state.get_accompaniment_summary(min_count=3)
    if frequent_accompaniments:
        variety_constraint = f"\nVARIETY: Used frequently this week: {frequent_accompaniments}. Choose different options."

    prompt = build_accompaniment_prompt(
        primary_name=primary.name,
        cuisine=cuisine,
        day=day,
        meal_type=meal_type,
        household_context=household_context,
        variety_constraint=variety_constraint,
    )

    cache = await get_llm_cache()
    resp = await cache.call_llm(
        prompt=prompt,
        system_prompt=ACCOMPANIMENT_SYSTEM_PROMPT,
        model=CHAT_MODEL,
        temperature=0.4,
        max_tokens=500,
        response_format=_schema_accompaniments(),
        reasoning={"effort": "none"}
    )
    
    data = json.loads(resp.strip())
    accompaniments = data.get("accompaniments", [])
    condiments = data.get("condiments", []) or []
    reasoning = data.get("reasoning", "").strip()
    
    # Log classified accompaniments
    print(f"    🍽️ Accompaniments chosen:")
    for acc in accompaniments:
        item = acc.get("item", "?")
        acc_type = acc.get("type", "?")
        note = acc.get("note", "")
        note_str = f" ({note})" if note else ""
        print(f"       [{acc_type.upper()}] {item}{note_str}")
    if condiments:
        print(f"    🧂 Optional condiments:")
        for c in condiments:
            item = (c.get("item") or "").strip()
            note = (c.get("note") or "").strip()
            note_str = f" ({note})" if note else ""
            if item:
                print(f"       {item}{note_str}")
    print(f"    💭 Reasoning: {reasoning}")
    
    return accompaniments, condiments


def _schema_pick_accompaniment() -> dict:
    """Schema for LLM to pick an accompaniment or reject all options."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pick_accompaniment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["pick", "none_suitable"],
                        "description": "'pick' if one works, 'none_suitable' if none are appropriate"
                    },
                    "recipe_id": {
                        "type": "string",
                        "description": "ID of chosen recipe (required if decision=pick)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of choice"
                    },
                },
                "required": ["decision", "recipe_id", "reason"],
                "additionalProperties": False,
            },
        },
    }


async def _llm_pick_accompaniment(
    description: str,
    candidates: List[Candidate],
    primary: Candidate,
    cuisine: str,
    state: AgentState,
    household_context: str = "",
) -> Optional[Candidate]:
    """
    Let LLM pick an accompaniment from search results with full meal context.
    
    The LLM considers:
    - What the main dish is
    - What accompaniments are already selected for this meal
    - Cuisine harmony
    - Balance and variety
    - Dietary restrictions from household context
    
    Args:
        household_context: Household description and dietary restrictions
    
    Returns:
        Chosen Candidate, or None if LLM rejects all options
    """
    if not candidates:
        return None
    
    # Build context about what's already in this meal
    # (We need to get the current meal's dishes from state)
    # Note: state.planned[day][meal_type] contains dishes already added
    
    # Format candidates for LLM
    candidate_briefs = [c.to_llm_brief() for c in candidates]
    
    prompt = build_accompaniment_pick_prompt(
        description=description,
        primary_name=primary.name,
        primary_cuisine=primary.cuisine_primary,
        primary_tags=primary.tag_names,
        cuisine=cuisine,
        candidate_briefs_json=json.dumps(candidate_briefs, indent=2, ensure_ascii=False),
        household_context=household_context,
    )

    cache = await get_llm_cache()
    resp = await cache.call_llm(
        prompt=prompt,
        system_prompt=AGENT_SYSTEM_PROMPT,
        model=CHAT_MODEL,
        temperature=0.3,
        max_tokens=300,
        response_format=_schema_pick_accompaniment(),
        reasoning={"effort": "none"}
    )
    
    data = json.loads(resp.strip())
    decision = data.get("decision", "")
    recipe_id = data.get("recipe_id", "").strip()
    reason = data.get("reason", "").strip()
    
    print(f"    🧠 LLM decision: {decision}")
    print(f"    💭 Reasoning: {reason}")
    
    if decision == "none_suitable":
        return None
    
    if decision == "pick" and recipe_id:
        # Find the candidate with this ID
        for c in candidates:
            if c.recipe_id == recipe_id:
                return c
        
        # LLM picked an ID not in the list - fail
        logger.warning(f"LLM picked recipe_id {recipe_id!r} not in candidates")
        return None
    
    return None


async def find_or_generate_accompaniment(
    client: MealieClient,
    description: str,
    cuisine: str,
    primary: Candidate,
    state: AgentState,
    rag: RecipeRAG,
    generation_cache: Dict[str, Candidate],
    household_context: str = "",
    servings: int = 4,
    dietary_restrictions: list = None,
) -> Optional[Candidate]:
    """
    Find an accompaniment in the recipe corpus, or generate a simple recipe if not found.
    
    Flow:
    1. Hybrid search (semantic + keyword) to find candidates
    2. LLM reviews candidates with full context (main dish, cuisine, meal harmony)
    3. LLM picks the best match OR says "none suitable"
    4. If none suitable, generate a simple recipe
    
    This ensures the LLM makes intelligent picks based on culinary reasoning,
    not just blind score-based selection.
    
    Args:
        description: What we're looking for (e.g., "garlic bread", "miso soup")
        cuisine: The cuisine context
        primary: The primary dish this accompanies
        state: Current planning state (to avoid duplicates)
        rag: RecipeRAG instance for hybrid search
        generation_cache: Cache by description to avoid duplicate LLM calls
        household_context: Household description and dietary restrictions
        servings: Number of servings for generated recipes
        dietary_restrictions: List of dietary restrictions to respect
    
    Returns:
        Candidate if found/generated, None if failed
    """
    # Cache key is by DESCRIPTION, not recipe_name
    # This ensures "french fries" lookup is cached regardless of what name LLM generates
    cache_key = f"{cuisine}:{description.lower().strip()}"
    
    # 1. Check if we've already found/generated this exact description this session
    if cache_key in generation_cache:
        cached = generation_cache[cache_key]
        # Don't reuse if already used in this week's plan
        if cached.recipe_id not in state.used_recipe_ids:
            print(f"    ♻️ Reusing from session cache: {cached.name}")
            return cached
        else:
            print(f"    ⚠️ Cached recipe already used this week, searching again...")
    
    print(f"    🔍 Searching for: {description!r} (hybrid: semantic + keyword)")
    
    # 2. Use HYBRID search (semantic embeddings + keyword FTS)
    # This handles synonyms like fries/chips while also boosting exact name matches
    try:
        results = rag.find_recipes_for_concept(description, top_k=15)
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        results = []
    
    # Filter out already-used recipes
    candidates = []
    for r in results:
        recipe_id = r.get("id")
        name = r.get("name", "")
        
        if not recipe_id or recipe_id in state.used_recipe_ids:
            continue
        if name.lower().strip() in state.used_recipe_names:
            continue
        
        # Extract tag names - handle both string and dict formats
        raw_tags = r.get("tags", [])
        tag_names = _extract_names_from_mealie_list(raw_tags) if isinstance(raw_tags, list) else []
        
        # Extract category names
        raw_categories = r.get("categories", [])
        category_names = _extract_names_from_mealie_list(raw_categories) if isinstance(raw_categories, list) else []
        
        candidates.append(Candidate(
            recipe_id=recipe_id,
            name=name,
            cuisine_primary=r.get("cuisine_primary"),
            tag_names=tag_names,
            category_names=category_names,
        ))
    
    # Debug: show candidates that will be shown to LLM
    if candidates:
        print(f"    📋 Showing LLM {len(candidates)} candidates for '{description}':")
        for i, c in enumerate(candidates[:8]):
            tags_preview = ", ".join(c.tag_names[:3]) if c.tag_names else "(no tags)"
            print(f"       {i+1}. {c.name} [{tags_preview}]")
        if len(candidates) > 8:
            print(f"       ... and {len(candidates) - 8} more")
    else:
        print(f"    ⚠️ No candidates found for {description!r}")
    
    # 3. Let LLM pick from candidates (same as primary dish selection)
    # LLM considers: main dish context, meal harmony, what's already picked
    if candidates:
        chosen = await _llm_pick_accompaniment(
            description=description,
            candidates=candidates[:10],  # Top 10 candidates
            primary=primary,
            cuisine=cuisine,
            state=state,
            household_context=household_context,
        )
        
        if chosen:
            print(f"    ✅ LLM picked: {chosen.name}")
            generation_cache[cache_key] = chosen
            return chosen
        else:
            print(f"    ⚠️ LLM rejected all candidates for '{description}'")
    
    # 3. No good match found - generate a simple recipe
    print(f"    💡 Not found in Mealie by keyword search. Generating: {description}")
    
    try:
        primary_ctx = {
            "primary_name": primary.name,
            "primary_cuisine_primary": primary.cuisine_primary or "",
            "primary_tags": primary.tag_names,
        }
        
        recipe_data = await generate_simple_accompaniment_recipe(
            description=description,
            cuisine=cuisine,
            primary_ctx=primary_ctx,
            servings=servings,
            dietary_restrictions=dietary_restrictions,
        )
        
        recipe_id, recipe_slug, is_new = await create_recipe_in_mealie(client, recipe_data, cuisine)
        
        # Track newly created recipes for image fetching later
        if is_new:
            state.track_created_recipe(recipe_slug, recipe_data["name"])
        
        # Post-process: apply tags, update local DB, index in RAG
        # This makes the generated recipe fully discoverable for future searches
        await post_process_generated_recipe(
            client=client,
            recipe_id=recipe_id,
            recipe_slug=recipe_slug,
            cuisine=cuisine,
            rag=rag,
        )
        
        generated = Candidate(
            recipe_id=recipe_id,
            name=recipe_data["name"],
            cuisine_primary=cuisine,
            tag_names=["AI-Generated", f"{cuisine} Cuisine"],
            category_names=[],
        )
        
        # Cache by description for future lookups
        generation_cache[cache_key] = generated
        
        print(f"    ✨ Generated: {generated.name} [{recipe_id}]")
        return generated
        
    except Exception as e:
        print(f"    ❌ Failed to generate {description}: {e}")
        return None


def validate_generated_recipe(recipe_data: dict) -> Tuple[bool, List[str]]:
    """
    Validate generated recipe has required content.
    
    Validation rules:
    - Must have at least 1 ingredient
    - Must have at least 1 instruction
    - Each instruction must be at least 10 characters (a real sentence)
    - Recipe name must not be empty
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: List[str] = []
    
    # Check recipe name
    name = recipe_data.get("name", "").strip()
    if not name:
        errors.append("Recipe name is empty")
    
    # Check ingredients
    ingredients = recipe_data.get("recipe_ingredient", [])
    if not isinstance(ingredients, list):
        errors.append(f"recipe_ingredient is not a list: {type(ingredients).__name__}")
    elif len(ingredients) < 1:
        errors.append("No ingredients (need at least 1)")
    
    # Check instructions
    instructions = recipe_data.get("recipe_instructions", [])
    if not isinstance(instructions, list):
        errors.append(f"recipe_instructions is not a list: {type(instructions).__name__}")
    elif len(instructions) < 1:
        errors.append("No instructions (need at least 1)")
    else:
        # Validate each instruction has meaningful content
        for i, step in enumerate(instructions):
            if isinstance(step, dict):
                text = step.get("text", "").strip()
            elif isinstance(step, str):
                text = step.strip()
            else:
                errors.append(f"Instruction {i+1} is invalid type: {type(step).__name__}")
                continue
            
            if len(text) < 10:
                errors.append(f"Instruction {i+1} too short ({len(text)} chars): '{text[:50]}'")
    
    return (len(errors) == 0, errors)


def _schema_simple_recipe() -> dict:
    """Schema for LLM to return a simple recipe."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "simple_recipe",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Recipe name"},
                    "description": {"type": "string", "description": "Brief description"},
                    "recipe_ingredient": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ingredients with quantities"
                    },
                    "recipe_instructions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "text": {"type": "string"}
                            },
                            "required": ["title", "text"],
                            "additionalProperties": False
                        },
                        "description": "Step-by-step instructions"
                    },
                    "recipe_yield": {"type": "string", "description": "Servings"},
                    "prep_time": {"type": "string", "description": "ISO 8601 duration (e.g., PT10M)"},
                    "cook_time": {"type": "string", "description": "ISO 8601 duration (e.g., PT15M)"},
                    "total_time": {"type": "string", "description": "ISO 8601 duration (e.g., PT25M)"},
                },
                "required": ["name", "description", "recipe_ingredient", "recipe_instructions", "recipe_yield", "prep_time", "cook_time", "total_time"],
                "additionalProperties": False,
            },
        },
    }


async def generate_simple_accompaniment_recipe(
    description: str,
    cuisine: str,
    primary_ctx: Dict[str, Any],
    max_attempts: int = 3,
    servings: int = 4,
    dietary_restrictions: list = None,
) -> Dict[str, Any]:
    """
    Generate a simple accompaniment recipe using LLM with validation and retry.
    
    IMPORTANT: Recipe name is FORCED to be the exact description (title-cased).
    This prevents duplicate recipes with different fancy names like
    "Crispy French Fries" vs "Crunchy French Fries".
    
    Args:
        description: What to generate (e.g., "garlic bread", "steamed rice")
        cuisine: Cuisine context
        primary_ctx: Info about the primary dish
        max_attempts: Maximum retry attempts for validation failures (default 3)
        servings: Number of servings (from household config)
        dietary_restrictions: List of dietary restrictions to respect
    
    Returns:
        Recipe dict in Mealie-compatible format
    
    Raises:
        RuntimeError: If recipe generation fails after max_attempts or on non-validation errors
    """
    # FORCE the recipe name to be exactly the description - no LLM creativity
    # This prevents "Crispy French Fries", "Crunchy French Fries", etc.
    canonical_name = description.strip().title()
    
    # Build prompt using centralized prompts.py
    prompt = build_simple_recipe_generation_prompt(
        description=description,
        primary_name=primary_ctx.get('primary_name', ''),
        cuisine=cuisine,
        canonical_name=canonical_name,
        servings=servings,
        dietary_restrictions=dietary_restrictions,
    )

    cache = await get_llm_cache()
    accumulated_errors: List[str] = []
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Temperature 0.4: balanced for consistency while allowing reasonable variation
            # (0.7 was too high for structured output, causing inconsistent/incomplete results)
            resp = await cache.call_llm(
                prompt=prompt,
                system_prompt=SIMPLE_RECIPE_GENERATION_SYSTEM_PROMPT,
                model=CHAT_MODEL,
                temperature=0.4,
                max_tokens=1000,
                response_format=_schema_simple_recipe(),
                reasoning={"effort": "none"}
            )
            
            # Parse JSON response
            try:
                recipe_data = json.loads(resp.strip())
            except json.JSONDecodeError as e:
                raise RuntimeError(f"LLM returned invalid JSON: {e}. Response: {resp[:200]}") from e
            
            # Validate we got a dict back (not a string or other type)
            if not isinstance(recipe_data, dict):
                raise RuntimeError(f"LLM returned {type(recipe_data).__name__} instead of dict. Response: {resp[:200]}")
            
            # Validate required fields exist
            required_fields = ["name", "recipe_ingredient", "recipe_instructions"]
            for field_name in required_fields:
                if field_name not in recipe_data:
                    raise RuntimeError(f"LLM response missing required field '{field_name}'. Response: {resp[:200]}")
            
            # FORCE the canonical name - override whatever LLM returned
            recipe_data["name"] = canonical_name
            
            # Validate recipe content quality
            is_valid, errors = validate_generated_recipe(recipe_data)
            
            if is_valid:
                print(f"    📝 LLM generated recipe: {recipe_data.get('name', '?')}")
                print(f"    📝 Ingredients: {len(recipe_data.get('recipe_ingredient', []))} items")
                print(f"    📝 Instructions: {len(recipe_data.get('recipe_instructions', []))} steps")
                return recipe_data
            else:
                # Validation failed - log and retry
                error_msg = f"Attempt {attempt}/{max_attempts} failed validation: {errors}"
                logger.warning(error_msg)
                print(f"    ⚠️ {error_msg}")
                accumulated_errors.extend(errors)
                
        except RuntimeError:
            # Non-validation errors (JSON parse, missing fields) - don't retry, re-raise
            raise
        except Exception as e:
            # Unexpected errors - don't retry, re-raise
            raise RuntimeError(f"Unexpected error generating recipe: {e}") from e
    
    # All attempts failed validation
    raise RuntimeError(
        f"Failed to generate valid recipe for '{description}' after {max_attempts} attempts. "
        f"Accumulated errors: {accumulated_errors}"
    )


def tool_search_recipes(rag: RecipeRAG, query: str, top_k: int, exclude_ids: Set[str]) -> List[Candidate]:
    """
    Tool: search recipes using RecipeRAG.
    Fail-fast on tool errors.
    """
    matches = rag.find_recipes_for_concept(query, top_k=top_k)
    candidates: List[Candidate] = []

    for r in matches:
        recipe_id = r.get("id")
        if not recipe_id or recipe_id in exclude_ids:
            continue
        name = r.get("name", "Unknown Recipe")
        cuisine_primary = r.get("cuisine_primary")

        # Prefer tags/categories (more stable than partial ingredient previews)
        raw_tags = r.get("tags") or []
        raw_categories = r.get("categories") or []

        def _extract_names(raw: Any) -> List[str]:
            names: List[str] = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and item.get("name"):
                        names.append(str(item["name"]))
                    elif isinstance(item, str) and item.strip():
                        names.append(item.strip())
            return names

        tag_names = _extract_names(raw_tags)
        category_names = _extract_names(raw_categories)

        candidates.append(
            Candidate(
                recipe_id=recipe_id,
                name=name,
                cuisine_primary=cuisine_primary,
                tag_names=tag_names,
                category_names=category_names,
            )
        )

    return candidates


def _extract_names_from_mealie_list(raw: Any) -> List[str]:
    names: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
            elif isinstance(item, str) and item.strip():
                names.append(item.strip())
    return names


def _extract_categories_from_mealie(recipe: Dict[str, Any]) -> List[str]:
    """
    Mealie recipes can represent categories in different shapes depending on API/version.
    We normalize to a list of names.
    """
    raw = recipe.get("recipeCategory")
    if raw is None:
        return []
    if isinstance(raw, list):
        return _extract_names_from_mealie_list(raw)
    if isinstance(raw, dict) and raw.get("name"):
        return [str(raw["name"])]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _fetch_mealie_recipe_metadata(client: MealieClient, recipe_id: str) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Fetch tags/categories from Mealie (source of truth) for a given recipe id.
    FAIL FAST: if Mealie is unreachable or returns an error, raise.
    """
    data = client.get_recipe_by_id(recipe_id)

    tags = _extract_names_from_mealie_list(data.get("tags") or [])
    categories = _extract_categories_from_mealie(data)
    cuisine_primary = None
    # If you have a specific cuisine tag naming convention, we can improve this later.
    # For now, keep cuisine_primary from the local index if present.
    return tags, categories, cuisine_primary


def tool_enrich_candidates_from_mealie(
    client: MealieClient,
    candidates: List[Candidate],
    cache: Dict[str, Tuple[List[str], List[str]]],
    limit: int = 10,
) -> None:
    """
    Enrich top candidates with Mealie tags/categories (truth source) before LLM decision.
    Only fetches for candidates that currently have no tags/categories.
    
    NOTE: Gracefully handles 404s (recipe deleted from Mealie but still in local index).
    Stale candidates are marked for removal rather than crashing.
    """
    to_check = []
    for c in candidates[:limit]:
        if c.tag_names or c.category_names:
            continue
        to_check.append(c)

    stale_ids = set()  # Track recipes that no longer exist in Mealie
    
    for c in to_check:
        if c.recipe_id in cache:
            tags, cats = cache[c.recipe_id]
            c.tag_names = tags
            c.category_names = cats
            continue

        try:
            tags, cats, _ = _fetch_mealie_recipe_metadata(client, c.recipe_id)
            cache[c.recipe_id] = (tags, cats)
            c.tag_names = tags
            c.category_names = cats
        except Exception as e:
            # Check if it's a "not found" error
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                # Recipe was deleted from Mealie but still in local index (stale data)
                logger.warning(f"Recipe {c.recipe_id} ({c.name}) not found in Mealie (404) - local index is stale")
                stale_ids.add(c.recipe_id)
            else:
                # Other errors - log but continue (don't crash the whole planning)
                logger.error(f"Failed to fetch metadata for {c.recipe_id}: {e}")
                # Leave tags/categories empty, LLM can still work with name
    
    # Remove stale candidates from the list
    if stale_ids:
        logger.warning(f"Removing {len(stale_ids)} stale recipes from candidates (deleted from Mealie)")
        candidates[:] = [c for c in candidates if c.recipe_id not in stale_ids]


def _debug_print_candidates(query: str, candidates: List[Candidate], limit: int = 5) -> None:
    """
    Debug helper: show what the retrieval tool actually returned.
    Keeps output small and high-signal.
    """
    print(f"    🔎 Tool search query: {query!r}")
    print(f"    📦 Candidates returned: {len(candidates)}")
    for i, c in enumerate(candidates[:limit], 1):
        tags = ", ".join(c.tag_names[:6]) if c.tag_names else "(no tags)"
        cats = ", ".join(c.category_names[:4]) if c.category_names else "(no categories)"
        cuisine = c.cuisine_primary or "(no cuisine_primary)"
        print(f"      {i}. {c.name}  [{c.recipe_id}]")
        print(f"         cuisine_primary: {cuisine}")
        print(f"         tags: {tags}")
        print(f"         categories: {cats}")


def _fuzzy_name_match(name1: str, name2: str, threshold: float = 0.75) -> bool:
    """
    Simple fuzzy name matching using token overlap (Jaccard similarity).
    Returns True if names are similar enough to be considered duplicates.
    """
    def tokenize(s: str) -> Set[str]:
        # Lowercase, remove punctuation, split into words
        s = s.lower()
        s = re.sub(r'[^\w\s]', ' ', s)
        return set(s.split())
    
    tokens1 = tokenize(name1)
    tokens2 = tokenize(name2)
    
    if not tokens1 or not tokens2:
        return False
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    jaccard = len(intersection) / len(union)
    return jaccard >= threshold


def search_mealie_recipes_by_keyword(client: MealieClient, query: str, per_page: int = 50) -> List[Dict[str, Any]]:
    """
    Search Mealie recipes using keyword/name search (NOT semantic).
    
    Uses Mealie's built-in search API which does text matching against recipe names.
    This is the correct approach for finding specific items like "garlic bread" or "miso soup".
    
    NOTE: Mealie search can be VERY slow (30-40 seconds on 16K recipes).
    This is why we use local embeddings/RAG for primary search.
    This function is only used for deduplication checks.
    
    Args:
        client: MealieClient instance
        query: Search term (e.g., "garlic bread", "miso soup")
        per_page: Number of results to return (default 50)
    
    Returns:
        List of recipe dicts with id, name, tags, etc.
    """
    try:
        items = client.search_recipes(query, per_page)
        
        # Debug: show what Mealie returned
        if items:
            names = [item.get("name", "?") for item in items[:5]]
            logger.debug(f"Mealie search returned {len(items)} results. Top 5: {names}")
        else:
            logger.debug(f"Mealie search returned 0 results for query={query!r}")
        
        return items
    except Exception as e:
        error_str = str(e).lower()
        if "timeout" in error_str or "timed out" in error_str:
            logger.warning(f"Mealie search timed out for query={query!r} - skipping dedup check")
            print(f"    ⚠️ Mealie search timed out - skipping duplicate check for '{query}'")
        else:
            logger.error(f"Mealie keyword search failed: {e}")
            print(f"    ⚠️ Mealie keyword search failed: {e}")
        return []


async def sync_index_with_mealie(client: MealieClient, rag: RecipeRAG) -> int:
    """
    Sync local index with Mealie - index any recipes missing from local DB.
    
    This prevents duplicate recipe creation by ensuring the dedup check
    has up-to-date data from Mealie.
    
    Returns number of recipes added to index.
    """
    print("🔄 Syncing local index with Mealie...")
    
    try:
        # Get all recipe IDs from local index
        with sqlite3.connect(str(DATA_DIR / "recipe_index.db")) as conn:
            cur = conn.execute("SELECT id FROM recipes")
            local_ids = {row[0] for row in cur.fetchall()}
        
        # Get all recipes from Mealie
        mealie_recipes = client.get_all_recipes()
        
        # Find recipes in Mealie but not in local index
        missing = []
        for r in mealie_recipes:
            if r.get('id') not in local_ids:
                missing.append(r.get('slug'))
        
        if not missing:
            print(f"   ✓ Index is up to date ({len(local_ids)} recipes)")
            return 0
        
        print(f"   Found {len(missing)} recipes to index...")
        
        # Index missing recipes
        indexed = 0
        for slug in missing[:50]:  # Limit to 50 to avoid long startup times
            try:
                recipe_data = client.get_recipe(slug)
                if rag.index_recipe(recipe_data, force=False):
                    indexed += 1
            except Exception as e:
                logger.debug(f"Failed to index {slug}: {e}")
                continue
        
        print(f"   ✅ Indexed {indexed} new recipes")
        return indexed
        
    except Exception as e:
        logger.warning(f"Index sync failed: {e}")
        print(f"   ⚠️ Index sync failed: {e}")
        return 0


async def find_existing_recipe_by_name(client: MealieClient, recipe_name: str, cuisine: str) -> Optional[Tuple[str, str]]:
    """
    Search for existing recipes with similar names (dedupe check).
    
    Uses LOCAL SQLite index first (fast!) then falls back to Mealie API only if needed.
    The local index is rebuilt from Mealie, so it should have all recipes.
    
    Returns (recipe_id, slug) tuple if a match is found, None otherwise.
    """
    recipe_name_lower = recipe_name.lower().strip()
    
    # FAST PATH: Check local SQLite index first (instant vs 30-40s Mealie search)
    try:
        with sqlite3.connect(str(DATA_DIR / "recipe_index.db")) as conn:
            # Exact match
            cur = conn.execute(
                "SELECT id, name, slug FROM recipes WHERE LOWER(name) = ?",
                (recipe_name_lower,)
            )
            row = cur.fetchone()
            if row:
                recipe_id, recipe_name_found, recipe_slug = row
                print(f"    🔍 Found exact match in local index: {recipe_name_found} [{recipe_id[:8]}...]")
                
                # Verify recipe actually exists in Mealie (index can be stale)
                try:
                    client.get_recipe(recipe_slug)
                    return (recipe_id, recipe_slug)
                except Exception:
                    print(f"    ⚠️ Recipe not found in Mealie (stale index entry), will create new")
                    # Remove stale entry from local index
                    conn.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
                    conn.commit()
            
            # Fuzzy match using LIKE for similar names
            cur = conn.execute(
                "SELECT id, name, slug FROM recipes WHERE LOWER(name) LIKE ? LIMIT 10",
                (f"%{recipe_name_lower}%",)
            )
            for row in cur.fetchall():
                if _fuzzy_name_match(recipe_name, row[1], threshold=0.75):
                    recipe_id, recipe_name_found, recipe_slug = row
                    print(f"    🔍 Found similar match in local index: {recipe_name_found} [{recipe_id[:8]}...] (fuzzy)")
                    
                    # Verify recipe actually exists in Mealie
                    try:
                        client.get_recipe(recipe_slug)
                        return (recipe_id, recipe_slug)
                    except Exception:
                        print(f"    ⚠️ Recipe not found in Mealie (stale index entry), will create new")
                        conn.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
                        conn.commit()
            
            logger.debug(f"No match in local index for '{recipe_name}'")
            return None
    except Exception as e:
        logger.warning(f"Local index search failed: {e}")
        # Don't fall back to slow Mealie API - local index is the source of truth for search
        # If local index is broken, that's a separate issue to fix
        return None


async def create_recipe_in_mealie(
    client: MealieClient,
    recipe_data: Dict[str, Any],
    cuisine: str,
) -> Tuple[str, str, bool]:
    """
    Create a new recipe in Mealie via API (with deduplication).
    
    Mealie API requires a two-step process:
    1. POST /api/recipes with {"name": "..."} to create a stub recipe (returns slug)
    2. PATCH /api/recipes/{slug} with full Recipe-Input data to populate content
    
    Deduplication strategy:
    1. Search Mealie for existing recipes with similar names
    2. Only create if no match found
    
    Note: Per-session caching by description happens at find_or_generate_accompaniment level.
    
    Returns (recipe_id, slug, is_new) tuple where is_new=True if recipe was created.
    """
    # Validate input is a dict (defensive against bad LLM responses)
    if not isinstance(recipe_data, dict):
        raise TypeError(f"create_recipe_in_mealie expected dict, got {type(recipe_data).__name__}: {str(recipe_data)[:100]}")
    
    if "name" not in recipe_data:
        raise ValueError(f"recipe_data missing 'name' field. Keys: {list(recipe_data.keys())}")
    
    recipe_name = recipe_data["name"]
    
    # Search Mealie for existing similar recipes (dedup against Mealie itself)
    existing = await find_existing_recipe_by_name(client, recipe_name, cuisine)
    if existing:
        existing_id, existing_slug = existing
        print(f"    🔍 Found existing recipe in Mealie: {recipe_name} [{existing_id}]")
        return existing_id, existing_slug, False  # is_new=False for existing recipe
    
    # STEP 1: Create stub recipe (Mealie's CreateRecipe schema only accepts "name")
    create_payload = {"name": recipe_name}
    
    print(f"    📤 Creating recipe in Mealie: POST /api/recipes")
    print(f"    📤 Payload: {create_payload}")
    
    try:
        result = client.create_recipe(create_payload)
        
        # Handle both response formats: Mealie may return string slug or dict with slug field
        if isinstance(result, str):
            slug = result
        elif isinstance(result, dict):
            slug = result.get("slug") or result.get("id")
            if not slug:
                raise RuntimeError(f"Mealie did not return slug after creation: {result}")
        else:
            raise RuntimeError(f"Unexpected response type from Mealie: {type(result).__name__}: {result}")
    except Exception as e:
        print(f"    ❌ Request failed: {e}")
        raise
    
    # STEP 2: Update the recipe with full content via PATCH
    # Convert LLM-generated ingredients (strings) to Mealie's RecipeIngredient format
    # IMPORTANT: Each ingredient MUST have a referenceId UUID, and SHOULD reference
    # existing foods/units from Mealie for proper shopping list aggregation.
    #
    # Uses LLM-based parsing from mealie_parse.py (no hardcoded rules) to properly
    # extract food names, quantities, and units from ingredient strings.
    raw_ingredients = recipe_data.get("recipe_ingredient", [])
    mealie_ingredients = []
    
    # Separate string ingredients (need parsing) from dict ingredients (already structured)
    string_ingredients = [ing for ing in raw_ingredients if isinstance(ing, str)]
    dict_ingredients = [ing for ing in raw_ingredients if isinstance(ing, dict)]
    
    # Parse string ingredients using LLM (batch call for efficiency)
    if string_ingredients:
        print(f"    🧠 Parsing {len(string_ingredients)} ingredients via LLM...")
        parsed = parse_ingredients_direct_llm(string_ingredients)
        
        # Load canonical foods for matching
        canonical_foods = get_canonical_foods()
        
        if parsed:
            for i, ing_data in enumerate(parsed):
                original_text = string_ingredients[i] if i < len(string_ingredients) else ""
                
                # Match food to canonical
                food_name = ing_data.get("food", "")
                matched_food = match_food_to_canonical(food_name, canonical_foods)
                
                # Look up/create unit
                unit_name = ing_data.get("unit")
                unit_obj = ensure_unit_object(unit_name) if unit_name else None
                
                ingredient_obj = {
                    "referenceId": str(uuid.uuid4()),  # REQUIRED: Mealie fails without this
                    "note": ing_data.get("note", ""),
                    "display": original_text,
                    "originalText": original_text,
                    "quantity": ing_data.get("quantity", 0) or 0,
                }
                
                # Add food reference if matched
                if matched_food:
                    ingredient_obj["food"] = {"id": matched_food["id"], "name": matched_food.get("name", "")}
                elif food_name:
                    # Food not in canonical list - include name without ID
                    # Shopping list will use 'note' field as fallback
                    ingredient_obj["food"] = {"name": food_name}
                
                # Add unit reference if found
                if unit_obj:
                    ingredient_obj["unit"] = {"id": unit_obj["id"], "name": unit_obj.get("name", "")}
                
                mealie_ingredients.append(ingredient_obj)
        else:
            # LLM parsing failed - use ingredients as-is with note field
            logger.warning("LLM ingredient parsing failed, using raw text in note field")
            for ing in string_ingredients:
                mealie_ingredients.append({
                    "referenceId": str(uuid.uuid4()),
                    "note": ing,
                    "display": ing,
                    "originalText": ing,
                })
    
    # Add dict ingredients (already structured)
    for ing in dict_ingredients:
        if "referenceId" not in ing:
            ing["referenceId"] = str(uuid.uuid4())
        mealie_ingredients.append(ing)
    
    # Convert LLM-generated instructions to Mealie's RecipeStep format
    # IMPORTANT: Instructions require specific fields: id, text, title, ingredientReferences
    raw_instructions = recipe_data.get("recipe_instructions", [])
    mealie_instructions = []
    for i, step in enumerate(raw_instructions):
        if isinstance(step, str):
            # Simple string instruction - use required format
            mealie_instructions.append({
                "id": None,  # REQUIRED: null for new instructions
                "text": step,
                "title": "",
                "ingredientReferences": []  # REQUIRED: empty list if no references
            })
        elif isinstance(step, dict):
            # Dict format - ensure all required fields exist
            mealie_instructions.append({
                "id": step.get("id"),  # Can be null for new instructions
                "text": step.get("text") or step.get("title") or "",
                "title": step.get("title", ""),
                "ingredientReferences": step.get("ingredientReferences", [])
            })
    
    # Build the update payload using Recipe-Input schema
    # Use None instead of empty strings for optional fields (Mealie may reject empty strings)
    prep_time = recipe_data.get("prep_time") or None
    cook_time = recipe_data.get("cook_time") or None
    total_time = recipe_data.get("total_time") or None
    recipe_yield = recipe_data.get("recipe_yield") or "4 servings"
    
    update_payload = {
        "name": recipe_name,
        "slug": slug,
        "description": recipe_data.get("description", ""),
        "recipeIngredient": mealie_ingredients,
        "recipeInstructions": mealie_instructions,
        "recipeYield": recipe_yield,
        "prepTime": prep_time,
        "performTime": cook_time,
        "totalTime": total_time,
        # Note: Tags removed - Mealie may require tags to exist before referencing
        # TODO: Create tags first or use existing tag IDs
    }
    
    # Debug: show time values
    print(f"    📤 Times: prep={prep_time}, cook={cook_time}, total={total_time}")
    
    print(f"    📤 Updating recipe: PATCH /api/recipes/{slug}")
    print(f"    📤 Payload keys: {list(update_payload.keys())}")
    print(f"    📤 Ingredients ({len(mealie_ingredients)}): {mealie_ingredients[:2]}...")
    print(f"    📤 Instructions ({len(mealie_instructions)}): {mealie_instructions[:1]}...")
    
    try:
        client.update_recipe(slug, update_payload)
    except Exception as e:
        print(f"    ❌ PATCH failed: {e}")
        raise
    
    # Fetch the recipe to get the ID
    recipe_obj = client.get_recipe(slug)
    recipe_id = recipe_obj.get("id")
    if not recipe_id:
        raise RuntimeError(f"Mealie did not return recipe ID for slug {slug}: {recipe_obj}")
    
    print(f"    ✨ Generated and created new recipe in Mealie: {recipe_name} [{recipe_id}]")
    return recipe_id, slug, True  # is_new=True for newly created recipe


async def post_process_generated_recipe(
    client: MealieClient,
    recipe_id: str,
    recipe_slug: str,
    cuisine: str,
    rag: RecipeRAG,
) -> None:
    """
    Make a generated recipe fully discoverable (first-class citizen).
    
    Called immediately after create_recipe_in_mealie() for generated recipes.
    
    Steps:
    1. Fetch full recipe from Mealie (need complete data for indexing)
    2. Apply tags: "{cuisine} Cuisine" and "AI-Generated"
    3. Update local SQLite database with cuisine_primary
    4. Index in RecipeRAG (creates embedding, adds to FTS, updates ANN index)
    
    Quality standards:
    - Atomic: If any step fails, log error but don't fail whole meal planning
    - Idempotent: Running twice on same recipe is safe
    - Observable: Log what was done
    
    Args:
        client: MealieClient instance
        recipe_id: Mealie recipe UUID
        recipe_slug: Mealie recipe slug (for API calls)
        cuisine: Cuisine for tagging (e.g., "Italian", "Japanese")
        rag: RecipeRAG instance for indexing
    """
    print(f"    🔄 Post-processing generated recipe: {recipe_slug}")
    
    try:
        # 1. Fetch full recipe from Mealie
        recipe_data = client.get_recipe(recipe_slug)
        
        # 2. Apply tags: "{cuisine} Cuisine" and "AI-Generated"
        tags_to_apply = [f"{cuisine} Cuisine", "AI-Generated"]
        existing_tags = recipe_data.get("tags", [])
        existing_tag_names = {t.get("name") for t in existing_tags if isinstance(t, dict)}
        
        # Get or create tag objects
        new_tag_objects = []
        for tag_name in tags_to_apply:
            if tag_name in existing_tag_names:
                continue  # Already has this tag
            
            # Look up existing tag by name
            tag_obj = _get_or_create_tag(client, tag_name)
            if tag_obj:
                new_tag_objects.append(tag_obj)
        
        if new_tag_objects:
            # Update recipe with new tags
            all_tags = existing_tags + new_tag_objects
            try:
                client.update_recipe(recipe_slug, {"tags": all_tags})
                applied_names = [t.get("name") for t in new_tag_objects]
                print(f"    🏷️  Applied tags: {applied_names}")
            except Exception as e:
                logger.warning(f"Failed to apply tags to {recipe_slug}: {e}")
        
        # 3. Update local SQLite database with cuisine_primary
        try:
            with sqlite3.connect(rag.db_path) as conn:
                conn.execute(
                    "UPDATE recipes SET cuisine_primary = ? WHERE id = ?",
                    (cuisine, recipe_id)
                )
                if conn.total_changes > 0:
                    print(f"    📊 Updated local DB: cuisine_primary = {cuisine}")
        except Exception as e:
            logger.warning(f"Failed to update local DB for {recipe_id}: {e}")
        
        # 4. Index in RecipeRAG (force=True to re-index even if exists)
        # Re-fetch to get updated data with tags
        updated_recipe_data = client.get_recipe(recipe_slug)
        
        if rag.index_recipe(updated_recipe_data, force=True):
            print(f"    🔍 Indexed in RAG (embedding + FTS + ANN)")
        else:
            logger.warning(f"Failed to index {recipe_slug} in RAG")
        
        print(f"    ✅ Post-processing complete for {recipe_slug}")
        
    except Exception as e:
        # Log but don't fail - recipe is still created, just not fully discoverable
        logger.error(f"Post-processing failed for {recipe_slug}: {e}")
        print(f"    ⚠️ Post-processing failed (recipe still created): {e}")


def _get_or_create_tag(client: MealieClient, tag_name: str) -> Optional[Dict[str, Any]]:
    """
    Get existing tag by name, or return None if not found.
    
    Note: We don't auto-create tags to avoid tag drift. Tags should be
    pre-seeded using tools/seed_mealie_tags.py.
    
    Args:
        client: MealieClient instance
        tag_name: Tag name to look up (e.g., "Italian Cuisine", "AI-Generated")
    
    Returns:
        Tag object dict with id/name/slug, or None if not found
    """
    try:
        tags = client.get_all_tags()
        for tag in tags:
            if tag.get("name") == tag_name:
                return {"id": tag["id"], "name": tag["name"], "slug": tag.get("slug")}
        
        # Tag doesn't exist - log and return None
        logger.info(f"Tag '{tag_name}' not found in Mealie - skipping")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to look up tag '{tag_name}': {e}")
        return None


def _print_plan(state: AgentState, dry_run: bool = True) -> None:
    print("\n" + "=" * 72)
    if dry_run:
        print("🧪 AGENTIC CHEF DRY RUN PLAN (NO MEALIE WRITES)")
    else:
        print("📝 AGENTIC CHEF MEAL PLAN")
    print("=" * 72)
    for i, day in enumerate(WEEK_DAYS):
        date_str = (state.week_start + timedelta(days=i)).strftime("%a %b %d")
        print(f"\n{date_str} ({day})")
        for meal_type in MEAL_TYPES:
            pick = state.planned.get(day, {}).get(meal_type)
            if pick and (pick.dishes or pick.notes):
                print(f"  {meal_type}:")
                for dish in pick.dishes:
                    print(f"    • {dish.candidate.name}")
                for note in pick.notes:
                    print(f"    • [{note.item_type.upper()}] {note.title}")
            else:
                print(f"  {meal_type}: (unfilled)")
    print("\n" + "=" * 72 + "\n")


async def write_plan_to_mealie(client: MealieClient, state: AgentState) -> None:
    """
    Write the planned week's meals to Mealie as mealplan entries.
    Idempotent: checks if entry already exists for date+entryType before creating.
    """
    print("📤 Writing meal plan to Mealie...")
    
    # Fetch existing mealplan entries for this week to avoid duplicates
    start_date = state.week_start.isoformat()
    end_date = (state.week_start + timedelta(days=6)).isoformat()
    
    data = client.get_meal_plans(start_date, end_date)
    if isinstance(data, list):
        existing_entries = data
    else:
        existing_entries = data.get("items", [])
    
    # Build map of (date, meal_type) -> list of entry IDs to delete before writing
    # We delete ALL entries for a slot before writing new plan (idempotent)
    existing_by_slot = {}
    for entry in existing_entries:
        entry_date = entry.get("date")
        entry_type = entry.get("entryType")
        entry_id = entry.get("id")
        
        # Only track lunch and dinner entries (all dishes use these types now)
        if entry_type in ["lunch", "dinner"]:
            key = (entry_date, entry_type)
            
            if key not in existing_by_slot:
                existing_by_slot[key] = []
            if entry_id:
                existing_by_slot[key].append(entry_id)
    
    # Create mealplan entries for each planned slot
    created_count = 0
    
    for i, day in enumerate(WEEK_DAYS):
        slot_date = (state.week_start + timedelta(days=i)).isoformat()
        
        for meal_type in MEAL_TYPES:
            meal = state.planned.get(day, {}).get(meal_type)
            if not meal or not meal.dishes:
                continue
            
            # Delete ALL existing entries for this slot (all dishes)
            # This ensures idempotency: re-running overwrites with new plan
            entries_to_delete = existing_by_slot.get((slot_date, meal_type), [])
            
            for entry_id in entries_to_delete:
                try:
                    client.delete_meal_plan_entry(entry_id)
                except Exception as e:
                    # Fail fast: if we can't delete, we'll create duplicates
                    raise RuntimeError(f"Failed to delete existing mealplan entry {entry_id} for {day} {meal_type}: {e}") from e
            
            if entries_to_delete:
                print(f"  🗑️  Deleted {len(entries_to_delete)} old entries for {day} {meal_type}")
            
            # Write ALL dishes for this meal (primary + accompaniments)
            # All dishes use the same entryType (lunch or dinner) to keep them grouped
            dishes_written = []
            notes_written = []
            
            for dish in meal.dishes:
                # Build mealplan entry payload for recipe
                payload = {
                    "date": slot_date,
                    "entryType": meal_type,
                    "recipeId": dish.candidate.recipe_id,
                    "title": "",  # Mealie will use recipe name
                    "text": "",  # No labels - helpers just see the recipes
                }
                
                try:
                    client.create_meal_plan_entry(payload)
                    dishes_written.append(dish.candidate.name)
                    created_count += 1
                except Exception as e:
                    # Fail fast: partial writes are bad
                    raise RuntimeError(f"Failed to create {day} {meal_type} dish '{dish.candidate.name}': {e}") from e
            
            # Write note entries (PREP/BUY items - no recipeId)
            # Ingredients are encoded in text field for shopping list extraction
            # UI should filter out ||INGREDIENTS: when displaying
            for note in meal.notes:
                payload = {
                    "date": slot_date,
                    "entryType": meal_type,
                    "title": note.title,
                    "text": note.text_with_ingredients(),  # Includes ingredients for shopping list
                    # No recipeId - this is a note entry
                }
                
                try:
                    client.create_meal_plan_entry(payload)
                    notes_written.append(f"[{note.item_type.upper()}] {note.title}")
                    created_count += 1
                except Exception as e:
                    raise RuntimeError(f"Failed to create {day} {meal_type} note '{note.title}': {e}") from e
            
            if dishes_written or notes_written:
                print(f"  ✅ Created {day} {meal_type}:")
                for dish_name in dishes_written:
                    print(f"      • {dish_name}")
                for note_name in notes_written:
                    print(f"      • {note_name}")
    
    print(f"\n📊 Summary: {created_count} dishes written to Mealie")
    if created_count > 0:
        from config import MEALIE_URL
        print(f"✅ Meal plan written to Mealie: {MEALIE_URL}/household/mealplan")


async def plan_week_agentic(
    start_date_str: Optional[str],
    candidate_k: int,
    max_refines: int,
    preferred_cuisines: Optional[List[str]] = None,
    dietary_restrictions: Optional[List[str]] = None,
    temp_prompt: str = "",
) -> AgentState:
    week_start = calculate_next_monday(start_date_str)
    end_date = week_start + timedelta(days=6)
    print(f"📆 Planning week: {week_start.strftime('%B %d')} - {end_date.strftime('%B %d, %Y')}\n")
    
    # Build household context from config (includes personal section)
    from prompts import get_household_context
    household_context = get_household_context()
    
    # Get servings for recipe generation
    from config import USER_CONFIG
    servings = USER_CONFIG.get("household", {}).get("servings", 4)
    
    # Show preferences if specified
    if preferred_cuisines:
        print(f"🌍 Preferred cuisines: {', '.join(preferred_cuisines)}")
    if dietary_restrictions:
        print(f"🚫 Dietary restrictions: {', '.join(dietary_restrictions)}")
    if temp_prompt:
        print(f"📝 Special instructions: {temp_prompt}")
    if preferred_cuisines or dietary_restrictions or temp_prompt:
        print()

    rag = RecipeRAG()
    # Force API mode - we create/update recipes then immediately read them
    client = MealieClient(use_direct_db=False)
    
    try:
        # Sync local index with Mealie before planning (prevents duplicate recipe creation)
        await sync_index_with_mealie(client, rag)
        
        db_stats = rag.analyze_recipe_database()
        
        # Get cuisine distribution WITH COUNTS so LLM knows how many recipes are in each
        cuisine_distribution = db_stats.get("cuisine_distribution", {}) if isinstance(db_stats, dict) else {}
        if not cuisine_distribution:
            raise RuntimeError("No cuisines found in local index (recipe_index.db). Cannot plan.")
        
        # Filter to cuisines with at least 10 recipes (avoid tiny buckets)
        MIN_RECIPES_PER_CUISINE = 10
        viable_cuisines = {k: v for k, v in cuisine_distribution.items() if v >= MIN_RECIPES_PER_CUISINE}
        
        if not viable_cuisines:
            # Fallback: use all cuisines if none have 10+
            viable_cuisines = cuisine_distribution
            logger.warning(f"No cuisines have {MIN_RECIPES_PER_CUISINE}+ recipes, using all cuisines")
        
        # Prioritize preferred cuisines by putting them first in the list
        available_cuisines = list(viable_cuisines.keys())
        if preferred_cuisines:
            # Normalize and match preferred cuisines (case-insensitive)
            preferred_lower = {c.lower().strip() for c in preferred_cuisines}
            matched_preferred = [c for c in available_cuisines if c.lower() in preferred_lower]
            other_cuisines = [c for c in available_cuisines if c.lower() not in preferred_lower]
            available_cuisines = matched_preferred + other_cuisines
            if matched_preferred:
                print(f"✅ Prioritizing cuisines: {', '.join(matched_preferred)}")
        
        print(f"📊 Cuisine distribution (>={MIN_RECIPES_PER_CUISINE} recipes):")
        for cuisine, count in sorted(viable_cuisines.items(), key=lambda x: -x[1])[:10]:
            print(f"   {cuisine}: {count} recipes")

        # Cook-count weights (Mealie truth)
        cook_recent, cook_lifetime = build_cook_counts(client, weeks_recent=12, years_lifetime=5)
        rng = random.Random()
        rng.seed(int(datetime.now().timestamp()))
        cache = HistoryCache(ttl_days=7)
        history = fetch_meal_history_processed(rag, cache, client, planning_start=week_start)

        state = AgentState(week_start=week_start, history=history.get("variety_constraints", {}))
        mealie_meta_cache: Dict[str, Tuple[List[str], List[str]]] = {}
        generation_cache: Dict[str, Candidate] = {}  # Per-session cache by description (not recipe_name)

        # Main planning loop - simplified architecture
        for day in WEEK_DAYS:
            for meal_type in MEAL_TYPES:
                print(f"\n{'='*60}")
                print(f"🍽️ Planning: {day} {meal_type}")
                print(f"{'='*60}")

                # 1. Choose cuisine for this slot (pass counts so LLM knows bucket sizes)
                slot_cuisine = await agent_choose_cuisine_for_slot(
                    state, day, meal_type, available_cuisines, viable_cuisines,
                    household_context=household_context,
                    temp_prompt=temp_prompt,
                )
                print(f"  🌍 Cuisine: {slot_cuisine} ({viable_cuisines.get(slot_cuisine, '?')} recipes)")

                # 2. Create empty meal container
                meal = PlannedMeal()
                if day not in state.planned:
                    state.planned[day] = {}
                state.planned[day][meal_type] = meal

                # 3. Pick primary dish
                print(f"  📋 Selecting primary dish...")
                primary = await _pick_primary_dish(
                    client=client,
                    state=state,
                    day=day,
                    meal_type=meal_type,
                    slot_cuisine=slot_cuisine,
                    rag=rag,
                    rng=rng,
                    cook_recent=cook_recent,
                    cook_lifetime=cook_lifetime,
                    candidate_k=candidate_k,
                    mealie_meta_cache=mealie_meta_cache,
                    household_context=household_context,
                )
                state.add_dish_to_meal(day, meal_type, primary)
                print(f"  ✅ Primary: {primary.name}")

                # 4. Determine what accompaniments complete this meal (with classification)
                accompaniments, condiments = await determine_meal_accompaniments(
                    primary=primary,
                    cuisine=slot_cuisine,
                    meal_type=meal_type,
                    day=day,
                    state=state,
                )

                # 5. Process each accompaniment by type
                for acc in accompaniments:
                    acc_type = acc.get("type", "").lower().strip()
                    acc_item = acc.get("item", "").strip()
                    acc_note = acc.get("note", "").strip()
                    acc_ingredients = acc.get("ingredients", [])
                    
                    # Validate type - fail fast on invalid
                    if acc_type not in ("recipe", "prep", "buy"):
                        raise RuntimeError(f"Invalid accompaniment type '{acc_type}' for item '{acc_item}'. Must be: recipe, prep, buy")
                    
                    if acc_type == "recipe":
                        # Find or generate recipe (existing flow)
                        acc_dish = await find_or_generate_accompaniment(
                            client=client,
                            description=acc_item,
                            cuisine=slot_cuisine,
                            primary=primary,
                            state=state,
                            rag=rag,
                            generation_cache=generation_cache,
                            household_context=household_context,
                            servings=servings,
                            dietary_restrictions=dietary_restrictions,
                        )
                        if acc_dish:
                            state.add_dish_to_meal(day, meal_type, acc_dish)
                            state.track_accompaniment(acc_item)  # Track for variety
                    else:
                        # PREP or BUY - create note entry with ingredients for shopping list
                        note = NoteItem(
                            title=acc_item,
                            text=acc_note,
                            item_type=acc_type,
                            ingredients=acc_ingredients if isinstance(acc_ingredients, list) else [],
                        )
                        state.add_note_to_meal(day, meal_type, note)
                        state.track_accompaniment(acc_item)  # Track for variety
                        ing_count = len(note.ingredients)
                        print(f"    📝 Added [{acc_type.upper()}]: {acc_item} ({ing_count} ingredients)")

                # 6. Add optional condiments as note entries (visible in Mealie + WhatsApp).
                # These do NOT consume accompaniment slots and should not affect shopping list.
                if isinstance(condiments, list) and condiments:
                    for c in condiments:
                        item = (c.get("item") or "").strip()
                        note_text = (c.get("note") or "").strip()
                        raw_ingredients = c.get("ingredients", [])
                        ingredients: List[str] = raw_ingredients if isinstance(raw_ingredients, list) else []
                        if not item:
                            continue

                        # Ensure condiments show up in WhatsApp (which prints title),
                        # but do NOT get treated as shopping-list items (we skip "Condiment:" titles).
                        title = f"Condiment: {item}"
                        note = NoteItem(
                            title=title,
                            text=note_text,
                            item_type="buy",
                            ingredients=ingredients,
                        )
                        state.add_note_to_meal(day, meal_type, note)
                        print(f"    🧂 Added [CONDIMENT]: {title}")

                # Print meal summary
                print(f"  📝 Complete meal: {meal.summary()}")

        return state
    finally:
        client.close()


async def _pick_primary_dish(
    client: MealieClient,
    state: AgentState,
    day: str,
    meal_type: str,
    slot_cuisine: str,
    rag: RecipeRAG,
    rng: random.Random,
    cook_recent: Dict[str, int],
    cook_lifetime: Dict[str, int],
    candidate_k: int,
    mealie_meta_cache: Dict[str, Tuple[List[str], List[str]]],
    household_context: str = "",
) -> Candidate:
    """
    Pick the primary dish for a meal slot from the cuisine bucket.
    Uses weighted sampling to prefer under-cooked recipes.
    
    Args:
        household_context: Household description and dietary restrictions
    """
    # Build cuisine bucket from local index
    with sqlite3.connect(rag.db_path) as conn:
        cur = conn.execute(
            "SELECT id, name, cuisine_primary FROM recipes WHERE cuisine_primary = ?",
            (slot_cuisine,),
        )
        pool: List[Candidate] = []
        for rid, name, cuisine_primary in cur.fetchall():
            if rid in state.used_recipe_ids:
                continue
            if name.lower().strip() in state.used_recipe_names:
                continue
            pool.append(
                Candidate(recipe_id=rid, name=name or "Unknown Recipe", cuisine_primary=cuisine_primary)
            )

    if not pool:
        raise RuntimeError(f"No recipes available for cuisine={slot_cuisine!r} after excluding already-used recipes.")

    # Weight by cook counts (prefer less-cooked recipes)
    def weight_for(rid: str) -> float:
        return (1.0 / (1 + cook_recent.get(rid, 0))) * (1.0 / (1 + cook_lifetime.get(rid, 0)))

    weights = [weight_for(c.recipe_id) for c in pool]
    candidates = _weighted_sample_without_replacement(rng, pool, weights, candidate_k)
    
    if not candidates:
        raise RuntimeError("Weighted sampling returned no candidates.")

    # Enrich top candidates with Mealie metadata
    tool_enrich_candidates_from_mealie(client, candidates, mealie_meta_cache, limit=min(10, len(candidates)))

    # Ask LLM to pick the best primary dish
    chosen = await agent_pick_from_sample_for_role(
        state=state,
        day=day,
        meal_type=meal_type,
        role="primary",
        slot_cuisine=slot_cuisine,
        candidates=candidates,
        household_context=household_context,
    )
    
    return chosen


def main() -> None:
    start_date_str = None
    candidate_k = 25
    max_refines = 1
    dry_run = False  # Default: write to Mealie

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--start-date" and i + 1 < len(sys.argv):
            start_date_str = sys.argv[i + 1]
            i += 2
        elif arg == "--candidate-k" and i + 1 < len(sys.argv):
            candidate_k = int(sys.argv[i + 1])
            i += 2
        elif arg == "--max-refines" and i + 1 < len(sys.argv):
            max_refines = int(sys.argv[i + 1])
            i += 2
        elif arg == "--dry-run":
            dry_run = True
            i += 1
        else:
            raise ValueError(
                "Usage: python chef_agentic.py [--start-date YYYY-MM-DD] [--candidate-k N] [--max-refines N] [--dry-run]"
            )

    print("\n" + "=" * 60)
    if dry_run:
        print("👨‍🍳 AGENTIC CHEF (DRY RUN) - Tool-Using Weekly Planner")
        print("💡 Diagnostic mode: will NOT write meal plan to Mealie")
    else:
        print("👨‍🍳 AGENTIC CHEF - Tool-Using Weekly Planner")
        print("📝 Will write meal plan to Mealie after planning")
    print("=" * 60 + "\n")

    import asyncio
    state = asyncio.run(plan_week_agentic(start_date_str, candidate_k, max_refines))
    _print_plan(state, dry_run=dry_run)
    
    if not dry_run:
        client = MealieClient()
        try:
            asyncio.run(write_plan_to_mealie(client, state))
        finally:
            client.close()
    else:
        print("💡 This was a DRY RUN (diagnostic mode). Omit --dry-run to write the plan to Mealie.")


if __name__ == "__main__":
    main()

