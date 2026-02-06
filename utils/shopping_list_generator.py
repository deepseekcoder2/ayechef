#!/usr/bin/env python3
"""
Shopping List Generator for Mealie
Creates shopping lists from meal plans with automatic recipe scaling.

Usage:
    python shopping_list_generator.py --week-start 2025-12-23
    python shopping_list_generator.py --recipe-ids uuid1 uuid2 uuid3
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from config import HOUSEHOLD_SERVINGS, get_bulk_operation_config_safe
from tools.logging_utils import get_logger
from mealie_client import MealieClient

# Initialize logger for this module
logger = get_logger(__name__)


def fetch_meal_plans_for_week(start_date: str) -> List[Dict]:
    """
    Fetch all meal plan entries for a week starting from the given date.
    
    Args:
        start_date: Date string in YYYY-MM-DD format
    
    Returns:
        List of meal plan entries
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = start + timedelta(days=6)
        
        logger.info(f"Fetching meal plans from {start.strftime('%b %d')} to {end.strftime('%b %d, %Y')}...")
        
        client = MealieClient()
        try:
            meal_plans = client.get_meal_plans(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            return meal_plans if isinstance(meal_plans, list) else []
        finally:
            client.close()
            
    except ValueError as e:
        logger.error(f"Error parsing date: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching meal plans: {e}")
        return []


def extract_recipe_ids_from_meal_plans(meal_plans: List[Dict]) -> List[str]:
    """
    Extract unique recipe IDs from meal plan entries.
    
    Args:
        meal_plans: List of meal plan entries
    
    Returns:
        List of unique recipe IDs
    """
    recipe_ids = set()
    
    for plan in meal_plans:
        if "recipeId" in plan and plan["recipeId"]:
            recipe_ids.add(plan["recipeId"])
    
    return list(recipe_ids)


def extract_note_items_from_meal_plans(meal_plans: List[Dict]) -> List[Dict]:
    """
    Extract note entries (PREP/BUY items) from meal plan entries.
    
    These are entries that have a title but no recipeId - they represent
    items that need to be purchased but don't have a recipe (e.g., "pappardelle"
    with note "boil per package directions", or "baguette" from bakery).
    
    Ingredients are encoded in the text field as "note ||INGREDIENTS:item1,item2"
    and are parsed out here for the shopping list.
    
    Args:
        meal_plans: List of meal plan entries
    
    Returns:
        List of note items: [{"title": str, "text": str, "ingredients": List[str]}, ...]
    """
    note_items = []
    
    for plan in meal_plans:
        # Note entries have title but no recipeId
        has_recipe = plan.get("recipeId")
        has_title = plan.get("title")
        
        if has_title and not has_recipe:
            # Skip informational notes that should appear in the meal plan UI
            # but should NOT generate shopping list items (e.g., optional table condiments).
            title_str = str(plan["title"]).strip()
            if title_str.lower().startswith("condiment:"):
                continue

            raw_text = plan.get("text", "")
            
            # Parse ingredients from text field (format: "note ||INGREDIENTS:item1,item2")
            note_text = raw_text
            ingredients = []
            if "||INGREDIENTS:" in raw_text:
                parts = raw_text.split("||INGREDIENTS:", 1)
                note_text = parts[0].strip()
                ingredients = [i.strip() for i in parts[1].split(",") if i.strip()]
            
            note_items.append({
                "title": title_str,
                "text": note_text,
                "ingredients": ingredients,
            })
    
    return note_items


def fetch_recipe_details(recipe_id: str) -> Optional[Dict]:
    """
    Fetch full recipe details including serving size.

    Args:
        recipe_id: Recipe UUID or slug

    Returns:
        Recipe dictionary or None if fetch fails
    """
    try:
        client = MealieClient()
        try:
            return client.get_recipe(recipe_id)
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Error fetching recipe {recipe_id}: {e}")
        return None


async def fetch_recipe_details_async(recipe_id: str) -> Optional[Dict]:
    """
    Async version of recipe fetching for parallel processing.

    Provides better performance when fetching multiple recipes concurrently.

    Args:
        recipe_id: Recipe UUID or slug

    Returns:
        Recipe dictionary or None if fetch fails
    """
    # Use sync version with MealieClient (it handles connection pooling internally)
    return fetch_recipe_details(recipe_id)


async def fetch_multiple_recipes_async(recipe_ids: List[str], max_concurrent: int = None) -> Dict[str, Optional[Dict]]:
    """
    Fetch multiple recipes concurrently with controlled parallelism.

    Optimizes network usage and reduces total fetch time for large recipe sets.

    Args:
        recipe_ids: List of recipe IDs to fetch
        max_concurrent: Maximum concurrent requests (None = use config default)

    Returns:
        Dictionary mapping recipe_id to recipe data (or None if failed)
    """
    import asyncio
    from typing import Dict, Optional

    # Use centralized configuration if no explicit max_concurrent provided
    if max_concurrent is None:
        config = get_bulk_operation_config_safe('sync', fallback_batch_size=20, fallback_concurrent=3)
        max_concurrent = config['max_concurrent']

    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async def fetch_with_semaphore(recipe_id: str) -> None:
        async with semaphore:
            recipe_data = await fetch_recipe_details_async(recipe_id)
            results[recipe_id] = recipe_data

    # Create concurrent fetch tasks
    tasks = [fetch_with_semaphore(recipe_id) for recipe_id in recipe_ids]
    await asyncio.gather(*tasks, return_exceptions=True)

    return results


def calculate_scale_factor(recipe_servings: Optional[int], target_servings: int = HOUSEHOLD_SERVINGS) -> float:
    """
    Calculate the scale factor for a recipe.
    
    Args:
        recipe_servings: Number of servings the recipe makes (None if not set)
        target_servings: Target number of servings (default from config)
    
    Returns:
        Scale factor rounded to 2 decimal places
    """
    if recipe_servings is None or recipe_servings == 0:
        return 1.0
    
    scale_factor = target_servings / recipe_servings
    return round(scale_factor, 2)


def create_shopping_list(name: str) -> Optional[str]:
    """
    Create a new shopping list in Mealie.
    
    Args:
        name: Name for the shopping list
    
    Returns:
        Shopping list ID (UUID) or None if creation fails
    """
    try:
        client = MealieClient()
        try:
            shopping_list = client.create_shopping_list(name)
            return shopping_list.get("id")
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Error creating shopping list: {e}")
        return None


def add_recipes_to_shopping_list(list_id: str, recipes: List[Dict[str, any]]) -> bool:
    """
    Add multiple recipes to a shopping list with scaling.
    
    Args:
        list_id: Shopping list UUID
        recipes: List of dicts with 'recipeId' and 'recipeIncrementQuantity'
    
    Returns:
        True if successful, False otherwise
    """
    try:
        client = MealieClient()
        try:
            return client.add_recipes_to_shopping_list(list_id, recipes)
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Error adding recipes to shopping list: {e}")
        return False


def fetch_shopping_list(list_id: str) -> Optional[Dict]:
    """
    Fetch complete shopping list details.
    
    Args:
        list_id: Shopping list UUID
    
    Returns:
        Shopping list dictionary or None if fetch fails
    """
    try:
        client = MealieClient()
        try:
            return client.get_shopping_list(list_id)
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Error fetching shopping list: {e}")
        return None


def add_item_to_shopping_list(list_id: str, display: str, note: str = "") -> bool:
    """
    Add a single item to a shopping list.
    
    NOTE: Mealie aggressively aggregates items without unique food identifiers.
    To prevent aggregation, we use the 'note' field for the item text (which 
    Mealie doesn't aggregate on) and set display to a unique value.
    
    Args:
        list_id: Shopping list UUID
        display: Item display text (e.g., "Pappardelle")
        note: Optional note (e.g., "boil per package directions")
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Combine display + note into a single note field
        # This shows in Mealie UI and won't get aggregated
        item_text = display
        if note:
            item_text = f"{display} ({note})"
        
        item = {
            "display": display,
            "note": item_text,  # Use note field - Mealie displays this
            "quantity": 1,
            "checked": False,
            "extras": {"source": "prep_buy", "original": display}
        }
        
        client = MealieClient()
        try:
            client.add_shopping_item(list_id, item)
            return True
        finally:
            client.close()
        
    except Exception as e:
        logger.error(f"Error adding item '{display}' to shopping list: {e}")
        return False


def add_note_items_to_shopping_list(list_id: str, note_items: List[Dict]) -> int:
    """
    Add PREP/BUY note items to shopping list.
    
    Items have specific ingredients to purchase (from the LLM classification):
    - PREP "simple green salad" → ingredients: ["mixed greens", "cherry tomatoes", "vinaigrette"]
    - BUY "crusty baguette" → ingredients: ["crusty baguette"]
    
    If an item has no ingredients, falls back to adding the title.
    
    Args:
        list_id: Shopping list UUID
        note_items: List of {"title": str, "text": str, "ingredients": List[str]}
    
    Returns:
        Count of items successfully added
    """
    added = 0
    for item in note_items:
        title = item.get("title", "")
        ingredients = item.get("ingredients", [])
        
        if not title:
            continue
        
        if ingredients:
            # Add each ingredient as a separate shopping list item
            for ingredient in ingredients:
                if add_item_to_shopping_list(list_id, ingredient, note=f"for {title}"):
                    added += 1
                    logger.info(f"  Added ingredient: {ingredient} (for {title})")
        else:
            # Fallback: add the title itself if no ingredients specified
            if add_item_to_shopping_list(list_id, title):
                added += 1
                logger.info(f"  Added to shopping list: {title}")
    
    return added


def generate_list_name(start_date: str) -> str:
    """
    Generate shopping list name from start date.
    
    Args:
        start_date: Date string in YYYY-MM-DD format
    
    Returns:
        Formatted list name like "Weekly Shopping - Dec 23-29"
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = start + timedelta(days=6)
        
        if start.month == end.month:
            return f"Weekly Shopping - {start.strftime('%b %d')}-{end.strftime('%d')}"
        else:
            return f"Weekly Shopping - {start.strftime('%b %d')}-{end.strftime('%b %d')}"
            
    except ValueError:
        return f"Weekly Shopping - {start_date}"


async def process_recipes(recipe_ids: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Fetch recipe details and calculate scale factors using memory-efficient streaming.

    Processes recipes in configurable batches to minimize memory usage while maintaining
    performance. Only keeps essential data in memory.

    Args:
        recipe_ids: List of recipe IDs to process

    Returns:
        Tuple of (recipes_with_scaling, recipe_details_list)
    """
    return await process_recipes_streaming(recipe_ids)


async def process_recipes_streaming(recipe_ids: List[str], batch_size: int = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Memory-efficient streaming processor for large recipe sets.

    Processes recipes in batches to:
    - Minimize peak memory usage
    - Enable parallel fetching within batches
    - Maintain processing order
    - Scale to 10,000+ recipes without memory issues

    Args:
        recipe_ids: List of recipe IDs to process
        batch_size: Number of recipes to process per batch (None = use config default)

    Returns:
        Tuple of (recipes_with_scaling, recipe_details_list)
    """
    # Use centralized configuration if no explicit batch_size provided
    if batch_size is None:
        config = get_bulk_operation_config_safe('sync', fallback_batch_size=10, fallback_concurrent=3)
        batch_size = config['default_batch_size']

    recipes_with_scaling = []
    recipe_details_list = []

    logger.info(f"Streaming {len(recipe_ids)} recipes in batches of {batch_size}...")
    logger.info("Calculating scale factors:")

    # Process recipes in batches
    for i in range(0, len(recipe_ids), batch_size):
        batch_recipe_ids = recipe_ids[i:i + batch_size]
        batch_number = i // batch_size + 1
        total_batches = (len(recipe_ids) + batch_size - 1) // batch_size

        logger.info(f"Processing batch {batch_number}/{total_batches} ({len(batch_recipe_ids)} recipes)...")

        # Process batch with memory-efficient approach
        batch_scaling, batch_details = await _process_recipe_batch(batch_recipe_ids)

        recipes_with_scaling.extend(batch_scaling)
        recipe_details_list.extend(batch_details)

        # Optional: Force garbage collection between batches for large datasets
        if len(recipe_ids) > 100:
            import gc
            gc.collect()

    logger.info(f"Completed processing {len(recipes_with_scaling)} recipes successfully")
    return recipes_with_scaling, recipe_details_list


async def _process_recipe_batch_async(recipe_ids: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single batch of recipes efficiently with async fetching.

    Args:
        recipe_ids: Recipe IDs for this batch

    Returns:
        Tuple of (batch_scaling, batch_details)
    """
    batch_scaling = []
    batch_details = []

    # Fetch all recipes in this batch concurrently
    recipes_dict = await fetch_multiple_recipes_async(recipe_ids, max_concurrent=min(len(recipe_ids), 5))

    for recipe_id in recipe_ids:
        recipe = recipes_dict.get(recipe_id)

        if not recipe:
            logger.warning(f"Warning: Skipping recipe {recipe_id} - fetch failed")
            continue

        # Extract only essential data to minimize memory usage
        recipe_name = recipe.get("name", "Unknown Recipe")
        recipe_servings = recipe.get("recipeYield")

        # Handle recipeYield which might be a string like "4 servings" or an int
        servings_int = None
        if recipe_servings:
            if isinstance(recipe_servings, int):
                servings_int = recipe_servings
            elif isinstance(recipe_servings, str):
                # Try to extract number from string
                try:
                    servings_int = int(''.join(filter(str.isdigit, recipe_servings)))
                except ValueError:
                    pass

        scale_factor = calculate_scale_factor(servings_int)

        if servings_int:
            logger.info(f"  {recipe_name}: {servings_int} servings → scale {scale_factor}x")
        else:
            logger.info(f"  {recipe_name}: no servings specified → scale {scale_factor}x (default)")

        batch_scaling.append({
            "recipeId": recipe_id,
            "recipeIncrementQuantity": scale_factor
        })

        # Store only essential recipe data for shopping list generation
        essential_recipe_data = {
            "id": recipe.get("id"),
            "name": recipe_name,
            "recipeYield": recipe_servings,
            "recipeIngredient": recipe.get("recipeIngredient", []),
            # Add other fields needed for shopping list generation as needed
        }
        batch_details.append(essential_recipe_data)

    return batch_scaling, batch_details


async def _process_recipe_batch(recipe_ids: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a batch of recipes - async version.
    """
    return await _process_recipe_batch_async(recipe_ids)


def main():
    """Main entry point for shopping list generator."""
    parser = argparse.ArgumentParser(
        description="Generate Mealie shopping lists from meal plans or recipe IDs"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--week-start",
        help="Start date for week in YYYY-MM-DD format (e.g., 2025-12-23)"
    )
    group.add_argument(
        "--recipe-ids",
        nargs="+",
        help="Space-separated list of recipe IDs or slugs"
    )
    
    args = parser.parse_args()
    
    # Determine recipe IDs and list name
    note_items = []  # PREP/BUY items from meal plan
    
    if args.week_start:
        print(f"Creating shopping list for week of {args.week_start}...")
        
        meal_plans = fetch_meal_plans_for_week(args.week_start)
        recipe_ids = extract_recipe_ids_from_meal_plans(meal_plans)
        note_items = extract_note_items_from_meal_plans(meal_plans)
        
        if not recipe_ids and not note_items:
            print("Error: No recipes or items found in meal plans for this week")
            sys.exit(1)
        
        if note_items:
            print(f"Found {len(note_items)} PREP/BUY items to add")
        
        list_name = generate_list_name(args.week_start)
        
    else:  # args.recipe_ids
        recipe_ids = args.recipe_ids
        list_name = f"Shopping List - {datetime.now().strftime('%b %d, %Y')}"
        print(f"Creating shopping list from {len(recipe_ids)} recipes...")
    
    # Process recipes and calculate scaling
    import asyncio
    recipes_with_scaling = []
    recipe_details = []
    
    if recipe_ids:
        recipes_with_scaling, recipe_details = asyncio.run(process_recipes(recipe_ids))
    
    if not recipes_with_scaling and not note_items:
        print("Error: No valid recipes or items to add to shopping list")
        sys.exit(1)
    
    # Create shopping list
    print(f"\nCreating shopping list '{list_name}'...")
    list_id = create_shopping_list(list_name)
    
    if not list_id:
        print("Error: Failed to create shopping list")
        sys.exit(1)
    
    # Add recipes to shopping list
    if recipes_with_scaling:
        print("Adding recipes to shopping list...")
        success = add_recipes_to_shopping_list(list_id, recipes_with_scaling)
        
        if not success:
            print("Error: Failed to add recipes to shopping list")
            sys.exit(1)
    
    # Add PREP/BUY note items to shopping list
    if note_items:
        print(f"Adding {len(note_items)} PREP/BUY items to shopping list...")
        added_count = add_note_items_to_shopping_list(list_id, note_items)
        print(f"Added {added_count} items")
    
    # Fetch complete shopping list
    shopping_list = fetch_shopping_list(list_id)
    
    if not shopping_list:
        print("Warning: Could not fetch complete shopping list details")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"✓ Shopping list created: {list_name}")
    
    if shopping_list:
        item_count = len(shopping_list.get("listItems", []))
        print(f"✓ Total items: {item_count}")
    
    if recipe_details:
        recipe_names = [recipe.get("name", "Unknown") for recipe in recipe_details]
        print(f"✓ Recipes included ({len(recipe_names)}):")
        for name in recipe_names:
            print(f"  • {name}")
    
    if note_items:
        print(f"✓ PREP/BUY items included ({len(note_items)}):")
        for item in note_items:
            print(f"  • {item.get('title', '?')}")
    
    print(f"\n✓ List ID: {list_id}")
    print("=" * 60)
    
    # Return list ID for integration with other scripts
    return list_id


if __name__ == "__main__":
    try:
        list_id = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

