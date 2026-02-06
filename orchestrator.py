#!/usr/bin/env python3
"""
Mealie Orchestrator - Weekly Meal Planning Pipeline
===================================================

Main entry point for the Mealie Recipe Management System.

Runs the full meal planning pipeline:
1. Validates environment and connections
2. Parses any unparsed recipe ingredients
3. Runs agentic AI chef to plan meals
4. Generates shopping list from meal plans
5. Aggregates and refines shopping list
6. Exports to WhatsApp format

USAGE:
    python orchestrator.py                    # Plan for next Monday
    python orchestrator.py --week-start DATE  # Plan for specific date
    python orchestrator.py --dry-run          # Test run without changes
    python orchestrator.py --help             # Show all options

TARGET: Complete pipeline in ~30-45 minutes (LLM processing is the bottleneck)
"""

import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

# Import configuration and validation
from config import validate_all, MEALIE_TOKEN, DATA_DIR
from tools.logging_utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Import component modules
from utils import shopping_list_generator
from utils import whatsapp_export
import mealie_parse
from mealie_client import MealieClient

# Import recipe image fetching (for post-meal-planning image retrieval)
from recipe_images import fetch_images_for_recipes, BRAVE_API_KEY


# =============================================================================
# PROGRESS DISPLAY UTILITIES
# =============================================================================

def print_header():
    """Print fancy header for the orchestrator."""
    print("\n" + "═" * 60)
    print("🍽️  WEEKLY MEAL PLANNER")
    print("═" * 60)


def print_step(step_num: int, total_steps: int, message: str):
    """Print step progress indicator."""
    print(f"\n[Step {step_num}/{total_steps}] {message}")


def print_success(total_time: float):
    """Print success summary."""
    print("\n" + "═" * 60)
    print(f"✅ SUCCESS! Total time: {int(total_time)} seconds")
    print("═" * 60)
    print("\n📱 Ready to paste in WhatsApp!")
    print("📄 Backup saved to: meal_plan.txt")
    print()


def print_error(step_name: str, error_message: str):
    """Print error message and exit."""
    logger.critical(f"❌ ERROR in {step_name}: {error_message}")
    print("\n" + "═" * 60)
    print(f"❌ ERROR in {step_name}")
    print("═" * 60)
    print(f"\n{error_message}\n")
    sys.exit(1)


def print_thinking_progress(message: str, expected_seconds: int):
    """Print progress dots during long operations."""
    print(f"🤖 {message}")
    print(f"   Expected time: ~{expected_seconds} seconds")


# =============================================================================
# DATE CALCULATION
# =============================================================================

def calculate_next_monday(date_str: Optional[str] = None) -> datetime:
    """
    Calculate next Monday or parse provided date.
    
    Args:
        date_str: Optional date in YYYY-MM-DD format
    
    Returns:
        datetime object for the start of the week (Monday)
    """
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print_error("Date Parsing", f"Invalid date format: {date_str}\nUse YYYY-MM-DD format")
    
    # Calculate next Monday from today
    today = datetime.now()
    days_ahead = (7 - today.weekday()) % 7 or 7  # weekday(): Monday=0, Sunday=6
    next_monday = today + timedelta(days=days_ahead)
    
    return next_monday.replace(hour=0, minute=0, second=0, microsecond=0)


# =============================================================================
# COMPONENT ORCHESTRATION
# =============================================================================

def run_validation(skip_validation: bool) -> bool:
    """
    Validate environment before starting.
    
    Args:
        skip_validation: If True, skip validation checks
    
    Returns:
        True if validation passed or was skipped
    """
    if skip_validation:
        logger.warning("⚠️  Skipping validation (--skip-validation flag set)")
        logger.warning("   This may cause errors if services are down!")
        return True
    
    logger.info("🔍 Validating environment...")
    print_step(0, 0, "🔍 Validating environment...")
    
    # Check MEALIE_TOKEN first
    if not MEALIE_TOKEN:
        print_error(
            "Configuration",
            "MEALIE_TOKEN environment variable not set!\n"
            "Export your token: export MEALIE_TOKEN='your-token-here'"
        )
    
    # Run full validation
    validation_start = time.time()
    if not validate_all():
        print_error(
            "Validation",
            "System validation failed. Please fix the errors above."
        )
    
    validation_time = time.time() - validation_start
    logger.info(f"✅ Validation complete ({validation_time:.1f}s)")
    print(f"✅ Validation complete ({validation_time:.1f}s)\n")
    
    return True


def validate_parsing_results() -> bool:
    """
    Validate that all recipes have properly parsed ingredients.
    Returns True if all recipes are properly parsed.
    """
    client = MealieClient()
    try:
        # Get all recipes
        recipes = client.get_all_recipes()

        total_recipes = 0
        unparsed_recipes = 0

        # Check each recipe (limit to first page for performance)
        from config import get_config_value
        max_recipes_check = get_config_value('ingredient_parsing', 'max_recipes_per_batch', 50)
        for recipe_summary in recipes[:max_recipes_check]:  # Check configurable number of recipes
            total_recipes += 1
            slug = recipe_summary['slug']

            # Get detailed recipe data
            recipe_detail = client.get_recipe(slug)

            # Check if recipe has unparsed ingredients
            ingredients = recipe_detail.get('recipeIngredient', [])
            if not ingredients:
                continue

            has_unparsed = False
            for ing in ingredients:
                if isinstance(ing, dict):
                    qty = ing.get('quantity', 0)
                    unit = ing.get('unit')
                    food = ing.get('food')

                    # Ingredient is unparsed if quantity=0 and both unit and food are None
                    if qty == 0.0 and unit is None and food is None:
                        has_unparsed = True
                        break

            if has_unparsed:
                unparsed_recipes += 1

        if unparsed_recipes > 0:
            print(f"   Found {unparsed_recipes} recipes with unparsed ingredients out of {total_recipes} checked")
            return False
        else:
            print(f"   Checked {total_recipes} recipes - all have properly parsed ingredients")
            return True

    except Exception as e:
        logger.warning(f"⚠️  Warning: Could not validate parsing results: {e}")
        print(f"⚠️  Warning: Could not validate parsing results: {e}")
        return True  # Don't fail the workflow for validation errors
    finally:
        client.close()


def check_recipe_quality_warning() -> dict:
    """
    Check for recipe quality issues and return warning info.
    
    Returns dict with:
        - has_issues: bool
        - unparsed: int
        - untagged: int  
        - unindexed: int
        - total: int
    """
    import sqlite3
    from pathlib import Path
    
    result = {
        "has_issues": False,
        "unparsed": 0,
        "untagged": 0,
        "unindexed": 0,
        "total_mealie": 0,
        "total_indexed": 0
    }
    
    client = MealieClient()
    try:
        # Get Mealie total
        recipes = client.get_all_recipes()
        result["total_mealie"] = len(recipes)
        
        # Check local DB
        db_path = DATA_DIR / "recipe_index.db"
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Total indexed
            cursor.execute("SELECT COUNT(*) FROM recipes")
            result["total_indexed"] = cursor.fetchone()[0]
            
            # Untagged count
            cursor.execute("""
                SELECT COUNT(*) FROM recipes 
                WHERE cuisine_primary IS NULL OR cuisine_primary = ''
            """)
            result["untagged"] = cursor.fetchone()[0]
            
            conn.close()
        
        # Unindexed
        result["unindexed"] = max(0, result["total_mealie"] - result["total_indexed"])
        
        # Check unparsed (uses cache, should be fast after first run)
        try:
            from mealie_parse import get_unparsed_slugs
            unparsed = get_unparsed_slugs()
            result["unparsed"] = len(unparsed) if unparsed else 0
        except Exception:
            pass
        
        result["has_issues"] = (result["unparsed"] + result["untagged"] + result["unindexed"]) > 0
        
    except Exception as e:
        logger.warning(f"Could not check recipe quality: {e}")
    finally:
        client.close()
    
    return result


async def run_chef_agent(week_start: datetime, dry_run: bool, candidate_k: int = 25, max_refines: int = 1, 
                         preferred_cuisines: Optional[List[str]] = None, dietary_restrictions: Optional[List[str]] = None,
                         temp_prompt: str = "") -> Tuple[list, Dict]:
    """
    Run agentic AI Chef to create meal plan.
    
    Args:
        week_start: Start date for the week
        dry_run: If True, plan but don't write to Mealie
        candidate_k: Number of candidates to sample per role (default 25)
        max_refines: Max retry attempts for role validation (default 1)
        temp_prompt: One-shot instructions for this planning session
    
    Returns:
        Tuple of (recipe_ids_list, agent_state_dict)
    """
    week_end = week_start + timedelta(days=6)
    week_str = f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}"
    
    print(f"📆 Planning week: {week_str}")
    
    if dry_run:
        print("\n🔷 DRY RUN MODE: Will plan meals but NOT write to Mealie")
    
    print_thinking_progress("Agentic Chef planning slots...", 600)  # 10 minutes estimate
    
    chef_start = time.time()
    
    try:
        # Import chef_agentic's planning function
        import chef_agentic
        
        # Run the agentic planning workflow (always run, even in dry-run)
        # This returns an AgentState object with the full plan
        week_start_str = week_start.strftime("%Y-%m-%d")
        state = await chef_agentic.plan_week_agentic(
            start_date_str=week_start_str,
            candidate_k=candidate_k,
            max_refines=max_refines,
            preferred_cuisines=preferred_cuisines,
            dietary_restrictions=dietary_restrictions,
            temp_prompt=temp_prompt,
        )
        
        # Extract recipe IDs from the planned state
        used_recipe_ids = list(state.used_recipe_ids)
        
        # Print the plan
        chef_agentic._print_plan(state, dry_run=dry_run)
        
        # Write to Mealie only if not dry-run
        if not dry_run:
            client = MealieClient()
            try:
                await chef_agentic.write_plan_to_mealie(client, state)
            finally:
                client.close()
        else:
            print("\n💡 DRY RUN: Skipping write to Mealie (plan was generated above)")
        
        chef_time = time.time() - chef_start

        # Count unique recipes and total meals
        unique_recipes = len(set(used_recipe_ids))
        total_meals = sum(len(meal.dishes) for day_meals in state.planned.values() for meal in day_meals.values())

        if dry_run:
            print(f"\n✅ Meal plan generated: {total_meals} dishes across 14 meals, {unique_recipes} unique recipes ({chef_time:.1f}s)")
        else:
            print(f"\n✅ Meal plan created: {total_meals} dishes across 14 meals, {unique_recipes} unique recipes ({chef_time:.1f}s)")

        # Return recipe IDs and state (as dict for compatibility)
        return used_recipe_ids, {"state": state, "week_start": week_start}
    
    except Exception as e:
        print_error("AI Chef", f"Failed to create meal plan:\n{str(e)}")




async def run_shopping_list_generator(week_start: datetime, dry_run: bool) -> Tuple[str, int]:
    """
    Generate shopping list from meal plan.
    
    Args:
        week_start: Start date for the week
        dry_run: If True, don't actually create shopping list
    
    Returns:
        Tuple of (list_id, item_count)
    """
    week_start_str = week_start.strftime("%Y-%m-%d")
    
    if dry_run:
        print("\n🔷 DRY RUN MODE: Would create shopping list but skipping")
        return "dry-run-list-id", 0
    
    shopping_start = time.time()
    
    try:
        # Fetch meal plans for the week
        meal_plans = shopping_list_generator.fetch_meal_plans_for_week(week_start_str)
        
        if not meal_plans:
            print_error("Shopping List", "No meal plans found for this week")
        
        # Extract recipe IDs
        recipe_ids = shopping_list_generator.extract_recipe_ids_from_meal_plans(meal_plans)
        
        if not recipe_ids:
            print_error("Shopping List", "No recipes found in meal plans")
        
        # Process recipes and calculate scaling
        recipes_with_scaling, recipe_details = await shopping_list_generator.process_recipes(recipe_ids)
        
        if not recipes_with_scaling:
            print_error("Shopping List", "No valid recipes to add to shopping list")
        
        # Generate list name
        list_name = shopping_list_generator.generate_list_name(week_start_str)
        
        # Create shopping list
        print(f"\nCreating shopping list '{list_name}'...")
        list_id = shopping_list_generator.create_shopping_list(list_name)
        
        if not list_id:
            print_error("Shopping List", "Failed to create shopping list in Mealie")
        
        # Add recipes to shopping list
        print("Adding recipes to shopping list...")
        success = shopping_list_generator.add_recipes_to_shopping_list(
            list_id, 
            recipes_with_scaling
        )
        
        if not success:
            print_error("Shopping List", "Failed to add recipes to shopping list")
        
        # Extract and add PREP/BUY note items from meal plan
        note_items = shopping_list_generator.extract_note_items_from_meal_plans(meal_plans)
        if note_items:
            print(f"Adding {len(note_items)} PREP/BUY items to shopping list...")
            added_count = shopping_list_generator.add_note_items_to_shopping_list(list_id, note_items)
            print(f"  Added {added_count} PREP/BUY items")
        
        # Fetch complete shopping list to get item count
        shopping_list = shopping_list_generator.fetch_shopping_list(list_id)
        item_count = len(shopping_list.get("listItems", [])) if shopping_list else 0
        
        shopping_time = time.time() - shopping_start
        
        print(f"\n✅ Shopping list created: {item_count} items ({shopping_time:.1f}s)")
        
        return list_id, item_count
    
    except Exception as e:
        print_error("Shopping List", f"Failed to create shopping list:\n{str(e)}")


async def run_whatsapp_export(list_id: str, week_start: datetime, dry_run: bool) -> bool:
    """
    Export shopping list and meal plan to WhatsApp format.
    
    Args:
        list_id: Shopping list UUID
        week_start: Start date for the week
        dry_run: If True, don't actually export
    
    Returns:
        True if export successful
    """
    if dry_run:
        print("\n🔷 DRY RUN MODE: Would export to WhatsApp but skipping")
        return True
    
    export_start = time.time()
    
    try:
        week_start_str = week_start.strftime("%Y-%m-%d")
        week_end = week_start + timedelta(days=6)
        week_end_str = week_end.strftime("%Y-%m-%d")
        
        # Fetch refined shopping list from Mealie
        print("Fetching shopping list...")
        client = MealieClient()
        try:
            shopping_list_data = whatsapp_export.fetch_shopping_list(client, list_id)
        finally:
            client.close()
        
        # Fetch meal plan
        date_range_display = whatsapp_export.get_date_range_display(week_start)
        print(f"Fetching meal plan for {date_range_display}...")
        client = MealieClient()
        try:
            meal_plan_data = whatsapp_export.fetch_meal_plan(client, week_start_str, week_end_str)
        finally:
            client.close()
        
        # Generate all dates in range
        all_dates = []
        current = week_start
        while current <= week_end:
            all_dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        # Group meal plans by date
        print("Formatting message...")
        grouped_plans = whatsapp_export.group_meal_plans_by_date(meal_plan_data)
        
        # Check for overnight prep requirements (fast - just checks tags)
        print("Checking for overnight prep requirements...")
        prep_warnings = whatsapp_export.detect_overnight_prep(grouped_plans, all_dates)
        
        # Build message
        message_lines = []
        
        # Shopping list section
        message_lines.extend(whatsapp_export.format_shopping_list(
            shopping_list_data, 
            date_range_display
        ))
        
        # Meal plan section
        message_lines.extend(whatsapp_export.format_meal_plan(
            grouped_plans, 
            all_dates, 
            prep_warnings
        ))
        
        # Join all lines
        message = '\n'.join(message_lines)
        
        # Save to file
        file_saved = whatsapp_export.save_to_file(message)
        
        # Copy to clipboard
        clipboard_copied = whatsapp_export.copy_to_clipboard(message)
        
        if clipboard_copied:
            clipboard_status = "✅ Copied to clipboard"
        elif file_saved:
            clipboard_status = "📄 Saved to data/meal_plan.txt (clipboard not available)"
        else:
            clipboard_status = "⚠️ Message generated but could not save or copy"
        
        export_time = time.time() - export_start
        
        print(f"\n✅ Message formatted and exported ({export_time:.1f}s)")
        print(f"   {clipboard_status}")
        
        return True
    
    except Exception as e:
        print_error("WhatsApp Export", f"Failed to export:\n{str(e)}")


async def run_shopping_list_aggregation(list_id: str, dry_run: bool = False) -> bool:
    """
    Aggregate shopping list items by converting units and grouping by food_id.
    
    This happens BEFORE refinement so the refined list has cleaner data.
    Conservative approach: only deletes items we successfully aggregate.
    
    Args:
        list_id: Shopping list UUID
        dry_run: If True, don't actually aggregate
        
    Returns:
        True if successful
    """
    try:
        if dry_run:
            print("🔷 DRY RUN MODE: Would aggregate shopping list but skipping")
            return True
        
        from shopping_aggregator import aggregate_and_write_to_mealie
        from mealie_shopping_integration import fetch_mealie_shopping_list
        
        # Fetch shopping list
        shopping_list = fetch_mealie_shopping_list(list_id)
        if not shopping_list:
            raise RuntimeError(f"Failed to fetch shopping list {list_id}")
        
        items = shopping_list.get("listItems", [])
        if not items:
            print("✅ No items to aggregate")
            return True
        
        print(f"📊 Aggregating {len(items)} items by unit conversion...")
        
        aggregation_start = time.time()
        
        # Run aggregation
        success = await aggregate_and_write_to_mealie(list_id, items)
        
        aggregation_time = time.time() - aggregation_start
        
        if success:
            print(f"✅ Shopping list aggregated ({aggregation_time:.1f}s)")
        else:
            print("❌ Aggregation failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Shopping list aggregation failed: {e}", exc_info=True)
        print(f"❌ Shopping list aggregation failed: {e}")
        raise


async def run_shopping_list_refinement(list_id: str, dry_run: bool = False) -> list:
    """
    Refine shopping list by filtering pantry items and cleaning display text.
    
    Uses v2 approach that works WITH Mealie's aggregation:
    1. Mealie aggregates quantities correctly when adding recipes
    2. We filter out pantry items (DELETE)
    3. We filter out garbage items (DELETE)
    4. We clean up ugly display text (PRESERVE quantities)
    
    This fixes the "1 gram cheese" bug from the old pipeline that corrupted quantities.
    
    Returns:
        list: Refined shopping list items (also written to Mealie)
    """
    try:
        if dry_run:
            print("🔷 DRY RUN MODE: Would refine shopping list but skipping")
            return []

        from mealie_shopping_integration import (
            fetch_mealie_shopping_list, 
            delete_mealie_shopping_items,
            add_mealie_shopping_item
        )
        from shopping_pipeline_v2 import refine_shopping_list, format_for_mealie

        # Fetch shopping list from Mealie
        shopping_list = fetch_mealie_shopping_list(list_id)
        if not shopping_list:
            raise RuntimeError(f"Failed to fetch shopping list {list_id} from Mealie")

        items = shopping_list.get("listItems", [])
        if not items:
            print("✅ No ingredients to refine")
            return []

        print(f"📋 Refining {len(items)} items (filter pantry, clean display, preserve quantities)...")

        # Run v2 refinement (filter + clean, preserves Mealie's quantities)
        result = refine_shopping_list(items)

        if not result.success:
            raise RuntimeError(f"Refinement failed: {result.errors}")

        # Log results
        print(f"   Items to keep: {len(result.items_to_keep)}")
        print(f"   Items to delete: {len(result.items_to_delete)}")
        if result.pantry_filtered:
            print(f"   Pantry filtered: {len(result.pantry_filtered)} items")
        if result.garbage_rejected:
            print(f"   Garbage rejected: {len(result.garbage_rejected)} items")

        # Delete filtered items from Mealie
        if result.items_to_delete:
            print(f"   Deleting {len(result.items_to_delete)} filtered items...")
            delete_mealie_shopping_items(result.items_to_delete)

        # Delete remaining items (we'll re-add with cleaned display)
        remaining_ids = [item.original_id for item in result.items_to_keep if item.original_id]
        if remaining_ids:
            print(f"   Replacing {len(remaining_ids)} items with cleaned versions...")
            delete_mealie_shopping_items(remaining_ids)

        # Format items for Mealie and add them back
        mealie_items = format_for_mealie(result.items_to_keep)
        
        successful = 0
        failed = 0
        for item in mealie_items:
            try:
                add_mealie_shopping_item(list_id, item)
                successful += 1
            except Exception as e:
                logger.warning(f"Failed to add item {item.get('display', 'unknown')}: {e}")
                failed += 1

        print(f"✅ Shopping list refined: {successful} items with correct quantities")
        if failed > 0:
            print(f"⚠️  {failed} items failed to add")
        
        # Return the refined items for downstream use
        return [{"display": item.cleaned_display, "food_name": item.food_name} for item in result.items_to_keep]

    except Exception as e:
        print(f"❌ Shopping list refinement failed: {e}")
        raise  # Fail fast - don't continue with unrefined list


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

async def main():
    """Main orchestration function - async for proper event loop management."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="AI-powered weekly meal planning system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py                    # Plan next week
  python orchestrator.py --week-start 2025-12-30
  python orchestrator.py --dry-run          # Test without making changes
  python orchestrator.py --skip-validation  # Skip health checks (faster)
        """
    )
    
    parser.add_argument(
        "--week-start",
        help="Start date for week in YYYY-MM-DD format (default: next Monday)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Mealie/LLM health checks (faster but risky)"
    )
    
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=25,
        help="Number of candidates to sample per role (default: 25)"
    )
    
    parser.add_argument(
        "--max-refines",
        type=int,
        default=1,
        help="Max retry attempts for role validation (default: 1)"
    )
    
    parser.add_argument(
        "--cuisines",
        help="Comma-separated list of preferred cuisines to prioritize"
    )
    
    parser.add_argument(
        "--restrictions",
        help="Comma-separated list of dietary restrictions to exclude"
    )
    
    parser.add_argument(
        "--temp-prompt",
        help="One-shot instructions for this planning session (e.g., 'Use up leftover chicken')"
    )
    
    args = parser.parse_args()
    
    # Start timer
    start_time = time.time()
    
    # Print header
    print_header()
    
    # Calculate week start date
    week_start = calculate_next_monday(args.week_start)
    
    if args.dry_run:
        print("\n🔷 DRY RUN MODE ENABLED - No changes will be made\n")
    
    # Step 0: Validation
    validation_start = time.time()
    run_validation(args.skip_validation)
    validation_time = time.time() - validation_start

    # Step 1: Check recipe quality (info only, no processing)
    print_step(1, 6, "Checking recipe collection...")
    quality = check_recipe_quality_warning()
    
    if quality["has_issues"]:
        # Only mention if significant (>10 recipes)
        issues = []
        if quality["unparsed"] > 10:
            issues.append(f"{quality['unparsed']} unparsed")
        if quality["untagged"] > 10:
            issues.append(f"{quality['untagged']} untagged")
        if quality["unindexed"] > 10:
            issues.append(f"{quality['unindexed']} unindexed")
        
        if issues:
            print(f"   ℹ️  Note: {', '.join(issues)} recipes could benefit from maintenance")
            print(f"   Using {quality['total_indexed']} indexed recipes for planning\n")
        else:
            print(f"   ✅ {quality['total_indexed']} recipes ready")
    else:
        print(f"   ✅ {quality['total_indexed']} recipes ready")

    # Step 2: AI Chef - Create meal plan (agentic approach - writes directly to Mealie)
    print_step(2, 6, "Planning meals with Agentic Chef...")
    # Parse cuisine/restriction preferences from args
    preferred_cuisines = None
    dietary_restrictions = None
    if args.cuisines:
        preferred_cuisines = [c.strip() for c in args.cuisines.split(',') if c.strip()]
    if args.restrictions:
        dietary_restrictions = [r.strip() for r in args.restrictions.split(',') if r.strip()]
    
    recipe_ids, meal_plan = await run_chef_agent(
        week_start, 
        args.dry_run, 
        candidate_k=args.candidate_k,
        max_refines=args.max_refines,
        preferred_cuisines=preferred_cuisines,
        dietary_restrictions=dietary_restrictions,
        temp_prompt=args.temp_prompt or "",
    )
    
    # Note: Agentic chef already writes meal plan entries to Mealie via write_plan_to_mealie()
    # No need for additional meal plan creation step

    # Step 3: Shopping List Generator
    print_step(3, 6, "Creating shopping list...")
    list_id, item_count = await run_shopping_list_generator(week_start, args.dry_run)

    # Step 4: Shopping List Aggregation (NEW - aggregates mixed units before refinement)
    print_step(4, 6, "Aggregating shopping list...")
    await run_shopping_list_aggregation(list_id, args.dry_run)

    # Step 5: Shopping List Refinement (filter pantry items)
    print_step(5, 6, "Refining shopping list...")
    # Fail fast - if refinement fails, stop the pipeline
    # Writes refined items to Mealie with proper foodId/unitId to prevent merge corruption
    refined_items = await run_shopping_list_refinement(list_id, args.dry_run)

    # Step 6: WhatsApp Export
    print_step(6, 6, "Formatting for WhatsApp...")
    # Use proper async concurrency within single event loop
    # Fetch from Mealie (now has clean refined items with proper display)
    try:
        export_success = await run_whatsapp_export(list_id, week_start, args.dry_run)
        if not export_success:
            print("⚠️  WhatsApp export failed")
            print("   Shopping list may still be available in Mealie UI")
    except Exception as e:
        print(f"⚠️  WhatsApp export failed: {e}")
        print("   Shopping list may still be available in Mealie UI")
    
    # Optional: Fetch images for newly created recipes
    # Runs at the very end, only if:
    # 1. Brave API is configured
    # 2. Recipes were actually created (not just using existing)
    # 3. Not in dry-run mode
    image_results = None
    state = meal_plan.get("state")
    created_recipes = state.created_recipes if state and hasattr(state, 'created_recipes') else []
    
    if not args.dry_run and BRAVE_API_KEY and created_recipes:
        logger.info(f"🖼️  Fetching images for {len(created_recipes)} newly created recipes...")
        print(f"\n🖼️  Fetching images for {len(created_recipes)} newly created recipes...")
        
        try:
            batch_result = await fetch_images_for_recipes(created_recipes)
            
            # Store results for job detail display
            image_results = {
                "success_count": batch_result.success_count,
                "skipped_count": batch_result.skipped_count,
                "rate_limited": batch_result.rate_limited,
                "results": [
                    {"recipe": r.recipe_name, "status": r.status, "message": r.message}
                    for r in batch_result.results
                ]
            }
            
            # Log summary
            logger.info(f"✅ Image fetch complete: {batch_result.success_count} success, {batch_result.skipped_count} skipped")
            print(f"   ✅ Images: {batch_result.success_count} fetched, {batch_result.skipped_count} skipped")
            
            if batch_result.rate_limited:
                logger.warning("⚠️  Brave API rate limit reached - some images skipped")
                print("   ⚠️  Rate limit reached - some images skipped")
                
        except Exception as e:
            # Image fetching should never fail the meal plan
            logger.warning(f"Image fetching failed (non-blocking): {e}")
            print(f"   ⚠️  Image fetching failed: {e}")
            image_results = {"error": str(e)}
    elif args.dry_run and created_recipes:
        print(f"\n🔷 DRY RUN: Would fetch images for {len(created_recipes)} recipes")
    elif not BRAVE_API_KEY and created_recipes:
        logger.debug("Skipping image fetch - BRAVE_API_KEY not configured")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print success summary
    print_success(total_time)
    
    # Print performance breakdown if verbose
    if total_time > 2700:  # 45 minutes
        print("⚠️  Warning: Total time exceeded 45 minutes")
        print("   This is unusually slow - check LLM API or network connection\n")
    
    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(130)

    except Exception as e:
        print("\n" + "═" * 60)
        print("❌ UNEXPECTED ERROR")
        print("═" * 60)
        print(f"\n{str(e)}\n")
        
        # Print full traceback for debugging
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        sys.exit(1)

