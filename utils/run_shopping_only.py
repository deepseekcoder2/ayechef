#!/usr/bin/env python3
"""
Shopping List Only Runner
==========================

Re-run just the shopping list generation and refinement steps without
redoing meal planning. Useful for:
- Testing shopping list refinement after bug fixes
- Regenerating lists with different parameters
- Running refinement on existing lists

Usage:
    # Generate new list + refine it
    python utils/run_shopping_only.py --week-start 2026-01-27
    
    # Only refine an existing list
    python utils/run_shopping_only.py --list-id <uuid> --refine-only
    
    # Generate list without refinement
    python utils/run_shopping_only.py --week-start 2026-01-27 --no-refine
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Optional

from utils.shopping_list_generator import (
    fetch_meal_plans_for_week,
    extract_recipe_ids_from_meal_plans,
    process_recipes,
    create_shopping_list,
    add_recipes_to_shopping_list,
    fetch_shopping_list,
    generate_list_name
)
from tools.logging_utils import get_logger

logger = get_logger(__name__)


async def generate_shopping_list(week_start: str) -> Optional[str]:
    """
    Generate shopping list from meal plans for the given week.
    
    Args:
        week_start: Week start date in YYYY-MM-DD format
        
    Returns:
        Shopping list ID or None if failed
    """
    print(f"\n[Step 1/2] Creating shopping list for week of {week_start}...")
    
    # Fetch meal plans
    meal_plans = fetch_meal_plans_for_week(week_start)
    if not meal_plans:
        print("❌ No meal plans found for this week")
        return None
    
    # Extract recipe IDs
    recipe_ids = extract_recipe_ids_from_meal_plans(meal_plans)
    if not recipe_ids:
        print("❌ No recipes found in meal plans")
        return None
    
    print(f"   Found {len(meal_plans)} meal plan entries with {len(recipe_ids)} unique recipes")
    
    # Process recipes and calculate scaling
    recipes_with_scaling, recipe_details = await process_recipes(recipe_ids)
    if not recipes_with_scaling:
        print("❌ No valid recipes to add to shopping list")
        return None
    
    # Create shopping list
    list_name = generate_list_name(week_start)
    print(f"   Creating list: {list_name}")
    list_id = create_shopping_list(list_name)
    
    if not list_id:
        print("❌ Failed to create shopping list")
        return None
    
    # Add recipes to shopping list
    success = add_recipes_to_shopping_list(list_id, recipes_with_scaling)
    if not success:
        print("❌ Failed to add recipes to shopping list")
        return None
    
    # Fetch complete shopping list for summary
    shopping_list = fetch_shopping_list(list_id)
    if shopping_list:
        item_count = len(shopping_list.get("listItems", []))
        print(f"✅ Shopping list created: {item_count} items")
    else:
        print("✅ Shopping list created")
    
    return list_id


async def refine_shopping_list(list_id: str) -> bool:
    """
    Run refinement on an existing shopping list.
    
    Uses shopping_pipeline_v2 which PRESERVES foodId/unitId from original items.
    This is critical - without these IDs, Mealie merges items incorrectly.
    
    Args:
        list_id: Shopping list UUID
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n[Step 2/2] Refining shopping list...")
    
    try:
        from mealie_shopping_integration import (
            fetch_mealie_shopping_list, 
            delete_mealie_shopping_items,
            add_mealie_shopping_item
        )
        from shopping_pipeline_v2 import refine_shopping_list as v2_refine, format_for_mealie

        # Fetch shopping list from Mealie
        shopping_list = fetch_mealie_shopping_list(list_id)
        if not shopping_list:
            print(f"❌ Failed to fetch shopping list {list_id}")
            return False

        items = shopping_list.get("listItems", [])
        if not items:
            print("✅ No ingredients to refine")
            return True

        print(f"   Processing {len(items)} items...")

        # Run v2 refinement (filter + clean, PRESERVES foodId/unitId)
        result = v2_refine(items)

        if not result.success:
            print(f"❌ Refinement failed: {result.errors}")
            return False

        # Log results
        print(f"   Items to keep: {len(result.items_to_keep)}")
        print(f"   Items to delete: {len(result.items_to_delete)}")
        if result.pantry_filtered:
            print(f"   Pantry filtered: {len(result.pantry_filtered)} items")

        # Delete filtered items from Mealie
        if result.items_to_delete:
            delete_mealie_shopping_items(result.items_to_delete)

        # Delete remaining items (we'll re-add with cleaned display)
        remaining_ids = [item.original_id for item in result.items_to_keep if item.original_id]
        if remaining_ids:
            delete_mealie_shopping_items(remaining_ids)

        # Format items for Mealie and add them back ONE BY ONE
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

        print(f"✅ Refinement completed: {successful} items added")
        if failed > 0:
            print(f"⚠️  {failed} items failed to add")
        
        return True

    except Exception as e:
        print(f"❌ Refinement failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Run shopping list generation and refinement without redoing meal planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate + refine shopping list for current week
    python utils/run_shopping_only.py --week-start 2026-01-27
    
    # Generate list without refinement
    python utils/run_shopping_only.py --week-start 2026-01-27 --no-refine
    
    # Refine an existing list
    python utils/run_shopping_only.py --list-id <uuid> --refine-only
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--week-start",
        help="Start date for week in YYYY-MM-DD format (e.g., 2026-01-27)"
    )
    group.add_argument(
        "--list-id",
        help="Existing shopping list ID to refine"
    )
    
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="Only refine existing list (use with --list-id)"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Generate list without refinement (use with --week-start)"
    )
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.refine_only and not args.list_id:
        print("❌ Error: --refine-only requires --list-id")
        return 1
    
    if args.no_refine and args.list_id:
        print("❌ Error: --no-refine cannot be used with --list-id")
        return 1
    
    print("\n" + "="*60)
    print("🛒 SHOPPING LIST RUNNER")
    print("="*60)
    
    try:
        if args.refine_only:
            # Just refine existing list
            success = await refine_shopping_list(args.list_id)
            if success:
                print("\n" + "="*60)
                print(f"✅ List refined successfully!")
                print(f"   List ID: {args.list_id}")
                print("="*60 + "\n")
                return 0
            else:
                return 1
        
        elif args.week_start:
            # Generate new list
            list_id = await generate_shopping_list(args.week_start)
            if not list_id:
                return 1
            
            # Optionally refine it
            if not args.no_refine:
                success = await refine_shopping_list(list_id)
                if not success:
                    print(f"\n⚠️  List created but refinement failed")
                    print(f"   You can retry refinement with:")
                    print(f"   python utils/run_shopping_only.py --list-id {list_id} --refine-only")
            
            print("\n" + "="*60)
            print(f"✅ Shopping list ready!")
            print(f"   List ID: {list_id}")
            print("="*60 + "\n")
            return 0
        
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
