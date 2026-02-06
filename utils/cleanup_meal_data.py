#!/usr/bin/env python3
"""
Cleanup Meal Plans and Shopping Lists
======================================

This script wipes all meal plans and shopping lists from Mealie while
preserving recipes, tags, and other imported data.

Use this to start fresh with meal planning without losing your recipe database.

IMPORTANT: This is a destructive operation! Make sure you want to delete
all meal plans and shopping lists before running this script.

Usage:
    python utils/cleanup_meal_data.py              # Interactive confirmation
    python utils/cleanup_meal_data.py --confirm    # Skip confirmation
    python utils/cleanup_meal_data.py --dry-run    # Preview only, no deletion
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from mealie_client import MealieClient
from tools.logging_utils import get_logger

logger = get_logger(__name__)


def fetch_all_shopping_lists(client: MealieClient) -> List[Dict]:
    """
    Fetch all shopping lists from Mealie.
    
    Args:
        client: MealieClient instance
    
    Returns:
        List of shopping list objects
    """
    try:
        logger.info("Fetching all shopping lists...")
        all_lists = client.get_all_shopping_lists()
        logger.info(f"Found {len(all_lists)} shopping list(s)")
        return all_lists
        
    except Exception as e:
        logger.error(f"Error fetching shopping lists: {e}")
        return []


def delete_shopping_list(client: MealieClient, list_id: str, list_name: str, dry_run: bool = False) -> bool:
    """
    Delete a shopping list by ID.
    
    Args:
        client: MealieClient instance
        list_id: Shopping list UUID
        list_name: Shopping list name (for logging)
        dry_run: If True, only simulate deletion
    
    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would delete shopping list: {list_name} ({list_id})")
        return True
    
    try:
        logger.info(f"Deleting shopping list: {list_name}")
        success = client.delete_shopping_list(list_id)
        if success:
            logger.info(f"✅ Deleted shopping list: {list_name}")
            return True
        else:
            logger.error(f"❌ Failed to delete shopping list {list_name}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error deleting shopping list {list_name}: {e}")
        return False


def fetch_all_meal_plans(client: MealieClient, weeks_back: int = 52) -> List[Dict]:
    """
    Fetch all meal plan entries from Mealie.
    
    Args:
        client: MealieClient instance
        weeks_back: Number of weeks to look back (default 52 = 1 year)
    
    Returns:
        List of meal plan entry objects
    """
    # Create a broad date range to capture all meal plans
    end_date = datetime.now().date() + timedelta(weeks=52)  # Look ahead 1 year too
    start_date = end_date - timedelta(weeks=weeks_back * 2)  # Look back double
    
    try:
        logger.info(f"Fetching all meal plans from {start_date} to {end_date}...")
        all_plans = client.get_meal_plans(start_date, end_date)
        logger.info(f"Found {len(all_plans)} meal plan entry/entries")
        return all_plans
        
    except Exception as e:
        logger.error(f"Error fetching meal plans: {e}")
        return []


def delete_meal_plan_entry(client: MealieClient, entry_id: str, entry_info: str, dry_run: bool = False) -> bool:
    """
    Delete a meal plan entry by ID.
    
    Args:
        client: MealieClient instance
        entry_id: Meal plan entry UUID
        entry_info: Human-readable info (for logging)
        dry_run: If True, only simulate deletion
    
    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would delete meal plan entry: {entry_info} ({entry_id})")
        return True
    
    try:
        success = client.delete_meal_plan_entry(entry_id)
        return success
        
    except Exception as e:
        logger.error(f"❌ Error deleting meal plan entry {entry_info}: {e}")
        return False


def cleanup_shopping_lists(client: MealieClient, dry_run: bool = False) -> Tuple[int, int]:
    """
    Delete all shopping lists.
    
    Args:
        client: MealieClient instance
        dry_run: If True, only simulate deletion
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    print("\n" + "="*60)
    print("🧹 CLEANING UP SHOPPING LISTS")
    print("="*60)
    
    lists = fetch_all_shopping_lists(client)
    
    if not lists:
        print("No shopping lists found.")
        return 0, 0
    
    print(f"\nFound {len(lists)} shopping list(s) to delete:")
    for lst in lists:
        list_name = lst.get("name", "Unnamed")
        list_id = lst.get("id", "Unknown")
        item_count = len(lst.get("listItems", []))
        print(f"  - {list_name} ({item_count} items)")
    
    if not dry_run:
        print()
    
    successful = 0
    failed = 0
    
    for lst in lists:
        list_name = lst.get("name", "Unnamed")
        list_id = lst.get("id")
        
        if not list_id:
            logger.warning(f"Skipping shopping list without ID: {list_name}")
            failed += 1
            continue
        
        if delete_shopping_list(client, list_id, list_name, dry_run):
            successful += 1
        else:
            failed += 1
    
    return successful, failed


def cleanup_meal_plans(client: MealieClient, dry_run: bool = False) -> Tuple[int, int]:
    """
    Delete all meal plan entries.
    
    Args:
        client: MealieClient instance
        dry_run: If True, only simulate deletion
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    print("\n" + "="*60)
    print("🧹 CLEANING UP MEAL PLANS")
    print("="*60)
    
    plans = fetch_all_meal_plans(client)
    
    if not plans:
        print("No meal plan entries found.")
        return 0, 0
    
    # Group by date and meal type for better reporting
    entries_by_date = {}
    for plan in plans:
        date = plan.get("date", "Unknown date")
        entry_type = plan.get("entryType", "Unknown type")
        recipe = plan.get("recipe", {})
        recipe_name = recipe.get("name", "Unknown recipe") if recipe else "No recipe"
        
        if date not in entries_by_date:
            entries_by_date[date] = []
        entries_by_date[date].append(f"{entry_type}: {recipe_name}")
    
    print(f"\nFound {len(plans)} meal plan entry/entries across {len(entries_by_date)} date(s):")
    
    # Show summary by date (limit output for readability)
    dates_shown = 0
    for date in sorted(entries_by_date.keys()):
        if dates_shown < 10:  # Show first 10 dates
            entries = entries_by_date[date]
            print(f"  {date}: {len(entries)} meal(s)")
            dates_shown += 1
        elif dates_shown == 10:
            remaining = len(entries_by_date) - 10
            if remaining > 0:
                print(f"  ... and {remaining} more date(s)")
            break
    
    if not dry_run:
        print()
    
    successful = 0
    failed = 0
    batch_size = 10
    
    for i, plan in enumerate(plans):
        entry_id = plan.get("id")
        date = plan.get("date", "Unknown")
        entry_type = plan.get("entryType", "Unknown")
        recipe = plan.get("recipe", {})
        recipe_name = recipe.get("name", "Unknown") if recipe else "No recipe"
        
        entry_info = f"{date} {entry_type} - {recipe_name}"
        
        if not entry_id:
            logger.warning(f"Skipping meal plan entry without ID: {entry_info}")
            failed += 1
            continue
        
        if delete_meal_plan_entry(client, entry_id, entry_info, dry_run):
            successful += 1
            # Show progress every batch_size items
            if not dry_run and (successful % batch_size == 0):
                print(f"  Progress: {successful}/{len(plans)} entries deleted...")
        else:
            failed += 1
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Clean up meal plans and shopping lists from Mealie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python utils/cleanup_meal_data.py               # Interactive confirmation
    python utils/cleanup_meal_data.py --confirm     # Skip confirmation
    python utils/cleanup_meal_data.py --dry-run     # Preview without deletion
        """
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🗑️  MEALIE CLEANUP TOOL")
    print("="*60)
    print("\nThis script will DELETE all meal plans and shopping lists.")
    print("Your recipes, tags, and other data will NOT be affected.")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE: No actual deletions will occur.")
    
    if not args.confirm and not args.dry_run:
        print("\n⚠️  WARNING: This operation cannot be undone!")
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("\n❌ Cleanup cancelled.")
            return 1
    
    # Initialize MealieClient
    client = MealieClient()
    try:
        # Clean up shopping lists
        shopping_success, shopping_failed = cleanup_shopping_lists(client, args.dry_run)
        
        # Clean up meal plans
        meal_success, meal_failed = cleanup_meal_plans(client, args.dry_run)
    
        # Summary
        print("\n" + "="*60)
        print("📊 CLEANUP SUMMARY")
        print("="*60)
        
        if args.dry_run:
            print("\n🔍 DRY RUN RESULTS:")
            print(f"  Shopping Lists: {shopping_success} would be deleted")
            print(f"  Meal Plans: {meal_success} entries would be deleted")
        else:
            print(f"\n✅ Shopping Lists Deleted: {shopping_success}")
            if shopping_failed > 0:
                print(f"❌ Shopping Lists Failed: {shopping_failed}")
            
            print(f"\n✅ Meal Plan Entries Deleted: {meal_success}")
            if meal_failed > 0:
                print(f"❌ Meal Plan Entries Failed: {meal_failed}")
            
            total_success = shopping_success + meal_success
            total_failed = shopping_failed + meal_failed
            
            print(f"\n📈 TOTAL: {total_success} items deleted, {total_failed} failed")
            
            if total_failed == 0:
                print("\n🎉 Cleanup completed successfully!")
            else:
                print("\n⚠️  Cleanup completed with some failures. Check logs above.")
        
        print("="*60 + "\n")
        
        return 0 if (shopping_failed + meal_failed) == 0 else 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
