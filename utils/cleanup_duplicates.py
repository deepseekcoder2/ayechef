#!/usr/bin/env python3
"""
Cleanup Invalid and Duplicate Recipes
=====================================

This script removes problematic recipes from Mealie:
- Duplicates: recipes with "(1)", "(2)", etc. in their names
- Invalid: recipes with no ingredients or no instructions

Usage:
    python utils/cleanup_duplicates.py --duplicates        # Remove duplicates only
    python utils/cleanup_duplicates.py --invalid           # Remove invalid recipes only
    python utils/cleanup_duplicates.py --all               # Remove both
    python utils/cleanup_duplicates.py --all --confirm     # Skip confirmation (web UI)
    python utils/cleanup_duplicates.py --all --dry-run     # Preview only
"""

import sys
import os
import re
import concurrent.futures

# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import List, Dict, Tuple, Optional
from mealie_client import MealieClient
from tools.progress_ui import ui
from tools.logging_utils import get_logger
from utils.recipe_validation import is_valid_recipe_content

logger = get_logger(__name__)

# Pattern to match duplicate indicators like "(1)", "(2)", "(3)", etc.
DUPLICATE_PATTERN = re.compile(r'\(\d+\)$')


def fetch_all_recipes(client: MealieClient) -> List[Dict]:
    """
    Fetch all recipes from Mealie.
    
    Args:
        client: MealieClient instance
    
    Returns:
        List of recipe objects (slug, name, id)
    """
    try:
        print("Fetching all recipes from Mealie...", flush=True)
        all_recipes = client.get_all_recipes()
        print(f"✅ Found {len(all_recipes)} total recipes in Mealie", flush=True)
        return all_recipes
        
    except Exception as e:
        print(f"Error fetching recipes: {e}", flush=True)
        logger.error(f"Error fetching recipes: {e}")
        return []


def find_duplicates(recipes: List[Dict]) -> List[Dict]:
    """
    Find recipes that are duplicates (have "(1)", "(2)", etc. in their name).
    
    Args:
        recipes: List of recipe objects
    
    Returns:
        List of duplicate recipe objects
    """
    duplicates = []
    
    for recipe in recipes:
        name = recipe.get("name", "")
        # Check if name ends with a duplicate indicator like "(1)"
        if DUPLICATE_PATTERN.search(name.strip()):
            duplicates.append(recipe)
    
    return duplicates


def fetch_recipe_full(client: MealieClient, slug: str) -> Optional[Dict]:
    """
    Fetch full recipe data including ingredients and instructions.
    
    Args:
        client: MealieClient instance
        slug: Recipe slug
    
    Returns:
        Full recipe data or None on error
    """
    try:
        return client.get_recipe(slug)
    except Exception as e:
        logger.debug(f"Error fetching recipe {slug}: {e}")
        return None


def get_invalid_candidates_from_local_db() -> List[Tuple[str, str]]:
    """
    Query local recipe database for recipes that might be invalid.
    
    Returns:
        List of (slug, name) tuples for candidate invalid recipes
    """
    import sqlite3
    from pathlib import Path
    from config import DATA_DIR
    
    db_path = DATA_DIR / "recipe_index.db"
    if not db_path.exists():
        print("⚠️  Local recipe database not found - will scan all recipes from API", flush=True)
        return []
    
    candidates = []
    try:
        with sqlite3.connect(db_path) as conn:
            # Find recipes with empty or null ingredients
            cursor = conn.execute("""
                SELECT slug, name FROM recipes 
                WHERE ingredients IS NULL 
                   OR ingredients = '' 
                   OR ingredients = '[]'
            """)
            candidates = cursor.fetchall()
        
        print(f"📊 Local DB: Found {len(candidates)} recipes with no ingredients in local index", flush=True)
        return candidates
        
    except Exception as e:
        print(f"⚠️  Error querying local DB: {e}", flush=True)
        return []


def find_invalid_recipes(client: MealieClient, recipes: List[Dict], max_workers: int = 10) -> List[Tuple[Dict, str]]:
    """
    Find recipes that have no ingredients or no instructions.
    
    Uses hybrid approach:
    1. Fast query of local DB for candidates
    2. Verify each candidate against Mealie API (source of truth)
    
    Args:
        client: MealieClient instance
        recipes: List of recipe objects (from list endpoint) - used as fallback
        max_workers: Number of parallel workers for fetching (unused, kept for compatibility)
    
    Returns:
        List of (recipe_dict, reason) tuples for invalid recipes
    """
    invalid = []
    
    # Step 1: Get candidates from local DB (fast)
    candidates = get_invalid_candidates_from_local_db()
    
    if not candidates:
        print("⚠️  No candidates from local DB. Falling back to full API scan...", flush=True)
        print("  (This will be slow - consider running maintenance to sync local DB)", flush=True)
        # Fall back to checking all recipes if local DB unavailable
        candidates = [(r.get("slug"), r.get("name")) for r in recipes if r.get("slug")]
    
    total = len(candidates)
    if total == 0:
        print("✅ No invalid recipe candidates found", flush=True)
        return []
    
    print(f"🔍 Verifying {total} candidates against Mealie API...", flush=True)
    
    verified = 0
    for slug, name in candidates:
        verified += 1
        
        # Show progress
        if verified % 10 == 0 or verified == total:
            print(f"  Verified {verified}/{total} candidates - {len(invalid)} confirmed invalid", flush=True)
        
        # Verify against Mealie API (source of truth)
        full_data = fetch_recipe_full(client, slug)
        
        if full_data is None:
            # Recipe doesn't exist in Mealie anymore - skip (might be already deleted)
            print(f"  ⚪ SKIPPED: {name} (not found in Mealie - may be already deleted)", flush=True)
            continue
        
        is_valid, reason = is_valid_recipe_content(full_data)
        if not is_valid:
            recipe_dict = {"slug": slug, "name": full_data.get("name", name)}
            invalid.append((recipe_dict, reason))
            print(f"  ❌ CONFIRMED INVALID: {name} - {reason}", flush=True)
        else:
            print(f"  ✅ OK: {name} (has content in Mealie, local DB out of sync)", flush=True)
    
    print(f"\n✅ Verification complete: {len(invalid)} invalid recipes confirmed out of {total} candidates", flush=True)
    return invalid


def bulk_delete_recipe(slug: str) -> bool:
    """
    Delete a recipe using the bulk delete endpoint.
    
    The regular DELETE endpoint fails with 500 errors on some corrupted recipes,
    but the bulk delete endpoint works reliably.
    
    Args:
        slug: Recipe slug
    
    Returns:
        True if successful, False otherwise
    """
    import requests
    from config import MEALIE_URL, MEALIE_TOKEN
    
    headers = {
        'Authorization': f'Bearer {MEALIE_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    payload = {'recipes': [slug]}
    try:
        r = requests.post(
            f'{MEALIE_URL}/api/recipes/bulk-actions/delete',
            headers=headers,
            json=payload,
            timeout=30
        )
        return r.status_code == 200
    except Exception:
        return False


def delete_recipe(client: MealieClient, slug: str, name: str, dry_run: bool = False) -> bool:
    """
    Delete a recipe by slug using bulk delete endpoint.
    
    Uses bulk delete because the regular DELETE endpoint fails with 500 errors
    on recipes with corrupted slug routing.
    
    Args:
        client: MealieClient instance (unused, kept for compatibility)
        slug: Recipe slug
        name: Recipe name (for logging)
        dry_run: If True, only simulate deletion
    
    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        print(f"  [DRY RUN] Would delete: {name}", flush=True)
        return True
    
    try:
        success = bulk_delete_recipe(slug)
        if success:
            print(f"  ✅ Deleted: {name}", flush=True)
            return True
        else:
            print(f"  ❌ Failed to delete {name}", flush=True)
            return False
            
    except Exception as e:
        print(f"  ❌ Error deleting {name}: {e}", flush=True)
        return False


def cleanup_rag_index(deleted_slugs: List[str]) -> int:
    """
    Remove deleted recipes from the local RAG index.
    
    Args:
        deleted_slugs: List of slugs that were deleted
    
    Returns:
        Number of recipes removed from index
    """
    if not deleted_slugs:
        return 0
    
    try:
        import sqlite3
        from pathlib import Path
        from config import DATA_DIR
        
        db_path = DATA_DIR / "recipe_index.db"
        if not db_path.exists():
            logger.warning("Recipe index database not found, skipping index cleanup")
            return 0
        
        removed = 0
        with sqlite3.connect(db_path) as conn:
            for slug in deleted_slugs:
                try:
                    # First get the id for this slug (needed for FTS table)
                    cursor = conn.execute("SELECT id FROM recipes WHERE slug = ?", (slug,))
                    row = cursor.fetchone()
                    
                    if row:
                        recipe_id = row[0]
                        # Delete from main table
                        conn.execute("DELETE FROM recipes WHERE slug = ?", (slug,))
                        # Delete from FTS index
                        conn.execute("DELETE FROM recipes_fts WHERE id = ?", (recipe_id,))
                        removed += 1
                except Exception:
                    pass  # Recipe might not be in index
            conn.commit()
        
        return removed
        
    except Exception as e:
        logger.warning(f"Error cleaning RAG index: {e}")
        return 0


def cleanup_duplicates(client: MealieClient, dry_run: bool = False) -> Tuple[int, int, List[str]]:
    """
    Find and delete all duplicate recipes.
    
    Args:
        client: MealieClient instance
        dry_run: If True, only simulate deletion
    
    Returns:
        Tuple of (successful_count, failed_count, deleted_slugs)
    """
    # Fetch all recipes
    recipes = fetch_all_recipes(client)
    if not recipes:
        print("No recipes found in Mealie.", flush=True)
        return 0, 0, []
    
    # Find duplicates
    print("🔍 Scanning for duplicate recipes...", flush=True)
    duplicates = find_duplicates(recipes)
    
    if not duplicates:
        print("✅ No duplicate recipes found!", flush=True)
        return 0, 0, []
    
    # List duplicates found
    print(f"Found {len(duplicates)} duplicate recipe(s):", flush=True)
    print("-" * 60, flush=True)
    for recipe in duplicates:
        name = recipe.get("name", "Unknown")
        slug = recipe.get("slug", "unknown")
        print(f"  • {name}", flush=True)
        print(f"    slug: {slug}", flush=True)
    print("-" * 60, flush=True)
    
    # Start deletion operation
    if dry_run:
        print(f"\n🔍 DRY RUN: Would delete {len(duplicates)} recipes", flush=True)
    else:
        print(f"\n🗑️  Deleting {len(duplicates)} duplicate recipes...", flush=True)
    
    successful = 0
    failed = 0
    deleted_slugs = []
    
    for i, recipe in enumerate(duplicates):
        name = recipe.get("name", "Unknown")
        slug = recipe.get("slug")
        
        if not slug:
            print(f"Skipping recipe without slug: {name}", flush=True)
            failed += 1
            continue
        
        if delete_recipe(client, slug, name, dry_run):
            successful += 1
            if not dry_run:
                deleted_slugs.append(slug)
        else:
            failed += 1
        
        # Show progress every 10 deletions
        if (successful + failed) % 10 == 0:
            print(f"  Progress: {successful + failed}/{len(duplicates)}", flush=True)
    
    # Complete
    print(f"\n✅ Duplicate cleanup complete: {successful} deleted, {failed} failed", flush=True)
    
    return successful, failed, deleted_slugs


def cleanup_invalid(client: MealieClient, dry_run: bool = False) -> Tuple[int, int, List[str]]:
    """
    Find and delete all invalid recipes (no ingredients or no instructions).
    
    Args:
        client: MealieClient instance
        dry_run: If True, only simulate deletion
    
    Returns:
        Tuple of (successful_count, failed_count, deleted_slugs)
    """
    # Fetch all recipes
    recipes = fetch_all_recipes(client)
    if not recipes:
        print("No recipes found in Mealie.", flush=True)
        return 0, 0, []
    
    # Find invalid recipes (this fetches full data for each)
    invalid_recipes = find_invalid_recipes(client, recipes)
    
    if not invalid_recipes:
        print("✅ No invalid recipes found! All recipes have ingredients and instructions.", flush=True)
        return 0, 0, []
    
    # Summary of invalid recipes (already printed during scan)
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY: {len(invalid_recipes)} invalid recipes to delete", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Start deletion operation
    if dry_run:
        print(f"\n🔍 DRY RUN: Would delete {len(invalid_recipes)} recipes", flush=True)
    else:
        print(f"\n🗑️  Deleting {len(invalid_recipes)} invalid recipes...", flush=True)
    
    successful = 0
    failed = 0
    deleted_slugs = []
    
    for recipe, reason in invalid_recipes:
        name = recipe.get("name", "Unknown")
        slug = recipe.get("slug")
        
        if not slug:
            print(f"Skipping recipe without slug: {name}", flush=True)
            failed += 1
            continue
        
        if delete_recipe(client, slug, name, dry_run):
            successful += 1
            if not dry_run:
                deleted_slugs.append(slug)
        else:
            failed += 1
        
        # Show progress every 10 deletions
        if (successful + failed) % 10 == 0:
            print(f"  Progress: {successful + failed}/{len(invalid_recipes)}", flush=True)
    
    # Complete
    print(f"\n✅ Invalid recipe cleanup complete: {successful} deleted, {failed} failed", flush=True)
    
    return successful, failed, deleted_slugs


def main():
    parser = argparse.ArgumentParser(
        description="Clean up invalid and duplicate recipes from Mealie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python utils/cleanup_duplicates.py --duplicates         # Remove duplicates only
    python utils/cleanup_duplicates.py --invalid            # Remove invalid recipes only
    python utils/cleanup_duplicates.py --all                # Remove both types
    python utils/cleanup_duplicates.py --all --confirm      # Skip confirmation (web UI)
    python utils/cleanup_duplicates.py --all --dry-run      # Preview without deletion
        """
    )
    parser.add_argument(
        "--duplicates",
        action="store_true",
        help="Remove duplicate recipes (names ending with (1), (2), etc.)"
    )
    parser.add_argument(
        "--invalid",
        action="store_true",
        help="Remove invalid recipes (no ingredients or no instructions)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove both duplicates and invalid recipes"
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
    
    # Default to --duplicates if no mode specified (backward compatibility)
    if not args.duplicates and not args.invalid and not args.all:
        args.duplicates = True
    
    # --all enables both
    if args.all:
        args.duplicates = True
        args.invalid = True
    
    print("\n" + "="*60)
    print("🧹 RECIPE CLEANUP")
    print("="*60)
    
    modes = []
    if args.duplicates:
        modes.append("duplicates (names with (1), (2), etc.)")
    if args.invalid:
        modes.append("invalid (no ingredients or no instructions)")
    
    print(f"\nCleanup mode(s): {', '.join(modes)}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE: No actual deletions will occur.", flush=True)
    
    if not args.confirm and not args.dry_run:
        print("\n⚠️  WARNING: This operation cannot be undone!", flush=True)
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Cleanup cancelled.", flush=True)
            return 1
    
    # Initialize MealieClient
    client = MealieClient()
    try:
        total_successful = 0
        total_failed = 0
        all_deleted_slugs = []
        
        # Run duplicate cleanup if requested
        if args.duplicates:
            print("\n" + "-"*60)
            print("PHASE 1: Duplicate Recipes")
            print("-"*60)
            successful, failed, deleted_slugs = cleanup_duplicates(client, args.dry_run)
            total_successful += successful
            total_failed += failed
            all_deleted_slugs.extend(deleted_slugs)
        
        # Run invalid cleanup if requested
        if args.invalid:
            print("\n" + "-"*60)
            print("PHASE 2: Invalid Recipes")
            print("-"*60)
            successful, failed, deleted_slugs = cleanup_invalid(client, args.dry_run)
            total_successful += successful
            total_failed += failed
            all_deleted_slugs.extend(deleted_slugs)
    
        # Clean up RAG index if we deleted recipes
        rag_removed = 0
        if all_deleted_slugs and not args.dry_run:
            print("📚 Cleaning up search index...", flush=True)
            rag_removed = cleanup_rag_index(all_deleted_slugs)
            if rag_removed > 0:
                print(f"  ✅ Removed {rag_removed} recipe(s) from search index", flush=True)
        
        # Summary
        print("\n" + "="*60)
        print("📊 CLEANUP SUMMARY")
        print("="*60)
        
        if args.dry_run:
            print("\n🔍 DRY RUN RESULTS:")
            print(f"  Recipes found: {total_successful}")
            print(f"  Would be deleted: {total_successful}")
        else:
            print(f"\n✅ Recipes Deleted: {total_successful}")
            if total_failed > 0:
                print(f"❌ Recipes Failed: {total_failed}")
            if rag_removed > 0:
                print(f"📚 Removed from search index: {rag_removed}")
            
            if total_failed == 0 and total_successful > 0:
                print("\n🎉 Cleanup completed successfully!")
            elif total_successful == 0:
                print("\n✅ No problematic recipes found - nothing to clean up!")
            else:
                print("\n⚠️  Cleanup completed with some failures. Check output above.")
        
        print("="*60 + "\n")
        
        return 0 if total_failed == 0 else 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
