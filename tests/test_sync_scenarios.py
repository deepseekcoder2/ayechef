#!/usr/bin/env python3
"""
Test script for incremental sync scenarios.
Verifies timestamp-based change detection against real Mealie data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
from config import MEALIE_URL, MEALIE_TOKEN, get_mealie_headers, DATA_DIR
from mealie_client import MealieClient


def fetch_mealie_recipes_with_timestamps() -> List[Dict]:
    """Fetch all recipes from Mealie with their timestamps."""
    print("📡 Fetching recipes from Mealie API...")
    client = MealieClient()
    try:
        recipes = client.get_all_recipes()
        print(f"   ✅ Fetched {len(recipes)} recipes from Mealie")
        return recipes
    finally:
        client.close()


def get_local_db_recipes() -> Dict[str, Dict]:
    """Get all recipes from local DB with their timestamps."""
    db_path = str(DATA_DIR / "recipe_index.db")
    
    if not os.path.exists(db_path):
        print("   ⚠️  Local DB does not exist yet")
        return {}
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT id, name, slug, updated_at 
            FROM recipes
        """)
        recipes = {}
        for row in cursor.fetchall():
            recipes[row[0]] = {
                "id": row[0],
                "name": row[1],
                "slug": row[2],
                "local_updated_at": row[3]
            }
        print(f"   ✅ Found {len(recipes)} recipes in local DB")
        return recipes


def analyze_timestamp_availability(mealie_recipes: List[Dict]) -> None:
    """Check what timestamp fields are available in Mealie responses."""
    print("\n" + "="*60)
    print("SCENARIO 0: Verify timestamp availability in Mealie API")
    print("="*60)
    
    if not mealie_recipes:
        print("❌ No recipes to analyze")
        return
    
    # Sample first 5 recipes
    sample_size = min(5, len(mealie_recipes))
    print(f"\nSampling {sample_size} recipes:")
    
    fields_found = {
        "updatedAt": 0,
        "dateUpdated": 0,
        "createdAt": 0,
        "dateAdded": 0
    }
    
    for i, recipe in enumerate(mealie_recipes[:sample_size]):
        print(f"\n  Recipe {i+1}: {recipe.get('name', 'Unknown')[:40]}")
        print(f"    id: {recipe.get('id', 'N/A')}")
        
        for field in fields_found.keys():
            value = recipe.get(field)
            if value:
                fields_found[field] += 1
                print(f"    {field}: {value}")
            else:
                print(f"    {field}: (not present)")
    
    print(f"\n  Summary across {sample_size} samples:")
    for field, count in fields_found.items():
        status = "✅" if count == sample_size else ("⚠️" if count > 0 else "❌")
        print(f"    {status} {field}: present in {count}/{sample_size} recipes")
    
    # Determine best field to use
    if fields_found["updatedAt"] > 0:
        print("\n  → Will use 'updatedAt' as primary timestamp field")
    elif fields_found["dateUpdated"] > 0:
        print("\n  → Will use 'dateUpdated' as fallback timestamp field")
    else:
        print("\n  ❌ No timestamp fields available! Cannot do incremental sync.")


def test_scenario_additions(mealie_recipes: List[Dict], local_recipes: Dict[str, Dict]) -> Tuple[Set[str], List[Dict]]:
    """Test: Find recipes in Mealie but not in local DB (new additions)."""
    print("\n" + "="*60)
    print("SCENARIO 3: Detect NEW recipes (in Mealie, not in local)")
    print("="*60)
    
    mealie_ids = {r.get("id") for r in mealie_recipes if r.get("id")}
    local_ids = set(local_recipes.keys())
    
    new_recipe_ids = mealie_ids - local_ids
    
    print(f"\n  Mealie has: {len(mealie_ids)} recipes")
    print(f"  Local has:  {len(local_ids)} recipes")
    print(f"  New (to add): {len(new_recipe_ids)} recipes")
    
    if new_recipe_ids and len(new_recipe_ids) <= 10:
        print("\n  New recipes:")
        mealie_by_id = {r.get("id"): r for r in mealie_recipes}
        for rid in list(new_recipe_ids)[:10]:
            recipe = mealie_by_id.get(rid, {})
            print(f"    - {recipe.get('name', 'Unknown')[:50]}")
    elif new_recipe_ids:
        print(f"\n  (Too many to list - showing first 5)")
        mealie_by_id = {r.get("id"): r for r in mealie_recipes}
        for rid in list(new_recipe_ids)[:5]:
            recipe = mealie_by_id.get(rid, {})
            print(f"    - {recipe.get('name', 'Unknown')[:50]}")
    
    # Return for use in other tests
    new_recipes = [r for r in mealie_recipes if r.get("id") in new_recipe_ids]
    return new_recipe_ids, new_recipes


def test_scenario_deletions(mealie_recipes: List[Dict], local_recipes: Dict[str, Dict]) -> Set[str]:
    """Test: Find recipes in local DB but not in Mealie (deleted)."""
    print("\n" + "="*60)
    print("SCENARIO 4: Detect DELETED recipes (in local, not in Mealie)")
    print("="*60)
    
    mealie_ids = {r.get("id") for r in mealie_recipes if r.get("id")}
    local_ids = set(local_recipes.keys())
    
    deleted_recipe_ids = local_ids - mealie_ids
    
    print(f"\n  Mealie has: {len(mealie_ids)} recipes")
    print(f"  Local has:  {len(local_ids)} recipes")
    print(f"  Deleted (to remove): {len(deleted_recipe_ids)} recipes")
    
    if deleted_recipe_ids and len(deleted_recipe_ids) <= 10:
        print("\n  Deleted recipes:")
        for rid in list(deleted_recipe_ids)[:10]:
            recipe = local_recipes.get(rid, {})
            print(f"    - {recipe.get('name', 'Unknown')[:50]} (id: {rid[:8]}...)")
    elif deleted_recipe_ids:
        print(f"\n  (Too many to list - showing first 5)")
        for rid in list(deleted_recipe_ids)[:5]:
            recipe = local_recipes.get(rid, {})
            print(f"    - {recipe.get('name', 'Unknown')[:50]} (id: {rid[:8]}...)")
    
    return deleted_recipe_ids


def test_scenario_modifications(mealie_recipes: List[Dict], local_recipes: Dict[str, Dict]) -> List[Dict]:
    """
    Test: Find recipes that exist in both but have been modified in Mealie.
    This requires comparing timestamps.
    """
    print("\n" + "="*60)
    print("SCENARIO 5-7: Detect MODIFIED recipes (timestamp comparison)")
    print("="*60)
    
    # Check if local DB has mealie_updated_at column
    db_path = str(DATA_DIR / "recipe_index.db")
    has_mealie_timestamp = False
    
    if os.path.exists(db_path):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(recipes)")
            columns = [row[1] for row in cursor.fetchall()]
            has_mealie_timestamp = "mealie_updated_at" in columns
    
    if not has_mealie_timestamp:
        print("\n  ⚠️  Local DB does not have 'mealie_updated_at' column yet")
        print("  → Cannot detect modifications without storing Mealie's timestamp")
        print("  → This is what we need to ADD in the fix")
        print("\n  Simulating what we WOULD detect if we had the column...")
    
    # Build lookup
    mealie_by_id = {r.get("id"): r for r in mealie_recipes if r.get("id")}
    
    # Find recipes that exist in both
    common_ids = set(mealie_by_id.keys()) & set(local_recipes.keys())
    print(f"\n  Recipes in both: {len(common_ids)}")
    
    # For demonstration, show some timestamp comparisons
    modified_recipes = []
    sample_count = 0
    
    print("\n  Sample timestamp comparisons (first 5 common recipes):")
    for recipe_id in list(common_ids)[:5]:
        mealie_recipe = mealie_by_id[recipe_id]
        local_recipe = local_recipes[recipe_id]
        
        mealie_ts = mealie_recipe.get('updatedAt') or mealie_recipe.get('dateUpdated') or 'N/A'
        local_ts = local_recipe.get('local_updated_at', 'N/A')
        
        print(f"\n    {mealie_recipe.get('name', 'Unknown')[:40]}")
        print(f"      Mealie updatedAt: {mealie_ts}")
        print(f"      Local updated_at: {local_ts}")
        
        # Parse and compare if both are valid
        if mealie_ts != 'N/A' and local_ts != 'N/A':
            try:
                # Parse Mealie timestamp (ISO format)
                if 'T' in str(mealie_ts):
                    mealie_dt = datetime.fromisoformat(mealie_ts.replace('Z', '+00:00'))
                else:
                    mealie_dt = datetime.fromisoformat(mealie_ts)
                
                # Parse local timestamp
                local_dt = datetime.fromisoformat(local_ts.replace('Z', '+00:00') if 'Z' in str(local_ts) else local_ts)
                
                # Compare (note: we're comparing apples to oranges here - 
                # local_updated_at is when WE indexed, not when Mealie was updated)
                print(f"      ⚠️  Note: These are different timestamps (when WE indexed vs when Mealie updated)")
            except Exception as e:
                print(f"      ⚠️  Could not parse timestamps: {e}")
        
        sample_count += 1
    
    print("\n  Key insight:")
    print("  → Current local 'updated_at' = when WE indexed the recipe")
    print("  → Mealie 'updatedAt' = when recipe was actually modified")
    print("  → We need to store Mealie's timestamp to compare properly")
    
    return modified_recipes


def test_scenario_no_changes(mealie_recipes: List[Dict], local_recipes: Dict[str, Dict]) -> None:
    """Test: Verify we can detect when nothing has changed."""
    print("\n" + "="*60)
    print("SCENARIO 2: Detect NO CHANGES (everything in sync)")
    print("="*60)
    
    mealie_ids = {r.get("id") for r in mealie_recipes if r.get("id")}
    local_ids = set(local_recipes.keys())
    
    new_count = len(mealie_ids - local_ids)
    deleted_count = len(local_ids - mealie_ids)
    
    if new_count == 0 and deleted_count == 0:
        print("\n  ✅ Recipe counts match!")
        print("  → With timestamp tracking, we could verify no modifications either")
    else:
        print(f"\n  ⚠️  Counts don't match: +{new_count} new, -{deleted_count} deleted")


def summarize_sync_requirements() -> None:
    """Summarize what the incremental sync needs to do."""
    print("\n" + "="*60)
    print("SUMMARY: Incremental Sync Requirements")
    print("="*60)
    
    print("""
  1. ADD 'mealie_updated_at' column to local DB
     → Stores Mealie's timestamp when we index each recipe
  
  2. On sync, compare:
     - mealie.updatedAt vs local.mealie_updated_at
     - If Mealie's is newer → re-index that recipe
  
  3. Detection logic:
     - NEW: in Mealie, not in local → ADD
     - DELETED: in local, not in Mealie → REMOVE  
     - MODIFIED: Mealie timestamp > local stored timestamp → RE-INDEX
     - UNCHANGED: timestamps match → SKIP
  
  4. First run after migration:
     - All recipes have mealie_updated_at = NULL
     - Need to do one full sync to populate timestamps
     - After that, incremental syncs are fast
    """)


def main():
    """Run all scenario tests."""
    print("="*60)
    print("INCREMENTAL SYNC SCENARIO TESTS")
    print("="*60)
    print(f"Mealie URL: {MEALIE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Fetch data
    mealie_recipes = fetch_mealie_recipes_with_timestamps()
    local_recipes = get_local_db_recipes()
    
    if not mealie_recipes:
        print("\n❌ Cannot proceed without Mealie data")
        return
    
    # Run all scenario tests
    analyze_timestamp_availability(mealie_recipes)
    test_scenario_no_changes(mealie_recipes, local_recipes)
    new_ids, new_recipes = test_scenario_additions(mealie_recipes, local_recipes)
    deleted_ids = test_scenario_deletions(mealie_recipes, local_recipes)
    test_scenario_modifications(mealie_recipes, local_recipes)
    
    # Summary
    summarize_sync_requirements()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # Final stats
    total_mealie = len(mealie_recipes)
    total_local = len(local_recipes)
    to_add = len(new_ids)
    to_remove = len(deleted_ids)
    
    print(f"""
  Mealie:     {total_mealie} recipes
  Local DB:   {total_local} recipes
  To add:     {to_add} recipes
  To remove:  {to_remove} recipes
  To check:   {total_local - to_remove} recipes (need timestamp comparison)
    """)


if __name__ == "__main__":
    main()
