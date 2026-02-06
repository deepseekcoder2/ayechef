#!/usr/bin/env python3
"""
REAL END-TO-END TEST of incremental sync.
Actually creates, modifies, and deletes recipes in Mealie.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import sqlite3
from datetime import datetime
from mealie_client import MealieClient


def get_local_recipe(recipe_id: str) -> dict:
    """Get recipe from local DB."""
    from config import DATA_DIR
    db_path = str(DATA_DIR / "recipe_index.db")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT id, name, mealie_updated_at FROM recipes WHERE id = ?",
            (recipe_id,)
        )
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "mealie_updated_at": row[2]}
        return None


def create_test_recipe() -> dict:
    """Create a test recipe in Mealie."""
    print("\n" + "="*60)
    print("STEP 1: CREATE test recipe in Mealie")
    print("="*60)
    
    recipe_data = {
        "name": f"TEST_SYNC_RECIPE_{datetime.now().strftime('%H%M%S')}",
        "description": "Test recipe for sync verification",
        "recipeIngredient": [
            {"note": "1 cup test ingredient"}
        ],
        "recipeInstructions": [
            {"text": "Step 1: Test the sync"}
        ]
    }
    
    print(f"  Creating recipe: {recipe_data['name']}")
    
    client = MealieClient()
    try:
        # Create recipe
        result = client.create_recipe(recipe_data)
        
        # Handle response - Mealie may return slug string or dict
        if isinstance(result, str):
            slug = result
        elif isinstance(result, dict):
            slug = result.get('slug', '')
        else:
            print(f"  ❌ Failed to create: unexpected response type")
            return None
        
        if not slug:
            print(f"  ❌ Failed to create: no slug returned")
            return None
        
        print(f"  ✅ Created recipe!")
        print(f"     Slug: {slug}")
        
        # Fetch full recipe to get ID and timestamps
        full_recipe = client.get_recipe(slug)
        print(f"     ID: {full_recipe.get('id')}")
        print(f"     updatedAt: {full_recipe.get('updatedAt')}")
        print(f"     dateUpdated: {full_recipe.get('dateUpdated')}")
        
        return full_recipe
    finally:
        client.close()


def modify_recipe(slug: str) -> dict:
    """Modify an existing recipe in Mealie."""
    print("\n" + "="*60)
    print("STEP 3: MODIFY test recipe in Mealie")
    print("="*60)
    
    client = MealieClient()
    try:
        # First get current recipe
        recipe = client.get_recipe(slug)
        old_updated_at = recipe.get('updatedAt')
        print(f"  Before modification:")
        print(f"     updatedAt: {old_updated_at}")
        
        # Modify description
        update_data = {
            'description': f"Modified at {datetime.now().isoformat()}"
        }
        
        print(f"  Updating description...")
        
        # Update recipe
        updated_recipe = client.update_recipe(slug, update_data)
        new_updated_at = updated_recipe.get('updatedAt')
        
        print(f"  After modification:")
        print(f"     updatedAt: {new_updated_at}")
        print(f"     Changed: {old_updated_at != new_updated_at}")
        
        return updated_recipe
    finally:
        client.close()


def delete_recipe(slug: str) -> bool:
    """Delete a recipe from Mealie."""
    print("\n" + "="*60)
    print("STEP 5: DELETE test recipe from Mealie")
    print("="*60)
    
    print(f"  Deleting recipe: {slug}")
    
    client = MealieClient()
    try:
        success = client.delete_recipe(slug)
        if success:
            print(f"  ✅ Recipe deleted from Mealie")
        else:
            print(f"  ❌ Failed to delete recipe")
        return success
    finally:
        client.close()


def run_sync_and_check():
    """Run sync and return results."""
    from utils.recipe_maintenance import sync_local_db_with_mealie
    
    print("  Running incremental sync...")
    start = time.time()
    added, removed, modified, unchanged = sync_local_db_with_mealie()
    elapsed = time.time() - start
    
    print(f"  Sync completed in {elapsed:.1f}s")
    print(f"  Results: +{added} added, -{removed} removed, ~{modified} modified, ={unchanged} unchanged")
    
    return added, removed, modified, unchanged


def main():
    print("="*60)
    print("REAL END-TO-END SYNC TEST")
    print("="*60)
    from config import MEALIE_URL
    print(f"Mealie URL: {MEALIE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    
    # =====================================================
    # STEP 1: Create test recipe
    # =====================================================
    recipe = create_test_recipe()
    if not recipe:
        print("❌ Test failed - could not create recipe")
        return
    
    recipe_id = recipe.get('id')
    recipe_slug = recipe.get('slug')
    
    # Check local DB - should NOT have this recipe yet
    local = get_local_recipe(recipe_id)
    print(f"\n  Local DB check: {'Found' if local else 'Not found'}")
    
    # =====================================================
    # STEP 2: Run sync - should detect NEW recipe
    # =====================================================
    print("\n" + "="*60)
    print("STEP 2: Run sync - expect NEW recipe detected")
    print("="*60)
    
    added, removed, modified, unchanged = run_sync_and_check()
    
    if added >= 1:
        print(f"  ✅ PASS: Sync detected new recipe(s)")
    else:
        print(f"  ❌ FAIL: Sync did not detect new recipe")
    
    # Verify recipe is now in local DB
    local = get_local_recipe(recipe_id)
    if local:
        print(f"  ✅ Recipe now in local DB")
        print(f"     mealie_updated_at: {local.get('mealie_updated_at')}")
    else:
        print(f"  ❌ Recipe NOT in local DB after sync")
    
    # =====================================================
    # STEP 3: Modify recipe in Mealie
    # =====================================================
    time.sleep(2)  # Ensure timestamp changes
    modified_recipe = modify_recipe(recipe_slug)
    if not modified_recipe:
        print("❌ Test failed - could not modify recipe")
        # Cleanup
        delete_recipe(recipe_slug)
        return
    
    # =====================================================
    # STEP 4: Run sync - should detect MODIFIED recipe
    # =====================================================
    print("\n" + "="*60)
    print("STEP 4: Run sync - expect MODIFIED recipe detected")
    print("="*60)
    
    # Check timestamps BEFORE sync
    local_before = get_local_recipe(recipe_id)
    print(f"  Before sync:")
    print(f"     Local mealie_updated_at: {local_before.get('mealie_updated_at') if local_before else 'N/A'}")
    print(f"     Mealie updatedAt:        {modified_recipe.get('updatedAt')}")
    
    added, removed, modified_count, unchanged = run_sync_and_check()
    
    if modified_count >= 1:
        print(f"  ✅ PASS: Sync detected modified recipe(s)")
    else:
        print(f"  ❌ FAIL: Sync did not detect modified recipe")
        print(f"          This is the CRITICAL test - if this fails, incremental sync doesn't work")
    
    # Verify timestamp was updated
    local_after = get_local_recipe(recipe_id)
    if local_after:
        print(f"  After sync:")
        print(f"     Local mealie_updated_at: {local_after.get('mealie_updated_at')}")
        if local_after.get('mealie_updated_at') != local_before.get('mealie_updated_at'):
            print(f"  ✅ Timestamp was updated in local DB")
        else:
            print(f"  ❌ Timestamp was NOT updated")
    
    # =====================================================
    # STEP 5: Delete recipe from Mealie
    # =====================================================
    if not delete_recipe(recipe_slug):
        print("❌ Test failed - could not delete recipe")
        return
    
    # =====================================================
    # STEP 6: Run sync - should detect DELETED recipe
    # =====================================================
    print("\n" + "="*60)
    print("STEP 6: Run sync - expect DELETED recipe detected")
    print("="*60)
    
    added, removed, modified_count, unchanged = run_sync_and_check()
    
    if removed >= 1:
        print(f"  ✅ PASS: Sync detected deleted recipe(s)")
    else:
        print(f"  ❌ FAIL: Sync did not detect deleted recipe")
    
    # Verify recipe is removed from local DB
    local = get_local_recipe(recipe_id)
    if local is None:
        print(f"  ✅ Recipe removed from local DB")
    else:
        print(f"  ❌ Recipe still in local DB after sync")
    
    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
