#!/usr/bin/env python3
"""
Quick verification that PREP/BUY items can be added to shopping list.

Run: python tests/verify_shopping_notes.py

This will:
1. Create a temporary test shopping list
2. Add a few test note items (simulating PREP/BUY)
3. Verify they appear in the list
4. Delete the test list
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mealie_client import MealieClient
from utils.shopping_list_generator import (
    add_item_to_shopping_list,
    add_note_items_to_shopping_list,
    fetch_shopping_list
)

def create_test_list() -> str:
    """Create a temporary test shopping list."""
    client = MealieClient()
    try:
        result = client.create_shopping_list("[TEST] Verify PREP/BUY Items")
        return result.get("id")
    finally:
        client.close()


def delete_test_list(list_id: str) -> bool:
    """Delete the test shopping list."""
    client = MealieClient()
    try:
        return client.delete_shopping_list(list_id)
    except Exception as e:
        print(f"Warning: Failed to delete test list: {e}")
        return False
    finally:
        client.close()


def main():
    print("=" * 60)
    print("VERIFYING PREP/BUY ITEMS -> SHOPPING LIST")
    print("=" * 60)
    
    # Step 1: Create test list
    print("\n1. Creating test shopping list...")
    list_id = create_test_list()
    print(f"   Created list: {list_id}")
    
    try:
        # Step 2: Simulate note items (what we get from meal plan)
        test_items = [
            {"title": "crusty baguette", "text": "from bakery"},
            {"title": "pappardelle", "text": "boil per package directions"},
            {"title": "apple slices", "text": "wash and slice"},
        ]
        
        print(f"\n2. Adding {len(test_items)} test items...")
        for item in test_items:
            print(f"   - {item['title']}")
        
        added_count = add_note_items_to_shopping_list(list_id, test_items)
        print(f"   Added: {added_count}/{len(test_items)} items")
        
        # Step 3: Verify items appear in list
        print("\n3. Fetching shopping list to verify...")
        shopping_list = fetch_shopping_list(list_id)
        list_items = shopping_list.get("listItems", [])
        
        print(f"   Found {len(list_items)} items in list:")
        for item in list_items:
            display = item.get("display", "")
            note = item.get("note", "")
            print(f"   - display='{display}' note='{note}'")
        
        # Step 4: Verify all items were added
        print("\n4. Verification:")
        expected_titles = {item["title"].lower() for item in test_items}
        found_titles = set()
        
        for item in list_items:
            # Check both display and note fields
            display = (item.get("display") or "").lower()
            note = (item.get("note") or "").lower()
            if display in expected_titles:
                found_titles.add(display)
            elif note in expected_titles:
                found_titles.add(note)
        
        if found_titles == expected_titles:
            print("   ✅ SUCCESS: All PREP/BUY items were added to shopping list")
            success = True
        else:
            missing = expected_titles - found_titles
            print(f"   ❌ FAILED: Missing items: {missing}")
            success = False
        
    finally:
        # Step 5: Cleanup
        print("\n5. Cleaning up test list...")
        if delete_test_list(list_id):
            print("   ✅ Test list deleted")
        else:
            print(f"   ⚠️  Manual cleanup needed: {list_id}")
    
    print("\n" + "=" * 60)
    if success:
        print("RESULT: PREP/BUY -> Shopping List integration WORKS")
    else:
        print("RESULT: PREP/BUY -> Shopping List integration BROKEN")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
