#!/usr/bin/env python3
"""
Quick script to add foods to Mealie's database.

Usage:
    python add_food.py "ginger" "lemongrass" "fish sauce"
    python add_food.py "ingredient name"
"""
import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mealie_client import MealieClient

def add_foods(foods: list[str]):
    client = MealieClient()
    try:
        # Get existing
        all_foods = client.get_all_foods()
        existing = {f['name'].lower() for f in all_foods}
        
        for food in foods:
            food = food.strip()
            if not food:
                continue
                
            if food.lower() in existing:
                print(f"⏭️  Already exists: {food}")
                continue
            
            try:
                client.create_food(food)
                print(f"✅ Added: {food}")
            except Exception as e:
                print(f"❌ Failed: {food} ({e})")
    finally:
        client.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_food.py 'food1' 'food2' ...")
        print("Example: python add_food.py 'sriracha' 'kimchi' 'gochujang'")
        sys.exit(1)
    
    add_foods(sys.argv[1:])
