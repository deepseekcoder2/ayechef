#!/usr/bin/env python3
"""
Diagnose the quality check discrepancy.
Compare what bulk_import_smart sees vs what Mealie actually has.
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
from config import MEALIE_URL, MEALIE_TOKEN, get_mealie_headers
from bulk_import_smart import assess_parsing_quality
from mealie_client import MealieClient

# Test recipe slug (one we imported earlier)
SLUG = "crispy-honey-buffalo-wings-with-blue-cheese-celery-slaw"

print("=" * 60)
print("DIAGNOSING QUALITY CHECK DISCREPANCY")
print("=" * 60)

# Method 1: Direct API call with headers from config
print("\n1. Direct API call with get_mealie_headers():")
response1 = requests.get(
    f"{MEALIE_URL}/api/recipes/{SLUG}",
    headers=get_mealie_headers(),
    timeout=30
)
print(f"   Status: {response1.status_code}")
if response1.status_code == 200:
    data1 = response1.json()
    print(f"   Recipe name: {data1.get('name')}")
    print(f"   Ingredient count: {len(data1.get('recipeIngredient', []))}")
    
    # Check first ingredient
    if data1.get('recipeIngredient'):
        ing = data1['recipeIngredient'][0]
        print(f"   First ingredient:")
        print(f"      quantity: {ing.get('quantity')} (type: {type(ing.get('quantity')).__name__})")
        print(f"      unit: {ing.get('unit')} (type: {type(ing.get('unit')).__name__})")
        print(f"      food: {ing.get('food')} (type: {type(ing.get('food')).__name__})")
    
    quality1 = assess_parsing_quality(data1)
    print(f"   Quality assessment: {quality1}")
else:
    print(f"   Error: {response1.text[:200]}")

# Method 2: Via MealieClient
print("\n2. Via MealieClient:")
client = MealieClient()
try:
    data2 = client.get_recipe(SLUG)
    print(f"   Recipe name: {data2.get('name')}")
    print(f"   Ingredient count: {len(data2.get('recipeIngredient', []))}")
    
    # Check first ingredient
    if data2.get('recipeIngredient'):
        ing = data2['recipeIngredient'][0]
        print(f"   First ingredient:")
        print(f"      quantity: {ing.get('quantity')} (type: {type(ing.get('quantity')).__name__})")
        print(f"      unit: {ing.get('unit')} (type: {type(ing.get('unit')).__name__})")
        print(f"      food: {ing.get('food')} (type: {type(ing.get('food')).__name__})")
    
    quality2 = assess_parsing_quality(data2)
    print(f"   Quality assessment: {quality2}")
finally:
    client.close()

# Method 3: Check what the quality check is actually checking
print("\n3. Detailed quality check analysis:")
if data1 and data1.get('recipeIngredient'):
    total = len(data1['recipeIngredient'])
    poor = 0
    for i, ing in enumerate(data1['recipeIngredient']):
        qty = ing.get('quantity', 0)
        unit = ing.get('unit')
        food = ing.get('food')
        
        is_good = qty > 0 and unit is not None and food is not None
        if not is_good:
            poor += 1
            print(f"   Ingredient {i+1} FAILED:")
            print(f"      qty={qty} (>0: {qty > 0})")
            print(f"      unit={unit} (not None: {unit is not None})")
            print(f"      food={food} (not None: {food is not None})")
    
    print(f"\n   Summary: {poor}/{total} ingredients failed quality check")
    if poor == 0:
        print("   Result: GOOD")
    else:
        print("   Result: POOR")

