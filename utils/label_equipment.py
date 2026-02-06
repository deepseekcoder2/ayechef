#!/usr/bin/env python3
"""
Equipment Labeling Utility
==========================

Identifies and labels kitchen equipment that was incorrectly imported as "foods"
in Mealie. This is typically caused by recipe scrapers including equipment lists
in the ingredients section.

Strategy:
1. Pattern-based detection for OBVIOUS equipment (knife, pan, scale, etc.)
2. Heuristic detection for equipment-like naming patterns
3. Manual review list for uncertain cases
4. LLM classification option for bulk uncertain cases

This is a ONE-TIME cleanup utility, not a runtime filter.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.logging_utils import get_logger
from mealie_client import MealieClient

logger = get_logger(__name__)

# Equipment patterns - these are NEVER food items
# Be specific to avoid false positives like "pot yogurt" or "microwave rice"
EQUIPMENT_PATTERNS = [
    # Cutting tools
    r'\bknife\b', r'\bknives\b', r'\bkitchen scissors\b', r'\bshears\b',
    # Cookware - specific patterns only
    r'\bsaucepan\b', r'\bfrying pan\b', r'\broasting tin\b', r'\bskillet\b',
    r'\bcasserole dish\b', r'\bbaking tray\b', r'\bgriddle\b',
    r'\bovenproof pan\b', r'\bcake tin\b', r'\bmuffin tin\b',
    r'\bSwiss roll.*tin\b', r'\bspringform\b',
    # Measuring - specific
    r'\bmeasuring scales?\b', r'\bmeasuring spoons?\b', r'\bmeasuring jug\b',
    r'\bweighing scales?\b',
    # Utensils
    r'\bspatula\b', r'\belectric.*whisk\b', r'\bhand whisk\b',
    r'\btongs\b', r'\bladle\b',
    r'\bwooden spoon\b', r'\bslotted spoon\b', r'\bserving spoon\b',
    r'\bvegetable peeler\b', r'\bgrater\b', r'\bzester\b', r'\bmandoline\b',
    r'\bsieve\b', r'\bstrainer\b', r'\bcolander\b', r'\bfunnel\b',
    # Boards - specific equipment contexts
    r'\bchopping board\b', r'\bcutting board\b', r'\bcake board\b',
    r'\bwooden board\b', r'\bpastry board\b',
    # Bowls - only equipment contexts
    r'\bmixing bowl\b',
    # Appliances - be specific
    r'\bblender\b', r'\bfood processor\b', r'\bstand mixer\b', r'\bhand mixer\b',
    # Baking equipment
    r'\bbaking sheet\b', r'\bcookie sheet\b', r'\bflat baking sheet\b',
    r'\bparchment paper\b', r'\bbaking paper\b', r'\bsilicone mat\b',
    r'\bcake lifter\b', r'\bpalette knife\b',
    # Misc equipment
    r'\b(?:food|kitchen)\s+thermometer\b',
    r'\bcooling rack\b', r'\bwire rack\b',
    r'\boven gloves?\b',
]

# Food context patterns - items matching these are NOT equipment
# These override equipment patterns to avoid false positives
FOOD_CONTEXT_PATTERNS = [
    # "pot" as container for food products
    r'\bpot\b.*\b(?:yogurt|yoghurt|cream|cheese|sauce|crabmeat|honey)\b',
    r'\b(?:yogurt|yoghurt|cream|cheese).*\bpot\b',
    r'\bgranola\s+pot\b',
    # "microwave" as cooking method, not appliance
    r'\bmicrowave\b.*\b(?:rice|vegetable|grain|noodle|pilau|sticky|wholegrain)\b',
    r'\bpouch\s+microwave\b',
    r'\bcooked\s+microwave\b',
    # "oven" as cooking method
    r'\boven[\s-](?:roasted|baked|fries?|ready|chips?)\b',
    # Noodle products
    r'\bstraight[\s-]to[\s-]wok\b',
    # Food names that happen to contain equipment words
    r'\bhot\s+pot\s+soup\b',
    r'\bstock\s+pot\b',  # Stock Pot brand
    r'\bsalad\s+bowl\b',  # serving context
    r'\btortilla\s+bowl\b',  # food item
    r'\bpan\s+juices?\b',  # cooking liquid
    r'\bpetit\s+four\s+case\b',  # baking cup
    r'\bplant\s+pot\b',  # gardening
    r'\bfoil\s+case\b',  # baking cup
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EQUIPMENT_PATTERNS]
COMPILED_FOOD_CONTEXT = [re.compile(p, re.IGNORECASE) for p in FOOD_CONTEXT_PATTERNS]


def is_likely_equipment(food_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a food name is likely kitchen equipment.
    
    Uses a two-pass approach:
    1. Check food context patterns - if matched, it's NOT equipment
    2. Check equipment patterns - if matched, it IS equipment
    
    Returns:
        Tuple of (is_equipment, matched_pattern)
    """
    name_lower = food_name.lower().strip()
    
    # First pass: check if this is a food item that happens to contain
    # equipment-like words (e.g., "pot yogurt", "microwave rice")
    for food_pattern in COMPILED_FOOD_CONTEXT:
        if food_pattern.search(name_lower):
            return False, None
    
    # Second pass: check equipment patterns
    for pattern in COMPILED_PATTERNS:
        if pattern.search(name_lower):
            return True, pattern.pattern
    
    return False, None


def get_all_unlabeled_foods(client: MealieClient) -> List[Dict]:
    """Fetch all foods that don't have a label assigned."""
    all_foods = client.get_all_foods()
    # Filter to unlabeled foods
    return [f for f in all_foods if not f.get('labelId')]


def get_equipment_label_id(client: MealieClient) -> Optional[str]:
    """Get or create the Equipment label ID."""
    # Try to find existing
    labels = client.get_all_labels()
    equipment = next((l for l in labels if l.get('name') == 'Equipment'), None)
    if equipment:
        return equipment['id']
    
    # Create if not exists
    try:
        new_label = client.create_label('Equipment', '#808080')
        return new_label['id']
    except Exception:
        return None


def label_food_as_equipment(food_id: str, label_id: str, client: MealieClient) -> bool:
    """
    Assign the Equipment label to a food.
    
    Note: Mealie's foods API requires PUT with full object, not PATCH.
    """
    try:
        # Get full food object first
        food = client.get_food(food_id)
        food['labelId'] = label_id
        
        # PUT the updated object
        client.update_food(food_id, food)
        return True
    except Exception:
        return False


def scan_for_equipment(dry_run: bool = True) -> None:
    """
    Scan all unlabeled foods and identify equipment.
    
    Args:
        dry_run: If True, only show what would be labeled. If False, apply labels.
    """
    client = MealieClient()
    try:
        print("=" * 60)
        print("EQUIPMENT DETECTION SCAN")
        print("=" * 60)
        
        # Get Equipment label ID
        equipment_label_id = get_equipment_label_id(client)
        if not equipment_label_id:
            print("ERROR: Could not get/create Equipment label")
            return
        
        print(f"Equipment label ID: {equipment_label_id}")
        print()
        
        # Get all unlabeled foods
        print("Fetching unlabeled foods from Mealie...")
        unlabeled_foods = get_all_unlabeled_foods(client)
        print(f"Found {len(unlabeled_foods)} unlabeled foods")
        print()
        
        # Detect equipment
        equipment_found = []
        for food in unlabeled_foods:
            name = food.get('name', '')
            is_equip, pattern = is_likely_equipment(name)
            if is_equip:
                equipment_found.append({
                    'id': food['id'],
                    'name': name,
                    'pattern': pattern
                })
        
        print(f"Detected {len(equipment_found)} equipment items:")
        print("-" * 60)
        for item in equipment_found:
            print(f"  • {item['name']}")
        print()
        
        if dry_run:
            print("DRY RUN - No changes made")
            print("Run with --apply to label these as Equipment")
        else:
            print("Applying Equipment labels...")
            success = 0
            for item in equipment_found:
                if label_food_as_equipment(item['id'], equipment_label_id, client):
                    success += 1
                    print(f"  ✓ {item['name']}")
                else:
                    print(f"  ✗ {item['name']} - FAILED")
            
            print()
            print(f"Labeled {success}/{len(equipment_found)} items as Equipment")
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Identify and label kitchen equipment in Mealie's food database"
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply labels (default is dry-run)'
    )
    parser.add_argument(
        '--check',
        type=str,
        help='Check if a specific food name is detected as equipment'
    )
    
    args = parser.parse_args()
    
    if args.check:
        is_equip, pattern = is_likely_equipment(args.check)
        if is_equip:
            print(f"'{args.check}' IS equipment (matched: {pattern})")
        else:
            print(f"'{args.check}' is NOT detected as equipment")
        return
    
    scan_for_equipment(dry_run=not args.apply)


if __name__ == '__main__':
    main()
