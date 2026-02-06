#!/usr/bin/env python3
"""
Shopping List Refinement v2
===========================

Simplified approach that works WITH Mealie's aggregation instead of replacing it.

Key insight: Mealie already aggregates quantities correctly when adding recipes.
We just need to:
1. Filter out pantry items (DELETE)
2. Filter out garbage items (DELETE)
3. Clean up ugly display text (preserve quantities)

This fixes the "1 gram cheese" bug where the old pipeline corrupted quantities.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from tools.logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION (loaded from config.yaml)
# =============================================================================

from config import USER_CONFIG

# Equipment label name - foods labeled as Equipment are filtered out
# Use utils/label_equipment.py to identify and label equipment in Mealie
EQUIPMENT_LABEL_NAME = "Equipment"

# Pantry items loaded from config.yaml - items you always have are filtered out
# Customize your pantry staples in config.yaml under pantry.staples
PANTRY_ITEMS = set(USER_CONFIG["pantry"]["staples"])


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RefinedItem:
    """An item that passed filtering with cleaned display."""
    original_id: str
    cleaned_display: str
    quantity: float
    unit: Optional[dict]
    unit_id: Optional[str]
    food: Optional[dict]
    food_id: Optional[str]
    food_name: str
    note: str = ""
    checked: bool = False
    position: int = 0


@dataclass
class RefinementResult:
    """Result of the refinement process."""
    success: bool
    items_to_keep: List[RefinedItem]
    items_to_delete: List[str]  # IDs of items to delete
    pantry_filtered: List[str]  # Food names filtered as pantry
    equipment_filtered: List[str]  # Food names filtered as equipment
    garbage_rejected: List[str]  # Display text of garbage items
    total_input: int
    errors: List[str] = field(default_factory=list)


# =============================================================================
# DISPLAY CLEANING
# =============================================================================

def clean_display_text(display: str, quantity: float, unit_name: Optional[str], food_name: str) -> str:
    """
    Clean up Mealie's ugly concatenated display text.
    
    Mealie creates displays like:
    "340 grams blue cheeses crumbled into chunks | such as dolcelatte, cubed | crumbled"
    
    We want:
    "340 g blue cheese"
    
    Also converts large quantities to better units:
    - 1250 ml → 1.25 L
    - 1500 g → 1.5 kg
    
    Args:
        display: Original display from Mealie
        quantity: The quantity value
        unit_name: The unit name (or None)
        food_name: The food name
    
    Returns:
        Cleaned display string
    """
    # Convert large quantities to better units
    final_qty = quantity
    final_unit = unit_name.lower() if unit_name else None
    
    if final_unit in ("gram", "grams", "g") and quantity >= 1000:
        final_qty = quantity / 1000
        final_unit = "kg"
    elif final_unit in ("milliliter", "milliliters", "ml") and quantity >= 1000:
        final_qty = quantity / 1000
        final_unit = "L"
    
    # Format quantity (remove .0 for whole numbers, keep 1-2 decimals otherwise)
    if final_qty == int(final_qty):
        qty_str = str(int(final_qty))
    elif final_qty * 10 == int(final_qty * 10):
        qty_str = f"{final_qty:.1f}"
    else:
        qty_str = f"{final_qty:.2f}".rstrip('0').rstrip('.')
    
    # Abbreviate common units
    unit_abbrev = {
        "gram": "g",
        "grams": "g",
        "kilogram": "kg",
        "kilograms": "kg",
        "kg": "kg",
        "milliliter": "ml",
        "milliliters": "ml",
        "ml": "ml",
        "liter": "L",
        "liters": "L",
        "l": "L",
        "tablespoon": "tbsp",
        "tablespoons": "tbsp",
        "teaspoon": "tsp",
        "teaspoons": "tsp",
    }
    
    unit_display = ""
    if final_unit:
        unit_display = unit_abbrev.get(final_unit.lower(), final_unit)
    
    # Use food name directly - Mealie's food database stores correct singular names
    # The ugly plurals ("pestoes", "mints") appear in display text which we ignore
    clean_food = food_name
    
    # Build clean display
    if unit_display:
        return f"{qty_str} {unit_display} {clean_food}"
    else:
        return f"{qty_str} {clean_food}"


# =============================================================================
# FILTERING LOGIC
# =============================================================================

def is_pantry_item(food_name: str) -> bool:
    """Check if food name matches a pantry staple."""
    if not food_name:
        return False
    
    food_lower = food_name.lower().strip()
    
    # Direct match
    if food_lower in PANTRY_ITEMS:
        return True
    
    # Check if any pantry item is contained in the food name
    # (e.g., "extra virgin olive oil" contains "olive oil")
    for pantry in PANTRY_ITEMS:
        if pantry in food_lower:
            return True
    
    return False


def is_equipment_item(item: dict) -> bool:
    """
    Check if item is kitchen equipment based on its food's label.
    
    Equipment filtering is based on Mealie's label system, not hardcoded keywords.
    Use utils/label_equipment.py to identify and label equipment foods.
    
    Args:
        item: Shopping list item dict from Mealie
        
    Returns:
        True if the item's food is labeled as Equipment
    """
    food = item.get("food")
    if not isinstance(food, dict):
        return False
    
    # Check by label name (in case label object is included)
    label = food.get("label")
    if isinstance(label, dict):
        label_name = label.get("name", "")
        if label_name.lower() == EQUIPMENT_LABEL_NAME.lower():
            return True
    
    # Also check labelId against known Equipment label
    # This is a fallback - the label object should contain the name
    label_id = food.get("labelId")
    if label_id:
        # We don't hardcode the label ID - if there's a labelId and label object
        # is missing, we can't determine if it's Equipment without an API call
        # The proper solution is to ensure label object is populated
        pass
    
    return False


def is_garbage_item(item: dict) -> Tuple[bool, str]:
    """
    Check if item is garbage/malformed.
    
    Returns:
        Tuple of (is_garbage, reason)
    """
    display = item.get("display", "").strip()
    quantity = item.get("quantity") or 0
    food = item.get("food")
    food_name = food.get("name") if isinstance(food, dict) else None
    
    # No food reference and no meaningful display
    if not food_name and quantity == 0:
        return True, f"no food and qty=0: '{display}'"
    
    # No food reference and no display text (quantity-only item)
    if not food_name and not display:
        return True, f"no food and no display (qty={quantity})"
    
    # Numeric-only display (catches "37", "123.5", etc.)
    if display and display.replace(".", "").replace("-", "").replace(" ", "").isdigit():
        return True, f"numeric-only: '{display}'"
    
    # Very short display with no food
    if len(display) < 3 and not food_name:
        return True, f"too short: '{display}'"
    
    # Display is just a fraction
    if re.match(r'^[\d/½¼¾⅓⅔⅛⅜⅝⅞\s]+$', display):
        return True, f"fraction-only: '{display}'"
    
    # Item has no food, no display, just quantity - definitely garbage
    if not food_name and (not display or display == str(int(quantity)) or display == str(quantity)):
        return True, f"qty-only item: '{display}' (qty={quantity})"
    
    return False, ""


# =============================================================================
# MAIN REFINEMENT FUNCTION
# =============================================================================

def refine_shopping_list(items: List[dict]) -> RefinementResult:
    """
    Refine a shopping list by filtering and cleaning.
    
    This preserves Mealie's aggregated quantities - we just filter and clean display.
    
    Args:
        items: List of shopping list items from Mealie (listItems array)
    
    Returns:
        RefinementResult with items to keep and items to delete
    """
    logger.info(f"Refining {len(items)} shopping list items")
    
    items_to_keep = []
    items_to_delete = []
    pantry_filtered = []
    equipment_filtered = []
    garbage_rejected = []
    errors = []
    
    for item in items:
        item_id = item.get("id")
        display = item.get("display", "").strip()
        quantity = item.get("quantity") or 0
        unit = item.get("unit")
        unit_id = item.get("unitId")
        food = item.get("food")
        food_id = item.get("foodId")
        extras = item.get("extras", {})
        
        # Extract names (handle food being None)
        unit_name = unit.get("name") if isinstance(unit, dict) else None
        food_name = food.get("name") if isinstance(food, dict) else None
        
        # Skip PREP/BUY items - they don't need cleaning and should stay untouched
        # These are items added from meal plan notes (e.g., "crusty baguette", "salad ingredients")
        if extras.get("source") == "prep_buy":
            logger.debug(f"Skipping PREP/BUY item (leaving untouched): {display}")
            continue
        
        # Check if garbage
        is_garbage, reason = is_garbage_item(item)
        if is_garbage:
            logger.debug(f"Rejecting garbage: {reason}")
            garbage_rejected.append(f"{display} ({reason})")
            if item_id:
                items_to_delete.append(item_id)
            continue
        
        # Check if equipment (based on food's label in Mealie)
        if is_equipment_item(item):
            logger.debug(f"Filtering equipment: {food_name}")
            equipment_filtered.append(food_name or display)
            if item_id:
                items_to_delete.append(item_id)
            continue
        
        # Check if pantry item
        if food_name and is_pantry_item(food_name):
            logger.debug(f"Filtering pantry item: {food_name}")
            pantry_filtered.append(food_name)
            if item_id:
                items_to_delete.append(item_id)
            continue
        
        # Item passes filters - clean it up
        if food_name:
            cleaned_display = clean_display_text(display, quantity, unit_name, food_name)
        else:
            # No food reference - use original display but try to clean it
            # Remove everything after | separator
            cleaned_display = display.split("|")[0].strip()
        
        refined = RefinedItem(
            original_id=item_id,
            cleaned_display=cleaned_display,
            quantity=quantity,
            unit=unit,
            unit_id=unit_id,
            food=food,
            food_id=food_id,
            food_name=food_name or display,
            note=item.get("note", ""),
            checked=item.get("checked", False),
            position=item.get("position", 0),
        )
        items_to_keep.append(refined)
    
    # Log summary
    logger.info(f"Refinement complete: {len(items_to_keep)} keep, {len(items_to_delete)} delete")
    logger.info(f"  Pantry filtered: {len(pantry_filtered)}")
    logger.info(f"  Equipment filtered: {len(equipment_filtered)}")
    logger.info(f"  Garbage rejected: {len(garbage_rejected)}")
    
    return RefinementResult(
        success=True,
        items_to_keep=items_to_keep,
        items_to_delete=items_to_delete,
        pantry_filtered=pantry_filtered,
        equipment_filtered=equipment_filtered,
        garbage_rejected=garbage_rejected,
        total_input=len(items),
        errors=errors,
    )


def format_for_mealie(items: List[RefinedItem]) -> List[dict]:
    """
    Convert refined items back to Mealie format.
    
    Includes food/unit IDs so Mealie generates clean display like "340 grams blue cheese"
    instead of ugly concatenated notes. The IDs must be included for proper resolution.
    
    NOTE: Mealie regenerates display from structured fields, so our cleaned_display is
    advisory. The actual display will be "{qty} {unit} {food}" which is what we want.
    """
    mealie_items = []
    
    for idx, item in enumerate(items):
        mealie_item = {
            "display": item.cleaned_display,  # Advisory - Mealie may regenerate
            "quantity": item.quantity,        # CRITICAL: use actual quantity
            "note": "",                       # Clear notes to avoid ugly concatenation
            "checked": item.checked,
            "position": idx + 1,
        }
        
        # Include food reference with ID (required for clean display)
        if item.food and item.food_id:
            mealie_item["food"] = item.food
            mealie_item["foodId"] = item.food_id
        
        # Include unit reference with ID (required for clean display)
        if item.unit and item.unit_id:
            mealie_item["unit"] = item.unit
            mealie_item["unitId"] = item.unit_id
        
        mealie_items.append(mealie_item)
    
    return mealie_items


# =============================================================================
# STANDALONE TEST
# =============================================================================

async def test_refinement():
    """Test the refinement with real Mealie data."""
    from mealie_client import MealieClient
    
    print("=" * 60)
    print("SHOPPING LIST REFINEMENT v2 TEST")
    print("=" * 60)
    
    client = MealieClient()
    try:
        # Get most recent shopping list
        lists = client.get_all_shopping_lists()
        if not lists:
            print("No shopping lists found")
            return
        
        list_id = lists[0]["id"]
        list_name = lists[0]["name"]
        print(f"Testing with: {list_name}")
        
        # Fetch full list
        shopping_list = client.get_shopping_list(list_id)
        items = shopping_list.get("listItems", [])
        print(f"Input items: {len(items)}")
        
        # Run refinement
        result = refine_shopping_list(items)
        
        print(f"\nResults:")
        print(f"  Items to keep: {len(result.items_to_keep)}")
        print(f"  Items to delete: {len(result.items_to_delete)}")
        print(f"  Pantry filtered: {len(result.pantry_filtered)}")
        print(f"  Equipment filtered: {len(result.equipment_filtered)}")
        print(f"  Garbage rejected: {len(result.garbage_rejected)}")
        
        # Show some examples
        print(f"\n=== SAMPLE CLEANED ITEMS ===")
        for item in result.items_to_keep[:10]:
            print(f"  qty={item.quantity}, display=\"{item.cleaned_display}\"")
        
        print(f"\n=== PANTRY ITEMS FILTERED ===")
        for name in result.pantry_filtered[:10]:
            print(f"  - {name}")
        
        print(f"\n=== EQUIPMENT FILTERED ===")
        for name in result.equipment_filtered[:10]:
            print(f"  - {name}")
        
        print(f"\n=== GARBAGE REJECTED ===")
        for reason in result.garbage_rejected[:10]:
            print(f"  - {reason}")
        
        # Format for Mealie
        mealie_items = format_for_mealie(result.items_to_keep)
        print(f"\n=== MEALIE FORMAT SAMPLE ===")
        for item in mealie_items[:5]:
            print(f"  qty={item['quantity']}, display=\"{item['display']}\"")
    finally:
        client.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_refinement())
