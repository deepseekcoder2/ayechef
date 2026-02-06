"""
Shopping List Aggregator

Converts and aggregates shopping list items by food_id using documented unit conversions.
Conservative approach: only deletes items we successfully aggregate; unknown units left untouched.

Sources:
- Densities: https://github.com/elliscode/ingredient-converter
- Item weights: USDA National Nutrient Database SR28
- Cross-validation: King Arthur Baking, Instacart ingredient charts
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from tools.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# SHOPPING-FRIENDLY ROUNDING
# ============================================================================

def round_for_shopping(qty: float, unit_type: str) -> float:
    """
    Round quantities to practical shopping values.
    
    Instead of precise decimals (4.92 kg, 127.8 g), round to increments
    that make sense when actually shopping.
    
    Args:
        qty: The quantity to round
        unit_type: One of 'kg', 'g', 'L', 'ml', 'count'
    
    Returns:
        Rounded quantity appropriate for shopping
    """
    if qty <= 0:
        return qty
    
    if unit_type == 'kg':
        # Round to nearest 0.25 kg (250g increments)
        return round(qty * 4) / 4
    
    elif unit_type == 'g':
        if qty >= 100:
            # Round to nearest 25g for larger amounts
            return round(qty / 25) * 25
        else:
            # Round to nearest 5g for smaller amounts
            return round(qty / 5) * 5
    
    elif unit_type == 'L':
        # Round to nearest 0.25 L (250ml increments)
        return round(qty * 4) / 4
    
    elif unit_type == 'ml':
        if qty >= 100:
            # Round to nearest 25ml for larger amounts
            return round(qty / 25) * 25
        else:
            # Round to nearest 5ml for smaller amounts
            return round(qty / 5) * 5
    
    elif unit_type == 'count':
        # Always round up for count items - if you need any, you need at least 1
        return max(1, round(qty))
    
    # Default: standard rounding
    return round(qty, 1)


# ============================================================================
# CONVERSION CONSTANTS (All Documented)
# ============================================================================

# Standard weight conversions
WEIGHT_TO_GRAMS = {
    "gram": 1, "grams": 1, "g": 1,
    "kilogram": 1000, "kilograms": 1000, "kg": 1000,
    "pound": 454, "pounds": 454, "lb": 454, "lbs": 454,
    "ounce": 28, "ounces": 28, "oz": 28,
}

# Standard volume conversions (US measurements)
VOLUME_TO_ML = {
    "milliliter": 1, "milliliters": 1, "ml": 1,
    "liter": 1000, "liters": 1000, "l": 1000,
    "cup": 240, "cups": 240,
    "tablespoon": 15, "tablespoons": 15, "tbsp": 15,
    "teaspoon": 5, "teaspoons": 5, "tsp": 5,
    "fluid ounce": 30, "fluid ounces": 30, "fl oz": 30,
}

# Density table (grams per milliliter)
# Source: https://github.com/elliscode/ingredient-converter
# Cross-validated with King Arthur Baking (butter: 0.942 g/ml) and Instacart
# Using 0.95 g/ml as consensus value for butter
DENSITY_G_PER_ML = {
    "butter": 0.95,        # Consensus: elliscode 0.955, KA 0.942, Instacart 0.942
    "milk": 0.959,
    "cream": 0.994,
    "heavy cream": 0.994,
    "yogurt": 1.03,
    "honey": 1.42,
    "maple syrup": 1.32,
    "olive oil": 0.92,
    "vegetable oil": 0.92,
    "flour": 0.529,        # all-purpose
    "all-purpose flour": 0.529,
    "sugar": 0.845,
    "granulated sugar": 0.845,
    "brown sugar": 0.93,
    "rice": 0.85,
    "oats": 0.341,
    "cocoa powder": 0.52,
    "peanut butter": 1.09,
    "mayonnaise": 0.91,
    "sour cream": 0.97,
    "cream cheese": 0.98,
    "parmesan": 0.42,      # grated
    "cheddar": 0.47,       # shredded
}

# Count-to-weight conversions (USDA National Nutrient Database SR28)
# Standard reference portions for "medium" size
# ONLY for foods where weight is useful for purchasing
USDA_STANDARD_PORTIONS = {
    "carrot": 61,          # USDA SR28 #11124: 1 medium, 7" long
    "carrots": 61,
    "potato": 213,         # USDA SR28 #11352: 1 medium, 2.5" diameter
    "potatoes": 213,
    "tomato": 123,         # USDA SR28 #11529: 1 medium, 2.5" diameter
    "tomatoes": 123,
    "onion": 110,          # USDA SR28 #11282: 1 medium, 2.5" diameter
    "onions": 110,
}

# Foods that should NEVER convert count to weight
# Weight is not useful for purchasing these items
ALWAYS_COUNT_FOODS = {
    "egg", "eggs",
    "lemon", "lemons", "lime", "limes", "orange", "oranges",
    "avocado", "avocados",
    "bell pepper", "bell peppers",
    "cucumber", "cucumbers",
    "eggplant", "eggplants", "aubergine", "aubergines",
}

# Informal unit to grams conversion (fallback for items not converted at parse time)
# Sources: USDA FoodData Central, America's Test Kitchen, howmuchisin.com
# Format: (food_name, unit) -> grams
INFORMAL_UNIT_WEIGHTS = {
    # Herbs - bunches (grocery store standard)
    ("parsley", "bunch"): 55,
    ("cilantro", "bunch"): 80,
    ("coriander", "bunch"): 80,
    ("basil", "bunch"): 70,
    ("mint", "bunch"): 90,
    ("dill", "bunch"): 21,
    ("thyme", "bunch"): 21,
    ("rosemary", "bunch"): 30,
    ("chive", "bunch"): 25,
    ("chives", "bunch"): 25,
    ("tarragon", "bunch"): 25,
    ("oregano", "bunch"): 25,
    ("sage", "bunch"): 20,
    
    # Herbs - sprigs
    ("parsley", "sprig"): 1,
    ("cilantro", "sprig"): 2,
    ("coriander", "sprig"): 2,
    ("thyme", "sprig"): 1,
    ("rosemary", "sprig"): 1.5,
    ("mint", "sprig"): 2,
    ("dill", "sprig"): 1,
    ("tarragon", "sprig"): 1,
    ("oregano", "sprig"): 1,
    ("sage", "sprig"): 1.5,
    
    # Garlic
    ("garlic", "head"): 55,
    ("garlic", "bulb"): 55,
    ("garlic", "clove"): 5,
    
    # Leafy greens - bunches
    ("spinach", "bunch"): 280,
    ("kale", "bunch"): 200,
    ("chard", "bunch"): 200,
    ("swiss chard", "bunch"): 200,
    ("green onion", "bunch"): 100,
    ("scallion", "bunch"): 100,
    ("spring onion", "bunch"): 100,
    ("watercress", "bunch"): 85,
    ("arugula", "bunch"): 125,
    ("rocket", "bunch"): 125,
    
    # Celery
    ("celery", "stalk"): 40,
    ("celery", "stick"): 40,
    ("celery", "rib"): 40,
    ("celery", "bunch"): 450,
    ("celery", "head"): 450,
    
    # Ginger
    ("ginger", "piece"): 15,
    ("ginger", "knob"): 25,
    ("ginger", "inch"): 10,
    ("ginger", "thumb"): 15,
    
    # Other produce with informal units
    ("broccoli", "head"): 225,
    ("cauliflower", "head"): 500,
    ("lettuce", "head"): 300,
    ("cabbage", "head"): 900,
    ("romaine", "head"): 300,
    ("iceberg", "head"): 500,
    
    # Handfuls (approximate)
    ("spinach", "handful"): 30,
    ("arugula", "handful"): 20,
    ("rocket", "handful"): 20,
    ("herbs", "handful"): 15,
    ("nuts", "handful"): 30,
    ("raisins", "handful"): 30,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AggregatedFood:
    """Aggregated quantities for a single food item."""
    food_id: str
    food_name: str
    label_id: Optional[str]
    
    # Aggregated quantities (in base units)
    total_grams: float = 0
    total_ml: float = 0
    total_count: float = 0
    
    # Track which original item IDs were successfully aggregated
    # (for selective deletion)
    aggregated_item_ids: list[str] = field(default_factory=list)


# ============================================================================
# UNIT CONVERSION FUNCTIONS
# ============================================================================

def normalize_name(name: Optional[str]) -> str:
    """Normalize food/unit names for lookup."""
    if not name:
        return ""
    return name.lower().strip()


def convert_to_grams(item: dict) -> Optional[float]:
    """
    Convert item quantity to grams if possible.
    
    Conversion priority:
    1. Direct weight units (g, kg, oz, lb)
    2. Volume to weight via density (cup, tbsp of butter, etc.)
    3. Informal units via lookup table (bunch, sprig, clove, etc.)
    
    Returns:
        Grams if convertible, None if not
    """
    quantity = item.get("quantity") or 1
    unit = item.get("unit")
    unit_name = normalize_name(unit.get("name") if unit else None)
    food_name = normalize_name(item.get("food", {}).get("name"))
    
    # Direct weight conversion
    if unit_name in WEIGHT_TO_GRAMS:
        return quantity * WEIGHT_TO_GRAMS[unit_name]
    
    # Volume to weight (if density available)
    if unit_name in VOLUME_TO_ML:
        if food_name in DENSITY_G_PER_ML:
            ml = quantity * VOLUME_TO_ML[unit_name]
            density = DENSITY_G_PER_ML[food_name]
            grams = ml * density
            logger.debug(f"  Volume→weight: {quantity} {unit_name} {food_name} = {ml}ml × {density} g/ml = {grams:.1f}g")
            return grams
    
    # Informal unit conversion (bunch, sprig, clove, head, etc.)
    if unit_name and food_name:
        # Try exact match first
        key = (food_name, unit_name)
        if key in INFORMAL_UNIT_WEIGHTS:
            grams_per_unit = INFORMAL_UNIT_WEIGHTS[key]
            grams = quantity * grams_per_unit
            logger.debug(f"  Informal→weight: {quantity} {unit_name} {food_name} = {quantity} × {grams_per_unit}g = {grams:.1f}g")
            return grams
        
        # Try singularizing the unit (bunches→bunch, stalks→stalk, cloves→clove)
        unit_singular = unit_name
        if unit_name.endswith('ches'):  # bunches, inches
            unit_singular = unit_name[:-2]  # bunches → bunch
        elif unit_name.endswith('fuls'):  # handfuls
            unit_singular = unit_name[:-1]  # handfuls → handful
        elif unit_name.endswith('s') and not unit_name.endswith('ss'):
            unit_singular = unit_name[:-1]  # sprigs→sprig, heads→head, cloves→clove
        
        if unit_singular != unit_name:
            key = (food_name, unit_singular)
            if key in INFORMAL_UNIT_WEIGHTS:
                grams_per_unit = INFORMAL_UNIT_WEIGHTS[key]
                grams = quantity * grams_per_unit
                logger.debug(f"  Informal→weight: {quantity} {unit_name} {food_name} = {quantity} × {grams_per_unit}g = {grams:.1f}g")
                return grams
    
    return None


def convert_to_ml(item: dict) -> Optional[float]:
    """
    Convert item quantity to milliliters if possible.
    
    Only converts direct volume measurements (not weight→volume).
    
    Returns:
        Milliliters if convertible, None if not
    """
    quantity = item.get("quantity") or 1
    unit = item.get("unit")
    unit_name = normalize_name(unit.get("name") if unit else None)
    
    # Direct volume conversion
    if unit_name in VOLUME_TO_ML:
        return quantity * VOLUME_TO_ML[unit_name]
    
    return None


def convert_count_to_grams(item: dict) -> Optional[float]:
    """
    Convert count-based items to grams using USDA standard portions.
    
    Only converts if:
    1. Item has no unit (count)
    2. Food is NOT in ALWAYS_COUNT_FOODS
    3. Food is in USDA_STANDARD_PORTIONS
    
    Returns:
        Grams if convertible, None if not
    """
    quantity = item.get("quantity") or 1
    unit = item.get("unit")
    
    # Only convert if no unit (count)
    if unit is not None:
        return None
    
    food_name = normalize_name(item.get("food", {}).get("name"))
    
    # Never convert count to weight for these
    if food_name in ALWAYS_COUNT_FOODS:
        logger.debug(f"  Count preserved: {quantity} {food_name} (ALWAYS_COUNT_FOODS)")
        return None
    
    # Convert if we have USDA data
    if food_name in USDA_STANDARD_PORTIONS:
        grams_per_item = USDA_STANDARD_PORTIONS[food_name]
        total_grams = quantity * grams_per_item
        logger.debug(f"  Count→weight: {quantity} {food_name} × {grams_per_item}g = {total_grams}g (USDA SR28)")
        return total_grams
    
    return None


def extract_count(item: dict) -> float:
    """
    Extract count quantity (items with no unit).
    
    Returns:
        Count quantity, or 0 if item has a unit
    """
    unit = item.get("unit")
    if unit is None:
        return item.get("quantity") or 1
    return 0


# ============================================================================
# CORE AGGREGATION LOGIC
# ============================================================================

def aggregate_shopping_list(items: list[dict]) -> list[AggregatedFood]:
    """
    Aggregate shopping list items by food_id with unit conversion.
    
    Conservative approach:
    - Only aggregates items we can successfully convert
    - Tracks aggregated item IDs for selective deletion
    - Unknown units are left untouched (not tracked for deletion)
    
    Args:
        items: List of shopping list item dicts from Mealie API
        
    Returns:
        List of AggregatedFood objects with consolidated quantities
    """
    aggregated: dict[str, AggregatedFood] = {}
    
    for item in items:
        food_id = item.get("foodId")
        
        # Skip items without food reference (manual entries, PREP/BUY notes)
        # These are valid in Mealie but can't be aggregated by food_id
        if not food_id:
            logger.debug(f"Skipping item without foodId: {item.get('display', item.get('note', 'unknown'))}")
            continue
        
        # Handle food being None (key exists but value is null)
        food = item.get("food") or {}
        food_name = food.get("name", "Unknown")
        
        # Initialize aggregated food entry if needed
        if food_id not in aggregated:
            aggregated[food_id] = AggregatedFood(
                food_id=food_id,
                food_name=food_name,
                label_id=item.get("labelId"),
            )
        
        agg = aggregated[food_id]
        item_id = item.get("id")
        unit = item.get("unit")
        unit_name = normalize_name(unit.get("name") if unit else None)
        quantity = item.get("quantity") or 1
        
        # Track if we successfully converted this item
        converted = False
        
        # Try weight conversion
        grams = convert_to_grams(item)
        if grams is not None:
            agg.total_grams += grams
            converted = True
            logger.debug(f"  ✓ Weight: {quantity} {unit_name} {food_name} → {grams:.1f}g")
        
        # Try volume conversion (only if not already converted to weight)
        # ONLY for true liquids - don't convert tbsp of herbs/dry goods to ml
        elif unit_name in VOLUME_TO_ML:
            food_name_norm = normalize_name(food_name)
            if food_name_norm in DENSITY_G_PER_ML:
                # Already handled in convert_to_grams above
                pass
            elif unit_name in ("liter", "liters", "l", "milliliter", "milliliters", "ml", 
                              "cup", "cups", "fluid ounce", "fluid ounces", "fl oz"):
                # True volume units for liquids - convert to ml
                ml = convert_to_ml(item)
                if ml is not None:
                    agg.total_ml += ml
                    converted = True
                    logger.debug(f"  ✓ Volume: {quantity} {unit_name} {food_name} → {ml:.1f}ml")
            # else: tablespoon/teaspoon for dry goods - leave untouched (don't convert)
        
        # Try count conversion
        elif unit is None:
            # Check if we should convert count to weight
            count_grams = convert_count_to_grams(item)
            if count_grams is not None:
                agg.total_grams += count_grams
                converted = True
            else:
                # Keep as count (ALWAYS_COUNT_FOODS or unknown food)
                agg.total_count += quantity
                converted = True
                logger.debug(f"  ✓ Count: {quantity} {food_name}")
        
        # Unknown unit - can't convert
        if not converted:
            logger.warning(f"  ⚠ Unknown unit '{unit_name}' for {food_name} (item {item_id}), leaving untouched")
        else:
            # Successfully converted - track for deletion
            if item_id:
                agg.aggregated_item_ids.append(item_id)
    
    # Filter out foods with no aggregated data
    result = [agg for agg in aggregated.values() if agg.aggregated_item_ids]
    
    logger.info(f"Aggregated {len(result)} food groups from {len(items)} items")
    return result


# ============================================================================
# MEALIE API INTEGRATION
# ============================================================================

async def write_aggregated_to_mealie(list_id: str, aggregated: list[AggregatedFood]) -> bool:
    """
    Write aggregated items back to Mealie shopping list.
    
    Conservative deletion: Only deletes items that were successfully aggregated.
    Items with unknown units are left untouched in the shopping list.
    
    Args:
        list_id: Mealie shopping list ID
        aggregated: List of AggregatedFood objects
        
    Returns:
        True if successful
    """
    from mealie_shopping_integration import (
        delete_mealie_shopping_items,
        add_mealie_shopping_item
    )
    from mealie_parse import ensure_unit_object
    
    # 1. Delete only items that were successfully aggregated
    all_aggregated_ids = []
    for agg in aggregated:
        all_aggregated_ids.extend(agg.aggregated_item_ids)
    
    if all_aggregated_ids:
        logger.info(f"Deleting {len(all_aggregated_ids)} aggregated items")
        delete_mealie_shopping_items(all_aggregated_ids)
    
    # 2. Look up unit IDs dynamically (no hardcoding)
    gram_unit = ensure_unit_object("gram")
    kg_unit = ensure_unit_object("kilogram")
    ml_unit = ensure_unit_object("milliliter")
    l_unit = ensure_unit_object("liter")
    
    # 3. Add aggregated items with smart unit selection
    added_count = 0
    for agg in aggregated:
        # Add weight item
        if agg.total_grams > 0:
            if agg.total_grams >= 1000:
                qty = round_for_shopping(agg.total_grams / 1000, 'kg')
                unit = kg_unit
                unit_str = "kg"
            else:
                qty = round_for_shopping(agg.total_grams, 'g')
                unit = gram_unit
                unit_str = "g"
            
            # Format quantity for display (remove .0 for whole numbers)
            qty_display = int(qty) if qty == int(qty) else qty
            display = f"{qty_display} {unit_str} {agg.food_name}"
            
            add_mealie_shopping_item(list_id, {
                "display": display,
                "quantity": qty,
                "unitId": unit["id"],
                "foodId": agg.food_id,
                "note": "",  # Empty to avoid concatenation
            })
            added_count += 1
            logger.info(f"  ✓ Added: {display}")
        
        # Add volume item (if couldn't convert to weight)
        if agg.total_ml > 0:
            if agg.total_ml >= 1000:
                qty = round_for_shopping(agg.total_ml / 1000, 'L')
                unit = l_unit
                unit_str = "L"
            else:
                qty = round_for_shopping(agg.total_ml, 'ml')
                unit = ml_unit
                unit_str = "ml"
            
            qty_display = int(qty) if qty == int(qty) else qty
            display = f"{qty_display} {unit_str} {agg.food_name}"
            
            add_mealie_shopping_item(list_id, {
                "display": display,
                "quantity": qty,
                "unitId": unit["id"],
                "foodId": agg.food_id,
                "note": "",
            })
            added_count += 1
            logger.info(f"  ✓ Added: {display}")
        
        # Add count item
        if agg.total_count > 0:
            qty = round_for_shopping(agg.total_count, 'count')
            display = f"{qty} {agg.food_name}"
            
            add_mealie_shopping_item(list_id, {
                "display": display,
                "quantity": qty,
                "unitId": None,  # No unit = count
                "foodId": agg.food_id,
                "note": "",
            })
            added_count += 1
            logger.info(f"  ✓ Added: {display}")
    
    logger.info(f"Added {added_count} aggregated items")
    return True


async def aggregate_and_write_to_mealie(list_id: str, items: list[dict]) -> bool:
    """
    Main entry point: Aggregate shopping list and write back to Mealie.
    
    Args:
        list_id: Mealie shopping list ID
        items: List of shopping list items from Mealie API
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"Starting aggregation for {len(items)} items")
        
        # Aggregate items
        aggregated = aggregate_shopping_list(items)
        
        if not aggregated:
            logger.info("No items to aggregate")
            return True
        
        # Write back to Mealie
        success = await write_aggregated_to_mealie(list_id, aggregated)
        
        if success:
            logger.info("✅ Aggregation complete")
        else:
            logger.error("❌ Failed to write aggregated items")
        
        return success
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}", exc_info=True)
        return False


# ============================================================================
# TESTING
# ============================================================================

async def test_aggregation():
    """Test aggregation with mock data covering all conversion scenarios."""
    
    print("\n" + "="*80)
    print("TESTING SHOPPING LIST AGGREGATION")
    print("="*80 + "\n")
    
    # Mock shopping list items
    mock_items = [
        # Butter: weight + volume (density conversion)
        {
            "id": "1",
            "foodId": "butter-id",
            "food": {"name": "butter"},
            "quantity": 405,
            "unit": {"name": "gram"},
        },
        {
            "id": "2",
            "foodId": "butter-id",
            "food": {"name": "butter"},
            "quantity": 7,
            "unit": {"name": "tablespoon"},
        },
        # Potato: multiple weight units
        {
            "id": "3",
            "foodId": "potato-id",
            "food": {"name": "potato"},
            "quantity": 1150,
            "unit": {"name": "gram"},
        },
        {
            "id": "4",
            "foodId": "potato-id",
            "food": {"name": "potato"},
            "quantity": 1,
            "unit": {"name": "kilogram"},
        },
        # Carrot: count + weight (USDA conversion)
        {
            "id": "5",
            "foodId": "carrot-id",
            "food": {"name": "carrot"},
            "quantity": 10,
            "unit": None,  # count
        },
        {
            "id": "6",
            "foodId": "carrot-id",
            "food": {"name": "carrot"},
            "quantity": 550,
            "unit": {"name": "gram"},
        },
        # Eggs: count + weight (keep separate - ALWAYS_COUNT_FOODS)
        {
            "id": "7",
            "foodId": "egg-id",
            "food": {"name": "eggs"},
            "quantity": 5,
            "unit": None,  # count
        },
        {
            "id": "8",
            "foodId": "egg-id",
            "food": {"name": "eggs"},
            "quantity": 250,
            "unit": {"name": "gram"},
        },
        # Unknown unit (should be left untouched)
        {
            "id": "9",
            "foodId": "butter-id",
            "food": {"name": "butter"},
            "quantity": 2,
            "unit": {"name": "stick"},  # Not in conversion tables
        },
    ]
    
    print(f"Testing with {len(mock_items)} mock items\n")
    
    # Run aggregation
    aggregated = aggregate_shopping_list(mock_items)
    
    print("\n" + "="*80)
    print("AGGREGATION RESULTS")
    print("="*80 + "\n")
    
    for agg in aggregated:
        print(f"{agg.food_name}:")
        if agg.total_grams > 0:
            print(f"  - {agg.total_grams:.1f}g")
        if agg.total_ml > 0:
            print(f"  - {agg.total_ml:.1f}ml")
        if agg.total_count > 0:
            print(f"  - {agg.total_count:.0f} count")
        print(f"  - Aggregated {len(agg.aggregated_item_ids)} items: {agg.aggregated_item_ids}")
        print()
    
    # Expected results
    print("="*80)
    print("EXPECTED RESULTS")
    print("="*80 + "\n")
    print("Butter:")
    print("  - 505g (405g + 7 tbsp × 15ml × 0.95 g/ml = 505g)")
    print("  - Aggregated items: ['1', '2']")
    print("  - Item '9' (2 sticks) LEFT UNTOUCHED (unknown unit)\n")
    print("Potato:")
    print("  - 2150g (1150g + 1000g)")
    print("  - Aggregated items: ['3', '4']\n")
    print("Carrot:")
    print("  - 1160g (10 × 61g USDA + 550g)")
    print("  - Aggregated items: ['5', '6']\n")
    print("Eggs:")
    print("  - 5 count (ALWAYS_COUNT_FOODS)")
    print("  - 250g (kept separate)")
    print("  - Aggregated items: ['7', '8'] (kept as separate entries)\n")
    
    print("="*80)
    print("✅ Test complete - verify results match expected values")
    print("="*80 + "\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_aggregation())
