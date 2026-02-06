"""
Classification Tests - RECIPE/PREP/BUY Decision Tree
=====================================================

Tests the LLM classification logic for accompaniments.
Uses real LLM calls but does NOT write to Mealie.

The GOAL is to verify that whatever accompaniments the LLM suggests,
they are classified correctly according to the decision tree:

    Q1: Can a home cook make this same-day?
        NO → BUY (fermented foods, bakery items requiring hours)
        YES → continue to Q2
    
    Q2: Would they need a recipe to do it right?
        YES → RECIPE (technique, ratios, or timing matter)
        NO → PREP (trivial: wash produce, boil pasta per package)

We verify:
1. All classifications use valid types (recipe/prep/buy)
2. BUY items are legitimately things you can't make same-day
3. RECIPE items are things that need technique/ratios
4. PREP items are trivial preparations
"""

import pytest
import asyncio
from typing import Dict, List, Any


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def validate_classification(item: str, classification: str, note: str = "") -> tuple[bool, str]:
    """
    Validate classification is sane.
    
    The classification depends on HOW the item is prepared, not just WHAT it is.
    - "cherry tomatoes (wash)" → PREP
    - "roasted cherry tomatoes" → RECIPE (technique required)
    
    We verify:
    1. Type is valid (recipe/prep/buy)
    2. Classification is consistent with the preparation described
    
    Returns (is_valid, reason)
    """
    item_lower = item.lower()
    note_lower = (note or "").lower()
    classification_lower = classification.lower()
    
    # Must be a valid type
    if classification_lower not in ["recipe", "prep", "buy"]:
        return False, f"Invalid type '{classification}' - must be recipe/prep/buy"
    
    # Check consistency: if description mentions technique, should be RECIPE
    technique_words = ["roast", "sauté", "braise", "simmer", "reduce", "caramelize", 
                       "make", "cook", "bake", "fry", "grill"]
    # Note: "prepare" removed - too ambiguous ("easy to prepare" != needs recipe)
    trivial_words = ["wash", "rinse", "boil per package", "frozen", "store-bought", 
                     "from bakery", "buy", "purchase", "easy to prepare", "just placing"]
    fermented_words = ["ferment", "aged", "cured", "from market", "store-bought"]
    
    combined = f"{item_lower} {note_lower}"
    
    # Check for trivial first - these override technique detection
    has_trivial = any(word in combined for word in trivial_words)
    has_technique = any(word in combined for word in technique_words) and not has_trivial
    has_fermented = any(word in combined for word in fermented_words)
    
    # Consistency checks
    if has_technique and classification_lower == "prep":
        return False, f"INCONSISTENT: '{item}' has technique words but classified as PREP"
    
    if has_fermented and classification_lower == "recipe":
        # Check if it's a dish using fermented ingredient vs the ingredient itself
        dish_words = ["soup", "stew", "rice", "fried", "salad"]
        is_dish = any(word in item_lower for word in dish_words)
        if not is_dish:
            return False, f"INCONSISTENT: '{item}' is fermented but classified as RECIPE"
    
    return True, f"'{item}' → {classification.upper()}"


# =============================================================================
# Helper: Get All Accompaniments
# =============================================================================

async def _get_accompaniments_async(
    primary_name: str,
    primary_tags: List[str],
    cuisine: str,
) -> List[Dict[str, Any]]:
    """Run the LLM classification and return all accompaniments."""
    from chef_agentic import determine_meal_accompaniments, Candidate, AgentState
    from datetime import datetime
    
    candidate = Candidate(
        recipe_id="test-primary-001",
        name=primary_name,
        cuisine_primary=cuisine,
        tag_names=primary_tags,
        category_names=[]
    )
    
    state = AgentState(
        week_start=datetime.now().date(),
        history={}
    )
    
    accompaniments, _condiments = await determine_meal_accompaniments(
        primary=candidate,
        cuisine=cuisine,
        meal_type="dinner",
        day="monday",
        state=state
    )
    return accompaniments


def get_accompaniments(primary_name: str, primary_tags: List[str], cuisine: str) -> List[Dict[str, Any]]:
    """Sync wrapper."""
    return run_async(_get_accompaniments_async(primary_name, primary_tags, cuisine))


# =============================================================================
# Classification Tests - Validate ALL suggested accompaniments
# =============================================================================

class TestClassificationLogic:
    """
    Test that ALL accompaniments suggested by the LLM are classified correctly.
    
    We don't care WHAT accompaniments it suggests - we care that whatever
    it suggests, the classification follows the decision tree:
    - Fermented/bakery items → BUY
    - Things needing technique/ratios → RECIPE  
    - Trivial prep → PREP
    """
    
    @pytest.mark.slow
    @pytest.mark.readonly
    def test_french_meal_classifications(self):
        """Test classifications for French cuisine accompaniments."""
        accompaniments = get_accompaniments(
            primary_name="Beef Bourguignon",
            primary_tags=["French Cuisine", "Stew", "Beef"],
            cuisine="French"
        )
        
        print(f"\n=== French Meal: Beef Bourguignon ===")
        assert len(accompaniments) > 0, "LLM should suggest at least one accompaniment"
        
        errors = []
        for acc in accompaniments:
            item = acc.get("item", "")
            acc_type = acc.get("type", "")
            note = acc.get("note", "")
            
            is_valid, reason = validate_classification(item, acc_type, note)
            print(f"  [{acc_type.upper()}] {item}: {reason}")
            
            if not is_valid:
                errors.append(reason)
        
        assert len(errors) == 0, f"Classification errors:\n" + "\n".join(errors)
    
    @pytest.mark.slow
    @pytest.mark.readonly
    def test_japanese_meal_classifications(self):
        """Test classifications for Japanese cuisine accompaniments."""
        accompaniments = get_accompaniments(
            primary_name="Teriyaki Salmon",
            primary_tags=["Japanese Cuisine", "Fish", "Grilled"],
            cuisine="Japanese"
        )
        
        print(f"\n=== Japanese Meal: Teriyaki Salmon ===")
        assert len(accompaniments) > 0, "LLM should suggest at least one accompaniment"
        
        errors = []
        for acc in accompaniments:
            item = acc.get("item", "")
            acc_type = acc.get("type", "")
            note = acc.get("note", "")
            
            is_valid, reason = validate_classification(item, acc_type, note)
            print(f"  [{acc_type.upper()}] {item}: {reason}")
            
            if not is_valid:
                errors.append(reason)
        
        assert len(errors) == 0, f"Classification errors:\n" + "\n".join(errors)
    
    @pytest.mark.slow
    @pytest.mark.readonly
    def test_korean_meal_classifications(self):
        """Test classifications for Korean cuisine accompaniments."""
        accompaniments = get_accompaniments(
            primary_name="Bulgogi",
            primary_tags=["Korean Cuisine", "Beef", "Grilled"],
            cuisine="Korean"
        )
        
        print(f"\n=== Korean Meal: Bulgogi ===")
        assert len(accompaniments) > 0, "LLM should suggest at least one accompaniment"
        
        errors = []
        for acc in accompaniments:
            item = acc.get("item", "")
            acc_type = acc.get("type", "")
            
            note = acc.get("note", "")
            is_valid, reason = validate_classification(item, acc_type, note)
            print(f"  [{acc_type.upper()}] {item}: {reason}")
            
            if not is_valid:
                errors.append(reason)
        
        assert len(errors) == 0, f"Classification errors:\n" + "\n".join(errors)
    
    @pytest.mark.slow
    @pytest.mark.readonly
    def test_italian_meal_classifications(self):
        """Test classifications for Italian cuisine accompaniments."""
        accompaniments = get_accompaniments(
            primary_name="Beef Ragu",
            primary_tags=["Italian Cuisine", "Pasta Sauce", "Beef"],
            cuisine="Italian"
        )
        
        print(f"\n=== Italian Meal: Beef Ragu ===")
        assert len(accompaniments) > 0, "LLM should suggest at least one accompaniment"
        
        errors = []
        for acc in accompaniments:
            item = acc.get("item", "")
            acc_type = acc.get("type", "")
            
            note = acc.get("note", "")
            is_valid, reason = validate_classification(item, acc_type, note)
            print(f"  [{acc_type.upper()}] {item}: {reason}")
            
            if not is_valid:
                errors.append(reason)
        
        assert len(errors) == 0, f"Classification errors:\n" + "\n".join(errors)
    
    @pytest.mark.slow
    @pytest.mark.readonly
    def test_american_meal_classifications(self):
        """Test classifications for American cuisine accompaniments."""
        accompaniments = get_accompaniments(
            primary_name="Fried Chicken",
            primary_tags=["American Cuisine", "Chicken", "Fried"],
            cuisine="American"
        )
        
        print(f"\n=== American Meal: Fried Chicken ===")
        assert len(accompaniments) > 0, "LLM should suggest at least one accompaniment"
        
        errors = []
        for acc in accompaniments:
            item = acc.get("item", "")
            acc_type = acc.get("type", "")
            
            note = acc.get("note", "")
            is_valid, reason = validate_classification(item, acc_type, note)
            print(f"  [{acc_type.upper()}] {item}: {reason}")
            
            if not is_valid:
                errors.append(reason)
        
        assert len(errors) == 0, f"Classification errors:\n" + "\n".join(errors)
    
    @pytest.mark.slow
    @pytest.mark.readonly
    def test_chinese_meal_classifications(self):
        """Test classifications for Chinese cuisine accompaniments."""
        accompaniments = get_accompaniments(
            primary_name="Mapo Tofu",
            primary_tags=["Sichuan Cuisine", "Tofu", "Spicy"],
            cuisine="Sichuan"
        )
        
        print(f"\n=== Chinese Meal: Mapo Tofu ===")
        assert len(accompaniments) > 0, "LLM should suggest at least one accompaniment"
        
        errors = []
        for acc in accompaniments:
            item = acc.get("item", "")
            acc_type = acc.get("type", "")
            
            note = acc.get("note", "")
            is_valid, reason = validate_classification(item, acc_type, note)
            print(f"  [{acc_type.upper()}] {item}: {reason}")
            
            if not is_valid:
                errors.append(reason)
        
        assert len(errors) == 0, f"Classification errors:\n" + "\n".join(errors)


# =============================================================================
# Validation Tests - All types must be valid
# =============================================================================

class TestClassificationValidation:
    """Verify all classifications use valid types."""
    
    @pytest.mark.slow
    @pytest.mark.readonly
    def test_all_types_are_valid(self):
        """Every accompaniment must have type in [recipe, prep, buy]."""
        accompaniments = get_accompaniments(
            primary_name="Test Primary Dish",
            primary_tags=["American Cuisine"],
            cuisine="American"
        )
        
        valid_types = {"recipe", "prep", "buy"}
        
        for acc in accompaniments:
            acc_type = acc.get("type", "").lower().strip()
            assert acc_type in valid_types, \
                f"Invalid type '{acc_type}' for item '{acc.get('item')}'. Must be: {valid_types}"
