"""
Unit Tests - Pure Python, No External Dependencies
===================================================

These tests verify internal logic without hitting Mealie or LLM APIs.
All tests marked as @pytest.mark.readonly.
"""

import pytest
import json


# =============================================================================
# Test: Accompaniment Schema
# =============================================================================

class TestAccompanimentSchema:
    """Verify the accompaniment classification schema is valid."""
    
    @pytest.mark.readonly
    def test_schema_structure(self):
        """Schema should have correct structure with type enum."""
        from chef_agentic import _schema_accompaniments
        
        schema = _schema_accompaniments()
        
        # Verify top-level structure
        assert schema["type"] == "json_schema"
        assert "json_schema" in schema
        assert schema["json_schema"]["name"] == "meal_accompaniments"
        assert schema["json_schema"]["strict"] == True
        
    @pytest.mark.readonly
    def test_schema_type_enum(self):
        """Schema should enforce RECIPE/PREP/BUY classification."""
        from chef_agentic import _schema_accompaniments
        
        schema = _schema_accompaniments()
        
        # Navigate to the type enum
        items_schema = schema["json_schema"]["schema"]["properties"]["accompaniments"]["items"]
        type_schema = items_schema["properties"]["type"]
        
        assert type_schema["type"] == "string"
        assert set(type_schema["enum"]) == {"recipe", "prep", "buy"}
        
    @pytest.mark.readonly
    def test_schema_required_fields(self):
        """Each accompaniment should require item, type, note fields."""
        from chef_agentic import _schema_accompaniments
        
        schema = _schema_accompaniments()
        
        items_schema = schema["json_schema"]["schema"]["properties"]["accompaniments"]["items"]
        required = items_schema["required"]
        
        assert "item" in required
        assert "type" in required
        assert "note" in required


# =============================================================================
# Test: Recipe Validation
# =============================================================================

class TestRecipeValidation:
    """Verify the recipe validation function catches invalid recipes."""
    
    @pytest.mark.readonly
    def test_valid_recipe_passes(self):
        """A complete recipe should pass validation."""
        from chef_agentic import validate_generated_recipe
        
        valid_recipe = {
            "name": "Steamed Jasmine Rice",
            "recipe_ingredient": ["2 cups jasmine rice", "3 cups water", "1 tsp salt"],
            "recipe_instructions": [
                {"title": "Step 1", "text": "Rinse the rice in cold water until the water runs clear."},
                {"title": "Step 2", "text": "Add rice and water to a pot, bring to boil."},
                {"title": "Step 3", "text": "Reduce heat, cover, and simmer for 15 minutes."},
            ]
        }
        
        is_valid, errors = validate_generated_recipe(valid_recipe)
        
        assert is_valid is True
        assert len(errors) == 0
        
    @pytest.mark.readonly
    def test_empty_name_fails(self):
        """Recipe with empty name should fail."""
        from chef_agentic import validate_generated_recipe
        
        recipe = {
            "name": "",
            "recipe_ingredient": ["rice"],
            "recipe_instructions": [{"text": "Cook the rice properly."}]
        }
        
        is_valid, errors = validate_generated_recipe(recipe)
        
        assert is_valid is False
        assert any("name" in e.lower() for e in errors)
        
    @pytest.mark.readonly
    def test_no_ingredients_fails(self):
        """Recipe without ingredients should fail."""
        from chef_agentic import validate_generated_recipe
        
        recipe = {
            "name": "Empty Recipe",
            "recipe_ingredient": [],
            "recipe_instructions": [{"text": "Do something with nothing."}]
        }
        
        is_valid, errors = validate_generated_recipe(recipe)
        
        assert is_valid is False
        assert any("ingredient" in e.lower() for e in errors)
        
    @pytest.mark.readonly
    def test_no_instructions_fails(self):
        """Recipe without instructions should fail."""
        from chef_agentic import validate_generated_recipe
        
        recipe = {
            "name": "No Instructions",
            "recipe_ingredient": ["rice"],
            "recipe_instructions": []
        }
        
        is_valid, errors = validate_generated_recipe(recipe)
        
        assert is_valid is False
        assert any("instruction" in e.lower() for e in errors)
        
    @pytest.mark.readonly
    def test_short_instruction_fails(self):
        """Instruction with less than 10 chars should fail."""
        from chef_agentic import validate_generated_recipe
        
        recipe = {
            "name": "Short Instructions",
            "recipe_ingredient": ["rice"],
            "recipe_instructions": [{"text": "Cook it."}]  # Only 8 chars
        }
        
        is_valid, errors = validate_generated_recipe(recipe)
        
        assert is_valid is False
        assert any("short" in e.lower() or "char" in e.lower() for e in errors)


# =============================================================================
# Test: NoteItem Data Structure
# =============================================================================

class TestNoteItem:
    """Verify the NoteItem dataclass for PREP/BUY items."""
    
    @pytest.mark.readonly
    def test_note_item_creation(self):
        """NoteItem should store title, text, and item_type."""
        from chef_agentic import NoteItem
        
        note = NoteItem(
            title="Pappardelle",
            text="boil per package directions",
            item_type="prep"
        )
        
        assert note.title == "Pappardelle"
        assert note.text == "boil per package directions"
        assert note.item_type == "prep"
        
    @pytest.mark.readonly
    def test_note_item_buy_type(self):
        """NoteItem should support 'buy' type."""
        from chef_agentic import NoteItem
        
        note = NoteItem(
            title="Crusty Baguette",
            text="from bakery",
            item_type="buy"
        )
        
        assert note.item_type == "buy"


# =============================================================================
# Test: PlannedMeal Summary
# =============================================================================

class TestPlannedMealSummary:
    """Verify PlannedMeal correctly summarizes dishes and notes."""
    
    @pytest.mark.readonly
    def test_meal_summary_with_notes(self):
        """Meal summary should include [TYPE] prefix for notes."""
        from chef_agentic import PlannedMeal, PlannedDish, NoteItem, Candidate
        
        # Create a meal with a dish and notes
        meal = PlannedMeal()
        meal.dishes.append(PlannedDish(
            candidate=Candidate(recipe_id="abc123", name="Beef Ragu")
        ))
        meal.notes.append(NoteItem(
            title="Pappardelle",
            text="boil per package",
            item_type="prep"
        ))
        meal.notes.append(NoteItem(
            title="Parmesan",
            text="grated",
            item_type="buy"
        ))
        
        summary = meal.summary()
        
        assert "Beef Ragu" in summary
        assert "[PREP] Pappardelle" in summary
        assert "[BUY] Parmesan" in summary


# =============================================================================
# Test: Fuzzy Name Matching
# =============================================================================

class TestFuzzyNameMatch:
    """Verify fuzzy name matching for deduplication."""
    
    @pytest.mark.readonly
    def test_exact_match(self):
        """Exact names should match."""
        from chef_agentic import _fuzzy_name_match
        
        assert _fuzzy_name_match("Steamed Rice", "Steamed Rice") is True
        
    @pytest.mark.readonly
    def test_case_insensitive_match(self):
        """Matching should be case-insensitive."""
        from chef_agentic import _fuzzy_name_match
        
        assert _fuzzy_name_match("Steamed Rice", "steamed rice") is True
        assert _fuzzy_name_match("STEAMED RICE", "Steamed Rice") is True
        
    @pytest.mark.readonly
    def test_similar_names_match(self):
        """Similar names should match (Jaccard >= 0.75)."""
        from chef_agentic import _fuzzy_name_match
        
        # "Steamed Jasmine Rice" vs "Jasmine Rice" share 2/3 tokens
        # Jaccard = 2/3 = 0.67, below threshold
        assert _fuzzy_name_match("Steamed Jasmine Rice", "Jasmine Rice", threshold=0.6) is True
        
    @pytest.mark.readonly
    def test_different_names_dont_match(self):
        """Different names should not match."""
        from chef_agentic import _fuzzy_name_match
        
        assert _fuzzy_name_match("Beef Stew", "Chicken Curry") is False
        assert _fuzzy_name_match("Pasta Carbonara", "Spaghetti Bolognese") is False
