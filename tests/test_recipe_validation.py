"""Tests for recipe validation functions."""
import pytest


class TestIsValidRecipeContent:
    """Tests for is_valid_recipe_content function."""
    
    def test_valid_recipe_with_ingredients_and_instructions(self):
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {
            "recipeIngredient": [{"note": "1 cup flour"}],
            "recipeInstructions": [{"text": "Mix ingredients"}]
        }
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is True
        assert reason == "Valid recipe content"
    
    def test_invalid_no_ingredients_no_instructions(self):
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {"recipeIngredient": [], "recipeInstructions": []}
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is False
        assert "No ingredients and no instructions" in reason
    
    def test_invalid_no_ingredients(self):
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {
            "recipeIngredient": [],
            "recipeInstructions": [{"text": "Do something"}]
        }
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is False
        assert "No ingredients found" in reason
    
    def test_invalid_no_instructions(self):
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {
            "recipeIngredient": [{"note": "1 item"}],
            "recipeInstructions": []
        }
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is False
        assert "No instructions found" in reason
    
    def test_handles_missing_keys(self):
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {}  # No keys at all
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is False
    
    def test_handles_none_values(self):
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {"recipeIngredient": None, "recipeInstructions": None}
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is False
    
    def test_rejects_empty_placeholder_dicts(self):
        """Empty placeholder dicts [{}] should be rejected."""
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {
            "recipeIngredient": [{}],  # Empty dict placeholder
            "recipeInstructions": [{}]  # Empty dict placeholder
        }
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is False
        assert "No ingredients" in reason or "No instructions" in reason
    
    def test_rejects_empty_string_content(self):
        """Entries with empty strings should be rejected."""
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {
            "recipeIngredient": [{"note": ""}, {"display": "   "}],
            "recipeInstructions": [{"text": ""}, {"text": "  "}]
        }
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is False
    
    def test_accepts_food_object_with_name(self):
        """Ingredients with food.name should be valid."""
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {
            "recipeIngredient": [{"food": {"name": "flour"}, "quantity": 1}],
            "recipeInstructions": [{"text": "Mix well"}]
        }
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is True
    
    def test_accepts_display_field(self):
        """Ingredients with display field should be valid."""
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {
            "recipeIngredient": [{"display": "1 cup flour"}],
            "recipeInstructions": [{"text": "Combine ingredients"}]
        }
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is True
    
    def test_accepts_plain_string_ingredients(self):
        """Plain string ingredients (legacy format) should be valid."""
        from utils.recipe_validation import is_valid_recipe_content
        recipe = {
            "recipeIngredient": ["1 cup flour", "2 eggs"],
            "recipeInstructions": ["Mix together", "Bake at 350F"]
        }
        is_valid, reason = is_valid_recipe_content(recipe)
        assert is_valid is True
