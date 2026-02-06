"""Recipe validation functions shared across import modules."""

# Error placeholders that indicate failed import (not real content)
ERROR_PLACEHOLDERS = [
    "could not detect ingredients",
    "could not detect instructions",
    "no ingredients found",
    "no instructions found",
]

# Title patterns that indicate technique/tutorial pages, not actual recipes
TUTORIAL_TITLE_PATTERNS = [
    "how to ",
]


def _is_error_placeholder(text: str) -> bool:
    """Check if text is an error placeholder, not real content."""
    if not text:
        return False
    text_lower = text.strip().lower()
    return any(placeholder in text_lower for placeholder in ERROR_PLACEHOLDERS)


def _has_meaningful_content(item: dict, content_fields: list[str]) -> bool:
    """Check if a dict has meaningful content in any of the specified fields."""
    if not isinstance(item, dict):
        # If it's a string, check if it's non-empty and not an error placeholder
        text = str(item).strip() if item else ""
        return bool(text) and not _is_error_placeholder(text)
    
    for field in content_fields:
        value = item.get(field)
        if value:
            text = str(value).strip()
            if text and not _is_error_placeholder(text):
                return True
    return False


def _count_meaningful_ingredients(ingredients: list) -> int:
    """Count ingredients that have actual content, not just empty placeholders."""
    if not ingredients:
        return 0
    
    # Fields that might contain ingredient text
    content_fields = ["display", "note", "originalText", "food", "name"]
    
    count = 0
    for ing in ingredients:
        if isinstance(ing, dict):
            # Check for food object with name
            food = ing.get("food")
            if food and isinstance(food, dict) and food.get("name"):
                count += 1
                continue
            # Check other content fields
            if _has_meaningful_content(ing, content_fields):
                count += 1
        elif isinstance(ing, str) and ing.strip():
            # Plain string ingredient
            count += 1
    
    return count


def _count_meaningful_instructions(instructions: list) -> int:
    """Count instructions that have actual content, not just empty placeholders."""
    if not instructions:
        return 0
    
    # Fields that might contain instruction text
    content_fields = ["text", "name", "summary"]
    
    count = 0
    for inst in instructions:
        if isinstance(inst, dict):
            if _has_meaningful_content(inst, content_fields):
                count += 1
        elif isinstance(inst, str) and inst.strip():
            # Plain string instruction
            count += 1
    
    return count


def is_valid_recipe_content(recipe_data: dict) -> tuple[bool, str]:
    """
    Validate that imported content is actually a recipe, not an informational page.
    
    A valid recipe must have:
    - At least 1 ingredient with actual content AND
    - At least 1 instruction with actual content AND
    - Not be a template (1 ingredient + 1 instruction is Mealie's default template)
    - Not be a tutorial/technique page (e.g., "How to...")
    
    Args:
        recipe_data: Full recipe data from Mealie API
        
    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    # Check for tutorial/technique titles
    name = recipe_data.get("name", "").lower()
    for pattern in TUTORIAL_TITLE_PATTERNS:
        if name.startswith(pattern):
            return False, f"Tutorial/technique page (title starts with '{pattern.strip()}')"
    
    ingredients = recipe_data.get("recipeIngredient", [])
    instructions = recipe_data.get("recipeInstructions", [])
    
    # Count MEANINGFUL content, not just list length
    ingredient_count = _count_meaningful_ingredients(ingredients)
    instruction_count = _count_meaningful_instructions(instructions)
    
    if ingredient_count == 0 and instruction_count == 0:
        return False, "No ingredients and no instructions (likely a glossary or reference page)"
    
    if ingredient_count == 0:
        return False, f"No ingredients found (list had {len(ingredients) if ingredients else 0} empty entries)"
    
    if instruction_count == 0:
        return False, f"No instructions found (list had {len(instructions) if instructions else 0} empty entries)"
    
    # Check for Mealie default template (1 ingredient + 1 instruction = likely incomplete)
    # Real recipes almost always have at least 2 ingredients OR 2 instructions
    if ingredient_count == 1 and instruction_count == 1:
        return False, "Only 1 ingredient and 1 instruction (likely Mealie template or failed import)"
    
    return True, "Valid recipe content"
