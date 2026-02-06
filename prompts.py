"""
AI Chef Prompts Configuration
==============================

This module contains all LLM prompts used by the AI Chef agent.
Separating prompts from code makes it easier to tune and experiment
with different prompt strategies without modifying the core logic.

Prompts are formatted as Python f-strings and expect certain variables
to be filled in at runtime (see comments for required variables).

HOUSEHOLD CONTEXT:
The get_household_context() function builds a context string from config.yaml
for use in prompts. This keeps user-specific details (household size, preferences)
separate from culinary domain knowledge (Asian meals = 2-3 dishes, etc.).
"""

from config import USER_CONFIG


def get_household_context() -> str:
    """
    Build household context string for LLM prompts from config.yaml.
    
    Combines:
    - household.description + servings (who's eating)
    - preferences.cuisines (preferred cuisines)
    - preferences.dietary_restrictions (hard restrictions)
    - personal.dietary (natural language dietary notes)
    - personal.cooking (kitchen constraints)
    
    Returns:
        String describing the household for use in prompts
    """
    h = USER_CONFIG["household"]
    p = USER_CONFIG.get("preferences", {})
    personal = USER_CONFIG.get("personal", {})
    
    # Start with household description
    description = h.get('description', '').strip()
    if description:
        context = f"{description} ({h['servings']} servings)"
    else:
        context = f"{h['servings']} servings"
    
    # Cuisine preferences (keywords)
    cuisines = p.get("cuisines", [])
    if cuisines:
        context += f". Cuisine preferences: {', '.join(cuisines)}"
    
    # Hard dietary restrictions (keywords for filtering)
    restrictions = p.get("dietary_restrictions", [])
    if restrictions:
        context += f". Dietary restrictions: {', '.join(restrictions)}"
    
    # Natural language dietary notes (softer guidance)
    dietary = personal.get("dietary", [])
    if dietary:
        context += f". Dietary notes: {'; '.join(dietary)}"
    
    # Kitchen constraints (time, equipment, skill)
    cooking = personal.get("cooking", "").strip()
    if cooking:
        context += f". Kitchen: {cooking}"
    
    return context


def get_household_context_detailed() -> str:
    """
    Build detailed household context for shopping list and meal planning prompts.
    
    This provides more context than get_household_context() for prompts that
    need additional details about pantry, preferences, etc.
    
    Returns:
        Detailed string describing household context
    """
    h = USER_CONFIG["household"]
    p = USER_CONFIG.get("preferences", {})
    pantry = USER_CONFIG.get("pantry", {})
    personal = USER_CONFIG.get("personal", {})
    
    # Build detailed context
    description = h.get('description', '').strip()
    if description:
        lines = [f"Household: {description} ({h['servings']} servings)"]
    else:
        lines = [f"Household: {h['servings']} servings"]
    
    lines.append(f"Meals planned: {', '.join(h['meal_types'])}")
    
    # Preferences
    cuisines = p.get("cuisines", [])
    if cuisines:
        lines.append(f"Preferred cuisines: {', '.join(cuisines)}")
    
    restrictions = p.get("dietary_restrictions", [])
    if restrictions:
        lines.append(f"Dietary restrictions: {', '.join(restrictions)}")
    
    # Personal context
    dietary = personal.get("dietary", [])
    if dietary:
        lines.append(f"Dietary notes: {'; '.join(dietary)}")
    
    cooking = personal.get("cooking", "").strip()
    if cooking:
        lines.append(f"Kitchen: {cooking}")
    
    def _format_list(label: str, items: list, max_items: int = 60) -> str:
        clean = [str(x).strip() for x in (items or []) if str(x).strip()]
        if not clean:
            return ""
        shown = clean[:max_items]
        suffix = f" (+{len(clean) - max_items} more)" if len(clean) > max_items else ""
        return f"{label}: {', '.join(shown)}{suffix}"

    # Pantry info (explicit list so prompts can use it without hardcoding)
    staples_line = _format_list("Pantry staples (exclude)", pantry.get("staples", []))
    if staples_line:
        lines.append(staples_line)

    # Shopping exclusions (separate from pantry staples)
    always_exclude = USER_CONFIG.get("shopping", {}).get("exclusions", {}).get("always_exclude", [])
    exclude_line = _format_list("Shopping exclusions (always exclude)", always_exclude)
    if exclude_line:
        lines.append(exclude_line)
    
    return ". ".join(lines)

# =============================================================================
# PROMPT TUNING NOTES
# =============================================================================
"""
When tuning prompts, consider:

1. TEMPERATURE: Currently set to 0.7 in chef_agentic.py
   - Lower (0.3-0.5): More consistent, less creative
   - Higher (0.8-1.0): More variety, potentially less practical

2. SYSTEM PROMPT adjustments:
   - Modify dish counts for Asian/Western meals
   - Add dietary restrictions or preferences
   - Emphasize seasonal ingredients
   - Add cost consciousness

3. USER PROMPT adjustments:
   - Include more context (weather, occasions, leftovers)
   - Add ingredient preferences/aversions
   - Mention specific nutrition goals

4. OUTPUT FORMAT:
   - Current format uses day names as keys
   - Could add metadata like cuisine type, prep time
   - Could request nutritional balance info

5. COMMON ISSUES:
   - AI suggests non-existent recipes → Ensure "use exact names" is emphasized
   - Too many complex meals → Adjust "mix preparation complexity" guideline
   - Imbalanced cuisines → Strengthen "balance cuisines throughout the week"
   - Invalid JSON → Keep "OUTPUT ONLY VALID JSON" and "no markdown" instructions
"""

# =============================================================================
# SHOPPING LIST REFINEMENT PROMPTS
# =============================================================================

SHOPPING_REFINEMENT_PROMPT = """You are a shopping list quality expert. Your mission is to produce clean, purchasable ingredients by cleaning and filtering the raw input.

PROCESSING APPROACH:
1. CLEAN items by removing preparation instructions and non-essential modifiers
2. FILTER OUT only truly non-purchasable items (pantry staples, decorations, vague items)
3. INCLUDE cleaned items that represent real ingredients you can buy

ITEMS TO REJECT (after cleaning):
- Pantry staples already in household (see exclusion list below)
- Decoration/garnish-only items with no substance (e.g., "for dusting", "to garnish")
- Completely vague quantities impossible to purchase (e.g., "to taste", "as needed" with no base quantity)
- Items that become empty/meaningless after cleaning preparation instructions

HOUSEHOLD CONTEXT:
{household_context}

RAW INGREDIENTS FROM MEALIE:
{raw_ingredients}

CLEANING EXAMPLES (CLEAN THESE, DON'T REJECT):
[OK] "2 chicken breasts, roughly chopped" → CLEAN to "2 chicken breasts" → INCLUDE
[OK] "500g beef, sliced thin" → CLEAN to "500g beef" → INCLUDE
[OK] "3 carrots, diced" → CLEAN to "3 carrots" → INCLUDE
[OK] "2 Tablespoons Grapeseed Oils" → CLEAN to "2 Tablespoons Grapeseed Oil" (fix pluralization) → INCLUDE

REJECTION EXAMPLES (REJECT THESE AFTER CLEANING):
[REJECT] "For Dusting" - no ingredient substance after cleaning
[REJECT] "To taste" (with no ingredient) - unquantifiable
[REJECT] "For garnish" (with no ingredient) - decoration only
[REJECT] "Cooked rice" → CLEAN to "rice" → REJECT (pantry staple)
[REJECT] "Soy sauce" → REJECT (pantry staple / shopping exclusion)

EXCLUSIONS (DO NOT INCLUDE):
- Pantry staples already in household (see HOUSEHOLD CONTEXT: Pantry staples)
- Shopping exclusions configured by the user (see HOUSEHOLD CONTEXT: Shopping exclusions)

INGREDIENT CLEANING RULES (Apply to ALL items first):
1. Remove ALL preparation instructions: chopped, diced, sliced, minced, grated, crushed, etc.
2. Remove ALL cooking state modifiers: cooked, steamed, baked, roasted, grilled, etc.
3. Remove ALL presentation terms: to serve, for garnish, for dusting, etc.
4. Fix pluralization errors in ingredient names (oils → oil)
5. KEEP the food name - this is the most important part!
6. Keep quantity and unit information intact

UNIT CONVERSION RULES (CRITICAL FOR AGGREGATION):
When combining items with DIFFERENT units, you MUST convert to a common unit first:

WEIGHT CONVERSIONS (always convert to grams, then to sensible output unit):
- 1 kilogram = 1000 grams
- 1 kg = 1000g
- 1 pound = 454 grams
- 1 lb = 454g
- 1 oz = 28 grams

VOLUME CONVERSIONS (always convert to milliliters, then to sensible output unit):
- 1 liter = 1000 milliliters
- 1 cup = 240 milliliters
- 1 tablespoon = 15 milliliters
- 1 teaspoon = 5 milliliters

OUTPUT UNIT SELECTION:
- Weights ≥ 1000g → use kg (e.g., 1500g → 1.5 kg)
- Weights < 1000g → use g (e.g., 450g stays as 450g)
- Volumes ≥ 1000ml → use liters
- Volumes < 1000ml → use ml

MIXED UNIT AGGREGATION EXAMPLE:
Input items:
- "1.25 kilograms potatoes" (= 1250g)
- "700 grams potatoes" (= 700g)
- "300 grams potatoes" (= 300g)
- "2 potatoes large" (≈ 400g estimate for 2 large potatoes)

Step 1: Convert all to grams: 1250 + 700 + 300 + 400 = 2650g
Step 2: Convert to sensible unit: 2650g = 2.65 kg
Step 3: Output: {{"display": "2.7 kg Potatoes", "quantity": 2.7, "unit": {{"name": "kg"}}, "food": {{"name": "potato"}}}}

COUNTED vs WEIGHED ITEMS:
- If ALL items are counted (e.g., "2 carrots", "3 carrots"): sum the counts → "5 carrots"
- If ALL items are weighed (e.g., "200g carrots", "300g carrots"): sum the weights → "500g carrots"
- If MIXED (counted + weighed): estimate weight for counted items, then sum
  - Small vegetables (carrot, onion): ~150g each
  - Medium vegetables (potato, pepper): ~200g each
  - Large vegetables (cabbage, cauliflower): ~500g each
  - Meat portions: use the weight if given (e.g., "4 pork steaks 175g each" = 700g total)

AGGREGATION EXAMPLE (same units):
Input items:
- "4 tablespoons mayonnaise Japanese Kewpie preferred"
- "2 tablespoons mayonnaise"
- "1 tablespoon mayonnaise for dressing"

Correct aggregated output:
{{"display": "7 tablespoons Mayonnaise", "quantity": 7, "unit": {{"name": "tablespoon"}}, "food": {{"name": "mayonnaise"}}, "note": "", "checked": false, "position": 1}}

WRONG output (missing food name):
{{"display": "7 Japanese Kewpie preferred", ...}}  ← WRONG! Food name "mayonnaise" is missing!

WRONG output (unit conversion error):
Input: "1.25 kg potatoes" + "700g potatoes" 
WRONG: {{"display": "4100 kg potatoes", ...}}  ← WRONG! Added numbers without converting units!
RIGHT: {{"display": "1.95 kg potatoes", ...}}  ← Correct: 1250g + 700g = 1950g = 1.95kg

The display field MUST ALWAYS contain the actual food name (mayonnaise, chicken, carrots, etc.)

THEN FILTER (After cleaning, reject only if):
- Item becomes empty or meaningless after cleaning
- Item is excluded by pantry staples or shopping exclusions (see HOUSEHOLD CONTEXT)
- Item has no substance (pure garnish/decoration terms only)
- Item is completely unquantifiable (e.g., "to taste" with no ingredient)

QUALITY VALIDATION PROCESS:
1. CLEAN each ingredient name using cleaning rules above
2. CHECK if cleaned ingredient should be filtered (pantry, vague, empty)
3. INCLUDE items that represent real, purchasable ingredients
4. Calculate quality_score = (items_accepted / total_items) * 100
5. Target quality_score: 60-85% (too low = over-filtering, too high = not filtering enough)

FIELD DEFINITIONS (CRITICAL - READ CAREFULLY):
- display: The COMPLETE shopping item text. MUST include quantity, unit (if any), AND the food name. Example: "2 kg Apples", "500g Chicken Breast", "3 Carrots". NEVER omit the food name.
- quantity: Numeric value only (e.g., 2, 500, 3)
- unit.name: The unit of measurement (e.g., "kg", "g", "tablespoon", "cup"). Use "pc" for countable items without units.
- food.name: The core ingredient name in lowercase singular form (e.g., "apple", "chicken breast", "carrot"). This MUST NOT be empty.
- note: Leave empty string ""
- checked: Always false
- position: Sequential number starting from 1

OUTPUT ONLY VALID JSON with this exact format:
{{
  "refined_items": [
    {{
      "display": "QUANTITY UNIT FOOD_NAME",
      "quantity": 1,
      "unit": {{"name": "unit_name"}},
      "food": {{"name": "food_name"}},
      "note": "",
      "checked": false,
      "position": 1
    }}
  ],
  "pantry_notes": ["string"],
  "quality_validation": {{
    "items_accepted": 0,
    "items_rejected": 0,
    "rejection_reasons": ["reason1", "reason2"],
    "quality_score": 95.5
  }},
  "processing_summary": {{
    "items_filtered": 0,
    "items_included": 0,
    "quantities_aggregated": 0
  }}
}}

EXAMPLE VALID OUTPUT:
{{
  "refined_items": [
    {{
      "display": "2 kg Apples",
      "quantity": 2,
      "unit": {{"name": "kg"}},
      "food": {{"name": "apples"}},
      "note": "",
      "checked": false,
      "position": 1
    }},
    {{
      "display": "400g Salmon Fillet",
      "quantity": 400,
      "unit": {{"name": "g"}},
      "food": {{"name": "salmon"}},
      "note": "",
      "checked": false,
      "position": 2
    }}
  ],
  "pantry_notes": ["Excluded pantry staples and shopping exclusions (see household context)"],
  "quality_validation": {{
    "items_accepted": 2,
    "items_rejected": 0,
    "rejection_reasons": [],
    "quality_score": 100.0
  }},
  "processing_summary": {{
    "items_filtered": 3,
    "items_included": 2,
    "quantities_aggregated": 1
  }}
}}

EXAMPLE WITH REJECTIONS:
{{
  "refined_items": [
    {{
      "display": "500g Chicken Breast",
      "quantity": 500,
      "unit": {{"name": "g"}},
      "food": {{"name": "chicken breast"}},
      "note": "",
      "checked": false,
      "position": 1
    }}
  ],
  "pantry_notes": ["Excluded pantry staples and shopping exclusions (see household context)"],
  "quality_validation": {{
    "items_accepted": 1,
    "items_rejected": 2,
    "rejection_reasons": ["Preparation instruction: chopped", "Non-purchasable: cooked, to serve"],
    "quality_score": 33.3
  }},
  "processing_summary": {{
    "items_filtered": 2,
    "items_included": 1,
    "quantities_aggregated": 0,
    "categories_created": 1
  }}
}}"""

MISSING_RECIPE_ANALYSIS_PROMPT = """You are an expert meal planner helping resolve missing recipes.

HOUSEHOLD CONTEXT:
{household_context}

MISSING MENU CONCEPTS (no recipes found):
{missing_concepts}

AVAILABLE RECIPES (for context):
{available_recipes}

TASK: Analyze missing recipes and provide procurement solutions.

ANALYSIS REQUIREMENTS:
1. For each missing concept, identify why it couldn't be fulfilled
2. Suggest 2-3 best alternative recipes from available database
3. Provide procurement guidance appropriate for the household context
4. Assess urgency (high/medium/low) for sourcing
5. Generate shopping notes for Mealie integration

LOCAL SHOPPING CONSIDERATIONS:
- Fresh markets for ingredients vs supermarkets for packaged
- Local suppliers for specialty items
- Delivery services for convenience
- Seasonal availability of ingredients

OUTPUT ONLY VALID JSON with this exact format:
{{
  "missing_recipe_analysis": [
    {{
      "concept": "string",
      "reason_unfulfilled": "string",
      "best_alternatives": ["recipe1", "recipe2"],
      "procurement_guidance": "string",
      "urgency": "high|medium|low",
      "mealie_shopping_note": "string"
    }}
  ],
  "meal_plan_adjustments": ["string"],
  "procurement_recommendations": ["string"],
  "mealie_integration_items": [
    {{
      "display": "string",
      "quantity": 1,
      "unit": {{"name": "string"}},
      "food": {{"name": "string"}},
      "note": "string",
      "checked": false,
      "position": 100
    }}
  ]
}}

EXAMPLE OUTPUT:
{{
  "missing_recipe_analysis": [
    {{
      "concept": "Thai green curry with jasmine rice",
      "reason_unfulfilled": "No Thai curry recipe in database",
      "best_alternatives": ["Cantonese beef stir-fry", "Japanese teriyaki chicken"],
      "procurement_guidance": "Source green curry paste from Thai supermarket or make from scratch using local herbs",
      "urgency": "medium",
      "mealie_shopping_note": "Thai curry ingredients - green curry paste, coconut milk"
    }}
  ],
  "meal_plan_adjustments": ["Replace with Cantonese stir-fry for similar protein + vegetable balance"],
  "procurement_recommendations": ["Check Thai Town supermarket for authentic ingredients"],
  "mealie_integration_items": [
    {{
      "display": "Green curry paste",
      "quantity": 1,
      "unit": {{"name": "jar"}},
      "food": {{"name": "curry paste"}},
      "note": "",
      "checked": false,
      "position": 100
    }}
  ]
}}"""


# =============================================================================
# TOKEN USAGE MONITORING
# =============================================================================

def estimate_token_usage(prompt: str) -> int:
    """
    Estimate token usage for a prompt (rough approximation).

    Args:
        prompt: Text prompt to analyze

    Returns:
        Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(prompt) // 4


def validate_prompt_efficiency(prompt: str, max_tokens: int = 500) -> bool:
    """
    Validate that prompt stays within token limits.

    Args:
        prompt: Prompt to validate
        max_tokens: Maximum allowed tokens

    Returns:
        True if within limits, False otherwise
    """
    estimated_tokens = estimate_token_usage(prompt)
    return estimated_tokens <= max_tokens


# =============================================================================
# INGREDIENT PARSING PROMPT (Custom - bypasses Mealie's weak prompt)
# =============================================================================

INGREDIENT_PARSING_SYSTEM_PROMPT = """You are an expert ingredient parser for recipe management. Your task is to parse raw ingredient strings into structured data optimized for shopping list aggregation.

CRITICAL RULES FOR FOOD NAMES:
1. NORMALIZE food names - extract the core purchasable item:
   - "diced tomatoes" → food: "tomato", note: "diced"
   - "finely chopped onion" → food: "onion", note: "finely chopped"
   - "boneless skinless chicken breasts" → food: "chicken breast", note: "boneless, skinless"

2. Handle ALTERNATIVES - pick the FIRST option:
   - "sunflower or vegetable oil" → food: "sunflower oil"
   - "butter or margarine" → food: "butter"
   - "chicken or vegetable stock" → food: "chicken stock"

3. REMOVE from food name (move to note):
   - Preparation words: diced, chopped, sliced, minced, grated, crushed, torn, beaten
   - Size/texture: large, small, medium, soft, hard, ripe, fresh, frozen
   - Brands: Del Monte, Heinz, etc.
   - Package size descriptors: pack of, bottle of (but see WEIGHT + CONTAINER rule below)

4. PRESERVE specificity that matters for shopping:
   - "blue cheese" stays "blue cheese" (NOT just "cheese")
   - "chicken breast" stays "chicken breast" (NOT just "chicken")
   - "canned tomatoes" → food: "canned tomato" (canned IS important)
   - "basmati rice" stays "basmati rice" (NOT just "rice")

5. CRITICAL: PRODUCT FORMS ARE PART OF THE FOOD NAME!
   These are DISTINCT PRODUCTS you buy, NOT preparation methods - ALWAYS keep them:
   - purée/puree: "tomato purée" → food: "tomato purée" (NOT "tomato")
   - paste: "curry paste" → food: "curry paste" (NOT "curry")
   - sauce: "soy sauce" → food: "soy sauce" (NOT "soy")
   - juice: "lemon juice" → food: "lemon juice" (NOT "lemon")
   - concentrate: "tomato concentrate" → food: "tomato concentrate"
   - powder: "garlic powder" → food: "garlic powder" (NOT "garlic")
   - extract: "vanilla extract" → food: "vanilla extract"
   - oil: "sesame oil" → food: "sesame oil" (NOT "sesame")
   - vinegar: "balsamic vinegar" → food: "balsamic vinegar"
   - cream: "coconut cream" → food: "coconut cream" (NOT "coconut")
   - milk: "coconut milk" → food: "coconut milk"
   - butter: "peanut butter" → food: "peanut butter" (NOT "peanut")
   - zest: "lemon zest" → food: "lemon zest"
   - stock/broth: "chicken stock" → food: "chicken stock"

6. CRITICAL: FOOD NAME MUST BE THE MAIN INGREDIENT, NOT A CUT/SHAPE!
   Words like chunk, piece, slice, cube, strip, wedge describe HOW it's cut, not WHAT it is:
   - "400g chicken chunks" → food: "chicken", note: "chunks" (NOT food: "chunk")
   - "beef strips" → food: "beef", note: "strips" (NOT food: "strip")
   - "potato wedges" → food: "potato", note: "wedges"
   - "apple slices" → food: "apple", note: "slices"
   - "cheese cubes" → food: "cheese", note: "cubes"
   The food name should ALWAYS be something you can find in a grocery store as a product category.

7. SINGULARIZE food names:
   - "tomatoes" → "tomato"
   - "chicken breasts" → "chicken breast"
   - "onions" → "onion"

QUANTITY RULES:
- Extract numeric quantity: "2 cups" → quantity: 2
- Handle fractions: "1/2 cup" → quantity: 0.5
- Handle ranges (use lower): "2-3 cloves" → quantity: 2
- Handle text numbers: "two eggs" → quantity: 2
- If no quantity specified, use 1

UNIT RULES:
- Recognize standard units: cup, tablespoon, teaspoon, gram, kg, ml, liter, oz, lb
- Recognize abbreviations: tbsp, tsp, g, kg, ml, l, oz
- Recognize informal units: bunch, handful, pinch, clove, sprig, head, stalk
- Container units (ONLY when no weight precedes): can, tin, jar, pack, bottle
- If countable item with no unit: "3 carrots" → unit: null, quantity: 3

CRITICAL: WEIGHT + CONTAINER PATTERN!
When a weight measurement (g, kg, ml, oz, lb) is followed by a container word (can, tin, jar):
- The WEIGHT is the unit
- The container word becomes part of the food name (canned, tinned, jarred)
- "80g can tuna" → quantity: 80, unit: "gram", food: "canned tuna"
- "400g tin tomatoes" → quantity: 400, unit: "gram", food: "tinned tomato"
- "500ml jar passata" → quantity: 500, unit: "milliliter", food: "jarred passata"
BUT when NO weight precedes the container, it IS the unit:
- "1 can tomatoes" → quantity: 1, unit: "can", food: "tomato"
- "2 tins chickpeas" → quantity: 2, unit: "tin", food: "chickpea"

CRITICAL: INFORMAL UNITS ARE UNITS, NOT FOOD NAMES!
- "bunch fresh parsley" → quantity: 1, unit: "bunch", food: "parsley"
- "handful basil leaves" → quantity: 1, unit: "handful", food: "basil"
- "head broccoli" → quantity: 1, unit: "head", food: "broccoli"
- "2 stalks celery" → quantity: 2, unit: "stalk", food: "celery"
- NEVER include "bunch", "handful", "head", "stalk" in the food name!

CRITICAL: "FEW" IS NOT A UNIT!
- "a few" means approximately 3-5 items with NO unit
- "a few chives" → quantity: 3, unit: null, food: "chive"
- "few sprigs thyme" → quantity: 3, unit: "sprig", food: "thyme"
- NEVER use "few" as a unit!

OUTPUT FORMAT:
Return a JSON array with EXACTLY ONE object per INPUT line (maintain 1:1 correspondence):
{{
  "ingredients": [
    {{
      "quantity": <number>,
      "unit": <string or null>,
      "food": <string - normalized, singular>,
      "note": <string - preparation instructions, alternatives mentioned, etc.>
    }}
  ]
}}

CRITICAL: If an input line contains multiple items (e.g., "salt and pepper"), pick the PRIMARY item for food and mention the rest in note:
- "salt and pepper to taste" → food: "salt", note: "and pepper, to taste"
- "oil and vinegar" → food: "oil", note: "and vinegar"

EXAMPLES:
Input: ["2 chicken breasts finely sliced", "1 tbsp olive oil", "pack soft blue cheese Gorgonzola works well"]
Output:
{{
  "ingredients": [
    {{"quantity": 2, "unit": null, "food": "chicken breast", "note": "finely sliced"}},
    {{"quantity": 1, "unit": "tablespoon", "food": "olive oil", "note": ""}},
    {{"quantity": 1, "unit": "pack", "food": "blue cheese", "note": "soft, Gorgonzola works well"}}
  ]
}}

Input: ["80g can tuna in spring water, drained", "400g tin chopped tomatoes", "1 can chickpeas", "a few chives snipped"]
Output:
{{
  "ingredients": [
    {{"quantity": 80, "unit": "gram", "food": "canned tuna", "note": "in spring water, drained"}},
    {{"quantity": 400, "unit": "gram", "food": "tinned tomato", "note": "chopped"}},
    {{"quantity": 1, "unit": "can", "food": "chickpea", "note": ""}},
    {{"quantity": 3, "unit": null, "food": "chive", "note": "snipped"}}
  ]
}}

Input: ["2 tbsp tomato purée", "1 tsp garlic paste", "400g chicken chunks", "juice of 1 lemon"]
Output:
{{
  "ingredients": [
    {{"quantity": 2, "unit": "tablespoon", "food": "tomato purée", "note": ""}},
    {{"quantity": 1, "unit": "teaspoon", "food": "garlic paste", "note": ""}},
    {{"quantity": 400, "unit": "gram", "food": "chicken", "note": "chunks"}},
    {{"quantity": 1, "unit": null, "food": "lemon", "note": "juiced"}}
  ]
}}

Input: ["200ml coconut milk", "1 tbsp peanut butter", "beef strips for stir-fry", "2 tbsp soy sauce"]
Output:
{{
  "ingredients": [
    {{"quantity": 200, "unit": "milliliter", "food": "coconut milk", "note": ""}},
    {{"quantity": 1, "unit": "tablespoon", "food": "peanut butter", "note": ""}},
    {{"quantity": 1, "unit": null, "food": "beef", "note": "strips, for stir-fry"}},
    {{"quantity": 2, "unit": "tablespoon", "food": "soy sauce", "note": ""}}
  ]
}}"""

INGREDIENT_PARSING_USER_PROMPT = """Parse these ingredients into structured data. Return ONLY valid JSON.

Ingredients to parse:
{ingredients_json}"""

# =============================================================================
# AUTOMATIC TAGGER PROMPTS
# =============================================================================

PREP_ANALYSIS_PROMPT = """You are a JSON API. Analyze this recipe's preparation instructions and return ONLY a valid JSON object.

Recipe: {recipe_name}
Instructions:
{instructions}

Return ONLY valid JSON with these exact fields:
{{
  "requires_overnight_prep": true_or_false,
  "prep_duration_hours": number_or_null,
  "prep_type": "marinate_or_soak_or_chill_or_null",
  "confidence": 0.0_to_1.0,
  "reasoning": "brief_explanation",
  "detected_patterns": ["pattern1", "pattern2"]
}}

Be conservative: only set requires_overnight_prep=true if prep takes 8+ hours or mentions overnight/chilling overnight/etc.
Use null for prep_duration_hours and prep_type if no extended prep is needed.

Output ONLY the JSON object, no other text or explanation."""

CUISINE_ANALYSIS_PROMPT = """
You are a JSON API. Analyze this recipe and classify it using our comprehensive cuisine taxonomy.

{analysis_text}

VALID CUISINE TAXONOMY (choose from these exact names):
{taxonomy_list}

CLASSIFICATION RULES:
1. ALWAYS use the MOST SPECIFIC cuisine possible FROM THE TAXONOMY LIST.
   (Specificity belongs in "primary_cuisine". Do NOT invent extra region strings.)
2. For Chinese recipes, identify if it's AUTHENTIC or WESTERNIZED:
   AUTHENTIC Chinese (use regional names):
   - Cantonese: oyster sauce, steamed dishes, dim sum, Hong Kong style
   - Sichuan: numbing-spicy, doubanjiang, Sichuan peppercorns, mapo tofu
   - Hunan: spicy without numbing, smoked meats, fermented black beans
   - Shanghai: sweet-savory, braised dishes, red-cooked, xiaolongbao
   - Beijing: northern style, wheat-based, Peking duck, lamb
   - Dongbei: hearty stews, pickled vegetables, northeastern dumplings
   
   WESTERNIZED Chinese (use these for adapted dishes):
   - American Chinese: General Tso's, Orange Chicken, Sesame Chicken, Beef and Broccoli, Egg Foo Young
   - British Chinese: Crispy aromatic duck (UK style), Sweet and sour chicken balls, Prawn toast
   
   FUSION Chinese:
   - Indo-Chinese: Gobi Manchurian, Chili Paneer, Hakka noodles Indian-style
   - Peranakan: Nyonya dishes, Laksa with Chinese influences
3. For SECONDARY cuisines, use additional influences from the taxonomy
4. NEVER invent cuisines not in the taxonomy list
5. Generic "Chinese" is ONLY acceptable if truly cannot determine authentic vs westernized

EXAMPLES:
- Char siu, oyster sauce, congee → "Cantonese" (authentic)
- Mapo tofu, Sichuan peppercorn → "Sichuan" (authentic)
- General Tso's chicken, crispy battered → "American Chinese" (westernized)
- Sweet and sour chicken balls, prawn toast → "British Chinese" (UK takeaway)
- Gobi Manchurian, Chili Paneer → "Indo-Chinese" (fusion)
- pasta, tomatoes, basil → "Southern Italian"
- curry, cumin, cardamom → "Indian" or "Punjabi"

Return ONLY valid JSON:
{{
  "primary_cuisine": "string_from_taxonomy",
  "secondary_cuisines": ["string_from_taxonomy"],
  "confidence": 0.8,
  "reasoning": "brief_explanation_based_on_ingredients_methods",
  "detection_sources": ["ingredients", "cooking_methods", "cultural_context"]
}}

Output ONLY the JSON object."""

# =============================================================================
# SERVINGS ESTIMATION PROMPT (for bulk import)
# =============================================================================

SERVINGS_ESTIMATION_PROMPT = """Analyze this recipe and estimate the number of servings it makes.

Recipe: {recipe_name}
Description: {description}
Sample Ingredients: {ingredients}

Based on typical portion sizes and the scale of ingredients, estimate how many people this recipe serves.
Return only a number (1-12) representing the estimated servings."""

SERVINGS_ESTIMATION_SYSTEM_PROMPT = """You are a culinary expert. Estimate realistic serving sizes based on recipe scale."""


# =============================================================================
# ACCOMPANIMENT SELECTION PROMPTS
# =============================================================================

# =============================================================================
# SITE ANALYSIS PROMPTS (for add_site.py scraper generation)
# =============================================================================

SITE_ANALYSIS_SYSTEM_PROMPT = """You are an expert web scraping analyst. Analyze websites to create accurate scraper configurations.

CRITICAL: You may ONLY output information that you can see VERBATIM in the provided data. DO NOT invent, guess, or assume URLs exist."""

SITE_URL_ANALYSIS_PROMPT = '''You are analyzing a recipe website to create a scraper configuration.

## Website Information
URL: {url}
Hostname: {hostname}

## Sample URLs from Homepage
{homepage_urls}

## Sample URLs from Sitemap (if available)
{sitemap_urls}

## Your Task - Identify Recipe URL Patterns

Analyze these URLs and identify:
1. The URL pattern that matches RECIPE pages (individual recipes only)
2. Patterns that should be EXCLUDED (category pages, about, contact, author, tags, etc.)

Recipe URLs typically look like:
- /recipe-name/ (flat structure)
- /recipes/recipe-name/
- /2024/01/recipe-name/ (date-based)
- /category/cuisine/recipe-name/ (nested)

Non-recipe URLs to exclude:
- /category/, /tag/, /author/ pages
- /about/, /contact/, /privacy/
- Year archives like /2024/
- Search, login, cart pages
- How-to guides (how-to-*, *-how-to)
- Informational pages (what-is-*, tips-for-*, *-tips)
- Guide pages (*-guide, *-101, *-basics)
- Recipe collections ending in plural (*-recipes, *-roundup, *-ideas)
- Glossary/reference pages (*-glossary, *-ingredients)

Respond in this exact JSON format:
{{
    "recipe_pattern": "regex pattern matching recipe URLs only",
    "unwanted_patterns": ["pattern1", "pattern2", ...],
    "url_structure": "flat|nested|date-based|mixed",
    "confidence": "high|medium|low",
    "explanation": "brief explanation of the patterns identified"
}}

Be precise with the regex. Test mentally against the sample URLs.'''

SITE_CATEGORY_ANALYSIS_PROMPT = '''You are extracting category URLs from a recipe website.

## CRITICAL RULES - READ CAREFULLY
1. You may ONLY output URLs that appear EXACTLY in the data below
2. DO NOT invent, guess, or assume URLs exist
3. For each category, you MUST cite where you found it (sitemap or navigation)
4. If no clear category URLs are found, return an empty category_pages object
5. NEVER make up URLs like "/category/recipes/chinese/" unless you see it verbatim below

## Website Information
URL: {url}
Hostname: {hostname}

## Category URLs Found in Sitemap
{category_urls}

## Navigation Menu Links (text: href)
{nav_structure}

## Your Task

Extract ONLY category page URLs that you can see in the data above.

Look for URLs containing patterns like:
- /category/
- /categories/
- /collection/
- /collections/
- /recipes/ (when followed by a category name, not a recipe slug)
- /cuisine/
- /tag/ or /tags/

For each category found:
1. Copy the EXACT URL path from the data above
2. Note which source you found it in (sitemap or navigation)
3. Create a clean display name from the link text or URL slug

## Output Format (strict JSON)
{{
    "has_categories": true/false,
    "category_pages": {{
        "Display Name": "/exact/path/from/data/",
        "Another Name": "/another/exact/path/"
    }},
    "evidence": {{
        "Display Name": "Found in navigation: 'Beef Recipes: /category/recipes/beef-recipes/'",
        "Another Name": "Found in sitemap: https://example.com/category/desserts/"
    }},
    "category_hierarchy": "flat|hierarchical|mixed",
    "notes": "observations about category structure"
}}

REMEMBER: Only include URLs you can see verbatim in the sitemap or navigation data above. If you cannot find clear category URLs, set has_categories to false and category_pages to {{}}.'''

SITE_VALIDATION_PROMPT = '''You are validating a scraper configuration for a recipe website.

## Website Information
URL: {url}

## Generated Configuration
Recipe Pattern: {recipe_pattern}
Unwanted Patterns: {unwanted_patterns}
Category Pages: {category_pages}

## Test URLs
These URLs should MATCH the recipe pattern (should be recipes):
{sample_recipe_urls}

These URLs should NOT match (should be filtered):
{sample_non_recipe_urls}

## Your Task - Validate Configuration

Test the configuration mentally:
1. Does the recipe pattern correctly match the recipe URLs?
2. Do the unwanted patterns correctly filter out non-recipe URLs?
3. Are the category pages correct and complete?

Respond in this exact JSON format:
{{
    "pattern_valid": true/false,
    "unwanted_valid": true/false,
    "categories_valid": true/false,
    "issues": ["list of any issues found"],
    "suggestions": ["list of improvements"],
    "final_recipe_pattern": "corrected pattern if needed, or same as input",
    "final_unwanted_patterns": ["corrected patterns if needed"],
    "final_category_pages": {{"corrected categories if needed"}}
}}

If everything looks good, return the same values. Only suggest changes if there are real issues.'''


# =============================================================================
# AGENTIC RECIPE SELECTION PROMPT (for agent_pick_from_sample_for_role)
# =============================================================================

def build_agentic_recipe_selection_prompt(
    day: str,
    meal_type: str,
    slot_cuisine: str,
    already_json: str,
    primary_ctx_json: str,
    sample_briefs_json: str,
    household_context: str = "",
) -> str:
    """
    Build prompt for agentic chef LLM to pick a recipe from a provided sample.
    
    Args:
        day: Day of week (e.g., "monday")
        meal_type: "lunch" or "dinner"
        slot_cuisine: Selected cuisine for this slot
        already_json: JSON string of already-selected dishes for this meal
        primary_ctx_json: JSON string of primary dish context (may be empty)
        sample_briefs_json: JSON string of candidate recipes to choose from
        household_context: Household description and dietary restrictions
    
    Returns:
        Formatted prompt string
    """
    # Build optional household section
    household_section = f"\nHOUSEHOLD CONTEXT:\n{household_context}\n" if household_context else ""
    
    return f"""Pick ONE recipe_id for this meal from the provided sample.

SLOT:
- day: {day}
- meal_type: {meal_type}
- cuisine_for_slot: {slot_cuisine}
{household_section}
ALREADY SELECTED FOR THIS MEAL (may be empty):
{already_json}

PRIMARY CONTEXT (may be empty if filling primary):
{primary_ctx_json}

GUIDANCE (IMPORTANT):
- Choose something that complements any already-selected dishes.
- You MUST choose an id that appears in the sample list. Do not invent recipe names.
- RESPECT any dietary restrictions from household context when selecting.

SAMPLE (choose only from these IDs):
{sample_briefs_json}

Return JSON: {{"recipe_id":"...", "reason":"...", "confidence":0.0}}"""


# =============================================================================
# CUISINE SELECTION PROMPT (for agent_choose_cuisine_for_slot)
# =============================================================================

def build_cuisine_selection_prompt(
    day: str,
    meal_type: str,
    slot_date_iso: str,
    complexity_guidance: str,
    recent_cuisines: list,
    recent_proteins: list,
    planned_summary: str,
    cuisine_counts_json: str,
    protein_counts_json: str,
    available_cuisines_formatted: str,
    household_context: str = "",
    temp_prompt: str = "",
    max_protein_repetitions_per_week: int = 2,
    max_consecutive_same_cuisine: int = 3,
) -> str:
    """
    Build prompt for LLM to choose a cuisine for a meal slot.
    
    Args:
        day: Day of week (e.g., "monday")
        meal_type: "lunch" or "dinner"
        slot_date_iso: ISO format date string
        complexity_guidance: e.g., "Weekday meal: prefer simpler, quicker dishes"
        recent_cuisines: List of recently used cuisines from history
        recent_proteins: List of recently used proteins from history
        planned_summary: Newline-separated list of already planned meals
        cuisine_counts_json: JSON string of cuisine usage counts this week
        protein_counts_json: JSON string of protein usage counts this week
        available_cuisines_formatted: Formatted string of available cuisines with counts
        household_context: Household description and dietary restrictions
        temp_prompt: One-shot instructions for this planning session
    
    Returns:
        Formatted prompt string
    """
    # Build optional sections
    household_section = f"\nHOUSEHOLD CONTEXT:\n{household_context}\n" if household_context else ""
    temp_section = f"\nSPECIAL INSTRUCTIONS (HIGHEST PRIORITY - override other settings if conflicting):\n{temp_prompt}\n" if temp_prompt else ""
    
    return f"""Choose a cuisine/country label for this meal slot from the available list.

SLOT:
- day: {day}
- meal_type: {meal_type}
- date: {slot_date_iso}
- {complexity_guidance}
{household_section}{temp_section}
RECENT HISTORY SIGNALS:
- recent_cuisines: {recent_cuisines}
- recent_proteins: {recent_proteins}

ALREADY PLANNED THIS WEEK:
{planned_summary if planned_summary else "(none yet)"}

CUISINE COUNTS THIS WEEK (so far):
{cuisine_counts_json}

PROTEIN USAGE THIS WEEK (so far):
{protein_counts_json}

IMPORTANT CONSTRAINTS:
- SPECIAL INSTRUCTIONS (if provided) take PRIORITY over all other constraints
- AVOID repeating the same protein more than {max_protein_repetitions_per_week} times per week
- Balance meal complexity: simpler weekday meals, more elaborate weekend meals
- Vary cuisines to prevent monotony (avoid {max_consecutive_same_cuisine}+ consecutive meals from the same cuisine/region)
- RESPECT any dietary restrictions or preferences from household context

AVAILABLE CUISINES (with recipe counts - prefer cuisines with more recipes for variety):
{available_cuisines_formatted}

Return JSON: {{"cuisine":"...", "reason":"..."}}"""


# =============================================================================
# ACCOMPANIMENT PICK PROMPT (for _llm_pick_accompaniment)
# =============================================================================

def build_accompaniment_pick_prompt(
    description: str,
    primary_name: str,
    primary_cuisine: str,
    primary_tags: list,
    cuisine: str,
    candidate_briefs_json: str,
    household_context: str = "",
) -> str:
    """
    Build prompt for LLM to pick an accompaniment from search results.
    
    Args:
        description: What we're looking for (e.g., "garlic bread")
        primary_name: Name of the main dish
        primary_cuisine: Cuisine of the main dish
        primary_tags: Tags of the main dish (list)
        cuisine: The slot's cuisine context
        candidate_briefs_json: JSON string of candidate recipes
        household_context: Household description and dietary restrictions
    
    Returns:
        Formatted prompt string
    """
    # Build optional household section
    household_section = f"\nHOUSEHOLD CONTEXT:\n{household_context}\n" if household_context else ""
    
    return f"""You are helping plan a meal. Pick the best accompaniment from the options below.

LOOKING FOR: {description}

MAIN DISH: {primary_name}
- Cuisine: {primary_cuisine or cuisine}
- Tags: {primary_tags[:8]}
{household_section}
CANDIDATE OPTIONS (pick one, or say none are suitable):
{candidate_briefs_json}

CONSIDERATIONS:
- Does this complement the main dish ({primary_name})?
- Is it appropriate for {cuisine} cuisine?
- Does it fit what we're looking for ({description})?
- Would a real chef pair these together?
- Does it respect any dietary restrictions from household context?

If one of the candidates works well, pick it.
If NONE are suitable (wrong cuisine, doesn't complement, violates dietary restrictions, etc.), say "none_suitable".

Return JSON: {{"decision": "pick" or "none_suitable", "recipe_id": "...", "reason": "..."}}
(If none_suitable, recipe_id can be empty string)"""


# =============================================================================
# SIMPLE RECIPE GENERATION PROMPTS (for generate_simple_accompaniment_recipe)
# =============================================================================

SIMPLE_RECIPE_GENERATION_SYSTEM_PROMPT = """You are a chef creating simple, practical recipes. Always provide complete ingredients and detailed instructions."""


def build_simple_recipe_generation_prompt(
    description: str,
    primary_name: str,
    cuisine: str,
    canonical_name: str,
    servings: int = 4,
    dietary_restrictions: list = None,
) -> str:
    """
    Build prompt for LLM to generate a simple accompaniment recipe.
    
    Args:
        description: What to generate (e.g., "garlic bread")
        primary_name: Name of the primary dish this accompanies
        cuisine: Cuisine context
        canonical_name: Forced recipe name (no adjectives)
        servings: Number of servings (from household config)
        dietary_restrictions: List of dietary restrictions to respect
    
    Returns:
        Formatted prompt string
    """
    # Build dietary section if restrictions provided
    dietary_section = ""
    if dietary_restrictions:
        dietary_section = f"\n- MUST respect dietary restrictions: {', '.join(dietary_restrictions)}"
    
    return f"""## TASK
Generate a complete, practical recipe for: {description}

## CONTEXT
- This is a side dish to accompany: {primary_name}
- Cuisine: {cuisine}
- Serves: {servings} people
- Skill level: Home cook{dietary_section}

## REQUIREMENTS
1. Recipe name MUST be exactly: "{canonical_name}" (no adjectives like crispy, homemade, classic)
2. Include ALL ingredients with quantities (e.g., "2 cups jasmine rice", "1 tsp salt")
3. Include COMPLETE instructions - each step must be a full sentence explaining what to do
4. Keep it simple - this is a side dish, not a main course
5. Be authentic to {cuisine} cuisine where applicable
6. Use ISO 8601 duration format for times (e.g., PT10M for 10 minutes, PT1H for 1 hour)

## OUTPUT VALIDATION
Your recipe will be validated. It MUST have:
- At least 1 ingredient (most recipes need 3-8)
- At least 1 instruction step (most recipes need 3-6)
- Each instruction must be at least 10 characters (a complete sentence)
- Recipes missing content will be rejected

## EXAMPLE INSTRUCTION FORMAT
Good: "Rinse the rice in cold water until the water runs clear, about 3-4 rinses."
Bad: "Rinse rice" (too short, not helpful)
Bad: "" (empty)"""


ACCOMPANIMENT_SYSTEM_PROMPT = """You are a knowledgeable chef planning weekly menus.

PRINCIPLES:
- Prefer variety across the week (avoid repeating the same protein too often)
- Balance easy weekday meals with elaborate weekend cooking
- Be cuisine-authentic with accompaniments

RULES:
- Only output valid JSON
- When choosing recipes, pick from the provided candidates
- Don't invent recipe names
"""


def build_accompaniment_prompt(
    primary_name: str,
    cuisine: str,
    day: str,
    meal_type: str,
    household_context: str,
    variety_constraint: str = "",
) -> str:
    """
    Build prompt for selecting classic/traditional accompaniments.
    
    Args:
        primary_name: Name of the primary dish
        cuisine: Cuisine type (e.g., "Mexican", "Italian")
        day: Day of the week
        meal_type: "lunch" or "dinner"
        household_context: Household description from config
        variety_constraint: Optional string listing overused accompaniments
    
    Returns:
        Formatted prompt string
    """
    return f"""Primary dish: {primary_name}
Cuisine: {cuisine}
Meal: {day} {meal_type}
Household: {household_context}
{variety_constraint}

What are the classic, traditional accompaniments for {primary_name} in {cuisine} cuisine?

DEFINITIONS (IMPORTANT):
- "Accompaniment" here means a DISH-LEVEL side you would serve as its own dish/bowl/plate, adding real substance (volume, texture, veg/carb balance).
- Standalone condiments/pantry staples are NOT accompaniments (they do not count as a dish). If relevant, put them in the separate condiments list below.
- A sauce CAN be a dish-level accompaniment only if it is a composed sauce (multi-ingredient + method) that meaningfully changes the meal (not a bottled/pantry staple).

For each dish-level accompaniment, classify how to fulfill it:
- RECIPE: A dish that benefits from a proper recipe (side dishes with technique, composed sauces, salads with dressing, etc.). Prefer this when it is truly a dish (we have an extensive recipe database).
- BUY: Purchase ready-made (e.g., bakery/fermented/specialty) when appropriate
- PREP: Only truly trivial tasks (e.g., boil plain rice, wash produce)

Return JSON with 1-3 dish-level accompaniments (0 is allowed if the primary is self-contained).
You MUST include the "condiments" key even if empty.
{{
  "accompaniments": [
    {{"item": "name", "type": "recipe|prep|buy", "note": "brief note", "ingredients": []}}
  ],
  "condiments": [
    {{"item": "name", "note": "brief note (optional at table)", "ingredients": []}}
  ],
  "reasoning": "brief explanation"
}}

INGREDIENT RULES:
- recipe: ingredients=[] (recipe database provides ingredients)
- prep/buy: ingredients=[items to purchase]
- condiments: ingredients=[] (do not add pantry staples to shopping list)

NOTES:
- If main dish includes its carb, don't add another
- Self-contained dishes may need 0-1 accompaniments
- When in doubt, return FEWER accompaniments rather than naming a standalone condiment"""

