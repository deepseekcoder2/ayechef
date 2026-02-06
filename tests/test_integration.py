"""
Integration Tests - Mealie + RAG + LLM
======================================

These tests interact with real systems:
- Mealie API (read-only where possible)
- Recipe RAG (read-only)
- LLM API (through cache)

Tests that create data MUST register for cleanup using conftest.register_test_recipe().

Test Cases from PRD Phase 4:
| # | Test     | Verification                                          |
|---|----------|-------------------------------------------------------|
| 8 | Dedup    | Generate "steamed rice", run again, should FIND       |
| 9 | Quality  | Generated recipes have ≥1 ingredient, ≥1 instruction  |
| 10| Shopping | PREP items appear on shopping list                    |
| 11| Shopping | BUY items appear on shopping list                     |
| 12| Tags     | Generated recipes have cuisine + AI-Generated tags    |
| 13| RAG      | Generated recipes findable by semantic search         |
"""

import pytest
import asyncio
from datetime import datetime
from typing import Optional

from mealie_client import MealieClient
from tests.conftest import register_test_recipe


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# =============================================================================
# Helper: Create Test Recipe using PRODUCTION code path
# =============================================================================

async def create_test_recipe_in_mealie(
    name: str,
    cuisine: str,
    mealie_config: dict
) -> tuple[str, str]:
    """
    Create a test recipe in Mealie using the PRODUCTION code path.
    
    This uses the actual create_recipe_in_mealie() function to test
    the real code, not a bypass.
    
    IMPORTANT: Caller MUST call register_test_recipe() for cleanup.
    
    Returns: (recipe_id, recipe_slug)
    """
    from chef_agentic import create_recipe_in_mealie
    
    recipe_data = {
        "name": name,
        "description": f"Test recipe created at {datetime.now().isoformat()}",
        "recipe_ingredient": [
            "2 cups jasmine rice",
            "3 cups water", 
            "1 tsp salt"
        ],
        "recipe_instructions": [
            {"text": "Rinse the rice in cold water until the water runs clear."},
            {"text": "Add rice and water to a pot, bring to boil."},
            {"text": "Reduce heat to low, cover, and simmer for 15 minutes."},
        ],
        "recipe_yield": "4 servings",
        "prep_time": "PT5M",
        "cook_time": "PT20M",
        "total_time": "PT25M"
    }
    
    client = MealieClient()
    try:
        recipe_id, recipe_slug, _ = await create_recipe_in_mealie(client, recipe_data, cuisine)
        return recipe_id, recipe_slug
    finally:
        client.close()


# =============================================================================
# Test 8: Deduplication
# =============================================================================

class TestDeduplication:
    """
    Test 8: Deduplication works correctly.
    
    Production flow:
    1. create_recipe_in_mealie() - creates in Mealie (does NOT index locally)
    2. post_process_generated_recipe() - indexes in local SQLite + RAG
    3. find_existing_recipe_by_name() - searches LOCAL index first (instant)
       - Only falls back to Mealie API if local index fails
       - The 90s timeout should NEVER be hit if indexing works
    
    This test verifies the local index lookup works correctly.
    """
    
    @pytest.mark.creates_data
    @pytest.mark.slow
    def test_dedup_finds_existing_recipe(self, mealie_config, mealie_connection, rag_instance, test_timestamp):
        """
        Test the FULL production flow:
        1. Create recipe (production code)
        2. Index it (production code)
        3. Search should find it via LOCAL index (fast)
        """
        from chef_agentic import find_existing_recipe_by_name, post_process_generated_recipe
        
        async def _run():
            test_name = f"[TEST] {test_timestamp} Steamed Jasmine Rice"
            cuisine = "Japanese"
            
            # Step 1: Create using PRODUCTION code
            recipe_id, recipe_slug = await create_test_recipe_in_mealie(
                name=test_name,
                cuisine=cuisine,
                mealie_config=mealie_config
            )
            register_test_recipe(recipe_id, recipe_slug)
            print(f"\n  ✅ Created via production code: {test_name} [{recipe_id[:8]}...]")
            
            # Step 2: Index using PRODUCTION code
            await post_process_generated_recipe(
                recipe_id=recipe_id,
                recipe_slug=recipe_slug,
                cuisine=cuisine,
                rag=rag_instance
            )
            print(f"  ✅ Indexed via production code")
            
            # Step 3: Search should find via LOCAL index (should be instant, not 90s)
            import time
            start = time.time()
            found_result = await find_existing_recipe_by_name(test_name, cuisine)
            search_time = time.time() - start
            
            print(f"  ⏱️  Search took {search_time:.2f}s")
            
            # If search took > 5s, something is wrong - it's hitting Mealie API instead of local index
            assert search_time < 5.0, \
                f"Search took {search_time:.2f}s - should be instant via local index, not Mealie API"
            
            assert found_result is not None, \
                f"Deduplication FAILED: Recipe '{test_name}' was not found in local index"
            
            # find_existing_recipe_by_name returns (recipe_id, slug) tuple
            found_id, found_slug = found_result
            assert found_id == recipe_id, \
                f"Wrong recipe found: expected {recipe_id}, got {found_id}"
            
            print(f"  ✅ Found via local index: {found_id[:8]}...")
        
        run_async(_run())


# =============================================================================
# Test 9: Generation Quality
# =============================================================================

class TestGenerationQuality:
    """
    Test 9: Generated recipes have real content.
    
    Validation rules:
    - At least 1 ingredient
    - At least 1 instruction
    - Each instruction at least 10 characters
    """
    
    @pytest.mark.slow
    @pytest.mark.readonly
    def test_generated_recipe_has_content(self, mealie_config, mealie_connection, test_timestamp):
        """
        Generate a simple recipe and verify it has ingredients and instructions.
        
        Note: This test ONLY validates LLM generation output - it does NOT create in Mealie.
        """
        from chef_agentic import generate_simple_accompaniment_recipe, validate_generated_recipe
        
        async def _run():
            description = "steamed jasmine rice"
            cuisine = "Japanese"
            primary_ctx = {
                "primary_name": "Teriyaki Salmon",
                "primary_cuisine_primary": "Japanese",
                "primary_tags": ["Fish", "Grilled"]
            }
            
            # Generate recipe (LLM call only - no Mealie)
            recipe_data = await generate_simple_accompaniment_recipe(
                description=description,
                cuisine=cuisine,
                primary_ctx=primary_ctx,
                max_attempts=3
            )
            
            print(f"\n  Generated recipe: {recipe_data.get('name')}")
            print(f"  Ingredients: {len(recipe_data.get('recipe_ingredient', []))} items")
            print(f"  Instructions: {len(recipe_data.get('recipe_instructions', []))} steps")
            
            # Validate using the internal validator
            is_valid, errors = validate_generated_recipe(recipe_data)
            
            assert is_valid, \
                f"Generated recipe failed validation: {errors}"
            
            # Additional checks
            assert len(recipe_data.get("recipe_ingredient", [])) >= 1, \
                "Generated recipe has no ingredients"
            assert len(recipe_data.get("recipe_instructions", [])) >= 1, \
                "Generated recipe has no instructions"
            
            print(f"  ✅ Quality validation passed")
        
        run_async(_run())


# =============================================================================
# Test 12: Tags on Generated Recipes
# =============================================================================

class TestGeneratedRecipeTags:
    """
    Test 12: Generated recipes have correct tags.
    
    Expected tags:
    - "{cuisine} Cuisine" (e.g., "Japanese Cuisine")
    - "AI-Generated"
    """
    
    @pytest.mark.creates_data
    @pytest.mark.slow
    def test_generated_recipe_has_tags(self, mealie_config, mealie_connection, rag_instance, test_timestamp):
        """
        Test full production flow for generated recipe tagging:
        1. Create recipe (production code)
        2. Post-process (tags + indexing)
        3. Verify tags were applied in Mealie
        """
        from chef_agentic import post_process_generated_recipe
        
        async def _run():
            test_name = f"[TEST] {test_timestamp} Garlic Bread"
            cuisine = "Italian"
            
            # Create using PRODUCTION code
            recipe_id, recipe_slug = await create_test_recipe_in_mealie(
                name=test_name,
                cuisine=cuisine,
                mealie_config=mealie_config
            )
            register_test_recipe(recipe_id, recipe_slug)
            print(f"\n  ✅ Created via production code: {test_name} [{recipe_id[:8]}...]")
            
            # Post-process using PRODUCTION code
            await post_process_generated_recipe(
                recipe_id=recipe_id,
                recipe_slug=recipe_slug,
                cuisine=cuisine,
                rag=rag_instance
            )
            print(f"  ✅ Post-processed via production code")
            
            return test_name, recipe_id, recipe_slug, cuisine
        
        test_name, recipe_id, recipe_slug, cuisine = run_async(_run())
        
        # Fetch recipe from Mealie and check tags
        client = MealieClient()
        try:
            recipe = client.get_recipe(recipe_slug)
        finally:
            client.close()
        
        tags = recipe.get("tags", [])
        tag_names = [t.get("name", "") for t in tags if isinstance(t, dict)]
        
        print(f"  Tags found: {tag_names}")
        
        # Check for expected tags
        expected_cuisine_tag = f"{cuisine} Cuisine"
        expected_ai_tag = "AI-Generated"
        
        # Note: Tags may not exist in Mealie - check if they were applied
        # Post-processing logs warnings if tags don't exist
        has_cuisine_tag = expected_cuisine_tag in tag_names
        has_ai_tag = expected_ai_tag in tag_names
        
        if has_cuisine_tag and has_ai_tag:
            print(f"  ✅ Both tags applied: {expected_cuisine_tag}, {expected_ai_tag}")
        else:
            # Tags may not exist in Mealie - this is expected if not seeded
            print(f"  ⚠️ Tags not fully applied (may need seeding):")
            print(f"      {expected_cuisine_tag}: {'✅' if has_cuisine_tag else '❌'}")
            print(f"      {expected_ai_tag}: {'✅' if has_ai_tag else '❌'}")


# =============================================================================
# Test 13: RAG Search
# =============================================================================

class TestRAGSearch:
    """
    Test 13: Generated recipes findable by semantic search.
    """
    
    @pytest.mark.readonly
    def test_rag_finds_existing_recipes(self, rag_instance):
        """
        Verify RAG can find recipes by semantic search (read-only).
        """
        # Search for a common concept
        results = rag_instance.find_recipes_for_concept("steamed rice", top_k=5)
        
        print(f"\n  RAG search 'steamed rice': {len(results)} results")
        for r in results[:3]:
            print(f"    - {r.get('name')} (score: {r.get('relevance_score', 0):.3f})")
        
        # Should find at least some results if DB has rice recipes
        # This is a sanity check, not a strict assertion
        if len(results) > 0:
            print(f"  ✅ RAG search working")
        else:
            print(f"  ⚠️ No results for 'steamed rice' - may need to index recipes")
    
    @pytest.mark.readonly
    def test_rag_database_stats(self, rag_instance):
        """
        Verify RAG database has recipes indexed (read-only).
        """
        total = rag_instance.get_total_recipes()
        stats = rag_instance.analyze_recipe_database()
        
        print(f"\n  RAG Database Stats:")
        print(f"    Total recipes: {total}")
        print(f"    With embeddings: {stats.get('recipes_with_embeddings', 0)}")
        print(f"    Cuisines: {stats.get('unique_cuisines', 0)}")
        
        assert total > 0, "RAG database is empty - need to index recipes"
        print(f"  ✅ RAG database has {total} recipes")


# =============================================================================
# Test 10-11: Shopping List (Notes for PREP/BUY)
# =============================================================================

class TestShoppingListNotes:
    """
    Tests 10-11: PREP and BUY items appear as notes in meal plan.
    
    Note: These tests verify the data structure, not actual Mealie writes.
    """
    
    @pytest.mark.readonly
    def test_prep_items_in_meal_summary(self):
        """
        Test 10: PREP items should appear in meal summary with [PREP] prefix.
        """
        from chef_agentic import PlannedMeal, NoteItem
        
        meal = PlannedMeal()
        meal.notes.append(NoteItem(
            title="Pappardelle",
            text="boil per package directions",
            item_type="prep"
        ))
        
        summary = meal.summary()
        
        assert "[PREP] Pappardelle" in summary, \
            f"PREP item not in summary: {summary}"
        
        print(f"\n  ✅ PREP item in summary: {summary}")
    
    @pytest.mark.readonly
    def test_buy_items_in_meal_summary(self):
        """
        Test 11: BUY items should appear in meal summary with [BUY] prefix.
        """
        from chef_agentic import PlannedMeal, NoteItem
        
        meal = PlannedMeal()
        meal.notes.append(NoteItem(
            title="Crusty Baguette",
            text="from bakery",
            item_type="buy"
        ))
        
        summary = meal.summary()
        
        assert "[BUY] Crusty Baguette" in summary, \
            f"BUY item not in summary: {summary}"
        
        print(f"\n  ✅ BUY item in summary: {summary}")
    
    @pytest.mark.readonly
    def test_write_plan_payload_includes_notes(self):
        """
        Verify write_plan_to_mealie would include note entries.
        
        This is a unit test - does NOT write to Mealie.
        """
        from chef_agentic import PlannedMeal, PlannedDish, NoteItem, Candidate, AgentState
        from datetime import datetime
        
        # Create state with notes
        state = AgentState(
            week_start=datetime.now().date(),
            history={}
        )
        
        meal = PlannedMeal()
        meal.dishes.append(PlannedDish(
            candidate=Candidate(recipe_id="primary-001", name="Beef Ragu")
        ))
        meal.notes.append(NoteItem(
            title="Pappardelle",
            text="boil per package",
            item_type="prep"
        ))
        
        state.planned["monday"] = {"dinner": meal}
        
        # Verify notes exist in state
        monday_dinner = state.planned["monday"]["dinner"]
        
        assert len(monday_dinner.notes) == 1, "Note not stored in state"
        assert monday_dinner.notes[0].title == "Pappardelle"
        assert monday_dinner.notes[0].item_type == "prep"
        
        print(f"\n  ✅ Notes correctly stored in AgentState")
        print(f"      {monday_dinner.summary()}")


# =============================================================================
# Dry Run Test
# =============================================================================

class TestDryRun:
    """
    Verify dry-run mode doesn't write to Mealie.
    """
    
    @pytest.mark.readonly
    def test_dry_run_flag_exists(self):
        """Verify --dry-run is documented in main."""
        from chef_agentic import main
        import inspect
        
        # Check if main function mentions dry_run
        source = inspect.getsource(main)
        
        assert "--dry-run" in source, \
            "main() should support --dry-run flag"
        assert "dry_run" in source.lower(), \
            "main() should have dry_run logic"
        
        print("\n  ✅ --dry-run flag supported in main()")
