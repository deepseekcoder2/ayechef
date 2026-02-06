"""
Pytest Configuration and Fixtures
=================================

Provides shared fixtures for the test suite:
- Read-only Mealie connection
- RAG instance (read-only)
- LLM cache (uses real API but cached)
- Cleanup registry for created test recipes

SAFETY: All fixtures are designed to be non-destructive.
Any test that creates data MUST register it for cleanup.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from mealie_client import MealieClient


# =============================================================================
# SAFETY: Test Recipe Cleanup Registry
# =============================================================================

_created_recipe_ids: List[str] = []
_created_recipe_slugs: List[str] = []


def register_test_recipe(recipe_id: str, recipe_slug: str = None):
    """Register a recipe for cleanup after tests."""
    if recipe_id and recipe_id not in _created_recipe_ids:
        _created_recipe_ids.append(recipe_id)
    if recipe_slug and recipe_slug not in _created_recipe_slugs:
        _created_recipe_slugs.append(recipe_slug)


def get_registered_recipes() -> List[str]:
    """Get list of recipe IDs registered for cleanup."""
    return _created_recipe_ids.copy()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def mealie_config():
    """Load Mealie configuration (read-only)."""
    from config import MEALIE_URL, MEALIE_TOKEN
    
    if not MEALIE_TOKEN:
        pytest.skip("MEALIE_TOKEN not configured - skipping integration tests")
    
    return {
        "url": MEALIE_URL,
    }


@pytest.fixture(scope="session")
def mealie_connection(mealie_config):
    """
    Verify Mealie connection is working (read-only check).
    
    This fixture validates we can connect to Mealie without modifying anything.
    """
    client = MealieClient()
    try:
        # Try to fetch recipes as a connectivity test
        client.get_all_recipes()
        return True
    except Exception as e:
        pytest.skip(f"Cannot connect to Mealie: {e}")
    finally:
        client.close()


@pytest.fixture(scope="session")
def rag_instance():
    """
    Get RecipeRAG instance (read-only).
    
    Uses existing recipe_index.db - does NOT create or modify recipes.
    """
    from recipe_rag import RecipeRAG
    return RecipeRAG()


@pytest.fixture(scope="function")
async def llm_cache():
    """Get LLM cache for tests (uses real API with caching)."""
    from batch_llm_processor import get_llm_cache
    return await get_llm_cache()


@pytest.fixture(scope="session")
def test_timestamp():
    """Generate unique timestamp for test recipe names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# Session Cleanup
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_recipes(mealie_config):
    """
    CRITICAL: Clean up any test recipes created during the session.
    
    This runs AFTER all tests complete to ensure no test data remains.
    """
    yield  # Run all tests first
    
    # Cleanup phase
    if not _created_recipe_ids:
        return
    
    print(f"\n🧹 Cleaning up {len(_created_recipe_ids)} test recipes...")
    
    client = MealieClient()
    try:
        for recipe_id in _created_recipe_ids:
            try:
                success = client.delete_recipe(recipe_id)
                if success:
                    print(f"   ✅ Deleted test recipe: {recipe_id[:8]}...")
                else:
                    print(f"   ⚠️ Failed to delete {recipe_id[:8]}...")
            except Exception as e:
                print(f"   ❌ Error deleting {recipe_id[:8]}...: {e}")
    finally:
        client.close()
    
    _created_recipe_ids.clear()
    _created_recipe_slugs.clear()


# =============================================================================
# Test Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "readonly: marks test as read-only (no database writes)"
    )
    config.addinivalue_line(
        "markers", "creates_data: marks test as creating data (requires cleanup)"
    )
    config.addinivalue_line(
        "markers", "slow: marks test as slow (may take >10 seconds)"
    )
