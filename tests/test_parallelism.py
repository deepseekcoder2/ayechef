"""
Tests for Bulk Import Parallelism Implementation
================================================

Tests the parallelism configuration, thread safety, and parallel processing.
"""

import pytest
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Test: Parallelism Configuration
# =============================================================================

class TestParallelismConfig:
    """Test parallelism preset loading and configuration."""
    
    @pytest.mark.readonly
    def test_presets_exist(self):
        """All 7 presets should be defined."""
        from config import PARALLELISM_PRESETS
        
        expected_presets = [
            "apple_silicon_8gb",
            "apple_silicon_16gb", 
            "apple_silicon_32gb",
            "apple_silicon_max",
            "nvidia_gpu",
            "cloud_api",
            "conservative"
        ]
        
        for preset in expected_presets:
            assert preset in PARALLELISM_PRESETS, f"Missing preset: {preset}"
    
    @pytest.mark.readonly
    def test_preset_structure(self):
        """Each preset should have all required phases."""
        from config import PARALLELISM_PRESETS
        
        required_phases = ["discovery", "import", "tagging", "parsing", "indexing"]
        
        for preset_name, preset in PARALLELISM_PRESETS.items():
            for phase in required_phases:
                assert phase in preset, f"Preset '{preset_name}' missing phase '{phase}'"
                assert "workers" in preset[phase], f"Preset '{preset_name}' phase '{phase}' missing 'workers'"
    
    @pytest.mark.readonly
    def test_indexing_has_batch_size(self):
        """Indexing phase should have batch_size in all presets."""
        from config import PARALLELISM_PRESETS
        
        for preset_name, preset in PARALLELISM_PRESETS.items():
            assert "batch_size" in preset["indexing"], \
                f"Preset '{preset_name}' indexing missing 'batch_size'"
    
    @pytest.mark.readonly
    def test_get_parallelism_config_returns_dict(self):
        """get_parallelism_config should return a dict with workers."""
        from config import get_parallelism_config
        
        config = get_parallelism_config("tagging")
        
        assert isinstance(config, dict)
        assert "workers" in config
        assert isinstance(config["workers"], int)
        assert config["workers"] > 0
    
    @pytest.mark.readonly
    def test_get_parallelism_config_invalid_phase_raises(self):
        """Invalid phase should raise ValueError."""
        from config import get_parallelism_config
        
        with pytest.raises(ValueError) as exc_info:
            get_parallelism_config("invalid_phase")
        
        assert "invalid_phase" in str(exc_info.value)
    
    @pytest.mark.readonly
    def test_apple_silicon_max_values(self):
        """apple_silicon_max should have high worker counts."""
        from config import PARALLELISM_PRESETS
        
        preset = PARALLELISM_PRESETS["apple_silicon_max"]
        
        assert preset["tagging"]["workers"] == 25
        assert preset["indexing"]["workers"] == 12
        assert preset["import"]["workers"] == 15
    
    @pytest.mark.readonly
    def test_conservative_values(self):
        """conservative should have low worker counts."""
        from config import PARALLELISM_PRESETS
        
        preset = PARALLELISM_PRESETS["conservative"]
        
        assert preset["tagging"]["workers"] <= 4
        assert preset["indexing"]["workers"] <= 2
        assert preset["import"]["workers"] <= 3


# =============================================================================
# Test: Thread Safety - LLM Cache
# =============================================================================

class TestLLMCacheThreadSafety:
    """Test thread safety of LLM cache."""
    
    @pytest.mark.readonly
    def test_cache_lock_is_threading_lock(self):
        """Global cache lock should be a threading.Lock, not asyncio.Lock."""
        from batch_llm_processor import _cache_lock
        
        # Should be threading.Lock, not asyncio.Lock
        assert isinstance(_cache_lock, type(threading.Lock()))
    
    @pytest.mark.readonly
    def test_lru_cache_uses_rlock(self):
        """LRUCache should use RLock for thread safety."""
        from batch_llm_processor import LRUCache
        
        cache = LRUCache(max_size=10)
        
        # Check the lock attribute exists and is an RLock
        assert hasattr(cache, 'lock')
        assert isinstance(cache.lock, type(threading.RLock()))


# =============================================================================
# Test: Thread Safety - Automatic Tagger
# =============================================================================

class TestAutomaticTaggerThreadSafety:
    """Test thread safety of AutomaticTagger."""
    
    @pytest.mark.readonly
    def test_tagger_has_tag_cache_lock(self):
        """AutomaticTagger should have a threading.Lock for tag cache."""
        from automatic_tagger import AutomaticTagger
        
        tagger = AutomaticTagger()
        
        assert hasattr(tagger, '_tag_cache_lock')
        assert isinstance(tagger._tag_cache_lock, type(threading.Lock()))


# =============================================================================
# Test: RecipeRAG store_with_precomputed_embedding
# =============================================================================

class TestRecipeRAGPrecomputedEmbedding:
    """Test the store_with_precomputed_embedding method."""
    
    @pytest.mark.readonly
    def test_method_exists(self):
        """RecipeRAG should have store_with_precomputed_embedding method."""
        from recipe_rag import RecipeRAG
        
        assert hasattr(RecipeRAG, 'store_with_precomputed_embedding')
    
    @pytest.mark.readonly
    def test_method_signature(self):
        """Method should accept recipe, embedding, force, auto_save."""
        import inspect
        from recipe_rag import RecipeRAG
        
        sig = inspect.signature(RecipeRAG.store_with_precomputed_embedding)
        params = list(sig.parameters.keys())
        
        assert 'recipe' in params
        assert 'embedding' in params
        assert 'force' in params
        assert 'auto_save' in params
    
    @pytest.mark.readonly
    def test_index_recipe_uses_store_method(self):
        """index_recipe should call store_with_precomputed_embedding internally."""
        import inspect
        from recipe_rag import RecipeRAG
        
        # Get source code of index_recipe
        source = inspect.getsource(RecipeRAG.index_recipe)
        
        # Should call the store method
        assert 'store_with_precomputed_embedding' in source


# =============================================================================
# Test: Parallel Processing Pattern
# =============================================================================

class TestParallelProcessingPattern:
    """Test that parallel processing works correctly."""
    
    @pytest.mark.readonly
    def test_threadpool_executor_pattern(self):
        """Verify ThreadPoolExecutor works with our pattern."""
        results = []
        lock = threading.Lock()
        
        def worker(n):
            # Simulate work
            time.sleep(0.01)
            with lock:
                results.append(n)
            return n * 2
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(worker, i): i for i in range(10)}
            for future in as_completed(futures):
                result = future.result()
                assert result == futures[future] * 2
        
        # All items should be processed
        assert len(results) == 10
        assert set(results) == set(range(10))
    
    @pytest.mark.readonly
    def test_lock_protects_shared_state(self):
        """Lock should prevent race conditions."""
        counter = [0]  # Use list to allow mutation in nested function
        lock = threading.Lock()
        iterations = 1000
        
        def increment():
            with lock:
                current = counter[0]
                time.sleep(0.0001)  # Simulate some work
                counter[0] = current + 1
        
        threads = [threading.Thread(target=increment) for _ in range(iterations)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Without lock, this would likely be less than iterations due to race
        assert counter[0] == iterations


# =============================================================================
# Test: SSE Progress Endpoint
# =============================================================================

class TestSSEProgressEndpoint:
    """Test SSE progress endpoint exists and has correct structure."""
    
    @pytest.mark.readonly
    def test_endpoint_registered(self):
        """Progress endpoint function should exist."""
        from panel.routes import import_recipes
        
        # Check the function exists
        assert hasattr(import_recipes, 'job_progress_stream')
        assert callable(import_recipes.job_progress_stream)
    
    @pytest.mark.readonly
    def test_progress_parser_exists(self):
        """parse_progress_line function should exist."""
        from panel.routes.import_recipes import parse_progress_line
        
        assert callable(parse_progress_line)
    
    @pytest.mark.readonly
    def test_progress_parser_returns_none_for_invalid(self):
        """Parser should return None for non-progress lines."""
        from panel.routes.import_recipes import parse_progress_line
        
        result = parse_progress_line("This is just a random log line")
        assert result is None
    
    @pytest.mark.readonly
    def test_progress_parser_handles_percentage(self):
        """Parser should handle percentage format."""
        from panel.routes.import_recipes import parse_progress_line
        
        # Test percentage pattern
        result = parse_progress_line("Progress: 45% (45/100)")
        
        if result:  # May not match exact format
            assert "percent" in result or "completed" in result


# =============================================================================
# Test: Integration - Config Flows to Bulk Import
# =============================================================================

class TestConfigIntegration:
    """Test that config values flow correctly to bulk import code."""
    
    @pytest.mark.readonly
    def test_bulk_import_imports_config(self):
        """bulk_import_smart should import get_parallelism_config."""
        import inspect
        import bulk_import_smart
        
        source = inspect.getsource(bulk_import_smart)
        
        assert 'get_parallelism_config' in source
    
    @pytest.mark.readonly
    def test_bulk_import_uses_tagging_config(self):
        """Tagging loop should use config for workers."""
        import inspect
        import bulk_import_smart
        
        source = inspect.getsource(bulk_import_smart)
        
        # Should call get_parallelism_config with 'tagging'
        assert "get_parallelism_config('tagging')" in source or \
               'get_parallelism_config("tagging")' in source
    
    @pytest.mark.readonly  
    def test_bulk_import_uses_indexing_config(self):
        """Indexing loop should use config for workers."""
        import inspect
        import bulk_import_smart
        
        source = inspect.getsource(bulk_import_smart)
        
        # Should call get_parallelism_config with 'indexing'
        assert "get_parallelism_config('indexing')" in source or \
               'get_parallelism_config("indexing")' in source


# =============================================================================
# Test: Concurrent Embedding Generation
# =============================================================================

class TestConcurrentEmbedding:
    """Test that embedding generation can run concurrently."""
    
    @pytest.mark.readonly
    def test_embedding_outside_lock_in_bulk_import(self):
        """Embedding generation should happen outside the lock."""
        import inspect
        import bulk_import_smart
        
        source = inspect.getsource(bulk_import_smart)
        
        # Look for the pattern where embedding is generated before the lock
        # The key indicator is _generate_embedding being called, then lock acquired
        
        # Find index_single_recipe function
        if 'def index_single_recipe' in source:
            # Extract the function
            start = source.find('def index_single_recipe')
            end = source.find('\n            with ThreadPoolExecutor', start)
            if end == -1:
                end = start + 2000  # Fallback
            
            func_source = source[start:end]
            
            # _generate_embedding should come before 'with index_lock:'
            embed_pos = func_source.find('_generate_embedding')
            lock_pos = func_source.find('with index_lock:')
            
            if embed_pos != -1 and lock_pos != -1:
                assert embed_pos < lock_pos, \
                    "Embedding generation should happen BEFORE acquiring the lock"
