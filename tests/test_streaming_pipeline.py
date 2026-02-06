"""
Tests for Phase 3 Streaming Pipeline
=====================================

Tests the streaming pipeline components including:
- PipelineState state management
- Worker functions (parse, tag, index)
- streaming_bulk_import function
"""

import pytest
import threading
import time
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# Test: PipelineState
# =============================================================================

class TestPipelineState:
    """Test PipelineState state management."""
    
    @pytest.fixture
    def state(self):
        """Create a fresh PipelineState for testing."""
        from panel.jobs.pipeline_state import PipelineState
        job_id = f"test-{int(time.time() * 1000)}"
        state = PipelineState(job_id)
        yield state
        # Cleanup
        if state._state_file.exists():
            state._state_file.unlink()
    
    @pytest.fixture
    def temp_state(self, tmp_path):
        """Create a PipelineState with a temporary directory for isolation."""
        from panel.jobs.pipeline_state import PipelineState, JOBS_DIR
        
        # Create a unique job_id to avoid conflicts
        job_id = f"test-temp-{int(time.time() * 1000)}"
        state = PipelineState(job_id)
        yield state
        # Cleanup
        if state._state_file.exists():
            state._state_file.unlink()
    
    @pytest.mark.readonly
    def test_init_creates_empty_state(self, state):
        """PipelineState should initialize with empty results."""
        assert state.results == {}
        assert state.job_id.startswith("test-")
    
    @pytest.mark.readonly
    def test_add_urls_no_duplicates(self, state):
        """add_urls should skip existing URLs."""
        urls = [
            "https://example.com/recipe1",
            "https://example.com/recipe2",
            "https://example.com/recipe1",  # Duplicate
        ]
        
        state.add_urls(urls)
        
        # Should only have 2 unique URLs
        assert len(state.results) == 2
        assert "https://example.com/recipe1" in state.results
        assert "https://example.com/recipe2" in state.results
    
    @pytest.mark.readonly
    def test_add_urls_preserves_existing(self, state):
        """add_urls should not overwrite existing URLs."""
        url = "https://example.com/recipe1"
        
        # Add URL first time
        state.add_urls([url])
        state.update_recipe(url, status='imported', slug='recipe-1')
        
        # Add same URL again
        state.add_urls([url])
        
        # Original data should be preserved
        assert state.results[url].status == 'imported'
        assert state.results[url].slug == 'recipe-1'
    
    @pytest.mark.readonly
    def test_add_urls_sets_pending_status(self, state):
        """add_urls should set initial status to pending."""
        url = "https://example.com/recipe1"
        
        state.add_urls([url])
        
        assert state.results[url].status == 'pending'
        assert state.results[url].phase == 'discovery'
    
    @pytest.mark.readonly
    def test_update_recipe_creates_if_missing(self, state):
        """update_recipe should create recipe if URL not in results."""
        url = "https://example.com/new-recipe"
        
        state.update_recipe(url, status='importing', slug='new-recipe')
        
        assert url in state.results
        assert state.results[url].status == 'importing'
        assert state.results[url].slug == 'new-recipe'
    
    @pytest.mark.readonly
    def test_update_recipe_updates_existing(self, state):
        """update_recipe should update existing recipe."""
        url = "https://example.com/recipe1"
        state.add_urls([url])
        
        state.update_recipe(url, status='parsed', quality='GOOD')
        
        assert state.results[url].status == 'parsed'
        assert state.results[url].quality == 'GOOD'
    
    @pytest.mark.readonly
    def test_update_recipe_sets_started_at(self, state):
        """update_recipe should set started_at for in-progress statuses."""
        url = "https://example.com/recipe1"
        state.add_urls([url])
        
        # Initially no started_at
        assert state.results[url].started_at is None
        
        # Update to importing status
        state.update_recipe(url, status='importing')
        
        # Should have started_at timestamp
        assert state.results[url].started_at is not None
    
    @pytest.mark.readonly
    def test_update_recipe_sets_completed_at(self, state):
        """update_recipe should set completed_at for terminal statuses."""
        url = "https://example.com/recipe1"
        state.add_urls([url])
        
        # Initially no completed_at
        assert state.results[url].completed_at is None
        
        # Update to indexed status
        state.update_recipe(url, status='indexed')
        
        # Should have completed_at timestamp
        assert state.results[url].completed_at is not None
    
    @pytest.mark.readonly
    def test_get_pending_for_phase_import(self, state):
        """get_pending_for_phase('import') returns URLs with status='pending'."""
        urls = [
            "https://example.com/recipe1",
            "https://example.com/recipe2",
            "https://example.com/recipe3",
        ]
        state.add_urls(urls)
        
        # Update some recipes to different statuses
        state.update_recipe(urls[1], status='imported')
        state.update_recipe(urls[2], status='parsed')
        
        pending = state.get_pending_for_phase('import')
        
        # Only recipe1 should be pending for import
        assert len(pending) == 1
        assert urls[0] in pending
    
    @pytest.mark.readonly
    def test_get_pending_for_phase_parsing(self, state):
        """get_pending_for_phase('parsing') returns URLs with status='imported'."""
        urls = [
            "https://example.com/recipe1",
            "https://example.com/recipe2",
            "https://example.com/recipe3",
        ]
        state.add_urls(urls)
        
        state.update_recipe(urls[0], status='imported')
        state.update_recipe(urls[1], status='imported')
        state.update_recipe(urls[2], status='parsed')
        
        pending = state.get_pending_for_phase('parsing')
        
        # recipe1 and recipe2 should be pending for parsing
        assert len(pending) == 2
        assert urls[0] in pending
        assert urls[1] in pending
    
    @pytest.mark.readonly
    def test_get_pending_for_phase_tagging(self, state):
        """get_pending_for_phase('tagging') returns URLs with status='parsed'."""
        urls = ["https://example.com/recipe1", "https://example.com/recipe2"]
        state.add_urls(urls)
        
        state.update_recipe(urls[0], status='parsed')
        state.update_recipe(urls[1], status='tagged')
        
        pending = state.get_pending_for_phase('tagging')
        
        assert len(pending) == 1
        assert urls[0] in pending
    
    @pytest.mark.readonly
    def test_get_pending_for_phase_indexing(self, state):
        """get_pending_for_phase('indexing') returns URLs with status='tagged'."""
        urls = ["https://example.com/recipe1", "https://example.com/recipe2"]
        state.add_urls(urls)
        
        state.update_recipe(urls[0], status='tagged')
        state.update_recipe(urls[1], status='indexed')
        
        pending = state.get_pending_for_phase('indexing')
        
        assert len(pending) == 1
        assert urls[0] in pending
    
    @pytest.mark.readonly
    def test_get_pending_for_phase_invalid(self, state):
        """get_pending_for_phase with invalid phase returns empty list."""
        state.add_urls(["https://example.com/recipe1"])
        
        pending = state.get_pending_for_phase('invalid_phase')
        
        assert pending == []
    
    @pytest.mark.readonly
    def test_get_progress_empty_state(self, state):
        """get_progress returns zeroes for empty state."""
        progress = state.get_progress()
        
        assert 'import' in progress
        assert 'parsing' in progress
        assert 'tagging' in progress
        assert 'indexing' in progress
        
        for phase in progress.values():
            assert phase.pending == 0
            assert phase.in_progress == 0
            assert phase.completed == 0
            assert phase.failed == 0
    
    @pytest.mark.readonly
    def test_get_progress_counts_correctly(self, state):
        """get_progress should count recipes in each phase correctly."""
        urls = [f"https://example.com/recipe{i}" for i in range(6)]
        state.add_urls(urls)
        
        # Set various statuses
        state.update_recipe(urls[0], status='pending')
        state.update_recipe(urls[1], status='importing')
        state.update_recipe(urls[2], status='imported')
        state.update_recipe(urls[3], status='parsed')
        state.update_recipe(urls[4], status='tagged')
        state.update_recipe(urls[5], status='indexed')
        
        progress = state.get_progress()
        
        # Import phase
        assert progress['import'].pending == 1  # pending
        assert progress['import'].in_progress == 1  # importing
        assert progress['import'].completed == 4  # imported, parsed, tagged, indexed
        
        # Parsing phase
        assert progress['parsing'].pending == 1  # imported
        assert progress['parsing'].in_progress == 0
        assert progress['parsing'].completed == 3  # parsed, tagged, indexed
        
        # Tagging phase
        assert progress['tagging'].pending == 1  # parsed
        assert progress['tagging'].completed == 2  # tagged, indexed
        
        # Indexing phase
        assert progress['indexing'].pending == 1  # tagged
        assert progress['indexing'].completed == 1  # indexed
    
    @pytest.mark.readonly
    def test_get_progress_counts_failed(self, state):
        """get_progress should count failed recipes correctly."""
        urls = ["https://example.com/recipe1", "https://example.com/recipe2"]
        state.add_urls(urls)
        
        state.update_recipe(urls[0], status='failed', phase='parsing', error='Parse error')
        state.update_recipe(urls[1], status='imported')
        
        progress = state.get_progress()
        
        # Failed recipe should be counted in its phase
        assert progress['parsing'].failed == 1
    
    @pytest.mark.readonly
    def test_save_creates_file(self, state):
        """save() should create state file."""
        state.add_urls(["https://example.com/recipe1"])
        
        result = state.save()
        
        assert result is True
        assert state._state_file.exists()
    
    @pytest.mark.readonly
    def test_save_writes_correct_data(self, state):
        """save() should write correct JSON data."""
        url = "https://example.com/recipe1"
        state.add_urls([url])
        state.update_recipe(url, status='imported', slug='recipe-1')
        
        state.save()
        
        # Read the file directly
        with open(state._state_file, 'r') as f:
            data = json.load(f)
        
        assert data['job_id'] == state.job_id
        assert url in data['results']
        assert data['results'][url]['status'] == 'imported'
        assert data['results'][url]['slug'] == 'recipe-1'
    
    @pytest.mark.readonly
    def test_load_restores_state(self, state):
        """load() should restore state from file."""
        url = "https://example.com/recipe1"
        state.add_urls([url])
        state.update_recipe(url, status='parsed', quality='GOOD')
        state.save()
        
        # Create new state with same job_id
        from panel.jobs.pipeline_state import PipelineState
        new_state = PipelineState(state.job_id)
        result = new_state.load()
        
        assert result is True
        assert url in new_state.results
        assert new_state.results[url].status == 'parsed'
        assert new_state.results[url].quality == 'GOOD'
    
    @pytest.mark.readonly
    def test_load_returns_false_if_no_file(self):
        """load() should return False if state file doesn't exist."""
        from panel.jobs.pipeline_state import PipelineState
        state = PipelineState("nonexistent-job-id-12345")
        
        result = state.load()
        
        assert result is False
    
    @pytest.mark.readonly
    def test_load_handles_corrupted_file(self, state):
        """load() should handle corrupted JSON gracefully."""
        # Create corrupted state file
        state._state_file.write_text("{ invalid json }")
        
        result = state.load()
        
        assert result is False
    
    @pytest.mark.readonly
    def test_to_dict_serialization(self, state):
        """to_dict() should serialize state correctly."""
        url = "https://example.com/recipe1"
        state.add_urls([url])
        state.update_recipe(url, status='tagged', slug='recipe-1', quality='GOOD')
        
        data = state.to_dict()
        
        assert data['job_id'] == state.job_id
        assert url in data['results']
        assert data['results'][url]['status'] == 'tagged'
        assert data['results'][url]['slug'] == 'recipe-1'
        assert data['results'][url]['quality'] == 'GOOD'
    
    @pytest.mark.readonly
    def test_from_dict_deserialization(self, state):
        """from_dict() should deserialize data correctly."""
        data = {
            'job_id': 'test-job-123',
            'results': {
                'https://example.com/recipe1': {
                    'url': 'https://example.com/recipe1',
                    'slug': 'recipe-1',
                    'status': 'indexed',
                    'phase': 'indexing',
                    'quality': 'GOOD',
                    'error': None,
                    'retry_count': 0,
                    'started_at': '2026-01-30T10:00:00',
                    'completed_at': '2026-01-30T10:05:00',
                }
            }
        }
        
        state.from_dict(data)
        
        assert state.job_id == 'test-job-123'
        assert 'https://example.com/recipe1' in state.results
        result = state.results['https://example.com/recipe1']
        assert result.status == 'indexed'
        assert result.slug == 'recipe-1'
        assert result.quality == 'GOOD'
    
    @pytest.mark.readonly
    def test_from_dict_handles_invalid_recipe(self, state):
        """from_dict() should handle invalid recipe data gracefully."""
        data = {
            'job_id': 'test-job-123',
            'results': {
                'https://example.com/recipe1': {
                    'url': 'https://example.com/recipe1',
                    'invalid_field': 'will cause TypeError',
                }
            }
        }
        
        # Should not raise, should create minimal valid entry
        state.from_dict(data)
        
        assert 'https://example.com/recipe1' in state.results
        assert state.results['https://example.com/recipe1'].status == 'failed'
    
    @pytest.mark.readonly
    def test_get_summary(self, state):
        """get_summary() should return correct statistics."""
        urls = [f"https://example.com/recipe{i}" for i in range(5)]
        state.add_urls(urls)
        
        state.update_recipe(urls[0], status='indexed')
        state.update_recipe(urls[1], status='indexed')
        state.update_recipe(urls[2], status='failed', error='Error')
        state.update_recipe(urls[3], status='parsing')
        state.update_recipe(urls[4], status='pending')
        
        summary = state.get_summary()
        
        assert summary['total'] == 5
        assert summary['completed'] == 2
        assert summary['failed'] == 1
        assert summary['in_progress'] == 2
        assert 'phases' in summary
    
    @pytest.mark.readonly
    def test_thread_safety_concurrent_updates(self, state):
        """Concurrent updates should not corrupt state."""
        urls = [f"https://example.com/recipe{i}" for i in range(100)]
        state.add_urls(urls)
        
        errors = []
        
        def update_recipe(url, status):
            try:
                state.update_recipe(url, status=status)
            except Exception as e:
                errors.append(e)
        
        # Concurrent updates
        threads = []
        for i, url in enumerate(urls):
            status = ['importing', 'imported', 'parsing', 'parsed', 'tagged', 'indexed'][i % 6]
            t = threading.Thread(target=update_recipe, args=(url, status))
            threads.append(t)
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should occur
        assert len(errors) == 0
        # All URLs should still be in results
        assert len(state.results) == 100
    
    @pytest.mark.readonly
    def test_thread_safety_concurrent_reads(self, state):
        """Concurrent reads should not cause issues."""
        urls = [f"https://example.com/recipe{i}" for i in range(50)]
        state.add_urls(urls)
        
        # Update some recipes
        for i, url in enumerate(urls[:25]):
            state.update_recipe(url, status='imported')
        
        results = []
        errors = []
        
        def read_pending():
            try:
                pending = state.get_pending_for_phase('import')
                results.append(len(pending))
            except Exception as e:
                errors.append(e)
        
        # Concurrent reads
        threads = [threading.Thread(target=read_pending) for _ in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        # All reads should return consistent count (25 pending)
        assert all(r == 25 for r in results)
    
    @pytest.mark.readonly
    def test_thread_safety_read_write_mix(self, state):
        """Mixed read/write operations should be thread-safe."""
        urls = [f"https://example.com/recipe{i}" for i in range(50)]
        state.add_urls(urls)
        
        errors = []
        
        def writer(url):
            try:
                time.sleep(0.001)  # Small delay to increase contention
                state.update_recipe(url, status='imported')
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                time.sleep(0.001)
                _ = state.get_progress()
                _ = state.get_pending_for_phase('import')
            except Exception as e:
                errors.append(e)
        
        threads = []
        for url in urls[:25]:
            threads.append(threading.Thread(target=writer, args=(url,)))
        for _ in range(25):
            threads.append(threading.Thread(target=reader))
        
        # Shuffle to mix readers and writers
        import random
        random.shuffle(threads)
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# =============================================================================
# Test: RecipeResult Dataclass
# =============================================================================

class TestRecipeResult:
    """Test RecipeResult dataclass."""
    
    @pytest.mark.readonly
    def test_default_values(self):
        """RecipeResult should have correct defaults."""
        from panel.jobs.pipeline_state import RecipeResult
        
        result = RecipeResult(url="https://example.com/recipe1")
        
        assert result.url == "https://example.com/recipe1"
        assert result.slug is None
        assert result.status == 'pending'
        assert result.phase == 'discovery'
        assert result.error is None
        assert result.quality is None
        assert result.retry_count == 0
        assert result.started_at is None
        assert result.completed_at is None
    
    @pytest.mark.readonly
    def test_custom_values(self):
        """RecipeResult should accept custom values."""
        from panel.jobs.pipeline_state import RecipeResult
        
        result = RecipeResult(
            url="https://example.com/recipe1",
            slug="recipe-1",
            status="indexed",
            phase="indexing",
            quality="GOOD",
        )
        
        assert result.slug == "recipe-1"
        assert result.status == "indexed"
        assert result.phase == "indexing"
        assert result.quality == "GOOD"


# =============================================================================
# Test: PhaseProgress Dataclass
# =============================================================================

class TestPhaseProgress:
    """Test PhaseProgress dataclass."""
    
    @pytest.mark.readonly
    def test_default_values(self):
        """PhaseProgress should have correct defaults."""
        from panel.jobs.pipeline_state import PhaseProgress
        
        progress = PhaseProgress()
        
        assert progress.pending == 0
        assert progress.in_progress == 0
        assert progress.completed == 0
        assert progress.failed == 0
    
    @pytest.mark.readonly
    def test_total_property(self):
        """total property should sum all counts."""
        from panel.jobs.pipeline_state import PhaseProgress
        
        progress = PhaseProgress(pending=5, in_progress=3, completed=10, failed=2)
        
        assert progress.total == 20
    
    @pytest.mark.readonly
    def test_percent_property(self):
        """percent property should calculate completion percentage."""
        from panel.jobs.pipeline_state import PhaseProgress
        
        progress = PhaseProgress(pending=0, in_progress=0, completed=75, failed=25)
        
        assert progress.percent == 75.0
    
    @pytest.mark.readonly
    def test_percent_empty(self):
        """percent property should return 0 for empty progress."""
        from panel.jobs.pipeline_state import PhaseProgress
        
        progress = PhaseProgress()
        
        assert progress.percent == 0.0


# =============================================================================
# Test: Worker Functions - Importability and Signatures
# =============================================================================

class TestWorkerFunctions:
    """Test worker function signatures and importability."""
    
    @pytest.mark.readonly
    def test_parse_worker_importable(self):
        """parse_single_recipe_standalone should be importable."""
        from mealie_parse import parse_single_recipe_standalone
        
        assert callable(parse_single_recipe_standalone)
    
    @pytest.mark.readonly
    def test_parse_worker_signature(self):
        """parse_single_recipe_standalone should have correct signature."""
        import inspect
        from mealie_parse import parse_single_recipe_standalone
        
        sig = inspect.signature(parse_single_recipe_standalone)
        params = list(sig.parameters.keys())
        
        assert 'slug' in params
        assert 'auto_tag' in params
        assert 'force_reparse' in params
    
    @pytest.mark.readonly
    def test_parse_worker_returns_bool(self):
        """parse_single_recipe_standalone should return bool."""
        import inspect
        from mealie_parse import parse_single_recipe_standalone
        
        sig = inspect.signature(parse_single_recipe_standalone)
        # Check return annotation if present
        if sig.return_annotation != inspect.Signature.empty:
            assert sig.return_annotation == bool
    
    @pytest.mark.readonly
    def test_tag_worker_importable(self):
        """tag_single_recipe should be importable."""
        from bulk_import_smart import tag_single_recipe
        
        assert callable(tag_single_recipe)
    
    @pytest.mark.readonly
    def test_tag_worker_signature(self):
        """tag_single_recipe should have correct signature."""
        import inspect
        from bulk_import_smart import tag_single_recipe
        
        sig = inspect.signature(tag_single_recipe)
        params = list(sig.parameters.keys())
        
        assert 'url' in params
        assert 'slug' in params
    
    @pytest.mark.readonly
    def test_tag_worker_returns_bool(self):
        """tag_single_recipe should return bool."""
        import inspect
        from bulk_import_smart import tag_single_recipe
        
        sig = inspect.signature(tag_single_recipe)
        if sig.return_annotation != inspect.Signature.empty:
            assert sig.return_annotation == bool
    
    @pytest.mark.readonly
    def test_index_worker_importable(self):
        """index_single_recipe_worker should be importable."""
        from bulk_import_smart import index_single_recipe_worker
        
        assert callable(index_single_recipe_worker)
    
    @pytest.mark.readonly
    def test_index_worker_signature(self):
        """index_single_recipe_worker should have correct signature."""
        import inspect
        from bulk_import_smart import index_single_recipe_worker
        
        sig = inspect.signature(index_single_recipe_worker)
        params = list(sig.parameters.keys())
        
        assert 'slug' in params
        assert 'rag' in params
    
    @pytest.mark.readonly
    def test_index_worker_has_lock(self):
        """index_single_recipe_worker should use thread-safe lock."""
        from bulk_import_smart import _rag_index_lock
        
        # Should be a threading.Lock
        assert isinstance(_rag_index_lock, type(threading.Lock()))


# =============================================================================
# Test: Streaming Pipeline Function
# =============================================================================

class TestStreamingPipeline:
    """Test streaming_bulk_import function."""
    
    @pytest.mark.readonly
    def test_streaming_bulk_import_exists(self):
        """streaming_bulk_import should be importable."""
        from bulk_import_smart import streaming_bulk_import
        
        assert callable(streaming_bulk_import)
    
    @pytest.mark.readonly
    def test_streaming_bulk_import_signature(self):
        """streaming_bulk_import should have correct signature."""
        import inspect
        from bulk_import_smart import streaming_bulk_import
        
        sig = inspect.signature(streaming_bulk_import)
        params = list(sig.parameters.keys())
        
        assert 'urls' in params
        assert 'job_id' in params
        assert 'dry_run' in params
    
    @pytest.mark.readonly
    def test_streaming_bulk_import_returns_pipeline_state(self):
        """streaming_bulk_import should return PipelineState."""
        import inspect
        from bulk_import_smart import streaming_bulk_import
        
        sig = inspect.signature(streaming_bulk_import)
        # Check return annotation
        if sig.return_annotation != inspect.Signature.empty:
            # Should mention PipelineState
            assert 'PipelineState' in str(sig.return_annotation)
    
    @pytest.mark.readonly
    def test_streaming_pipeline_uses_queue(self):
        """streaming_bulk_import should use queues for streaming."""
        import inspect
        from bulk_import_smart import streaming_bulk_import
        
        source = inspect.getsource(streaming_bulk_import)
        
        # Should use Queue for inter-phase communication
        assert 'Queue' in source
        assert 'import_queue' in source or 'parse_queue' in source
    
    @pytest.mark.readonly
    def test_streaming_pipeline_uses_threadpool(self):
        """streaming_bulk_import should use ThreadPoolExecutor."""
        import inspect
        from bulk_import_smart import streaming_bulk_import
        
        source = inspect.getsource(streaming_bulk_import)
        
        assert 'ThreadPoolExecutor' in source
    
    @pytest.mark.readonly
    def test_streaming_pipeline_creates_state(self):
        """streaming_bulk_import should create PipelineState with job_id."""
        import inspect
        from bulk_import_smart import streaming_bulk_import
        
        source = inspect.getsource(streaming_bulk_import)
        
        assert 'PipelineState' in source
        assert 'job_id' in source
    
    @pytest.mark.readonly
    def test_streaming_pipeline_loads_existing_state(self):
        """streaming_bulk_import should load existing state for resume."""
        import inspect
        from bulk_import_smart import streaming_bulk_import
        
        source = inspect.getsource(streaming_bulk_import)
        
        # Should call state.load() for resume capability
        assert 'state.load()' in source or '.load()' in source


# =============================================================================
# Test: Pipeline Phase Constants
# =============================================================================

class TestPipelinePhaseConstants:
    """Test phase mapping constants in PipelineState."""
    
    @pytest.mark.readonly
    def test_phase_pending_status_defined(self):
        """PHASE_PENDING_STATUS should map phases to required statuses."""
        from panel.jobs.pipeline_state import PipelineState
        
        expected_phases = ['import', 'parsing', 'tagging', 'indexing']
        
        for phase in expected_phases:
            assert phase in PipelineState.PHASE_PENDING_STATUS
    
    @pytest.mark.readonly
    def test_phase_pending_status_values(self):
        """PHASE_PENDING_STATUS should have correct values."""
        from panel.jobs.pipeline_state import PipelineState
        
        assert PipelineState.PHASE_PENDING_STATUS['import'] == 'pending'
        assert PipelineState.PHASE_PENDING_STATUS['parsing'] == 'imported'
        assert PipelineState.PHASE_PENDING_STATUS['tagging'] == 'parsed'
        assert PipelineState.PHASE_PENDING_STATUS['indexing'] == 'tagged'
    
    @pytest.mark.readonly
    def test_status_phase_mapping_defined(self):
        """STATUS_PHASE should map statuses to phases."""
        from panel.jobs.pipeline_state import PipelineState
        
        expected_statuses = [
            'pending', 'importing', 'imported',
            'parsing', 'parsed',
            'tagging', 'tagged',
            'indexing', 'indexed',
            'failed'
        ]
        
        for status in expected_statuses:
            assert status in PipelineState.STATUS_PHASE


# =============================================================================
# Test: Resume Functionality
# =============================================================================

class TestResumeFunctionality:
    """Test resume capability of streaming pipeline."""
    
    @pytest.fixture
    def persisted_state(self):
        """Create a persisted PipelineState for resume testing."""
        from panel.jobs.pipeline_state import PipelineState
        
        job_id = f"resume-test-{int(time.time() * 1000)}"
        state = PipelineState(job_id)
        
        # Add some URLs and set various states
        urls = [
            "https://example.com/recipe1",
            "https://example.com/recipe2",
            "https://example.com/recipe3",
        ]
        state.add_urls(urls)
        state.update_recipe(urls[0], status='indexed')  # Completed
        state.update_recipe(urls[1], status='parsing')  # In progress
        state.update_recipe(urls[2], status='pending')  # Not started
        
        state.save()
        
        yield state
        
        # Cleanup
        if state._state_file.exists():
            state._state_file.unlink()
    
    @pytest.mark.readonly
    def test_resume_loads_existing_state(self, persisted_state):
        """Resume should load existing state from file."""
        from panel.jobs.pipeline_state import PipelineState
        
        new_state = PipelineState(persisted_state.job_id)
        loaded = new_state.load()
        
        assert loaded is True
        assert len(new_state.results) == 3
    
    @pytest.mark.readonly
    def test_resume_preserves_completed(self, persisted_state):
        """Resume should preserve completed recipes."""
        from panel.jobs.pipeline_state import PipelineState
        
        new_state = PipelineState(persisted_state.job_id)
        new_state.load()
        
        assert new_state.results["https://example.com/recipe1"].status == 'indexed'
    
    @pytest.mark.readonly
    def test_resume_identifies_pending(self, persisted_state):
        """Resume should identify pending work correctly."""
        from panel.jobs.pipeline_state import PipelineState
        
        new_state = PipelineState(persisted_state.job_id)
        new_state.load()
        
        # recipe3 should still be pending for import
        pending_import = new_state.get_pending_for_phase('import')
        assert "https://example.com/recipe3" in pending_import


# =============================================================================
# Test: Integration - Worker Function Import in Streaming Pipeline
# =============================================================================

class TestWorkerIntegration:
    """Test that streaming pipeline correctly imports worker functions."""
    
    @pytest.mark.readonly
    def test_pipeline_imports_parse_worker(self):
        """streaming_bulk_import should use parse_single_recipe_standalone."""
        import inspect
        from bulk_import_smart import streaming_bulk_import
        
        source = inspect.getsource(streaming_bulk_import)
        
        assert 'parse_single_recipe_standalone' in source
    
    @pytest.mark.readonly
    def test_pipeline_imports_tag_worker(self):
        """streaming_bulk_import should use tag_single_recipe."""
        import inspect
        from bulk_import_smart import streaming_bulk_import
        
        source = inspect.getsource(streaming_bulk_import)
        
        assert 'tag_single_recipe' in source
    
    @pytest.mark.readonly
    def test_pipeline_imports_index_worker(self):
        """streaming_bulk_import should use index_single_recipe_worker."""
        import inspect
        from bulk_import_smart import streaming_bulk_import
        
        source = inspect.getsource(streaming_bulk_import)
        
        assert 'index_single_recipe_worker' in source


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in pipeline components."""
    
    @pytest.fixture
    def state(self):
        """Create a fresh PipelineState for testing."""
        from panel.jobs.pipeline_state import PipelineState
        job_id = f"error-test-{int(time.time() * 1000)}"
        state = PipelineState(job_id)
        yield state
        if state._state_file.exists():
            state._state_file.unlink()
    
    @pytest.mark.readonly
    def test_update_recipe_stores_error(self, state):
        """update_recipe should store error message."""
        url = "https://example.com/recipe1"
        state.add_urls([url])
        
        state.update_recipe(url, status='failed', error='Parse error: invalid JSON')
        
        assert state.results[url].status == 'failed'
        assert state.results[url].error == 'Parse error: invalid JSON'
    
    @pytest.mark.readonly
    def test_failed_recipes_tracked_in_progress(self, state):
        """Failed recipes should be tracked in progress."""
        urls = ["https://example.com/recipe1", "https://example.com/recipe2"]
        state.add_urls(urls)
        
        state.update_recipe(urls[0], status='failed', phase='parsing', error='Error')
        
        progress = state.get_progress()
        
        assert progress['parsing'].failed == 1
    
    @pytest.mark.readonly
    def test_summary_counts_failed(self, state):
        """get_summary should count failed recipes."""
        urls = [f"https://example.com/recipe{i}" for i in range(5)]
        state.add_urls(urls)
        
        state.update_recipe(urls[0], status='indexed')
        state.update_recipe(urls[1], status='failed', error='Import failed')
        state.update_recipe(urls[2], status='failed', error='Parse failed')
        
        summary = state.get_summary()
        
        assert summary['failed'] == 2
        assert summary['completed'] == 1
