"""Pipeline state tracking for streaming recipe import with resume capability."""
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import DATA_DIR

logger = logging.getLogger(__name__)

DATA_DIR.mkdir(exist_ok=True)

# Jobs state directory
JOBS_DIR = DATA_DIR / 'jobs'
JOBS_DIR.mkdir(exist_ok=True)


@dataclass
class RecipeResult:
    """Per-recipe state tracking."""
    url: str
    slug: Optional[str] = None
    name: Optional[str] = None  # Recipe name (captured during processing, no extra API call)
    status: str = 'pending'  # pending, importing, imported, parsing, parsed, 
                              # tagging, tagged, indexing, indexed, failed
    phase: str = 'discovery'  # Current phase: discovery, import, parsing, tagging, indexing
    error: Optional[str] = None
    quality: Optional[str] = None  # GOOD, POOR, UNKNOWN
    retry_count: int = 0
    started_at: Optional[str] = None  # ISO format timestamps
    completed_at: Optional[str] = None


@dataclass
class PhaseProgress:
    """Progress tracking for a single phase."""
    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    
    @property
    def total(self) -> int:
        return self.pending + self.in_progress + self.completed + self.failed
    
    @property
    def percent(self) -> float:
        return (self.completed / self.total * 100) if self.total > 0 else 0.0


class PipelineState:
    """
    Tracks pipeline state for resume capability.
    
    Thread-safe with atomic file writes.
    State persisted to data/jobs/{job_id}_state.json
    """
    
    # Phase status mapping: which status indicates "pending" for each phase
    PHASE_PENDING_STATUS = {
        'import': 'pending',
        'parsing': 'imported',
        'tagging': 'parsed',
        'indexing': 'tagged',
    }
    
    # Status to phase mapping for progress calculation
    STATUS_PHASE = {
        'pending': 'import',
        'importing': 'import',
        'imported': 'parsing',
        'parsing': 'parsing',
        'parsed': 'tagging',
        'tagging': 'tagging',
        'tagged': 'indexing',
        'indexing': 'indexing',
        'indexed': 'complete',
        'failed': 'failed',
    }
    
    def __init__(self, job_id: str):
        """Initialize with job_id, create empty results dict."""
        self.job_id = job_id
        self.results: Dict[str, RecipeResult] = {}
        self._lock = threading.Lock()
        # Ensure jobs directory exists (may not exist if module loaded in subprocess)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._state_file = JOBS_DIR / f"{job_id}_state.json"
    
    def add_urls(self, urls: List[str]) -> None:
        """Add URLs as pending recipes (skip if already exists)."""
        with self._lock:
            for url in urls:
                if url not in self.results:
                    self.results[url] = RecipeResult(url=url)
    
    def update_recipe(self, url: str, **kwargs) -> None:
        """Update recipe state (thread-safe)."""
        with self._lock:
            if url not in self.results:
                self.results[url] = RecipeResult(url=url)
            
            recipe = self.results[url]
            for key, value in kwargs.items():
                if hasattr(recipe, key):
                    setattr(recipe, key, value)
            
            # Auto-set timestamps
            if kwargs.get('status') in ('importing', 'parsing', 'tagging', 'indexing'):
                if recipe.started_at is None:
                    recipe.started_at = datetime.now().isoformat()
            
            if kwargs.get('status') in ('indexed', 'failed'):
                recipe.completed_at = datetime.now().isoformat()
    
    def get_pending_for_phase(self, phase: str) -> List[str]:
        """Get URLs pending for specific phase."""
        required_status = self.PHASE_PENDING_STATUS.get(phase)
        if required_status is None:
            return []
        
        with self._lock:
            return [
                url for url, recipe in self.results.items()
                if recipe.status == required_status
            ]
    
    def get_progress(self) -> Dict[str, PhaseProgress]:
        """Get progress for all phases."""
        phases = {
            'import': PhaseProgress(),
            'parsing': PhaseProgress(),
            'tagging': PhaseProgress(),
            'indexing': PhaseProgress(),
        }
        
        with self._lock:
            for recipe in self.results.values():
                status = recipe.status
                
                if status == 'failed':
                    # Failed recipes count as failed in their current phase
                    phase = recipe.phase
                    if phase in phases:
                        phases[phase].failed += 1
                    continue
                
                # Map status to phase and state within that phase
                if status == 'pending':
                    phases['import'].pending += 1
                elif status == 'importing':
                    phases['import'].in_progress += 1
                elif status == 'imported':
                    phases['import'].completed += 1
                    phases['parsing'].pending += 1
                elif status == 'parsing':
                    phases['import'].completed += 1
                    phases['parsing'].in_progress += 1
                elif status == 'parsed':
                    phases['import'].completed += 1
                    phases['parsing'].completed += 1
                    phases['tagging'].pending += 1
                elif status == 'tagging':
                    phases['import'].completed += 1
                    phases['parsing'].completed += 1
                    phases['tagging'].in_progress += 1
                elif status == 'tagged':
                    phases['import'].completed += 1
                    phases['parsing'].completed += 1
                    phases['tagging'].completed += 1
                    phases['indexing'].pending += 1
                elif status == 'indexing':
                    phases['import'].completed += 1
                    phases['parsing'].completed += 1
                    phases['tagging'].completed += 1
                    phases['indexing'].in_progress += 1
                elif status == 'indexed':
                    phases['import'].completed += 1
                    phases['parsing'].completed += 1
                    phases['tagging'].completed += 1
                    phases['indexing'].completed += 1
        
        return phases
    
    def save(self) -> bool:
        """Persist state to disk. Returns True on success."""
        with self._lock:
            data = self.to_dict()
        
        # Ensure directory exists before writing
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use unique temp file name to avoid race conditions between threads
        temp_file = self._state_file.with_suffix(f'.tmp.{uuid.uuid4().hex[:8]}')
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self._state_file)  # os.replace is atomic and cross-platform
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to save state: {e}")
            # Clean up temp file if it exists
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
            return False
    
    def load(self) -> bool:
        """Load state from disk if exists. Returns True if loaded."""
        with self._lock:
            # Check existence inside lock to avoid TOCTOU race
            if not self._state_file.exists():
                return False
            
            try:
                with open(self._state_file, 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                # File may have been deleted between exists() check and open()
                return False
            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.warning(f"Failed to load state file {self._state_file}: {e}")
                return False
            self.from_dict(data)
        
        return True
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            'job_id': self.job_id,
            'results': {
                url: asdict(recipe) 
                for url, recipe in self.results.items()
            }
        }
    
    def from_dict(self, data: dict) -> None:
        """Deserialize from dict."""
        self.job_id = data.get('job_id', self.job_id)
        self.results = {}
        
        # Get known field names for RecipeResult to filter unknown fields
        known_fields = {f.name for f in fields(RecipeResult)}
        
        for url, recipe_data in data.get('results', {}).items():
            try:
                # Filter to only known fields for forward/backward compatibility
                filtered_data = {k: v for k, v in recipe_data.items() if k in known_fields}
                # Ensure URL key matches the stored URL (prevents corruption)
                filtered_data['url'] = url
                self.results[url] = RecipeResult(**filtered_data)
            except TypeError as e:
                logger.warning(f"Skipping invalid recipe data for {url}: {e}")
                # Create minimal valid entry
                self.results[url] = RecipeResult(url=url, status='failed', error=str(e))
    
    def get_summary(self) -> dict:
        """Get a summary of the pipeline state."""
        progress = self.get_progress()
        
        with self._lock:
            total = len(self.results)
            completed = sum(1 for r in self.results.values() if r.status == 'indexed')
            failed = sum(1 for r in self.results.values() if r.status == 'failed')
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'in_progress': total - completed - failed,
            'phases': {
                name: {
                    'pending': p.pending,
                    'in_progress': p.in_progress,
                    'completed': p.completed,
                    'failed': p.failed,
                    'percent': round(p.percent, 1),
                }
                for name, p in progress.items()
            }
        }
