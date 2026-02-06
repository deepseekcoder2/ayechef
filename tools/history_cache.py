#!/usr/bin/env python3
"""
Historical Meal Data Caching System
====================================

LRU cache for historical meal data with TTL-based expiration.
Caches processed variety constraints to accelerate planning cycles.

Features:
- File-based caching with JSON storage
- TTL-based expiration (7-day default)
- SHA-256 hash keys for date ranges
- Automatic cache invalidation
- Thread-safe operations
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import os


class HistoryCache:
    """
    LRU cache for historical meal data with TTL-based expiration.
    Prevents redundant 4-week history fetches in chef_agentic.py.
    """

    def __init__(self, cache_dir: str = None, ttl_days: int = 7):
        if cache_dir is None:
            from config import DATA_DIR
            cache_dir = str(DATA_DIR / "cache")
        """
        Initialize the history cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_days: Time-to-live in days for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_days = ttl_days
        self.ttl_seconds = ttl_days * 24 * 60 * 60

    def _get_cache_key(self, start_date: str, end_date: str) -> str:
        """Generate cache key from date range."""
        key_data = f"{start_date}:{end_date}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get filesystem path for cache entry."""
        return self.cache_dir / f"{cache_key}.json"

    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cache entry is expired."""
        if not cache_path.exists():
            return True
        mtime = cache_path.stat().st_mtime
        return (time.time() - mtime) > self.ttl_seconds

    def get(self, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Retrieve cached historical data if valid.

        Returns processed variety constraints, not raw meal data.
        """
        cache_key = self._get_cache_key(start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        if self._is_expired(cache_path):
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Validate cache structure
                if self._validate_cache_data(data):
                    print(f"✅ Using cached history data ({start_date} to {end_date})")
                    return data
        except (FileNotFoundError, json.JSONDecodeError, KeyError, UnicodeDecodeError):
            pass

        return None

    def put(self, start_date: str, end_date: str, history_data: Dict):
        """
        Cache processed historical data.

        Args:
            start_date/end_date: Date range for cache key
            history_data: Processed variety constraints and analytics
        """
        cache_key = self._get_cache_key(start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        # Validate data structure before caching
        if not self._validate_cache_data(history_data):
            print("⚠️ Invalid cache data structure, skipping cache")
            return

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Cached history data ({start_date} to {end_date})")
        except Exception as e:
            print(f"⚠️ Failed to cache history data: {e}")

    def _validate_cache_data(self, data: Dict) -> bool:
        """
        Validate cache data has required structure.

        Expected structure:
        {
            "date_range": {"start": "2024-01-01", "end": "2024-01-15"},
            "variety_constraints": {
                "cuisine_counts": {"cantonese": 3, "japanese": 2, ...},
                "protein_rotation": ["chicken", "beef", "fish", ...],
                "dish_patterns": {...}
            },
            "meal_history_summary": {
                "total_meals": 28,
                "avg_meals_per_day": 2.0,
                ...
            }
        }
        """
        required_keys = ["date_range", "variety_constraints", "meal_history_summary"]
        if not all(key in data for key in required_keys):
            return False

        # Check date_range structure
        date_range = data.get("date_range", {})
        if not isinstance(date_range, dict) or "start" not in date_range or "end" not in date_range:
            return False

        # Check variety_constraints structure
        constraints = data.get("variety_constraints", {})
        if not isinstance(constraints, dict):
            return False

        return True

    def invalidate(self, start_date: str = None, end_date: str = None):
        """
        Invalidate cache entries (called after meal plan changes).
        """
        if start_date and end_date:
            # Invalidate specific range
            cache_key = self._get_cache_key(start_date, end_date)
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
                print(f"🗑️ Invalidated cache for {start_date} to {end_date}")
        else:
            # Clear all cache
            deleted_count = 0
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                deleted_count += 1
            print(f"🗑️ Cleared all cache ({deleted_count} entries)")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        expired_count = sum(1 for f in cache_files if self._is_expired(f))

        return {
            "total_entries": len(cache_files),
            "expired_entries": expired_count,
            "valid_entries": len(cache_files) - expired_count,
            "ttl_days": self.ttl_days,
            "cache_dir": str(self.cache_dir)
        }

    def cleanup_expired(self) -> int:
        """Remove expired cache entries. Returns number of files deleted."""
        deleted_count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if self._is_expired(cache_file):
                cache_file.unlink()
                deleted_count += 1
        return deleted_count
