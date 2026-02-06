#!/usr/bin/env python3
"""
Cache Invalidation Utility
==========================

Utility script to invalidate history cache entries when meal plans are modified.
Call this script after creating, updating, or deleting meal plans to ensure
fresh history analysis on next planning cycle.

Usage:
    python invalidate_cache.py                    # Clear all cache
    python invalidate_cache.py --start 2024-01-01 --end 2024-01-15  # Clear specific range
    python invalidate_cache.py --stats           # Show cache statistics
    python invalidate_cache.py --cleanup         # Remove expired entries only
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from tools.history_cache import HistoryCache


def main():
    parser = argparse.ArgumentParser(description="History cache invalidation utility")
    parser.add_argument("--start", help="Start date for range invalidation (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date for range invalidation (YYYY-MM-DD)")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--cleanup", action="store_true", help="Remove expired entries only")

    args = parser.parse_args()

    cache = HistoryCache()

    if args.stats:
        # Show cache statistics
        stats = cache.get_stats()
        print("📊 History Cache Statistics")
        print("=" * 40)
        print(f"Total entries:     {stats['total_entries']}")
        print(f"Valid entries:     {stats['valid_entries']}")
        print(f"Expired entries:   {stats['expired_entries']}")
        print(f"TTL (days):        {stats['ttl_days']}")
        print(f"Cache directory:   {stats['cache_dir']}")
        print("=" * 40)

    elif args.cleanup:
        # Remove expired entries
        deleted = cache.cleanup_expired()
        print(f"🧹 Cleaned up {deleted} expired cache entries")

    elif args.start and args.end:
        # Invalidate specific date range
        cache.invalidate(args.start, args.end)
        print(f"🗑️ Invalidated cache for {args.start} to {args.end}")

    else:
        # Clear all cache
        stats_before = cache.get_stats()
        cache.invalidate()
        print(f"🗑️ Cleared all cache ({stats_before['total_entries']} entries)")


if __name__ == "__main__":
    main()
