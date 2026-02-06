#!/usr/bin/env python3
"""
Rebuild All Search Indexes
==========================

This script rebuilds both the ANN (vector/semantic) and FTS (full-text) search
indexes from scratch. Use this as a recovery tool when search is returning
wrong results or errors.

What it does:
1. Rebuilds the USearch ANN index from existing embeddings in recipe_index.db
2. Rebuilds the SQLite FTS5 full-text search index

Usage:
    python utils/rebuild_search_index.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.logging_utils import get_logger

logger = get_logger(__name__)


def rebuild_search_indexes():
    """
    Rebuild both ANN and FTS search indexes.
    
    Returns:
        True if both succeeded, False if either failed
    """
    print("=" * 60)
    print("🔧 REBUILD SEARCH INDEXES")
    print("=" * 60)
    print("\nThis will rebuild all search indexes from scratch.")
    print("Your recipe data will NOT be affected.\n")
    
    success = True
    
    # Step 1: Rebuild ANN (vector) index
    print("-" * 60)
    print("📊 Step 1: Rebuilding ANN (semantic search) index...")
    print("-" * 60)
    
    try:
        from utils.rebuild_ann_index import rebuild_ann_index
        ann_success = rebuild_ann_index()
        if ann_success:
            print("✅ ANN index rebuilt successfully\n")
        else:
            print("❌ ANN index rebuild failed\n")
            success = False
    except Exception as e:
        logger.error(f"ANN index rebuild error: {e}")
        print(f"❌ ANN index rebuild error: {e}\n")
        success = False
    
    # Step 2: Rebuild FTS (full-text search) index
    print("-" * 60)
    print("📝 Step 2: Rebuilding FTS (text search) index...")
    print("-" * 60)
    
    try:
        from recipe_rag import RecipeRAG
        rag = RecipeRAG()
        rag.rebuild_fts_index()
        print("✅ FTS index rebuilt successfully\n")
    except Exception as e:
        logger.error(f"FTS index rebuild error: {e}")
        print(f"❌ FTS index rebuild error: {e}\n")
        success = False
    
    # Summary
    print("=" * 60)
    if success:
        print("🎉 All search indexes rebuilt successfully!")
    else:
        print("⚠️  Some indexes failed to rebuild. Check errors above.")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = rebuild_search_indexes()
    sys.exit(0 if success else 1)
