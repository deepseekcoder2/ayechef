#!/usr/bin/env python3
"""
Rebuild ANN Index from Existing Embeddings
==========================================

This script rebuilds the USearch ANN index from existing embeddings in recipe_index.db.
Used after changing embedding dimensions or when the index gets corrupted.

Usage:
    python rebuild_ann_index.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import numpy as np
from recipe_ann_index import RecipeANNIndex
from tools.logging_utils import get_logger

logger = get_logger(__name__)


def rebuild_ann_index(db_path=None):
    if db_path is None:
        from config import DATA_DIR
        db_path = str(DATA_DIR / "recipe_index.db")
    """
    Rebuild the ANN index from embeddings stored in SQLite database.
    
    Args:
        db_path: Path to the recipe database
    """
    print("🔧 Rebuilding ANN index from existing embeddings...")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all recipes with embeddings
    cursor.execute("SELECT id, embedding FROM recipes WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    
    if not rows:
        print("❌ No recipes with embeddings found in database")
        conn.close()
        return False
    
    print(f"📊 Found {len(rows)} recipes with embeddings")
    
    # Extract recipe IDs and embeddings
    recipe_ids = []
    embeddings = []
    
    for recipe_id, embedding_bytes in rows:
        # Convert bytes back to numpy array
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        recipe_ids.append(recipe_id)
        embeddings.append(embedding)
    
    # Check embedding dimensions
    if embeddings:
        dim = len(embeddings[0])
        print(f"🔍 Detected embedding dimension: {dim}")
    
    # Initialize ANN index with correct dimensions
    ann_index = RecipeANNIndex(dimension=dim)
    
    # Build index
    print("🏗️  Building USearch ANN index...")
    ann_index.build_index(embeddings, recipe_ids)
    
    # Save index to disk
    print("💾 Saving ANN index to disk...")
    ann_index.save()
    
    print("✅ ANN index rebuild complete!")
    print(f"   - {len(recipe_ids)} recipes indexed")
    print(f"   - Dimension: {dim}")
    print(f"   - Index file: recipe_usearch.index")
    print(f"   - Mapping file: recipe_usearch.map.pkl")
    
    conn.close()
    return True


if __name__ == "__main__":
    success = rebuild_ann_index()
    if not success:
        exit(1)
