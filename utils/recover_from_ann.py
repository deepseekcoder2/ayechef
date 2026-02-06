#!/usr/bin/env python3
"""
Fast Recovery from ANN Index
============================

Recovers the local recipe_index.db using:
1. Existing embeddings from data/recipe_usearch.index (18,886 vectors)
2. Recipe metadata from Mealie's database (fast bulk fetch)

This avoids regenerating embeddings, making recovery take minutes instead of hours.

Usage:
    python utils/recover_from_ann.py           # Dry run (preview)
    python utils/recover_from_ann.py --run     # Execute recovery
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional
from usearch.index import Index, MetricKind

from mealie_client import MealieClient
from config import validate_all, DATA_DIR
from tools.logging_utils import get_logger

logger = get_logger(__name__)

# Paths
ANN_INDEX_PATH = DATA_DIR / "recipe_usearch.index"
ANN_MAP_PATH = DATA_DIR / "recipe_usearch.map.pkl"
LOCAL_DB_PATH = DATA_DIR / "recipe_index.db"


def load_existing_embeddings() -> tuple[Dict[str, np.ndarray], Set[str]]:
    """
    Load embeddings from existing ANN index.
    
    Returns:
        embeddings_by_uuid: Dict mapping recipe UUID to embedding vector
        recipe_ids: Set of all recipe UUIDs in the index
    """
    print("📂 Loading existing ANN index...")
    
    if not ANN_INDEX_PATH.exists():
        raise FileNotFoundError(f"ANN index not found: {ANN_INDEX_PATH}")
    if not ANN_MAP_PATH.exists():
        raise FileNotFoundError(f"ANN map not found: {ANN_MAP_PATH}")
    
    # Load index
    index = Index(ndim=4096, metric=MetricKind.Cos, dtype='f32')
    index.load(str(ANN_INDEX_PATH))
    print(f"   Loaded {len(index)} vectors from index")
    
    # Load mapping (internal_id -> recipe_uuid)
    with open(ANN_MAP_PATH, 'rb') as f:
        id_map = pickle.load(f)
    print(f"   Loaded {len(id_map)} ID mappings")
    
    # Extract all embeddings
    embeddings_by_uuid = {}
    for internal_id, recipe_uuid in id_map.items():
        try:
            embedding = index[internal_id]
            embeddings_by_uuid[recipe_uuid] = np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to extract embedding for {recipe_uuid}: {e}")
    
    print(f"   Extracted {len(embeddings_by_uuid)} embeddings")
    return embeddings_by_uuid, set(embeddings_by_uuid.keys())


def fetch_all_recipes_from_mealie() -> list:
    """Fetch all recipe data from Mealie using fast bulk DB mode."""
    print("📡 Fetching all recipes from Mealie...")
    
    client = MealieClient()
    print(f"   Mode: {client.mode}")
    
    if client.mode == 'db':
        # Use bulk fetch for maximum speed
        recipes = client.get_all_recipes_full()
    else:
        # Fall back to API
        print("   ⚠️  Not in DB mode - this will be slower")
        recipes = client.get_all_recipes_full()
    
    client.close()
    print(f"   Fetched {len(recipes)} recipes")
    return recipes


def create_searchable_text(recipe: dict) -> str:
    """Create searchable text from recipe data."""
    parts = []
    
    if recipe.get("name"):
        parts.append(f"Recipe: {recipe['name']}")
    
    if recipe.get("description"):
        parts.append(f"Description: {recipe['description']}")
    
    # Ingredients
    ingredients = recipe.get("recipeIngredient", [])
    if ingredients:
        ing_texts = []
        for ing in ingredients:
            if isinstance(ing, dict):
                food = ing.get('food', {})
                if isinstance(food, dict) and food.get('name'):
                    ing_texts.append(food['name'])
                elif ing.get('display'):
                    ing_texts.append(ing['display'])
            elif isinstance(ing, str):
                ing_texts.append(ing)
        if ing_texts:
            parts.append(f"Ingredients: {', '.join(ing_texts[:20])}")
    
    # Tags
    tags = recipe.get("tags", [])
    if tags:
        tag_names = [t.get('name', '') for t in tags if isinstance(t, dict)]
        if tag_names:
            parts.append(f"Tags: {', '.join(tag_names)}")
    
    # Instructions
    instructions = recipe.get("recipeInstructions", [])
    if instructions:
        inst_texts = []
        for inst in instructions[:5]:  # First 5 steps
            if isinstance(inst, dict):
                inst_texts.append(inst.get('text', '')[:200])
            elif isinstance(inst, str):
                inst_texts.append(inst[:200])
        if inst_texts:
            parts.append(f"Instructions: {' '.join(inst_texts)}")
    
    return "\n".join(parts)


def extract_cuisine_from_tags(tags: list) -> tuple:
    """
    Extract cuisine information from tags.
    
    Inlined from RecipeRAG._extract_cuisine_from_tags to avoid loading ANN index.
    
    Returns:
        Tuple of (primary_cuisine, secondary_cuisines_json, region, confidence)
    """
    primary_cuisine = None
    secondary_cuisines = []
    region = None
    confidence = 0.0

    for tag in tags:
        if not isinstance(tag, dict):
            continue

        tag_name = tag.get("name", "")
        if not tag_name:
            continue

        # Parse cuisine tags (format: "Cuisine: Italian - Northern" or "Italian Cuisine")
        if "Cuisine:" in tag_name or " Cuisine" in tag_name:
            confidence = 1.0  # Tagged cuisines are authoritative

            # Extract cuisine name
            if "Cuisine:" in tag_name:
                # Format: "Cuisine: Italian - Northern"
                cuisine_part = tag_name.split("Cuisine:")[1].strip()
            else:
                # Format: "Italian Cuisine"
                cuisine_part = tag_name.split(" Cuisine")[0].strip()

            # Handle regional variants (Cuisine - Region)
            if " - " in cuisine_part:
                cuisine_name, region_name = cuisine_part.split(" - ", 1)
                cuisine_name = cuisine_name.strip()
                region_name = region_name.strip()

                if not primary_cuisine:
                    primary_cuisine = cuisine_name
                    region = region_name
                else:
                    secondary_cuisines.append(cuisine_name)
            else:
                # Simple cuisine name
                if not primary_cuisine:
                    primary_cuisine = cuisine_part
                else:
                    secondary_cuisines.append(cuisine_part)

        # Also check for region tags
        elif "Region:" in tag_name:
            if not region:
                region = tag_name.split("Region:")[1].strip()

    # Convert secondary cuisines to JSON string
    secondary_cuisines_json = json.dumps(secondary_cuisines) if secondary_cuisines else None

    return primary_cuisine, secondary_cuisines_json, region, confidence


def summarize_ingredients(ingredients: list) -> list:
    """Summarize ingredients list."""
    result = []
    for ing in ingredients:
        if isinstance(ing, dict):
            food = ing.get('food', {})
            if isinstance(food, dict) and food.get('name'):
                result.append(food['name'])
            elif ing.get('display'):
                result.append(ing['display'])
        elif isinstance(ing, str):
            result.append(ing)
    return result[:30]  # Limit to 30 ingredients


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Convert embedding to bytes for SQLite storage."""
    return embedding.astype(np.float32).tobytes()


def recover_database(dry_run: bool = True):
    """
    Main recovery function.
    
    Args:
        dry_run: If True, only preview what would happen
    """
    print("=" * 60)
    print("🔄 Recipe Database Recovery from ANN Index")
    print("=" * 60)
    
    # Step 1: Load existing embeddings
    embeddings_by_uuid, existing_uuids = load_existing_embeddings()
    
    # Step 2: Fetch recipes from Mealie
    recipes = fetch_all_recipes_from_mealie()
    
    # Step 3: Match recipes to embeddings
    matched = []
    unmatched = []
    
    for recipe in recipes:
        recipe_id = recipe.get('id')
        if not recipe_id:
            continue
        
        if recipe_id in embeddings_by_uuid:
            matched.append((recipe, embeddings_by_uuid[recipe_id]))
        else:
            unmatched.append(recipe)
    
    print(f"\n📊 Recovery Summary:")
    print(f"   Recipes in Mealie:     {len(recipes)}")
    print(f"   Embeddings available:  {len(embeddings_by_uuid)}")
    print(f"   Matched (recoverable): {len(matched)}")
    print(f"   Unmatched (need embed): {len(unmatched)}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - No changes made")
        print(f"   Run with --run to execute recovery")
        
        if unmatched:
            print(f"\n   Unmatched recipes (sample):")
            for recipe in unmatched[:5]:
                print(f"      - {recipe.get('name', 'Unknown')} ({recipe.get('id', 'no-id')[:8]}...)")
        return
    
    # Step 4: Recover to database
    print(f"\n💾 Writing {len(matched)} recipes to database...")
    
    recovered = 0
    with sqlite3.connect(LOCAL_DB_PATH) as conn:
        for recipe, embedding in matched:
            try:
                recipe_id = recipe.get('id')
                name = recipe.get('name', '')
                slug = recipe.get('slug', '')
                description = recipe.get('description', '')
                org_url = recipe.get('orgURL', '') or ''
                
                searchable_text = create_searchable_text(recipe)
                tags = json.dumps(recipe.get('tags', []))
                categories = json.dumps(recipe.get('recipeCategory', []))
                ingredients = summarize_ingredients(recipe.get('recipeIngredient', []))
                ingredients_json = json.dumps(ingredients)
                
                cuisine_primary, cuisine_secondary, cuisine_region, cuisine_confidence = extract_cuisine_from_tags(recipe.get('tags', []))
                
                mealie_updated_at = recipe.get('updatedAt') or recipe.get('dateUpdated') or ''
                embedding_bytes = embedding_to_bytes(embedding)
                
                conn.execute('''
                    INSERT OR REPLACE INTO recipes
                    (id, name, slug, org_url, description, searchable_text, tags, categories, 
                     ingredients, cuisine_primary, cuisine_secondary, cuisine_region, 
                     cuisine_confidence, embedding, updated_at, mealie_updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ''', (recipe_id, name, slug, org_url, description, searchable_text, tags, 
                      categories, ingredients_json, cuisine_primary, cuisine_secondary, 
                      cuisine_region, cuisine_confidence, embedding_bytes, mealie_updated_at))
                
                # Also update FTS
                ingredients_text = " ".join(ingredients)
                try:
                    tags_list = json.loads(tags) if tags else []
                    tags_text = " ".join([tag.get("name", "") for tag in tags_list if isinstance(tag, dict)])
                except:
                    tags_text = ""
                
                conn.execute('''
                    INSERT OR REPLACE INTO recipes_fts
                    (id, name, description, searchable_text, tags, ingredients)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (recipe_id, name, description, searchable_text, tags_text, ingredients_text))
                
                recovered += 1
                
                if recovered % 1000 == 0:
                    print(f"   Recovered {recovered}/{len(matched)}...")
                    conn.commit()
            
            except Exception as e:
                logger.warning(f"Failed to recover {recipe.get('name', 'unknown')}: {e}")
        
        conn.commit()
    
    print(f"\n✅ Recovery complete!")
    print(f"   Recovered: {recovered} recipes")
    print(f"   Need embedding: {len(unmatched)} recipes")
    
    if unmatched:
        print(f"\n📝 To embed remaining {len(unmatched)} recipes, run:")
        print(f"   python utils/recipe_maintenance.py --sync")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Recover recipe database from ANN index")
    parser.add_argument("--run", action="store_true", help="Execute recovery (default: dry run)")
    args = parser.parse_args()
    
    # Validate system
    if not validate_all():
        print("❌ System validation failed")
        sys.exit(1)
    
    try:
        recover_database(dry_run=not args.run)
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
