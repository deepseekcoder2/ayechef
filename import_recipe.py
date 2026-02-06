#!/usr/bin/env python3
"""
Import Single Recipe with Full Processing
==========================================

Imports a single recipe URL through the full pipeline:
1. Import via Mealie API
2. Parse ingredients
3. Tag with cuisine/prep requirements
4. Update RAG index

Usage:
    python import_recipe.py https://example.com/recipe-url
"""

import sys
import argparse
from mealie_client import MealieClient
from config import MEALIE_URL
from tools.logging_utils import get_logger
from utils.url_utils import normalize_url
from utils.recipe_validation import is_valid_recipe_content

logger = get_logger(__name__)


def check_duplicate(client: MealieClient, url: str) -> str | None:
    """Check if recipe URL already exists in Mealie. Returns slug if found.
    
    Uses normalized URL comparison to catch duplicates with different
    formatting (trailing slashes, www prefix, http/https variations).
    """
    print(f"🔍 Checking for duplicates...")
    
    normalized_input = normalize_url(url)
    
    # Search through recipes to find matching orgURL
    recipes = client.get_all_recipes()
    
    for recipe in recipes:
        org_url = recipe.get('orgURL', '')
        if org_url and normalize_url(org_url) == normalized_input:
            return recipe.get('slug')
    
    return None


def import_recipe(client: MealieClient, url: str, skip_duplicate_check: bool = False) -> str:
    """
    Import recipe via Mealie API with collision detection. Returns slug or raises exception.
    
    Uses import_recipe_smart() internally to prevent Mealie's slug corruption bug
    by detecting name collisions BEFORE import.
    
    Args:
        client: MealieClient instance
        url: Recipe URL to import
        skip_duplicate_check: If True, skip the expensive duplicate check.
                              Use when URLs are already pre-filtered (e.g., from sitemap diff).
    """
    from bulk_import_smart import import_recipe_smart
    
    # Check for URL duplicates first (unless pre-filtered)
    if not skip_duplicate_check:
        existing_slug = check_duplicate(client, url)
        if existing_slug:
            print(f"⚠️ Recipe already exists: {existing_slug}")
            print(f"   Skipping import to avoid duplicate.")
            return existing_slug
    
    print(f"📥 Importing recipe from: {url}")
    
    # Use smart import with collision detection
    slug, name, was_qualified = import_recipe_smart(client, url)
    
    if not slug:
        raise Exception(f"Import failed - no slug returned")
    
    if was_qualified:
        print(f"   📝 Name qualified: {name}")
    
    print(f"✅ Imported: {slug}")
    return slug


def verify_recipe_exists(client: MealieClient, slug: str) -> dict:
    """Verify recipe exists and return its data."""
    return client.get_recipe(slug)


def delete_recipe(client: MealieClient, slug: str) -> bool:
    """Delete a recipe by slug. Returns True if successful."""
    try:
        return client.delete_recipe(slug)
    except Exception as e:
        logger.warning(f"Failed to delete recipe {slug}: {e}")
        return False


def parse_recipe_ingredients(slug: str):
    """Parse ingredients for a single recipe."""
    print(f"🔍 Parsing ingredients...")
    
    import subprocess
    
    try:
        # Call mealie_parse with specific slug
        result = subprocess.run(
            ['python', 'mealie_parse.py', '--slugs', slug],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"   ✅ Parsing complete")
        else:
            print(f"   ⚠️ Parsing returned code {result.returncode}")
            if result.stderr:
                # Just show last line of error
                last_line = result.stderr.strip().split('\n')[-1]
                print(f"   {last_line[:100]}")
                
    except subprocess.TimeoutExpired:
        print(f"   ⚠️ Parsing timed out")
    except Exception as e:
        print(f"   ⚠️ Parsing error: {e}")


def tag_recipe(client: MealieClient, slug: str):
    """Tag recipe with cuisine and prep requirements."""
    print(f"🏷️ Tagging recipe...")
    
    try:
        from automatic_tagger import AutomaticTagger
        import asyncio
        
        async def do_tag():
            tagger = AutomaticTagger()
            
            # Get recipe data
            recipe = client.get_recipe(slug)
            recipe_id = recipe.get('id')
            
            # Check if already tagged
            existing_tags = [t.get('name', '') for t in recipe.get('tags', [])]
            if existing_tags:
                print(f"   ✓ Already tagged: {', '.join(existing_tags)}")
                return True
            
            # Analyze recipe
            analysis = await tagger.analyze_recipe(recipe)
            
            if analysis and analysis.recommended_tags:
                # Apply tags to Mealie
                result = tagger.apply_tags_to_mealie(recipe_id, analysis, dry_run=False)
                if result.get('tags_added'):
                    print(f"   ✅ Tagged: {', '.join(result['tags_added'])}")
                    return True
                else:
                    print(f"   ⚠️ No tags applied")
                    return False
            else:
                print(f"   ⚠️ No tags identified")
                return False
        
        asyncio.run(do_tag())
        
    except ImportError:
        print(f"   ⚠️ Tagger not available")
    except Exception as e:
        print(f"   ⚠️ Tagging error: {e}")


def update_rag_index(client: MealieClient, slug: str):
    """Update RAG index with new recipe."""
    print(f"📚 Updating search index...")
    
    try:
        from recipe_rag import RecipeRAG
        
        rag = RecipeRAG()
        
        # Get recipe data
        recipe = client.get_recipe(slug)
        
        # Index the recipe (force=True to update if exists)
        if rag.index_recipe(recipe, force=True):
            print(f"   ✅ Added to search index")
        else:
            print(f"   ⚠️ Indexing returned False")
        
    except ImportError:
        print(f"   ⚠️ RAG not available")
    except Exception as e:
        print(f"   ⚠️ Indexing error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Import a single recipe with full processing")
    parser.add_argument("url", help="Recipe URL to import")
    parser.add_argument("--skip-parse", action="store_true", help="Skip ingredient parsing")
    parser.add_argument("--skip-tag", action="store_true", help="Skip tagging")
    parser.add_argument("--skip-index", action="store_true", help="Skip RAG indexing")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"SINGLE RECIPE IMPORT")
    print(f"{'='*60}\n")
    
    # Force API mode - we create then immediately verify/read
    client = MealieClient(use_direct_db=False)
    
    try:
        # Step 1: Import via Mealie
        slug = import_recipe(client, args.url)
        
        # Step 2: Verify it exists
        recipe = verify_recipe_exists(client, slug)
        print(f"📄 Recipe: {recipe.get('name', slug)}")
        
        # Step 2.5: Validate recipe content (reject glossaries, reference pages, etc.)
        is_valid, reason = is_valid_recipe_content(recipe)
        if not is_valid:
            print(f"🗑️ Rejecting: {reason}")
            if delete_recipe(client, slug):
                print(f"   ✅ Deleted invalid import: {slug}")
            else:
                print(f"   ⚠️ Could not delete {slug} - please remove manually")
            raise Exception(f"Not a valid recipe: {reason}")
        
        # Step 3: Parse ingredients
        if not args.skip_parse:
            parse_recipe_ingredients(slug)
        
        # Step 4: Tag
        if not args.skip_tag:
            tag_recipe(client, slug)
        
        # Step 5: Update RAG index
        if not args.skip_index:
            update_rag_index(client, slug)
        
        print(f"\n{'='*60}")
        print(f"✅ COMPLETE")
        print(f"{'='*60}")
        print(f"\nView recipe: {MEALIE_URL}/g/home/r/{slug}")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
