#!/usr/bin/env python3
"""
Smart Bulk Recipe Importer for Mealie with Selective Re-parsing
===============================================================

Features:
- Pre-flight duplicate checking (prevents recipe mess)
- User confirmation before importing
- Progress tracking and detailed reporting
- Rate limiting to respect API limits
- Robust error handling
- Automatic parsing quality assessment
- Selective re-parsing for poorly parsed recipes (future feature)

Usage:
    python bulk_import_smart.py <recipe_file.txt>
    python bulk_import_smart.py mainland_chinese_recipes.txt --dry-run
    python bulk_import_smart.py recipes.txt --yes  # Skip confirmation + auto-processing

Features:
- ✅ Smart duplicate detection before import
- ✅ Automatic ingredient parsing after import
- ✅ Automatic cuisine classification and tagging
- ✅ Automatic RecipeRAG indexing for menu planning
- ✅ One-command recipe import to production-ready
"""

import sys
import os
import time
import requests
import subprocess
import asyncio
import threading
import uuid
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Set, Optional
from config import MEALIE_URL, MEALIE_TOKEN, get_mealie_headers, validate_all, get_parallelism_config, mealie_rate_limit, DATA_DIR
from panel.jobs.pipeline_state import PipelineState
from mealie_parse import parse_single_recipe_standalone
from mealie_client import MealieClient
from tools.progress_ui import ui, create_progress_bar, Timer
from tools.logging_utils import get_logger
from utils.url_utils import normalize_url
from utils.recipe_validation import is_valid_recipe_content
from prompts import SERVINGS_ESTIMATION_PROMPT, SERVINGS_ESTIMATION_SYSTEM_PROMPT

# Initialize logger for this module
logger = get_logger(__name__)


def get_existing_recipe_urls() -> Set[str]:
    """
    Get all existing recipe URLs from Mealie for duplicate checking.
    
    Uses MealieClient which queries the AUTHORITATIVE source:
    - DB mode (use_direct_db=True): Queries /mealie-data/mealie.db directly (~10ms)
    - API mode (default): Uses Mealie API pagination (~30s for large collections)
    
    Previously this used RecipeRAG's local cache which could become stale,
    causing duplicate imports when the cache wasn't synced with Mealie.
    
    URLs are normalized for consistent comparison (handles trailing slashes,
    www prefix, http/https variations).

    Returns:
        Set of normalized orgURLs (original recipe URLs)
    """
    from mealie_client import MealieClient
    
    client = MealieClient()  # Respects use_direct_db config
    try:
        mode = "DB" if client.mode == "db" else "API"
        print(f"🔍 Checking existing recipes for duplicates (Mealie {mode})...")
        
        existing_urls = client.get_all_recipe_urls()
        
        print(f"✅ Found {len(existing_urls)} existing recipe URLs in Mealie")
        return existing_urls
        
    except Exception as e:
        logger.critical(f"❌ CRITICAL FAILURE: Cannot check for duplicate recipes: {e}")
        print(f"❌ CRITICAL FAILURE: Cannot check for duplicate recipes: {e}")
        print("❌ FAST FAILURE: Duplicate checking is required for safe imports")
        print("❌ FIX REQUIRED: Ensure Mealie connection works before importing")
        sys.exit(1)  # IMMEDIATE FAILURE - duplicate checking is mandatory
    finally:
        client.close()


def read_recipe_urls(filename: str) -> List[str]:
    """
    Read recipe URLs from a text file.

    Args:
        filename: Path to the text file containing URLs

    Returns:
        List of URLs (empty lines and comments are skipped)
    """
    if not os.path.exists(filename):
        logger.error(f"❌ Error: File '{filename}' not found")
        print(f"❌ Error: File '{filename}' not found")
        print(f"   Please create a text file with one recipe URL per line")
        sys.exit(1)

    urls = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                urls.append(line)
            elif line.startswith('#'):
                continue  # Skip comments

    print(f"📄 Read {len(urls)} recipe URLs from {filename}")
    return urls


def analyze_import_list(urls: List[str], existing_urls: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Analyze URLs to import vs existing duplicates.
    
    URLs are normalized for comparison to catch duplicates with different
    formatting (trailing slashes, www prefix, http/https).

    Args:
        urls: List of URLs to potentially import
        existing_urls: Set of normalized URLs that already exist in Mealie

    Returns:
        Tuple of (to_import: List[str], duplicates: List[str])
        Note: Returns original URLs (not normalized) for import
    """
    to_import = []
    duplicates = []

    for url in urls:
        # Normalize for comparison, but keep original URL for import
        if normalize_url(url) in existing_urls:
            duplicates.append(url)
        else:
            to_import.append(url)

    return to_import, duplicates


def assess_parsing_quality(recipe_data: dict) -> str:
    """
    STRICT REQUIREMENT: Assess parsing quality and ENFORCE high standards.

    Args:
        recipe_data: Full recipe data from Mealie API

    Returns:
        "GOOD" if ingredients meet strict parsing requirements
        "POOR" if ingredients fail quality standards (will cause import failure)
        "INVALID" if content is not a valid recipe (no ingredients/instructions)
    """
    # First check if this is valid recipe content at all
    is_valid, reason = is_valid_recipe_content(recipe_data)
    if not is_valid:
        return "INVALID"
    
    ingredients = recipe_data.get("recipeIngredient", [])

    total_ingredients = len(ingredients)
    poorly_parsed = 0

    for ingredient in ingredients:
        quantity = ingredient.get("quantity", 0)
        unit = ingredient.get("unit")
        food = ingredient.get("food")

        # STRICT CRITERIA: Ingredient must have quantity, unit, AND food to be considered well-parsed
        if not (quantity > 0 and unit is not None and food is not None):
            poorly_parsed += 1

    # STRICT REQUIREMENT: ALL ingredients must be properly parsed (100% success rate)
    # No tolerance for poor parsing - quality must be perfect
    if poorly_parsed > 0:
        return "POOR"

    return "GOOD"


def get_recipe(slug: str) -> dict:
    """
    Get full recipe data by slug.
    
    IMPORTANT: Always uses API mode because this is typically called
    immediately after import for quality assessment, and DB mode may
    not see fresh writes.

    Args:
        slug: Recipe slug

    Returns:
        Full recipe data dictionary
    """
    try:
        from mealie_client import MealieClient
        client = MealieClient(use_direct_db=False)
        try:
            return client.get_recipe(slug)
        finally:
            client.close()
    except Exception as e:
        logger.critical(f"❌ CRITICAL FAILURE: Cannot fetch recipe {slug}: {e}")
        print(f"❌ CRITICAL FAILURE: Cannot fetch recipe {slug}: {e}")
        print("❌ FAST FAILURE: Recipe data access is required for quality assessment")
        sys.exit(1)  # IMMEDIATE FAILURE - cannot continue without recipe data


def import_recipe(url: str) -> Tuple[bool, str]:
    """
    Import a single recipe URL into Mealie.

    Args:
        url: Recipe URL to import

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        from mealie_client import MealieClient, MealieAPIError
        client = MealieClient()
        try:
            result = client.create_recipe_from_url(url, include_tags=False)
            # MealieClient returns recipe data dict or slug string
            if isinstance(result, dict):
                recipe_slug = result.get('slug', '')
            else:
                recipe_slug = str(result).strip().strip('"')
            
            if recipe_slug and not recipe_slug.startswith('"') and "error" not in recipe_slug.lower() and "not found" not in recipe_slug.lower():
                return True, recipe_slug
            else:
                return False, f"API returned: {recipe_slug}"
        except MealieAPIError as e:
            error_msg = f"HTTP {e.status_code}"
            if e.response_body:
                try:
                    import json
                    error_detail = json.loads(e.response_body)
                    if isinstance(error_detail, dict) and 'detail' in error_detail:
                        error_msg += f": {error_detail['detail']}"
                    elif isinstance(error_detail, str):
                        error_msg += f": {error_detail}"
                except:
                    error_msg += f": {e.response_body[:200]}"
            return False, error_msg
        finally:
            client.close()
    except requests.exceptions.Timeout:
        return False, "Timeout (recipe scraping took too long)"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def import_recipe_with_quality_check(url: str, dry_run: bool = False) -> Tuple[bool, str, str]:
    """
    Import a recipe and check parsing quality.

    Args:
        url: Recipe URL to import
        dry_run: If True, simulate without actual import

    Returns:
        Tuple of (success: bool, slug_or_message: str, quality: str)
        quality will be "GOOD", "POOR", or "UNKNOWN" (if fetch failed)
    """
    if dry_run:
        return True, "dry-run", "UNKNOWN"

    # Import the recipe
    success, result = import_recipe(url)

    if not success:
        return False, result, "UNKNOWN"

    slug = result

    # Get full recipe data to check parsing quality
    recipe_data = get_recipe(slug)
    if not recipe_data:
        return True, slug, "UNKNOWN"  # Import succeeded but couldn't check quality

    # STRICT REQUIREMENT: Assess parsing quality - poor quality causes immediate failure
    quality = assess_parsing_quality(recipe_data)

    if quality == "INVALID":
        # Not a recipe - it's a glossary, reference page, or other non-recipe content
        logger.warning(f"🗑️ INVALID: Not a recipe - {slug} has no ingredients/instructions")
        print(f"🗑️ INVALID: Not a recipe (no ingredients/instructions)")
        print(f"   Likely a glossary, reference page, or navigation page")
        # Remove the invalid content from Mealie
        try:
            from mealie_client import MealieClient
            client = MealieClient()
            try:
                client.delete_recipe(slug)
                print(f"   ✅ Removed invalid content from Mealie")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not remove invalid content: {e}")
                print(f"   ⚠️ Could not remove invalid content: {e}")
            finally:
                client.close()
        except Exception as e:
            logger.warning(f"   ⚠️ Could not remove invalid content: {e}")
            print(f"   ⚠️ Could not remove invalid content: {e}")

        return False, "Not a valid recipe (no ingredients/instructions)", quality

    if quality == "POOR":
        logger.error(f"❌ CRITICAL: Parsing quality assessment: {quality} for recipe {slug}")
        print(f"❌ CRITICAL: Parsing quality assessment: {quality}")
        print(f"❌ FAST FAILURE: Recipe {slug} has poor parsing quality - rejecting import")
        print(f"❌ Quality enforcement: Recipe must have 100% properly parsed ingredients")
        # Remove the failed recipe from Mealie since parsing failed
        try:
            from mealie_client import MealieClient
            client = MealieClient()
            try:
                client.delete_recipe(slug)
                print(f"✅ Removed poorly parsed recipe from Mealie")
            except Exception as e:
                logger.warning(f"⚠️  Could not remove failed recipe: {e}")
                print(f"⚠️  Could not remove failed recipe: {e}")
            finally:
                client.close()
        except Exception as e:
            logger.warning(f"⚠️  Could not remove failed recipe: {e}")
            print(f"⚠️  Could not remove failed recipe: {e}")

        return False, f"Rejected due to poor parsing quality", quality

    return True, slug, quality


def import_recipe_batch(urls: List[str], dry_run: bool = False) -> Dict[str, Tuple[bool, str, str]]:
    """
    Import multiple recipes with progress tracking and quality assessment.
    Now uses parallel fetching for 20x speedup.

    Args:
        urls: List of URLs to import
        dry_run: If True, simulate import without actually importing

    Returns:
        Dict mapping URL to (success, message, quality) tuples
        quality: "GOOD", "POOR", "INVALID" (not a recipe), or "UNKNOWN"
    """
    results = {}

    for i, url in enumerate(urls, 1):
        if dry_run:
            print(f"[{i}/{len(urls)}] 🔍 Would import: {url}")
            results[url] = (True, "dry-run", "UNKNOWN")
            time.sleep(0.1)  # Fast simulation
        else:
            print(f"[{i}/{len(urls)}] 📥 Importing: {url}")
            success, result, quality = import_recipe_with_quality_check(url, dry_run)
            results[url] = (success, result, quality)

            if success and quality == "POOR":
                print(f"    ⚠️  Parsing quality: {quality} - may need re-parsing")
            elif success:
                print(f"    ✅ Parsing quality: {quality}")

            time.sleep(1.5)  # Rate limiting - be nice to the API

    return results


def import_recipe_batch_parallel(urls: List[str], dry_run: bool = False) -> Dict[str, Tuple[bool, str, str]]:
    """
    Import multiple recipes with parallel processing and connection pooling.
    BEFORE: Sequential processing (5-10 minutes for 100 recipes)
    AFTER: Parallel processing (30-60 seconds for 100 recipes)

    Args:
        urls: List of URLs to import
        dry_run: If True, simulate import without actually importing

    Returns:
        Dict mapping URL to (success, message, quality) tuples
    """
    from tools.parallel_processor import ParallelProcessor

    # Get parallelism config once
    import_config = get_parallelism_config('import')
    import_workers = import_config.get('workers', 6)  # Default matches cloud_api preset

    # Initialize MealieClient
    # CRITICAL: Force API mode - after importing, we need to read back fresh data
    # and DB mode's read-only connection won't see the just-written recipes
    client = MealieClient(use_direct_db=False)

    try:
        # Check for duplicates
        print("Checking for duplicates...")
        duplicates = client.check_duplicate_urls(urls)

        # Filter to new URLs only
        new_urls = [url for url, exists in duplicates.items() if not exists]

        if dry_run:
            print(f"📊 Dry run: {len(new_urls)} new recipes, {len(urls) - len(new_urls)} duplicates")
            results = {url: (True, "dry-run", "UNKNOWN") for url in urls}
            return results

        # Import new recipes
        print(f"Importing {len(new_urls)} recipes...")
        processor = ParallelProcessor(max_workers=import_workers)

        def import_single_recipe(url):
            return import_recipe_from_url(url, client)

        results = processor.process_batch(new_urls, import_single_recipe, "Import")
        print(f"Import complete: {len(new_urls)} recipes")

        # Build final results dict
        final_results = {}
        new_url_index = 0

        for url in urls:
            if url in duplicates and duplicates[url]:
                # This was a duplicate
                final_results[url] = (False, "duplicate", "UNKNOWN")
            else:
                # This was imported (or attempted)
                result = results[new_url_index]
                success, slug, quality = result[0], result[1], result[2]  # Ignore recipe_data (4th element)
                final_results[url] = (success, slug, quality)
                new_url_index += 1

        success_count = sum(1 for r in results if r and r[0])
        print(f"✅ Import complete: {success_count}/{len(new_urls)} recipes imported")

        return final_results

    finally:
        client.close()


def import_recipe_from_url(url: str, client: 'MealieClient') -> Tuple[bool, str, str, dict]:
    """
    Import single recipe using MealieClient with pre-import collision detection.

    Uses import_recipe_smart() internally to prevent Mealie's slug corruption bug
    by detecting name collisions BEFORE import.

    Args:
        url: Recipe URL to import
        client: MealieClient instance

    Returns:
        Tuple of (success: bool, slug_or_message: str, quality: str, recipe_data: dict)
        recipe_data is passed through pipeline to avoid redundant API fetches
    """
    try:
        # Use smart import with collision detection
        slug, name, was_qualified = import_recipe_smart(client, url)
        
        if not slug:
            return False, "Import failed: no slug returned", "UNKNOWN", None
        
        if was_qualified:
            logger.info(f"📝 Name qualified to avoid collision: {name}")
        
        # Get full recipe data for quality assessment and pipeline
        try:
            full_recipe_data = client.get_recipe(slug)
        except Exception as e:
            logger.warning(f"Could not fetch full recipe data for {slug}: {e}")
            return True, slug, "UNKNOWN", None
        
        if full_recipe_data:
            quality = assess_parsing_quality(full_recipe_data)
            return True, slug, quality, full_recipe_data
        else:
            return True, slug, "UNKNOWN", None

    except Exception as e:
        # Handle MealieClientError and other exceptions
        from mealie_client import MealieClientError, MealieAPIError
        error_msg = str(e)
        if isinstance(e, MealieAPIError) and hasattr(e, 'status_code'):
            error_msg = f"HTTP {e.status_code}: {error_msg}"
        elif isinstance(e, MealieClientError):
            error_msg = f"MealieClient error: {error_msg}"
        return False, f"Import failed: {error_msg}", "UNKNOWN", None


def import_recipe_smart(
    client: 'MealieClient',
    url: str,
    skip_collision_check: bool = False
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Import a recipe with pre-import collision detection.
    
    This is the RECOMMENDED import method that prevents Mealie's slug corruption bug.
    
    Flow:
    1. Scrape URL to get recipe metadata (without creating)
    2. Check if recipe name would collide with existing recipe
    3. If collision: create with site-qualified name
    4. If no collision: create normally
    
    Args:
        client: MealieClient instance
        url: Recipe URL to import
        skip_collision_check: If True, skip collision detection (use for known-unique imports)
    
    Returns:
        Tuple of (slug, name, was_qualified)
        - slug: Created recipe slug (None if failed)
        - name: Final recipe name (may be qualified)
        - was_qualified: True if name was modified to avoid collision
    
    Raises:
        Exception: If import fails
    """
    from utils.collision_detection import should_qualify_name
    from utils.url_utils import normalize_url
    
    normalized_url = normalize_url(url)
    
    # Step 1: Pre-scrape to get recipe name
    logger.info(f"📡 Pre-scraping: {url}")
    try:
        scraped_data = client.scrape_recipe_url(url)
    except Exception as e:
        logger.error(f"❌ Failed to scrape {url}: {e}")
        raise Exception(f"Scrape failed: {e}")
    
    original_name = scraped_data.get('name', '').strip()
    if not original_name:
        logger.warning(f"⚠️ No recipe name found in scraped data for {url}")
        # Fall back to normal import
        result = client.create_recipe_from_url(url, include_tags=False)
        slug = result if isinstance(result, str) else result.get('slug')
        return (slug, None, False)
    
    # Step 2: Check for collision (unless skipped)
    was_qualified = False
    final_name = original_name
    
    if not skip_collision_check:
        was_qualified, final_name = should_qualify_name(original_name, url)
        if was_qualified:
            logger.info(f"📝 Qualifying name: '{original_name}' → '{final_name}'")
    
    # Step 3: Create recipe
    if was_qualified:
        # Create with qualified name using manual creation
        # (can't use create_from_url as it would use original name)
        logger.info(f"📥 Creating with qualified name: {final_name}")
        
        # Create stub with qualified name
        stub_result = client.create_recipe({'name': final_name})
        slug = stub_result if isinstance(stub_result, str) else stub_result.get('slug')
        
        if not slug:
            raise Exception(f"Failed to create recipe stub for {final_name}")
        
        # Convert scraped JSON-LD to Mealie update format
        update_data = _convert_scraped_to_mealie_format(scraped_data, normalized_url)
        
        # Update with full recipe data
        client.update_recipe(slug, update_data)
        logger.info(f"✅ Created: {slug} (qualified name)")
        
    else:
        # No collision - use standard import
        logger.info(f"📥 Importing normally: {final_name}")
        result = client.create_recipe_from_url(url, include_tags=False)
        slug = result if isinstance(result, str) else result.get('slug')
        logger.info(f"✅ Created: {slug}")
    
    return (slug, final_name, was_qualified)


def _convert_scraped_to_mealie_format(scraped_data: Dict, org_url: str) -> Dict:
    """
    Convert JSON-LD scraped data to Mealie's recipe update format.
    
    Args:
        scraped_data: Data from test-scrape-url endpoint (JSON-LD format)
        org_url: Original URL for orgURL field
    
    Returns:
        Dict suitable for MealieClient.update_recipe()
    """
    update = {
        'orgURL': org_url,
    }
    
    # Map JSON-LD fields to Mealie fields
    field_mapping = {
        'description': 'description',
        'prepTime': 'prepTime',
        'cookTime': 'cookTime', 
        'totalTime': 'totalTime',
        'recipeYield': 'recipeYield',
    }
    
    for ld_field, mealie_field in field_mapping.items():
        if ld_field in scraped_data and scraped_data[ld_field]:
            value = scraped_data[ld_field]
            # Handle recipeYield which may be a list
            if isinstance(value, list):
                value = value[0] if value else ''
            update[mealie_field] = value
    
    # Handle ingredients (list of strings → list of dicts)
    if 'recipeIngredient' in scraped_data:
        ingredients = []
        for ing_text in scraped_data['recipeIngredient']:
            ingredients.append({
                'note': ing_text,
                'display': ing_text,
            })
        update['recipeIngredient'] = ingredients
    
    # Handle instructions (list of HowToStep → list of dicts)
    if 'recipeInstructions' in scraped_data:
        instructions = []
        for step in scraped_data['recipeInstructions']:
            if isinstance(step, dict):
                text = step.get('text', '')
            else:
                text = str(step)
            instructions.append({'text': text})
        update['recipeInstructions'] = instructions
    
    # Handle images (list of URLs → first image)
    if 'image' in scraped_data:
        images = scraped_data['image']
        if isinstance(images, list) and images:
            update['image'] = images[0]
        elif isinstance(images, str):
            update['image'] = images
    
    # Handle nutrition
    if 'nutrition' in scraped_data and isinstance(scraped_data['nutrition'], dict):
        nutrition = scraped_data['nutrition']
        update['nutrition'] = {
            'calories': nutrition.get('calories', ''),
            'fatContent': nutrition.get('fatContent', ''),
            'carbohydrateContent': nutrition.get('carbohydrateContent', ''),
            'proteinContent': nutrition.get('proteinContent', ''),
            'fiberContent': nutrition.get('fiberContent', ''),
            'sodiumContent': nutrition.get('sodiumContent', ''),
            'sugarContent': nutrition.get('sugarContent', ''),
        }
    
    # Handle categories (recipeCategory in JSON-LD)
    if 'recipeCategory' in scraped_data:
        cats = scraped_data['recipeCategory']
        if isinstance(cats, list):
            update['recipeCategory'] = [{'name': c} for c in cats if c]
        elif cats:
            update['recipeCategory'] = [{'name': cats}]
    
    return update


def print_analysis(to_import: List[str], duplicates: List[str]):
    """Print detailed analysis of what will be imported."""
    print(f"\n📊 IMPORT ANALYSIS:")
    print(f"✅ New recipes to import: {len(to_import)}")
    print(f"⏭️  Duplicates to skip: {len(duplicates)}")

    if duplicates:
        print(f"\nDuplicates found (first 5 shown):")
        for dup in duplicates[:5]:
            print(f"  ⏭️  {dup}")
        if len(duplicates) > 5:
            print(f"  ... and {len(duplicates) - 5} more")

    if to_import:
        print(f"\nRecipes to import (first 5 shown):")
        for recipe in to_import[:5]:
            print(f"  ✅ {recipe}")
        if len(to_import) > 5:
            print(f"  ... and {len(to_import) - 5} more")


async def run_automatic_post_processing(results: Dict[str, Tuple[bool, str, str]], dry_run: bool) -> bool:
    """
    Run automatic post-processing after successful imports.

    Args:
        results: Dict of url -> (success, slug, quality) for import results
        dry_run: Whether this was a dry run

    Returns:
        True if post-processing completed successfully
    """
    if dry_run:
        return True

    successful_imports = [(url, slug) for url, (success, slug, quality) in results.items() if success]
    if not successful_imports:
        return True

    print(f"\n🔧 Starting automatic post-processing for {len(successful_imports)} recipes...")
    success = True

    try:
        # Parse ingredients for newly imported recipes only
        print("Parsing ingredients...")
        import mealie_parse

        # Extract slugs from successful imports for targeted parsing
        imported_slugs = [slug for url, slug in successful_imports]
        if imported_slugs:
            try:
                import subprocess
                # Call mealie_parse.py as subprocess to avoid event loop conflicts
                cmd = [
                    sys.executable, "mealie_parse.py",
                    "--slugs", ",".join(imported_slugs)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
                if result.returncode == 0:
                    print(f"Parsing completed for {len(imported_slugs)} imported recipes")
                else:
                    logger.error(f"❌ CRITICAL FAILURE: Parsing subprocess failed with exit code {result.returncode}")
                    print(f"❌ CRITICAL FAILURE: Parsing subprocess failed with exit code {result.returncode}")
                    print(f"❌ STDOUT: {result.stdout}")
                    print(f"❌ STDERR: {result.stderr}")
                    print("❌ FAST FAILURE: Ingredient parsing is mandatory for recipe usability")
                    sys.exit(1)  # IMMEDIATE FAILURE - parsing is required
            except subprocess.TimeoutExpired:
                print("❌ CRITICAL FAILURE: Parsing timed out after 5 minutes")
                print("❌ FAST FAILURE: Ingredient parsing is mandatory for recipe usability")
                sys.exit(1)
            except Exception as parse_error:
                print(f"❌ CRITICAL FAILURE: Parsing failed with error: {parse_error}")
                print("❌ FAST FAILURE: Ingredient parsing is mandatory for recipe usability")
                sys.exit(1)  # IMMEDIATE FAILURE - parsing is required
        else:
            print("No recipes to parse")
            success = False

        # Step 2: Enrich metadata for imported recipes (tagging + servings inference)
        print("🏷️  Step 2: Automatic cuisine and prep tagging...")
        try:
            await enrich_imported_recipes_metadata(successful_imports)
            print("✅ Metadata enrichment complete")
        except Exception as e:
            print(f"❌ Metadata enrichment failed: {e}")
            print("⚠️  Recipes imported but may lack complete metadata")
            success = False

        # Step 3: Update RecipeRAG index for new recipes (parallelized)
        print("🔍 Step 3: Updating RecipeRAG index...")
        try:
            import threading
            from recipe_rag import RecipeRAG
            rag = RecipeRAG()

            # Get indexing parallelism config
            indexing_config = get_parallelism_config('indexing')
            indexing_workers = indexing_config.get('workers', 6)

            # Thread-safe lock for index operations (ANN index is not thread-safe)
            index_lock = threading.Lock()

            # Shared client for all index workers (requests.Session with HTTPAdapter is thread-safe)
            from mealie_client import MealieClient
            shared_index_client = MealieClient(use_direct_db=False)

            def index_single_recipe(url_slug_tuple: Tuple[str, str]) -> Tuple[str, bool]:
                """
                Index a single recipe with parallel embedding generation.
                Embedding generation runs in parallel (thread-safe).
                Only ANN index storage is serialized under lock.
                """
                url, slug = url_slug_tuple
                try:
                    # Fetch recipe data (I/O bound - parallel)
                    recipe_data = fetch_recipe_data_by_slug(slug, client=shared_index_client)
                    if not recipe_data:
                        return (slug, False)
                    
                    # Generate embedding OUTSIDE lock (CPU/GPU bound - parallel, thread-safe)
                    searchable_text = rag._create_searchable_text(recipe_data)
                    embedding = rag._generate_embedding(searchable_text)
                    if embedding is None:
                        logger.warning(f"⚠️  Skipping recipe {slug} - embedding generation failed")
                        return (slug, False)
                    
                    # Only storage under lock (fast, protects ANN index)
                    with index_lock:
                        if rag.store_with_precomputed_embedding(recipe_data, embedding, force=True, auto_save=False):
                            return (slug, True)
                    
                    return (slug, False)
                except Exception as e:
                    logger.error(f"Failed to index recipe {slug}: {e}")
                    return (slug, False)

            # Process in parallel
            print(f"   Using {indexing_workers} workers for parallel indexing...")
            indexed_count = 0

            with ThreadPoolExecutor(max_workers=indexing_workers) as executor:
                futures = {executor.submit(index_single_recipe, item): item for item in successful_imports}

                for future in as_completed(futures):
                    url, slug = futures[future]
                    try:
                        result_slug, success = future.result()
                        if success:
                            indexed_count += 1
                    except Exception as e:
                        print(f"⚠️  Failed to index recipe {slug}: {e}")

            # Clean up shared client after all workers finish
            shared_index_client.close()

            # Save index once after all recipes are indexed (instead of after each one)
            if rag.ann_index is not None and indexed_count > 0:
                try:
                    rag.ann_index.save()
                    print(f"💾 Saved ANN index with {indexed_count} new recipes")
                except Exception as e:
                    print(f"❌ CRITICAL: Failed to save ANN index: {e}")
                    print(f"⚠️  {indexed_count} recipes indexed in memory but not persisted to disk")
                    print(f"⚠️  Re-run indexing to persist these recipes")
                    # Don't exit - recipes are still in database and can be re-indexed

            print(f"✅ Indexed {indexed_count}/{len(successful_imports)} recipes in RecipeRAG")

        except Exception as e:
            print(f"❌ CRITICAL FAILURE: RecipeRAG indexing failed: {e}")
            print("❌ FAST FAILURE: Recipe search indexing is required for menu planning")
            sys.exit(1)  # IMMEDIATE FAILURE - indexing is required

        # Step 3: Final verification
        print("✅ Step 3: Post-processing verification...")
        try:
            total_indexed = get_recipe_rag_count()
            print(f"   RecipeRAG now contains {total_indexed} indexed recipes")
        except:
            print("   Could not verify RecipeRAG count")

        print("🎉 Post-processing complete!")
        print("   Recipes are now ready for menu planning")

    except subprocess.TimeoutExpired:
        print("❌ Post-processing timed out (10 minutes)")
        success = False
    except Exception as e:
        print(f"❌ Post-processing failed: {e}")
        success = False

    return success


def fetch_recipe_data_by_slug(slug: str, client: 'MealieClient' = None) -> Dict:
    """Fetch full recipe data by slug for indexing.
    
    IMPORTANT: Always uses API mode because this is typically called 
    immediately after import, and DB mode may not see fresh writes.
    
    Args:
        slug: Recipe slug to fetch
        client: Optional MealieClient instance to reuse. If None, creates
                a temporary one (and closes it after). Pass a shared client
                when calling in a loop to avoid per-call overhead.
    """
    try:
        from mealie_client import MealieClient
        client_owned = client is None
        if client_owned:
            client = MealieClient(use_direct_db=False)
        try:
            return client.get_recipe(slug)
        finally:
            if client_owned:
                client.close()
    except Exception as e:
        print(f"⚠️  Could not fetch recipe {slug}: {e}")
        return None


def _get_site_display_name(url: str) -> str:
    """
    Extract a display-friendly site name from a URL.
    
    Uses centralized SITE_DISPLAY_NAMES from collision_detection module.
    """
    from utils.collision_detection import get_site_display_name
    return get_site_display_name(url)


def _fix_duplicate_name_suffix(slug: str, recipe_name: str, org_url: str) -> tuple[Optional[str], Optional[str]]:
    """
    DEPRECATED: This function is no longer used.
    
    Post-import renaming causes Mealie's slug routing bug. Use pre-import
    collision detection via import_recipe_smart() instead.
    
    Kept for reference only - will be removed in future version.
    """
    import warnings
    warnings.warn(
        "_fix_duplicate_name_suffix() is deprecated. "
        "Use import_recipe_smart() for pre-import collision detection.",
        DeprecationWarning,
        stacklevel=2
    )
    return None, None


# =============================================================================
# STANDALONE WORKER FUNCTIONS FOR STREAMING PIPELINE
# =============================================================================
# These functions are designed to be imported and used by queue workers.
# They handle all errors gracefully and never raise exceptions.
# =============================================================================

def tag_single_recipe(url: str, slug: str, recipe_data: dict = None,
                      client: 'MealieClient' = None) -> bool:
    """
    Tag a single recipe by slug.

    For use as a queue worker in streaming pipeline.

    Args:
        url: Original recipe URL (for logging/tracking)
        slug: Recipe slug to tag
        recipe_data: Optional recipe data (avoids redundant API fetch if provided)
        client: Optional MealieClient instance to reuse. Passed to AutomaticTagger
                to avoid creating a new client per call.

    Returns:
        True if tagging succeeded, False otherwise
    """
    try:
        from automatic_tagger import AutomaticTagger

        # Use provided data or fetch if needed
        if recipe_data is None:
            recipe_data = fetch_recipe_data_by_slug(slug, client=client)
        if not recipe_data:
            logger.warning(f"⚠️  tag_single_recipe: Could not fetch recipe data for {slug}")
            return False

        # Get recipe ID for tagging
        recipe_id = recipe_data.get('id')
        if not recipe_id:
            logger.warning(f"⚠️  tag_single_recipe: Recipe {slug} has no ID")
            return False

        recipe_name = recipe_data.get('name', slug)

        # Create tagger and analyze recipe (pass client to avoid duplicate MealieClient)
        tagger = AutomaticTagger(client=client)
        analysis = asyncio.run(tagger.analyze_recipe(recipe_data))

        if analysis is None:
            logger.warning(f"⚠️  tag_single_recipe: Analysis failed for {recipe_name}")
            return False

        # Apply tags to Mealie (pass recipe_data to avoid redundant lookup that fails in DB mode)
        result = tagger.apply_tags_to_mealie(recipe_id, analysis, recipe_data=recipe_data)

        # Check for errors in result
        if result.get('errors'):
            logger.warning(f"⚠️  tag_single_recipe: Tag application had errors for {recipe_name}: {result['errors']}")
            # Still return True if some tags were added successfully
            return len(result.get('tags_added', [])) > 0

        if result.get('tags_added'):
            logger.info(f"✅ tag_single_recipe: Tagged {recipe_name} with {len(result['tags_added'])} tags")
        else:
            logger.info(f"ℹ️  tag_single_recipe: No new tags needed for {recipe_name}")

        return True

    except Exception as e:
        logger.error(f"❌ tag_single_recipe: Failed to tag recipe {slug}: {e}")
        return False


# Thread-safe lock for RAG index operations (shared across workers)
_rag_index_lock = threading.Lock()


def index_single_recipe_worker(slug: str, rag: 'RecipeRAG' = None, recipe_data: dict = None) -> bool:
    """
    Index a single recipe in RAG system.

    For use as a queue worker in streaming pipeline.
    Generates embedding OUTSIDE lock, stores INSIDE lock.

    Args:
        slug: Recipe slug to index
        rag: Optional RAG instance (creates one if not provided)
        recipe_data: Optional recipe data (avoids redundant API fetch if provided)

    Returns:
        True if indexing succeeded, False otherwise
    """
    try:
        from recipe_rag import RecipeRAG

        # Create RAG instance if not provided
        if rag is None:
            rag = RecipeRAG()

        # Use provided data or fetch if needed
        if recipe_data is None:
            recipe_data = fetch_recipe_data_by_slug(slug)
        if not recipe_data:
            logger.warning(f"⚠️  index_single_recipe_worker: Could not fetch recipe data for {slug}")
            return False

        recipe_name = recipe_data.get('name', slug)

        # Generate embedding OUTSIDE lock (CPU/GPU bound - thread-safe, can run in parallel)
        searchable_text = rag._create_searchable_text(recipe_data)
        embedding = rag._generate_embedding(searchable_text)

        if embedding is None:
            logger.warning(f"⚠️  index_single_recipe_worker: Embedding generation failed for {recipe_name}")
            return False

        # Store in RAG index INSIDE lock (protects ANN index which is not thread-safe)
        with _rag_index_lock:
            success = rag.store_with_precomputed_embedding(
                recipe_data,
                embedding,
                force=True,
                auto_save=False  # Caller should save index after batch completes
            )

        if success:
            logger.info(f"✅ index_single_recipe_worker: Indexed {recipe_name}")
        else:
            logger.warning(f"⚠️  index_single_recipe_worker: Storage failed for {recipe_name}")

        return success

    except Exception as e:
        logger.error(f"❌ index_single_recipe_worker: Failed to index recipe {slug}: {e}")
        return False


def get_recipe_rag_count() -> int:
    """Get total recipes in RecipeRAG index."""
    try:
        from recipe_rag import RecipeRAG
        rag = RecipeRAG()
        return rag.get_total_recipes()
    except:
        return 0


def streaming_bulk_import(
    urls: List[str],
    job_id: Optional[str] = None,
    dry_run: bool = False
) -> 'PipelineState':
    """
    Streaming pipeline: import → parse → tag → index
    Each phase pulls from previous queue, pushes to next.

    Args:
        urls: List of recipe URLs to import
        job_id: Optional job ID for state persistence (generates UUID if None)
        dry_run: If True, skip actual imports

    Returns:
        PipelineState with results
    """
    # Initialize state
    if job_id is None:
        job_id = str(uuid.uuid4())

    state = PipelineState(job_id)
    if state.load():
        logger.info(f"Resuming job {job_id} with {len(state.results)} existing recipes")
    state.add_urls(urls)
    state.save()

    # Create queues - pass recipe_data through to avoid redundant API fetches
    import_queue = Queue()    # URLs to import
    parse_queue = Queue()     # (url, slug, recipe_data) - data passed from import
    tag_queue = Queue()       # (url, slug, recipe_data) - data passed from parse
    index_queue = Queue()     # (url, slug, recipe_data) - data passed from tag

    # Sentinel
    DONE = object()

    # Circuit breaker - stop pipeline after 20 consecutive failures
    stop_event = threading.Event()
    consecutive_failures = [0]  # List to allow modification in nested functions
    failure_lock = threading.Lock()
    FAILURE_THRESHOLD = 20

    def record_success():
        """Reset consecutive failure counter on any success."""
        with failure_lock:
            consecutive_failures[0] = 0

    def record_failure(phase: str) -> bool:
        """Increment failure counter. Returns True if threshold hit."""
        with failure_lock:
            consecutive_failures[0] += 1
            if consecutive_failures[0] >= FAILURE_THRESHOLD:
                if not stop_event.is_set():
                    print(f"\n{'='*50}")
                    print(f"🛑 PIPELINE STOPPED: {FAILURE_THRESHOLD} consecutive failures")
                    print(f"   Last failure in: {phase} phase")
                    print(f"   Something is broken. Check Mealie and try again.")
                    print(f"{'='*50}\n")
                    logger.error(f"Pipeline stopped: {FAILURE_THRESHOLD} consecutive failures in {phase}")
                stop_event.set()
                return True
        return False

    # Load pending URLs
    for url in state.get_pending_for_phase('import'):
        import_queue.put(url)

    # Get worker counts from config
    import_workers = get_parallelism_config('import').get('workers', 6)
    parse_workers = get_parallelism_config('parsing').get('workers', 6)
    tag_workers = get_parallelism_config('tagging').get('workers', 10)
    index_workers = get_parallelism_config('indexing').get('workers', 6)

    # Create MealieClient for import workers
    # CRITICAL: Force API mode - after importing, we need to read back fresh data
    # and DB mode's read-only connection won't see the just-written recipes
    client = MealieClient(use_direct_db=False)

    # Add sentinels for import workers
    for _ in range(import_workers):
        import_queue.put(DONE)

    # Import worker
    def import_worker():
        while True:
            if stop_event.is_set():
                break
            item = import_queue.get()
            if item is DONE or stop_event.is_set():
                break
            url = item
            state.update_recipe(url, status='importing', phase='import')
            state.save()

            success, slug, quality, recipe_data = import_recipe_from_url(url, client)

            if success:
                record_success()
                state.update_recipe(url, status='imported', slug=slug, quality=quality)
                print(f"📥 Imported: {slug}")
                parse_queue.put((url, slug, recipe_data))
            else:
                state.update_recipe(url, status='failed', error=slug or 'Import failed', phase='import')
                print(f"❌ Import failed: {url[:60]}...")
                record_failure('import')
            state.save()

    # Parse worker
    def parse_worker():
        # Single client for this worker's lifetime (API mode for fresh writes)
        from mealie_client import MealieClient
        parse_client = MealieClient(use_direct_db=False)
        try:
            while True:
                if stop_event.is_set():
                    break
                item = parse_queue.get()
                if item is DONE:
                    if not stop_event.is_set():
                        parse_queue.put(DONE)  # Pass to next worker
                    break
                if stop_event.is_set():
                    break
                url, slug, recipe_data = item
                state.update_recipe(url, status='parsing', phase='parsing')
                state.save()

                success = parse_single_recipe_standalone(slug)

                if success:
                    record_success()
                    state.update_recipe(url, status='parsed')
                    print(f"🔍 Parsed: {slug}")
                    # Re-fetch data since parsing may have modified it
                    # (parsed ingredients added) - this is the only re-fetch needed
                    updated_data = fetch_recipe_data_by_slug(slug, client=parse_client) or recipe_data
                    tag_queue.put((url, slug, updated_data))
                else:
                    state.update_recipe(url, status='failed', error='Parsing failed', phase='parsing')
                    print(f"❌ Parse failed: {slug}")
                    record_failure('parsing')
                state.save()
        finally:
            parse_client.close()

    # Tag worker
    def tag_worker():
        # Single client for this worker's lifetime (API mode for fresh writes)
        from mealie_client import MealieClient
        tag_client = MealieClient(use_direct_db=False)
        try:
            while True:
                if stop_event.is_set():
                    break
                item = tag_queue.get()
                if item is DONE:
                    if not stop_event.is_set():
                        tag_queue.put(DONE)
                    break
                if stop_event.is_set():
                    break
                url, slug, recipe_data = item
                state.update_recipe(url, status='tagging', phase='tagging')
                state.save()

                success = tag_single_recipe(url, slug, recipe_data, client=tag_client)

                if success:
                    record_success()
                    state.update_recipe(url, status='tagged')
                    print(f"🏷️  Tagged: {slug}")
                    index_queue.put((url, slug, recipe_data))
                else:
                    state.update_recipe(url, status='failed', error='Tagging failed', phase='tagging')
                    print(f"❌ Tag failed: {slug}")
                    record_failure('tagging')
                state.save()
        finally:
            tag_client.close()

    # Create shared RAG instance for all index workers (avoids reloading 2GB model)
    from recipe_rag import RecipeRAG
    shared_rag = RecipeRAG()
    print("📇 RAG index loaded for indexing phase")

    # Index worker
    def index_worker():
        while True:
            if stop_event.is_set():
                break
            item = index_queue.get()
            if item is DONE or stop_event.is_set():
                break
            url, slug, recipe_data = item
            state.update_recipe(url, status='indexing', phase='indexing')
            state.save()

            success = index_single_recipe_worker(slug, rag=shared_rag, recipe_data=recipe_data)

            if success:
                record_success()
                state.update_recipe(url, status='indexed')
                print(f"📇 Indexed: {slug}")
            else:
                state.update_recipe(url, status='failed', error='Indexing failed', phase='indexing')
                print(f"❌ Index failed: {slug}")
                record_failure('indexing')
            state.save()

    # Execute pipeline
    total_workers = import_workers + parse_workers + tag_workers + index_workers

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        # Start all workers
        import_futures = [executor.submit(import_worker) for _ in range(import_workers)]
        parse_futures = [executor.submit(parse_worker) for _ in range(parse_workers)]
        tag_futures = [executor.submit(tag_worker) for _ in range(tag_workers)]
        index_futures = [executor.submit(index_worker) for _ in range(index_workers)]

        # Wait for imports to finish, then signal parse
        for f in import_futures:
            f.result()
        parse_queue.put(DONE)

        # Wait for parsing to finish, then signal tag
        for f in parse_futures:
            f.result()
        tag_queue.put(DONE)

        # Wait for tagging to finish, then signal index
        for f in tag_futures:
            f.result()
        for _ in range(index_workers):
            index_queue.put(DONE)

        # Wait for indexing to finish
        for f in index_futures:
            f.result()

    # Save final RAG index (use shared instance)
    try:
        if shared_rag.ann_index is not None:
            shared_rag.ann_index.save()
            logger.info("RAG index saved")
    except Exception as e:
        logger.warning(f"Failed to save RAG index: {e}")

    # Clean up client
    client.close()

    state.save()

    # Print summary
    summary = state.get_summary()
    print(f"\n{'='*50}")
    if stop_event.is_set():
        print(f"⚠️  PIPELINE STOPPED (circuit breaker triggered)")
    else:
        print(f"✅ PIPELINE COMPLETE")
    print(f"{'='*50}")
    print(f"   Completed: {summary['completed']}")
    print(f"   Failed:    {summary['failed']}")
    print(f"   Total:     {summary['total']}")
    print(f"{'='*50}")
    if stop_event.is_set():
        logger.warning(f"Pipeline stopped by circuit breaker: {summary['completed']} completed, {summary['failed']} failed")
    else:
        logger.info(f"Pipeline complete: {summary['completed']} completed, {summary['failed']} failed")

    return state


# =============================================================================
# BATCHED BULK IMPORT (New Implementation)
# =============================================================================

def batched_bulk_import(
    urls: List[str],
    job_id: Optional[str] = None,
    batch_size: int = 100
) -> 'PipelineState':
    """
    Batched sequential pipeline: import → parse → tag → index
    
    Each phase completes for entire batch before next phase starts.
    Uses Mealie bulk endpoint for efficient imports.
    
    Args:
        urls: List of recipe URLs to import
        job_id: Optional job ID for state persistence
        batch_size: URLs per batch (default 100, Mealie handles internally)
        
    Returns:
        PipelineState with results
    """
    # Initialize state
    if job_id is None:
        job_id = str(uuid.uuid4())[:8]
    
    state = PipelineState(job_id)
    if state.load():
        logger.info(f"Resuming job {job_id} with {len(state.results)} existing recipes")
    state.add_urls(urls)
    state.save()
    
    # Dedupe against local cache AND dedupe the input URLs themselves
    existing_urls = get_existing_recipe_urls()
    
    # CRITICAL: Deduplicate input URLs to prevent importing same URL multiple times
    # Sitemaps can contain duplicate URLs, and without deduplication we'd send
    # duplicates to Mealie's bulk import endpoint, creating duplicate recipes
    seen_urls = set()
    urls_to_import = []
    for url in urls:
        normalized = normalize_url(url)
        if normalized not in existing_urls and normalized not in seen_urls:
            seen_urls.add(normalized)
            urls_to_import.append(url)
    
    if not urls_to_import:
        print("✅ All URLs already imported")
        return state
    
    print(f"📥 Importing {len(urls_to_import)} new recipes in batches of {batch_size}")
    
    # Process in batches
    total_imported = 0
    for batch_start in range(0, len(urls_to_import), batch_size):
        batch_urls = urls_to_import[batch_start:batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(urls_to_import) + batch_size - 1) // batch_size
        
        print(f"\n{'='*50}")
        print(f"BATCH {batch_num}/{total_batches} ({len(batch_urls)} recipes)")
        print(f"{'='*50}")
        
        # Phase 1: Import batch via Mealie bulk endpoint
        print(f"\n📥 Phase 1: Importing...")
        slugs, url_to_slug = _import_batch_bulk(batch_urls, state)
        
        if not slugs:
            print(f"⚠️ No recipes imported in this batch")
            continue
        
        print(f"   ✅ Imported {len(slugs)} recipes")
        total_imported += len(slugs)
        
        # Phase 2: Parse batch
        print(f"\n🔍 Phase 2: Parsing ingredients...")
        _parse_batch(slugs, state)
        
        # Filter to only successfully parsed recipes - don't waste time on garbage
        parsed_slugs = _get_slugs_with_status(state, slugs, 'parsed')
        failed_parse = len(slugs) - len(parsed_slugs)
        if failed_parse > 0:
            print(f"   ⚠️ {failed_parse} recipes failed parsing - skipping from pipeline")
            # Clean up failed recipes from Mealie (they're garbage without parsed ingredients)
            _cleanup_failed_recipes(state, slugs, 'parsing')
        
        if not parsed_slugs:
            print(f"   ⚠️ No recipes to process after parsing")
            state.save()
            continue
        
        # Phase 3: Tag batch (only parsed recipes)
        print(f"\n🏷️ Phase 3: Tagging...")
        _tag_batch(parsed_slugs, state)
        
        # Filter to only successfully tagged recipes
        tagged_slugs = _get_slugs_with_status(state, parsed_slugs, 'tagged')
        failed_tag = len(parsed_slugs) - len(tagged_slugs)
        if failed_tag > 0:
            print(f"   ⚠️ {failed_tag} recipes failed tagging - skipping from indexing")
        
        if not tagged_slugs:
            print(f"   ⚠️ No recipes to index after tagging")
            state.save()
            continue
        
        # Phase 4: Index batch (only tagged recipes)
        print(f"\n📚 Phase 4: Indexing...")
        _index_batch(tagged_slugs, state)
        
        state.save()
    
    # Final summary
    summary = state.get_summary()
    print(f"\n{'='*50}")
    print(f"✅ IMPORT COMPLETE")
    print(f"{'='*50}")
    print(f"   Total imported: {total_imported}")
    print(f"   Completed: {summary['completed']}")
    print(f"   Failed: {summary['failed']}")
    
    return state


def _import_batch_bulk(urls: List[str], state: 'PipelineState') -> Tuple[List[str], Dict[str, str]]:
    """
    Import URLs via Mealie bulk endpoint.
    
    POST /api/recipes/create/url/bulk
    Poll until complete, extract successful slugs.
    
    Returns:
        Tuple of (list of slugs, dict mapping url to slug)
    """
    # Mark all as importing
    for url in urls:
        state.update_recipe(url, status='importing', phase='import')
    
    # Build bulk request
    payload = {"imports": [{"url": url} for url in urls]}
    
    try:
        with mealie_rate_limit():
            response = requests.post(
                f"{MEALIE_URL}/api/recipes/create/url/bulk",
                json=payload,
                headers=get_mealie_headers(),
                timeout=30
            )
        
        if response.status_code != 202:
            logger.error(f"❌ Bulk import failed: {response.status_code}")
            for url in urls:
                state.update_recipe(url, status='failed', error=f"HTTP {response.status_code}", phase='import')
            return [], {}
        
        report_id = response.json().get("reportId")
        if not report_id:
            logger.error("❌ No reportId in bulk response")
            return [], {}
        
        print(f"   Polling report {report_id}...")
        
    except Exception as e:
        logger.error(f"❌ Bulk import request failed: {e}")
        for url in urls:
            state.update_recipe(url, status='failed', error=str(e), phase='import')
        return [], {}
    
    # Poll until complete with timeout protection
    poll_count = 0
    MAX_POLL_ATTEMPTS = 300  # 5 minutes at 1s intervals
    MAX_TOTAL_TIMEOUT = 600  # 10 minutes total timeout
    poll_start_time = time.time()
    
    while True:
        poll_count += 1
        elapsed_time = time.time() - poll_start_time
        
        # Check timeout conditions
        if poll_count > MAX_POLL_ATTEMPTS:
            logger.error(f"❌ Bulk import polling exceeded {MAX_POLL_ATTEMPTS} attempts")
            print(f"   ❌ TIMEOUT: Polling exceeded {MAX_POLL_ATTEMPTS} attempts")
            for url in urls:
                state.update_recipe(url, status='failed', error='Polling timeout exceeded', phase='import')
            return [], {}
        
        if elapsed_time > MAX_TOTAL_TIMEOUT:
            logger.error(f"❌ Bulk import polling exceeded {MAX_TOTAL_TIMEOUT}s total timeout")
            print(f"   ❌ TIMEOUT: Total polling time exceeded {MAX_TOTAL_TIMEOUT}s")
            for url in urls:
                state.update_recipe(url, status='failed', error='Total timeout exceeded', phase='import')
            return [], {}
        
        try:
            with mealie_rate_limit():
                report_response = requests.get(
                    f"{MEALIE_URL}/api/groups/reports/{report_id}",
                    headers=get_mealie_headers(),
                    timeout=30
                )
            
            if report_response.status_code != 200:
                logger.warning(f"⚠️ Report fetch failed: {report_response.status_code}")
                time.sleep(2)
                continue
            
            report = report_response.json()
            status = report.get("status", "unknown")
            entries = report.get("entries", [])
            
            success_count = sum(1 for e in entries if e.get("success"))
            print(f"   [{poll_count}] Status: {status} | {success_count}/{len(urls)} succeeded | {elapsed_time:.0f}s elapsed")
            
            if status != "in-progress":
                break
            
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"⚠️ Poll error: {e}")
            time.sleep(2)
    
    # Process results - first check which URLs succeeded
    # Try to match entries to URLs by URL field first, fall back to index if not available
    successful_urls = []
    
    # Build URL lookup for matching by URL field
    normalized_url_map = {normalize_url(url): url for url in urls}
    
    for i, entry in enumerate(entries):
        # Try to match by URL field first (more reliable than index)
        entry_url = entry.get("url") or entry.get("originalUrl")
        if entry_url:
            normalized_entry_url = normalize_url(entry_url)
            url = normalized_url_map.get(normalized_entry_url)
        else:
            # Fall back to index-based matching if no URL in entry
            url = urls[i] if i < len(urls) else None
        
        if entry.get("success"):
            # Check if slug is directly in response (some Mealie versions)
            slug = entry.get("slug") or entry.get("recipeSlug")
            if slug and url:
                successful_urls.append((url, slug))
            elif url:
                successful_urls.append((url, None))  # Need to look up
        else:
            error = entry.get("message") or entry.get("exception") or "Unknown error"
            if url:
                state.update_recipe(url, status='failed', error=error, phase='import')
                logger.warning(f"   ❌ Failed: {url[:50]}... - {error}")
    
    # If we have successful imports but no slugs, look them up by URL
    # This is ONE API call to match all successful URLs
    slugs = []
    url_to_slug = {}
    
    urls_needing_lookup = [url for url, slug in successful_urls if slug is None]
    
    if urls_needing_lookup:
        # Query recently created recipes and match by orgURL (ONE request)
        url_to_recipe = _lookup_recipes_by_urls(urls_needing_lookup)
        
        # Retry logic: if lookup returns empty but we have URLs needing lookup, retry once after 2s
        if not url_to_recipe and urls_needing_lookup:
            logger.warning(f"⚠️ Recipe lookup returned empty for {len(urls_needing_lookup)} URLs, retrying in 2s...")
            print(f"   ⚠️ Recipe lookup returned empty, retrying...")
            time.sleep(2)
            url_to_recipe = _lookup_recipes_by_urls(urls_needing_lookup)
            
            if not url_to_recipe:
                logger.error(f"❌ CRITICAL: Recipe lookup still empty after retry for {len(urls_needing_lookup)} URLs")
                print(f"   ❌ CRITICAL: Recipe lookup failed after retry - {len(urls_needing_lookup)} recipes imported but not found")
                print(f"   ❌ This may indicate a Mealie API issue or indexing delay")
        
        for url, slug in successful_urls:
            if slug is None:
                # Look up from our query
                recipe_info = url_to_recipe.get(normalize_url(url))
                if recipe_info:
                    slug = recipe_info.get('slug')
            
            if slug:
                slugs.append(slug)
                url_to_slug[url] = slug
                state.update_recipe(url, status='imported', slug=slug, phase='import')
            else:
                # Import succeeded but couldn't find the recipe - this shouldn't happen
                logger.error(f"   ❌ CRITICAL: Import succeeded but recipe not found: {url[:50]}...")
                print(f"   ❌ Recipe not found after import: {url[:50]}...")
                state.update_recipe(url, status='failed', error='Recipe not found after import', phase='import')
    else:
        # All slugs were in the response directly
        for url, slug in successful_urls:
            if slug:
                slugs.append(slug)
                url_to_slug[url] = slug
                state.update_recipe(url, status='imported', slug=slug, phase='import')
    
    return slugs, url_to_slug


def _lookup_recipes_by_urls(urls: List[str]) -> Dict[str, dict]:
    """
    Look up recipes by their original URLs.
    
    Makes ONE API call to get recently created recipes, then matches by orgURL.
    This is needed because Mealie's bulk import response doesn't include slugs.
    
    IMPORTANT: Always uses API mode (not DB mode) because this is called immediately
    after bulk import, and DB mode's read-only connection may not see fresh writes.
    
    Args:
        urls: List of original recipe URLs to look up
        
    Returns:
        Dict mapping normalized URL to recipe info {slug, name, id}
    """
    # Normalize URLs for matching
    normalized_urls = {normalize_url(url) for url in urls}
    
    try:
        # Query recently created recipes (sorted by newest first)
        # Get enough to cover our batch plus some buffer
        # CRITICAL: Force API mode - DB mode won't see freshly imported recipes
        from mealie_client import MealieClient
        client = MealieClient(use_direct_db=False)
        try:
            # Get all recipes and sort by creation date (MealieClient doesn't support ordering params)
            # We'll get all and filter/sort in memory
            all_recipes = client.get_all_recipes()
            # Sort by dateAdded (most recent first) - approximate since we can't order via API
            items = sorted(all_recipes, key=lambda r: r.get('dateAdded', ''), reverse=True)[:len(urls) * 2]
        finally:
            client.close()
        
        # Build lookup by orgURL
        url_to_recipe = {}
        for recipe in items:
            org_url = recipe.get('orgURL')
            if org_url:
                normalized = normalize_url(org_url)
                if normalized in normalized_urls:
                    url_to_recipe[normalized] = {
                        'slug': recipe.get('slug'),
                        'name': recipe.get('name'),
                        'id': recipe.get('id')
                    }
        
        logger.info(f"Looked up {len(url_to_recipe)}/{len(urls)} recipes by URL")
        return url_to_recipe
        
    except Exception as e:
        logger.error(f"Failed to look up recipes by URL: {e}")
        return {}


def _parse_batch(slugs: List[str], state: 'PipelineState') -> None:
    """Parse ingredients for a batch of recipes."""
    # Build slug_to_url mapping once at start to avoid O(n²) lookups
    slug_to_url = _build_slug_to_url_map(state, slugs)
    
    for slug in slugs:
        try:
            # Use pre-built mapping instead of repeated linear search
            url = slug_to_url.get(slug)
            if url:
                state.update_recipe(url, status='parsing', phase='parsing')
            
            success = parse_single_recipe_standalone(slug)
            
            if success and url:
                state.update_recipe(url, status='parsed')
                print(f"   ✅ Parsed: {slug}")
            elif url:
                state.update_recipe(url, status='failed', error='Parsing failed', phase='parsing')
                print(f"   ❌ Parse failed: {slug}")
                
        except Exception as e:
            logger.warning(f"⚠️ Parse error for {slug}: {e}")


def _tag_batch(slugs: List[str], state: 'PipelineState') -> None:
    """Tag a batch of recipes."""
    from mealie_client import MealieClient
    
    # Build slug_to_url mapping once at start to avoid O(n²) lookups
    slug_to_url = _build_slug_to_url_map(state, slugs)
    
    # Single client for the entire batch (API mode for fresh writes)
    client = MealieClient(use_direct_db=False)
    try:
        for slug in slugs:
            try:
                # Use pre-built mapping instead of repeated linear search
                url = slug_to_url.get(slug)
                if url:
                    state.update_recipe(url, status='tagging', phase='tagging')
                
                # Fetch recipe data (we're already doing this for tagging - capture name here)
                recipe_data = fetch_recipe_data_by_slug(slug, client=client)
                if not recipe_data:
                    continue
                
                # Capture recipe name from data we already fetched (no extra API call)
                recipe_name = recipe_data.get('name')
                org_url = recipe_data.get('orgURL') or url
                
                # NOTE: Collision detection now happens at import time via import_recipe_smart()
                # Post-import renaming is deprecated due to Mealie's slug routing bug.
                # Recipes should never have (1) suffix if imported via smart import.
                
                if url and recipe_name:
                    state.update_recipe(url, name=recipe_name)
                
                success = tag_single_recipe(url or "", slug, recipe_data, client=client)
                
                if success and url:
                    state.update_recipe(url, status='tagged')
                    print(f"   ✅ Tagged: {recipe_name or slug}")
                elif url:
                    state.update_recipe(url, status='failed', error='Tagging failed', phase='tagging')
                    
            except Exception as e:
                logger.warning(f"⚠️ Tag error for {slug}: {e}")
    finally:
        client.close()


def _index_batch(slugs: List[str], state: 'PipelineState') -> None:
    """Index a batch of recipes using batched embedding generation."""
    from mealie_client import MealieClient
    from recipe_rag import RecipeRAG
    
    # Build slug_to_url mapping once at start to avoid O(n²) lookups
    slug_to_url = _build_slug_to_url_map(state, slugs)
    
    # Fetch all recipe data with a single shared client (API mode for fresh writes)
    recipes = []
    client = MealieClient(use_direct_db=False)
    try:
        for slug in slugs:
            url = slug_to_url.get(slug)
            if url:
                state.update_recipe(url, status='indexing', phase='indexing')
            
            recipe_data = fetch_recipe_data_by_slug(slug, client=client)
            if recipe_data:
                recipes.append(recipe_data)
    finally:
        client.close()
    
    if not recipes:
        return
    
    # Use batch indexing with exception handling
    try:
        rag = RecipeRAG()
        success_count = rag.index_recipes_batch(recipes, force=True)
        
        print(f"   ✅ Indexed {success_count}/{len(recipes)} recipes")
        
        # Update state for successful indexes
        for recipe in recipes:
            slug = recipe.get('slug')
            url = slug_to_url.get(slug)
            if url:
                state.update_recipe(url, status='indexed')
                
    except Exception as e:
        error_msg = f"RAG indexing failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        print(f"   ❌ CRITICAL: {error_msg}")
        
        # Mark all recipes in this batch as failed
        for recipe in recipes:
            slug = recipe.get('slug')
            url = slug_to_url.get(slug)
            if url:
                state.update_recipe(url, status='failed', error=error_msg, phase='indexing')
        
        print(f"   ❌ Marked {len(recipes)} recipes as failed in indexing phase")


def _find_url_for_slug(state: 'PipelineState', slug: str) -> Optional[str]:
    """Find the original URL for a recipe slug in the pipeline state.
    
    Note: For batch operations, use _build_slug_to_url_map() instead to avoid O(n²) lookups.
    """
    for url, result in state.results.items():
        if result.slug == slug:
            return url
    return None


def _build_slug_to_url_map(state: 'PipelineState', slugs: List[str]) -> Dict[str, str]:
    """
    Build a mapping from slug to URL for a batch of slugs.
    
    This avoids O(n²) performance when looking up URLs for multiple slugs
    by doing a single pass through the state results.
    
    Args:
        state: PipelineState with results
        slugs: List of slugs to build mapping for
        
    Returns:
        Dict mapping slug to original URL
    """
    slug_set = set(slugs)
    slug_to_url = {}
    
    for url, result in state.results.items():
        if result.slug in slug_set:
            slug_to_url[result.slug] = url
            # Early exit if we've found all slugs
            if len(slug_to_url) == len(slug_set):
                break
    
    return slug_to_url


def _get_slugs_with_status(state: 'PipelineState', slugs: List[str], required_status: str) -> List[str]:
    """
    Get CURRENT slugs for recipes with the required status.
    
    This prevents garbage recipes (failed parsing) from polluting later phases.
    
    NOTE: Returns the CURRENT slugs from state, which may differ from input slugs
    if recipes were renamed during tagging (Mealie auto-updates slugs on name change).
    
    The input `slugs` parameter serves as a scope limiter - only recipes that
    originally had one of these slugs will be considered. But the RETURNED
    slugs are the current values (which may have changed due to renames).
    
    Args:
        state: PipelineState with results
        slugs: List of original slugs in this batch (may be outdated after rename)
        required_status: Status that recipes must have (e.g., 'parsed', 'tagged')
        
    Returns:
        List of CURRENT slugs that have the required status
    """
    # Simple approach: collect all recipes with required status
    # The pipeline ensures only current batch recipes are in state with this status
    result_slugs = []
    for url, result in state.results.items():
        if result.status == required_status and result.slug:
            result_slugs.append(result.slug)
    
    return result_slugs
    return result_slugs


def _cleanup_failed_recipes(state: 'PipelineState', slugs: List[str], failed_phase: str) -> None:
    """
    Delete recipes that failed from Mealie to avoid garbage accumulation.
    
    Recipes that fail parsing are useless (no structured ingredients) and should
    be removed rather than left as trash in the database.
    
    Args:
        state: PipelineState with results
        slugs: List of all slugs in the batch
        failed_phase: Phase where failures occurred (for filtering)
    """
    from mealie_client import MealieClient
    
    client = MealieClient()
    try:
        for slug in slugs:
            for url, result in state.results.items():
                if result.slug == slug and result.status == 'failed' and result.phase == failed_phase:
                    # Delete from Mealie
                    try:
                        client.delete_recipe(slug)
                        print(f"   🗑️ Deleted failed recipe: {slug}")
                        logger.info(f"Deleted failed recipe {slug} from Mealie")
                    except Exception as e:
                        logger.warning(f"Could not delete failed recipe {slug}: {e}")
                    break
    finally:
        client.close()


def print_final_report(results: Dict[str, Tuple[bool, str, str]], duplicates: List[str], dry_run: bool, skip_post_process: bool = False):
    """Print comprehensive final report."""
    print(f"\n{'🎭 DRY RUN COMPLETE!' if dry_run else '🎉 IMPORT COMPLETE!'}")

    if not dry_run:
        successful = sum(1 for success, _, _ in results.values() if success)
        failed = len(results) - successful

        # Count parsing quality
        good_quality = sum(1 for success, _, quality in results.values()
                          if success and quality == "GOOD")
        poor_quality = sum(1 for success, _, quality in results.values()
                          if success and quality == "POOR")

        print(f"✅ Successfully imported: {successful}")
        print(f"❌ Failed to import: {failed}")
        print(f"⏭️  Skipped duplicates: {len(duplicates)}")

        if successful > 0:
            print(f"📊 Parsing quality assessment:")
            print(f"   ✅ Good parsing: {good_quality}")
            print(f"   ⚠️  Poor parsing: {poor_quality}")
            print(f"   ❓ Unknown quality: {successful - good_quality - poor_quality}")

        if failed > 0:
            print(f"\nFailed imports:")
            for url, (success, message, _) in results.items():
                if not success:
                    print(f"  ❌ {url}")
                    print(f"     Reason: {message}")

    # Run automatic post-processing for all successful imports
    if not dry_run and not skip_post_process:
        successful_imports = sum(1 for success, _, _ in results.values() if success)
        if successful_imports > 0:
            import asyncio
            post_processing_success = asyncio.run(run_automatic_post_processing(results, dry_run))
            if post_processing_success:
                print(f"\n🎯 All {successful_imports} recipes meet strict quality standards!")
                print("   ✅ Ingredients perfectly parsed and scaled to 4 servings")
                print("   ✅ Cuisine and prep requirements automatically tagged")
                print("   ✅ RecipeRAG index updated for semantic search")
                print("   ✅ Quality enforcement: 100% parsing success rate achieved")
            else:
                print(f"\n⚠️  Post-processing completed with issues.")
                print("   Recipes may still be usable but check parsing quality.")

        # NOTE: Quality enforcement happens during post-processing (parsing step)
        # Initial import quality assessment is before parsing - ignore it here
        # Post-processing already validated everything and would have exited if parsing failed
    elif skip_post_process:
        print("\n⏭️  Skipping automatic post-processing (--no-post-process flag)")
        print("   Run parsing, tagging, and indexing separately")

    print(f"\nTotal recipes in import list: {len(results) + len(duplicates)}")
    print(f"Recipes processed: {len(results)}")


def confirm_import(to_import: List[str], duplicates: List[str], dry_run: bool, skip_confirmation: bool = False) -> bool:
    """Get user confirmation before proceeding with import."""
    if dry_run:
        print("\n🔍 DRY RUN MODE - No actual imports will be made")
        return True

    if skip_confirmation:
        print(f"\n🚀 Auto-confirming import of {len(to_import)} recipes (--yes flag)")
        return True

    if not to_import:
        print("\n⚠️  No new recipes to import (all are duplicates)")
        return False

    while True:
        try:
            response = input(f"\n🚀 Proceed with importing {len(to_import)} recipes? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                print("Import cancelled by user")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
        except KeyboardInterrupt:
            print("\nImport cancelled by user")
            return False


def create_argument_parser():
    """Create and return the argument parser."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Smart Bulk Recipe Importer for Mealie with Selective Re-parsing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bulk_import_smart.py mainland_chinese_recipes.txt
  python bulk_import_smart.py recipes.txt --dry-run
  python bulk_import_smart.py recipes.txt --yes  # Skip confirmation + auto-processing
  python bulk_import_smart.py recipes.txt --yes --job-id my-import-2026
  python bulk_import_smart.py --resume my-import-2026
        """
    )
    
    parser.add_argument('recipe_file', nargs='?', type=str,
                        help='Path to text file containing recipe URLs (one per line)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate import without actually importing')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt and auto-process')
    parser.add_argument('--no-post-process', action='store_true',
                        help='Skip automatic post-processing (parsing, tagging, indexing)')
    parser.add_argument('--job-id', type=str,
                        help='Job ID for state tracking (auto-generated if not provided)')
    parser.add_argument('--resume', type=str,
                        help='Resume a previous job by its ID')
    
    return parser


def main():
    """Main execution function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle resume mode
    if args.resume:
        return handle_resume(args.resume)
    
    # Normal mode requires recipe file
    if not args.recipe_file:
        parser.print_help()
        print("\nError: recipe_file is required unless using --resume")
        sys.exit(1)
    
    recipe_file = args.recipe_file
    dry_run = args.dry_run
    skip_confirmation = args.yes
    skip_post_process = args.no_post_process
    job_id = args.job_id

    # Validate environment
    print("🔍 Validating environment...")
    if not validate_all():
        print("❌ Environment validation failed. Please fix issues above.")
        sys.exit(1)

    # Read recipe URLs
    urls = read_recipe_urls(recipe_file)
    if not urls:
        print("❌ No valid URLs found in file")
        sys.exit(1)

    # Get existing recipe URLs for duplicate checking
    existing_urls = get_existing_recipe_urls()

    # Analyze what to import vs skip
    to_import, duplicates = analyze_import_list(urls, existing_urls)

    # Show analysis and get confirmation
    print_analysis(to_import, duplicates)

    if not confirm_import(to_import, duplicates, dry_run, skip_confirmation):
        sys.exit(0)

    # Generate or use provided job ID
    if job_id is None:
        job_id = str(uuid.uuid4())[:8]  # Short UUID for readability
    
    print(f"\n📋 Job ID: {job_id}")
    print(f"   (Use --resume {job_id} to resume if interrupted)\n")

    # Import recipes using streaming pipeline
    print(f"{'🔍 Running dry run...' if dry_run else '🚀 Starting streaming import pipeline...'}")
    
    if dry_run:
        # Dry run uses old parallel method
        results = import_recipe_batch_parallel(to_import, dry_run=True)
        print_final_report(results, duplicates, dry_run, skip_post_process)
    else:
        # Real import uses batched pipeline (Import → Parse → Tag → Index)
        state = batched_bulk_import(to_import, job_id=job_id, batch_size=100)
        
        # Print job summary
        print_job_summary(state, duplicates)


def handle_resume(job_id: str) -> None:
    """
    Resume a previously interrupted import job.
    
    Args:
        job_id: The job ID to resume
    """
    print(f"🔄 Attempting to resume job: {job_id}")
    
    # Load existing state
    state = PipelineState(job_id)
    if not state.load():
        print(f"❌ Error: No saved state found for job ID '{job_id}'")
        print(f"   State file would be at: {state._state_file}")
        print(f"\nAvailable jobs:")
        
        # List available job files
        from pathlib import Path
        jobs_dir = DATA_DIR / 'jobs'
        if jobs_dir.exists():
            job_files = list(jobs_dir.glob('*_state.json'))
            if job_files:
                for f in job_files[:10]:  # Show up to 10
                    job_name = f.stem.replace('_state', '')
                    print(f"   - {job_name}")
                if len(job_files) > 10:
                    print(f"   ... and {len(job_files) - 10} more")
            else:
                print("   (no jobs found)")
        else:
            print("   (jobs directory not found)")
        
        sys.exit(1)
    
    # Get summary of loaded state
    summary = state.get_summary()
    print(f"✅ Loaded job state:")
    print(f"   Total recipes: {summary['total']}")
    print(f"   Completed: {summary['completed']}")
    print(f"   Failed: {summary['failed']}")
    print(f"   In progress: {summary['in_progress']}")
    
    # Get pending URLs for import phase
    pending_import = state.get_pending_for_phase('import')
    pending_parsing = state.get_pending_for_phase('parsing')
    pending_tagging = state.get_pending_for_phase('tagging')
    pending_indexing = state.get_pending_for_phase('indexing')
    
    total_pending = len(pending_import) + len(pending_parsing) + len(pending_tagging) + len(pending_indexing)
    
    if total_pending == 0:
        print(f"\n✅ Job {job_id} is already complete!")
        print_job_summary(state, [])
        return
    
    print(f"\n📊 Pending work:")
    print(f"   Import: {len(pending_import)} recipes")
    print(f"   Parsing: {len(pending_parsing)} recipes")
    print(f"   Tagging: {len(pending_tagging)} recipes")
    print(f"   Indexing: {len(pending_indexing)} recipes")
    
    # Validate environment
    print("\n🔍 Validating environment...")
    if not validate_all():
        print("❌ Environment validation failed. Please fix issues above.")
        sys.exit(1)
    
    # Resume the pipeline
    print(f"\n🚀 Resuming batched import pipeline...")
    
    # Collect all URLs that need any processing
    # batched_bulk_import will handle loading state and resuming
    all_urls = list(state.results.keys())
    
    # Run batched import (it will load state and skip completed items)
    state = batched_bulk_import(all_urls, job_id=job_id, batch_size=100)
    
    # Print final summary
    print_job_summary(state, [])


def print_job_summary(state: 'PipelineState', duplicates: List[str]) -> None:
    """
    Print comprehensive job summary from PipelineState.
    
    Args:
        state: PipelineState with results
        duplicates: List of duplicate URLs that were skipped
    """
    summary = state.get_summary()
    phases = summary['phases']
    
    print(f"\n{'='*60}")
    print(f"🎉 IMPORT JOB COMPLETE: {state.job_id}")
    print(f"{'='*60}")
    
    print(f"\n📊 Overall Results:")
    print(f"   Total recipes: {summary['total']}")
    print(f"   ✅ Completed: {summary['completed']}")
    print(f"   ❌ Failed: {summary['failed']}")
    if duplicates:
        print(f"   ⏭️  Skipped duplicates: {len(duplicates)}")
    
    print(f"\n📈 Phase Breakdown:")
    for phase_name in ['import', 'parsing', 'tagging', 'indexing']:
        phase = phases[phase_name]
        status_icon = "✅" if phase['failed'] == 0 else "⚠️"
        print(f"   {status_icon} {phase_name.capitalize()}: {phase['completed']} completed, {phase['failed']} failed")
    
    # Show failed recipes if any
    failed_recipes = [
        (url, recipe) for url, recipe in state.results.items()
        if recipe.status == 'failed'
    ]
    
    if failed_recipes:
        print(f"\n❌ Failed Recipes ({len(failed_recipes)}):")
        for url, recipe in failed_recipes[:10]:  # Show first 10
            print(f"   - {url}")
            print(f"     Phase: {recipe.phase}, Error: {recipe.error}")
        if len(failed_recipes) > 10:
            print(f"   ... and {len(failed_recipes) - 10} more")
        
        print(f"\n💡 To retry failed recipes, run:")
        print(f"   python bulk_import_smart.py --resume {state.job_id}")
    
    print(f"\n📁 State saved to: {state._state_file}")


async def enrich_imported_recipes_metadata(successful_imports: List[Tuple[str, str]]) -> None:
    """
    Enrich metadata for newly imported recipes.
    Now uses parallel processing for faster tagging.

    Args:
        successful_imports: List of (url, slug) tuples for successful imports
    """
    from automatic_tagger import AutomaticTagger

    # Get tagging parallelism config
    tagging_config = get_parallelism_config('tagging')
    tagging_workers = tagging_config.get('workers', 15)

    # Shared client for all enrichment workers (requests.Session with HTTPAdapter is thread-safe)
    from mealie_client import MealieClient
    shared_enrich_client = MealieClient(use_direct_db=False)

    tagger = AutomaticTagger(client=shared_enrich_client)
    enriched_count = 0
    failed_recipes = []

    def process_single_recipe(url_slug_tuple: Tuple[str, str]) -> Tuple[str, bool, str, bool]:
        """
        Process a single recipe for metadata enrichment (runs in thread).

        Returns:
            Tuple of (slug, success, message, was_enriched)
        """
        url, slug = url_slug_tuple
        try:
            # Fetch full recipe data
            recipe_data = fetch_recipe_data_by_slug(slug, client=shared_enrich_client)
            if not recipe_data:
                return (slug, False, "Could not fetch recipe data", False)

            # Check what metadata is missing
            missing_fields = identify_missing_metadata(recipe_data)

            if missing_fields:
                recipe_name = recipe_data.get('name', slug)

                # Use automatic tagger for comprehensive analysis
                # Run async code in this thread's own event loop
                analysis = asyncio.run(tagger.analyze_recipe(recipe_data))

                # Apply tags to Mealie (pass recipe_data to avoid redundant lookup)
                recipe_id = recipe_data.get('id')
                if recipe_id:
                    tagger.apply_tags_to_mealie(recipe_id, analysis, recipe_data=recipe_data)

                # Try to infer servings if missing
                if 'servings' in missing_fields:
                    asyncio.run(infer_missing_servings(recipe_data))

                return (slug, True, f"Enriched {recipe_name}: {', '.join(missing_fields)}", True)
            else:
                return (slug, True, "No missing fields", False)

        except Exception as e:
            return (slug, False, str(e), False)

    # Process recipes in parallel using ThreadPoolExecutor
    print(f"   Using {tagging_workers} workers for parallel tagging...")

    with ThreadPoolExecutor(max_workers=tagging_workers) as executor:
        futures = {executor.submit(process_single_recipe, item): item for item in successful_imports}

        for future in as_completed(futures):
            url, slug = futures[future]
            try:
                result_slug, success, message, was_enriched = future.result()
                if success and was_enriched:
                    print(f"   ✅ {message}")
                    enriched_count += 1
                elif not success:
                    print(f"   ❌ CRITICAL FAILURE: Metadata enrichment failed for {result_slug}: {message}")
                    print(f"   ❌ Recipe will be imported without required metadata - this breaks the system")
                    failed_recipes.append((result_slug, message))
            except Exception as e:
                print(f"   ❌ CRITICAL FAILURE: Metadata enrichment failed for {slug}: {e}")
                print(f"   ❌ Recipe will be imported without required metadata - this breaks the system")
                failed_recipes.append((slug, str(e)))

    # Clean up shared client after all workers finish
    shared_enrich_client.close()

    # Raise error if any recipes failed (preserves original fail-fast behavior)
    if failed_recipes:
        failed_slugs = [slug for slug, _ in failed_recipes]
        raise RuntimeError(f"Metadata enrichment failed for {len(failed_recipes)} recipes: {', '.join(failed_slugs)}. Cannot continue with incomplete recipe data.")

    if enriched_count > 0:
        print(f"✅ Enriched metadata for {enriched_count} recipes")


def identify_missing_metadata(recipe_data: Dict) -> List[str]:
    """
    Identify which metadata fields are missing from a recipe.

    Args:
        recipe_data: Full recipe data from Mealie

    Returns:
        List of missing metadata field names
    """
    missing = []

    # Check servings
    if not recipe_data.get('recipeYield') or recipe_data.get('recipeYield') == 0:
        missing.append('servings')

    # Check cuisine tags
    tags = recipe_data.get('tags', [])
    cuisine_tags = [tag for tag in tags if isinstance(tag, dict) and
                   tag.get('name', '').lower() in get_cuisine_keywords()]
    if len(cuisine_tags) == 0:
        missing.append('cuisine_tags')

    # Check timing information
    has_timing = any([
        recipe_data.get('prepTime'),
        recipe_data.get('cookTime'),
        recipe_data.get('totalTime')
    ])
    if not has_timing:
        missing.append('timing')

    # Check prep requirements
    if not any('Prep' in tag.get('name', '') or 'Overnight' in tag.get('name', '')
               for tag in tags if isinstance(tag, dict)):
        missing.append('prep_requirements')

    return missing


async def infer_missing_servings(recipe_data: Dict) -> None:
    """
    Try to infer serving size from recipe content using LLM.

    Args:
        recipe_data: Recipe data that might be missing servings
    """
    try:
        from batch_llm_processor import get_llm_cache

        recipe_name = recipe_data.get('name', 'Unknown Recipe')
        description = recipe_data.get('description', '')[:200]
        ingredients = recipe_data.get('recipeIngredient', [])[:5]

        prompt = SERVINGS_ESTIMATION_PROMPT.format(
            recipe_name=recipe_name,
            description=description,
            ingredients=', '.join(str(i) for i in ingredients)
        )

        cache = await get_llm_cache()
        response = await cache.call_llm(
            prompt=prompt,
            system_prompt=SERVINGS_ESTIMATION_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=10
        )

        try:
            estimated_servings = int(response.strip())
            if 1 <= estimated_servings <= 12:
                # Update recipe in Mealie
                update_recipe_servings(recipe_data.get('id'), estimated_servings)
                print(f"   📏 Inferred {estimated_servings} servings for {recipe_name}")
        except ValueError:
            print(f"❌ CRITICAL: Servings inference returned invalid response")
            print(f"❌ FAST FAILURE: Recipe scaling requires valid serving information")
            raise RuntimeError("Servings inference failed: Invalid response from LLM")

    except Exception as e:
        print(f"❌ CRITICAL FAILURE: Servings inference failed: {e}")
        print(f"❌ Recipe scaling will be broken without serving information")
        raise RuntimeError(f"Servings inference failed: {e}. Cannot continue with unscalable recipes.") from e


def update_recipe_servings(recipe_id: str, servings: int) -> None:
    """
    Update recipe servings in Mealie.

    Args:
        recipe_id: Mealie recipe ID (can be slug or UUID)
        servings: Number of servings
    """
    try:
        from mealie_client import MealieClient
        client = MealieClient()
        try:
            client.update_recipe(recipe_id, {"recipeYield": servings})
        finally:
            client.close()

    except Exception as e:
        print(f"❌ CRITICAL FAILURE: Failed to update recipe servings: {e}")
        raise RuntimeError(f"Recipe servings update failed: {e}. Cannot continue with broken scaling.") from e


def get_cuisine_keywords() -> Set[str]:
    """Get comprehensive cuisine keywords for validation."""
    return {
        # Asian cuisines
        'chinese', 'cantonese', 'sichuan', 'shandong', 'japanese', 'korean',
        'thai', 'vietnamese', 'indian', 'malaysian', 'indonesian', 'filipino',
        'singaporean', 'taiwanese', 'hong kong', 'macanese',

        # European cuisines
        'italian', 'french', 'spanish', 'german', 'british', 'irish',
        'greek', 'turkish', 'russian', 'polish', 'hungarian', 'portuguese',

        # American cuisines
        'american', 'mexican', 'tex-mex', 'cajun', 'creole', 'southern',
        'southwestern', 'californian', 'new england',

        # Middle Eastern
        'middle eastern', 'lebanese', 'iranian', 'israeli', 'moroccan',
        'egyptian', 'turkish', 'persian',

        # Other
        'mediterranean', 'fusion', 'international'
    }


if __name__ == "__main__":
    main()
