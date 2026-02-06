#!/usr/bin/env python3
"""
Recipe Image Fetching via Brave Search API
============================================

Auto-fetch images for AI-generated recipes using Brave Search API.
Supports two entry points:

1. After meal planning: fetch images for newly created recipes
2. Backfill utility: add images to existing recipes without them

Features:
- Brave Search API image search
- Dimension filtering for quality images
- Optional LLM vision validation
- Graceful degradation when API not configured

Usage:
    from recipe_images import fetch_and_apply_image, backfill_missing_images

    # Single recipe
    result = await fetch_and_apply_image("chicken-tikka-masala", "Chicken Tikka Masala")

    # Batch backfill (auto-detects count, caps at 100)
    batch_result = await backfill_missing_images()
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import requests

from config import (
    BRAVE_API_KEY,
    BRAVE_API_URL,
    get_image_search_config,
)
from tools.logging_utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Load configuration from centralized config.py (avoids duplication)
_config = get_image_search_config()


# =============================================================================
# EXCEPTIONS
# =============================================================================

class RateLimitError(Exception):
    """Raised when Brave API rate limit is hit."""
    pass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ImageResult:
    """Result of attempting to fetch and apply an image to a single recipe."""
    status: Literal['success', 'skipped', 'rate_limited', 'error']
    recipe_slug: str
    recipe_name: str
    message: Optional[str] = None
    image_url: Optional[str] = None


@dataclass
class BatchResult:
    """Result of batch image backfill operation."""
    success_count: int
    skipped_count: int
    rate_limited: bool
    results: List[ImageResult] = field(default_factory=list)


# =============================================================================
# BRAVE SEARCH API
# =============================================================================

def search_brave_images(query: str) -> List[Dict]:
    """
    Search for images using Brave Search API.
    
    Args:
        query: Search query string (e.g., "Chicken Tikka Masala recipe")
        
    Returns:
        List of image results with url, width, height
        Empty list on error or if API not configured
        
    Raises:
        None - returns empty list on all errors for graceful degradation
    """
    if not BRAVE_API_KEY:
        logger.debug("Brave API not configured, skipping image search")
        return []
    
    try:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY,
        }
        
        params = {
            "q": query,
            "count": 10,  # Fetch more than max_attempts to have fallbacks
        }
        
        response = requests.get(
            BRAVE_API_URL,
            headers=headers,
            params=params,
            timeout=15
        )
        
        # Handle specific error codes
        if response.status_code == 429:
            logger.warning("Brave API rate limit reached (429)")
            raise RateLimitError("Brave API rate limit reached")
        
        if response.status_code == 401:
            logger.error("Brave API authentication failed (401) - check BRAVE_API_KEY")
            return []
        
        if response.status_code in (500, 503):
            logger.warning(f"Brave API server error ({response.status_code})")
            return []
        
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        # Extract relevant fields
        images = []
        for result in results:
            image_info = {
                "url": result.get("properties", {}).get("url") or result.get("url"),
                "width": result.get("properties", {}).get("width", 0),
                "height": result.get("properties", {}).get("height", 0),
                "title": result.get("title", ""),
            }
            if image_info["url"]:
                images.append(image_info)
        
        logger.debug(f"Brave search returned {len(images)} images for query: {query}")
        return images
        
    except RateLimitError:
        raise  # Re-raise rate limit errors for caller to handle
    except requests.exceptions.Timeout:
        logger.warning(f"Brave API timeout for query: {query}")
        return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"Brave API request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in Brave search: {e}")
        return []


# =============================================================================
# IMAGE VALIDATION
# =============================================================================

def validate_image_dimensions(image: Dict, min_width: int, min_height: int) -> bool:
    """
    Check if image meets minimum dimension requirements.
    
    Args:
        image: Image dict with 'width' and 'height' keys
        min_width: Minimum width in pixels
        min_height: Minimum height in pixels
        
    Returns:
        True if image meets requirements, False otherwise
    """
    width = image.get("width", 0)
    height = image.get("height", 0)
    
    # If dimensions unknown, accept the image (let Mealie validate)
    if width == 0 or height == 0:
        logger.debug(f"Image dimensions unknown, accepting: {image.get('url', 'unknown')[:50]}")
        return True
    
    meets_requirements = width >= min_width and height >= min_height
    
    if not meets_requirements:
        logger.debug(f"Image too small ({width}x{height}), need {min_width}x{min_height}")
    
    return meets_requirements


def validate_image_with_vision(image_url: str) -> bool:
    """
    Validate that an image is food-related using LLM vision.
    
    This is an optional validation step that uses vision models to verify
    the image actually shows food rather than unrelated content.
    
    Args:
        image_url: URL of the image to validate
        
    Returns:
        True if image is validated as food (or validation fails gracefully)
        
    Note:
        On any error, returns True to allow using the image anyway.
        This ensures graceful degradation - vision validation is a nice-to-have.
    """
    if not _config.get('use_vision_validation', True):
        return True
    
    logger.info(f"🔍 Validating image with vision model...")
    
    try:
        # Import vision validation dependencies
        from config import CHAT_API_KEY, CHAT_API_URL, CHAT_MODEL
        
        if not CHAT_API_KEY:
            logger.debug("LLM not configured, skipping vision validation")
            return True
        
        # Call vision model to validate image
        headers = {
            "Authorization": f"Bearer {CHAT_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Is this image showing food or a recipe dish? Reply with just 'yes' or 'no'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        response = requests.post(
            f"{CHAT_API_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").lower().strip()
        
        is_food = answer.startswith("yes")
        
        if is_food:
            logger.info(f"✅ Vision: confirmed food image")
        else:
            logger.info(f"❌ Vision: rejected (not food): {image_url[:50]}...")
        
        return is_food
        
    except Exception as e:
        # Graceful degradation - if vision fails, accept the image
        logger.warning(f"Vision validation failed, accepting image: {e}")
        return True


# =============================================================================
# MEALIE API INTEGRATION
# =============================================================================



def apply_image_to_recipe(recipe_slug: str, image_url: str) -> bool:
    """
    Apply an image URL to a recipe in Mealie.
    
    Mealie will download and store the image automatically.
    
    Args:
        recipe_slug: Recipe slug in Mealie
        image_url: URL of the image to apply
        
    Returns:
        True on success, False on failure
    """
    from mealie_client import MealieClient
    
    try:
        client = MealieClient()
        try:
            success = client.upload_recipe_image(recipe_slug, image_url)
            if success:
                logger.info(f"Successfully applied image to recipe: {recipe_slug}")
            else:
                logger.warning(f"Failed to apply image to {recipe_slug}")
            return success
        finally:
            client.close()
            
    except Exception as e:
        logger.error(f"Unexpected error applying image to {recipe_slug}: {e}")
        return False


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

async def fetch_and_apply_image(recipe_slug: str, recipe_name: str) -> ImageResult:
    """
    Search for and apply an image to a single recipe.
    
    Steps:
    1. Search Brave for "{recipe_name} {query_suffix}"
    2. Filter results by minimum dimensions
    3. (Optional) Validate with vision model
    4. POST image URL to Mealie
    5. Return result
    
    Args:
        recipe_slug: Recipe slug in Mealie
        recipe_name: Recipe name for search query
        
    Returns:
        ImageResult with status and details
    """
    # Early return if Brave API not configured
    if not BRAVE_API_KEY:
        return ImageResult(
            status='skipped',
            recipe_slug=recipe_slug,
            recipe_name=recipe_name,
            message='Brave API not configured'
        )
    
    # Build search query
    query_suffix = _config.get('query_suffix', 'recipe')
    search_query = f"{recipe_name} {query_suffix}"
    
    try:
        # Search for images
        images = search_brave_images(search_query)
        
        if not images:
            logger.info(f"No images found for: {recipe_name}")
            return ImageResult(
                status='skipped',
                recipe_slug=recipe_slug,
                recipe_name=recipe_name,
                message='No images found'
            )
        
        # Configuration
        min_width = _config.get('min_width', 400)
        min_height = _config.get('min_height', 300)
        max_attempts = _config.get('max_attempts', 3)
        
        # Try images until one works
        attempts = 0
        for image in images:
            if attempts >= max_attempts:
                break
            
            image_url = image.get('url')
            if not image_url:
                continue
            
            attempts += 1
            
            # Check dimensions
            if not validate_image_dimensions(image, min_width, min_height):
                continue
            
            # Optional vision validation
            if not validate_image_with_vision(image_url):
                continue
            
            # Try to apply to Mealie
            if apply_image_to_recipe(recipe_slug, image_url):
                return ImageResult(
                    status='success',
                    recipe_slug=recipe_slug,
                    recipe_name=recipe_name,
                    image_url=image_url,
                    message=f'Applied image from {image_url[:50]}...'
                )
        
        # No suitable image found after all attempts
        return ImageResult(
            status='skipped',
            recipe_slug=recipe_slug,
            recipe_name=recipe_name,
            message=f'No suitable image found after {attempts} attempts'
        )
        
    except RateLimitError:
        return ImageResult(
            status='rate_limited',
            recipe_slug=recipe_slug,
            recipe_name=recipe_name,
            message='Brave API rate limit reached'
        )
    except Exception as e:
        logger.error(f"Error fetching image for {recipe_name}: {e}")
        return ImageResult(
            status='error',
            recipe_slug=recipe_slug,
            recipe_name=recipe_name,
            message=str(e)
        )


async def find_recipes_without_images(limit: int = 100) -> List[Dict]:
    """
    Query Mealie for recipes that have no image.
    
    Checks if the `image` field is empty/null. This is the fast path that works
    for normal recipes (new imports, AI-generated recipes without images).
    
    Note: Recipes corrupted by migration may have the `image` field populated
    but no actual image file. Those require a separate repair operation.
    
    Args:
        limit: Maximum number of recipes to return
        
    Returns:
        List of {slug, name} dicts for recipes without images
    """
    from mealie_client import MealieClient
    
    recipes_without_images = []
    
    try:
        client = MealieClient()
        try:
            recipes = client.get_all_recipes()
            
            for recipe in recipes:
                slug = recipe.get("slug")
                name = recipe.get("name")
                image_field = recipe.get("image")
                
                if not slug:
                    continue
                
                # Empty image field = no image
                # (Recipes with images have a 4-char token in this field)
                if not image_field:
                    recipes_without_images.append({
                        "slug": slug,
                        "name": name,
                    })
                    
                    if len(recipes_without_images) >= limit:
                        break
        finally:
            client.close()
        
        logger.info(f"Found {len(recipes_without_images)} recipes without images")
        return recipes_without_images
        
    except Exception as e:
        logger.error(f"Error finding recipes without images: {e}")
        return []


async def _process_recipes_batch(
    recipes: List[Dict],
    delay: float = 0.5,
    log_progress: bool = False,
    add_remaining_on_rate_limit: bool = False,
) -> BatchResult:
    """
    Shared batch processing logic for image fetching.
    
    Args:
        recipes: List of dicts with 'slug' and 'name' keys
        delay: Seconds to wait between API calls (be nice to APIs)
        log_progress: If True, log each recipe result
        add_remaining_on_rate_limit: If True, add remaining recipes as skipped when rate limited
        
    Returns:
        BatchResult with summary and individual results
    """
    results = []
    success_count = 0
    skipped_count = 0
    rate_limited = False
    
    for idx, recipe in enumerate(recipes):
        slug = recipe.get("slug")
        name = recipe.get("name")
        
        if not slug or not name:
            continue
        
        result = await fetch_and_apply_image(slug, name)
        results.append(result)
        
        if result.status == 'success':
            success_count += 1
            if log_progress:
                logger.info(f"✓ {name}")
        elif result.status == 'rate_limited':
            rate_limited = True
            if log_progress:
                logger.warning("Rate limited, stopping batch")
            
            # Optionally add remaining recipes as skipped
            if add_remaining_on_rate_limit:
                for r in recipes[idx + 1:]:
                    results.append(ImageResult(
                        status='skipped',
                        recipe_slug=r.get('slug', ''),
                        recipe_name=r.get('name', ''),
                        message='Skipped due to rate limiting'
                    ))
                    skipped_count += 1
            break
        else:
            skipped_count += 1
            if log_progress:
                logger.info(f"⊘ {name}: {result.message}")
        
        # Small delay between requests to be nice to APIs
        await asyncio.sleep(delay)
    
    return BatchResult(
        success_count=success_count,
        skipped_count=skipped_count,
        rate_limited=rate_limited,
        results=results
    )


async def backfill_missing_images() -> BatchResult:
    """
    Find recipes without images and fetch images for them.
    
    Auto-detects how many recipes need images and caps at 100 per run
    to respect API rate limits.
    
    Returns:
        BatchResult with summary and individual results
    """
    # Early return if Brave API not configured
    if not BRAVE_API_KEY:
        print("❌ Brave API not configured, skipping image backfill")
        return BatchResult(
            success_count=0,
            skipped_count=0,
            rate_limited=False,
            results=[]
        )
    
    # Find ALL recipes without images (up to 1000 for counting)
    recipes = await find_recipes_without_images(limit=1000)
    
    if not recipes:
        print("✅ All recipes have images - nothing to do!")
        return BatchResult(
            success_count=0,
            skipped_count=0,
            rate_limited=False,
            results=[]
        )
    
    total_missing = len(recipes)
    print(f"Found {total_missing} recipes without images")
    
    # Cap at 100 to respect API rate limits
    if total_missing > 100:
        print(f"⚠️  Processing first 100 to respect API rate limits. Run again to process more.")
        recipes = recipes[:100]
    
    print(f"\nProcessing {len(recipes)} recipes...")
    print("-" * 40)
    
    result = await _process_recipes_batch(
        recipes,
        delay=0.5,
        log_progress=True,
        add_remaining_on_rate_limit=False
    )
    
    logger.info(f"Backfill complete: {result.success_count} success, {result.skipped_count} skipped, rate_limited={result.rate_limited}")
    
    # Re-index successfully updated recipes to update local search index
    # This ensures the local index reflects the updated image metadata
    if result.success_count > 0:
        print(f"\n🔄 Re-indexing {result.success_count} recipes with new images...", flush=True)
        try:
            from recipe_rag import RecipeRAG
            rag = RecipeRAG()
            
            # Get slugs that were successfully updated
            successful_slugs = [r.recipe_slug for r in result.results if r.status == 'success']
            
            if successful_slugs:
                # Fetch and index in batches
                recipes_data = []
                from mealie_client import MealieClient
                client = MealieClient()
                try:
                    for slug in successful_slugs:
                        try:
                            recipe = client.get_recipe(slug)
                            recipes_data.append(recipe)
                        except Exception as e:
                            logger.warning(f"Failed to fetch {slug} for re-indexing: {e}")
                finally:
                    client.close()
                
                if recipes_data:
                    indexed = rag.index_recipes_batch(recipes_data, force=True)
                    print(f"✅ Re-indexed {indexed}/{len(recipes_data)} recipes", flush=True)
        except Exception as e:
            logger.warning(f"⚠️  Re-indexing failed (images added but local index not updated): {e}")
            print(f"⚠️  Re-indexing failed: {e}", flush=True)
    
    return result


# =============================================================================
# BATCH PROCESSING FOR MEAL PLANNING
# =============================================================================

async def fetch_images_for_recipes(recipes: List[Dict]) -> BatchResult:
    """
    Fetch images for a list of newly created recipes.
    
    Intended for use after meal planning to add images to AI-generated recipes.
    
    Args:
        recipes: List of dicts with 'slug' and 'name' keys
        
    Returns:
        BatchResult with summary and individual results
    """
    if not BRAVE_API_KEY:
        logger.info("Brave API not configured, skipping image fetching")
        return BatchResult(
            success_count=0,
            skipped_count=len(recipes),
            rate_limited=False,
            results=[
                ImageResult(
                    status='skipped',
                    recipe_slug=r.get('slug', ''),
                    recipe_name=r.get('name', ''),
                    message='Brave API not configured'
                )
                for r in recipes
            ]
        )
    
    return await _process_recipes_batch(
        recipes,
        delay=0.3,
        log_progress=False,
        add_remaining_on_rate_limit=True
    )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch images for recipes using Brave Search API")
    parser.add_argument("--backfill", action="store_true",
                       help="Backfill images for recipes without images (auto-detects count, caps at 100)")
    parser.add_argument("--recipe", metavar="SLUG",
                       help="Fetch image for a specific recipe by slug")
    parser.add_argument("--name", metavar="NAME",
                       help="Recipe name (required with --recipe)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Search for images but don't apply them")
    args = parser.parse_args()
    
    # Check API key
    if not BRAVE_API_KEY:
        print("❌ BRAVE_API_KEY not configured")
        print("   Set via environment variable or data/secrets.yaml")
        exit(1)
    
    async def main():
        if args.recipe:
            if not args.name:
                print("❌ --name required when using --recipe")
                exit(1)
            
            print(f"🔍 Searching for image: {args.name}")
            result = await fetch_and_apply_image(args.recipe, args.name)
            
            if result.status == 'success':
                print(f"✅ Applied image: {result.image_url}")
            else:
                print(f"⚠️  {result.status}: {result.message}")
        elif args.backfill:
            print("🔍 Scanning for recipes without images...")
            result = await backfill_missing_images()
            
            print(f"\n{'='*60}")
            print(f"✅ Success: {result.success_count}")
            print(f"⊘ Skipped: {result.skipped_count}")
            if result.rate_limited:
                print(f"⚠️  Rate limited - some recipes skipped")
            print(f"{'='*60}")
        else:
            parser.print_help()
    
    asyncio.run(main())
