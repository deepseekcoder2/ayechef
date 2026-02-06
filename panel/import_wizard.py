"""
Import Wizard - Business logic for progressive confidence import flow.
"""
from typing import List, Dict, Optional, TypedDict
from urllib.parse import urlparse
import random
import concurrent.futures

from recipe_urls import get_scraper_for_url, is_site_supported, reload_scrapers
from recipe_urls._utils import get_site_origin
from import_site import fetch_sitemap_urls, filter_recipe_urls, get_existing_recipe_urls
from utils.url_utils import normalize_url
from mealie_client import MealieClient


class AnalysisResult(TypedDict):
    """Result of site analysis."""
    supported: bool
    host: str
    total_recipes: int
    already_imported: int
    new_recipes: int
    sample_urls: List[str]
    sample_names: List[str]
    has_categories: bool
    categories: List[str]
    category_counts: Dict[str, int]  # Recipe count per category
    # Compatibility check - did a single test import work?
    compatible: bool
    compatibility_error: Optional[str]
    test_recipe_name: Optional[str]  # Name of recipe that was tested


def test_single_import(url: str) -> tuple[bool, str, Optional[str]]:
    """
    Test if Mealie can import a single recipe from this URL.
    
    Returns:
        (success, error_message, recipe_name)
        
    If successful, deletes the test recipe to avoid clutter.
    """
    # Force API mode - we create then immediately read/delete
    client = MealieClient(use_direct_db=False)
    try:
        print(f"  Testing Mealie compatibility with: {url}")
        
        # Try to import via MealieClient
        try:
            recipe_data = client.create_recipe_from_url(url, include_tags=False)
            slug = recipe_data.get('slug') if isinstance(recipe_data, dict) else recipe_data
            
            # Get recipe name before we delete it
            try:
                recipe = client.get_recipe(slug)
                recipe_name = recipe.get('name', slug)
                recipe_id = recipe.get('id')
            except Exception:
                recipe_name = slug
                recipe_id = None
            
            # Delete the test recipe to avoid clutter
            if recipe_id:
                print(f"  ✓ Import worked! Cleaning up test recipe...")
                client.delete_recipe(slug)
            
            return True, None, recipe_name
        
        except Exception as e:
            error_str = str(e)
            
            # Translate common errors to user-friendly messages
            if 'BAD_RECIPE_DATA' in error_str:
                return False, "This site doesn't use standard recipe format. Mealie can't extract recipe data from it.", None
            elif 'SCRAPE_ERROR' in error_str or 'timeout' in error_str.lower():
                return False, "Mealie couldn't access this site. It may have anti-bot protection.", None
            else:
                return False, f"Mealie import failed: {error_str[:100]}", None
                
    except Exception as e:
        error_str = str(e)
        if 'timeout' in error_str.lower() or 'timed out' in error_str.lower():
            return False, "Request timed out. The site may be slow or blocking requests.", None
        return False, f"Connection error: {error_str}", None
    finally:
        client.close()


def analyze_site(url: str) -> AnalysisResult:
    """
    Analyze a site for import.
    
    This is SYNCHRONOUS - caller should show loading state.
    
    Returns structured analysis with recipe counts and compatibility status.
    """
    reload_scrapers()
    
    parsed = urlparse(url)
    host = parsed.netloc.replace('www.', '')
    
    # Check if supported by our scraper
    if not is_site_supported(url):
        return {
            'supported': False,
            'host': host,
            'total_recipes': 0,
            'already_imported': 0,
            'new_recipes': 0,
            'sample_urls': [],
            'sample_names': [],
            'has_categories': False,
            'categories': [],
            'category_counts': {},
            'compatible': False,
            'compatibility_error': 'Site not learned yet. Use "Learn Site" first.',
            'test_recipe_name': None
        }
    
    # Get scraper info
    scraper_class = get_scraper_for_url(url)
    has_categories = scraper_class.has_categories() if scraper_class else False
    categories = list(scraper_class.get_categories()) if has_categories else []
    
    # Fetch category counts sequentially with polite delays
    # Being a good citizen - don't hammer websites
    category_counts = {}
    if has_categories and scraper_class:
        import time
        print(f"Fetching recipe counts for {len(categories)} categories (this takes a moment)...")
        
        for i, category in enumerate(categories):
            try:
                recipe_urls = scraper_class.scrape_category_urls(category, url)
                category_counts[category] = len(recipe_urls)
                print(f"  [{i+1}/{len(categories)}] {category}: {len(recipe_urls)} recipes")
            except Exception as e:
                category_counts[category] = 0
                print(f"  [{i+1}/{len(categories)}] {category}: failed ({e})")
            
            # Polite delay between requests - don't get shadowbanned
            if i < len(categories) - 1:
                time.sleep(0.5)
    
    # Fetch all recipe URLs via sitemap
    print(f"Fetching sitemap for {host}...")
    all_urls = fetch_sitemap_urls(url)
    
    # Filter to recipe URLs only
    recipe_urls = filter_recipe_urls(all_urls, url)
    
    # Deduplicate URLs (sitemaps can have duplicates)
    recipe_urls = list(dict.fromkeys(recipe_urls))
    total_recipes = len(recipe_urls)
    
    # Get existing recipes from Mealie (already normalized)
    existing_urls = get_existing_recipe_urls()
    
    # Calculate new vs existing (normalize URLs for comparison, deduplicate)
    seen = set()
    new_urls = []
    for u in recipe_urls:
        normalized = normalize_url(u)
        if normalized not in existing_urls and normalized not in seen:
            seen.add(normalized)
            new_urls.append(u)
    already_imported = total_recipes - len(new_urls)
    
    # Get samples (random 5 from new, or all if < 5)
    sample_urls = random.sample(new_urls, min(5, len(new_urls))) if new_urls else []
    
    def extract_recipe_name(url: str) -> str:
        """Extract recipe name from URL, handling trailing slashes."""
        # Remove trailing slash and get last path segment
        path = url.rstrip('/').split('/')[-1]
        if not path:
            return 'Unknown Recipe'
        # Clean up the name
        name = path.replace('-', ' ').replace('_', ' ').title()
        return name[:40] if name else 'Unknown Recipe'
    
    sample_names = [extract_recipe_name(u) for u in sample_urls]
    
    # === COMPATIBILITY CHECK ===
    # Test ONE import to verify Mealie can actually scrape this site
    compatible = True
    compatibility_error = None
    test_recipe_name = None
    
    if new_urls:
        # Pick a random URL to test
        test_url = random.choice(new_urls)
        print(f"Testing Mealie compatibility...")
        compatible, compatibility_error, test_recipe_name = test_single_import(test_url)
        
        if not compatible:
            print(f"  ✗ Site incompatible: {compatibility_error}")
    
    return {
        'supported': True,
        'host': host,
        'total_recipes': total_recipes,
        'already_imported': already_imported,
        'new_recipes': len(new_urls),
        'sample_urls': sample_urls,
        'sample_names': sample_names,
        'has_categories': has_categories,
        'categories': categories,
        'category_counts': category_counts,
        'compatible': compatible,
        'compatibility_error': compatibility_error,
        'test_recipe_name': test_recipe_name
    }
