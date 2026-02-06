"""
Recipe URL scrapers with auto-discovery.

To add a new site:
1. Create a .py file in recipe_urls/sites/
2. Define a class inheriting from AbstractScraper
3. Set RECIPE_PATTERN, UNWANTED_PATTERNS, and host()
4. Optionally set CATEGORY_PAGES or CATEGORIES for category-based import

That's it. No other files need to be modified.
"""

import pkgutil
import importlib
from typing import Optional, List, Dict

from recipe_urls._abstract import AbstractScraper


def _discover_scrapers() -> dict:
    """Auto-discover all scraper classes from sites/ and data/scrapers/ directories."""
    import importlib.util
    from pathlib import Path
    import sys
    import logging
    
    scrapers = {}
    
    # 1. Load built-in scrapers from recipe_urls/sites/
    from recipe_urls import sites
    
    for _, module_name, _ in pkgutil.iter_modules(sites.__path__):
        if module_name.startswith('_'):
            continue
        
        module = importlib.import_module(f'recipe_urls.sites.{module_name}')
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) 
                and issubclass(attr, AbstractScraper) 
                and attr is not AbstractScraper
                and hasattr(attr, 'host')):
                try:
                    host = attr.host()
                    scrapers[host] = attr
                except NotImplementedError:
                    pass
    
    # 2. Load user-generated scrapers from data/scrapers/
    from config import DATA_DIR
    user_scrapers_dir = DATA_DIR / "scrapers"
    if user_scrapers_dir.exists():
        for scraper_file in user_scrapers_dir.glob("*.py"):
            if scraper_file.name.startswith('_'):
                continue
            
            file_module_name = scraper_file.stem
            try:
                spec = importlib.util.spec_from_file_location(
                    f"user_scrapers.{file_module_name}", 
                    scraper_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) 
                            and issubclass(attr, AbstractScraper) 
                            and attr is not AbstractScraper
                            and hasattr(attr, 'host')):
                            try:
                                host = attr.host()
                                scrapers[host] = attr
                            except NotImplementedError:
                                pass
            except Exception as e:
                logging.warning(f"Failed to load user scraper {scraper_file}: {e}")
    
    return scrapers


# Auto-discover on import
SCRAPERS = _discover_scrapers()
SITE_ORIGINS = list(SCRAPERS.keys())


def reload_scrapers() -> None:
    """
    Reload scrapers from disk.
    
    Call this after add_site.py creates a new scraper to pick up
    newly learned sites without restarting the worker.
    """
    global SCRAPERS, SITE_ORIGINS
    SCRAPERS = _discover_scrapers()
    SITE_ORIGINS = list(SCRAPERS.keys())


def scrape_urls(base_url: str) -> Optional[List[str]]:
    """Scrape recipe URLs from a supported site."""
    from recipe_urls._utils import get_site_origin
    
    origin = get_site_origin(base_url)
    scraper_class = SCRAPERS.get(origin)
    
    if not scraper_class:
        raise ValueError(f"Unsupported website: {base_url}. Supported: {', '.join(SITE_ORIGINS)}")
    
    return scraper_class(base_url).scrape()


def get_scraper(host: str) -> Optional[type]:
    """Get scraper class for a given host."""
    return SCRAPERS.get(host)


def is_site_supported(url: str) -> bool:
    """Check if a URL's host is supported."""
    from recipe_urls._utils import parse_hostname
    hostname = parse_hostname(url)
    
    # Check exact match
    if hostname in SCRAPERS:
        return True
    
    # Check with/without www
    if hostname.startswith('www.'):
        return hostname[4:] in SCRAPERS
    else:
        return f"www.{hostname}" in SCRAPERS


def get_supported_sites() -> List[str]:
    """Get list of supported site hostnames."""
    return sorted(SITE_ORIGINS)


def detect_url_type(url: str) -> dict:
    """
    Detect if a URL is a single recipe or a site homepage.
    
    Returns:
        dict with keys:
        - type: 'recipe', 'site', or 'unknown'
        - supported: bool - whether we have a scraper for this site
        - host: str - the site hostname
        - has_categories: bool - whether the site has category configuration
        - categories: list - category names if available
    """
    from recipe_urls._utils import parse_hostname, get_site_origin
    
    hostname = parse_hostname(url)
    origin = get_site_origin(url)
    scraper_class = SCRAPERS.get(origin)
    
    result = {
        'type': 'unknown',
        'supported': scraper_class is not None,
        'host': hostname,
        'has_categories': False,
        'categories': [],
    }
    
    if not scraper_class:
        # Not a supported site - can't determine type
        return result
    
    # Add category information
    result['has_categories'] = scraper_class.has_categories()
    if result['has_categories']:
        result['categories'] = scraper_class.get_categories()
    
    # Check if URL matches recipe pattern
    recipe_pattern = scraper_class.RECIPE_PATTERN
    unwanted_patterns = scraper_class.UNWANTED_PATTERNS
    
    if recipe_pattern and recipe_pattern.search(url):
        # Check it's not an unwanted URL (category, about, etc.)
        if not any(p.search(url) for p in unwanted_patterns):
            result['type'] = 'recipe'
            return result
    
    # If it's just the homepage or a non-recipe page, treat as site
    result['type'] = 'site'
    return result


def get_site_categories(url: str) -> Dict[str, any]:
    """
    Get category information for a supported site.
    
    Args:
        url: Site URL
        
    Returns:
        dict with keys:
        - supported: bool
        - has_categories: bool
        - categories: list of category names
        - uses_category_pages: bool - True if site needs category page scraping
    """
    from recipe_urls._utils import get_site_origin
    
    origin = get_site_origin(url)
    scraper_class = SCRAPERS.get(origin)
    
    if not scraper_class:
        return {
            'supported': False,
            'has_categories': False,
            'categories': [],
            'uses_category_pages': False,
        }
    
    return {
        'supported': True,
        'has_categories': scraper_class.has_categories(),
        'categories': scraper_class.get_categories(),
        'uses_category_pages': scraper_class.uses_category_pages(),
    }


def get_category_urls(url: str, category: str) -> List[str]:
    """
    Get recipe URLs for a specific category.
    
    For sites using CATEGORY_PAGES, this scrapes the category page.
    For sites using CATEGORIES (URL patterns), this returns an empty list
    (filtering should be done on the full URL list instead).
    
    Args:
        url: Site base URL
        category: Category name
        
    Returns:
        List of recipe URLs in the category
    """
    from recipe_urls._utils import get_site_origin
    
    origin = get_site_origin(url)
    scraper_class = SCRAPERS.get(origin)
    
    if not scraper_class or not scraper_class.has_categories():
        return []
    
    if scraper_class.uses_category_pages():
        return scraper_class.scrape_category_urls(category, url)
    
    # For URL pattern-based categories, return empty
    # Caller should use filter_urls_by_categories instead
    return []


def get_scraper_for_url(url: str) -> Optional[type]:
    """
    Get the scraper class for a URL.
    
    Args:
        url: Any URL from the site
        
    Returns:
        Scraper class or None if not supported
    """
    from recipe_urls._utils import get_site_origin
    
    origin = get_site_origin(url)
    return SCRAPERS.get(origin)


__all__ = [
    'SCRAPERS', 
    'SITE_ORIGINS', 
    'reload_scrapers',
    'scrape_urls', 
    'get_scraper', 
    'get_scraper_for_url',
    'is_site_supported',
    'get_supported_sites',
    'detect_url_type',
    'get_site_categories',
    'get_category_urls',
    'AbstractScraper'
]
