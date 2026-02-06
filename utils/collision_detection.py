"""
Pre-import collision detection for recipe names.

Prevents Mealie's slug corruption bug by detecting name collisions BEFORE import
and returning site-qualified names when needed.

Usage:
    from utils.collision_detection import check_name_collision, get_qualified_name
    
    # Check if name would collide
    existing = check_name_collision("Banana Bread")
    if existing:
        # Name collides, need to qualify it
        new_name = get_qualified_name("Banana Bread", "https://thewoksoflife.com/...")
        # new_name = "Banana Bread (The Woks of Life)"
"""

import sqlite3
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse

from config import DATA_DIR

logger = logging.getLogger(__name__)

# Database path for recipe index
RECIPE_INDEX_DB = str(DATA_DIR / "recipe_index.db")

# Map of known domains to display names (shared with bulk_import_smart.py)
SITE_DISPLAY_NAMES = {
    'thewoksoflife.com': 'The Woks of Life',
    'bbcgoodfood.com': 'BBC Good Food',
    'seriouseats.com': 'Serious Eats',
    'budgetbytes.com': 'Budget Bytes',
    'bbc.co.uk': 'BBC',
    'abuelascounter.com': "Abuela's Counter",
    'forktospoon.com': 'Fork to Spoon',
    'mykoreankitchen.com': 'My Korean Kitchen',
    'justonecookbook.com': 'Just One Cookbook',
    'recipetineats.com': 'RecipeTin Eats',
    'minimalistbaker.com': 'Minimalist Baker',
    'cookieandkate.com': 'Cookie and Kate',
    'smittenkitchen.com': 'Smitten Kitchen',
    'bonappetit.com': 'Bon Appétit',
    'food52.com': 'Food52',
    'epicurious.com': 'Epicurious',
    'allrecipes.com': 'Allrecipes',
    'foodnetwork.com': 'Food Network',
    'delish.com': 'Delish',
    'tasty.co': 'Tasty',
}


def get_site_display_name(url: str) -> str:
    """
    Extract a display-friendly site name from a URL.
    
    Args:
        url: Full URL (e.g., "https://thewoksoflife.com/banana-bread/")
    
    Returns:
        Display name (e.g., "The Woks of Life")
    
    Examples:
        >>> get_site_display_name("https://thewoksoflife.com/recipe/")
        'The Woks of Life'
        >>> get_site_display_name("https://www.bbcgoodfood.com/recipes/...")
        'BBC Good Food'
        >>> get_site_display_name("https://unknown-site.com/recipe")
        'Unknown Site'
    """
    if not url:
        return "Unknown Source"
    
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower().replace('www.', '')
        
        # Check known sites first
        if hostname in SITE_DISPLAY_NAMES:
            return SITE_DISPLAY_NAMES[hostname]
        
        # Fallback: Convert domain to title case
        # e.g., "my-recipe-site.com" -> "My Recipe Site"
        domain = hostname.split('.')[0]
        return domain.replace('-', ' ').title()
        
    except Exception:
        return "Unknown Source"


def check_name_collision(recipe_name: str) -> Optional[Tuple[str, str, str]]:
    """
    Check if a recipe name already exists in the local index.
    
    Args:
        recipe_name: Recipe name to check (case-insensitive)
    
    Returns:
        Tuple of (id, name, slug) if collision exists, None otherwise
    """
    if not recipe_name:
        return None
    
    try:
        with sqlite3.connect(RECIPE_INDEX_DB) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, slug FROM recipes WHERE LOWER(name) = LOWER(?)",
                (recipe_name.strip(),)
            )
            row = cursor.fetchone()
            if row:
                return (row[0], row[1], row[2])
    except Exception as e:
        logger.warning(f"Could not check name collision for '{recipe_name}': {e}")
    
    return None


def get_qualified_name(recipe_name: str, source_url: str) -> str:
    """
    Create a site-qualified recipe name to avoid collision.
    
    Args:
        recipe_name: Original recipe name
        source_url: URL the recipe came from
    
    Returns:
        Qualified name with site identifier: "Recipe Name (Site Name)"
    
    Example:
        >>> get_qualified_name("Banana Bread", "https://thewoksoflife.com/banana-bread/")
        'Banana Bread (The Woks of Life)'
    """
    site_name = get_site_display_name(source_url)
    return f"{recipe_name} ({site_name})"


def should_qualify_name(recipe_name: str, source_url: str) -> Tuple[bool, str]:
    """
    Check if recipe name needs qualification and return the appropriate name.
    
    This is the main entry point for pre-import collision detection.
    
    Args:
        recipe_name: Recipe name from scraped data
        source_url: URL being imported
    
    Returns:
        Tuple of (needs_qualification, final_name)
        - If no collision: (False, original_name)
        - If collision: (True, qualified_name)
    
    Example:
        >>> should_qualify_name("Banana Bread", "https://thewoksoflife.com/...")
        (True, 'Banana Bread (The Woks of Life)')  # if collision exists
        (False, 'Banana Bread')  # if no collision
    """
    existing = check_name_collision(recipe_name)
    
    if existing:
        existing_id, existing_name, existing_slug = existing
        qualified_name = get_qualified_name(recipe_name, source_url)
        logger.info(
            f"Name collision detected: '{recipe_name}' exists as '{existing_name}' ({existing_slug}). "
            f"Using qualified name: '{qualified_name}'"
        )
        return (True, qualified_name)
    
    return (False, recipe_name)
