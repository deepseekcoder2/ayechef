"""Theme management for the panel web UI."""
import json
from pathlib import Path
from typing import Optional

# Path to themes definition file
THEMES_PATH = Path(__file__).parent / 'themes.json'

# Default theme if none configured
DEFAULT_THEME_ID = 'morning-light'

# Cached themes data
_themes_cache: Optional[dict] = None


def load_themes() -> list[dict]:
    """Load all available themes from themes.json.
    
    Returns:
        List of theme dictionaries with id, name, type, colors, chart, etc.
    """
    global _themes_cache
    
    if _themes_cache is not None:
        return _themes_cache['themes']
    
    if not THEMES_PATH.exists():
        raise FileNotFoundError(f"Themes file not found: {THEMES_PATH}")
    
    with open(THEMES_PATH) as f:
        data = json.load(f)
    
    _themes_cache = data
    return data['themes']


def get_theme(theme_id: str) -> dict:
    """Get a specific theme by ID.
    
    Args:
        theme_id: The theme identifier (e.g., 'morning-light', 'night-kitchen')
        
    Returns:
        Theme dictionary with all color definitions
        
    Raises:
        ValueError: If theme_id not found
    """
    themes = load_themes()
    
    for theme in themes:
        if theme['id'] == theme_id:
            return theme
    
    # Fall back to default if requested theme not found
    for theme in themes:
        if theme['id'] == DEFAULT_THEME_ID:
            return theme
    
    # Last resort: return first theme
    return themes[0]


def get_themes_json() -> str:
    """Get themes as JSON string for client-side use.
    
    Returns:
        JSON string containing array of theme objects
    """
    themes = load_themes()
    return json.dumps(themes)


def get_theme_choices() -> list[tuple[str, str, str]]:
    """Get themes formatted for a select dropdown.
    
    Returns:
        List of tuples: (id, name, type) for each theme
    """
    themes = load_themes()
    return [(t['id'], t['name'], t['type']) for t in themes]


def reload_themes():
    """Force reload themes from disk (useful after editing themes.json)."""
    global _themes_cache
    _themes_cache = None
    return load_themes()


def validate_theme_id(theme_id: str) -> str:
    """Validate and return a theme ID, falling back to default if invalid.
    
    Args:
        theme_id: Theme ID to validate
        
    Returns:
        Valid theme ID (original or default)
    """
    themes = load_themes()
    valid_ids = {t['id'] for t in themes}
    
    if theme_id in valid_ids:
        return theme_id
    
    return DEFAULT_THEME_ID
