"""
Utility functions for recipe URL scraping.
"""

from typing import Optional
from urllib.parse import urlparse


def get_site_origin(base_url: str) -> Optional[str]:
    """
    Extract and validate site origin from URL.
    
    Args:
        base_url: Full URL to parse
        
    Returns:
        Normalized hostname if supported, None if not supported or invalid
    """
    # Lazy import to avoid circular dependency
    from recipe_urls import SITE_ORIGINS
    
    if not isinstance(base_url, str):
        return None
    
    parsed_url = urlparse(base_url)
    hostname = parsed_url.hostname or parsed_url.netloc
    scheme = parsed_url.scheme

    if not all([scheme, hostname]):
        return None
    
    if scheme not in ['https', 'http']:
        return None
    
    normalized_domain = hostname.lower()
    
    # Check exact match first
    if normalized_domain in SITE_ORIGINS:
        return normalized_domain
    
    # Check without www prefix
    if normalized_domain.startswith('www.'):
        without_www = normalized_domain[4:]
        for origin in SITE_ORIGINS:
            if origin == without_www or origin == f"www.{without_www}":
                return origin
    else:
        # Check with www prefix
        with_www = f"www.{normalized_domain}"
        if with_www in SITE_ORIGINS:
            return with_www
    
    # Site not supported
    return None


def parse_hostname(url: str) -> str:
    """Extract hostname from URL without validation."""
    parsed = urlparse(url)
    return (parsed.hostname or parsed.netloc or "").lower()
