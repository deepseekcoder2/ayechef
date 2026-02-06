"""URL utility functions shared across import modules."""


def normalize_url(url: str) -> str:
    """
    Normalize a URL for consistent duplicate comparison.
    
    Handles:
    - Trailing slashes
    - www prefix
    - http/https
    - Whitespace
    - Case
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL string, or empty string if input is None/empty
    """
    if not url:
        return ""
    url = url.strip().lower()
    url = url.rstrip('/')
    if url.startswith('http://'):
        url = 'https://' + url[7:]
    if '://www.' in url:
        url = url.replace('://www.', '://')
    return url
