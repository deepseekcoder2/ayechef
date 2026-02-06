"""
Abstract base class for recipe URL scrapers.

Each site scraper inherits from this and defines:
- RECIPE_PATTERN: regex matching recipe URLs
- UNWANTED_PATTERNS: list of regexes for URLs to exclude
- CATEGORIES: dict mapping category names to URL patterns (optional)
- host(): classmethod returning the site domain

Categories enable bulk import with category-based selection.
If CATEGORIES is empty, bulk import shows "Import All" only.
"""

from typing import List, Optional, Dict
import re
import requests
from requests.exceptions import HTTPError, RequestException
from bs4 import BeautifulSoup
from urllib.parse import urlparse


class AbstractScraper:
    """Base scraper class - handles HTTP requests and link extraction."""
    
    RECIPE_PATTERN = None
    UNWANTED_PATTERNS = []
    
    # Category configuration - supports two patterns:
    # 
    # Pattern A: Category page URLs (for sites with flat recipe URLs)
    # CATEGORY_PAGES = {
    #     'Chinese': '/category/recipes/chinese-recipes/',
    #     'Japanese': '/category/recipes/japanese-recipes/',
    # }
    # During import, we scrape each category page to get recipe URLs.
    #
    # Pattern B: URL regex patterns (for sites with category in recipe URL)
    # CATEGORIES = {
    #     'Chinese': re.compile(r'/chinese/'),
    #     'Japanese': re.compile(r'/japanese/'),
    # }
    # We can filter recipe URLs directly by pattern matching.
    #
    # Sites can define one or both. If both are defined, CATEGORY_PAGES takes precedence.
    
    CATEGORY_PAGES: Dict[str, str] = {}  # Category name -> category page URL path
    CATEGORIES: Dict[str, re.Pattern] = {}  # Category name -> URL pattern (fallback)
    CUSTOM_HREF = ("a", {"href": True})

    def __init__(self, base_url: Optional[str] = None, html: Optional[str] = None):
        self.base_url = base_url
        self.http_status = None  # Store for diagnostics

        if not html:
            try:
                response = requests.get(
                    url=self.base_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    },
                    timeout=30
                )
                self.http_status = response.status_code
                
                # Don't rely on HTTP status codes - many sites return 404 but still serve content
                # Only fail on actual connection/request failures, not status codes
                # Check if we got HTML content regardless of status code
                content_type = response.headers.get('content-type', '')
                has_html_content = (
                    'text/html' in content_type or 
                    len(response.content) > 1000  # Has substantial content
                )
                
                if response.status_code == 403:
                    # 403 Forbidden is a real block - don't try to parse
                    raise Exception(f"Access to {self.base_url} is forbidden (403).")
                elif response.status_code >= 500:
                    # Server errors are real failures
                    raise Exception(f"Server error: {response.status_code}")
                elif not has_html_content and response.status_code >= 400:
                    # No content AND error status - real failure
                    raise Exception(f"HTTP error {response.status_code} with no content")
                
                # Use whatever content we got (handles soft-404s, redirects, etc.)
                self.html = response.content
                self.soup = BeautifulSoup(self.html, "html.parser")

            except requests.exceptions.HTTPError as e:
                # This shouldn't happen anymore since we don't call raise_for_status()
                raise Exception(f"HTTP error occurred: {e}") from e

            except RequestException as e:
                raise Exception(f"Request failed: {e}.") from e

            except Exception as e:
                if "forbidden" in str(e).lower() or "server error" in str(e).lower():
                    raise  # Re-raise our own exceptions
                raise Exception(f"Unexpected error accessing {self.base_url}: {e}.") from e
        else:
            self.html = html
            self.soup = BeautifulSoup(self.html, "html.parser")

    def scrape(self) -> List[str]:
        """Extract all recipe URLs from the page."""
        try:
            tag, attrs = self.CUSTOM_HREF
            attrs["href"] = True
            href_links = [a["href"] for a in self.soup.find_all(tag, attrs)]

        except (TypeError, AttributeError) as e:
            raise ValueError(f"Failed to extract href links: {e}") from e

        unique_links = {
            self._concat_host(link) 
            for link in href_links
            if self.RECIPE_PATTERN and self.RECIPE_PATTERN.search(link)
            and not any(pattern.search(link) for pattern in self.UNWANTED_PATTERNS)
        }

        return list(unique_links)

    def _concat_host(self, link: str) -> str:
        """Ensure link has full URL with host."""
        if self.base_url and 'http' in self.base_url:
            base_parsed = urlparse(self.base_url)
            base_domain = f"{base_parsed.scheme}://{base_parsed.netloc}"
            return link if base_parsed.netloc in link else base_domain + link
        return link

    @classmethod
    def host(cls) -> str:
        """Return the hostname this scraper handles. Must be overridden."""
        raise NotImplementedError("Subclass must implement host()")

    @classmethod
    def has_categories(cls) -> bool:
        """Check if this scraper has category configuration."""
        return bool(cls.CATEGORY_PAGES) or bool(cls.CATEGORIES)

    @classmethod
    def uses_category_pages(cls) -> bool:
        """Check if this scraper uses category page scraping (vs URL pattern matching)."""
        return bool(cls.CATEGORY_PAGES)

    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of category names for this scraper."""
        if cls.CATEGORY_PAGES:
            return list(cls.CATEGORY_PAGES.keys())
        return list(cls.CATEGORIES.keys())

    @classmethod
    def get_category_page_url(cls, category: str, base_url: str) -> Optional[str]:
        """
        Get the full URL for a category page.
        
        Args:
            category: Category name
            base_url: Site base URL (e.g., 'https://thewoksoflife.com')
            
        Returns:
            Full category page URL, or None if not found
        """
        if category not in cls.CATEGORY_PAGES:
            return None
        
        path = cls.CATEGORY_PAGES[category]
        # Ensure base_url doesn't end with / and path starts with /
        base_url = base_url.rstrip('/')
        if not path.startswith('/'):
            path = '/' + path
        return base_url + path

    @classmethod
    def get_category_for_url(cls, url: str) -> Optional[str]:
        """
        Determine which category a recipe URL belongs to.
        Only works for sites using CATEGORIES (URL pattern matching).
        For CATEGORY_PAGES sites, use scrape_category_urls() instead.
        
        Args:
            url: Recipe URL to categorize
            
        Returns:
            Category name if matched, None otherwise
        """
        for category_name, pattern in cls.CATEGORIES.items():
            if pattern.search(url):
                return category_name
        return None

    @classmethod
    def filter_urls_by_categories(cls, urls: List[str], categories: List[str]) -> List[str]:
        """
        Filter URLs to only include those in the specified categories.
        Only works for sites using CATEGORIES (URL pattern matching).
        
        Args:
            urls: List of recipe URLs
            categories: List of category names to include
            
        Returns:
            Filtered list of URLs matching the specified categories
        """
        if not categories or not cls.CATEGORIES:
            return urls
        
        filtered = []
        for url in urls:
            category = cls.get_category_for_url(url)
            if category in categories:
                filtered.append(url)
        return filtered

    @classmethod
    def categorize_urls(cls, urls: List[str]) -> Dict[str, List[str]]:
        """
        Group URLs by their categories.
        Only works for sites using CATEGORIES (URL pattern matching).
        
        Args:
            urls: List of recipe URLs
            
        Returns:
            Dict mapping category names to lists of URLs.
            Includes 'Uncategorized' for URLs that don't match any category.
        """
        categorized: Dict[str, List[str]] = {name: [] for name in cls.CATEGORIES.keys()}
        categorized['Uncategorized'] = []
        
        for url in urls:
            category = cls.get_category_for_url(url)
            if category:
                categorized[category].append(url)
            else:
                categorized['Uncategorized'].append(url)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}

    @classmethod
    def scrape_category_urls(cls, category: str, base_url: str, max_pages: int = 50) -> List[str]:
        """
        Scrape recipe URLs from a category page with pagination support.
        Used for sites with CATEGORY_PAGES configuration.
        
        Args:
            category: Category name
            base_url: Site base URL
            max_pages: Maximum number of pages to fetch (safety limit)
            
        Returns:
            List of recipe URLs found across all category pages
        """
        category_url = cls.get_category_page_url(category, base_url)
        if not category_url:
            return []
        
        all_urls = []
        page = 1
        
        while page <= max_pages:
            # Construct paginated URL (WordPress style: /page/N/)
            if page == 1:
                page_url = category_url
            else:
                # Ensure URL ends with / before adding page/N/
                page_url = category_url.rstrip('/') + f'/page/{page}/'
            
            try:
                scraper = cls(base_url=page_url)
                urls = scraper.scrape()
                
                if not urls:
                    # No more recipes found - end of pagination
                    break
                
                # Check for duplicate URLs (we've looped back to page 1)
                new_urls = [u for u in urls if u not in all_urls]
                if not new_urls:
                    break
                
                all_urls.extend(new_urls)
                page += 1
                
            except Exception as e:
                # Page doesn't exist or error - end pagination
                if page == 1:
                    # First page failed - return empty
                    return []
                break
        
        return all_urls
