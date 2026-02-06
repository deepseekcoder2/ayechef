#!/usr/bin/env python3
"""
Mealie Unified Client
=====================

Single entry point for all Mealie operations with two backend modes:
- API mode (default): Uses Mealie REST API with connection pooling and retries
- Direct DB mode: Reads directly from Mealie's SQLite database (read-only)

The client automatically selects the backend based on configuration:
- Check `USER_CONFIG.get('mealie', {}).get('use_direct_db', False)`
- If `use_direct_db=True` and DB is accessible, use DB for reads
- All writes always go through the API (Mealie must handle its own schema)

Usage:
    from mealie_client import MealieClient, MealieClientError
    
    client = MealieClient()  # Auto-selects mode based on config
    
    # Read operations (use DB if enabled, else API)
    recipes = client.get_all_recipes()
    recipe = client.get_recipe("chicken-tikka-masala")
    
    # Write operations (always API)
    new_recipe = client.create_recipe_from_url("https://example.com/recipe")
    client.update_recipe("chicken-tikka-masala", {"servings": 6})

Architecture:
    MealieClient (public facade)
    ├── _MealieAPIAdapter (internal - handles all API calls)
    │   ├── Connection pooling via requests.Session
    │   ├── Retry strategy with exponential backoff
    │   └── Rate limiting via mealie_rate_limit()
    └── _MealieDBAdapter (internal - handles direct SQLite reads)
        ├── Read-only SQLite connection
        └── Maps to Mealie's internal schema

Author: AI Recipe System
"""

import sqlite3
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import date, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    MEALIE_URL,
    MEALIE_TOKEN,
    USER_CONFIG,
    mealie_rate_limit,
    get_mealie_headers,
)
from tools.logging_utils import get_logger

# Module logger
logger = get_logger(__name__)

# =============================================================================
# IDENTIFIER HELPERS
# =============================================================================

# Strict UUID patterns to avoid mis-classifying typical slugs.
# - 36-char canonical UUID with version/variant validation
# - 32-char hex UUID (no dashes)
_UUID_CANONICAL_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-"
    r"[0-9a-fA-F]{12}$"
)
_UUID_HEX_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def _is_uuid_identifier(value: Any) -> bool:
    """
    Return True if `value` looks like a UUID (dashed or 32-hex).

    This is used to route `MealieClient.get_recipe()` calls to either the
    slug path or the ID path. We keep the heuristic strict to reduce the
    chance of accidentally treating normal recipe slugs as UUIDs.
    """
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    return bool(_UUID_CANONICAL_RE.match(v) or _UUID_HEX_RE.match(v))


# =============================================================================
# EXCEPTIONS
# =============================================================================

class MealieClientError(Exception):
    """
    Base exception for MealieClient errors.
    
    Provides context about what operation failed and why.
    
    Attributes:
        message: Human-readable error description
        operation: The operation that failed (e.g., "get_recipe", "create_recipe")
        details: Additional context (e.g., HTTP status code, response body)
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.operation = operation
        self.details = details or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with context."""
        parts = [self.message]
        if self.operation:
            parts.insert(0, f"[{self.operation}]")
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"({detail_str})")
        return " ".join(parts)


class MealieAPIError(MealieClientError):
    """Exception raised for API-specific errors (HTTP failures, timeouts)."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if status_code is not None:
            details['status_code'] = status_code
        if response_body:
            # Truncate long response bodies
            details['response'] = response_body[:200] + "..." if len(response_body) > 200 else response_body
        super().__init__(message, operation, details)
        self.status_code = status_code
        self.response_body = response_body


class MealieDBError(MealieClientError):
    """Exception raised for database-specific errors (connection, query failures)."""
    pass


# =============================================================================
# INTERNAL: API ADAPTER
# =============================================================================

class _MealieAPIAdapter:
    """
    Internal adapter for Mealie REST API operations.
    
    Features:
    - HTTP connection pooling via requests.Session
    - Automatic retry with exponential backoff for transient failures
    - Rate limiting integration via mealie_rate_limit()
    - Comprehensive error handling with context
    
    This class is internal - external code should use MealieClient.
    """
    
    # Retry configuration
    RETRY_TOTAL = 3
    RETRY_BACKOFF_FACTOR = 0.5
    RETRY_STATUS_FORCELIST = [429, 500, 502, 503, 504]
    
    # Connection pool configuration
    POOL_CONNECTIONS = 10
    POOL_MAXSIZE = 20
    
    # Default timeout (seconds)
    DEFAULT_TIMEOUT = 30
    
    def __init__(self, base_url: str, token: str, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the API adapter with connection pooling.
        
        Args:
            base_url: Mealie base URL (e.g., "http://localhost:9925")
            token: Mealie API token (JWT)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api"
        self.token = token
        self.timeout = timeout
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.RETRY_TOTAL,
            backoff_factor=self.RETRY_BACKOFF_FACTOR,
            status_forcelist=self.RETRY_STATUS_FORCELIST,
            allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        
        # Connection pool adapter
        adapter = HTTPAdapter(
            pool_connections=self.POOL_CONNECTIONS,
            pool_maxsize=self.POOL_MAXSIZE,
            max_retries=retry_strategy,
        )
        
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Default headers
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        })
        
        logger.debug(f"_MealieAPIAdapter initialized: base_url={base_url}")
    
    def close(self) -> None:
        """Clean up connections."""
        self.session.close()
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Union[Dict[str, Any], bool]:
        """
        Perform an HTTP request with rate limiting and error handling.
        
        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (e.g., "/recipes" or "/recipes/{slug}")
            data: JSON body for POST/PATCH requests
            params: Query parameters
            timeout: Override default timeout
        
        Returns:
            Parsed JSON response, empty dict for 204, or True for DELETE
        
        Raises:
            MealieAPIError: On HTTP or network errors
        """
        url = f"{self.api_base}{endpoint}"
        timeout = timeout or self.timeout
        
        try:
            with mealie_rate_limit():
                if method == "GET":
                    response = self.session.get(url, params=params, timeout=timeout)
                elif method == "POST":
                    response = self.session.post(url, json=data, params=params, timeout=timeout)
                elif method == "PATCH":
                    response = self.session.patch(url, json=data, timeout=timeout)
                elif method == "PUT":
                    response = self.session.put(url, json=data, timeout=timeout)
                elif method == "DELETE":
                    response = self.session.delete(url, params=params, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # DELETE returns True on success
            if method == "DELETE":
                return True
            
            # Handle empty responses (204 No Content, etc.)
            if response.status_code == 204 or not response.text:
                return {}
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            raise MealieAPIError(
                f"HTTP error: {e}",
                operation=method,
                status_code=e.response.status_code if e.response else None,
                response_body=e.response.text if e.response else None,
            )
        except requests.exceptions.Timeout:
            raise MealieAPIError(
                f"Request timed out after {timeout}s",
                operation=method,
                details={'endpoint': endpoint, 'timeout': timeout},
            )
        except requests.exceptions.RequestException as e:
            raise MealieAPIError(
                f"Network error: {e}",
                operation=method,
                details={'endpoint': endpoint},
            )
    
    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform a GET request."""
        return self._request("GET", endpoint, params=params, timeout=timeout)
    
    def _post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform a POST request."""
        return self._request("POST", endpoint, data=data, params=params, timeout=timeout)
    
    def _patch(
        self,
        endpoint: str,
        data: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform a PATCH request."""
        return self._request("PATCH", endpoint, data=data, timeout=timeout)
    
    def _put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform a PUT request."""
        return self._request("PUT", endpoint, data=data, timeout=timeout)
    
    def _delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> bool:
        """Perform a DELETE request."""
        return self._request("DELETE", endpoint, params=params, timeout=timeout)
    
    def _get_paginated(
        self,
        endpoint: str,
        page_size: int = 100,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all pages from a paginated endpoint.
        
        Args:
            endpoint: API endpoint
            page_size: Items per page (use -1 for all if supported)
            params: Additional query parameters
            timeout: Override default timeout
        
        Returns:
            List of all items from all pages
        
        Raises:
            MealieAPIError: On HTTP or network errors
        """
        all_items = []
        page = 1
        params = params or {}
        
        while True:
            page_params = {**params, 'page': page, 'perPage': page_size}
            data = self._get(endpoint, params=page_params, timeout=timeout)
            
            items = data.get('items', [])
            all_items.extend(items)
            
            # Check if more pages exist
            total_pages = data.get('total_pages', 1)
            if page >= total_pages:
                break
            page += 1
        
        return all_items
    
    # -------------------------------------------------------------------------
    # Recipe operations
    # -------------------------------------------------------------------------
    
    def get_all_recipes(self) -> List[Dict[str, Any]]:
        """Fetch all recipes with pagination."""
        return self._get_paginated('/recipes', page_size=100)
    
    def get_recipe(self, slug: str) -> Dict[str, Any]:
        """Fetch a single recipe by slug."""
        return self._get(f'/recipes/{slug}')
    
    def get_recipe_by_id(self, recipe_id: str) -> Dict[str, Any]:
        """Fetch a single recipe by ID."""
        return self._get(f'/recipes/{recipe_id}')
    
    def search_recipes(self, query: str, per_page: int = 50) -> List[Dict[str, Any]]:
        """
        Search recipes by keyword/name.
        
        Uses Mealie's built-in search API which does text matching against recipe names.
        
        Args:
            query: Search term (e.g., "garlic bread", "miso soup")
            per_page: Number of results to return (default 50)
        
        Returns:
            List of recipe dicts with id, name, tags, etc.
        """
        params = {"search": query, "perPage": per_page}
        data = self._get('/recipes', params=params, timeout=90)
        return data.get('items', [])
    
    def get_all_recipe_urls(self) -> Set[str]:
        """Fetch all original URLs from recipes for duplicate checking."""
        from utils.url_utils import normalize_url
        
        recipes = self.get_all_recipes()
        urls = set()
        for recipe in recipes:
            if recipe.get('orgURL'):
                urls.add(normalize_url(recipe['orgURL']))
        return urls
    
    def get_recipes_batch(self, slugs: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple recipes by slug in parallel.
        
        Thread Safety: Uses ThreadPoolExecutor with requests.Session which is
        thread-safe for concurrent reads. The mealie_rate_limit() semaphore
        controls actual API concurrency (default: 2 for SQLite backend).
        
        Args:
            slugs: List of recipe slugs to fetch
        
        Returns:
            Dict mapping slug to recipe data (missing slugs are omitted)
        """
        import concurrent.futures
        
        if not slugs:
            return {}
        
        # Get configured concurrency from config (default 2 for SQLite safety)
        max_concurrent = USER_CONFIG.get('mealie', {}).get('max_concurrent_requests', 2)
        max_workers = min(max_concurrent, len(slugs))
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_slug = {
                executor.submit(self._fetch_single_safe, slug): slug
                for slug in slugs
            }
            
            for future in concurrent.futures.as_completed(future_to_slug):
                slug = future_to_slug[future]
                try:
                    recipe = future.result()
                    if recipe:
                        results[slug] = recipe
                except Exception as e:
                    logger.error(f"Failed to fetch recipe {slug}: {e}")
        
        return results
    
    def _fetch_single_safe(self, slug: str) -> Optional[Dict[str, Any]]:
        """Fetch single recipe, returning None on error (for parallel use)."""
        try:
            return self.get_recipe(slug)
        except Exception:
            return None
    
    def check_duplicate_urls(self, urls: List[str]) -> Dict[str, bool]:
        """
        Check which URLs already exist in Mealie.
        
        Args:
            urls: List of URLs to check
        
        Returns:
            Dict mapping URL to exists boolean
        """
        from utils.url_utils import normalize_url
        
        existing_urls = self.get_all_recipe_urls()
        
        results = {}
        for url in urls:
            results[url] = normalize_url(url) in existing_urls
        
        return results
    
    def create_recipe_from_url(self, url: str, include_tags: bool = False) -> Dict[str, Any]:
        """
        Create a recipe from a URL using Mealie's scraper.
        
        Args:
            url: Recipe URL to import
            include_tags: Whether to include tags from source
        
        Returns:
            Created recipe data (contains 'slug' for the new recipe)
        """
        payload = {"url": url, "includeTags": include_tags}
        return self._post('/recipes/create/url', data=payload, timeout=60)
    
    def scrape_recipe_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape recipe data from URL WITHOUT creating the recipe.
        
        Uses Mealie's test-scrape-url endpoint to extract recipe metadata
        for pre-import collision detection.
        
        Args:
            url: Recipe URL to scrape
        
        Returns:
            Scraped recipe data in JSON-LD format with keys:
            - name: Recipe title
            - description: Recipe description
            - recipeIngredient: List of ingredient strings
            - recipeInstructions: List of instruction objects
            - image: List of image URLs
            - prepTime, cookTime, totalTime: ISO 8601 durations
            - recipeYield: Serving info
            - nutrition: Nutrition data
            - etc.
        
        Raises:
            MealieAPIError: If scraping fails (invalid URL, site not supported, etc.)
        """
        payload = {"url": url}
        return self._post('/recipes/test-scrape-url', data=payload, timeout=60)
    
    def create_recipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new recipe manually.
        
        Mealie's CreateRecipe schema only accepts 'name' initially.
        Use update_recipe() to add full details after creation.
        
        Args:
            data: Recipe data (must contain 'name')
        
        Returns:
            Created recipe stub (use slug to update with full data)
        """
        if 'name' not in data:
            raise MealieClientError("Recipe data must contain 'name'", operation="create_recipe")
        
        # Create stub with just the name
        result = self._post('/recipes', data={'name': data['name']})
        
        # Handle both response formats: Mealie may return string slug or dict with slug field
        if isinstance(result, str):
            slug = result
        elif isinstance(result, dict):
            slug = result.get('slug', '')
        else:
            raise MealieClientError(
                f"Unexpected response type from recipe creation: {type(result).__name__}",
                operation="create_recipe"
            )
        
        if not slug:
            raise MealieClientError(
                "Recipe creation returned empty slug",
                operation="create_recipe",
                details={'response': str(result)[:200]}
            )
        
        # If additional fields provided, update immediately
        if len(data) > 1:
            update_data = {k: v for k, v in data.items() if k != 'name'}
            if update_data:
                return self.update_recipe(slug, update_data)
        
        return result
    
    def update_recipe(self, slug: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing recipe.
        
        Args:
            slug: Recipe slug (or ID)
            data: Fields to update
        
        Returns:
            Updated recipe data
        """
        return self._patch(f'/recipes/{slug}', data=data)
    
    def upload_recipe_image(self, slug: str, image_url: str) -> bool:
        """
        Upload an image to a recipe by URL.
        
        Mealie will download and store the image automatically.
        
        Args:
            slug: Recipe slug (or ID)
            image_url: URL of the image to upload
        
        Returns:
            True on success, False on failure
        """
        try:
            self._post(f'/recipes/{slug}/image', data={'url': image_url}, timeout=30)
            return True
        except MealieAPIError:
            return False
    
    def delete_recipe(self, slug: str) -> bool:
        """
        Delete a recipe.
        
        Args:
            slug: Recipe slug (or ID)
        
        Returns:
            True on success
        """
        return self._delete(f'/recipes/{slug}')
    
    # -------------------------------------------------------------------------
    # Foods, Units, Tags operations
    # -------------------------------------------------------------------------
    
    def get_all_foods(self) -> List[Dict[str, Any]]:
        """Fetch all foods (ingredients database)."""
        # Use perPage=-1 to get all in one request
        data = self._get('/foods', params={'perPage': -1})
        return data.get('items', [])
    
    def get_all_units(self) -> List[Dict[str, Any]]:
        """Fetch all units."""
        data = self._get('/units', params={'perPage': -1})
        return data.get('items', [])
    
    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Fetch all tags."""
        data = self._get('/organizers/tags', params={'perPage': -1})
        return data.get('items', [])
    
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Fetch all categories."""
        data = self._get('/organizers/categories', params={'perPage': -1})
        return data.get('items', [])
    
    def create_food(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Create a new food entry.
        
        Args:
            name: Food name
            **kwargs: Additional fields (description, labelId, etc.)
        
        Returns:
            Created food data
        """
        data = {'name': name, **kwargs}
        return self._post('/foods', data=data)
    
    def create_unit(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Create a new unit.
        
        Args:
            name: Unit name
            **kwargs: Additional fields (abbreviation, description, etc.)
        
        Returns:
            Created unit data
        """
        data = {'name': name, **kwargs}
        return self._post('/units', data=data)
    
    def create_tag(self, name: str) -> Dict[str, Any]:
        """
        Create a new tag.
        
        Args:
            name: Tag name
        
        Returns:
            Created tag data
        """
        return self._post('/organizers/tags', data={'name': name})
    
    def get_food(self, food_id: str) -> Dict[str, Any]:
        """
        Fetch a single food by ID.
        
        Args:
            food_id: Food ID
        
        Returns:
            Food dictionary
        """
        return self._get(f'/foods/{food_id}')
    
    def update_food(self, food_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing food.
        
        Args:
            food_id: Food ID
            data: Fields to update (must include all required fields)
        
        Returns:
            Updated food data
        """
        return self._put(f'/foods/{food_id}', data=data)
    
    def get_all_labels(self) -> List[Dict[str, Any]]:
        """Fetch all labels."""
        data = self._get('/groups/labels', params={'perPage': -1})
        return data.get('items', [])
    
    def create_label(self, name: str, color: str = '#808080') -> Dict[str, Any]:
        """
        Create a new label.
        
        Args:
            name: Label name
            color: Label color (hex code, default: '#808080')
        
        Returns:
            Created label data
        """
        return self._post('/groups/labels', data={'name': name, 'color': color})
    
    # -------------------------------------------------------------------------
    # Meal plan operations
    # -------------------------------------------------------------------------
    
    def get_meal_plans(
        self,
        start_date: Union[str, date],
        end_date: Union[str, date]
    ) -> List[Dict[str, Any]]:
        """
        Fetch meal plan entries for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD string or date object)
            end_date: End date (YYYY-MM-DD string or date object)
        
        Returns:
            List of meal plan entries
        """
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()
        
        params = {'start_date': start_date, 'end_date': end_date}
        logger.debug(f"API get_meal_plans: requesting {params}")
        data = self._get('/households/mealplans', params=params, timeout=60)
        logger.debug(f"API get_meal_plans: raw response type={type(data).__name__}, preview={str(data)[:200] if data else 'None'}")
        
        # Response may be a list directly or wrapped in 'items'
        # Handle None response (API returns null) and missing/null items
        if data is None:
            logger.warning("API get_meal_plans: received None response")
            return []
        if isinstance(data, list):
            logger.debug(f"API get_meal_plans: returning list with {len(data)} items")
            return data
        items = data.get('items')
        result = items if isinstance(items, list) else []
        logger.debug(f"API get_meal_plans: extracted {len(result)} items from dict response")
        return result
    
    def create_meal_plan_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a meal plan entry.
        
        Args:
            data: Meal plan entry data (date, entryType, recipeId, etc.)
        
        Returns:
            Created meal plan entry
        """
        return self._post('/households/mealplans', data=data)
    
    def delete_meal_plan_entry(self, entry_id: str) -> bool:
        """
        Delete a meal plan entry.
        
        Args:
            entry_id: Meal plan entry ID
        
        Returns:
            True on success
        """
        return self._delete(f'/households/mealplans/{entry_id}')
    
    # -------------------------------------------------------------------------
    # Shopping list operations
    # -------------------------------------------------------------------------
    
    def get_all_shopping_lists(self) -> List[Dict[str, Any]]:
        """
        Fetch all shopping lists.
        
        Returns:
            List of shopping list dictionaries
        """
        return self._get_paginated('/households/shopping/lists', page_size=100)
    
    def get_shopping_list(self, list_id: str) -> Dict[str, Any]:
        """
        Fetch a shopping list by ID.
        
        Args:
            list_id: Shopping list ID
        
        Returns:
            Shopping list data with 'listItems' array
        """
        return self._get(f'/households/shopping/lists/{list_id}')
    
    def create_shopping_list(self, name: str) -> Dict[str, Any]:
        """
        Create a new shopping list.
        
        Args:
            name: Shopping list name
        
        Returns:
            Created shopping list data
        """
        return self._post('/households/shopping/lists', data={'name': name})
    
    def delete_shopping_list(self, list_id: str) -> bool:
        """
        Delete a shopping list.
        
        Args:
            list_id: Shopping list ID
        
        Returns:
            True on success
        """
        return self._delete(f'/households/shopping/lists/{list_id}')
    
    def add_recipes_to_shopping_list(self, list_id: str, recipes: List[Dict[str, Any]]) -> bool:
        """
        Add multiple recipes to a shopping list with scaling.
        
        Args:
            list_id: Shopping list ID
            recipes: List of dicts with 'recipeId' and 'recipeIncrementQuantity'
        
        Returns:
            True on success
        """
        try:
            self._post(f'/households/shopping/lists/{list_id}/recipe', data=recipes)
            return True
        except MealieAPIError:
            return False
    
    def add_shopping_item(self, list_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a single item to a shopping list.
        
        IMPORTANT: Shopping list writes MUST be sequential. Mealie corrupts
        data when receiving bulk/parallel writes. Use add_shopping_items()
        for multiple items - it handles sequential internally.
        
        Args:
            list_id: Shopping list ID
            item: Item data (display, quantity, unit, food, note, etc.)
        
        Returns:
            Created shopping list item
        """
        # Handle quantity carefully: 0 is a valid value, only default to 1 if None
        quantity = item.get('quantity')
        if quantity is None:
            quantity = 1
        
        payload = {
            'shoppingListId': list_id,
            'display': item.get('display', '').strip(),
            'quantity': quantity,
            'note': item.get('note', ''),
            'checked': item.get('checked', False),
            'position': item.get('position', 0),
        }
        
        # Add optional fields
        if item.get('unit'):
            payload['unit'] = item['unit']
        if item.get('food'):
            payload['food'] = item['food']
        if item.get('foodId'):
            payload['foodId'] = item['foodId']
        if item.get('unitId'):
            payload['unitId'] = item['unitId']
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        return self._post('/households/shopping/items', data=payload)
    
    def add_shopping_items(
        self,
        list_id: str,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add multiple items to a shopping list SEQUENTIALLY.
        
        CRITICAL: Mealie corrupts shopping lists when receiving bulk parallel
        writes. This method adds items one-by-one to prevent data corruption.
        
        Args:
            list_id: Shopping list ID
            items: List of item data dicts
        
        Returns:
            List of created items (may be shorter than input if some failed)
        """
        results = []
        for item in items:
            try:
                result = self.add_shopping_item(list_id, item)
                results.append(result)
            except MealieAPIError as e:
                logger.warning(f"Failed to add shopping item '{item.get('display', 'unknown')}': {e}")
                # Continue with remaining items
        return results
    
    def delete_shopping_items(self, item_ids: List[str]) -> bool:
        """
        Delete multiple shopping list items.
        
        Args:
            item_ids: List of shopping list item IDs
        
        Returns:
            True on success
        """
        if not item_ids:
            return True
        
        return self._delete('/households/shopping/items', params={'ids': item_ids})


# =============================================================================
# INTERNAL: DATABASE ADAPTER
# =============================================================================

class _MealieDBAdapter:
    """
    Internal adapter for direct Mealie database reads.
    
    Reads directly from Mealie's SQLite database for faster bulk operations.
    Connection is read-only to prevent accidental corruption.
    
    This class is internal - external code should use MealieClient.
    
    Database Path:
        Hardcoded to /mealie-data/mealie.db (standard Docker volume mount)
    
    Mealie Schema (simplified):
        - recipes: id, slug, name, description, recipe_ingredient, ...
        - ingredient_foods: id, name, description, label_id, ...
        - ingredient_units: id, name, abbreviation, description, ...
        - tags: id, name, slug, ...
        - recipe_tags: recipe_id, tag_id (join table)
        - meal_plans: id, date, entry_type, recipe_id, ...
    """
    
    # Default DB path (Docker volume mount location)
    DEFAULT_DB_PATH = '/mealie-data/mealie.db'
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database adapter.
        
        Args:
            db_path: Path to Mealie's SQLite database
        
        Raises:
            MealieDBError: If database is not accessible
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        
        # Verify database exists and is accessible
        if not Path(self.db_path).exists():
            raise MealieDBError(
                f"Mealie database not found at {self.db_path}",
                operation="init",
                details={'db_path': self.db_path},
            )
        
        # Test connection
        try:
            conn = self._get_connection()
            conn.close()
        except sqlite3.Error as e:
            raise MealieDBError(
                f"Cannot connect to Mealie database: {e}",
                operation="init",
                details={'db_path': self.db_path},
            )
        
        logger.debug(f"_MealieDBAdapter initialized: db_path={self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a read-only database connection.
        
        Returns:
            SQLite connection in read-only mode
        """
        # Use URI mode for read-only access
        conn = sqlite3.connect(
            f'file:{self.db_path}?mode=ro',
            uri=True,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row to a dictionary."""
        return dict(row)
    
    def _rows_to_list(self, rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
        """Convert a list of rows to a list of dictionaries."""
        return [self._row_to_dict(row) for row in rows]
    
    def _format_uuid(self, db_id: Optional[str]) -> Optional[str]:
        """
        Convert a dashless database UUID to standard dashed format.
        
        Mealie's SQLite database stores UUIDs without dashes (32 chars),
        but the REST API returns them with dashes (36 chars). This method
        ensures consistency with API response format.
        
        Args:
            db_id: UUID from database (32 chars, no dashes) or None
        
        Returns:
            UUID with dashes (36 chars) or None if input is None/empty
        
        Examples:
            '000be282ac574aeeb09d77e580878ddb' -> '000be282-ac57-4aee-b09d-77e580878ddb'
            None -> None
            '' -> ''
        """
        if not db_id:
            return db_id
        
        # Already has dashes - return as-is
        if len(db_id) == 36 and '-' in db_id:
            return db_id
        
        # Convert 32-char hex to 8-4-4-4-12 format
        if len(db_id) == 32:
            return f"{db_id[:8]}-{db_id[8:12]}-{db_id[12:16]}-{db_id[16:20]}-{db_id[20:]}"
        
        # Unexpected format - return as-is and log warning
        logger.warning(f"Unexpected UUID format: {db_id!r} (len={len(db_id)})")
        return db_id
    
    def _normalize_uuid_for_query(self, uuid_str: Optional[str]) -> Optional[str]:
        """
        Convert a dashed UUID to dashless format for database queries.
        
        The database stores UUIDs without dashes, so when looking up by ID,
        we need to strip dashes from input UUIDs.
        
        Args:
            uuid_str: UUID (may have dashes or not)
        
        Returns:
            UUID without dashes, suitable for DB query
        """
        if not uuid_str:
            return uuid_str
        return uuid_str.replace('-', '')
    
    # -------------------------------------------------------------------------
    # Recipe operations
    # -------------------------------------------------------------------------
    
    def get_all_recipes(self) -> List[Dict[str, Any]]:
        """
        Fetch all recipes from the database.
        
        Returns:
            List of recipe dictionaries matching Mealie's API response format
        """
        # Query Mealie's actual schema - note the column names differ from API
        query = """
            SELECT 
                id, slug, name, description, image, 
                total_time, prep_time, cook_time, perform_time,
                recipe_yield, recipe_servings, rating,
                org_url, date_added, date_updated, update_at,
                "recipeCuisine"
            FROM recipes
            ORDER BY name
        """
        
        with self._connection() as conn:
            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        
        # Convert to list of dicts with API-compatible keys
        results = []
        for row in rows:
            recipe = dict(zip(columns, row))
            # CRITICAL: Convert ID from dashless DB format to dashed API format
            # This prevents ID mismatch between DB mode and local index
            recipe['id'] = self._format_uuid(recipe.get('id'))
            # Map DB column names to API response format
            recipe['orgURL'] = recipe.pop('org_url', None)
            recipe['servings'] = recipe.pop('recipe_servings', None)
            recipe['cuisine'] = recipe.pop('recipeCuisine', None)
            # Pop both date fields unconditionally, then choose non-None value
            update_at = recipe.pop('update_at', None)
            date_updated = recipe.pop('date_updated', None)
            recipe['updatedAt'] = update_at or date_updated
            results.append(recipe)
        
        return results
    
    def get_recipe_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single recipe by slug with FULL data (ingredients, instructions, tags).
        
        This returns complete recipe data matching the API response format,
        enabling full DB mode operation without fallback to API.
        
        Args:
            slug: Recipe slug
        
        Returns:
            Full recipe dictionary or None if not found
        """
        return self._get_recipe_full(slug_filter=slug)
    
    def _get_recipe_full(
        self, 
        slug_filter: Optional[str] = None, 
        id_filter: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Internal method to fetch complete recipe data from database.
        
        Fetches main recipe data plus:
        - Ingredients with resolved food/unit names
        - Instructions
        - Tags
        - Categories
        
        Args:
            slug_filter: Filter by slug (mutually exclusive with id_filter)
            id_filter: Filter by ID (mutually exclusive with slug_filter)
        
        Returns:
            Full recipe dictionary matching API response format, or None
        """
        # Build filter clause
        if slug_filter:
            where_clause = "WHERE r.slug = ?"
            filter_value = slug_filter
        elif id_filter:
            where_clause = "WHERE r.id = ?"
            # Normalize ID for DB query (strip dashes)
            filter_value = self._normalize_uuid_for_query(id_filter)
        else:
            raise ValueError("Must provide either slug_filter or id_filter")
        
        # Main recipe query
        recipe_query = f"""
            SELECT 
                r.id, r.slug, r.name, r.description, r.image,
                r.total_time, r.prep_time, r.cook_time, r.perform_time,
                r.recipe_yield, r.recipe_servings, r.rating,
                r.org_url, r.date_added, r.date_updated, r.update_at,
                r."recipeCuisine",
                r.recipe_yield_quantity, r.last_made
            FROM recipes r
            {where_clause}
        """
        
        with self._connection() as conn:
            # Fetch main recipe
            cursor = conn.execute(recipe_query, (filter_value,))
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            
            if not row:
                return None
            
            recipe = dict(zip(columns, row))
            recipe_id = recipe['id']  # Dashless ID for joins
            
            # Convert ID to dashed format for output
            recipe['id'] = self._format_uuid(recipe_id)
            
            # Map DB column names to API response format
            recipe['orgURL'] = recipe.pop('org_url', None)
            recipe['servings'] = recipe.pop('recipe_servings', None)
            recipe['cuisine'] = recipe.pop('recipeCuisine', None)
            update_at = recipe.pop('update_at', None)
            date_updated = recipe.pop('date_updated', None)
            recipe['updatedAt'] = update_at or date_updated
            recipe['lastMade'] = recipe.pop('last_made', None)
            recipe['yieldQuantity'] = recipe.pop('recipe_yield_quantity', None)
            
            # Fetch ingredients with resolved food/unit names
            ingredients_query = """
                SELECT 
                    ri.position, ri.quantity, ri.note, ri.original_text,
                    ri.unit_id, ri.food_id, ri.reference_id,
                    iu.name AS unit_name, iu.abbreviation AS unit_abbr,
                    iu.plural_name AS unit_plural, iu.plural_abbreviation AS unit_plural_abbr,
                    if.name AS food_name, if.plural_name AS food_plural,
                    if.description AS food_description
                FROM recipes_ingredients ri
                LEFT JOIN ingredient_units iu ON ri.unit_id = iu.id
                LEFT JOIN ingredient_foods if ON ri.food_id = if.id
                WHERE ri.recipe_id = ?
                ORDER BY ri.position
            """
            cursor = conn.execute(ingredients_query, (recipe_id,))
            ingredient_rows = cursor.fetchall()
            
            recipe['recipeIngredient'] = []
            for ing_row in ingredient_rows:
                ing = {
                    'quantity': ing_row[1],
                    'note': ing_row[2] or '',
                    'originalText': ing_row[3] or '',
                    'display': ing_row[3] or '',  # Use original_text as display
                }
                
                # Add unit object if present
                if ing_row[4]:  # unit_id
                    ing['unit'] = {
                        'id': self._format_uuid(ing_row[4]),
                        'name': ing_row[7] or '',
                        'abbreviation': ing_row[8] or '',
                        'pluralName': ing_row[9] or '',
                        'pluralAbbreviation': ing_row[10] or '',
                    }
                
                # Add food object if present
                if ing_row[5]:  # food_id
                    ing['food'] = {
                        'id': self._format_uuid(ing_row[5]),
                        'name': ing_row[11] or '',
                        'pluralName': ing_row[12] or '',
                        'description': ing_row[13] or '',
                    }
                
                recipe['recipeIngredient'].append(ing)
            
            # Fetch instructions
            instructions_query = """
                SELECT position, title, text, summary
                FROM recipe_instructions
                WHERE recipe_id = ?
                ORDER BY position
            """
            cursor = conn.execute(instructions_query, (recipe_id,))
            instruction_rows = cursor.fetchall()
            
            recipe['recipeInstructions'] = [
                {
                    'id': str(idx),  # API uses string IDs
                    'title': row[1] or '',
                    'text': row[2] or '',
                    'summary': row[3] or '',
                }
                for idx, row in enumerate(instruction_rows)
            ]
            
            # Fetch tags
            tags_query = """
                SELECT t.id, t.name, t.slug
                FROM recipes_to_tags rtt
                JOIN tags t ON rtt.tag_id = t.id
                WHERE rtt.recipe_id = ?
            """
            cursor = conn.execute(tags_query, (recipe_id,))
            tag_rows = cursor.fetchall()
            
            recipe['tags'] = [
                {
                    'id': self._format_uuid(row[0]),
                    'name': row[1] or '',
                    'slug': row[2] or '',
                }
                for row in tag_rows
            ]
            
            # Fetch categories
            categories_query = """
                SELECT c.id, c.name, c.slug
                FROM recipes_to_categories rtc
                JOIN categories c ON rtc.category_id = c.id
                WHERE rtc.recipe_id = ?
            """
            cursor = conn.execute(categories_query, (recipe_id,))
            category_rows = cursor.fetchall()
            
            recipe['recipeCategory'] = [
                {
                    'id': self._format_uuid(row[0]),
                    'name': row[1] or '',
                    'slug': row[2] or '',
                }
                for row in category_rows
            ]
            
            return recipe
    
    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single recipe by ID with FULL data (ingredients, instructions, tags).
        
        Args:
            recipe_id: Recipe ID (UUID, may have dashes or not)
        
        Returns:
            Full recipe dictionary or None if not found
        """
        return self._get_recipe_full(id_filter=recipe_id)
    
    def get_recipes_full_batch(self, slugs: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple complete recipes by slug in a single batch.
        
        This is MUCH faster than individual API calls for bulk operations.
        Uses efficient SQL queries to fetch all data at once.
        
        Args:
            slugs: List of recipe slugs to fetch
        
        Returns:
            Dict mapping slug to full recipe data (missing slugs omitted)
        """
        if not slugs:
            return {}
        
        # For now, fetch individually (already fast with direct DB)
        # Could be further optimized with bulk queries if needed
        results = {}
        for slug in slugs:
            recipe = self.get_recipe_by_slug(slug)
            if recipe:
                results[slug] = recipe
        
        return results
    
    def get_all_recipes_full(self) -> List[Dict[str, Any]]:
        """
        Fetch ALL recipes with FULL data (ingredients, instructions, tags).
        
        This is the fastest way to get complete data for all recipes,
        useful for full database rebuilds. Uses bulk SQL queries.
        
        Returns:
            List of full recipe dictionaries
        
        Performance:
            - Single query for all recipes
            - Bulk queries for ingredients, instructions, tags, categories
            - Much faster than N individual API calls
        """
        logger.info("📚 Fetching all recipes with full data from DB...")
        
        # Step 1: Get all recipe base data
        recipe_query = """
            SELECT 
                r.id, r.slug, r.name, r.description, r.image,
                r.total_time, r.prep_time, r.cook_time, r.perform_time,
                r.recipe_yield, r.recipe_servings, r.rating,
                r.org_url, r.date_added, r.date_updated, r.update_at,
                r."recipeCuisine",
                r.recipe_yield_quantity, r.last_made
            FROM recipes r
            ORDER BY r.name
        """
        
        with self._connection() as conn:
            cursor = conn.execute(recipe_query)
            columns = [desc[0] for desc in cursor.description]
            recipe_rows = cursor.fetchall()
            
            # Build recipe dict keyed by dashless ID
            recipes_by_id = {}
            for row in recipe_rows:
                recipe = dict(zip(columns, row))
                db_id = recipe['id']  # Dashless ID
                
                # Convert to API format
                recipe['id'] = self._format_uuid(db_id)
                recipe['orgURL'] = recipe.pop('org_url', None)
                recipe['servings'] = recipe.pop('recipe_servings', None)
                recipe['cuisine'] = recipe.pop('recipeCuisine', None)
                update_at = recipe.pop('update_at', None)
                date_updated = recipe.pop('date_updated', None)
                recipe['updatedAt'] = update_at or date_updated
                recipe['lastMade'] = recipe.pop('last_made', None)
                recipe['yieldQuantity'] = recipe.pop('recipe_yield_quantity', None)
                
                # Initialize empty arrays
                recipe['recipeIngredient'] = []
                recipe['recipeInstructions'] = []
                recipe['tags'] = []
                recipe['recipeCategory'] = []
                
                recipes_by_id[db_id] = recipe
            
            logger.info(f"   Loaded {len(recipes_by_id)} recipe bases")
            
            # Step 2: Bulk fetch all ingredients
            ingredients_query = """
                SELECT 
                    ri.recipe_id, ri.position, ri.quantity, ri.note, ri.original_text,
                    ri.unit_id, ri.food_id,
                    iu.name AS unit_name, iu.abbreviation AS unit_abbr,
                    iu.plural_name AS unit_plural, iu.plural_abbreviation AS unit_plural_abbr,
                    if.name AS food_name, if.plural_name AS food_plural,
                    if.description AS food_description
                FROM recipes_ingredients ri
                LEFT JOIN ingredient_units iu ON ri.unit_id = iu.id
                LEFT JOIN ingredient_foods if ON ri.food_id = if.id
                ORDER BY ri.recipe_id, ri.position
            """
            cursor = conn.execute(ingredients_query)
            
            for row in cursor.fetchall():
                recipe_id = row[0]
                if recipe_id not in recipes_by_id:
                    continue
                
                ing = {
                    'quantity': row[2],
                    'note': row[3] or '',
                    'originalText': row[4] or '',
                    'display': row[4] or '',
                }
                
                if row[5]:  # unit_id
                    ing['unit'] = {
                        'id': self._format_uuid(row[5]),
                        'name': row[7] or '',
                        'abbreviation': row[8] or '',
                        'pluralName': row[9] or '',
                        'pluralAbbreviation': row[10] or '',
                    }
                
                if row[6]:  # food_id
                    ing['food'] = {
                        'id': self._format_uuid(row[6]),
                        'name': row[11] or '',
                        'pluralName': row[12] or '',
                        'description': row[13] or '',
                    }
                
                recipes_by_id[recipe_id]['recipeIngredient'].append(ing)
            
            logger.info("   Loaded ingredients")
            
            # Step 3: Bulk fetch all instructions
            instructions_query = """
                SELECT recipe_id, position, title, text, summary
                FROM recipe_instructions
                ORDER BY recipe_id, position
            """
            cursor = conn.execute(instructions_query)
            
            for row in cursor.fetchall():
                recipe_id = row[0]
                if recipe_id not in recipes_by_id:
                    continue
                
                instruction = {
                    'id': str(row[1]),
                    'title': row[2] or '',
                    'text': row[3] or '',
                    'summary': row[4] or '',
                }
                recipes_by_id[recipe_id]['recipeInstructions'].append(instruction)
            
            logger.info("   Loaded instructions")
            
            # Step 4: Bulk fetch all tags
            tags_query = """
                SELECT rtt.recipe_id, t.id, t.name, t.slug
                FROM recipes_to_tags rtt
                JOIN tags t ON rtt.tag_id = t.id
            """
            cursor = conn.execute(tags_query)
            
            for row in cursor.fetchall():
                recipe_id = row[0]
                if recipe_id not in recipes_by_id:
                    continue
                
                tag = {
                    'id': self._format_uuid(row[1]),
                    'name': row[2] or '',
                    'slug': row[3] or '',
                }
                recipes_by_id[recipe_id]['tags'].append(tag)
            
            logger.info("   Loaded tags")
            
            # Step 5: Bulk fetch all categories
            categories_query = """
                SELECT rtc.recipe_id, c.id, c.name, c.slug
                FROM recipes_to_categories rtc
                JOIN categories c ON rtc.category_id = c.id
            """
            cursor = conn.execute(categories_query)
            
            for row in cursor.fetchall():
                recipe_id = row[0]
                if recipe_id not in recipes_by_id:
                    continue
                
                category = {
                    'id': self._format_uuid(row[1]),
                    'name': row[2] or '',
                    'slug': row[3] or '',
                }
                recipes_by_id[recipe_id]['recipeCategory'].append(category)
            
            logger.info("   Loaded categories")
        
        result = list(recipes_by_id.values())
        logger.info(f"✅ Fetched {len(result)} complete recipes from DB")
        return result
    
    def get_all_recipe_urls(self) -> Set[str]:
        """
        Fetch all original URLs from recipes.
        
        Returns:
            Set of normalized URLs
        """
        from utils.url_utils import normalize_url
        
        query = "SELECT org_url FROM recipes WHERE org_url IS NOT NULL AND org_url != ''"
        
        with self._connection() as conn:
            cursor = conn.execute(query)
            rows = cursor.fetchall()
        
        return {normalize_url(row['org_url']) for row in rows}
    
    # -------------------------------------------------------------------------
    # Foods, Units, Tags operations
    # -------------------------------------------------------------------------
    
    def get_all_foods(self) -> List[Dict[str, Any]]:
        """Fetch all foods from the database."""
        # Query Mealie's actual schema
        query = """
            SELECT 
                id, name, plural_name, description,
                label_id, on_hand,
                created_at, update_at
            FROM ingredient_foods
            ORDER BY name
        """
        
        with self._connection() as conn:
            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        
        # Convert to list of dicts with API-compatible keys
        results = []
        for row in rows:
            food = dict(zip(columns, row))
            # Convert IDs from dashless DB format to dashed API format
            food['id'] = self._format_uuid(food.get('id'))
            # Map DB column names to API response format
            food['labelId'] = self._format_uuid(food.pop('label_id', None))
            food['pluralName'] = food.pop('plural_name', None)
            food['onHand'] = food.pop('on_hand', False)
            food['createdAt'] = food.pop('created_at', None)
            food['updatedAt'] = food.pop('update_at', None)
            results.append(food)
        
        return results
    
    def get_all_units(self) -> List[Dict[str, Any]]:
        """Fetch all units from the database."""
        # Query Mealie's actual schema
        query = """
            SELECT 
                id, name, plural_name, abbreviation, plural_abbreviation,
                description, fraction, use_abbreviation,
                created_at, update_at
            FROM ingredient_units
            ORDER BY name
        """
        
        with self._connection() as conn:
            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        
        # Convert to list of dicts with API-compatible keys
        results = []
        for row in rows:
            unit = dict(zip(columns, row))
            # Convert ID from dashless DB format to dashed API format
            unit['id'] = self._format_uuid(unit.get('id'))
            # Map DB column names to API response format
            unit['pluralName'] = unit.pop('plural_name', None)
            unit['pluralAbbreviation'] = unit.pop('plural_abbreviation', None)
            unit['useAbbreviation'] = unit.pop('use_abbreviation', False)
            unit['createdAt'] = unit.pop('created_at', None)
            unit['updatedAt'] = unit.pop('update_at', None)
            results.append(unit)
        
        return results
    
    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Fetch all tags from the database."""
        # Query Mealie's actual schema
        query = """
            SELECT id, name, slug, group_id
            FROM tags
            ORDER BY name
        """
        
        with self._connection() as conn:
            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        
        # Convert to list of dicts with API-compatible keys
        results = []
        for row in rows:
            tag = dict(zip(columns, row))
            # Convert IDs from dashless DB format to dashed API format
            tag['id'] = self._format_uuid(tag.get('id'))
            # Map DB column names to API response format
            tag['groupId'] = self._format_uuid(tag.pop('group_id', None))
            results.append(tag)
        
        return results
    
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Fetch all categories from the database."""
        # Query Mealie's actual schema
        query = """
            SELECT id, name, slug, group_id
            FROM categories
            ORDER BY name
        """
        
        with self._connection() as conn:
            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        
        # Convert to list of dicts with API-compatible keys
        results = []
        for row in rows:
            category = dict(zip(columns, row))
            # Convert IDs from dashless DB format to dashed API format
            category['id'] = self._format_uuid(category.get('id'))
            # Map DB column names to API response format
            category['groupId'] = self._format_uuid(category.pop('group_id', None))
            results.append(category)
        
        return results
    
    # -------------------------------------------------------------------------
    # Meal plan operations
    # -------------------------------------------------------------------------
    
    def get_meal_plans(
        self,
        start_date: Union[str, date],
        end_date: Union[str, date]
    ) -> List[Dict[str, Any]]:
        """
        Fetch meal plan entries for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD string or date object)
            end_date: End date (YYYY-MM-DD string or date object)
        
        Returns:
            List of meal plan entries
        """
        if isinstance(start_date, date):
            start_date = start_date.isoformat()
        if isinstance(end_date, date):
            end_date = end_date.isoformat()
        
        query = """
            SELECT 
                mp.id, mp.date, mp.entry_type AS entryType,
                mp.title, mp.text, mp.recipe_id AS recipeId,
                r.name AS recipeName, r.slug AS recipeSlug
            FROM meal_plans mp
            LEFT JOIN recipes r ON mp.recipe_id = r.id
            WHERE mp.date >= ? AND mp.date <= ?
            ORDER BY mp.date, mp.entry_type
        """
        
        with self._connection() as conn:
            cursor = conn.execute(query, (start_date, end_date))
            rows = cursor.fetchall()
        
        # Convert to list of dicts with formatted UUIDs
        results = []
        for row in rows:
            entry = self._row_to_dict(row)
            # Convert IDs from dashless DB format to dashed API format
            entry['id'] = self._format_uuid(entry.get('id'))
            entry['recipeId'] = self._format_uuid(entry.get('recipeId'))
            results.append(entry)
        
        return results
    
    # -------------------------------------------------------------------------
    # Shopping list operations (read-only)
    # -------------------------------------------------------------------------
    
    def check_recipes_parsed_status(self, slugs: List[str]) -> Dict[str, bool]:
        """
        Check if recipes have parsed ingredients using a single bulk query.
        
        A recipe is "parsed" if any of its ingredients has food_id NOT NULL.
        This is much faster than fetching full recipe data for each recipe.
        
        Args:
            slugs: List of recipe slugs to check
            
        Returns:
            Dict mapping slug to is_parsed boolean.
            Recipes not found in database are omitted from results.
        """
        if not slugs:
            return {}
        
        # SQLite has a parameter limit (typically 999), so batch large queries
        BATCH_SIZE = 500
        results = {}
        
        with self._connection() as conn:
            for i in range(0, len(slugs), BATCH_SIZE):
                batch = slugs[i:i + BATCH_SIZE]
                placeholders = ','.join('?' * len(batch))
                
                query = f"""
                    SELECT r.slug, 
                           MAX(CASE WHEN ri.food_id IS NOT NULL THEN 1 ELSE 0 END) as is_parsed
                    FROM recipes r
                    LEFT JOIN recipes_ingredients ri ON r.id = ri.recipe_id
                    WHERE r.slug IN ({placeholders})
                    GROUP BY r.slug
                """
                
                cursor = conn.execute(query, batch)
                for row in cursor.fetchall():
                    slug = row[0]
                    is_parsed = bool(row[1])
                    results[slug] = is_parsed
        
        return results
    
    def get_shopping_list(self, list_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a shopping list with its items.
        
        Args:
            list_id: Shopping list ID
        
        Returns:
            Shopping list data with 'listItems' array, or None if not found
        """
        # Fetch list metadata
        list_query = """
            SELECT id, name, created_at, update_at
            FROM shopping_lists
            WHERE id = ?
        """
        
        # Fetch list items
        items_query = """
            SELECT 
                id, display, quantity, checked, position, note,
                food_id AS foodId, unit_id AS unitId,
                shopping_list_id AS shoppingListId
            FROM shopping_list_items
            WHERE shopping_list_id = ?
            ORDER BY position, id
        """
        
        # Normalize input ID for DB query
        db_list_id = self._normalize_uuid_for_query(list_id)
        
        with self._connection() as conn:
            cursor = conn.execute(list_query, (db_list_id,))
            list_row = cursor.fetchone()
            
            if not list_row:
                return None
            
            cursor = conn.execute(items_query, (db_list_id,))
            item_rows = cursor.fetchall()
        
        result = self._row_to_dict(list_row)
        # Convert list ID from dashless DB format to dashed API format
        result['id'] = self._format_uuid(result.get('id'))
        
        # Convert item IDs
        items = []
        for row in item_rows:
            item = self._row_to_dict(row)
            item['id'] = self._format_uuid(item.get('id'))
            item['foodId'] = self._format_uuid(item.get('foodId'))
            item['unitId'] = self._format_uuid(item.get('unitId'))
            item['shoppingListId'] = self._format_uuid(item.get('shoppingListId'))
            items.append(item)
        result['listItems'] = items
        
        return result


# =============================================================================
# PUBLIC: MEALIE CLIENT
# =============================================================================

class MealieClient:
    """
    Unified client for all Mealie operations.
    
    This is the public facade that external code should use. It automatically
    selects between API and direct database modes based on configuration.
    
    Configuration:
        - `USER_CONFIG.get('mealie', {}).get('use_direct_db', False)` controls mode
        - DB path is hardcoded to /mealie-data/mealie.db
        - If use_direct_db=True but DB not accessible, raises MealieClientError
    
    Behavior:
        - Read operations: Use DB adapter if enabled and available, else API
        - Write operations: Always use API (Mealie must handle its own schema)
    
    Usage:
        client = MealieClient()
        
        # Reads (use DB if enabled)
        recipes = client.get_all_recipes()
        recipe = client.get_recipe("chicken-tikka-masala")
        
        # Writes (always API)
        client.create_recipe_from_url("https://example.com/recipe")
        client.update_recipe("my-recipe", {"servings": 4})
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        use_direct_db: Optional[bool] = None,
        db_path: Optional[str] = None,
    ):
        """
        Initialize the MealieClient.
        
        Args:
            base_url: Override Mealie URL (default: from config)
            token: Override Mealie token (default: from config)
            use_direct_db: Override DB mode (default: from USER_CONFIG)
            db_path: Override DB path (default: /mealie-data/mealie.db)
        
        Raises:
            MealieClientError: If use_direct_db=True but DB is not accessible
        """
        self._base_url = base_url or MEALIE_URL
        self._token = token or MEALIE_TOKEN
        
        # Determine mode from config if not overridden
        if use_direct_db is None:
            use_direct_db = USER_CONFIG.get('mealie', {}).get('use_direct_db', False)
        
        self._use_direct_db = use_direct_db
        self._db_path = db_path
        
        # Initialize API adapter (always needed for writes)
        self._api = _MealieAPIAdapter(self._base_url, self._token)
        
        # Initialize DB adapter if requested
        self._db: Optional[_MealieDBAdapter] = None
        if self._use_direct_db:
            try:
                self._db = _MealieDBAdapter(db_path)
                logger.info("MealieClient initialized in DB mode (reads from database)")
            except MealieDBError as e:
                # Fail clearly if DB mode requested but not available
                raise MealieClientError(
                    f"Direct DB mode requested but database not accessible: {e}",
                    operation="init",
                    details={'db_path': db_path or _MealieDBAdapter.DEFAULT_DB_PATH},
                )
        else:
            logger.info("MealieClient initialized in API mode")
    
    def close(self) -> None:
        """Clean up resources."""
        self._api.close()
    
    @property
    def mode(self) -> str:
        """Return current operating mode: 'db' or 'api'."""
        return 'db' if self._db else 'api'
    
    # =========================================================================
    # READ OPERATIONS (use DB if enabled, else API)
    # =========================================================================
    
    def get_all_recipes(self) -> List[Dict[str, Any]]:
        """
        Fetch all recipes (summary data only).
        
        Returns:
            List of recipe dictionaries with basic fields
        """
        if self._db:
            return self._db.get_all_recipes()
        return self._api.get_all_recipes()
    
    def get_all_recipes_full(self) -> List[Dict[str, Any]]:
        """
        Fetch ALL recipes with FULL data (ingredients, instructions, tags).
        
        DB MODE ONLY: This method requires direct database access.
        Falls back to individual API calls if not in DB mode.
        
        Use this for bulk operations like full database rebuilds - 
        it's MUCH faster than fetching recipes one at a time.
        
        Returns:
            List of complete recipe dictionaries
        
        Performance:
            - DB mode: ~5 bulk SQL queries for all data
            - API mode: N individual HTTP requests (slow)
        """
        if self._db:
            return self._db.get_all_recipes_full()
        
        # Fallback to API mode - fetch all recipes one by one
        logger.warning("get_all_recipes_full() called without DB mode - this will be slow")
        summaries = self._api.get_all_recipes()
        full_recipes = []
        for summary in summaries:
            slug = summary.get('slug')
            if slug:
                try:
                    full_recipes.append(self._api.get_recipe(slug))
                except Exception as e:
                    logger.warning(f"Failed to fetch full recipe {slug}: {e}")
        return full_recipes
    
    def get_recipe(self, slug: str) -> Dict[str, Any]:
        """
        Fetch a single recipe by identifier.

        The identifier may be either:
        - a recipe **slug** (e.g. "chicken-tikka-masala")
        - a recipe **UUID** (dashed or dashless)
        
        Args:
            slug: Recipe slug or UUID
        
        Returns:
            Recipe dictionary
        
        Raises:
            MealieClientError: If recipe not found
        """
        if _is_uuid_identifier(slug):
            # Route UUIDs to ID lookup (fixes DB-mode failures when callers pass recipeId).
            if self._db:
                result = self._db.get_recipe_by_id(slug)
                if result is None:
                    raise MealieClientError(f"Recipe not found: {slug}", operation="get_recipe")
                return result
            return self._api.get_recipe_by_id(slug)

        # Otherwise treat as slug
        if self._db:
            result = self._db.get_recipe_by_slug(slug)
            if result is None:
                raise MealieClientError(f"Recipe not found: {slug}", operation="get_recipe")
            return result
        return self._api.get_recipe(slug)
    
    def get_recipe_by_id(self, recipe_id: str) -> Dict[str, Any]:
        """
        Fetch a single recipe by ID.
        
        Args:
            recipe_id: Recipe ID (UUID)
        
        Returns:
            Recipe dictionary
        
        Raises:
            MealieClientError: If recipe not found
        """
        if self._db:
            result = self._db.get_recipe_by_id(recipe_id)
            if result is None:
                raise MealieClientError(f"Recipe not found: {recipe_id}", operation="get_recipe_by_id")
            return result
        return self._api.get_recipe_by_id(recipe_id)
    
    def search_recipes(self, query: str, per_page: int = 50) -> List[Dict[str, Any]]:
        """
        Search recipes by keyword/name.
        
        Uses Mealie's built-in search API which does text matching against recipe names.
        
        Args:
            query: Search term (e.g., "garlic bread", "miso soup")
            per_page: Number of results to return (default 50)
        
        Returns:
            List of recipe dicts with id, name, tags, etc.
        """
        # Search is only available via API (not DB)
        return self._api.search_recipes(query, per_page)
    
    def get_all_recipe_urls(self) -> Set[str]:
        """
        Fetch all original URLs from recipes for duplicate checking.
        
        Returns:
            Set of normalized URLs
        """
        if self._db:
            return self._db.get_all_recipe_urls()
        return self._api.get_all_recipe_urls()
    
    def get_recipes_batch(self, slugs: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple recipes by slug.
        
        In DB mode, fetches complete recipe data directly from database.
        In API mode, uses parallel HTTP requests.
        
        Args:
            slugs: List of recipe slugs to fetch
        
        Returns:
            Dict mapping slug to full recipe data (missing slugs are omitted)
        """
        if self._db:
            # DB mode - fetch with full data, no HTTP overhead
            return self._db.get_recipes_full_batch(slugs)
        # API mode - parallel HTTP requests
        return self._api.get_recipes_batch(slugs)
    
    def check_duplicate_urls(self, urls: List[str]) -> Dict[str, bool]:
        """
        Check which URLs already exist in Mealie.
        
        Args:
            urls: List of URLs to check
        
        Returns:
            Dict mapping URL to exists boolean
        """
        # Uses get_all_recipe_urls internally which respects DB mode
        from utils.url_utils import normalize_url
        
        existing_urls = self.get_all_recipe_urls()
        
        results = {}
        for url in urls:
            results[url] = normalize_url(url) in existing_urls
        
        return results
    
    def get_all_foods(self) -> List[Dict[str, Any]]:
        """
        Fetch all foods (ingredients database).
        
        Returns:
            List of food dictionaries
        """
        if self._db:
            return self._db.get_all_foods()
        return self._api.get_all_foods()
    
    def get_all_units(self) -> List[Dict[str, Any]]:
        """
        Fetch all units.
        
        Returns:
            List of unit dictionaries
        """
        if self._db:
            return self._db.get_all_units()
        return self._api.get_all_units()
    
    def get_all_tags(self) -> List[Dict[str, Any]]:
        """
        Fetch all tags.
        
        Returns:
            List of tag dictionaries
        """
        if self._db:
            return self._db.get_all_tags()
        return self._api.get_all_tags()
    
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """
        Fetch all categories.
        
        Returns:
            List of category dictionaries
        """
        if self._db:
            return self._db.get_all_categories()
        return self._api.get_all_categories()
    
    def get_meal_plans(
        self,
        start_date: Union[str, date],
        end_date: Union[str, date]
    ) -> List[Dict[str, Any]]:
        """
        Fetch meal plan entries for a date range.
        
        Note: Always uses API - meal plans are not part of DB mode's
        "recipe data" scope and Mealie's schema may differ.
        
        Args:
            start_date: Start date (YYYY-MM-DD string or date object)
            end_date: End date (YYYY-MM-DD string or date object)
        
        Returns:
            List of meal plan entries
        """
        # Always use API for meal plan operations (consistent with create/delete)
        return self._api.get_meal_plans(start_date, end_date)
    
    def get_all_shopping_lists(self) -> List[Dict[str, Any]]:
        """
        Fetch all shopping lists.
        
        Returns:
            List of shopping list dictionaries
        """
        # Shopping lists are only available via API (not DB)
        return self._api.get_all_shopping_lists()
    
    def get_shopping_list(self, list_id: str) -> Dict[str, Any]:
        """
        Fetch a shopping list by ID.
        
        Args:
            list_id: Shopping list ID
        
        Returns:
            Shopping list data with 'listItems' array
        
        Raises:
            MealieClientError: If shopping list not found
        """
        # Shopping list schemas have varied across Mealie versions; DB-mode read adapters
        # are intentionally limited to *recipe* data for stability. Always use the API here.
        return self._api.get_shopping_list(list_id)
    
    def check_recipes_parsed_status(self, slugs: List[str]) -> Dict[str, bool]:
        """
        Check parsing status for multiple recipes.
        
        In DB mode, uses a single bulk SQL query to check if any ingredient
        in each recipe has food_id NOT NULL (indicating parsed ingredients).
        
        In API mode, returns an empty dict - callers should use the local
        index or fetch full recipe data instead.
        
        Args:
            slugs: List of recipe slugs to check
            
        Returns:
            Dict mapping slug to is_parsed boolean.
            In API mode, always returns empty dict.
        """
        if self._db:
            return self._db.check_recipes_parsed_status(slugs)
        # API mode: return empty dict (caller should use local index instead)
        return {}
    
    # =========================================================================
    # WRITE OPERATIONS (always API)
    # =========================================================================
    
    def create_recipe_from_url(self, url: str, include_tags: bool = False) -> Dict[str, Any]:
        """
        Create a recipe from a URL using Mealie's scraper.
        
        Args:
            url: Recipe URL to import
            include_tags: Whether to include tags from source
        
        Returns:
            Created recipe data
        """
        return self._api.create_recipe_from_url(url, include_tags)
    
    def scrape_recipe_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape recipe data from URL WITHOUT creating the recipe.
        
        Use this for pre-import collision detection:
        1. Scrape to get recipe name
        2. Check if name exists
        3. If collision, create with modified name
        
        Args:
            url: Recipe URL to scrape
        
        Returns:
            Scraped recipe data (JSON-LD format from source site)
        """
        return self._api.scrape_recipe_url(url)
    
    def create_recipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new recipe manually.
        
        Args:
            data: Recipe data (must contain 'name')
        
        Returns:
            Created recipe data
        """
        return self._api.create_recipe(data)
    
    def update_recipe(self, slug: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing recipe.
        
        Args:
            slug: Recipe slug (or ID)
            data: Fields to update
        
        Returns:
            Updated recipe data
        """
        return self._api.update_recipe(slug, data)
    
    def upload_recipe_image(self, slug: str, image_url: str) -> bool:
        """
        Upload an image to a recipe by URL.
        
        Mealie will download and store the image automatically.
        
        Args:
            slug: Recipe slug (or ID)
            image_url: URL of the image to upload
        
        Returns:
            True on success, False on failure
        """
        return self._api.upload_recipe_image(slug, image_url)
    
    def delete_recipe(self, slug: str) -> bool:
        """
        Delete a recipe.
        
        Args:
            slug: Recipe slug (or ID)
        
        Returns:
            True on success
        """
        return self._api.delete_recipe(slug)
    
    def create_meal_plan_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a meal plan entry.
        
        Args:
            data: Meal plan entry data (date, entryType, recipeId, etc.)
        
        Returns:
            Created meal plan entry
        """
        return self._api.create_meal_plan_entry(data)
    
    def delete_meal_plan_entry(self, entry_id: str) -> bool:
        """
        Delete a meal plan entry.
        
        Args:
            entry_id: Meal plan entry ID
        
        Returns:
            True on success
        """
        return self._api.delete_meal_plan_entry(entry_id)
    
    def create_shopping_list(self, name: str) -> Dict[str, Any]:
        """
        Create a new shopping list.
        
        Args:
            name: Shopping list name
        
        Returns:
            Created shopping list data
        """
        return self._api.create_shopping_list(name)
    
    def delete_shopping_list(self, list_id: str) -> bool:
        """
        Delete a shopping list.
        
        Args:
            list_id: Shopping list ID
        
        Returns:
            True on success
        """
        return self._api.delete_shopping_list(list_id)
    
    def add_recipes_to_shopping_list(self, list_id: str, recipes: List[Dict[str, Any]]) -> bool:
        """
        Add multiple recipes to a shopping list with scaling.
        
        Args:
            list_id: Shopping list ID
            recipes: List of dicts with 'recipeId' and 'recipeIncrementQuantity'
        
        Returns:
            True on success
        """
        return self._api.add_recipes_to_shopping_list(list_id, recipes)
    
    def add_shopping_item(self, list_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a single item to a shopping list.
        
        Args:
            list_id: Shopping list ID
            item: Item data (display, quantity, unit, food, note, etc.)
        
        Returns:
            Created shopping list item
        """
        return self._api.add_shopping_item(list_id, item)
    
    def add_shopping_items(
        self,
        list_id: str,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add multiple items to a shopping list SEQUENTIALLY.
        
        CRITICAL: Mealie corrupts shopping lists with parallel writes.
        This method adds items one-by-one to prevent data corruption.
        
        Args:
            list_id: Shopping list ID
            items: List of item data dicts
        
        Returns:
            List of created items
        """
        return self._api.add_shopping_items(list_id, items)
    
    def delete_shopping_items(self, item_ids: List[str]) -> bool:
        """
        Delete multiple shopping list items.
        
        Args:
            item_ids: List of shopping list item IDs
        
        Returns:
            True on success
        """
        return self._api.delete_shopping_items(item_ids)
    
    def create_food(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Create a new food entry.
        
        Args:
            name: Food name
            **kwargs: Additional fields (description, labelId, etc.)
        
        Returns:
            Created food data
        """
        return self._api.create_food(name, **kwargs)
    
    def create_unit(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Create a new unit.
        
        Args:
            name: Unit name
            **kwargs: Additional fields (abbreviation, description, etc.)
        
        Returns:
            Created unit data
        """
        return self._api.create_unit(name, **kwargs)
    
    def create_tag(self, name: str) -> Dict[str, Any]:
        """
        Create a new tag.
        
        Args:
            name: Tag name
        
        Returns:
            Created tag data
        """
        return self._api.create_tag(name)
    
    def get_food(self, food_id: str) -> Dict[str, Any]:
        """
        Fetch a single food by ID.
        
        Args:
            food_id: Food ID
        
        Returns:
            Food dictionary
        """
        return self._api.get_food(food_id)
    
    def update_food(self, food_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing food.
        
        Args:
            food_id: Food ID
            data: Fields to update (must include all required fields)
        
        Returns:
            Updated food data
        """
        return self._api.update_food(food_id, data)
    
    def get_all_labels(self) -> List[Dict[str, Any]]:
        """
        Fetch all labels.
        
        Returns:
            List of label dictionaries
        """
        return self._api.get_all_labels()
    
    def create_label(self, name: str, color: str = '#808080') -> Dict[str, Any]:
        """
        Create a new label.
        
        Args:
            name: Label name
            color: Label color (hex code, default: '#808080')
        
        Returns:
            Created label data
        """
        return self._api.create_label(name, color)


# =============================================================================
# MODULE SELF-TEST
# =============================================================================

if __name__ == "__main__":
    """
    Run basic connectivity test.
    
    Usage: python mealie_client.py
    """
    import sys
    
    print("=" * 60)
    print("MealieClient Self-Test")
    print("=" * 60)
    
    try:
        client = MealieClient()
        print(f"Mode: {client.mode}")
        
        # Test recipe fetch
        print("\nFetching recipes...")
        recipes = client.get_all_recipes()
        print(f"Found {len(recipes)} recipes")
        
        if recipes:
            sample = recipes[0]
            print(f"Sample recipe: {sample.get('name', 'Unknown')} ({sample.get('slug', 'no-slug')})")
        
        # Test foods fetch
        print("\nFetching foods...")
        foods = client.get_all_foods()
        print(f"Found {len(foods)} foods")
        
        # Test units fetch
        print("\nFetching units...")
        units = client.get_all_units()
        print(f"Found {len(units)} units")
        
        # Test tags fetch
        print("\nFetching tags...")
        tags = client.get_all_tags()
        print(f"Found {len(tags)} tags")
        
        client.close()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        sys.exit(0)
        
    except MealieClientError as e:
        print(f"\n❌ MealieClient error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
