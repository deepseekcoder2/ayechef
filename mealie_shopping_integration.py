#!/usr/bin/env python3
"""
Mealie Shopping List Integration Module
========================================

CRUD operations for Mealie shopping list items with connection pooling support.
Handles bulk operations for shopping list refinement and management.

Features:
- Bulk delete shopping list items by IDs
- Add individual shopping list items
- Extract ingredients from Mealie shopping list format
- Connection pooling for performance
- Comprehensive error handling

API Endpoints Used:
- DELETE /api/households/shopping/items?ids={item_ids}
- POST /api/households/shopping/items
- GET /api/households/shopping/lists/{list_id} (for extraction)

Usage:
    from mealie_shopping_integration import (
        delete_mealie_shopping_items,
        add_mealie_shopping_item,
        extract_ingredients_from_mealie_list
    )

    # Delete multiple items
    success = delete_mealie_shopping_items(["item-id-1", "item-id-2"])

    # Add new item
    success = add_mealie_shopping_item("list-id", item_data)

    # Extract ingredients from list
    ingredients = extract_ingredients_from_mealie_list(shopping_list)
"""

import asyncio
from typing import List, Dict, Any, Optional
from mealie_client import MealieClient, MealieClientError
from config import MEALIE_URL, get_mealie_headers
from tools.logging_utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


def delete_mealie_shopping_items(item_ids: List[str]) -> bool:
    """
    DELETE /api/households/shopping/items?ids={item_ids}

    Bulk delete shopping list items by their IDs.

    Args:
        item_ids: List of shopping list item UUIDs to delete

    Returns:
        bool: True if successful, False otherwise

    Raises:
        ValueError: If item_ids is empty or contains invalid UUIDs
        MealieClientError: For network/API errors
    """
    if not item_ids:
        logger.warning("No item IDs provided for deletion")
        return True

    # Validate UUID format (basic check)
    for item_id in item_ids:
        if not isinstance(item_id, str) or len(item_id) != 36:
            raise ValueError(f"Invalid item ID format: {item_id}")

    client = MealieClient()
    try:
        logger.info(f"Deleting {len(item_ids)} shopping list items")
        result = client.delete_shopping_items(item_ids)
        logger.info("Successfully deleted shopping list items")
        return result
    except MealieClientError as e:
        logger.error(f"Error deleting shopping list items: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting shopping list items: {e}")
        return False
    finally:
        client.close()


def add_mealie_shopping_item(list_id: str, item_data: dict) -> bool:
    """
    POST /api/households/shopping/items

    Add a new item to a Mealie shopping list.
    
    NOTE: Mealie has a bulk endpoint (/api/households/shopping/items/create-bulk)
    but it automatically aggregates/merges similar items, which breaks our use case
    where we need each refined LLM item to remain separate. Single POST per item
    is the correct approach for granular control.

    Args:
        list_id: The shopping list ID (though API doesn't require it in URL, kept for consistency)
        item_data: Shopping list item data in Mealie format:
            {
                "display": "string",          # Required: Display text
                "quantity": 1,                # Optional: Default 1
                "unit": {"name": "string"},   # Optional: Unit object
                "food": {"name": "string"},   # Optional: Food object
                "note": "string",             # Optional: Note text
                "checked": false,             # Optional: Default false
                "position": 100               # Optional: Position in list
            }

    Returns:
        bool: True if successful, False otherwise

    Raises:
        ValueError: If required fields are missing or invalid
        MealieClientError: For network/API errors
    """
    if not item_data or not isinstance(item_data, dict):
        raise ValueError("Item data must be a non-empty dictionary")

    if "display" not in item_data or not item_data["display"].strip():
        raise ValueError("Item data must contain non-empty 'display' field")

    # Prepare item payload matching MealieClient.add_shopping_item format
    item_payload = {
        "display": item_data["display"].strip(),
        "quantity": item_data.get("quantity") or 1,  # Handle null quantities
        "unit": item_data.get("unit"),
        "food": item_data.get("food"),
        "note": item_data.get("note", ""),
        "checked": item_data.get("checked", False),
        "position": item_data.get("position", 100)
    }
    
    # Add foodId and unitId if provided (critical for preserving display text)
    if item_data.get("foodId"):
        item_payload["foodId"] = item_data["foodId"]
    if item_data.get("unitId"):
        item_payload["unitId"] = item_data["unitId"]

    # Remove None values to avoid sending null to API
    item_payload = {k: v for k, v in item_payload.items() if v is not None}

    client = MealieClient()
    try:
        logger.info(f"Adding shopping list item: {item_payload['display']}")
        client.add_shopping_item(list_id, item_payload)
        logger.info("Successfully added shopping list item")
        return True
    except MealieClientError as e:
        logger.error(f"Error adding shopping list item: {e}")
        logger.error(f"Payload: {item_payload}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error adding shopping list item: {e}")
        return False
    finally:
        client.close()


def extract_ingredients_from_mealie_list(shopping_list: dict) -> List[str]:
    """
    Extract display strings from Mealie shopping list's listItems array.

    Args:
        shopping_list: Mealie shopping list response containing 'listItems' array

    Returns:
        List[str]: List of ingredient display strings

    Raises:
        ValueError: If shopping_list format is invalid
    """
    if not shopping_list or not isinstance(shopping_list, dict):
        raise ValueError("Shopping list must be a non-empty dictionary")

    if "listItems" not in shopping_list:
        logger.warning("Shopping list does not contain 'listItems' array")
        return []

    list_items = shopping_list["listItems"]
    if not isinstance(list_items, list):
        raise ValueError("'listItems' must be a list")

    ingredients = []
    for item in list_items:
        if isinstance(item, dict) and "display" in item:
            display_text = item["display"].strip()
            if display_text:  # Only add non-empty display texts
                ingredients.append(display_text)
        else:
            logger.warning(f"Invalid list item format: {item}")

    logger.info(f"Extracted {len(ingredients)} ingredients from shopping list")
    return ingredients


def extract_ingredients_with_food_ids(shopping_list: dict) -> List[dict]:
    """
    Extract ingredients with their food/unit references from Mealie shopping list.
    
    This preserves foodId, unitId, and full food/unit objects so that when we 
    write refined items back to Mealie, the display text is preserved instead
    of being regenerated.

    Args:
        shopping_list: Mealie shopping list response containing 'listItems' array

    Returns:
        List[dict]: List of items with display, foodId, unitId, food, unit, etc.
    """
    if not shopping_list or not isinstance(shopping_list, dict):
        raise ValueError("Shopping list must be a non-empty dictionary")

    if "listItems" not in shopping_list:
        logger.warning("Shopping list does not contain 'listItems' array")
        return []

    list_items = shopping_list["listItems"]
    if not isinstance(list_items, list):
        raise ValueError("'listItems' must be a list")

    ingredients = []
    for item in list_items:
        if isinstance(item, dict) and "display" in item:
            display_text = item["display"].strip()
            if display_text:
                food_obj = item.get("food")
                unit_obj = item.get("unit")
                
                ingredients.append({
                    "display": display_text,
                    "quantity": item.get("quantity") or 1,  # Handle null quantities
                    "foodId": item.get("foodId"),
                    "unitId": item.get("unitId"),
                    "food": food_obj,  # Full food object with id
                    "unit": unit_obj,  # Full unit object with id
                    "food_name": food_obj.get("name") if isinstance(food_obj, dict) else None,
                    "unit_name": unit_obj.get("name") if isinstance(unit_obj, dict) else None,
                    "note": item.get("note", "")
                })
        else:
            logger.warning(f"Invalid list item format: {item}")

    logger.info(f"Extracted {len(ingredients)} ingredients with food/unit refs from shopping list")
    return ingredients


def build_food_unit_lookup(ingredients_with_ids: List[dict]) -> dict:
    """
    Build a lookup table from food_name to full food/unit data.
    
    When the LLM returns aggregated items with food.name, we use this lookup
    to find the original foodId, unitId, and full objects from Mealie.
    
    Args:
        ingredients_with_ids: List from extract_ingredients_with_food_ids()
    
    Returns:
        dict: Mapping of lowercase food_name to {'foodId', 'unitId', 'food', 'unit'}
    """
    lookup = {}
    for item in ingredients_with_ids:
        food_name = item.get("food_name")
        if food_name:
            key = food_name.lower()
            # Only add if not already present (first occurrence wins)
            if key not in lookup:
                lookup[key] = {
                    "foodId": item.get("foodId"),
                    "unitId": item.get("unitId"),
                    "food": item.get("food"),
                    "unit": item.get("unit")
                }
    
    logger.info(f"Built food/unit lookup with {len(lookup)} entries")
    return lookup


def build_food_id_lookup(ingredients_with_ids: List[dict]) -> dict:
    """
    Build a lookup table from food_name to foodId.
    
    DEPRECATED: Use build_food_unit_lookup() instead for full object support.
    
    Args:
        ingredients_with_ids: List from extract_ingredients_with_food_ids()
    
    Returns:
        dict: Mapping of lowercase food_name to foodId
    """
    lookup = {}
    for item in ingredients_with_ids:
        food_name = item.get("food_name")
        food_id = item.get("foodId")
        if food_name and food_id:
            lookup[food_name.lower()] = food_id
    
    logger.info(f"Built food ID lookup with {len(lookup)} entries")
    return lookup


def fetch_mealie_shopping_list(list_id: str) -> Optional[dict]:
    """
    GET /api/households/shopping/lists/{list_id}

    Fetch a complete Mealie shopping list by ID.

    Args:
        list_id: The shopping list UUID

    Returns:
        dict: Complete shopping list data or None if failed
    """
    if not list_id or not isinstance(list_id, str):
        raise ValueError("List ID must be a non-empty string")

    client = MealieClient()
    try:
        logger.info(f"Fetching shopping list: {list_id}")
        shopping_list = client.get_shopping_list(list_id)
        logger.info(f"Successfully fetched shopping list with {len(shopping_list.get('listItems', []))} items")
        return shopping_list
    except MealieClientError as e:
        logger.error(f"Error fetching shopping list: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching shopping list: {e}")
        return None
    finally:
        client.close()


async def add_multiple_mealie_shopping_items_async(list_id: str, items_data: List[dict]) -> tuple[int, int]:
    """
    Async version of bulk shopping item addition using aiohttp for true parallelism.

    This provides maximum performance by avoiding thread pool overhead and enabling
    true concurrent HTTP requests when needed.

    Args:
        list_id: The shopping list ID
        items_data: List of item data dictionaries

    Returns:
        tuple: (successful_count, failed_count)
    """
    if not items_data:
        logger.warning("No items provided for async bulk addition")
        return 0, 0

    # Use async bulk API implementation
    return await add_multiple_mealie_shopping_items_async_bulk(list_id, items_data)


async def add_multiple_mealie_shopping_items_async_bulk(list_id: str, items_data: List[dict]) -> tuple[int, int]:
    """
    Async implementation of bulk shopping item addition using aiohttp.

    Provides true parallelism for maximum performance in production environments.
    """
    if not items_data:
        logger.warning("No items provided for async bulk addition")
        return 0, 0

    logger.info(f"Async bulk adding {len(items_data)} shopping list items")

    try:
        import aiohttp
    except ImportError:
        logger.warning("aiohttp not available, falling back to sync bulk implementation")
        return add_multiple_mealie_shopping_items_bulk(list_id, items_data)

    url = f"{MEALIE_URL}/api/households/shopping/items/create-bulk"
    headers = get_mealie_headers()

    # Prepare bulk payload - same format as sync version
    bulk_payload = []
    for item_data in items_data:
        try:
            # Validate required fields
            if not item_data or not isinstance(item_data, dict):
                logger.error("Invalid item data: must be a non-empty dictionary")
                continue

            if "display" not in item_data or not item_data["display"].strip():
                logger.error("Invalid item data: missing or empty 'display' field")
                continue

            # Transform to bulk API format
            bulk_item = {
                "shoppingListId": list_id,
                "display": item_data["display"].strip(),
                "quantity": item_data.get("quantity") or 1,  # Handle null quantities
                "note": item_data.get("note", ""),
                "checked": item_data.get("checked", False),
                "position": item_data.get("position", 0)
            }

            # Add optional fields only if they exist and are not None
            if item_data.get("unit") is not None:
                bulk_item["unit"] = item_data["unit"]
            if item_data.get("food") is not None:
                bulk_item["food"] = item_data["food"]

            bulk_payload.append(bulk_item)

        except Exception as e:
            logger.error(f"Error preparing item {item_data.get('display', 'unknown')} for async bulk operation: {e}")
            continue

    if not bulk_payload:
        logger.error("No valid items to add after validation")
        return 0, len(items_data)

    # Configure connection pooling for optimal performance
    connector = aiohttp.TCPConnector(
        limit=20,  # Connection pool size
        limit_per_host=10,  # Per host limit
        ttl_dns_cache=300,  # DNS cache TTL
        use_dns_cache=True
    )

    timeout = aiohttp.ClientTimeout(
        total=30,  # Total timeout
        connect=10,  # Connection timeout
        sock_read=20  # Socket read timeout
    )

    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        ) as session:

            logger.info(f"Sending async bulk request with {len(bulk_payload)} items")
            async with session.post(url, json=bulk_payload) as response:
                if response.status == 201:
                    # Parse response to determine success/failure counts
                    response_data = await response.json()
                    logger.info(f"Async bulk API response received: {len(response_data) if isinstance(response_data, list) else 'non-list response'}")

                    # The bulk API returns a list of created items
                    if isinstance(response_data, list):
                        successful = len(response_data)
                        failed = len(bulk_payload) - successful
                    else:
                        # Fallback: assume all successful if we get a non-list response
                        successful = len(bulk_payload)
                        failed = 0

                    logger.info(f"Async bulk addition complete: {successful} successful, {failed} failed")
                    return successful, failed

                else:
                    error_text = await response.text()
                    logger.error(f"Async bulk API failed with status {response.status}: {error_text}")
                    # Fallback to sync implementation
                    logger.warning("Async bulk API failed, falling back to sync implementation")
                    return add_multiple_mealie_shopping_items_bulk(list_id, items_data)

    except asyncio.TimeoutError:
        logger.error("Async bulk request timed out")
        return add_multiple_mealie_shopping_items_bulk(list_id, items_data)
    except aiohttp.ClientError as e:
        logger.error(f"Async bulk request failed with network error: {e}")
        return add_multiple_mealie_shopping_items_bulk(list_id, items_data)
    except Exception as e:
        logger.error(f"Unexpected error in async bulk operation: {e}")
        return add_multiple_mealie_shopping_items_bulk(list_id, items_data)


def add_multiple_mealie_shopping_items(list_id: str, items_data: List[dict]) -> tuple[int, int]:
    """
    Add multiple shopping list items using bulk API operations for maximum performance.

    Uses the /api/households/shopping/items/create-bulk endpoint for parallel processing.

    Args:
        list_id: The shopping list ID
        items_data: List of item data dictionaries

    Returns:
        tuple: (successful_count, failed_count)
    """
    if not items_data:
        logger.warning("No items provided for bulk addition")
        return 0, 0

    # Use bulk API for better performance (parallel processing)
    return add_multiple_mealie_shopping_items_bulk(list_id, items_data)


def add_multiple_mealie_shopping_items_bulk(list_id: str, items_data: List[dict]) -> tuple[int, int]:
    """
    Add multiple shopping list items using MealieClient.add_shopping_items.

    MealieClient.add_shopping_items handles sequential writes internally to prevent
    data corruption. This method validates items and uses the client for addition.

    Args:
        list_id: The shopping list ID
        items_data: List of item data dictionaries

    Returns:
        tuple: (successful_count, failed_count)
    """
    if not items_data:
        logger.warning("No items provided for bulk addition")
        return 0, 0

    logger.info(f"Adding {len(items_data)} shopping list items using MealieClient")

    # Prepare and validate items
    validated_items = []
    for item_data in items_data:
        try:
            # Validate required fields
            if not item_data or not isinstance(item_data, dict):
                logger.error("Invalid item data: must be a non-empty dictionary")
                continue

            if "display" not in item_data or not item_data["display"].strip():
                logger.error("Invalid item data: missing or empty 'display' field")
                continue

            # Transform to MealieClient format
            item = {
                "display": item_data["display"].strip(),
                "quantity": item_data.get("quantity") or 1,  # Handle null quantities
                "note": item_data.get("note", ""),
                "checked": item_data.get("checked", False),
                "position": item_data.get("position", 0)
            }

            # Add optional fields only if they exist and are not None
            if item_data.get("unit") is not None:
                item["unit"] = item_data["unit"]
            if item_data.get("food") is not None:
                item["food"] = item_data["food"]
            if item_data.get("foodId"):
                item["foodId"] = item_data["foodId"]
            if item_data.get("unitId"):
                item["unitId"] = item_data["unitId"]

            # Remove None values
            item = {k: v for k, v in item.items() if v is not None}
            validated_items.append(item)

        except Exception as e:
            logger.error(f"Error preparing item {item_data.get('display', 'unknown')} for bulk operation: {e}")
            continue

    if not validated_items:
        logger.error("No valid items to add after validation")
        return 0, len(items_data)

    client = MealieClient()
    try:
        logger.info(f"Adding {len(validated_items)} validated items")
        results = client.add_shopping_items(list_id, validated_items)
        
        # MealieClient.add_shopping_items returns list of created items
        successful = len(results)
        failed = len(validated_items) - successful

        logger.info(f"Bulk addition complete: {successful} successful, {failed} failed")
        return successful, failed
    except MealieClientError as e:
        logger.error(f"Error adding shopping items: {e}")
        # Fallback to individual API calls if bulk fails
        logger.warning("Bulk operation failed, falling back to individual API calls")
        return _add_multiple_mealie_shopping_items_fallback(list_id, items_data)
    except Exception as e:
        logger.error(f"Unexpected error in bulk operation: {e}")
        # Fallback to individual API calls
        logger.warning("Bulk operation failed, falling back to individual API calls")
        return _add_multiple_mealie_shopping_items_fallback(list_id, items_data)
    finally:
        client.close()


def _add_multiple_mealie_shopping_items_fallback(list_id: str, items_data: List[dict]) -> tuple[int, int]:
    """
    Fallback method using individual API calls when bulk operations fail.

    This maintains backward compatibility and reliability while bulk operations
    are being optimized.

    Args:
        list_id: The shopping list ID
        items_data: List of item data dictionaries

    Returns:
        tuple: (successful_count, failed_count)
    """
    logger.info(f"Using fallback: Adding {len(items_data)} shopping list items individually")

    successful = 0
    failed = 0

    for item_data in items_data:
        try:
            if add_mealie_shopping_item(list_id, item_data):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error adding item {item_data.get('display', 'unknown')}: {e}")
            failed += 1

    logger.info(f"Fallback addition complete: {successful} successful, {failed} failed")
    return successful, failed
