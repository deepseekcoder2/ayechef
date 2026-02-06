#!/usr/bin/env python3
"""
Direct Pipeline Executor
========================

Production-ready pipeline execution engine that eliminates subprocess isolation fragility.

FEATURES:
- ✅ Direct function calls with proper async/await chains
- ✅ Comprehensive error recovery with retry logic and exponential backoff
- ✅ Timeout handling for all operations
- ✅ Centralized configuration management
- ✅ Structured logging and monitoring
- ✅ Input validation and graceful degradation
- ✅ Integration with centralized config system

OPERATIONS:
- Ingredient parsing: Batch processing of unparsed recipes with quality validation
- Shopping list refinement: LLM-powered pantry filtering and categorization

ARCHITECTURAL ROLE:
This component is the "subprocess-free execution engine" that replaced fragile
subprocess-based execution with robust direct function calls, as documented
in the system README and PRD.
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
from config import get_pipeline_config, get_config_value
from tools.logging_utils import get_logger

logger = get_logger(__name__)


class OperationStatus(Enum):
    """Status constants for operations."""
    SUCCESS = "success"
    FAILURE = "failure"
    DEGRADED = "degraded"
    PENDING = "pending"
    RUNNING = "running"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    enable_validation: bool = True
    max_llm_retries: int = 3
    max_network_retries: int = 2
    ingredient_parsing_timeout: int = 900
    shopping_refinement_timeout: int = 300


@dataclass
class OperationResult:
    """Result of a pipeline operation."""
    status: OperationStatus
    data: Any = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    degraded_mode: bool = False


class PipelineError(Enum):
    """Error types for pipeline operations."""
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    LLM_ERROR = "llm_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


class DirectPipelineExecutor:
    """
    Production-ready pipeline execution engine for Aye Chef.

    This class implements the core "Direct Pipeline Executor" pattern that replaces
    subprocess-based execution with direct function calls, eliminating the fragility
    and serialization issues that plagued the original system.

    Key Features:
    - Async execution with proper timeout and retry logic
    - Centralized configuration management
    - Comprehensive error classification and recovery
    - Structured logging and monitoring integration
    - Input validation and graceful degradation
    - Support for dry-run operations for testing

    Operations Supported:
    - Ingredient parsing: Process unparsed recipes from Mealie database
    - Shopping list refinement: Apply LLM-powered filtering and categorization

    Configuration:
    Uses centralized config from config.py with fallback to local defaults.
    Supports timeout configuration, retry limits, and validation settings.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the executor with configuration."""
        # Use centralized config if available, otherwise fall back to local config
        if config is None:
            try:
                centralized_config = get_pipeline_config()
                self.config = PipelineConfig(
                    enable_validation=centralized_config.get("features", {}).get("enable_validation", True),
                    max_llm_retries=centralized_config.get("retries", {}).get("max_llm_retries", 3),
                    max_network_retries=centralized_config.get("retries", {}).get("max_network_retries", 2),
                    ingredient_parsing_timeout=centralized_config.get("timeouts", {}).get("ingredient_parsing", 900),
                    shopping_refinement_timeout=centralized_config.get("timeouts", {}).get("shopping_refinement", 300)
                )
            except Exception as e:
                logger.warning(f"Failed to load centralized config, using defaults: {e}")
                self.config = PipelineConfig()
        else:
            self.config = config

        self.monitoring = PipelineMonitoring()

    async def _execute_with_timeout_and_retry(
        self,
        operation: Callable[[], Awaitable[Any]],
        operation_name: str,
        timeout: float,
        max_retries: int,
        retry_delay: float = 1.0
    ) -> tuple[Any, int]:
        """
        Execute an operation with timeout and retry logic.

        Args:
            operation: Async callable to execute
            operation_name: Name for logging
            timeout: Timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries

        Returns:
            tuple: (result, retry_count)
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"{operation_name}: Attempt {attempt + 1}/{max_retries + 1}")

                # Execute with timeout
                result = await asyncio.wait_for(operation(), timeout=timeout)
                return result, attempt

            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"{operation_name}: Timeout on attempt {attempt + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff

            except Exception as e:
                # For non-timeout errors, don't retry
                logger.error(f"{operation_name}: Non-retryable error on attempt {attempt + 1}: {e}")
                raise e

        # All retries exhausted due to timeout
        raise asyncio.TimeoutError(f"{operation_name}: All {max_retries + 1} attempts timed out")

    def _classify_error(self, exception: Exception) -> str:
        """
        Classify an exception into error categories.

        Args:
            exception: The exception to classify

        Returns:
            str: Error type from PipelineError enum
        """
        import requests

        if isinstance(exception, asyncio.TimeoutError):
            return PipelineError.TIMEOUT_ERROR.value
        elif isinstance(exception, requests.exceptions.RequestException):
            return PipelineError.NETWORK_ERROR.value
        elif "LLM" in str(exception) or "model" in str(exception).lower():
            return PipelineError.LLM_ERROR.value
        elif "validation" in str(exception).lower():
            return PipelineError.VALIDATION_ERROR.value
        else:
            return PipelineError.UNKNOWN_ERROR.value

    async def run_ingredient_parsing_async(self, dry_run: bool = False) -> OperationResult:
        """
        Run ingredient parsing operation asynchronously with timeout and retry logic.

        Args:
            dry_run: If True, simulate the operation without making changes

        Returns:
            OperationResult with success status and data
        """
        start_time = time.time()
        logger.info("Starting ingredient parsing operation" + (" (dry run)" if dry_run else ""))

        try:
            if dry_run:
                # Simulate parsing operation
                logger.debug("Dry run mode: simulating parsing operation")
                await asyncio.sleep(0.1)  # Simulate work
                result = OperationResult(
                    status=OperationStatus.SUCCESS,
                    data={"simulated": True, "message": "Dry run - no changes made"},
                    execution_time=time.time() - start_time
                )
                logger.info(f"Ingredient parsing dry run completed successfully in {result.execution_time:.2f}s")
                return result

            async def _parse_operation():
                logger.debug("Fetching unparsed recipe slugs")
                # Import and run actual parsing logic
                from mealie_parse import parse_unparsed_recipes_batch, get_unparsed_slugs

                # Get unparsed recipes
                unparsed_slugs = get_unparsed_slugs()
                if not unparsed_slugs:
                    logger.info("No unparsed recipes found")
                    return {"parsed_recipes": 0, "message": "No unparsed recipes found"}

                # Limit to prevent timeouts
                slugs_to_parse = unparsed_slugs  # Parse ALL unparsed recipes
                logger.info(f"Processing {len(slugs_to_parse)} recipes (limited from {len(unparsed_slugs)} total)")

                # Build recipe list for parsing (name is only used for logging, slug suffices)
                # parse_unparsed_recipes_batch() fetches full recipe data internally
                unparsed_recipes = [{'slug': slug, 'name': slug} for slug in slugs_to_parse]
                
                from mealie_client import MealieClient
                client = MealieClient()

                try:

                    if unparsed_recipes:
                        logger.debug(f"Starting parsing for {len(unparsed_recipes)} recipes")
                        results = await parse_unparsed_recipes_batch(unparsed_recipes, auto_tag=True)
                        success_count = sum(1 for r in results if r and r.get('success'))

                        logger.info(f"Parsing completed: {success_count}/{len(unparsed_recipes)} recipes successful")
                        return {
                            "parsed_recipes": len(unparsed_recipes),
                            "successful_parses": success_count,
                            "message": f"Parsed {success_count}/{len(unparsed_recipes)} recipes successfully"
                        }
                    else:
                        logger.warning("No valid recipes found to parse after fetching")
                        return {"parsed_recipes": 0, "message": "No valid recipes found to parse"}

                finally:
                    client.close()

            # Execute with timeout and retry
            logger.debug(f"Executing parsing with timeout={self.config.ingredient_parsing_timeout}s, max_retries={self.config.max_network_retries}")
            result_data, retry_count = await self._execute_with_timeout_and_retry(
                operation=_parse_operation,
                operation_name="ingredient_parsing",
                timeout=self.config.ingredient_parsing_timeout,
                max_retries=self.config.max_network_retries
            )

            execution_time = time.time() - start_time
            logger.info(f"Ingredient parsing completed successfully in {execution_time:.2f}s (retries: {retry_count})")
            return OperationResult(
                status=OperationStatus.SUCCESS,
                data=result_data,
                execution_time=execution_time,
                retry_count=retry_count
            )

        except Exception as e:
            error_type = self._classify_error(e)
            execution_time = time.time() - start_time
            logger.error(f"Operation failed after {execution_time:.2f}s: {e} (type: {error_type})")
            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=str(e),
                error_type=error_type,
                execution_time=execution_time
            )

    async def run_shopping_list_refinement(self, shopping_list_id: str, dry_run: bool = False) -> OperationResult:
        """
        Run shopping list refinement operation asynchronously with timeout and retry logic.

        Uses shopping_pipeline_v2 (rule-based approach that preserves Mealie's quantities).

        Complete workflow:
        1. Fetch shopping list from Mealie
        2. Apply v2 refinement (filter pantry items, reject garbage, clean display)
        3. Delete filtered items from Mealie
        4. Re-add cleaned items with proper foodId/unitId

        Args:
            shopping_list_id: ID of the shopping list to refine
            dry_run: If True, simulate the operation without making changes

        Returns:
            OperationResult with success status and data
        """
        start_time = time.time()
        logger.info(f"Starting shopping list refinement for ID: {shopping_list_id}" + (" (dry run)" if dry_run else ""))

        # Validate input
        if not shopping_list_id or not isinstance(shopping_list_id, str):
            logger.error(f"Invalid shopping list ID: {shopping_list_id}")
            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message="Invalid shopping list ID",
                error_type=PipelineError.VALIDATION_ERROR.value,
                execution_time=time.time() - start_time
            )

        try:
            if dry_run:
                # Simulate refinement operation
                logger.debug("Dry run mode: simulating refinement operation")
                await asyncio.sleep(0.1)  # Simulate work
                result = OperationResult(
                    status=OperationStatus.SUCCESS,
                    data={"simulated": True, "message": "Dry run - no changes made"},
                    execution_time=time.time() - start_time
                )
                logger.info(f"Shopping list refinement dry run completed successfully in {result.execution_time:.2f}s")
                return result

            async def _refinement_operation():
                # Step 1: Fetch shopping list from Mealie
                from mealie_shopping_integration import (
                    fetch_mealie_shopping_list,
                    delete_mealie_shopping_items,
                    add_mealie_shopping_item
                )
                from shopping_pipeline_v2 import refine_shopping_list, format_for_mealie

                shopping_list = fetch_mealie_shopping_list(shopping_list_id)

                if not shopping_list:
                    raise Exception(f"Failed to fetch shopping list {shopping_list_id}")

                items = shopping_list.get("listItems", [])
                if not items:
                    return {"message": "No items to refine", "items_processed": 0}

                # Step 2: Apply v2 refinement (rule-based, preserves quantities)
                result = refine_shopping_list(items)

                if not result.success:
                    raise RuntimeError(f"Refinement failed: {result.errors}")

                # Step 3: Delete filtered items from Mealie
                if result.items_to_delete:
                    logger.info(f"Deleting {len(result.items_to_delete)} filtered items...")
                    delete_mealie_shopping_items(result.items_to_delete)

                # Step 4: Delete remaining items (we'll re-add with cleaned display)
                remaining_ids = [item.original_id for item in result.items_to_keep if item.original_id]
                if remaining_ids:
                    logger.info(f"Replacing {len(remaining_ids)} items with cleaned versions...")
                    delete_mealie_shopping_items(remaining_ids)

                # Step 5: Format items for Mealie and add them back
                mealie_items = format_for_mealie(result.items_to_keep)

                successful = 0
                failed = 0
                for item in mealie_items:
                    try:
                        add_mealie_shopping_item(shopping_list_id, item)
                        successful += 1
                    except Exception as e:
                        logger.warning(f"Failed to add item {item.get('display', 'unknown')}: {e}")
                        failed += 1

                result_data = {
                    "items_processed": len(items),
                    "items_to_keep": len(result.items_to_keep),
                    "items_to_delete": len(result.items_to_delete),
                    "pantry_filtered": len(result.pantry_filtered),
                    "garbage_rejected": len(result.garbage_rejected),
                    "items_added": successful,
                    "items_failed": failed,
                }

                # Check for degraded success
                if failed > 0:
                    result_data["degraded"] = True
                    result_data["degraded_reason"] = f"{failed} items failed to add"

                return result_data

            # Execute with timeout and retry
            result_data, retry_count = await self._execute_with_timeout_and_retry(
                operation=_refinement_operation,
                operation_name="shopping_list_refinement",
                timeout=self.config.shopping_refinement_timeout,
                max_retries=self.config.max_llm_retries
            )

            # Determine status based on result
            status = OperationStatus.SUCCESS
            error_message = None

            if result_data.get("degraded"):
                status = OperationStatus.DEGRADED
                error_message = f"Refinement completed in degraded mode: {result_data.get('degraded_reason', 'Unknown issues')}"

            execution_time = time.time() - start_time
            log_level = logger.warning if status == OperationStatus.DEGRADED else logger.info
            log_level(f"Shopping list refinement completed ({status.value}) in {execution_time:.2f}s (retries: {retry_count})")
            if error_message:
                logger.warning(f"Refinement note: {error_message}")

            return OperationResult(
                status=status,
                data=result_data,
                error_message=error_message,
                execution_time=execution_time,
                retry_count=retry_count,
                degraded_mode=result_data.get("degraded", False)
            )

        except Exception as e:
            error_type = self._classify_error(e)
            execution_time = time.time() - start_time
            logger.error(f"Operation failed after {execution_time:.2f}s: {e} (type: {error_type})")
            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=str(e),
                error_type=error_type,
                execution_time=execution_time
            )


class PipelineMonitoring:
    """Simple monitoring for pipeline operations."""

    def __init__(self):
        self.operations = {}

    def record_operation(self, operation_name: str, status: str, duration: float):
        """Record an operation for monitoring."""
        self.operations[operation_name] = {
            'status': status,
            'duration': duration,
            'timestamp': time.time()
        }

    def get_operation_metrics(self) -> Dict[str, Any]:
        """Get metrics for all operations."""
        return self.operations.copy()
