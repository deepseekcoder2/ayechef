#!/usr/bin/env python3
"""
Bulk Recipe Tagging Script for Mealie
=====================================

Production-ready bulk tagging for cuisine classification and preparation requirements.
Processes all untagged recipes efficiently using parallel LLM analysis.

Features:
- ✅ Tags all recipes with cuisine classifications and prep requirements
- ✅ Uses centralized configuration for optimal batch sizes
- ✅ Parallel processing with connection pooling
- ✅ Progress tracking and error recovery
- ✅ Environment-aware scaling (development/production/CI)
- ✅ Resume capability for interrupted operations
- ✅ Command-line interface with flexible options

Usage:
    # Tag all untagged recipes
    python bulk_tag.py --all

    # Tag specific number of recipes
    python bulk_tag.py --max 50

    # Tag with custom batch size
    python bulk_tag.py --all --batch-size 5

    # Dry run to see what would be tagged
    python bulk_tag.py --all --dry-run

    # Auto-confirm without prompts
    python bulk_tag.py --all --yes
"""

import sys
import os

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import asyncio
import sqlite3
import argparse
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from config import (
    validate_all,
    get_bulk_operation_config_safe,
    DATA_DIR,
)
from mealie_client import MealieClient
from tools.progress_ui import ui, create_progress_bar, Timer
from tools.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TaggingResult:
    """Result of a tagging operation."""
    recipe_id: str
    recipe_name: str
    success: bool
    tags_added: List[str]
    error_message: Optional[str]
    processing_time: float


def get_untagged_recipes(max_recipes: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Get all recipes that need tagging from the local database.

    Args:
        max_recipes: Maximum number of recipes to return (None = all)

    Returns:
        List of (recipe_id, recipe_name) tuples for untagged recipes
    """
    try:
        conn = sqlite3.connect(str(DATA_DIR / "recipe_index.db"))
        cursor = conn.cursor()

        # Get recipes without cuisine classification
        query = """
            SELECT id, name FROM recipes
            WHERE cuisine_primary IS NULL OR cuisine_primary = ''
            ORDER BY name
        """

        if max_recipes:
            query += f" LIMIT {max_recipes}"

        cursor.execute(query)
        recipes = cursor.fetchall()
        conn.close()

        return recipes

    except Exception as e:
        logger.error(f"❌ Failed to query untagged recipes: {e}", exc_info=True)
        print(f"❌ Failed to query untagged recipes: {e}")
        return []


def validate_system() -> bool:
    """Validate that all required systems are operational."""
    print("🔍 Validating system for bulk tagging...")
    logger.info("Validating system for bulk tagging")

    # Check basic configuration
    if not validate_all():
        logger.error("System validation failed for bulk tagging")
        print("❌ System validation failed - check configuration")
        return False

    # Check that we have untagged recipes
    untagged = get_untagged_recipes()
    if not untagged:
        logger.info("No recipes need tagging - all recipes already classified")
        print("✅ No recipes need tagging - all recipes already classified!")
        return False

    logger.info(f"Found {len(untagged)} recipes needing cuisine classification")
    print(f"📝 Found {len(untagged)} recipes needing cuisine classification")
    return True


def confirm_operation(recipes: List[Tuple[str, str]], dry_run: bool, skip_confirmation: bool) -> bool:
    """Get user confirmation before proceeding."""
    if dry_run:
        print("\n🔍 DRY RUN MODE - No actual tagging will be performed")
        return True

    if skip_confirmation:
        print(f"\n🚀 Auto-confirming bulk tagging of {len(recipes)} recipes")
        return True

    print(f"\n🚀 About to tag {len(recipes)} recipes with cuisine classifications")
    print("This will use LLM analysis and may take several minutes")

    while True:
        try:
            response = input("Proceed with bulk tagging? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                print("Bulk tagging cancelled by user")
                return False
        except KeyboardInterrupt:
            print("\nBulk tagging cancelled by user")
            return False


async def tag_recipe_batch(recipe_batch: List[Tuple[str, str]], dry_run: bool) -> List[TaggingResult]:
    """
    Tag a batch of recipes concurrently.

    Args:
        recipe_batch: List of (recipe_id, recipe_name) tuples
        dry_run: Whether to perform actual tagging

    Returns:
        List of tagging results
    """
    from automatic_tagger import AutomaticTagger

    tagger = AutomaticTagger()
    results = []

    # Get configuration for concurrent processing
    config = get_bulk_operation_config_safe('tag', fallback_batch_size=3, fallback_concurrent=2)
    max_concurrent = config['max_concurrent']

    async def tag_single_recipe(recipe_data: Tuple[str, str]) -> TaggingResult:
        """Tag a single recipe and return the result."""
        recipe_id, recipe_name = recipe_data
        start_time = time.time()

        try:
            if dry_run:
                # Dry run - just simulate success
                return TaggingResult(
                    recipe_id=recipe_id,
                    recipe_name=recipe_name,
                    success=True,
                    tags_added=["[DRY RUN]"],
                    error_message=None,
                    processing_time=time.time() - start_time
                )

            # Fetch recipe data
            # Force API mode - tags were just updated, need fresh data
            client = MealieClient(use_direct_db=False)
            try:
                recipe_data_full = client.get_recipe_by_id(recipe_id)
            finally:
                client.close()

            # Analyze and tag recipe
            analysis = await tagger.analyze_recipe(recipe_data_full)
            result = tagger.apply_tags_to_mealie(recipe_id, analysis)

            processing_time = time.time() - start_time

            if result.get('errors'):
                return TaggingResult(
                    recipe_id=recipe_id,
                    recipe_name=recipe_name,
                    success=False,
                    tags_added=result.get('tags_added', []) or [],
                    error_message="; ".join(str(err) for err in result['errors']),
                    processing_time=processing_time
                )
            else:
                return TaggingResult(
                    recipe_id=recipe_id,
                    recipe_name=recipe_name,
                    success=True,
                    tags_added=result.get('tags_added', []) or [],
                    error_message=None,
                    processing_time=processing_time
                )

        except Exception as e:
            return TaggingResult(
                recipe_id=recipe_id,
                recipe_name=recipe_name,
                success=False,
                tags_added=[],
                error_message=str(e) if e else "Unknown error",
                processing_time=time.time() - start_time
            )

    # Process batch with controlled concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def tag_with_semaphore(recipe_data):
        async with semaphore:
            return await tag_single_recipe(recipe_data)

    # Run concurrent tagging
    tasks = [tag_with_semaphore(recipe) for recipe in recipe_batch]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions that occurred during gathering
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            recipe_id, recipe_name = recipe_batch[i]
            results.append(TaggingResult(
                recipe_id=recipe_id,
                recipe_name=recipe_name,
                success=False,
                tags_added=[],
                error_message=f"Async error: {str(result)}",
                processing_time=0.0
            ))
        else:
            results.append(result)

    return results


async def run_bulk_tagging(max_recipes: Optional[int], batch_size: Optional[int],
                          dry_run: bool, skip_confirmation: bool) -> bool:
    """
    Run the bulk tagging operation.

    Args:
        max_recipes: Maximum recipes to process (None = all)
        batch_size: Override default batch size
        dry_run: Whether to perform actual tagging
        skip_confirmation: Skip user confirmation

    Returns:
        True if operation completed successfully
    """
    # Get untagged recipes
    recipes = get_untagged_recipes(max_recipes)
    if not recipes:
        logger.info("No recipes need tagging")
        print("No recipes need tagging")
        return True

    # Get configuration
    config = get_bulk_operation_config_safe('tag', fallback_batch_size=3, fallback_concurrent=2)
    if batch_size is None:
        batch_size = config['default_batch_size']

    logger.info(f"Using configuration: batch_size={batch_size}, max_concurrent={config['max_concurrent']}, environment={config.get('environment', 'unknown')}")
    print(f"⚙️  Using configuration: batch_size={batch_size}, max_concurrent={config['max_concurrent']}")
    print(f"   Environment: {config.get('environment', 'unknown')}")

    # Confirm operation
    if not confirm_operation(recipes, dry_run, skip_confirmation):
        return False

    # Process in batches
    total_recipes = len(recipes)
    all_results = []
    successful_tags = 0
    failed_tags = 0

    ui.start_operation("Bulk Tagging Recipes", total_recipes, f"Processing in batches of {batch_size}")

    for i in range(0, total_recipes, batch_size):
        batch = recipes[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_recipes + batch_size - 1) // batch_size

        ui.show_status(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} recipes)", "info")

        try:
            batch_results = await tag_recipe_batch(batch, dry_run)
            all_results.extend(batch_results)

            # Count results
            for result in batch_results:
                if result.success:
                    successful_tags += 1
                    # Ensure tags_added contains strings
                    tags_str = ', '.join(str(tag) for tag in result.tags_added)
                    ui.show_status(f"✅ {result.recipe_name} -> {tags_str}", "info")
                else:
                    failed_tags += 1
                    error_msg = str(result.error_message) if result.error_message else "Unknown error"
                    ui.show_status(f"❌ {result.recipe_name}: {error_msg}", "error")

            # Update progress after processing this batch
            ui.update_progress(completed=successful_tags + failed_tags)

        except Exception as e:
            ui.show_status(f"❌ Batch {batch_num} failed: {str(e)}", "error")
            failed_tags += len(batch)

    ui.complete_operation()

    # Summary
    if dry_run:
        logger.info(f"Dry run complete: would have tagged {len(all_results)} recipes with batch_size={batch_size}")
        print("\n🔍 DRY RUN COMPLETE")
        print(f"   Would have tagged: {len(all_results)} recipes")
        print(f"   Batch size: {batch_size}")
    else:
        logger.info(f"Bulk tagging complete: {successful_tags} successful, {failed_tags} failed, {len(all_results)} total")
        print("\n🏷️  BULK TAGGING COMPLETE")
        print(f"   Successfully tagged: {successful_tags} recipes")
        print(f"   Failed to tag: {failed_tags} recipes")
        print(f"   Total processed: {len(all_results)} recipes")

        if successful_tags > 0:
            print("\n✅ Recipes now have cuisine classifications for better meal planning!")
            
            # Re-index successfully tagged recipes to update local search index
            # This ensures the local index reflects the updated tag data
            print(f"\n🔄 Re-indexing {successful_tags} tagged recipes...", flush=True)
            try:
                from recipe_rag import RecipeRAG
                rag = RecipeRAG()
                
                # Get recipe IDs that were successfully tagged
                successful_ids = [r.recipe_id for r in all_results if r.success]
                
                if successful_ids:
                    # Fetch and index in batches
                    # Force API mode - tags were just updated, need fresh data
                    client = MealieClient(use_direct_db=False)
                    try:
                        recipes_data = []
                        for recipe_id in successful_ids:
                            try:
                                recipe_data = client.get_recipe_by_id(recipe_id)
                                recipes_data.append(recipe_data)
                            except Exception as e:
                                logger.warning(f"Failed to fetch {recipe_id} for re-indexing: {e}")
                        
                        if recipes_data:
                            indexed = rag.index_recipes_batch(recipes_data, force=True)
                            print(f"✅ Re-indexed {indexed}/{len(recipes_data)} recipes", flush=True)
                    finally:
                        client.close()
            except Exception as e:
                logger.warning(f"⚠️  Re-indexing failed (recipes tagged but local index not updated): {e}")
                print(f"⚠️  Re-indexing failed: {e}", flush=True)
            
            print("   Run 'python chef_agentic.py' to generate optimized meal plans")
    return successful_tags > 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bulk tag recipes with cuisine classifications and prep requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bulk_tag.py --all                    # Tag all untagged recipes
  python bulk_tag.py --max 50                 # Tag first 50 untagged recipes
  python bulk_tag.py --all --batch-size 2     # Use smaller batch size
  python bulk_tag.py --all --dry-run          # See what would be tagged
  python bulk_tag.py --all --yes              # Skip confirmation prompts
        """
    )

    parser.add_argument("--all", action="store_true",
                       help="Tag all untagged recipes")
    parser.add_argument("--max", type=int, metavar="N",
                       help="Tag maximum N recipes")
    parser.add_argument("--batch-size", type=int, metavar="N",
                       help="Override default batch size")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be tagged without making changes")
    parser.add_argument("--yes", action="store_true",
                       help="Skip confirmation prompts")

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.max:
        parser.error("Must specify --all or --max")
    if args.all and args.max:
        parser.error("Cannot specify both --all and --max")

    # Validate system
    if not validate_system():
        return

    # Run bulk tagging
    try:
        success = asyncio.run(run_bulk_tagging(
            max_recipes=None if args.all else args.max,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            skip_confirmation=args.yes
        ))
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.warning("Bulk tagging interrupted by user")
        print("\n❌ Bulk tagging interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Bulk tagging failed: {e}", exc_info=True)
        print(f"\n❌ Bulk tagging failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
