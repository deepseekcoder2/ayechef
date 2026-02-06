#!/usr/bin/env python3
"""
Recipe Maintenance Script
=========================

Automated maintenance for the meal planning system.
Runs periodic checks and processing for optimal system health.

Features:
- Process any remaining unparsed recipes
- Re-index recipes missing from RecipeRAG
- Clean up and optimize database
- Health checks and reporting

Usage:
    python recipe_maintenance.py              # Full maintenance
    python recipe_maintenance.py --quick      # Quick check only
    python recipe_maintenance.py --force      # Force reprocessing
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import subprocess
from typing import List, Dict, Set
from config import MEALIE_URL, MEALIE_TOKEN, get_mealie_headers, validate_all, mealie_rate_limit, DATA_DIR
from tools.logging_utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


def check_unparsed_recipes() -> List[Dict]:
    """
    Check for unparsed recipes using cached local index.
    
    Uses get_unparsed_slugs() as the single source of truth, which:
    - First checks local SQLite index (instant for known recipes)
    - Only queries Mealie API for recipes with unknown status
    - Updates local index with results for future fast lookups
    
    Returns:
        List of dicts with 'id', 'slug', 'name' keys for compatibility
        with parse_unparsed_recipes_batch().
    """
    from mealie_parse import get_unparsed_slugs
    from recipe_rag import RecipeRAG
    
    try:
        print("🔍 Scanning recipes for parsing status...", flush=True)
        
        # Get unparsed slugs using optimized function (uses local index)
        unparsed_slugs = get_unparsed_slugs()
        
        if not unparsed_slugs:
            print("✅ Found 0 recipes needing parsing", flush=True)
            return []
        
        # Get recipe names from local index for display purposes
        rag = RecipeRAG()
        unparsed_recipes = []
        
        for slug in unparsed_slugs:
            # Try to get name from local index
            recipe_info = rag.get_recipe_by_slug(slug)
            if recipe_info:
                unparsed_recipes.append({
                    "id": recipe_info.get("id", ""),
                    "slug": slug,
                    "name": recipe_info.get("name", slug)
                })
            else:
                # Recipe not in local index, use slug as name
                unparsed_recipes.append({
                    "id": "",
                    "slug": slug,
                    "name": slug
                })
        
        print(f"✅ Found {len(unparsed_recipes)} recipes needing parsing", flush=True)
        return unparsed_recipes

    except Exception as e:
        logger.error(f"❌ Error checking unparsed recipes: {e}")
        print(f"❌ Error checking unparsed recipes: {e}", flush=True)
        return []


def check_rag_index_completeness() -> tuple[int, int]:
    """Check if RecipeRAG index is complete."""
    print("🔍 Checking RecipeRAG index completeness...", flush=True)

    try:
        # Get total recipes in Mealie
        from mealie_client import MealieClient
        client = MealieClient()
        try:
            all_recipes = client.get_all_recipes()
            total_mealie = len(all_recipes)
        finally:
            client.close()

        # Get total in RecipeRAG
        from recipe_rag import RecipeRAG
        rag = RecipeRAG()
        total_rag = rag.get_total_recipes()

        print(f"✅ Mealie: {total_mealie} recipes, RecipeRAG: {total_rag} indexed", flush=True)

        return total_mealie, total_rag

    except Exception as e:
        logger.error(f"❌ Error checking RAG completeness: {e}")
        print(f"❌ Error checking RAG completeness: {e}", flush=True)
        return 0, 0


def populate_mealie_timestamps() -> int:
    """
    One-time migration: Populate mealie_updated_at for existing recipes.
    
    This is run after adding the mealie_updated_at column to avoid re-indexing
    all recipes. It only updates the timestamp column, not the content.
    
    Returns:
        Number of recipes updated
    """
    import sqlite3
    from mealie_client import MealieClient
    
    print("🔄 Populating Mealie timestamps (one-time migration)...", flush=True)
    
    try:
        # Fetch all recipe summaries from Mealie
        print("  📡 Fetching recipe summaries from Mealie...", flush=True)
        client = MealieClient()
        mealie_recipes = client.get_all_recipes()
        client.close()
        
        mealie_by_id = {r.get("id"): r for r in mealie_recipes if r.get("id")}
        print(f"  Found {len(mealie_by_id)} recipes in Mealie", flush=True)
        
        # Update timestamps in local DB
        db_path = str(DATA_DIR / "recipe_index.db")
        updated = 0
        
        with sqlite3.connect(db_path) as conn:
            for recipe_id, recipe in mealie_by_id.items():
                mealie_ts = recipe.get('updatedAt') or recipe.get('dateUpdated') or ''
                if mealie_ts:
                    conn.execute(
                        "UPDATE recipes SET mealie_updated_at = ? WHERE id = ? AND (mealie_updated_at IS NULL OR mealie_updated_at = '')",
                        (mealie_ts, recipe_id)
                    )
                    if conn.total_changes > updated:
                        updated = conn.total_changes
            conn.commit()
        
        print(f"  ✅ Updated timestamps for {updated} recipes", flush=True)
        return updated
        
    except Exception as e:
        logger.error(f"❌ Error populating timestamps: {e}")
        print(f"❌ Error populating timestamps: {e}", flush=True)
        return 0


def sync_local_db_with_mealie(force_full: bool = False) -> tuple[int, int, int, int]:
    """
    Incremental sync of local DB with Mealie using timestamp comparison.
    
    Only re-indexes recipes that have actually changed, making it fast for
    regular maintenance (seconds instead of hours for 17k+ recipes).
    
    Detection logic:
    - NEW: in Mealie but not in local → add
    - DELETED: in local but not in Mealie → remove
    - MODIFIED: Mealie's updatedAt > stored mealie_updated_at → re-index
    - UNCHANGED: timestamps match → skip
    
    Args:
        force_full: If True, re-index ALL recipes (use for first-time setup or recovery)
    
    Returns:
        Tuple of (added, removed, modified, unchanged) counts
    """
    import sqlite3
    from mealie_client import MealieClient, MealieAPIError
    from recipe_rag import RecipeRAG
    
    print("🔄 Syncing local database with Mealie...", flush=True)
    
    added = 0
    removed = 0
    modified = 0
    unchanged = 0
    
    try:
        # Step 1: Get all recipe summaries from Mealie (includes updatedAt)
        print("  📡 Fetching recipe summaries from Mealie...", flush=True)
        from mealie_client import MealieClient
        client = MealieClient()
        mealie_recipes = client.get_all_recipes()
        
        mealie_by_id = {r.get("id"): r for r in mealie_recipes if r.get("id")}
        mealie_ids = set(mealie_by_id.keys())
        print(f"  Found {len(mealie_ids)} recipes in Mealie", flush=True)
        
        # Step 2: Get all recipe IDs and timestamps from local DB
        rag = RecipeRAG()
        local_timestamps = rag.get_all_recipe_timestamps()
        local_ids = set(local_timestamps.keys())
        print(f"  Found {len(local_ids)} recipes in local DB", flush=True)
        
        # Step 2.5: Fix mismatched slugs (Mealie data inconsistency fix)
        # The list API returns correct slugs, but recipe detail may have stale ones
        db_path = str(DATA_DIR / "recipe_index.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, slug FROM recipes")
            local_slugs = {row[0]: row[1] for row in cursor.fetchall()}
        
        slugs_fixed = 0
        for recipe_id, local_slug in local_slugs.items():
            mealie_recipe = mealie_by_id.get(recipe_id)
            if mealie_recipe:
                correct_slug = mealie_recipe.get('slug', '')
                if correct_slug and local_slug != correct_slug:
                    with sqlite3.connect(db_path) as conn:
                        conn.execute("UPDATE recipes SET slug = ? WHERE id = ?", 
                                   (correct_slug, recipe_id))
                        conn.commit()
                    slugs_fixed += 1
        
        if slugs_fixed > 0:
            print(f"  🔧 Fixed {slugs_fixed} mismatched slugs", flush=True)
        
        # Check for migration scenario: if most recipes have no timestamps, 
        # populate them instead of re-indexing everything
        recipes_without_ts = sum(1 for ts in local_timestamps.values() if not ts)
        if recipes_without_ts > 0 and recipes_without_ts >= len(local_ids) * 0.9 and not force_full:
            print(f"  ⚠️  {recipes_without_ts}/{len(local_ids)} recipes missing timestamps (migration needed)", flush=True)
            print("  🔄 Running timestamp migration (no re-indexing)...", flush=True)
            
            # Populate timestamps only
            db_path = str(DATA_DIR / "recipe_index.db")
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                for recipe_id in local_ids:
                    mealie_recipe = mealie_by_id.get(recipe_id)
                    if mealie_recipe:
                        mealie_ts = mealie_recipe.get('updatedAt') or mealie_recipe.get('dateUpdated') or ''
                        if mealie_ts:
                            conn.execute(
                                "UPDATE recipes SET mealie_updated_at = ? WHERE id = ?",
                                (mealie_ts, recipe_id)
                            )
                conn.commit()
            
            print(f"  ✅ Timestamps populated for {len(local_ids)} recipes", flush=True)
            # Re-fetch timestamps after migration
            local_timestamps = rag.get_all_recipe_timestamps()
        
        # Step 3: Find recipes to REMOVE (deleted from Mealie)
        to_remove = local_ids - mealie_ids
        if to_remove:
            # SAFETY CHECK: Prevent catastrophic deletion if Mealie API returned bad data
            # If more than 50% of recipes would be deleted, abort!
            deletion_ratio = len(to_remove) / len(local_ids) if local_ids else 0
            if deletion_ratio > 0.5 and len(to_remove) > 100:
                print(f"  ⚠️  SAFETY ABORT: Would delete {len(to_remove)}/{len(local_ids)} recipes ({deletion_ratio*100:.0f}%)", flush=True)
                print(f"  ⚠️  This looks like a Mealie API error. Skipping deletion.", flush=True)
                logger.error(f"Safety abort: would delete {len(to_remove)}/{len(local_ids)} recipes")
            else:
                print(f"  🗑️  Removing {len(to_remove)} deleted recipes...", flush=True)
                db_path = str(DATA_DIR / "recipe_index.db")
                with sqlite3.connect(db_path) as conn:
                    for recipe_id in to_remove:
                        conn.execute("DELETE FROM recipes WHERE id = ?", (recipe_id,))
                        conn.execute("DELETE FROM recipes_fts WHERE id = ?", (recipe_id,))
                        removed += 1
                    conn.commit()
            
            # Rebuild ANN index from remaining SQLite embeddings
            # USearch doesn't support individual removal, so we rebuild after deletions
            print(f"  🔧 Rebuilding ANN index after deletions...", flush=True)
            try:
                from utils.rebuild_ann_index import rebuild_ann_index
                rebuild_ann_index(db_path)
                print(f"  ✅ ANN index rebuilt", flush=True)
            except Exception as e:
                logger.warning(f"⚠️  ANN index rebuild failed (will be stale until next rebuild): {e}")
                print(f"  ⚠️  ANN index rebuild failed: {e}", flush=True)
            
            print(f"  ✅ Removed {removed} deleted recipes", flush=True)
        
        # Step 4: Identify what needs indexing
        to_add = []      # New recipes (not in local)
        to_update = []   # Modified recipes (timestamp changed)
        
        for recipe_id, mealie_recipe in mealie_by_id.items():
            mealie_ts = mealie_recipe.get('updatedAt') or mealie_recipe.get('dateUpdated') or ''
            
            if recipe_id not in local_ids:
                # NEW recipe
                to_add.append((recipe_id, mealie_recipe))
            elif force_full:
                # Force mode - update everything
                to_update.append((recipe_id, mealie_recipe))
            else:
                # Check if modified
                stored_ts = local_timestamps.get(recipe_id) or ''
                
                if not stored_ts:
                    # No stored timestamp = need to sync (first run after migration)
                    to_update.append((recipe_id, mealie_recipe))
                elif mealie_ts > stored_ts:
                    # Mealie is newer = recipe was modified
                    to_update.append((recipe_id, mealie_recipe))
                else:
                    # Timestamps match = unchanged
                    unchanged += 1
        
        total_to_process = len(to_add) + len(to_update)
        
        if total_to_process == 0:
            print(f"  ✅ All {unchanged} recipes are up to date!", flush=True)
            return added, removed, modified, unchanged
        
        print(f"  📊 To process: {len(to_add)} new, {len(to_update)} modified, {unchanged} unchanged", flush=True)
        
        # Step 5: Process recipes that need indexing using batched approach
        all_to_process = to_add + to_update
        add_ids = {r[0] for r in to_add}
        recipes_to_index = []
        
        # OPTIMIZATION: In DB mode with many recipes, use bulk fetch (5-10x faster)
        # This fetches all full recipe data in ~5 SQL queries instead of N individual calls
        use_bulk_fetch = client.mode == 'db' and total_to_process > 100
        
        if use_bulk_fetch:
            print(f"  📥 Bulk fetching {total_to_process} recipes from database...", flush=True)
            
            # Get all full recipes in one batch
            all_full_recipes = client.get_all_recipes_full()
            full_recipes_by_id = {r.get('id'): r for r in all_full_recipes}
            
            print(f"  ✅ Loaded {len(full_recipes_by_id)} full recipes", flush=True)
            
            for recipe_id, mealie_summary in all_to_process:
                # Convert recipe_id to dashed format for lookup (DB mode returns dashed IDs now)
                formatted_id = recipe_id if '-' in recipe_id else f"{recipe_id[:8]}-{recipe_id[8:12]}-{recipe_id[12:16]}-{recipe_id[16:20]}-{recipe_id[20:]}"
                recipe_detail = full_recipes_by_id.get(formatted_id)
                
                if not recipe_detail:
                    logger.warning(f"⚠️  Recipe {mealie_summary.get('name')} not found in bulk fetch")
                    continue
                
                # Get Mealie's timestamp for storage
                mealie_ts = mealie_summary.get('updatedAt') or mealie_summary.get('dateUpdated') or ''
                if mealie_ts and 'updatedAt' not in recipe_detail:
                    recipe_detail['updatedAt'] = mealie_ts
                
                recipes_to_index.append((recipe_id, recipe_detail, recipe_id in add_ids))
        else:
            # Standard approach: fetch one at a time (still fast in DB mode)
            print(f"  📥 Fetching {total_to_process} recipe details...", flush=True)
            
            for i, (recipe_id, mealie_summary) in enumerate(all_to_process, 1):
                if i % 100 == 0:
                    print(f"  Fetching: {i}/{total_to_process}", flush=True)
                
                try:
                    slug = mealie_summary.get('slug')
                    if not slug:
                        continue
                    
                    try:
                        recipe_detail = client.get_recipe(slug)
                        
                        # CRITICAL: Override slug with known-good one from list API
                        # The detail response may have a corrupted/stale slug field
                        # (e.g., from Mealie data inconsistencies after bulk operations)
                        recipe_detail['slug'] = slug
                    except MealieAPIError as e:
                        if e.status_code in (403, 500):
                            # Recipe has corrupted slug routing (Mealie bug after rename)
                            # Log but don't spam - this is a known issue
                            logger.debug(f"Recipe {slug} has corrupted routing (status {e.status_code})")
                        else:
                            logger.warning(f"⚠️  Could not fetch recipe {slug}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"⚠️  Could not fetch recipe {slug}: {e}")
                        continue
                    
                    # Get Mealie's timestamp for storage
                    mealie_ts = mealie_summary.get('updatedAt') or mealie_summary.get('dateUpdated') or ''
                    if mealie_ts and 'updatedAt' not in recipe_detail:
                        recipe_detail['updatedAt'] = mealie_ts
                    
                    recipes_to_index.append((recipe_id, recipe_detail, recipe_id in add_ids))
                            
                except Exception as e:
                    logger.debug(f"Error fetching {mealie_summary.get('name')}: {e}")
        
        # Batch index all recipes at once (single embedding call)
        if recipes_to_index:
            print(f"  📊 Batch indexing {len(recipes_to_index)} recipes...", flush=True)
            
            recipe_data_list = [r[1] for r in recipes_to_index]
            indexed_count = rag.index_recipes_batch(recipe_data_list, force=True)
            
            # Count adds vs updates
            for recipe_id, _, is_add in recipes_to_index:
                if is_add:
                    added += 1
                else:
                    modified += 1
            
            print(f"  ✅ Batch indexed {indexed_count} recipes", flush=True)
        
            print(f"  ✅ Sync complete: +{added} added, -{removed} removed, ~{modified} modified, ={unchanged} unchanged", flush=True)
            
            # Step 7: Repair recipes with missing cuisine_primary
            # This handles recipes that were indexed before tags were added
            repaired = _repair_missing_cuisines(mealie_ids, mealie_by_id, rag)
            if repaired > 0:
                print(f"  ✅ Repaired {repaired} recipes with missing cuisine", flush=True)
            
            return added, removed, modified, unchanged
        
    except Exception as e:
        logger.error(f"❌ Error syncing local DB: {e}")
        print(f"❌ Error syncing local DB: {e}", flush=True)
        return added, removed, modified, unchanged
    finally:
        client.close()


def _repair_missing_cuisines(mealie_ids: set, mealie_by_id: dict, rag) -> int:
    """
    Repair recipes that have cuisine tags in Mealie but NULL cuisine_primary in local DB.
    
    This handles recipes that were indexed before tags were added - since the timestamp
    comparison skips unchanged recipes, cuisine_primary would remain NULL forever.
    
    Args:
        mealie_ids: Set of recipe IDs that exist in Mealie
        mealie_by_id: Dict mapping recipe ID to Mealie summary data
        rag: RecipeRAG instance for extracting cuisine from tags
    
    Returns:
        Number of recipes repaired
    """
    import sqlite3
    
    db_path = str(DATA_DIR / "recipe_index.db")
    repaired = 0
    
    try:
        # Find recipes with missing cuisine_primary
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, slug FROM recipes 
                WHERE cuisine_primary IS NULL OR cuisine_primary = ''
            """)
            broken_recipes = cursor.fetchall()
        
        if not broken_recipes:
            return 0
        
        # Filter to only recipes that still exist in Mealie
        broken_recipes = [(rid, slug) for rid, slug in broken_recipes if rid in mealie_ids]
        
        if not broken_recipes:
            return 0
        
        print(f"  🔧 Repairing {len(broken_recipes)} recipes with missing cuisine...", flush=True)
        
        # Create client for fetching recipes
        from mealie_client import MealieAPIError
        client = MealieClient()
        try:
            for recipe_id, _local_slug in broken_recipes:
                # IMPORTANT: Use slug from LIST API (mealie_by_id), not local DB
                # Local DB slug may be stale due to Mealie's slug routing bug
                mealie_recipe = mealie_by_id.get(recipe_id)
                if not mealie_recipe:
                    continue
                slug = mealie_recipe.get('slug')
                if not slug:
                    continue
                
                try:
                    # Fetch full recipe from Mealie to get tags
                    recipe_data = client.get_recipe(slug)
                except MealieAPIError as e:
                    if e.status_code in (403, 500):
                        logger.warning(f"Recipe {slug} has corrupted routing (status {e.status_code}), skipping")
                    else:
                        logger.debug(f"Could not fetch recipe {slug} for repair: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Could not fetch recipe {slug} for repair: {e}")
                    continue
                tags = recipe_data.get("tags", [])
                
                if not tags:
                    # No tags in Mealie, nothing to repair
                    continue
                
                # Extract cuisine from tags
                cuisine_primary, cuisine_secondary, cuisine_region, cuisine_confidence = rag._extract_cuisine_from_tags(tags)
                
                if not cuisine_primary:
                    # Tags exist but no cuisine tag found
                    continue
                
                # Update only the cuisine fields in local DB
                with sqlite3.connect(db_path) as conn:
                    conn.execute("""
                        UPDATE recipes 
                        SET cuisine_primary = ?, 
                            cuisine_secondary = ?, 
                            cuisine_region = ?, 
                            cuisine_confidence = ?
                        WHERE id = ?
                    """, (cuisine_primary, cuisine_secondary, cuisine_region, cuisine_confidence, recipe_id))
                    conn.commit()
                
                repaired += 1
                logger.debug(f"Repaired cuisine for {slug}: {cuisine_primary}")
                
        finally:
            client.close()
        
        return repaired
        
    except Exception as e:
        logger.warning(f"Error in cuisine repair pass: {e}")
        return repaired


def run_maintenance(force: bool = False, quick: bool = False, sync_only: bool = False) -> bool:
    """Run full maintenance suite."""
    print("🛠️  Starting recipe maintenance...", flush=True)
    print("=" * 50, flush=True)

    success = True

    try:
        # Quick check only
        if quick:
            print("⚡ Quick maintenance mode", flush=True)

            # Check system health
            if not validate_all():
                print("❌ System validation failed", flush=True)
                return False

            # Check unparsed recipes
            unparsed = check_unparsed_recipes()
            if unparsed:
                print(f"⚠️  {len(unparsed)} recipes need parsing", flush=True)

            # Check RAG completeness - compare counts AND check for stale entries
            total_mealie, total_rag = check_rag_index_completeness()
            if total_rag != total_mealie:
                diff = total_rag - total_mealie
                if diff > 0:
                    print(f"⚠️  Local DB has {diff} stale/deleted recipes (needs sync)", flush=True)
                else:
                    print(f"⚠️  Local DB missing {-diff} recipes (needs sync)", flush=True)

            print("✅ Quick maintenance complete", flush=True)
            return True

        # Sync-only mode - just sync DB, skip parsing and other steps
        if sync_only:
            print("🔄 Sync-only mode - syncing local DB with Mealie...", flush=True)
            added, removed, modified, unchanged = sync_local_db_with_mealie()
            print(f"\n✅ Sync complete: +{added} added, -{removed} removed, ~{modified} modified, ={unchanged} unchanged", flush=True)
            return True

        # Full maintenance
        print("🔧 Full maintenance mode", flush=True)

        # Step 1: Process unparsed recipes
        print("\n📋 Step 1: Processing unparsed recipes...", flush=True)
        unparsed = check_unparsed_recipes()

        if unparsed or force:
            print(f"🚀 Parsing {len(unparsed)} recipes...", flush=True)
            # Use Popen for streaming output
            parse_proc = subprocess.Popen(
                [sys.executable, "mealie_parse.py", "--scan-unparsed", "--auto-tag", "--yes"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            # Stream output in real-time
            for line in parse_proc.stdout:
                print(f"  {line.rstrip()}", flush=True)
            parse_proc.wait()

            if parse_proc.returncode == 0:
                print("✅ Recipe parsing completed", flush=True)
            else:
                logger.warning("⚠️  Recipe parsing had issues")
                print("⚠️  Recipe parsing had issues", flush=True)
                success = False
        else:
            print("✅ No unparsed recipes found", flush=True)

        # Step 2: Sync local database with Mealie FIRST (must happen before tagging)
        # This ensures the local DB reflects current Mealie state before any operations
        print("\n🔄 Step 2: Syncing local database with Mealie...", flush=True)
        total_mealie, total_rag = check_rag_index_completeness()
        print(f"  Mealie has {total_mealie}, local has {total_rag}", flush=True)
        
        # Always run incremental sync - it's fast now (only processes changed recipes)
        # The force flag triggers a full re-index of everything
        added, removed, modified, unchanged = sync_local_db_with_mealie(force_full=force)
        
        if added > 0 or removed > 0 or modified > 0:
            print(f"✅ Sync completed: +{added} added, -{removed} removed, ~{modified} modified", flush=True)
        else:
            print(f"✅ All {unchanged} recipes already up to date", flush=True)

        # Step 3: Tag untagged recipes (parsed but missing cuisine/prep tags)
        # Now that local DB is synced, tagging will work on accurate data
        print("\n🏷️  Step 3: Tagging untagged recipes...", flush=True)
        try:
            # Check for untagged recipes in local DB
            import sqlite3
            db_path = str(DATA_DIR / "recipe_index.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM recipes 
                    WHERE cuisine_primary IS NULL OR cuisine_primary = ''
                """)
                untagged_count = cursor.fetchone()[0]
            
            if untagged_count > 0 or force:
                print(f"🏷️  Found {untagged_count} untagged recipes, running bulk tagger...", flush=True)
                # Use Popen for streaming output
                tag_proc = subprocess.Popen(
                    [sys.executable, "utils/bulk_tag.py", "--all", "--yes"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                for line in tag_proc.stdout:
                    print(f"  {line.rstrip()}", flush=True)
                tag_proc.wait()
                
                if tag_proc.returncode == 0:
                    print("✅ Tagging completed", flush=True)
                else:
                    logger.warning("⚠️  Tagging had issues")
                    print("⚠️  Tagging had issues", flush=True)
                    success = False
            else:
                print("✅ All recipes are tagged", flush=True)
        except Exception as e:
            logger.warning(f"⚠️  Tagging check failed: {e}")
            print(f"⚠️  Tagging check failed: {e}", flush=True)

        # Step 4: Database optimization
        print("\n🗃️  Step 4: Database optimization...", flush=True)
        try:
            from recipe_rag import RecipeRAG
            rag = RecipeRAG()

            # Rebuild FTS index for better search performance
            rag.rebuild_fts_index()
            print("✅ Full-text search index rebuilt", flush=True)

        except Exception as e:
            logger.warning(f"⚠️  Database optimization failed: {e}")
            print(f"⚠️  Database optimization failed: {e}", flush=True)
            success = False

        # Step 5: Final health check
        print("\n🏥 Step 5: Final health check...", flush=True)
        if validate_all():
            print("✅ System health check passed", flush=True)
        else:
            logger.warning("⚠️  System health check failed")
            print("⚠️  System health check failed", flush=True)
            success = False

        print("\n" + "=" * 50, flush=True)
        if success:
            print("🎉 Maintenance completed successfully!", flush=True)
        else:
            print("⚠️  Maintenance completed with some issues", flush=True)

        return success

    except Exception as e:
        print(f"❌ Maintenance failed: {e}", flush=True)
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Recipe maintenance automation")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing of all recipes")
    parser.add_argument("--quick", action="store_true",
                       help="Quick check without processing")
    parser.add_argument("--sync", action="store_true",
                       help="Sync local DB with Mealie only (add missing, remove deleted, update stale)")

    args = parser.parse_args()

    if args.quick and args.force:
        logger.error("❌ Cannot use --quick and --force together")
        print("❌ Cannot use --quick and --force together", flush=True)
        sys.exit(1)
    
    if args.sync and (args.quick or args.force):
        logger.error("❌ --sync cannot be combined with --quick or --force")
        print("❌ --sync cannot be combined with --quick or --force", flush=True)
        sys.exit(1)

    success = run_maintenance(force=args.force, quick=args.quick, sync_only=args.sync)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
