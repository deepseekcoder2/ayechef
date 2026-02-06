#!/usr/bin/env python3
"""
Test Import - Import a small batch of recipes to verify the actual pipeline works.

Uses the SAME batched_bulk_import pipeline as full imports, just with fewer recipes.
This ensures the test actually validates what will run in production.

Usage:
    python test_import.py https://example.com --job-id abc123
    python test_import.py https://example.com --count 10 --job-id abc123
"""

import sys
import json
import argparse
import random
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

from recipe_urls import is_site_supported, reload_scrapers
from import_site import fetch_sitemap_urls, filter_recipe_urls, get_existing_recipe_urls
from utils.url_utils import normalize_url
from bulk_import_smart import batched_bulk_import
from config import MEALIE_URL, DATA_DIR


@dataclass
class TestImportResults:
    """Complete test import results."""
    success: bool
    total: int
    succeeded: int
    failed: int
    filtered: int  # Non-recipes correctly filtered out (not failures)
    skipped: int
    recipes: List[Dict]
    # Timing data for estimating full import
    total_seconds: float = 0.0
    avg_seconds_per_recipe: float = 0.0
    estimated_full_import_time: str = None  # Human-readable estimate for N recipes
    # Index verification
    index_verified: int = 0
    index_missing: int = 0
    

def get_test_urls(site_url: str, count: int = 10) -> List[str]:
    """Get N random new recipe URLs from a site for testing."""
    reload_scrapers()
    
    if not is_site_supported(site_url):
        raise ValueError(f"Site not supported: {site_url}")
    
    # Fetch all recipe URLs
    print(f"📡 Fetching recipe URLs from sitemap...")
    all_urls = fetch_sitemap_urls(site_url)
    recipe_urls = filter_recipe_urls(all_urls, site_url)
    
    print(f"   Found {len(recipe_urls)} total recipes")
    
    # Filter out already imported (normalize URLs for consistent comparison)
    print(f"📋 Checking for existing recipes...")
    existing = get_existing_recipe_urls()  # Returns normalized URLs
    
    def safe_normalize_url(url: str) -> str:
        """Safely normalize URL, returning original on any error."""
        try:
            return normalize_url(url)
        except Exception:
            return url
    
    # Deduplicate and filter to new URLs only
    seen = set()
    new_urls = []
    for url in recipe_urls:
        normalized = safe_normalize_url(url)
        if normalized not in existing and normalized not in seen:
            seen.add(normalized)
            new_urls.append(url)
    
    print(f"   {len(new_urls)} new recipes available (deduplicated)")
    
    if not new_urls:
        return []
    
    # Random sample
    return random.sample(new_urls, min(count, len(new_urls)))


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{int(minutes)}-{int(minutes * 1.2)} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}-{hours * 1.2:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f}-{days * 1.2:.1f} days"


def run_test_import(site_url: str, count: int = 10, job_id: str = None, new_recipe_count: int = None) -> TestImportResults:
    """Run test import using the ACTUAL batched pipeline.
    
    Uses batched_bulk_import - the same code path as full imports.
    This ensures the test validates the real production pipeline.
    
    Args:
        site_url: Site URL to import from
        count: Number of recipes to test (default: 10)
        job_id: Job ID for saving results
        new_recipe_count: Total new recipes available (for time estimate)
    """
    
    # Get test URLs
    test_urls = get_test_urls(site_url, count)
    
    if not test_urls:
        return TestImportResults(
            success=False,
            total=0,
            succeeded=0,
            failed=0,
            skipped=0,
            recipes=[]
        )
    
    print(f"\n{'='*60}")
    print(f"TEST IMPORT: {len(test_urls)} recipes (batched pipeline)")
    print(f"{'='*60}")
    
    # Track timing
    start_time = time.time()
    
    # Use the ACTUAL batched pipeline - same as full imports
    # batch_size matches the test count since it's small
    try:
        state = batched_bulk_import(test_urls, job_id=job_id, batch_size=count)
    except Exception as e:
        # Return partial results on failure
        total_seconds = time.time() - start_time
        print(f"\n❌ Pipeline error: {e}")
        return TestImportResults(
            success=False,
            total=len(test_urls),
            succeeded=0,
            failed=len(test_urls),
            skipped=0,
            recipes=[{'url': u, 'status': 'failed', 'error': str(e)} for u in test_urls],
            total_seconds=total_seconds
        )
    
    # Calculate timing
    total_seconds = time.time() - start_time
    
    # Verify indexing: check that recipes are actually in RAG index
    print(f"\n🔍 Verifying RAG index...")
    try:
        from recipe_rag import RecipeRAG
        rag = RecipeRAG()
        rag_available = True
    except Exception as e:
        print(f"   ⚠️ RAG initialization failed: {e}")
        print(f"   Skipping index verification...")
        rag = None
        rag_available = False
    
    # Extract results from pipeline state and verify indexing
    succeeded = 0
    failed = 0
    index_verified = 0
    index_missing = 0
    recipes = []
    
    filtered = 0  # Correctly filtered non-recipes (not failures)
    
    for url, result in state.results.items():
        slug = result.slug
        status = result.status or 'unknown'
        
        # Verify recipe is actually in RAG index (no extra API call - local DB check)
        is_in_index = False
        if rag_available and rag and url:
            try:
                is_in_index = rag.has_url(normalize_url(url))
            except Exception:
                pass  # Skip verification on error
        
        # If state says indexed but not in RAG, that's a problem
        if status == 'indexed' and not is_in_index:
            print(f"   ⚠️ Index mismatch: {slug} marked indexed but not in RAG")
            index_missing += 1
        elif status == 'indexed' and is_in_index:
            index_verified += 1
        
        # Determine if this was a correctly-filtered non-recipe vs actual failure
        # Filtered = failed at parsing phase (no ingredients = not a recipe)
        is_filtered = (status == 'failed' and result.phase == 'parsing')
        
        recipe_info = {
            'url': url,
            'status': 'success' if status == 'indexed' else ('filtered' if is_filtered else 'failed'),
            'slug': slug,
            'name': result.name,  # Captured during tagging phase (no extra API call)
            'mealie_url': f"{MEALIE_URL}/g/home/r/{slug}" if slug else None,
            'error': result.error,
            'in_rag_index': is_in_index
        }
        recipes.append(recipe_info)
        
        if status == 'indexed':
            succeeded += 1
        elif is_filtered:
            filtered += 1  # Correctly filtered, not a failure
        else:
            failed += 1
    
    if index_verified > 0:
        print(f"   ✅ Verified {index_verified} recipes in RAG index")
    if index_missing > 0:
        print(f"   ❌ {index_missing} recipes missing from RAG index!")
    
    # Calculate metrics
    avg_seconds = total_seconds / len(test_urls) if test_urls else 0
    
    # Estimate full import time (based on succeeded recipes, not filtered garbage)
    estimated_full_time = None
    if new_recipe_count and succeeded > 0:
        # Adjust estimate: assume similar filter rate in full import
        filter_rate = filtered / len(test_urls) if test_urls else 0
        effective_recipes = new_recipe_count * (1 - filter_rate)
        estimated_seconds = avg_seconds * effective_recipes
        estimated_full_time = format_duration(estimated_seconds)
    
    # Success criteria: 
    # - All indexed recipes verified in RAG
    # - No actual failures (filtered non-recipes don't count as failures)
    # A test passes if: succeeded + filtered == total (no real failures)
    overall_success = (failed == 0) and (index_missing == 0)
    
    if filtered > 0:
        print(f"   🗑️ {filtered} non-recipes correctly filtered out")
    
    test_results = TestImportResults(
        success=overall_success,
        total=len(test_urls),
        succeeded=succeeded,
        failed=failed,
        filtered=filtered,
        skipped=0,
        recipes=recipes,
        total_seconds=total_seconds,
        avg_seconds_per_recipe=avg_seconds,
        estimated_full_import_time=estimated_full_time,
        index_verified=index_verified,
        index_missing=index_missing
    )
    
    # Save to JSON if job_id provided
    if job_id:
        try:
            results_dir = DATA_DIR / 'import_tests'
            results_dir.mkdir(parents=True, exist_ok=True)
            results_path = results_dir / f"{job_id}.json"
            
            with open(results_path, 'w') as f:
                json.dump(asdict(test_results), f, indent=2)
            
            print(f"\n📄 Results saved to: {results_path}")
        except Exception as e:
            print(f"\n⚠️ Warning: Failed to save results to file: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    if filtered > 0:
        print(f"TEST RESULTS: {succeeded} recipes imported, {filtered} non-recipes filtered")
    else:
        print(f"TEST RESULTS: {succeeded}/{len(test_urls)} recipes imported")
    if failed > 0:
        print(f"   ❌ {failed} actual failures")
    print(f"RAG INDEX: {index_verified}/{succeeded} verified" + (f" ⚠️ {index_missing} missing!" if index_missing else " ✅"))
    print(f"{'='*60}")
    
    print(f"\n⏱️ Timing:")
    print(f"   Total test time: {total_seconds:.1f} seconds")
    print(f"   Average per recipe: {avg_seconds:.1f} seconds")
    if estimated_full_time:
        print(f"   Estimated full import ({new_recipe_count} recipes): {estimated_full_time}")
    
    if overall_success:
        print(f"\n✅ Test passed - pipeline is working")
    else:
        if failed > 0:
            print(f"\n⚠️ Test failed - {failed} recipes failed to process")
        elif index_missing > 0:
            print(f"\n⚠️ Test failed - {index_missing} recipes missing from RAG index")
    
    # Print links for successful imports
    print(f"\n📎 View imported recipes:")
    for r in recipes:
        if r['status'] == 'success':
            display_name = r['name'] or r['slug']  # Fallback to slug if name not captured
            print(f"   • {display_name}: {r['mealie_url']}")
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description="Test import recipes from a site")
    parser.add_argument("url", help="Site URL to test import from")
    parser.add_argument("--count", "-n", type=int, default=10, help="Number of recipes to test (default: 10)")
    parser.add_argument("--job-id", help="Job ID for saving results")
    parser.add_argument("--new-count", type=int, help="Total new recipes available (for time estimate)")
    
    args = parser.parse_args()
    
    results = run_test_import(args.url, args.count, args.job_id, args.new_count)
    
    # Exit code based on success
    sys.exit(0 if results.success else 1)


if __name__ == "__main__":
    main()
