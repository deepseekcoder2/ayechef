#!/usr/bin/env python3
"""
Import Recipes from Website
============================

Entry point for importing recipes from a website into Mealie.
Supports category-based filtering for sites with category configuration.

Usage:
    python import_site.py https://thewoksoflife.com
    python import_site.py https://thewoksoflife.com --sitemap
    python import_site.py https://thewoksoflife.com --categories "Chinese,Japanese"
    python import_site.py https://www.budgetbytes.com --dry-run
"""

import sys
import os
import re
import argparse
import requests
import concurrent.futures
from urllib.parse import urlparse
from typing import List, Set, Dict
from xml.etree import ElementTree as ET

from config import DATA_DIR
from recipe_urls import scrape_urls, is_site_supported, get_supported_sites, SCRAPERS
from recipe_urls._utils import get_site_origin
from mealie_client import MealieClient
from tools.logging_utils import get_logger
from utils.url_utils import normalize_url

logger = get_logger(__name__)


def fetch_sitemap_urls(base_url: str) -> List[str]:
    """
    Fetch all URLs from a site's sitemap.
    Handles sitemap index files that point to multiple sitemaps.
    """
    parsed = urlparse(base_url)
    base_domain = f"{parsed.scheme}://{parsed.netloc}"
    
    # Common sitemap locations
    sitemap_urls_to_try = [
        f"{base_domain}/sitemap.xml",
        f"{base_domain}/sitemap_index.xml",
        f"{base_domain}/sitemap-index.xml",
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    all_urls = []
    
    def parse_sitemap(url: str) -> List[str]:
        """Parse a single sitemap or sitemap index."""
        try:
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Handle namespace
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            urls = []
            
            # Check if this is a sitemap index
            sitemap_refs = root.findall('.//sm:sitemap/sm:loc', ns)
            if sitemap_refs:
                print(f"  Found sitemap index with {len(sitemap_refs)} sitemaps")
                for ref in sitemap_refs:
                    sitemap_url = ref.text
                    # Process sitemaps that likely contain recipes:
                    # - post-sitemap (WordPress standard)
                    # - recipe in name
                    # - numbered sitemaps like sitemap_1.xml (common pattern)
                    # Skip category, author, tag, page sitemaps
                    skip_patterns = ['category', 'author', 'tag-sitemap', 'page-sitemap']
                    should_skip = any(p in sitemap_url.lower() for p in skip_patterns)
                    
                    if not should_skip:
                        print(f"    Processing: {sitemap_url}")
                        urls.extend(parse_sitemap(sitemap_url))
                return urls
            
            # Regular sitemap - extract URLs
            loc_elements = root.findall('.//sm:url/sm:loc', ns)
            for loc in loc_elements:
                urls.append(loc.text)
            
            return urls
            
        except Exception as e:
            logger.warning(f"Failed to parse sitemap {url}: {e}")
            return []
    
    # Try each sitemap location
    for sitemap_url in sitemap_urls_to_try:
        print(f"  Trying: {sitemap_url}")
        try:
            response = requests.head(sitemap_url, headers=headers, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                all_urls = parse_sitemap(response.url)  # Use final URL after redirects
                if all_urls:
                    break
        except:
            continue
    
    return all_urls


def filter_recipe_urls(urls: List[str], base_url: str) -> List[str]:
    """Filter URLs using the site's scraper patterns."""
    try:
        origin = get_site_origin(base_url)
        scraper_class = SCRAPERS.get(origin)
        
        if not scraper_class:
            # No scraper - return all URLs
            return urls
        
        # Use the scraper's patterns to filter
        recipe_pattern = scraper_class.RECIPE_PATTERN
        unwanted_patterns = scraper_class.UNWANTED_PATTERNS
        
        filtered = []
        for url in urls:
            # Check if matches recipe pattern
            if recipe_pattern and not recipe_pattern.search(url):
                continue
            
            # Check if matches any unwanted pattern
            if any(p.search(url) for p in unwanted_patterns):
                continue
            
            filtered.append(url)
        
        return filtered
        
    except Exception as e:
        logger.warning(f"Could not filter URLs: {e}")
        return urls


def get_existing_recipe_urls() -> Set[str]:
    """Fetch all existing recipe URLs from Mealie for deduplication.
    
    URLs are normalized for consistent comparison.
    """
    print("Checking existing recipes in Mealie...")
    client = MealieClient()
    try:
        existing_urls = client.get_all_recipe_urls()
        print(f"  Found {len(existing_urls)} existing recipes")
        return existing_urls
    finally:
        client.close()


def save_urls_to_file(urls: List[str], output_path: str) -> None:
    """Save URLs to text file for bulk_import_smart.py."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for url in sorted(urls):
            f.write(url + '\n')


def get_urls_by_categories(base_url: str, categories: List[str]) -> Dict[str, List[str]]:
    """
    Get recipe URLs from specified categories with per-category tracking.
    
    Args:
        base_url: Site base URL
        categories: List of category names to include
        
    Returns:
        Dict mapping category name to list of recipe URLs found
        Includes '_all' key with deduplicated list of all URLs
        Includes '_failed' key with list of failed category names
    """
    origin = get_site_origin(base_url)
    scraper_class = SCRAPERS.get(origin)
    
    if not scraper_class or not scraper_class.has_categories():
        return {'_all': [], '_failed': categories}
    
    results = {}
    failed_categories = []
    all_urls = []
    
    for category in categories:
        print(f"  Fetching category: {category}")
        try:
            category_urls = scraper_class.scrape_category_urls(category, base_url)
            results[category] = category_urls
            if category_urls:
                print(f"    ✅ Found {len(category_urls)} recipes")
                all_urls.extend(category_urls)
            else:
                print(f"    ⚠️  Found 0 recipes (empty or page structure changed)")
                failed_categories.append(category)
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            results[category] = []
            failed_categories.append(category)
            logger.warning(f"Failed to fetch category {category}: {e}")
    
    # Deduplicate (recipes might be in multiple categories)
    results['_all'] = list(set(all_urls))
    results['_failed'] = failed_categories
    
    return results


def validate_urls(urls: List[str], max_workers: int = 10) -> Dict[str, bool]:
    """
    Validate URLs using HEAD requests to check for 404s.
    
    Args:
        urls: List of URLs to validate
        max_workers: Maximum concurrent requests
        
    Returns:
        Dict mapping URL to validity (True = valid, False = 404 or error)
    """
    print(f"  Validating {len(urls)} URLs...")
    results = {}
    valid_count = 0
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    def check_url(url: str) -> tuple:
        try:
            response = requests.head(
                url,
                headers=headers,
                timeout=10,
                allow_redirects=True
            )
            return url, response.status_code < 400
        except:
            return url, False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_url, url) for url in urls]
        for future in concurrent.futures.as_completed(futures):
            url, valid = future.result()
            results[url] = valid
            if valid:
                valid_count += 1
    
    invalid_count = len(urls) - valid_count
    if invalid_count > 0:
        print(f"    {invalid_count} URLs returned 404 or error (will be skipped)")
    print(f"    {valid_count} URLs are valid")
    
    return results


def list_site_categories(url: str) -> None:
    """List available categories for a site."""
    origin = get_site_origin(url)
    scraper_class = SCRAPERS.get(origin)
    
    if not scraper_class:
        print(f"Site not supported: {url}")
        return
    
    if not scraper_class.has_categories():
        print(f"Site has no category configuration: {url}")
        print("Categories need to be configured in the scraper.")
        return
    
    categories = scraper_class.get_categories()
    print(f"\nAvailable categories for {scraper_class.host()}:")
    for cat in sorted(categories):
        print(f"  - {cat}")
    print(f"\nTotal: {len(categories)} categories")
    print("\nTo import specific categories:")
    print(f"  python import_site.py {url} --categories \"{','.join(categories[:3])}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Import recipes from a website into Mealie",
        epilog="""
Examples:
  python import_site.py https://thewoksoflife.com --sitemap
  python import_site.py https://thewoksoflife.com --categories "Chinese,Japanese"
  python import_site.py https://thewoksoflife.com --list-categories
  python import_site.py https://thewoksoflife.com --sitemap --collect-only  # Just save URLs
        """
    )
    parser.add_argument("url", help="Website URL to scrape recipes from")
    parser.add_argument("--output", "-o", help="Output file path (default: data/imports/<domain>_recipes.txt)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without importing")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication against Mealie")
    parser.add_argument("--sitemap", action="store_true", help="Use sitemap for full site coverage (recommended)")
    parser.add_argument("--list-sites", action="store_true", help="List supported sites and exit")
    parser.add_argument("--list-categories", action="store_true", help="List available categories for the site")
    parser.add_argument("--categories", help="Comma-separated list of categories to import (e.g., 'Chinese,Japanese')")
    parser.add_argument("--validate-urls", action="store_true", help="Check URLs for 404s before importing")
    parser.add_argument("--collect-only", action="store_true", help="Only collect URLs, don't import (saves to file)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()

    # List supported sites
    if args.list_sites:
        print("Supported sites:")
        for site in sorted(get_supported_sites()):
            print(f"  - {site}")
        return

    # List categories for the site
    if args.list_categories:
        list_site_categories(args.url)
        return

    # Extract domain for output filename
    parsed = urlparse(args.url)
    domain = parsed.netloc.replace('www.', '').replace('.', '_')
    
    # Check if site is supported (for filtering)
    site_supported = is_site_supported(args.url)
    if not site_supported:
        print(f"\n  Site not supported: {args.url}")
        print(f"\n  Options:")
        print(f"    1. Add support: python add_site.py {args.url}")
        print(f"\n  Currently supported sites:")
        for site in sorted(get_supported_sites()):
            print(f"    - {site}")
        sys.exit(1)

    # Parse categories if provided
    selected_categories = []
    if args.categories:
        selected_categories = [c.strip() for c in args.categories.split(',')]
        print(f"\nImporting from categories: {', '.join(selected_categories)}")

    # Scrape URLs
    category_results = None  # Track per-category results for reporting
    if selected_categories:
        # Category-based import
        print(f"\nFetching recipes by category from {args.url}...")
        category_results = get_urls_by_categories(args.url, selected_categories)
        urls = category_results['_all']
        failed_cats = category_results.get('_failed', [])
        success_cats = [c for c in selected_categories if c not in failed_cats]
        
        print(f"\n  Category Summary:")
        print(f"    ✅ Successful: {len(success_cats)}/{len(selected_categories)} categories")
        if failed_cats:
            print(f"    ❌ Failed: {len(failed_cats)} categories")
            for cat in failed_cats[:5]:
                count = len(category_results.get(cat, []))
                print(f"       - {cat}: {count} recipes")
            if len(failed_cats) > 5:
                print(f"       ... and {len(failed_cats) - 5} more")
        print(f"  Total recipes found: {len(urls)}")
    elif args.sitemap:
        print(f"\nFetching recipes from {args.url} sitemap...")
        try:
            urls = fetch_sitemap_urls(args.url)
            print(f"  Raw URLs from sitemap: {len(urls)}")
            urls = filter_recipe_urls(urls, args.url)
            print(f"  After filtering: {len(urls)} recipe URLs")
            # Deduplicate URLs (sitemaps can have duplicates)
            urls = list(dict.fromkeys(urls))  # Preserves order, removes duplicates
            print(f"  After deduplication: {len(urls)} unique URLs")
        except Exception as e:
            print(f"  Error: {e}")
            sys.exit(1)
    else:
        print(f"\nScraping recipes from {args.url} homepage...")
        print("  (Use --sitemap for full site coverage)")
        print("  (Use --categories for category-based import)")
        try:
            urls = scrape_urls(args.url)
        except Exception as e:
            print(f"  Error: {e}")
            sys.exit(1)
    
    print(f"  Found {len(urls)} recipe URLs")

    if not urls:
        print("  No recipes found. The site structure may have changed.")
        sys.exit(1)

    # Validate URLs if requested
    if args.validate_urls:
        url_validity = validate_urls(urls)
        urls = [u for u in urls if url_validity.get(u, False)]
        print(f"  After validation: {len(urls)} valid URLs")

    # Deduplicate against Mealie
    new_urls = urls
    duplicates = []
    
    if not args.no_dedup:
        try:
            existing = get_existing_recipe_urls()
            # Use normalized URLs for comparison, but keep original for import
            new_urls = [u for u in urls if normalize_url(u) not in existing]
            duplicates = [u for u in urls if normalize_url(u) in existing]
            print(f"  Skipped {len(duplicates)} duplicates (already in Mealie)")
        except Exception as e:
            print(f"  Warning: Could not check Mealie for duplicates: {e}")
            print("  Proceeding with all URLs...")

    # Output path - include categories in filename if specified
    if selected_categories:
        cat_suffix = '_'.join(c.replace(' ', '').replace('-', '')[:10] for c in selected_categories[:3])
        output_path = args.output or str(DATA_DIR / "imports" / f"{domain}_{cat_suffix}_recipes.txt")
    else:
        output_path = args.output or str(DATA_DIR / "imports" / f"{domain}_recipes.txt")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"DISCOVERY SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total found:    {len(urls)}")
    print(f"  Duplicates:     {len(duplicates)}")
    print(f"  New recipes:    {len(new_urls)}")
    if selected_categories and category_results:
        # Show which categories actually contributed recipes
        failed_cats = category_results.get('_failed', [])
        success_cats = [c for c in selected_categories if c not in failed_cats and len(category_results.get(c, [])) > 0]
        print(f"  Categories with recipes: {len(success_cats)}/{len(selected_categories)}")
        if success_cats:
            print(f"  Contributing categories:")
            for cat in success_cats[:5]:
                count = len(category_results.get(cat, []))
                print(f"    ✅ {cat}: {count} recipes")
            if len(success_cats) > 5:
                print(f"    ... and {len(success_cats) - 5} more")

    if args.dry_run:
        print(f"\n  [DRY RUN] Would import {len(new_urls)} recipes")
        print(f"\n  Sample URLs:")
        for url in new_urls[:10]:
            print(f"    - {url}")
        if len(new_urls) > 10:
            print(f"    ... and {len(new_urls) - 10} more")
        return

    if not new_urls:
        print("\n  No new recipes to import.")
        return

    # Always save URLs to file for reference/resume capability
    save_urls_to_file(new_urls, output_path)
    print(f"  Saved URLs to: {output_path}")

    # If --collect-only, stop here
    if args.collect_only:
        print(f"\n{'=' * 50}")
        print(f"COLLECTION COMPLETE (--collect-only mode)")
        print(f"{'=' * 50}")
        print(f"  To import these recipes, run:")
        print(f"    python bulk_import_smart.py {output_path} --yes")
        return

    # Otherwise, run the actual import
    print(f"\n{'=' * 50}")
    print(f"STARTING IMPORT")
    print(f"{'=' * 50}")
    print(f"  Importing {len(new_urls)} recipes into Mealie...")
    print(f"  This will take approximately {estimate_import_time(len(new_urls))}")
    print()

    # Run bulk_import_smart.py with the saved file
    import subprocess
    cmd = [sys.executable, "bulk_import_smart.py", output_path, "--yes"]
    
    try:
        # Run with real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n{'=' * 50}")
            print(f"✅ IMPORT COMPLETE")
            print(f"{'=' * 50}")
        else:
            print(f"\n{'=' * 50}")
            print(f"❌ IMPORT FAILED (exit code: {process.returncode})")
            print(f"{'=' * 50}")
            sys.exit(process.returncode)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Import interrupted by user")
        print(f"  To resume, run:")
        print(f"    python bulk_import_smart.py {output_path} --yes")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed to run import: {e}")
        print(f"  To retry manually, run:")
        print(f"    python bulk_import_smart.py {output_path} --yes")
        sys.exit(1)


def estimate_import_time(num_recipes: int) -> str:
    """Estimate import time based on recipe count."""
    # Rough estimate: ~3 seconds per recipe (import + parse + tag + index)
    # With parallelism, effective rate is about 1 recipe per second
    seconds = num_recipes * 1.5
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''}"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


if __name__ == "__main__":
    main()
