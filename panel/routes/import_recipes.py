"""Import Recipes routes - bulk recipe import from websites.

Supports:
- Single recipe import
- Bulk import with category selection
- Category discovery for supported sites
- SSE progress streaming for jobs
"""
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, Response, stream_with_context
from panel.tools.registry import TOOLS
from panel.jobs import create_job, run_tool_job, list_jobs, get_job, get_output
from recipe_urls import (
    get_supported_sites, detect_url_type, get_site_categories,
    get_scraper_for_url, is_site_supported, reload_scrapers
)
from config import MEALIE_URL, MEALIE_TOKEN, get_mealie_headers
import requests
import uuid
import concurrent.futures
import json
import re
import time
from urllib.parse import urlparse

bp = Blueprint('import_recipes', __name__)

# Estimated recipe counts for known sites (rough approximations)
SITE_INFO = {
    'thewoksoflife.com': {'count': '1,200+', 'time': '4-6 hours'},
    'budgetbytes.com': {'count': '1,500+', 'time': '6-8 hours'},
    'bbc.co.uk': {'count': '10,000+', 'time': '3-5 days'},
    'seriouseats.com': {'count': '5,000+', 'time': '1-2 days'},
}


@bp.route('/', strict_slashes=False)
def index():
    """Import Recipes page - single input for any URL."""
    # Reload scrapers to pick up newly learned sites
    reload_scrapers()
    
    # Get supported sites
    supported_sites = sorted(get_supported_sites())
    
    # Get all import jobs (active and recent)
    all_jobs = list_jobs(limit=50)
    import_jobs = [j for j in all_jobs if j.tool_id in ('import_site', 'import_recipe')]
    active_imports = [j for j in import_jobs if j.status in ('pending', 'running')]
    recent_imports = [j for j in import_jobs if j.status not in ('pending', 'running')][:10]
    
    return render_template('import/index.html',
                          supported_sites=supported_sites,
                          site_info=SITE_INFO,
                          active_imports=active_imports,
                          recent_imports=recent_imports)


@bp.route('/detect', methods=['POST'])
def detect():
    """Detect URL type and return appropriate response with category info."""
    url = request.form.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Add protocol if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    # Reload scrapers to pick up newly learned sites
    reload_scrapers()
    
    try:
        detection = detect_url_type(url)
        detection['url'] = url
        
        # Add site info if available
        parsed = urlparse(url)
        hostname = parsed.netloc.lower().replace('www.', '')
        if hostname in SITE_INFO:
            detection['site_info'] = SITE_INFO[hostname]
        
        return jsonify(detection)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'url': url,
            'type': 'unknown',
            'supported': False,
            'has_categories': False,
            'categories': [],
            'host': urlparse(url).netloc
        })


@bp.route('/categories', methods=['POST'])
def get_categories():
    """
    Get available categories for a site.
    
    Returns:
        JSON with:
        - supported: bool
        - has_categories: bool  
        - categories: list of {name, sample_count} objects
        - uses_category_pages: bool
    """
    url = request.form.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Add protocol if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    # Reload scrapers to pick up newly learned sites
    reload_scrapers()
    
    # Get category info
    cat_info = get_site_categories(url)
    
    if not cat_info['supported']:
        return jsonify({
            'supported': False,
            'error': 'Site is not supported. Use add_site.py to add support.',
            'has_categories': False,
            'categories': []
        })
    
    # Format categories with names
    categories = []
    for name in cat_info['categories']:
        categories.append({
            'name': name,
            'selected': False  # Default to not selected
        })
    
    return jsonify({
        'supported': True,
        'has_categories': cat_info['has_categories'],
        'categories': categories,
        'uses_category_pages': cat_info['uses_category_pages']
    })


@bp.route('/category-preview', methods=['POST'])
def category_preview():
    """
    Get sample recipe URLs for a category (for preview).
    
    This scrapes the category page to get sample recipes.
    """
    url = request.form.get('url', '').strip()
    category = request.form.get('category', '').strip()
    
    if not url or not category:
        return jsonify({'error': 'URL and category required'}), 400
    
    # Add protocol if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    scraper_class = get_scraper_for_url(url)
    if not scraper_class:
        return jsonify({'error': 'Site not supported'}), 400
    
    if not scraper_class.has_categories():
        return jsonify({'error': 'Site has no category configuration'}), 400
    
    try:
        # Get recipe URLs from category page
        recipe_urls = scraper_class.scrape_category_urls(category, url)
        
        # Return sample (first 10) and total count
        return jsonify({
            'category': category,
            'total_count': len(recipe_urls),
            'sample_urls': recipe_urls[:10],
            'sample_names': [u.split('/')[-2] or u.split('/')[-1] for u in recipe_urls[:10]]
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch category: {str(e)}'}), 500


@bp.route('/category-counts', methods=['POST'])
def category_counts():
    """
    Fetch recipe counts for all categories in parallel.
    
    Returns counts for each category so users can make informed decisions.
    """
    url = request.form.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Add protocol if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    scraper_class = get_scraper_for_url(url)
    if not scraper_class:
        return jsonify({'error': 'Site not supported'}), 400
    
    if not scraper_class.has_categories():
        return jsonify({'error': 'Site has no category configuration'}), 400
    
    categories = scraper_class.get_categories()
    
    def fetch_category_count(category):
        """Fetch count for a single category."""
        try:
            recipe_urls = scraper_class.scrape_category_urls(category, url)
            return {'category': category, 'count': len(recipe_urls), 'error': None}
        except Exception as e:
            return {'category': category, 'count': 0, 'error': str(e)}
    
    # Fetch all categories in parallel (limit workers to avoid overwhelming the site)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_category_count, cat): cat for cat in categories}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[result['category']] = {
                'count': result['count'],
                'error': result['error']
            }
    
    return jsonify({
        'url': url,
        'categories': results
    })


@bp.route('/recipe', methods=['POST'])
def import_single_recipe():
    """Import a single recipe URL with full processing pipeline."""
    url = request.form.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Create a job to run the full import pipeline
    from urllib.parse import urlparse
    parsed = urlparse(url)
    recipe_name = parsed.path.split('/')[-1][:30] or 'recipe'
    
    job_id = str(uuid.uuid4())
    create_job(job_id, 'import_recipe', f"Import: {recipe_name}")
    
    form_data = {'url': url}
    run_tool_job(job_id, 'import_recipe', form_data)
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'redirect': f'/jobs/{job_id}'
    })


@bp.route('/start', methods=['POST'])
def start():
    """
    Start a bulk recipe import job.
    
    Accepts:
    - url: Site URL (required)
    - sitemap: Use sitemap for discovery
    - categories[]: List of category names to import (optional)
    - validate_urls: Check URLs for 404s before import (optional)
    """
    url = request.form.get('url', '').strip()
    use_sitemap = request.form.get('sitemap')
    categories = request.form.getlist('categories[]') or request.form.getlist('categories')
    validate_urls = request.form.get('validate_urls')
    
    if not url:
        return redirect(url_for('import_recipes.index'))
    
    # Add protocol if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    # Reload scrapers to pick up newly learned sites
    reload_scrapers()
    
    # Block unsupported sites from bulk import
    if not is_site_supported(url):
        # Return error for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'error': 'Site is not supported for bulk import',
                'message': 'Use add_site.py to add support for this site first.'
            }), 400
        # Redirect for form submissions
        return redirect(url_for('import_recipes.index'))
    
    # Extract domain for job name
    domain = urlparse(url).netloc or url[:30]
    
    # Create descriptive job name
    if categories:
        job_name = f"Import: {domain} ({len(categories)} categories)"
    else:
        job_name = f"Import: {domain} (all recipes)"
    
    job_id = str(uuid.uuid4())
    create_job(job_id, 'import_site', job_name)
    
    form_data = {'url': url}
    if use_sitemap:
        form_data['sitemap'] = 'on'
    if categories:
        form_data['categories'] = ','.join(categories)
    if validate_urls:
        form_data['validate_urls'] = 'on'
    
    run_tool_job(job_id, 'import_site', form_data)
    
    # Return JSON for AJAX requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': True,
            'job_id': job_id,
            'redirect': f'/jobs/{job_id}'
        })
    
    return redirect(url_for('jobs.job_detail', job_id=job_id))


def validate_urls_batch(urls: list, max_workers: int = 10) -> dict:
    """
    Validate URLs by checking for 404s using HEAD requests.
    
    Args:
        urls: List of URLs to validate
        max_workers: Maximum concurrent requests
        
    Returns:
        Dict mapping URL to (valid: bool, status_code: int)
    """
    results = {}
    
    def check_url(url: str) -> tuple:
        try:
            response = requests.head(
                url,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=10,
                allow_redirects=True
            )
            return url, response.status_code < 400, response.status_code
        except:
            return url, False, 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_url, url) for url in urls]
        for future in concurrent.futures.as_completed(futures):
            url, valid, status = future.result()
            results[url] = {'valid': valid, 'status_code': status}
    
    return results


@bp.route('/learn-site', methods=['POST'])
def learn_site():
    """
    Start a job to learn/analyze a new site.
    
    This runs the add_site analysis to create a scraper configuration.
    """
    url = request.form.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Add protocol if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    # Reload scrapers to check current state
    reload_scrapers()
    
    # Check if already supported
    if is_site_supported(url):
        return jsonify({
            'error': 'Site is already supported',
            'supported': True
        }), 400
    
    # Extract domain for job name
    parsed = urlparse(url)
    hostname = parsed.netloc.replace('www.', '')
    
    job_id = str(uuid.uuid4())
    create_job(job_id, 'add_site', f"Learn: {hostname}")
    
    form_data = {'url': url}
    run_tool_job(job_id, 'add_site', form_data)
    
    return jsonify({
        'success': True,
        'job_id': job_id
    })


@bp.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze a site for import - returns recipe counts and samples.
    
    This is SYNCHRONOUS. Frontend should show a loading spinner.
    """
    url = request.form.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Add protocol if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    # Reload scrapers to pick up newly learned sites
    reload_scrapers()
    
    try:
        from panel.import_wizard import analyze_site
        result = analyze_site(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'supported': False
        }), 500


@bp.route('/refresh-counts', methods=['POST'])
def refresh_counts():
    """
    Fast refresh of Mealie-side counts only.
    
    Used when we have cached site data but need fresh Mealie counts.
    Returns only the "already_imported" count for the given host.
    
    This is FAST - just queries Mealie API, no external scraping.
    """
    host = request.form.get('host', '').strip()
    
    if not host:
        return jsonify({'error': 'No host provided'}), 400
    
    # Normalize host (remove www. prefix)
    host = host.lower().replace('www.', '')
    
    try:
        from import_site import get_existing_recipe_urls, normalize_url
        
        # Get all existing recipe URLs from Mealie
        existing_urls = get_existing_recipe_urls()
        
        # Count how many are from this host
        count = 0
        for url in existing_urls:
            try:
                parsed = urlparse(url)
                url_host = parsed.netloc.lower().replace('www.', '')
                if url_host == host:
                    count += 1
            except:
                continue
        
        return jsonify({
            'host': host,
            'already_imported': count
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'host': host,
            'already_imported': 0
        }), 500


@bp.route('/test', methods=['POST'])
def start_test():
    """
    Start a test import job (10 recipes with full pipeline).
    
    Returns job_id to poll for completion.
    """
    url = request.form.get('url', '').strip()
    new_count = request.form.get('new_count', '').strip()  # Total new recipes for time estimate
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Add protocol if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    # Reload scrapers
    reload_scrapers()
    
    # Check if supported
    if not is_site_supported(url):
        return jsonify({'error': 'Site not supported'}), 400
    
    # Create job
    parsed = urlparse(url)
    hostname = parsed.netloc.replace('www.', '')
    
    job_id = str(uuid.uuid4())
    create_job(job_id, 'test_import', f"Test: {hostname} (10 recipes)")
    
    form_data = {
        'url': url,
        'count': '10',
        'job_id': job_id
    }
    if new_count:
        form_data['new_count'] = new_count
    
    run_tool_job(job_id, 'test_import', form_data)
    
    return jsonify({
        'success': True,
        'job_id': job_id
    })


@bp.route('/test/<job_id>/results', methods=['GET'])
def get_test_results(job_id):
    """
    Get structured results from a test import job.
    
    Returns JSON with success status and individual recipe results.
    """
    from pathlib import Path
    
    from config import DATA_DIR
    results_path = DATA_DIR / "import_tests" / f"{job_id}.json"
    
    if not results_path.exists():
        # Check if job is still running
        job = get_job(job_id)
        if job and job.status == 'running':
            return jsonify({'status': 'running'}), 202
        elif job and job.status == 'pending':
            return jsonify({'status': 'pending'}), 202
        else:
            return jsonify({'error': 'Results not found'}), 404
    
    with open(results_path) as f:
        results = json.load(f)
    
    return jsonify(results)


def parse_progress_line(line: str) -> dict | None:
    """
    Parse a single output line and extract progress information.
    
    Recognizes patterns like:
    - "Progress: 45% (45/100)"
    - "Progress: 100/100 complete"
    - "[45/100] 📥 Importing: https://example.com/recipe-name"
    - "Completed: 95/100 items in 123.4s"
    - Phase markers like "🚀 Importing", "🏷️ Tagging", "📊 Indexing"
    
    Returns:
        Dict with progress info or None if line doesn't contain progress.
    """
    # Pattern: "Progress: 45% (45/100)"
    match = re.search(r'Progress:\s*(\d+)%\s*\((\d+)/(\d+)\)', line)
    if match:
        percent = int(match.group(1))
        completed = int(match.group(2))
        total = int(match.group(3))
        return {
            'phase': 'importing',
            'completed': completed,
            'total': total,
            'percent': percent
        }
    
    # Pattern: "Progress: 100/100 complete"
    match = re.search(r'Progress:\s*(\d+)/(\d+)\s*complete', line)
    if match:
        completed = int(match.group(1))
        total = int(match.group(2))
        percent = int(completed / total * 100) if total > 0 else 0
        return {
            'phase': 'importing',
            'completed': completed,
            'total': total,
            'percent': percent
        }
    
    # Pattern: "[45/100] 📥 Importing: URL"
    match = re.search(r'\[(\d+)/(\d+)\]\s*📥\s*Importing:\s*(.+)', line)
    if match:
        completed = int(match.group(1))
        total = int(match.group(2))
        url = match.group(3).strip()
        # Extract recipe name from URL
        recipe_name = url.rstrip('/').split('/')[-1].replace('-', ' ').title()[:50]
        percent = int(completed / total * 100) if total > 0 else 0
        return {
            'phase': 'importing',
            'completed': completed,
            'total': total,
            'percent': percent,
            'current_recipe': recipe_name
        }
    
    # Pattern: "Completed: 95/100 items in 123.4s"
    match = re.search(r'Completed:\s*(\d+)/(\d+)\s*items?\s*in\s*([\d.]+)s', line)
    if match:
        completed = int(match.group(1))
        total = int(match.group(2))
        elapsed = float(match.group(3))
        percent = int(completed / total * 100) if total > 0 else 100
        return {
            'phase': 'completed',
            'completed': completed,
            'total': total,
            'percent': percent,
            'elapsed_seconds': elapsed
        }
    
    # Phase markers
    if '🚀' in line and ('Import' in line or 'import' in line):
        # Extract total from patterns like "📊 Processing 100 items"
        match = re.search(r'(\d+)\s*items?', line)
        total = int(match.group(1)) if match else 0
        return {
            'phase': 'starting',
            'message': line.strip()[:100],
            'total': total
        }
    
    if '🏷️' in line or 'Tagging' in line or 'tagging' in line:
        return {
            'phase': 'tagging',
            'message': line.strip()[:100]
        }
    
    if '📊' in line and ('Index' in line or 'index' in line):
        return {
            'phase': 'indexing',
            'message': line.strip()[:100]
        }
    
    # Pattern: Status messages with counts "✅ Successfully imported X recipes"
    match = re.search(r'✅.*?(\d+)\s*recipes?', line, re.IGNORECASE)
    if match:
        return {
            'phase': 'summary',
            'success_count': int(match.group(1)),
            'message': line.strip()[:100]
        }
    
    # Pattern: Error count "❌ Failed: X recipes"
    match = re.search(r'❌.*?(\d+)\s*(?:recipes?|failed|errors?)', line, re.IGNORECASE)
    if match:
        return {
            'phase': 'summary',
            'failed_count': int(match.group(1)),
            'message': line.strip()[:100]
        }
    
    return None


@bp.route('/api/jobs/<job_id>/progress')
def job_progress_stream(job_id):
    """
    SSE endpoint for real-time job progress.
    
    Streams progress updates as Server-Sent Events while the job is running.
    
    Events:
    - progress: Current progress with phase, completed, total, percent
    - complete: Job finished (success or partial success)
    - error: Job failed or not found
    
    Example event:
        event: progress
        data: {"phase": "importing", "completed": 45, "total": 100, "percent": 45}
    """
    def generate():
        try:
            # Check if job exists
            job = get_job(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
                return
            
            # Tell clients to retry after 5 seconds if connection drops
            yield "retry: 5000\n\n"
            
            # Track what we've already sent
            last_line_count = 0
            last_progress = None
            poll_count = 0
            max_polls = 3600  # Max 1 hour of polling (at 1s intervals)
            
            while poll_count < max_polls:
                poll_count += 1
                
                # Get current job status
                job = get_job(job_id)
                if not job:
                    yield f"event: error\ndata: {json.dumps({'message': 'Job disappeared'})}\n\n"
                    return
                
                # Read output and parse new lines
                output = get_output(job_id)
                lines = output.splitlines()
                
                # Process only new lines
                new_lines = lines[last_line_count:]
                last_line_count = len(lines)
                
                # Parse each new line for progress
                for line in new_lines:
                    progress = parse_progress_line(line)
                    if progress and progress != last_progress:
                        last_progress = progress
                        yield f"event: progress\ndata: {json.dumps(progress)}\n\n"
                
                # Check terminal states
                if job.status == 'completed':
                    # Extract final stats from output
                    total_imported = 0
                    failed = 0
                    for line in lines:
                        match = re.search(r'Completed:\s*(\d+)/(\d+)', line)
                        if match:
                            total_imported = int(match.group(1))
                            failed = int(match.group(2)) - total_imported
                    
                    yield f"event: complete\ndata: {json.dumps({'success': True, 'total_imported': total_imported, 'failed': failed, 'status': 'completed'})}\n\n"
                    return
                
                elif job.status == 'failed':
                    error_msg = job.error or 'Job failed'
                    yield f"event: error\ndata: {json.dumps({'message': error_msg})}\n\n"
                    return
                
                elif job.status == 'cancelled':
                    yield f"event: complete\ndata: {json.dumps({'success': False, 'status': 'cancelled', 'message': 'Job was cancelled'})}\n\n"
                    return
                
                # Send keepalive comment every 15 seconds to prevent proxy timeouts
                if poll_count % 15 == 0:
                    yield ":keepalive\n\n"
                
                # Poll interval
                time.sleep(1)
            
            # Timeout
            yield f"event: error\ndata: {json.dumps({'message': 'Progress stream timed out'})}\n\n"
        
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Stream error: {str(e)}'})}\n\n"
            return
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
            'Connection': 'keep-alive'
        }
    )
