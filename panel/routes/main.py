"""Main routes - single page control panel."""
from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from panel.tools.registry import TOOLS, get_maintenance_tools_by_group
from panel.jobs import create_job, run_tool_job, list_jobs, force_cancel_all_active
from panel.health_checks import get_recipe_quality_quick, is_ready_for_planning
from config import CONFIG_PATH, DATA_DIR, get_credential_status, save_secrets, reload_credentials
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import yaml
import requests

bp = Blueprint('main', __name__)


def get_config():
    """Load config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return None


def save_config(config: dict):
    """Save config to data/config.yaml."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def check_mealie():
    """Check Mealie connection."""
    config = get_config()
    if not config:
        return {'connected': False, 'url': None}
    
    url = config.get('connection', {}).get('mealie_url', '')
    if not url:
        return {'connected': False, 'url': None}
    
    try:
        response = requests.get(f"{url}/api/app/about", timeout=3)
        if response.status_code == 200:
            return {'connected': True, 'url': url}
    except:
        pass
    return {'connected': False, 'url': url}


def get_next_monday():
    """Get next Monday's date."""
    today = datetime.now()
    days_ahead = (7 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')


def get_last_job(tool_id):
    """Get the most recent job for a tool."""
    jobs = list_jobs(limit=50)
    for job in jobs:
        if job.tool_id == tool_id:
            return job
    return None


@bp.route('/')
def index():
    """Single page control panel."""
    config = get_config()
    
    # First run - no config
    if not config:
        return render_template('index.html', 
                             first_run=True,
                             mealie={},
                             config=None)
    
    mealie = check_mealie()
    last_plan = get_last_job('plan_week')
    last_import = get_last_job('import_site')
    
    # Check for active jobs
    jobs = list_jobs(limit=10)
    active_job = next((j for j in jobs if j.status in ('pending', 'running')), None)
    
    # Check recipe quality (quick check - local DB only)
    recipe_quality = get_recipe_quality_quick() if mealie.get('connected') else None
    
    # Check if system is ready for meal planning
    planning_ready = is_ready_for_planning()
    
    # Get maintenance tools organized by group
    maintenance_groups = get_maintenance_tools_by_group()
    
    # Check if backfill_images should be shown (requires Brave API key)
    from config import load_brave_api_key
    has_brave_api = bool(load_brave_api_key())
    
    # Filter out backfill_images from routine if no Brave API key configured
    if not has_brave_api:
        maintenance_groups['routine'] = [
            t for t in maintenance_groups['routine'] 
            if t.id != 'backfill_images'
        ]
    
    # Get credential status for settings display
    credentials = get_credential_status()
    
    return render_template('index.html',
                          first_run=False,
                          mealie=mealie,
                          config=config,
                          next_monday=get_next_monday(),
                          last_plan=last_plan,
                          last_import=last_import,
                          active_job=active_job,
                          recipe_quality=recipe_quality,
                          planning_ready=planning_ready,
                          maintenance_groups=maintenance_groups,
                          credentials=credentials)


@bp.route('/plan', methods=['POST'])
def plan():
    """Start meal planning job."""
    start_date = request.form.get('start_date') or get_next_monday()
    dry_run = request.form.get('dry_run')
    cuisines = request.form.get('cuisines', '').strip()
    restrictions = request.form.get('restrictions', '').strip()
    temp_prompt = request.form.get('temp_prompt', '').strip()
    
    job_id = str(uuid.uuid4())
    create_job(job_id, 'plan_week', f"Meal Plan: {start_date}")
    
    form_data = {'start_date': start_date}
    if dry_run:
        form_data['dry_run'] = 'on'
    if cuisines:
        form_data['cuisines'] = cuisines
    if restrictions:
        form_data['restrictions'] = restrictions
    if temp_prompt:
        form_data['temp_prompt'] = temp_prompt
    
    run_tool_job(job_id, 'plan_week', form_data)
    return redirect(url_for('jobs.job_detail', job_id=job_id))


@bp.route('/import', methods=['POST'])
def import_recipes():
    """Start import job."""
    url = request.form.get('url', '').strip()
    if not url:
        return redirect(url_for('main.index'))
    
    use_sitemap = request.form.get('sitemap')
    
    from urllib.parse import urlparse
    domain = urlparse(url).netloc or url[:30]
    
    job_id = str(uuid.uuid4())
    create_job(job_id, 'import_site', f"Import: {domain}")
    
    form_data = {'url': url}
    if use_sitemap:
        form_data['sitemap'] = 'on'
    
    run_tool_job(job_id, 'import_site', form_data)
    return redirect(url_for('jobs.job_detail', job_id=job_id))


@bp.route('/maintenance/<tool_id>', methods=['POST'])
def maintenance(tool_id):
    """Run maintenance tool."""
    tool = TOOLS.get(tool_id)
    if not tool:
        return "Tool not found", 404
    
    job_id = str(uuid.uuid4())
    create_job(job_id, tool_id, tool.name)
    run_tool_job(job_id, tool_id, dict(request.form))
    return redirect(url_for('jobs.job_detail', job_id=job_id))


@bp.route('/settings', methods=['POST'])
def save_settings():
    """Save settings."""
    from panel.themes import validate_theme_id
    import copy
    
    # Load existing config to preserve sections not in the form
    existing_config = get_config() or {}
    
    # Get theme preference
    theme_id = request.form.get('theme', 'morning-light')
    theme_id = validate_theme_id(theme_id)

    def _ensure_dict(root: dict, key: str) -> dict:
        if not isinstance(root.get(key), dict):
            root[key] = {}
        return root[key]

    from typing import Optional

    def _to_int(value: str, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
        try:
            n = int(str(value).strip())
        except Exception:
            n = default
        if min_value is not None:
            n = max(min_value, n)
        if max_value is not None:
            n = min(max_value, n)
        return n
    
    # Start from existing config to preserve unknown keys/sections.
    # This avoids clobbering advanced settings not present in the form.
    config = copy.deepcopy(existing_config) if isinstance(existing_config, dict) else {}

    # Core settings
    _ensure_dict(config, "ui")["theme"] = theme_id
    _ensure_dict(config, "connection")["mealie_url"] = request.form.get('mealie_url', '').rstrip('/')
    _ensure_dict(config, "mealie")["use_direct_db"] = request.form.get('data_mode') == 'db'

    config["household"] = {
        'servings': _to_int(request.form.get('servings', 4), 4, min_value=1, max_value=100),
        'meal_types': request.form.getlist('meal_types') or ['lunch', 'dinner'],
        'description': request.form.get('description', ''),
    }

    config["preferences"] = {
        'cuisines': [c.strip() for c in request.form.get('cuisines', '').split(',') if c.strip()],
        'dietary_restrictions': [r.strip() for r in request.form.get('restrictions', '').split(',') if r.strip()],
    }

    config["personal"] = {
        'dietary': [d.strip() for d in request.form.get('personal_dietary', '').split('\n') if d.strip()],
        'cooking': request.form.get('personal_cooking', '').strip(),
    }

    config["pantry"] = {
        'staples': [s.strip() for s in request.form.get('staples', '').split('\n') if s.strip()],
    }

    config["llm"] = {
        'chat_model': request.form.get('chat_model', 'openai/gpt-4o-mini'),
        'embedding_model': request.form.get('embedding_model', 'nvidia/llama-embed-nemotron-8b'),
    }

    _ensure_dict(config, "image_search")["use_vision_validation"] = request.form.get('vision_validation') == 'on'

    # AI Tuning (optional)
    max_protein = _to_int(request.form.get('max_protein_repeats_per_week', 2), 2, min_value=1, max_value=14)
    max_streak = _to_int(request.form.get('max_consecutive_same_cuisine', 3), 3, min_value=1, max_value=14)
    _ensure_dict(_ensure_dict(config, "meal_planning"), "variety")
    config["meal_planning"]["variety"]["max_protein_repetitions_per_week"] = max_protein
    config["meal_planning"]["variety"]["max_consecutive_same_cuisine"] = max_streak

    always_exclude = [
        s.strip() for s in request.form.get('shopping_always_exclude', '').split('\n') if s.strip()
    ]
    _ensure_dict(_ensure_dict(config, "shopping"), "exclusions")
    config["shopping"]["exclusions"]["always_exclude"] = always_exclude

    # Reset AI tuning to defaults (only affects AI tuning keys)
    if request.form.get("reset_ai_tuning"):
        try:
            if isinstance(config.get("meal_planning"), dict) and isinstance(config["meal_planning"].get("variety"), dict):
                config["meal_planning"].pop("variety", None)
                if not config["meal_planning"]:
                    config.pop("meal_planning", None)
            if isinstance(config.get("shopping"), dict) and isinstance(config["shopping"].get("exclusions"), dict):
                config["shopping"].pop("exclusions", None)
                if not config["shopping"]:
                    config.pop("shopping", None)
        except Exception:
            # Best-effort reset; settings save should still proceed
            pass
    
    save_config(config)

    # Reload runtime config so prompt builders see new values without restart.
    try:
        from config import reload_user_config
        reload_user_config()
    except Exception:
        # If reload fails, the saved config will still apply on next restart.
        pass
    return redirect(url_for('main.index'))


@bp.route('/credentials', methods=['POST'])
def save_credentials():
    """Save API credentials."""
    mealie_token = request.form.get('mealie_token', '').strip() or None
    openrouter_key = request.form.get('openrouter_api_key', '').strip() or None
    brave_key = request.form.get('brave_api_key', '').strip() or None
    
    # Only save non-empty values (preserves existing if field left blank)
    save_secrets(
        mealie_token=mealie_token,
        openrouter_api_key=openrouter_key,
        brave_api_key=brave_key
    )
    
    # Reload credentials into running config
    reload_credentials()
    
    return redirect(url_for('main.index'))


@bp.route('/test-connection', methods=['POST'])
def test_connection():
    """Test Mealie connection."""
    url = request.form.get('mealie_url', '').rstrip('/')
    if not url:
        return jsonify({'success': False, 'error': 'URL required'})
    
    try:
        response = requests.get(f"{url}/api/app/about", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return jsonify({'success': True, 'message': f"Connected to Mealie v{data.get('version', '?')}"})
        return jsonify({'success': False, 'error': f'HTTP {response.status_code}'})
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'Timeout'})
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'error': 'Cannot connect'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/reset-worker', methods=['POST'])
def reset_worker():
    """Force cancel all active jobs and reset worker state."""
    count = force_cancel_all_active()
    if count:
        return jsonify({
            'success': True, 
            'cancelled': count,
            'message': f'Reset complete. Cancelled {count} job(s).'
        })
    return jsonify({
        'success': True,
        'cancelled': 0,
        'message': 'No active jobs to cancel.'
    })
