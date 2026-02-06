"""Diagnostics routes - system health and prerequisites."""
import re
import subprocess
import sys
from flask import Blueprint, render_template, jsonify, redirect, url_for, flash
from panel.health_checks import run_all_checks

bp = Blueprint('diagnostics', __name__)


@bp.route('/', strict_slashes=False)
def index():
    """System status page - shows prerequisites and health."""
    checks = run_all_checks()
    
    # Count issues
    error_count = sum(1 for c in checks.values() if c["status"] == "error")
    warning_count = sum(1 for c in checks.values() if c["status"] == "warning")
    
    return render_template('diagnostics/index.html', 
                          checks=checks,
                          error_count=error_count,
                          warning_count=warning_count)


@bp.route('/refresh')
def refresh():
    """Re-run all checks and return JSON."""
    checks = run_all_checks()
    return jsonify(checks)


@bp.route('/seed-tags', methods=['POST'])
def seed_tags():
    """Seed canonical cuisine tags into Mealie."""
    try:
        # Run the seed script
        result = subprocess.run(
            [sys.executable, 'tools/seed_mealie_tags.py', '--apply'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # Parse "✅ Created N tags." from output
            match = re.search(r'Created (\d+) tags', result.stdout)
            created_count = int(match.group(1)) if match else 0
            
            if created_count > 0:
                flash(f'Successfully created {created_count} cuisine tags in Mealie', 'success')
            else:
                flash('All cuisine tags already exist in Mealie', 'info')
        else:
            flash(f'Error seeding tags: {result.stderr[:100]}', 'error')
            
    except subprocess.TimeoutExpired:
        flash('Tag seeding timed out', 'error')
    except Exception as e:
        flash(f'Error: {str(e)[:100]}', 'error')
    
    return redirect(url_for('diagnostics.index'))
