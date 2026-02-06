"""Insights routes - collection personality and exploration stats."""

from flask import Blueprint, render_template

from panel.insights import (
    get_collection_story,
    get_coverage_stats,
    get_this_week_stats,
)

bp = Blueprint('insights', __name__)


@bp.route('/', strict_slashes=False)
def index():
    """Your Kitchen - personality insights about your recipe collection."""
    # Fetch collection data
    collection_story = get_collection_story()
    coverage = get_coverage_stats()
    this_week = get_this_week_stats()
    
    return render_template(
        'insights/index.html',
        collection_story=collection_story,
        coverage=coverage,
        this_week=this_week,
    )
