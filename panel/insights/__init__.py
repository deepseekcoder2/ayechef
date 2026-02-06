"""Insights data module for collection personality and exploration stats."""

from .queries import (
    get_cuisine_distribution,
    get_source_sites,
    get_index_health,
    get_most_cooked_recipes,
    get_cuisine_over_time,
    get_collection_story,
    get_coverage_stats,
    get_this_week_stats,
)

__all__ = [
    "get_cuisine_distribution",
    "get_source_sites",
    "get_index_health",
    "get_most_cooked_recipes",
    "get_cuisine_over_time",
    "get_collection_story",
    "get_coverage_stats",
    "get_this_week_stats",
]
