"""
Insights data queries and aggregation.

This module provides data fetching functions for the insights page,
including collection personality typing and exploration statistics.
"""

import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

from config import DATA_DIR
from tools.logging_utils import get_logger

logger = get_logger(__name__)

# Local recipe index database path
RECIPE_INDEX_DB = DATA_DIR / "recipe_index.db"


@contextmanager
def _get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a connection to the recipe index database with automatic cleanup."""
    conn = sqlite3.connect(RECIPE_INDEX_DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# COLLECTION STORY DATA
# =============================================================================

def get_collection_story() -> Dict[str, Any]:
    """
    Generate the collection story with multi-dimensional MBTI-style archetype.
    
    Uses existing analyze_recipe_database() from recipe_rag to derive dimensions
    from actual data: cuisine distribution, protein families, and dish types.
    
    Returns dict with:
        - total_recipes: int
        - top_cuisines: List of (cuisine, count, percentage) tuples
        - dimensions: Dict of dimension data (breadth, protein, method)
        - archetype_code: 3-letter code (e.g., "S-C-W" for Specialist-Carnivore-Wok)
        - archetype_traits: List of trait descriptions
        - archetype_summary: One-line summary of all dimensions
    """
    result = {
        "total_recipes": 0,
        "top_cuisines": [],
        "unique_cuisines": 0,
        "dimensions": {},
        "archetype_code": "",
        "archetype_traits": [],
        "archetype_summary": "",
    }
    
    if not RECIPE_INDEX_DB.exists():
        return result
    
    try:
        # Get comprehensive data from recipe_rag's existing analyzer
        from recipe_rag import RecipeRAG
        rag = RecipeRAG()
        db_stats = rag.analyze_recipe_database()
        
        total = db_stats.get("total_recipes", 0)
        result["total_recipes"] = total
        
        if total == 0:
            return result
        
        cuisine_dist = db_stats.get("cuisine_distribution", {})
        protein_families = db_stats.get("protein_families", {})
        dish_types = db_stats.get("dish_types", {})
        
        # =====================================================================
        # DIMENSION 1: BREADTH (from cuisine distribution)
        # How focused vs diverse is the collection?
        # =====================================================================
        sorted_cuisines = sorted(cuisine_dist.items(), key=lambda x: x[1], reverse=True)
        result["unique_cuisines"] = len(sorted_cuisines)
        
        # Calculate percentages for top cuisines
        top_cuisines = []
        for cuisine, count in sorted_cuisines[:5]:
            pct = round(count / total * 100, 1)
            top_cuisines.append((cuisine, count, pct))
        result["top_cuisines"] = top_cuisines
        
        if top_cuisines:
            top_cuisine = top_cuisines[0][0]
            top_pct = top_cuisines[0][2]
            top2_pct = top_cuisines[1][2] if len(top_cuisines) > 1 else 0
            top3_pct = sum(c[2] for c in top_cuisines[:3])
            
            # Dominance ratio: how far ahead is #1 from #2?
            dominance_ratio = top_pct / top2_pct if top2_pct > 0 else float('inf')
            
            # SPECIALIST: One cuisine clearly dominates
            # Either very high concentration (≥30%) with significant lead (≥2.5x second place)
            # Or extremely high concentration (≥50%) regardless of lead
            if (top_pct >= 30 and dominance_ratio >= 2.5) or top_pct >= 50:
                breadth_code = "S"  # Specialist
                breadth_label = "Specialist"
                breadth_desc = f"{top_pct:.0f}% {top_cuisine} ({dominance_ratio:.1f}x lead)"
            
            # FOCUSED: Top few cuisines make up majority, but no single domination
            elif top3_pct >= 50:
                breadth_code = "F"  # Focused
                breadth_label = "Focused"
                breadth_desc = f"Top 3 = {top3_pct:.0f}%"
            
            # EXPLORER: More distributed collection
            else:
                breadth_code = "E"  # Explorer
                breadth_label = "Explorer"
                breadth_desc = f"Top 3 only {top3_pct:.0f}%"
            
            result["dimensions"]["breadth"] = {
                "code": breadth_code,
                "label": breadth_label,
                "description": breadth_desc,
                "top_cuisine": top_cuisine,
                "top_pct": top_pct,
            }
        
        # =====================================================================
        # DIMENSION 2: PROTEIN PROFILE (from protein_families)
        # What's the protein preference?
        # =====================================================================
        if protein_families:
            total_protein = sum(protein_families.values())
            
            if total_protein > 0:
                # Calculate percentages
                protein_pcts = {k: (v / total_protein * 100) for k, v in protein_families.items()}
                
                # Group proteins
                meat_pct = protein_pcts.get("beef", 0) + protein_pcts.get("pork", 0) + protein_pcts.get("lamb", 0)
                poultry_pct = protein_pcts.get("chicken", 0) + protein_pcts.get("turkey", 0) + protein_pcts.get("duck", 0)
                seafood_pct = protein_pcts.get("fish", 0) + protein_pcts.get("seafood", 0)
                plant_pct = protein_pcts.get("tofu", 0) + protein_pcts.get("vegetarian", 0)
                
                # Determine dominant protein profile
                profiles = [
                    ("M", "Meat-Forward", meat_pct),
                    ("P", "Poultry-Centric", poultry_pct),
                    ("S", "Seafood-Leaning", seafood_pct),
                    ("V", "Plant-Forward", plant_pct),
                ]
                profiles.sort(key=lambda x: x[2], reverse=True)
                
                top_profile = profiles[0]
                
                # Check if balanced (no profile > 35%)
                if top_profile[2] < 35:
                    protein_code = "B"
                    protein_label = "Balanced"
                    protein_desc = "Mixed proteins"
                else:
                    protein_code = top_profile[0]
                    protein_label = top_profile[1]
                    protein_desc = f"{top_profile[2]:.0f}% of protein mentions"
                
                result["dimensions"]["protein"] = {
                    "code": protein_code,
                    "label": protein_label,
                    "description": protein_desc,
                    "breakdown": {k: round(v, 1) for k, v in protein_pcts.items() if v > 0},
                }
        
        # =====================================================================
        # DIMENSION 3: COOKING METHOD (from dish_types)
        # What's the dominant cooking style?
        # =====================================================================
        if dish_types:
            total_methods = sum(dish_types.values())
            
            if total_methods > 0:
                method_pcts = {k: (v / total_methods * 100) for k, v in dish_types.items() if v > 0}
                sorted_methods = sorted(method_pcts.items(), key=lambda x: x[1], reverse=True)
                
                # Get top method
                if sorted_methods:
                    top_method, top_method_pct = sorted_methods[0]
                    
                    # Map to code and label
                    method_codes = {
                        "stir_fry": ("W", "Wok"),
                        "grilled": ("G", "Grill"),
                        "braised": ("L", "Low-Slow"),  # Low and slow
                        "steamed": ("T", "Steam"),
                        "baked": ("B", "Baker"),
                        "soup": ("O", "One-Pot"),
                        "pasta": ("N", "Noodles"),
                        "rice": ("R", "Rice"),
                        "salad": ("F", "Fresh"),
                        "other": ("X", "Mixed"),
                    }
                    
                    method_code, method_label = method_codes.get(top_method, ("X", "Mixed"))
                    
                    # Check if varied (top method < 25%)
                    if top_method_pct < 25:
                        method_code = "X"
                        method_label = "Varied"
                        method_desc = "No dominant method"
                    else:
                        method_desc = f"{top_method_pct:.0f}% {top_method.replace('_', ' ')}"
                    
                    result["dimensions"]["method"] = {
                        "code": method_code,
                        "label": method_label,
                        "description": method_desc,
                        "breakdown": {k.replace('_', ' '): round(v, 1) for k, v in sorted_methods[:5]},
                    }
        
        # =====================================================================
        # BUILD ARCHETYPE
        # =====================================================================
        codes = []
        traits = []
        
        if "breadth" in result["dimensions"]:
            dim = result["dimensions"]["breadth"]
            codes.append(dim["code"])
            traits.append(f"{dim['label']}: {dim['description']}")
        
        if "protein" in result["dimensions"]:
            dim = result["dimensions"]["protein"]
            codes.append(dim["code"])
            traits.append(f"{dim['label']}: {dim['description']}")
        
        if "method" in result["dimensions"]:
            dim = result["dimensions"]["method"]
            codes.append(dim["code"])
            traits.append(f"{dim['label']}: {dim['description']}")
        
        result["archetype_code"] = "-".join(codes) if codes else ""
        result["archetype_traits"] = traits
        
        # Build summary sentence
        summaries = []
        if "breadth" in result["dimensions"]:
            b = result["dimensions"]["breadth"]
            if b["code"] == "S":
                summaries.append(f"deep in {b['top_cuisine']}")
            elif b["code"] == "F":
                summaries.append(f"focused around {b['top_cuisine']}")
            else:
                summaries.append("culinarily adventurous")
        
        if "protein" in result["dimensions"]:
            p = result["dimensions"]["protein"]
            if p["code"] == "V":
                summaries.append("plant-forward")
            elif p["code"] == "S":
                summaries.append("seafood-loving")
            elif p["code"] == "M":
                summaries.append("meat-centric")
            elif p["code"] == "P":
                summaries.append("poultry-focused")
        
        if "method" in result["dimensions"]:
            m = result["dimensions"]["method"]
            if m["code"] == "W":
                summaries.append("wok-driven")
            elif m["code"] == "G":
                summaries.append("grill-inclined")
            elif m["code"] == "L":
                summaries.append("slow-cook oriented")
            elif m["code"] == "B":
                summaries.append("oven-focused")
        
        if summaries:
            result["archetype_summary"] = (
                f"A collection of {total:,} recipes that's {', '.join(summaries)}."
            )
        
    except Exception as e:
        logger.error(f"Error generating collection story: {e}")
    
    return result


def get_coverage_stats() -> Dict[str, Any]:
    """
    Get recipe coverage statistics - how much of the collection has been explored.
    
    Returns dict with:
        - total_recipes: int
        - explored_count: int (distinct recipes ever planned)
        - coverage_percent: float
        - unexplored_count: int
    """
    result = {
        "total_recipes": 0,
        "explored_count": 0,
        "coverage_percent": 0.0,
        "unexplored_count": 0,
    }
    
    if not RECIPE_INDEX_DB.exists():
        return result
    
    try:
        # Get total from local index
        with _get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM recipes")
            total = cursor.fetchone()[0]
            result["total_recipes"] = total
        
        # Get explored recipes from Mealie meal plans
        from mealie_client import MealieClient
        client = MealieClient(use_direct_db=False)
        try:
            # Get all meal plans (go back far - 5 years)
            end_date = date.today()
            start_date = end_date - timedelta(days=365 * 5)
            meal_plans = client.get_meal_plans(start_date, end_date)
            
            # Count distinct recipe IDs
            explored_ids = set()
            for entry in meal_plans:
                recipe_id = entry.get('recipeId')
                if recipe_id:
                    explored_ids.add(recipe_id)
            
            explored = len(explored_ids)
            result["explored_count"] = explored
            result["coverage_percent"] = round(explored / total * 100, 1) if total > 0 else 0
            result["unexplored_count"] = max(0, total - explored)
        finally:
            client.close()
            
    except Exception as e:
        logger.error(f"Error fetching coverage stats: {e}")
    
    return result


def get_this_week_stats() -> Dict[str, Any]:
    """
    Get statistics for the current/most recent planned week.
    
    Returns dict with:
        - week_label: str (e.g., "Feb 3–9")
        - meal_count: int
        - cuisine_counts: Dict[str, int]
        - protein_counts: Dict[str, int] (inferred from recipe tags/ingredients)
        - fallback_count: int (recipes without IDs - AI generated)
        - has_data: bool
    """
    result = {
        "week_label": "",
        "meal_count": 0,
        "cuisine_counts": {},
        "protein_counts": {},
        "fallback_count": 0,
        "has_data": False,
    }
    
    try:
        from mealie_client import MealieClient
        client = MealieClient(use_direct_db=False)
        
        try:
            # Find the most recent week with data
            end_date = date.today() + timedelta(days=14)  # Include next 2 weeks
            start_date = date.today() - timedelta(days=7)
            
            meal_plans = client.get_meal_plans(start_date, end_date)
            
            if not meal_plans:
                return result
            
            # Group by week (find the week with most entries)
            weeks: Dict[str, List] = defaultdict(list)
            for entry in meal_plans:
                entry_date_str = entry.get('date')
                if not entry_date_str:
                    continue
                try:
                    if isinstance(entry_date_str, str):
                        entry_date = date.fromisoformat(entry_date_str[:10])
                    else:
                        entry_date = entry_date_str
                    # Get week start (Monday)
                    week_start = entry_date - timedelta(days=entry_date.weekday())
                    weeks[week_start.isoformat()].append(entry)
                except (ValueError, TypeError):
                    continue
            
            if not weeks:
                return result
            
            # Find the most recent week with data
            latest_week = max(weeks.keys())
            week_entries = weeks[latest_week]
            
            # Format week label
            week_start = date.fromisoformat(latest_week)
            week_end = week_start + timedelta(days=6)
            result["week_label"] = f"{week_start.strftime('%b %d')}–{week_end.strftime('%d')}"
            result["meal_count"] = len(week_entries)
            result["has_data"] = True
            
            # Get recipe IDs and count fallbacks
            recipe_ids = []
            for entry in week_entries:
                recipe_id = entry.get('recipeId')
                if recipe_id:
                    recipe_ids.append(recipe_id)
                else:
                    # Entry without recipe ID is a fallback/note
                    result["fallback_count"] += 1
            
            # Get cuisine info from local index
            if recipe_ids and RECIPE_INDEX_DB.exists():
                # Build slug mapping from meal plan entries
                id_to_slug = {}
                for entry in week_entries:
                    recipe_id = entry.get('recipeId')
                    recipe = entry.get('recipe') or {}
                    slug = recipe.get('slug')
                    if recipe_id and slug:
                        id_to_slug[recipe_id] = slug
                
                slugs = list(id_to_slug.values())
                if slugs:
                    with _get_connection() as conn:
                        placeholders = ','.join(['?'] * len(slugs))
                        cursor = conn.execute(f"""
                            SELECT cuisine_primary
                            FROM recipes
                            WHERE slug IN ({placeholders})
                        """, slugs)
                        
                        for row in cursor:
                            cuisine = row['cuisine_primary']
                            if cuisine:
                                result["cuisine_counts"][cuisine] = result["cuisine_counts"].get(cuisine, 0) + 1
        finally:
            client.close()
            
    except Exception as e:
        logger.error(f"Error fetching this week stats: {e}")
    
    return result


def get_cuisine_distribution(limit: int = 15) -> List[Dict[str, Any]]:
    """
    Get distribution of recipes by primary cuisine.
    
    Args:
        limit: Maximum number of cuisines to return
        
    Returns:
        List of dicts with 'cuisine' and 'count' keys, sorted by count descending
    """
    if not RECIPE_INDEX_DB.exists():
        return []
    
    try:
        with _get_connection() as conn:
            cursor = conn.execute("""
                SELECT cuisine_primary AS cuisine, COUNT(*) AS count
                FROM recipes
                WHERE cuisine_primary IS NOT NULL AND cuisine_primary != ''
                GROUP BY cuisine_primary
                ORDER BY count DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching cuisine distribution: {e}")
        return []


def get_source_sites(limit: int = 15) -> List[Dict[str, Any]]:
    """
    Get distribution of recipes by source website.
    
    Extracts domain from org_url and counts recipes per domain.
    
    Args:
        limit: Maximum number of sources to return
        
    Returns:
        List of dicts with 'domain' and 'count' keys, sorted by count descending
    """
    if not RECIPE_INDEX_DB.exists():
        return []
    
    try:
        with _get_connection() as conn:
            cursor = conn.execute("""
                SELECT org_url FROM recipes WHERE org_url IS NOT NULL AND org_url != ''
            """)
            
            # Count domains in Python for better domain extraction
            domain_counts: Dict[str, int] = defaultdict(int)
            for row in cursor:
                url = row['org_url']
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc or parsed.path.split('/')[0]
                    # Remove www. prefix for cleaner display
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    if domain:
                        domain_counts[domain] += 1
                except Exception:
                    continue
            
            # Sort by count and limit
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
            return [{"domain": domain, "count": count} for domain, count in sorted_domains]
    except Exception as e:
        logger.error(f"Error fetching source sites: {e}")
        return []


def get_index_health() -> Dict[str, Any]:
    """
    Get health statistics for the recipe index.
    
    Returns:
        Dict with total_recipes, tagged_count, untagged_count, 
        embedded_count, and percentage values
    """
    empty_result = {
        "total_recipes": 0,
        "tagged_count": 0,
        "untagged_count": 0,
        "embedded_count": 0,
        "tagged_percent": 0,
        "embedded_percent": 0,
    }
    
    if not RECIPE_INDEX_DB.exists():
        return empty_result
    
    try:
        with _get_connection() as conn:
            # Total recipes
            cursor = conn.execute("SELECT COUNT(*) FROM recipes")
            total = cursor.fetchone()[0]
            
            # Tagged (has cuisine_primary)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM recipes 
                WHERE cuisine_primary IS NOT NULL AND cuisine_primary != ''
            """)
            tagged = cursor.fetchone()[0]
            
            # Embedded (has embedding)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM recipes 
                WHERE embedding IS NOT NULL
            """)
            embedded = cursor.fetchone()[0]
            
            return {
                "total_recipes": total,
                "tagged_count": tagged,
                "untagged_count": total - tagged,
                "embedded_count": embedded,
                "tagged_percent": round(tagged / total * 100, 1) if total > 0 else 0,
                "embedded_percent": round(embedded / total * 100, 1) if total > 0 else 0,
            }
    except Exception as e:
        logger.error(f"Error fetching index health: {e}")
        return empty_result


def get_most_cooked_recipes(weeks: int = 12, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Get the most frequently cooked recipes from meal plan history.
    
    Args:
        weeks: Number of weeks to look back
        limit: Maximum number of recipes to return
        
    Returns:
        List of dicts with 'name', 'slug', 'count' keys, sorted by count descending
    """
    try:
        from mealie_client import MealieClient
        
        # Force API mode - Mealie DB is in a separate container
        client = MealieClient(use_direct_db=False)
        try:
            # Calculate date range
            end_date = date.today()
            start_date = end_date - timedelta(weeks=weeks)
            
            # Get meal plans
            meal_plans = client.get_meal_plans(start_date, end_date)
            
            # Count recipes
            recipe_counts: Dict[str, Dict[str, Any]] = {}
            for entry in meal_plans:
                recipe_id = entry.get('recipeId')
                if not recipe_id:
                    continue
                
                # Get recipe details - API returns nested 'recipe' object
                recipe = entry.get('recipe') or {}
                recipe_name = recipe.get('name') if recipe else None
                recipe_slug = recipe.get('slug') if recipe else None
                    
                if recipe_id not in recipe_counts:
                    recipe_counts[recipe_id] = {
                        "name": recipe_name or 'Unknown',
                        "slug": recipe_slug or '',
                        "count": 0
                    }
                recipe_counts[recipe_id]["count"] += 1
            
            # Sort by count and limit
            sorted_recipes = sorted(
                recipe_counts.values(), 
                key=lambda x: x['count'], 
                reverse=True
            )[:limit]
            
            return sorted_recipes
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Error fetching most cooked recipes: {e}")
        return []


def get_cuisine_over_time(weeks: int = 12) -> Dict[str, Any]:
    """
    Get cuisine distribution over time from meal plan history.
    
    Joins meal plan data with local recipe index to get cuisine info.
    
    Args:
        weeks: Number of weeks to look back
        
    Returns:
        Dict with:
            - labels: List of week labels (e.g., "Jan 6", "Jan 13")
            - datasets: List of cuisine datasets with name and weekly values
    """
    try:
        from mealie_client import MealieClient
        
        if not RECIPE_INDEX_DB.exists():
            return {"labels": [], "datasets": []}
        
        # Force API mode - Mealie DB is in a separate container
        client = MealieClient(use_direct_db=False)
        try:
            # Calculate date range
            end_date = date.today()
            start_date = end_date - timedelta(weeks=weeks)
            
            # Get meal plans
            meal_plans = client.get_meal_plans(start_date, end_date)
            
            # Build recipe ID to slug mapping from meal plans
            recipe_slugs: Dict[str, str] = {}
            for entry in meal_plans:
                recipe_id = entry.get('recipeId')
                # API returns nested 'recipe' object
                recipe = entry.get('recipe') or {}
                slug = recipe.get('slug') if recipe else None
                if recipe_id and slug:
                    recipe_slugs[recipe_id] = slug
            
            if not recipe_slugs:
                return {"labels": [], "datasets": []}
            
            # Get cuisine mappings from local index (by slug)
            slug_to_cuisine: Dict[str, str] = {}
            with _get_connection() as conn:
                placeholders = ','.join(['?'] * len(recipe_slugs))
                cursor = conn.execute(f"""
                    SELECT slug, cuisine_primary
                    FROM recipes
                    WHERE slug IN ({placeholders})
                """, list(recipe_slugs.values()))
                
                for row in cursor:
                    if row['cuisine_primary']:
                        slug_to_cuisine[row['slug']] = row['cuisine_primary']
            
            # Create week buckets
            week_buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            week_labels: List[Tuple[date, str]] = []
            
            # Generate week labels
            current = start_date
            while current <= end_date:
                week_start = current - timedelta(days=current.weekday())  # Monday
                label = week_start.strftime("%b %d")
                week_labels.append((week_start, label))
                current += timedelta(weeks=1)
            
            # Deduplicate and sort week labels
            seen_weeks = set()
            unique_labels = []
            for week_start, label in week_labels:
                if week_start not in seen_weeks:
                    seen_weeks.add(week_start)
                    unique_labels.append((week_start, label))
            unique_labels.sort(key=lambda x: x[0])
            
            # Count cuisines per week
            for entry in meal_plans:
                recipe_id = entry.get('recipeId')
                entry_date_str = entry.get('date')
                
                if not recipe_id or not entry_date_str:
                    continue
                
                # Parse date
                try:
                    if isinstance(entry_date_str, str):
                        entry_date = date.fromisoformat(entry_date_str[:10])
                    else:
                        entry_date = entry_date_str
                except (ValueError, TypeError):
                    continue
                
                # Find week bucket
                week_start = entry_date - timedelta(days=entry_date.weekday())
                week_label = week_start.strftime("%b %d")
                
                # Get cuisine
                slug = recipe_slugs.get(recipe_id)
                cuisine = slug_to_cuisine.get(slug, "Unknown") if slug else "Unknown"
                
                week_buckets[week_label][cuisine] += 1
            
            # Get top cuisines across all weeks
            cuisine_totals: Dict[str, int] = defaultdict(int)
            for week_data in week_buckets.values():
                for cuisine, count in week_data.items():
                    cuisine_totals[cuisine] += count
            
            top_cuisines = sorted(cuisine_totals.keys(), key=lambda c: cuisine_totals[c], reverse=True)[:8]
            
            # Build datasets
            labels = [label for _, label in unique_labels]
            datasets = []
            
            for cuisine in top_cuisines:
                # Explicitly build values list to ensure it's JSON serializable
                cuisine_values = []
                for _, label in unique_labels:
                    count = week_buckets.get(label, {}).get(cuisine, 0)
                    cuisine_values.append(int(count))
                datasets.append({
                    "name": str(cuisine),
                    "values": cuisine_values
                })
            
            return {
                "labels": list(labels),
                "datasets": list(datasets)
            }
        finally:
            client.close()
    except Exception as e:
        logger.error(f"Error fetching cuisine over time: {e}")
        return {"labels": [], "datasets": []}
