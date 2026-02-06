"""Health check functions for system diagnostics."""
import os
import requests
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path
from mealie_client import MealieClient, MealieClientError


def check_config_file() -> Dict[str, Any]:
    """Check if config.yaml exists and has required fields."""
    from config import CONFIG_PATH
    config_path = CONFIG_PATH
    
    if not config_path.exists():
        return {
            "status": "error",
            "message": "config.yaml not found",
            "fix": "Copy config.yaml.example to data/config.yaml and fill in your settings"
        }
    
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check required fields (actual structure from config.py)
        missing = []
        if not config.get("connection", {}).get("mealie_url"):
            missing.append("connection.mealie_url")
        
        if missing:
            return {
                "status": "warning",
                "message": f"Missing: {', '.join(missing)}",
                "fix": "Edit config.yaml and add the missing values"
            }
        
        return {"status": "ok", "message": "Config file valid"}
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading config: {str(e)[:50]}",
            "fix": "Check config.yaml syntax (must be valid YAML)"
        }


def check_mealie_connection() -> Dict[str, Any]:
    """Check if Mealie API is reachable."""
    try:
        from config import MEALIE_URL, MEALIE_TOKEN
        
        if not MEALIE_URL:
            return {
                "status": "error",
                "message": "Mealie URL not configured",
                "fix": "Set connection.mealie_url in config.yaml"
            }
        
        if not MEALIE_TOKEN:
            return {
                "status": "error",
                "message": "Mealie token not configured",
                "fix": "Add via Settings → API Credentials or set MEALIE_TOKEN env var"
            }
        
        # Test connectivity by attempting to fetch recipes
        client = MealieClient()
        try:
            client.get_all_recipes()
            return {"status": "ok", "message": "Connected"}
        except MealieClientError as e:
            if "401" in str(e) or "Invalid" in str(e) or "token" in str(e).lower():
                return {
                    "status": "error",
                    "message": "Invalid API token",
                    "fix": "Update token via Settings → API Credentials"
                }
            else:
                return {
                    "status": "error",
                    "message": f"API error: {str(e)[:50]}",
                    "fix": "Check connection.mealie_url in config.yaml"
                }
        finally:
            client.close()
    
    except MealieClientError as e:
        error_str = str(e)
        if "Cannot connect" in error_str or "Connection" in error_str:
            return {
                "status": "error",
                "message": "Cannot connect to Mealie",
                "fix": "Check connection.mealie_url and ensure Mealie is running"
            }
        return {
            "status": "error",
            "message": error_str[:50],
            "fix": "Check Mealie configuration"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)[:50],
            "fix": "Check Mealie configuration"
        }


def check_openrouter_api() -> Dict[str, Any]:
    """Check if OpenRouter API key is configured and valid."""
    try:
        from config import CHAT_API_KEY
        
        if not CHAT_API_KEY:
            return {
                "status": "error",
                "message": "API key not configured",
                "fix": "Add via Settings → API Credentials or set OPENROUTER_API_KEY env var"
            }
        
        # Test the API with a minimal request
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {CHAT_API_KEY}"},
            timeout=10
        )
        
        if response.status_code == 200:
            return {"status": "ok", "message": "API key valid"}
        elif response.status_code == 401:
            return {
                "status": "error",
                "message": "Invalid API key",
                "fix": "Update key via Settings → API Credentials"
            }
        else:
            return {
                "status": "warning",
                "message": f"API returned {response.status_code}",
                "fix": "API key may be valid but rate limited"
            }
    
    except Exception as e:
        return {
            "status": "warning",
            "message": f"Could not verify: {str(e)[:30]}",
            "fix": "Check internet connection"
        }


def check_embedding_model() -> Dict[str, Any]:
    """Check if embedding model is downloaded."""
    try:
        from config import USER_CONFIG
        model_name = USER_CONFIG.get("llm", {}).get("embedding_model", "nvidia/llama-embed-nemotron-8b")
        
        # Check if model files exist in cache
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        
        # Also check Docker volume location
        docker_cache = Path("/root/.cache/huggingface/hub")
        
        # Model folder name format: models--org--name
        model_folder = f"models--{model_name.replace('/', '--')}"
        
        if (cache_dir / model_folder).exists() or (docker_cache / model_folder).exists():
            return {"status": "ok", "message": "Model downloaded"}
        
        # Model not found in cache - required but will auto-download
        return {
            "status": "warning", 
            "message": "Not yet downloaded (~2GB)",
            "fix": "Required for meal planning (~16GB download). Pre-download with: docker compose exec worker python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('nvidia/llama-embed-nemotron-8b')\""
        }
    
    except Exception as e:
        return {
            "status": "warning",
            "message": str(e)[:50],
            "fix": "Check embedding model configuration"
        }


def check_recipe_index() -> Dict[str, Any]:
    """Check if recipe index exists and has data."""
    from config import DATA_DIR
    try:
        db_path = DATA_DIR / "recipe_index.db"
        
        if not db_path.exists():
            return {
                "status": "warning",
                "message": "Recipe index not built",
                "fix": "Index will build automatically on first meal plan, or run the Index Recipes tool"
            }
        
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM recipes")
        count = cur.fetchone()[0]
        conn.close()
        
        if count == 0:
            return {
                "status": "warning",
                "message": "Index empty (no recipes)",
                "fix": "Import some recipes first, then rebuild index"
            }
        
        return {"status": "ok", "message": f"{count:,} recipes indexed"}
    
    except Exception as e:
        return {
            "status": "warning",
            "message": str(e)[:50],
            "fix": "Rebuild recipe index using maintenance tools"
        }


def check_recipe_quality() -> Dict[str, Any]:
    """
    Check recipe quality - untagged, unindexed, unparsed counts.
    
    Returns dict with:
        - status: "ok", "warning", or "error"
        - message: Human-readable summary
        - details: Dict with unparsed, untagged, unindexed counts
        - fix: Suggested action
    """
    from config import DATA_DIR
    import sqlite3
    
    details = {
        "unparsed": 0,
        "untagged": 0,
        "unindexed": 0,
        "total_mealie": 0,
        "total_indexed": 0
    }
    
    try:
        # Check local DB first (fast)
        db_path = DATA_DIR / "recipe_index.db"
        if not db_path.exists():
            return {
                "status": "warning",
                "message": "Recipe index not built",
                "details": details,
                "fix": "Run Full Maintenance to build the recipe index"
            }
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count untagged (no cuisine classification)
        cursor.execute("""
            SELECT COUNT(*) FROM recipes 
            WHERE cuisine_primary IS NULL OR cuisine_primary = ''
        """)
        details["untagged"] = cursor.fetchone()[0]
        
        # Count total indexed
        cursor.execute("SELECT COUNT(*) FROM recipes")
        details["total_indexed"] = cursor.fetchone()[0]
        
        conn.close()
        
        # Get Mealie total (one API call)
        try:
            client = MealieClient()
            try:
                recipes = client.get_all_recipes()
                details["total_mealie"] = len(recipes)
                details["unindexed"] = max(0, details["total_mealie"] - details["total_indexed"])
            finally:
                client.close()
        except Exception:
            pass  # If API fails, just report what we know from local DB
        
        # Check unparsed using cached function (only if reasonable number of recipes)
        # Skip for very large collections to avoid slow dashboard load
        if details["total_mealie"] <= 5000:
            try:
                # Import here to avoid circular dependencies
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from mealie_parse import get_unparsed_slugs
                unparsed = get_unparsed_slugs()
                details["unparsed"] = len(unparsed) if unparsed else 0
            except Exception:
                pass  # If parsing check fails, continue without it
        
        # Determine status
        total_issues = details["unparsed"] + details["untagged"] + details["unindexed"]
        
        if total_issues == 0:
            return {
                "status": "ok",
                "message": "All recipes processed",
                "details": details,
                "fix": None
            }
        
        # Build message
        issues = []
        if details["unparsed"] > 0:
            issues.append(f"{details['unparsed']} unparsed")
        if details["untagged"] > 0:
            issues.append(f"{details['untagged']} untagged")
        if details["unindexed"] > 0:
            issues.append(f"{details['unindexed']} unindexed")
        
        return {
            "status": "warning",
            "message": f"{total_issues} recipes need processing ({', '.join(issues)})",
            "details": details,
            "fix": "Run Full Maintenance to process all recipes"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Check failed: {str(e)[:50]}",
            "details": details,
            "fix": "Check database connectivity"
        }


def check_sync_staleness() -> Dict[str, Any]:
    """
    Check if recipe index sync is stale (> 7 days since last sync).
    
    This helps detect when users edit recipes in Mealie's UI but don't
    run sync, which can cause the cached ingredients_parsed status to
    become inaccurate.
    
    Returns:
        Dict with status, message, and optional fix suggestion.
    """
    from config import DATA_DIR
    try:
        db_path = DATA_DIR / "recipe_index.db"
        if not db_path.exists():
            # No index = nothing to be stale
            return {
                "status": "ok",
                "message": "No recipe index yet",
                "details": {"last_sync": None, "days_ago": None}
            }
        
        # Import RecipeRAG to get last sync timestamp
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from recipe_rag import RecipeRAG
        
        rag = RecipeRAG(db_path=str(db_path))
        last_sync = rag.get_last_sync_timestamp()
        
        if last_sync is None:
            # Index exists but no recipes with timestamps
            return {
                "status": "warning",
                "message": "No sync timestamp found",
                "details": {"last_sync": None, "days_ago": None},
                "fix": "Run 'Repair & Optimize' to sync recipes"
            }
        
        # Calculate days since last sync
        now = datetime.now(timezone.utc)
        # Ensure last_sync is timezone-aware for comparison
        if last_sync.tzinfo is None:
            last_sync = last_sync.replace(tzinfo=timezone.utc)
        
        days_ago = (now - last_sync).days
        
        details = {
            "last_sync": last_sync.isoformat(),
            "days_ago": days_ago
        }
        
        if days_ago > 7:
            return {
                "status": "warning",
                "message": f"Last sync was {days_ago} days ago",
                "details": details,
                "fix": "Recipe index hasn't synced in over 7 days. Run 'Repair & Optimize' to ensure parsing status is accurate."
            }
        elif days_ago > 3:
            # Informational note for 3-7 days
            return {
                "status": "ok",
                "message": f"Last sync {days_ago} days ago",
                "details": details
            }
        else:
            return {
                "status": "ok",
                "message": f"Recently synced ({days_ago}d ago)" if days_ago > 0 else "Synced today",
                "details": details
            }
    
    except Exception as e:
        return {
            "status": "warning",
            "message": f"Could not check: {str(e)[:30]}",
            "details": {"last_sync": None, "days_ago": None},
            "fix": "Check database connectivity"
        }


def get_recipe_quality_quick() -> Dict[str, Any]:
    """
    Quick recipe quality check for dashboard (fast, local DB only).
    
    Only checks untagged count from local DB - no API calls, no slow scans.
    Use check_recipe_quality() for full diagnostics.
    """
    from config import DATA_DIR
    import sqlite3
    
    try:
        db_path = DATA_DIR / "recipe_index.db"
        if not db_path.exists():
            return {"needs_maintenance": True, "untagged": 0, "total": 0}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count untagged
        cursor.execute("""
            SELECT COUNT(*) FROM recipes 
            WHERE cuisine_primary IS NULL OR cuisine_primary = ''
        """)
        untagged = cursor.fetchone()[0]
        
        # Count total
        cursor.execute("SELECT COUNT(*) FROM recipes")
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "needs_maintenance": untagged > 0,
            "untagged": untagged,
            "total": total
        }
        
    except Exception:
        return {"needs_maintenance": False, "untagged": 0, "total": 0}


def check_cuisine_distribution() -> Dict[str, Any]:
    """
    Check if recipes are tagged and distributed across cuisines.
    
    This is CRITICAL for meal planning - chef_agentic.py requires at least
    one cuisine with 10+ recipes to function.
    """
    from config import DATA_DIR
    import sqlite3
    
    try:
        db_path = DATA_DIR / "recipe_index.db"
        if not db_path.exists():
            return {
                "status": "error",
                "message": "Recipe index not built",
                "details": {"cuisines": [], "max_count": 0, "viable_count": 0},
                "fix": "Import recipes and run maintenance"
            }
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get cuisine distribution
        cursor.execute("""
            SELECT cuisine_primary, COUNT(*) as count
            FROM recipes
            WHERE cuisine_primary IS NOT NULL AND cuisine_primary != ''
            GROUP BY cuisine_primary
            ORDER BY count DESC
        """)
        distribution = cursor.fetchall()
        conn.close()
        
        if not distribution:
            return {
                "status": "error",
                "message": "No recipes have cuisine tags",
                "details": {"cuisines": [], "max_count": 0, "viable_count": 0},
                "fix": "Run maintenance to tag recipes with cuisines"
            }
        
        # Check if any cuisine has 10+ recipes (minimum for meal planning)
        MIN_REQUIRED = 10
        max_cuisine, max_count = distribution[0]
        viable_count = sum(1 for _, count in distribution if count >= MIN_REQUIRED)
        
        details = {
            "cuisines": distribution[:5],  # Top 5 cuisines
            "max_count": max_count,
            "viable_count": viable_count,
            "total_cuisines": len(distribution)
        }
        
        if max_count < MIN_REQUIRED:
            return {
                "status": "error",
                "message": f"Largest cuisine has only {max_count} recipes (need {MIN_REQUIRED}+)",
                "details": details,
                "fix": "Import more recipes or run maintenance to tag existing ones"
            }
        
        return {
            "status": "ok",
            "message": f"{viable_count} cuisines ready ({max_cuisine}: {max_count})",
            "details": details
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Check failed: {str(e)[:50]}",
            "details": {"cuisines": [], "max_count": 0, "viable_count": 0},
            "fix": "Check database connectivity"
        }


def check_mealie_tags() -> Dict[str, Any]:
    """
    Check if canonical cuisine tags exist in Mealie.
    
    This is RECOMMENDED (not required) - helps with Mealie organization
    but meal planning uses local cuisine_primary, not Mealie tags.
    """
    try:
        from config import MEALIE_URL, MEALIE_TOKEN
        
        if not MEALIE_URL or not MEALIE_TOKEN:
            return {
                "status": "warning",
                "message": "Mealie not configured",
                "fix": "Configure Mealie connection first"
            }
        
        # Fetch all tags from Mealie
        client = MealieClient()
        try:
            mealie_tags_list = client.get_all_tags()
        except Exception:
            return {
                "status": "warning",
                "message": "Could not check Mealie tags",
                "fix": "Check Mealie connection"
            }
        finally:
            client.close()
        
        # Get canonical tags from taxonomy
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from cuisine_taxonomy import canonical_cuisine_tag_names
            canonical_tags = set(canonical_cuisine_tag_names())
        except ImportError:
            return {
                "status": "warning",
                "message": "Could not load cuisine taxonomy",
                "fix": "Check cuisine_taxonomy.py exists"
            }
        
        mealie_tags = {tag["name"].lower() for tag in mealie_tags_list if tag.get("name")}
        canonical_lower = {t.lower() for t in canonical_tags}
        
        found = len(canonical_lower & mealie_tags)
        missing = len(canonical_lower - mealie_tags)
        total = len(canonical_tags)
        
        details = {
            "found": found,
            "missing": missing,
            "total": total
        }
        
        if missing == 0:
            return {
                "status": "ok",
                "message": f"All {total} cuisine tags present",
                "details": details
            }
        elif found >= 50:
            return {
                "status": "ok",
                "message": f"{found}/{total} tags present",
                "details": details
            }
        elif found > 0:
            return {
                "status": "warning",
                "message": f"{missing} cuisine tags missing from Mealie",
                "details": details,
                "fix": "Seed tags for better Mealie organization"
            }
        else:
            return {
                "status": "warning",
                "message": "No cuisine tags in Mealie",
                "details": details,
                "fix": "Seed tags to organize recipes in Mealie"
            }
            
    except Exception as e:
        return {
            "status": "warning",
            "message": f"Could not check: {str(e)[:30]}",
            "fix": "Check Mealie connection"
        }


def is_ready_for_planning() -> Dict[str, Any]:
    """
    Check if all critical prerequisites for meal planning are met.
    
    Uses short-circuit logic: if credentials (mealie/openrouter) are missing,
    downstream checks (index, cuisine) are skipped since they would fail
    as cascading errors, inflating the issue count and confusing users.
    
    Returns:
        Dict with:
            - ready: bool - True if meal planning can proceed
            - blocking: List[str] - Names of failing critical checks
            - message: str - Human-readable summary
    """
    blocking = []
    
    # Phase 1: Foundation checks (config + credentials)
    foundation_checks = {
        "config": check_config_file,
        "mealie": check_mealie_connection,
        "openrouter": check_openrouter_api,
    }
    
    for name, check_fn in foundation_checks.items():
        result = check_fn()
        if result.get("status") == "error":
            blocking.append(name)
    
    # Phase 2: Data checks — only run if credentials are configured
    # (these depend on a working Mealie connection)
    if "mealie" not in blocking:
        data_checks = {
            "index": check_recipe_index,
            "cuisine": check_cuisine_distribution,
        }
        for name, check_fn in data_checks.items():
            result = check_fn()
            if result.get("status") == "error":
                blocking.append(name)
    
    if not blocking:
        return {
            "ready": True,
            "blocking": [],
            "message": "Ready to plan meals"
        }
    else:
        return {
            "ready": False,
            "blocking": blocking,
            "message": f"{len(blocking)} issue{'s' if len(blocking) != 1 else ''} to resolve"
        }


def run_all_checks() -> Dict[str, Dict[str, Any]]:
    """Run all health checks and return results."""
    return {
        "config": check_config_file(),
        "mealie": check_mealie_connection(),
        "openrouter": check_openrouter_api(),
        "embedding": check_embedding_model(),
        "index": check_recipe_index(),
        "sync_staleness": check_sync_staleness(),
        "cuisine": check_cuisine_distribution(),
        "recipe_quality": check_recipe_quality(),
        "mealie_tags": check_mealie_tags(),
    }
