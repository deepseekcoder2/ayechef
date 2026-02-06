"""
Tool Registry - SINGLE source of truth for all CLI tools.

Adding a new tool = adding ONE dict entry here.
No new routes. No new templates. It auto-appears in the UI.

Tool Groups:
- 'routine': Regular maintenance tasks (safe, run often)
- 'cleanup': Data cleanup tasks (destructive, run when needed)
- 'advanced': Recovery/debug tools (hidden by default)
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional

# Tool groups for UI organization
ToolGroup = Literal['routine', 'cleanup', 'advanced']

@dataclass
class ToolParam:
    """Parameter definition for a tool."""
    name: str
    label: str
    type: str  # 'text', 'date', 'checkbox', 'select'
    required: bool = False
    default: Optional[str] = None
    prefix: Optional[str] = None  # e.g., '--diagnose=' for CLI arg
    options: Optional[List[str]] = None  # for 'select' type

@dataclass
class Tool:
    """Tool definition."""
    id: str
    name: str
    description: str
    command: List[str]  # base command
    category: str  # 'planning', 'import', 'maintenance'
    group: Optional[ToolGroup] = None  # UI grouping for maintenance tools
    params: List[ToolParam] = field(default_factory=list)
    dangerous: bool = False  # shows warning indicator
    confirm_message: Optional[str] = None  # if set, shows confirmation modal before running

# ═══════════════════════════════════════════════════════════════════════════════
# ALL TOOLS DEFINED HERE - This is the ONLY place to add/modify tools
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS: dict[str, Tool] = {
    # === PLANNING ===
    'plan_week': Tool(
        id='plan_week',
        name='Plan This Week',
        description='Generate meal plan and shopping list',
        command=['python', 'orchestrator.py'],
        category='planning',
        params=[
            ToolParam('start_date', 'Week starting', 'date', required=True, prefix='--week-start='),
            ToolParam('cuisines', 'Preferred cuisines', 'text', prefix='--cuisines='),
            ToolParam('restrictions', 'Dietary restrictions', 'text', prefix='--restrictions='),
            ToolParam('temp_prompt', 'Special instructions', 'text', prefix='--temp-prompt='),
            ToolParam('dry_run', 'Diagnostic mode (no changes)', 'checkbox', prefix='--dry-run'),
        ]
    ),
    
    # === IMPORT ===
    'import_recipe': Tool(
        id='import_recipe',
        name='Import Recipe',
        description='Import a single recipe with full processing',
        command=['python', 'import_recipe.py'],
        category='import',
        params=[
            ToolParam('url', 'Recipe URL', 'text', required=True),
        ]
    ),
    'import_site': Tool(
        id='import_site',
        name='Import from Website',
        description='Discover and fully import recipes from a website (scrape → parse → tag → index)',
        command=['python', 'import_site.py', '--yes'],  # Auto-confirm for job execution
        category='import',
        params=[
            ToolParam('url', 'Website URL', 'text', required=True),
            ToolParam('sitemap', 'Use sitemap (recommended)', 'checkbox', default='true', prefix='--sitemap'),
            ToolParam('categories', 'Collections to import (comma-separated)', 'text', prefix='--categories='),
        ]
    ),
    'add_site': Tool(
        id='add_site',
        name='Learn New Site',
        description='Analyze a website to enable recipe imports',
        command=['python', 'add_site.py', '--yes'],  # Auto-confirm for job execution
        category='import',
        params=[
            ToolParam('url', 'Website URL', 'text', required=True),
        ]
    ),
    'test_import': Tool(
        id='test_import',
        name='Test Import',
        description='Test import a few recipes to verify scraper works',
        command=['python', 'test_import.py'],
        category='import',
        params=[
            ToolParam('url', 'Website URL', 'text', required=True),
            ToolParam('count', 'Number of recipes', 'text', default='10', prefix='--count='),
            ToolParam('job_id', 'Job ID', 'text', prefix='--job-id='),
            ToolParam('new_count', 'Total new recipes', 'text', prefix='--new-count='),
        ]
    ),
    
    # =========================================================================
    # MAINTENANCE TOOLS - Organized by group for UI rendering
    # =========================================================================
    
    # === ROUTINE: Safe tasks to run regularly ===
    'repair_optimize': Tool(
        id='repair_optimize',
        name='Repair & Optimize',
        description='Scans for issues and fixes them: parses ingredients, syncs with Mealie, tags recipes, rebuilds search. Run after importing recipes or if search seems off.',
        command=['python', 'utils/recipe_maintenance.py'],
        category='maintenance',
        group='routine',
    ),
    'health_check': Tool(
        id='health_check',
        name='Health Check',
        description='Checks for problems without making changes. Shows unparsed recipes and sync status. Run to diagnose issues before taking action.',
        command=['python', 'utils/recipe_maintenance.py', '--quick'],
        category='maintenance',
        group='routine',
    ),
    'backfill_images': Tool(
        id='backfill_images',
        name='Add Missing Images',
        description='Fetches images from web search for recipes that don\'t have one. Run after imports or for AI-generated recipes.',
        command=['python', 'recipe_images.py', '--backfill'],
        category='maintenance',
        group='routine',
    ),
    
    # === CLEANUP: Destructive tasks that delete data ===
    'cleanup_recipes': Tool(
        id='cleanup_recipes',
        name='Remove Bad Recipes',
        description='Deletes duplicates (names ending in (1), (2)) and invalid recipes (missing ingredients or instructions). Run after bulk imports that created duplicates.',
        command=['python', 'utils/cleanup_duplicates.py', '--all', '--confirm'],
        category='maintenance',
        group='cleanup',
        dangerous=True,
        confirm_message='This will permanently delete duplicate recipes (names with (1), (2), etc.) and invalid recipes (missing ingredients or instructions). This cannot be undone.',
    ),
    'cleanup_meals': Tool(
        id='cleanup_meals',
        name='Clear Meal History',
        description='Deletes ALL meal plans and shopping lists. Recipes are kept. Run to start fresh with meal planning.',
        command=['python', 'utils/cleanup_meal_data.py', '--confirm'],
        category='maintenance',
        group='cleanup',
        dangerous=True,
        confirm_message='This will permanently delete ALL meal plans and shopping lists. Your recipes will NOT be affected. This cannot be undone.',
    ),
    
    # === ADVANCED: Recovery and debugging tools (hidden by default) ===
    'rebuild_search': Tool(
        id='rebuild_search',
        name='Rebuild Search Index',
        description='Rebuilds semantic and text search indexes from scratch. Only run if search returns wrong results or errors.',
        command=['python', 'utils/rebuild_search_index.py'],
        category='maintenance',
        group='advanced',
        dangerous=True,
        confirm_message='This will rebuild all search indexes from scratch. This is a recovery operation that may take several minutes.',
    ),
    'clear_cache': Tool(
        id='clear_cache',
        name='Clear Planning Cache',
        description='Clears cached meal history used to avoid repeating recent meals. Run if suggestions seem stuck on old patterns.',
        command=['python', 'tools/invalidate_cache.py'],
        category='maintenance',
        group='advanced',
    ),
    'fix_equipment': Tool(
        id='fix_equipment',
        name='Fix Equipment Labels',
        description='Corrects kitchen tools (blender, pan) incorrectly imported as food ingredients. Run if shopping lists include equipment.',
        command=['python', 'utils/label_equipment.py', '--apply'],
        category='maintenance',
        group='advanced',
    ),
    'diagnose_recipe': Tool(
        id='diagnose_recipe',
        name='Debug Recipe',
        description='Shows detailed diagnostic for one recipe: tags, parsing status, index status. Use to troubleshoot a specific recipe.',
        command=['python', 'automatic_tagger.py'],
        category='maintenance',
        group='advanced',
        params=[
            ToolParam('slug', 'Recipe slug', 'text', required=True, prefix='--diagnose='),
        ]
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_tools_by_category() -> dict[str, List[Tool]]:
    """Group tools by category for UI rendering."""
    grouped = {}
    for tool in TOOLS.values():
        if tool.category not in grouped:
            grouped[tool.category] = []
        grouped[tool.category].append(tool)
    return grouped


def get_maintenance_tools_by_group() -> dict[ToolGroup, List[Tool]]:
    """
    Get maintenance tools organized by group for UI rendering.
    
    Returns:
        Dict with keys 'routine', 'cleanup', 'advanced' containing lists of tools.
        Tools without a group are excluded.
    """
    grouped: dict[ToolGroup, List[Tool]] = {
        'routine': [],
        'cleanup': [],
        'advanced': [],
    }
    for tool in TOOLS.values():
        if tool.category == 'maintenance' and tool.group:
            grouped[tool.group].append(tool)
    return grouped


def get_maintenance_tools() -> List[Tool]:
    """Get all maintenance tools as a flat list (for backward compatibility)."""
    return [t for t in TOOLS.values() if t.category == 'maintenance']

def build_command(tool: Tool, form_data: dict) -> List[str]:
    """Build full command from tool definition and user input."""
    cmd = list(tool.command)
    for param in (tool.params or []):
        value = form_data.get(param.name)
        # Use default value if no value provided and param has a default
        if not value and param.default is not None:
            value = param.default
        if value:
            if param.type == 'checkbox':
                if value == 'on' and param.prefix:
                    cmd.append(param.prefix.rstrip('='))
            elif param.prefix:
                if param.prefix.endswith('='):
                    cmd.append(f"{param.prefix}{value}")
                else:
                    cmd.extend([param.prefix, value])
            else:
                cmd.append(value)
    return cmd
