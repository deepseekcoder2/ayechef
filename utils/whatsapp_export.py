#!/usr/bin/env python3
"""
WhatsApp Export Script
Fetches shopping list and meal plan from Mealie, formats for WhatsApp, and copies to clipboard.
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import subprocess
from datetime import datetime, timedelta
from mealie_client import MealieClient
from config import MEALIE_URL
from tools.logging_utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# Unicode fraction mapping for cleaning display text
UNICODE_FRACTIONS = {
    '½': 0.5, '⅓': 1/3, '⅔': 2/3, '¼': 0.25, '¾': 0.75,
    '⅕': 0.2, '⅖': 0.4, '⅗': 0.6, '⅘': 0.8,
    '⅙': 1/6, '⅚': 5/6, '⅛': 0.125, '⅜': 0.375, '⅝': 0.625, '⅞': 0.875,
}

# Superscript/subscript digits for vulgar fractions like ¹/₁₀
SUPERSCRIPT_DIGITS = {'⁰': 0, '¹': 1, '²': 2, '³': 3, '⁴': 4, '⁵': 5, '⁶': 6, '⁷': 7, '⁸': 8, '⁹': 9}
SUBSCRIPT_DIGITS = {'₀': 0, '₁': 1, '₂': 2, '₃': 3, '₄': 4, '₅': 5, '₆': 6, '₇': 7, '₈': 8, '₉': 9}


def clean_unicode_fractions(text: str) -> str:
    """
    Replace unicode fractions with decimal equivalents.
    
    Examples:
        "1 ½ cups" -> "1.5 cups"
        "1 ¹/₁₀ kilograms" -> "1.1 kilograms"
        "2 ⁴/₁₉ kg" -> "2.21 kg"
    """
    import re
    
    # First handle vulgar fractions like ¹/₁₀ (superscript/subscript)
    # Pattern: optional whole number + space + superscript digits + / + subscript digits
    def replace_vulgar_fraction(match):
        whole = match.group(1) or ""
        super_str = match.group(2)
        sub_str = match.group(3)
        
        # Parse superscript numerator
        numerator = 0
        for c in super_str:
            if c in SUPERSCRIPT_DIGITS:
                numerator = numerator * 10 + SUPERSCRIPT_DIGITS[c]
        
        # Parse subscript denominator
        denominator = 0
        for c in sub_str:
            if c in SUBSCRIPT_DIGITS:
                denominator = denominator * 10 + SUBSCRIPT_DIGITS[c]
        
        if denominator == 0:
            return match.group(0)  # Can't divide by zero, return original
        
        fraction_value = numerator / denominator
        
        if whole:
            whole_num = float(whole.strip())
            total = whole_num + fraction_value
        else:
            total = fraction_value
        
        # Format nicely (remove unnecessary decimals)
        if total == int(total):
            return str(int(total))
        else:
            return f"{total:.2f}".rstrip('0').rstrip('.')
    
    # Match: optional "number " + superscript digits + "/" + subscript digits
    vulgar_pattern = r'(\d+\s+)?([⁰¹²³⁴⁵⁶⁷⁸⁹]+)/([₀₁₂₃₄₅₆₇₈₉]+)'
    text = re.sub(vulgar_pattern, replace_vulgar_fraction, text)
    
    # Then handle simple unicode fractions like ½, ¼, etc.
    for frac_char, frac_value in UNICODE_FRACTIONS.items():
        if frac_char in text:
            # Check if preceded by a whole number
            pattern = r'(\d+)\s*' + re.escape(frac_char)
            def replace_with_whole(m):
                whole = float(m.group(1))
                total = whole + frac_value
                if total == int(total):
                    return str(int(total))
                return f"{total:.2f}".rstrip('0').rstrip('.')
            
            text = re.sub(pattern, replace_with_whole, text)
            
            # Replace standalone fractions
            if frac_value == int(frac_value):
                text = text.replace(frac_char, str(int(frac_value)))
            else:
                text = text.replace(frac_char, f"{frac_value:.2f}".rstrip('0').rstrip('.'))
    
    return text


def fetch_shopping_list(client: MealieClient, list_id: str):
    """Fetch shopping list from Mealie API."""
    try:
        return client.get_shopping_list(list_id)
    except Exception as e:
        logger.error(f"❌ Error fetching shopping list: {e}")
        print(f"❌ Error fetching shopping list: {e}")
        sys.exit(1)


def fetch_meal_plan(client: MealieClient, start_date: str, end_date: str):
    """Fetch meal plan from Mealie API for the given date range."""
    try:
        plans = client.get_meal_plans(start_date, end_date)
        # Return in the same format as before (with 'items' key)
        return {"items": plans}
    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not fetch meal plan: {e}")
        print(f"⚠️ Warning: Could not fetch meal plan: {e}")
        return {"items": []}


def format_date(date_obj):
    """Format date as 'MON Dec 23'."""
    day_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    day_name = day_names[date_obj.weekday()]
    month_day = date_obj.strftime("%b %d")
    return f"{day_name} {month_day}"


def get_date_range_display(start_date):
    """Get date range display for header like 'Dec 23-29'."""
    end_date = start_date + timedelta(days=6)
    
    # If same month
    if start_date.month == end_date.month:
        return f"{start_date.strftime('%b')} {start_date.day}-{end_date.day}"
    else:
        return f"{start_date.strftime('%b %d')}-{end_date.strftime('%b %d')}"


def has_overnight_prep_tag(recipe):
    """Check if recipe has overnight preparation tags (legacy and new formats)."""
    if not recipe or 'tags' not in recipe:
        return False

    tags = recipe.get('tags', [])
    for tag in tags:
        if isinstance(tag, dict):
            tag_name = tag.get('name', '')
            # Check for both legacy and new formats
            if tag_name == 'Overnight Preparation' or 'Prep' in tag_name:
                return True
    return False


async def detect_overnight_prep_automatic(recipe):
    """
    DEPRECATED: This function is wasteful - it makes LLM calls for every recipe.
    Use has_overnight_prep_tag() instead to check existing tags.
    
    Kept for backwards compatibility but should not be used.
    """
    raise RuntimeError(
        "detect_overnight_prep_automatic() is deprecated and should not be called. "
        "Use has_overnight_prep_tag() instead to check existing recipe tags without LLM calls."
    )


def group_meal_plans_by_date(meal_plans):
    """Group meal plan entries by date."""
    grouped = {}
    
    for entry in meal_plans.get('items', []):
        date = entry.get('date')
        if not date:
            continue
        
        if date not in grouped:
            grouped[date] = []
        
        grouped[date].append(entry)
    
    return grouped


def detect_overnight_prep(grouped_plans, all_dates):
    """
    Detect overnight preparation requirements by checking recipe tags.
    Returns dict mapping date -> list of prep warnings for that day.
    """
    prep_warnings = {}

    # Create a mapping of date to previous date
    date_to_prev = {}
    for i in range(1, len(all_dates)):
        date_to_prev[all_dates[i]] = all_dates[i-1]

    # Check each meal for overnight prep requirements
    for date_str, entries in grouped_plans.items():
        for entry in entries:
            recipe = entry.get('recipe')
            if recipe:
                recipe_name = recipe.get('name', 'Unknown Recipe')

                # Check tags directly (no LLM calls needed - recipes already have tags)
                requires_overnight = has_overnight_prep_tag(recipe)

                if requires_overnight:
                    # Add warning to previous day
                    if date_str in date_to_prev:
                        prev_date = date_to_prev[date_str]
                        if prev_date not in prep_warnings:
                            prep_warnings[prev_date] = []

                        prep_warnings[prev_date].append(recipe_name)
                        print(f"⚠️  {recipe_name} needs advance prep (serving {date_str}, warning on {prev_date})")

    return prep_warnings


def clean_ingredient_display(raw_display):
    """
    Clean up messy ingredient display to show only purchasable items.

    Examples:
    - "2 chicken breasts" → "2 chicken breasts"
    - "1 tablespoon olive oil" → "1 tablespoon olive oil"
    - "to serve" → None (filter out)
    - "optional" → None (filter out)
    - "2 dr peppers deseeded and sliced" → "2 red peppers"
    - "200 grams trimmed and cut into 4 pieces" → None (incomplete)
    """
    if not raw_display or not raw_display.strip():
        return None

    display = raw_display.strip().lower()

    # Filter out non-purchasable items and incomplete entries
    non_purchasable = [
        'to serve', 'optional', 'for drizzling', 'freeze the whites',
        'we used', 'halved', 'finely chopped', 'roughly chopped',
        'thinly sliced', 'coarsely grated', 'picked and chopped',
        'skinned, stoned and chopped', 'cut into wedges', 'cut into 6 wedges',
        'sliced into wedges', 'drained and roughly chopped', 'large, thickly sliced',
        'small, finely chopped', 'small, chopped', 'medium, chopped',
        'finely sliced', 'deseeded and sliced', 'tough stalks removed and finely chopped',
        'finely sliced or shredded', 'sliced', 'trimmed and cut into 4 pieces',
        'cut into bite-sized pieces', 'juiced', 'chopped into small pieces',
        'at least 70% cocoa solids', 'unpeeled', 'shredded', 'blitzed to breadcrumbs',
        'topped and tailed, finely minced', 'or gochugaru works great too',
        'grated', 'minced', 'diced', 'chopped', 'sliced', 'peeled',
        'fresh or frozen', 'drained', 'rinsed', 'washed'
    ]

    if any(phrase in display for phrase in non_purchasable):
        return None

    # Filter out items that are just single words or incomplete
    words = display.split()
    if len(words) < 2:
        return None

    # Filter out items that start with numbers but have no clear food name
    # (like "1 chopped", "2 medium", etc.)
    if words[0].isdigit() and len(words) == 2 and words[1] in ['chopped', 'medium', 'large', 'small', 'finely']:
        return None

    # Handle common parsing errors
    display = display.replace('dr peppers', 'red peppers')
    display = display.replace('dr pepper', 'red pepper')

    # Clean up quantity/unit/food format
    # Remove multiple separators and keep only the main ingredient
    display = display.split('|')[0].strip()  # Take first part before |
    display = display.split(',')[0].strip()  # Take first part before ,
    display = display.split(' - ')[0].strip()  # Take first part before -

    # Remove "or" clauses (like "or gochugaru" alternatives)
    if ' or ' in display:
        display = display.split(' or ')[0].strip()

    # Capitalize first letter of each word for readability
    display = ' '.join(word.capitalize() for word in display.split())

    # Final filter: must have at least quantity + food, or unit + food
    words = display.split()
    if len(words) >= 2:
        # Check if it looks like a valid ingredient (number/unit + food)
        first_word = words[0].lower()
        if first_word.isdigit() or first_word in ['tablespoon', 'teaspoon', 'cup', 'gram', 'kilogram', 'milliliter', 'liter', 'can', 'clove', 'bunch']:
            return display

    return None  # Filter out if doesn't match expected format


def format_shopping_list(shopping_list_data, date_range_display):
    """
    Format shopping list as a simple flat list.
    
    No categories - just a clean alphabetized list of items.
    Categories are inconsistent (many items end up in "Other") so they
    confuse more than they help.
    """
    lines = []
    lines.append(f"🛒 SHOPPING LIST ({date_range_display})")
    lines.append("━━━━━━━━━━━━━━━━")

    list_items = shopping_list_data.get('listItems', [])
    if not list_items:
        lines.append("(No items in shopping list)")
    else:
        # Collect all items as a flat list
        all_items = []
        
        for item in list_items:
            display = item.get('display', '').strip()
            if display:
                # Clean unicode fractions (e.g., "1 ¹/₁₀ kg" -> "1.1 kg")
                display = clean_unicode_fractions(display)
                all_items.append(display)
        
        if not all_items:
            lines.append("(No valid items in shopping list)")
        else:
            # Remove duplicates and sort alphabetically
            unique_items = sorted(list(set(all_items)))
            
            lines.append("")
            for item in unique_items:
                lines.append(f"☐ {item}")

    lines.append("\n━━━━━━━━━━━━━━━━")
    lines.append("")

    return lines


def format_meal_plan(grouped_plans, all_dates, prep_warnings):
    """Format meal plan section."""
    lines = []
    lines.append("📅 THIS WEEK'S MENU")
    lines.append("")
    
    if not grouped_plans:
        lines.append("(No meals planned for this week)")
        return lines
    
    for date_str in all_dates:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        lines.append(format_date(date_obj))
        
        entries = grouped_plans.get(date_str, [])
        
        if not entries:
            lines.append("(No meals planned)")
        else:
            # Group by meal type
            meals_by_type = {}
            for entry in entries:
                entry_type = entry.get('entryType', 'other')
                if entry_type not in meals_by_type:
                    meals_by_type[entry_type] = []
                meals_by_type[entry_type].append(entry)
            
            # Sort meal types (lunch before dinner)
            meal_order = ['breakfast', 'lunch', 'dinner', 'side']
            sorted_types = sorted(meals_by_type.keys(), 
                                key=lambda x: meal_order.index(x) if x in meal_order else 999)
            
            for meal_type in sorted_types:
                meal_entries = meals_by_type[meal_type]
                lines.append(f"🍽️ {meal_type.capitalize()}:")
                
                for entry in meal_entries:
                    recipe = entry.get('recipe')
                    if recipe:
                        recipe_name = recipe.get('name', 'Unknown Recipe')
                        recipe_slug = recipe.get('slug', '')
                        recipe_url = f"{MEALIE_URL}/g/home/r/{recipe_slug}" if recipe_slug else ""
                        
                        if recipe_url:
                            lines.append(f"{recipe_name} ({recipe_url})")
                        else:
                            lines.append(recipe_name)
                    else:
                        # Note entry (PREP/BUY item) - show title
                        title = entry.get('title', '')
                        if title:
                            lines.append(title)
        
        # Add prep warnings for this day
        if date_str in prep_warnings:
            for recipe_name in prep_warnings[date_str]:
                lines.append(f"⚠️ PREP TONIGHT: {recipe_name} needs advance preparation")
        
        lines.append("")
    
    return lines


def copy_to_clipboard(text):
    """Copy text to macOS clipboard using pbcopy."""
    try:
        subprocess.run('pbcopy', text=True, input=text, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.info(f"⚠️ Warning: Could not copy to clipboard: {e}")
        print(f"⚠️ Warning: Could not copy to clipboard: {e}")
        print("(This is expected on non-Mac systems)")
        return False


def save_to_file(text, filename="meal_plan.txt"):
    """Save message to file as backup (in data/ directory for Docker compatibility)."""
    from config import DATA_DIR
    
    # Write to data/ directory (writable in Docker)
    DATA_DIR.mkdir(exist_ok=True)
    filepath = DATA_DIR / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Message saved to: {filepath}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not save to file: {e}")
        print(f"⚠️ Warning: Could not save to file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Export Mealie shopping list and meal plan to WhatsApp format'
    )
    parser.add_argument('--list-id', required=True, help='Shopping list UUID')
    parser.add_argument('--week-start', required=True, help='Week start date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Parse and validate week start date
    try:
        week_start = datetime.strptime(args.week_start, "%Y-%m-%d")
    except ValueError:
        print("❌ Error: Invalid date format. Use YYYY-MM-DD (e.g., 2025-12-23)")
        sys.exit(1)
    
    # Calculate week end date
    week_end = week_start + timedelta(days=6)
    
    # Generate all dates in range
    all_dates = []
    current = week_start
    while current <= week_end:
        all_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    # Get date range display
    date_range_display = get_date_range_display(week_start)
    
    # Initialize MealieClient
    client = MealieClient()
    try:
        print("Fetching shopping list...")
        shopping_list_data = fetch_shopping_list(client, args.list_id)
        
        print(f"Fetching meal plan for {date_range_display}...")
        meal_plan_data = fetch_meal_plan(
            client,
            week_start.strftime("%Y-%m-%d"),
            week_end.strftime("%Y-%m-%d")
        )
    
        print("Formatting message...")
        
        # Group meal plans by date
        grouped_plans = group_meal_plans_by_date(meal_plan_data)
        
        print("Checking for overnight prep requirements...")
        prep_warnings = detect_overnight_prep(grouped_plans, all_dates)
        
        # Build message
        message_lines = []
        
        # Shopping list section
        message_lines.extend(format_shopping_list(shopping_list_data, date_range_display))
        
        # Meal plan section
        message_lines.extend(format_meal_plan(grouped_plans, all_dates, prep_warnings))
        
        # Join all lines
        message = '\n'.join(message_lines)
        
        # Save to file
        save_to_file(message)
        
        # Copy to clipboard
        if copy_to_clipboard(message):
            print("Message copied to clipboard! Ready to paste in WhatsApp.")
        else:
            print("Message ready in meal_plan.txt (clipboard copy not available)")
        
        print("\n✅ Done!")
    finally:
        client.close()


async def enhanced_whatsapp_export(list_id: str, week_start: datetime) -> str:
    """
    Export refined Mealie shopping list as a simple flat list.

    No categories - just a clean alphabetized list. Categories are inconsistent
    (many items end up in "Other") so they confuse more than they help.

    Args:
        list_id: Mealie shopping list ID
        week_start: Start date of the week

    Returns:
        str: WhatsApp-formatted message with shopping list
    """
    try:
        from mealie_shopping_integration import fetch_mealie_shopping_list

        print(f"📱 Fetching refined shopping list {list_id}...")

        # Fetch refined shopping list
        shopping_list_data = fetch_mealie_shopping_list(list_id)
        if not shopping_list_data:
            print("❌ Failed to fetch shopping list")
            return "❌ Failed to fetch shopping list"

        # Calculate date range
        date_range_display = get_date_range_display(week_start)

        # Format as simple flat list
        message_lines = []

        # Header
        message_lines.append("🛒 SHOPPING LIST")
        message_lines.append(f"📅 {date_range_display}")
        message_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        list_items = shopping_list_data.get('listItems', [])
        if not list_items:
            message_lines.append("📝 No items in shopping list")
        else:
            # Collect all items as a flat list
            all_items = []
            for item in list_items:
                display = item.get('display', '').strip()
                if display:
                    # Clean unicode fractions (e.g., "1 ¹/₁₀ kg" -> "1.1 kg")
                    display = clean_unicode_fractions(display)
                    all_items.append(display)

            if not all_items:
                message_lines.append("📝 No valid items in shopping list")
            else:
                # Remove duplicates and sort alphabetically
                unique_items = sorted(list(set(all_items)))
                
                message_lines.append("")
                for item in unique_items:
                    message_lines.append(f"☐ {item}")

        message_lines.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        message_lines.append("✅ Pantry staples already filtered out")

        # Add meal plan section if available
        try:
            # Note: This function is async and may need a client passed in
            # For now, we'll use the same pattern but need to handle client
            # Since this is an async function, we'll need to pass client when called
            from mealie_client import MealieClient
            client = MealieClient()
            try:
                meal_plan_data = fetch_meal_plan(
                    client,
                    week_start.strftime("%Y-%m-%d"),
                    (week_start + timedelta(days=6)).strftime("%Y-%m-%d")
                )
            finally:
                client.close()

            if meal_plan_data.get('items'):
                message_lines.append("\n\n📅 THIS WEEK'S MENU")
                message_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                # Group meal plans by date
                grouped_plans = {}
                all_dates = []

                for item in meal_plan_data['items']:
                    date_str = item['date']
                    if date_str not in grouped_plans:
                        grouped_plans[date_str] = []
                        all_dates.append(date_str)
                    grouped_plans[date_str].append(item)

                all_dates.sort()

                for date_str in all_dates:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    message_lines.append(f"\n{format_date(date_obj)}")

                    entries = grouped_plans.get(date_str, [])
                    for entry in entries:
                        recipe = entry.get('recipe')
                        if recipe:
                            item_title = recipe.get('name', 'Unknown Recipe')
                            # Check for overnight prep
                            if has_overnight_prep_tag(recipe):
                                item_title += " 🌙"
                        else:
                            # Note entry (PREP/BUY item)
                            item_title = entry.get('title', '')
                        
                        if item_title:
                            meal_type = entry.get('entryType', 'meal').title()
                            message_lines.append(f"   {meal_type}: {item_title}")

        except Exception as e:
            print(f"⚠️  Could not fetch meal plan: {e}")
            message_lines.append("\n\n📅 Meal plan not available")

        # Join and return message
        message = '\n'.join(message_lines)

        # Save to file
        save_to_file(message)

        print("✅ Enhanced WhatsApp export complete")
        return message

    except Exception as e:
        error_msg = f"❌ Enhanced export failed: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


if __name__ == "__main__":
    main()

