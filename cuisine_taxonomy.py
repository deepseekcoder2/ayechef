"""
Canonical cuisine taxonomy for tagging/planning.

This module is intentionally lightweight (no LLM/torch imports) so it can be used by:
- tag seeding scripts
- planners
- taggers

without pulling in heavy ML dependencies.
"""

from __future__ import annotations

from typing import Dict, List


def taxonomy_groups() -> Dict[str, List[str]]:
    """
    Canonical cuisine display names, grouped for readability.
    These values are the ONLY allowed cuisine strings in the cuisine classifier.
    """
    return {
        "Chinese Regional (PREFER SPECIFIC)": [
            "Cantonese",
            "Sichuan",
            "Hunan",
            "Dongbei",
            "Shanghai",
            "Beijing",
            "Fujian",
            "Hong Kong",
            "Taiwanese",
            "Hakka",
        ],
        "Westernized Chinese (for adapted dishes)": ["American Chinese", "British Chinese"],
        "Chinese Fusion": ["Indo-Chinese", "Macanese", "Peranakan"],
        "Other East Asian": ["Japanese", "Korean"],
        "Southeast Asian": ["Thai", "Vietnamese", "Malaysian", "Indonesian", "Singaporean", "Filipino"],
        "South Asian": ["Indian", "Punjabi"],
        "European": [
            "Italian",
            "Northern Italian",
            "Southern Italian",
            "French",
            "Provençal",
            "Spanish",
            "Greek",
            "British",
            "German",
            "Bavarian",
            "Austrian",
            "Swiss",
            "Polish",
            "Hungarian",
            "Scandinavian",
            "Portuguese",
            "Irish",
            "Russian",
        ],
        "Middle Eastern": ["Lebanese", "Turkish", "Persian", "Moroccan"],
        "Latin American": ["Mexican", "Brazilian", "Peruvian", "Tex-Mex"],
        "American": ["American", "Cajun", "Southern US", "Canadian"],
        "Caribbean": ["Jamaican", "Caribbean"],
        "Oceania": ["Australian", "New Zealand"],
        "Fusion/Pan-Regional": ["Fusion", "Pan-Asian"],
        "Generic (USE ONLY AS LAST RESORT)": ["Chinese", "Asian", "European", "Mediterranean"],
    }


def canonical_cuisine_names() -> List[str]:
    """
    Flat, canonical cuisine display names (stable ordering; de-duplicated).
    """
    flat: List[str] = []
    seen = set()
    for cuisines in taxonomy_groups().values():
        for c in cuisines:
            if c not in seen:
                seen.add(c)
                flat.append(c)
    return flat


def format_cuisine_tag_name(cuisine: str) -> str:
    return f"{cuisine} Cuisine"


def canonical_cuisine_tag_names() -> List[str]:
    return [format_cuisine_tag_name(c) for c in canonical_cuisine_names()]

