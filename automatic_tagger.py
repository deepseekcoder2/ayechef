#!/usr/bin/env python3
"""
Automatic Tagging System for Mealie Recipes
============================================

Intelligent dual tagging engine that automatically detects:
1. Overnight preparation requirements by parsing recipe instructions
2. Specific regional cuisine classifications using multi-modal analysis

Leverages local LLM (nemotron-3-nano) for sophisticated natural language understanding
and Mealie API for seamless tag management.

Features:
- LLM-powered instruction parsing for prep time detection
- Multi-layer cuisine classification (name, ingredients, methods, cultural context)
- Comprehensive regional taxonomy (50+ specific regions)
- Automatic Mealie tag creation and management
- Confidence scoring and fallback handling
- Backward compatibility with manual tags

Usage:
    from automatic_tagger import AutomaticTagger

    tagger = AutomaticTagger()
    analysis = tagger.analyze_recipe(recipe_data)
    tagger.apply_tags_to_mealie(recipe_id, analysis)
"""

import json
import re
import os
import asyncio
import sqlite3
import threading
from typing import Dict, List, Tuple, Optional, Any, NamedTuple, Set
from dataclasses import dataclass
from config import get_bulk_operation_config_safe, DATA_DIR
from mealie_client import MealieClient, MealieClientError
from batch_llm_processor import get_llm_cache
from tools.logging_utils import get_logger
from prompts import PREP_ANALYSIS_PROMPT, CUISINE_ANALYSIS_PROMPT
from cuisine_taxonomy import canonical_cuisine_names, taxonomy_groups

# Initialize logger for this module
logger = get_logger(__name__)


class TagFormatter:
    """
    Centralized tag naming utility to ensure all tags follow Mealie API requirements.

    FORBIDDEN FORMAT: PREFIX: WORD (causes 500 errors in some Mealie versions/endpoints)
    ACCEPTED FORMATS:
    - WORD SUFFIX (e.g., "Italian Cuisine", "Overnight Prep", "Sichuan Region")
    """

    @staticmethod
    def create_prep_tag(prep_type: str = "Overnight") -> str:
        """Create prep tags in accepted format."""
        return f"{prep_type} Prep"

    @staticmethod
    def create_region_tag(region: str) -> str:
        """Create region tags in accepted format."""
        return f"{region} Region"

    @staticmethod
    def create_cuisine_tag(cuisine: str, region: str = None) -> str:
        """Create cuisine tags in Mealie-accepted format."""
        # AVOID "Cuisine: X - Y" format - causes 500 errors in Mealie
        # Use only "X Cuisine" format which is accepted by Mealie API
        # NOTE: We intentionally do NOT combine region strings into cuisine tags.
        # Regional specificity must be represented in the cuisine name itself
        # (e.g., "Northern Italian", "Sichuan", "American Chinese") to avoid
        # combinatorial tag drift and duplicates.
        return f"{cuisine} Cuisine"

    @staticmethod
    def validate_tag_format(tag_name: str) -> bool:
        """Validate that a tag name follows accepted formats."""
        # Reject any tag with the forbidden PREFIX: WORD pattern
        if ": " in tag_name and not " - " in tag_name:
            return False
        return True

    @staticmethod
    def ensure_valid_tag(tag_name: str) -> str:
        """Ensure a tag name is valid, converting if necessary."""
        if not TagFormatter.validate_tag_format(tag_name):
            # Convert forbidden formats to accepted ones
            if tag_name.startswith("Prep: "):
                word = tag_name.split(": ")[1]
                return TagFormatter.create_prep_tag(word)
            elif tag_name.startswith("Region: "):
                word = tag_name.split(": ")[1]
                return TagFormatter.create_region_tag(word)
            elif tag_name.startswith("Cuisine: "):
                # Legacy format - convert to accepted format
                word = tag_name.split(": ")[1]
                # Remove region specifier if present (e.g., "Italian - Southern" -> "Italian")
                if " - " in word:
                    word = word.split(" - ")[0]
                return TagFormatter.create_cuisine_tag(word)
            else:
                # QUALITY FIRST: Unknown malformed format - should not occur with proper input validation
                raise ValueError(f"Tag name in unknown malformed format: '{tag_name}'. "
                               "This should not occur with proper input validation from LLM.")
        return tag_name

    @staticmethod
    def validate_cuisine_name(cuisine: str) -> bool:
        """Validate that a cuisine name follows required format (no colons, clean format)."""
        if not cuisine or not isinstance(cuisine, str):
            return False
        # Reject any cuisine name with colon (malformed LLM output)
        if ":" in cuisine:
            return False
        # Reject empty or whitespace-only strings
        if not cuisine.strip():
            return False
        return True

    @staticmethod
    def get_tag_type(tag_name: str) -> str:
        """Get the type of tag (cuisine, region, prep) from the name."""
        # Our canonical formats use suffixes ("X Cuisine", "X Region", "X Prep")
        if isinstance(tag_name, str) and tag_name.endswith(" Cuisine"):
            return "cuisine"
        elif isinstance(tag_name, str) and tag_name.endswith(" Region"):
            return "region"
        elif isinstance(tag_name, str) and tag_name.endswith(" Prep"):
            return "prep"
        else:
            return "other"


@dataclass
class PrepAnalysis:
    """Analysis result for preparation requirements."""
    requires_overnight_prep: bool
    prep_duration_hours: Optional[int]
    prep_type: Optional[str]  # 'marinate', 'soak', 'chill', 'rest', etc.
    confidence: float
    reasoning: str
    detected_patterns: List[str]


@dataclass
class CuisineClassification:
    """Cuisine classification result."""
    primary_cuisine: Optional[str]
    secondary_cuisines: List[str]
    region_specific: Optional[str]
    confidence: float
    reasoning: str
    detection_sources: List[str]


@dataclass
class TaggingAnalysis:
    """Complete analysis result for a recipe."""
    prep_analysis: PrepAnalysis
    cuisine_analysis: CuisineClassification
    recommended_tags: List[str]
    confidence_score: float


class AutomaticTagger:
    """
    Comprehensive automatic tagging engine using local LLM for intelligent analysis.
    """

    def __init__(self, client: Optional[MealieClient] = None):
        """
        Initialize the automatic tagger with comprehensive cuisine taxonomy.
        
        Args:
            client: Optional MealieClient instance. If None, creates a new one.
        """
        self.cuisine_taxonomy = self._load_cuisine_taxonomy()
        self.prep_patterns = self._load_prep_patterns()
        self._tag_cache_lock = threading.Lock()  # Thread-safe tag cache access
        self._category_cache_lock = threading.Lock()  # Thread-safe category cache access
        self.client = client if client is not None else MealieClient()
        self._client_owned = client is None  # Track if we own the client

    def _load_cuisine_taxonomy(self) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive cuisine taxonomy with regional specificity.
        Organized by continent/region with specific sub-regions and indicators.
        """
        return {
            # ASIAN CUISINES
            "chinese_cantonese": {
                "name": "Cantonese",
                "region": "Chinese",
                "subregion": "Guangdong",
                "indicators": {
                    "names": ["cantonese", "guangdong", "hong kong", "dim sum", "yum cha"],
                    "ingredients": ["oyster sauce", "hoisin", "shaoxing wine", "doubanjiang", "fermented black beans"],
                    "methods": ["steam", "stir-fry", "roast goose", "clay pot"],
                    "dishes": ["char siu", "wonton", "congee", "egg tart"]
                }
            },
            "chinese_sichuan": {
                "name": "Sichuan",
                "region": "Chinese",
                "subregion": "Sichuan",
                "indicators": {
                    "names": ["sichuan", "szechuan", "chongqing", "chengdu"],
                    "ingredients": ["sichuan peppercorn", "doubanjiang", "gochugaru", "pixian", "er jing tiao"],
                    "methods": ["numbing", "spicy", "dry pot", "twice cooked"],
                    "dishes": ["mapo tofu", "kung pao", "dan dan", "fish fragrant"]
                }
            },
            "chinese_dongbei": {
                "name": "Dong Bei",
                "region": "Chinese",
                "subregion": "Northeast",
                "indicators": {
                    "names": ["dong bei", "dongbei", "northeast china", "heilongjiang", "jilin", "liaoning"],
                    "ingredients": ["sauerkraut", "pickled cabbage", "cornmeal", "buckwheat", "sticky rice"],
                    "methods": ["pickle", "ferment", "stew", "dumpling"],
                    "dishes": ["sauerkraut fish", "nong min stew", "baba ganoush", "cold noodles"]
                }
            },
            "chinese_hunan": {
                "name": "Hunan",
                "region": "Chinese",
                "subregion": "Hunan",
                "indicators": {
                    "names": ["hunan", "xiang", "changsha"],
                    "ingredients": ["chili oil", "cumin", "broad bean paste", "coppa chili"],
                    "methods": ["dry fry", "spicy stir-fry", "smoked"],
                    "dishes": ["stir-fried meat and chili", "腊味", "腊肠"]
                }
            },
            "japanese_tokyo": {
                "name": "Tokyo",
                "region": "Japanese",
                "subregion": "Kanto",
                "indicators": {
                    "names": ["tokyo", "edomae", "kanto", "shojin"],
                    "ingredients": ["nigiri", "tamagoyaki", "uni", "hamachi"],
                    "methods": ["sushi", "tempura", "kaiseki", "washoku"],
                    "dishes": ["sukiyaki", "monjayaki", "anko"]
                }
            },
            "korean_seoul": {
                "name": "Seoul",
                "region": "Korean",
                "subregion": "Gyeonggi",
                "indicators": {
                    "names": ["seoul", "gyeonggi", "hanjeongsik"],
                    "ingredients": ["gochujang", "doenjang", "kimchi", "bulgogi"],
                    "methods": ["ferment", "grill", "bibimbap", "korean bbq"],
                    "dishes": ["bibimbap", "samgyetang", "dakgalbi"]
                }
            },
            "thai_central": {
                "name": "Central Thai",
                "region": "Thai",
                "subregion": "Bangkok",
                "indicators": {
                    "names": ["bangkok", "central thai", "ayutthaya"],
                    "ingredients": ["fish sauce", "palm sugar", "kaffir lime", "galangal"],
                    "methods": ["curry", "stir-fry", "tom yum", "pad thai"],
                    "dishes": ["pad thai", "tom yum", "green curry"]
                }
            },
            "indian_punjabi": {
                "name": "Punjabi",
                "region": "Indian",
                "subregion": "Punjab",
                "indicators": {
                    "names": ["punjabi", "punjab", "amritsar", "ludhiana"],
                    "ingredients": ["ghee", "cumin", "cardamom", "fenugreek", "ajwain"],
                    "methods": ["butter chicken", "tandoori", "dum", "kadai"],
                    "dishes": ["butter chicken", "dal makhani", "chole"]
                }
            },

            # EUROPEAN CUISINES
            "italian_northern": {
                "name": "Northern Italian",
                "region": "Italian",
                "subregion": "Piedmont/Lombardy",
                "indicators": {
                    "names": ["piedmont", "lombardy", "milan", "turin", "risotto"],
                    "ingredients": ["arborio rice", "barolo wine", "fontina", "porcini"],
                    "methods": ["risotto", "polenta", "ossobuco", "vitello"],
                    "dishes": ["risotto alla milanese", "ossobuco", "polenta"]
                }
            },
            "italian_southern": {
                "name": "Southern Italian",
                "region": "Italian",
                "subregion": "Campania/Sicily",
                "indicators": {
                    "names": ["campania", "sicily", "naples", "sicilian", "calabrian"],
                    "ingredients": ["san marzano tomatoes", "mozzarella", "basil", "orecchiette"],
                    "methods": ["pizza", "pasta", "caprese", "pesto"],
                    "dishes": ["margherita pizza", "pasta al pomodoro", "cacio e pepe"]
                }
            },
            "french_provencal": {
                "name": "Provencal",
                "region": "French",
                "subregion": "Provence",
                "indicators": {
                    "names": ["provence", "provencal", "marseille", "nice"],
                    "ingredients": ["herbes de provence", "olive oil", "lavender", "ratatouille"],
                    "methods": ["ratatouille", "bouillabaisse", "tapenade", "socca"],
                    "dishes": ["ratatouille", "bouillabaisse", "salade nicoise"]
                }
            },

            # LATIN AMERICAN CUISINES
            "mexican_northern": {
                "name": "Northern Mexican",
                "region": "Mexican",
                "subregion": "Sonora/Chihuahua",
                "indicators": {
                    "names": ["sonora", "chihuahua", "tijuana", "northern mexican"],
                    "ingredients": ["poblano", "menudo", "machaca", "carne asada"],
                    "methods": ["carne asada", "menudo", "chimichanga", "salsa"],
                    "dishes": ["carne asada", "menudo", "machaca"]
                }
            },
            "brazilian_bahian": {
                "name": "Bahian",
                "region": "Brazilian",
                "subregion": "Bahia",
                "indicators": {
                    "names": ["bahia", "bahian", "salvador", "recife"],
                    "ingredients": ["dende oil", "coconut milk", "shrimp", "cassava"],
                    "methods": ["moqueca", "vatapa", "acaraje", "feijoada"],
                    "dishes": ["moqueca", "acaraje", "vatapa"]
                }
            },

            # MIDDLE EASTERN CUISINES
            "lebanese": {
                "name": "Lebanese",
                "region": "Middle Eastern",
                "subregion": "Lebanon",
                "indicators": {
                    "names": ["lebanese", "beirut", "lebanon"],
                    "ingredients": ["tahini", "sumac", "za'atar", "pomegranate"],
                    "methods": ["mezze", "kibbeh", "tabouli", "falafel"],
                    "dishes": ["hummus", "tabouli", "falafel"]
                }
            },

            # AMERICAN CUISINES
            "cajun": {
                "name": "Cajun",
                "region": "American",
                "subregion": "Louisiana",
                "indicators": {
                    "names": ["cajun", "creole", "louisiana", "new orleans"],
                    "ingredients": ["cajun seasoning", "andouille", "crawfish", "okra"],
                    "methods": ["gumbo", "jambalaya", "etouffee", "blackening"],
                    "dishes": ["gumbo", "jambalaya", "crawfish etouffee"]
                }
            },

            # FUSION AND MODERN
            "california_cuisine": {
                "name": "California Cuisine",
                "region": "Fusion",
                "subregion": "Modern American",
                "indicators": {
                    "names": ["california", "californian", "farm to table", "alice waters"],
                    "ingredients": ["seasonal", "local", "organic", "microgreens"],
                    "methods": ["farm to table", "seasonal", "light", "fresh"],
                    "dishes": ["grilled fish", "seasonal vegetables", "fresh pasta"]
                }
            }
        }

    def _load_prep_patterns(self) -> Dict[str, Any]:
        """
        Comprehensive prep time detection patterns.
        Focus on overnight and extended preparation requirements.
        """
        return {
            "time_keywords": [
                "overnight", "24 hours", "48 hours", "72 hours", "next day",
                "following day", "tomorrow", "two days", "three days"
            ],
            "prep_actions": [
                "marinate", "soak", "chill", "refrigerate", "rest", "cure",
                "brine", "pickle", "ferment", "age", "dry", "air dry"
            ],
            "time_patterns": [
                r'(\d+)\s*(?:to|[-])\s*(\d+)\s*hours?',
                r'(\d+)\s*hours?',
                r'(\d+)\s*days?',
                r'at least (\d+)\s*hours?',
                r'up to (\d+)\s*hours?'
            ],
            "overnight_threshold": 8  # Hours - anything 8+ is considered overnight prep
        }

    def _get_prep_analysis_schema(self) -> dict:
        """
        JSON schema for prep time analysis structured output.
        Uses only basic JSON Schema features supported by LM Studio.
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "prep_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "requires_overnight_prep": {
                            "type": "boolean"
                        },
                        "prep_duration_hours": {
                            "type": ["number", "null"]
                        },
                        "prep_type": {
                            "type": ["string", "null"]
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "reasoning": {
                            "type": "string"
                        },
                        "detected_patterns": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["requires_overnight_prep", "confidence", "reasoning", "detected_patterns"]
                }
            }
        }

    def _is_known_cuisine(self, cuisine_name: str) -> bool:
        """
        Check if a cuisine name is in our known cuisine taxonomy.

        Uses the canonical taxonomy from cuisine_taxonomy.py as the single source of truth,
        plus additional broad categories that may be returned by LLMs.

        Args:
            cuisine_name: The cuisine name to validate

        Returns:
            True if cuisine is known, False otherwise
        """
        if not cuisine_name or not isinstance(cuisine_name, str):
            return False

        # Normalize to lowercase for comparison
        normalized = cuisine_name.lower().strip()

        # Build known cuisines from canonical taxonomy (SINGLE SOURCE OF TRUTH)
        # This prevents drift between the taxonomy file and this validation
        known_cuisines = {c.lower() for c in canonical_cuisine_names()}
        
        # Add broad categories that LLMs may return (not in canonical list but acceptable)
        broad_categories = {
            # Broad regional categories
            'asian', 'east asian', 'southeast asian', 'south asian', 'central asian',
            'european', 'western', 'eastern european', 'north american', 'oceania',
            'latin american', 'middle eastern',
            
            # Generic/fallback
            'generic', 'international', 'world cuisine',
            
            # Accented variations
            'provencal',  # provençal without accent
        }
        known_cuisines.update(broad_categories)

        # Check exact match first
        if normalized in known_cuisines:
            return True

        # Check for regional variants (e.g., "northern italian" contains "italian")
        for known in known_cuisines:
            if known in normalized:
                return True

        return False

    def _get_taxonomy_list(self) -> str:
        """
        Generate a formatted list of valid cuisines for LLM prompt.
        Organized by region for better comprehension.
        """
        formatted_lines = []
        for region, cuisines in taxonomy_groups().items():
            formatted_lines.append(f"{region}: {', '.join(cuisines)}")

        return "\n".join(formatted_lines)

    def _get_taxonomy_enum(self) -> List[str]:
        """Backward-compatible wrapper for the canonical cuisine list."""
        return canonical_cuisine_names()


    async def analyze_recipe(self, recipe_data: Dict[str, Any]) -> TaggingAnalysis:
        """
        Perform comprehensive analysis of a recipe for automatic tagging.

        Args:
            recipe_data: Full recipe data from Mealie API

        Returns:
            Complete tagging analysis with prep and cuisine classifications
        """
        logger.info(f"🔍 Analyzing recipe: {recipe_data.get('name', 'Unknown')}")

        # Analyze preparation requirements
        prep_analysis = await self._analyze_prep_requirements(recipe_data)

        # Analyze cuisine classification
        cuisine_analysis = await self._analyze_cuisine(recipe_data)

        # Handle cuisine analysis failure - validation errors should propagate
        if cuisine_analysis is None:
            # This should only happen for infrastructure failures (network/LLM unavailable)
            logger.warning(f"⚠️  Cuisine analysis failed for recipe: {recipe_data.get('name', 'Unknown')}")
            logger.warning("⚠️  Infrastructure issue detected - returning None for graceful degradation")
            return None

        # Generate recommended tags
        recommended_tags = self._generate_recommended_tags(prep_analysis, cuisine_analysis)

        # Debug: Log what we're generating
        logger.info(f"🔍 Recipe analysis complete: prep={prep_analysis.requires_overnight_prep}, cuisine='{cuisine_analysis.primary_cuisine}', tags={recommended_tags}")

        # Calculate overall confidence
        confidence_score = (prep_analysis.confidence + cuisine_analysis.confidence) / 2

        return TaggingAnalysis(
            prep_analysis=prep_analysis,
            cuisine_analysis=cuisine_analysis,
            recommended_tags=recommended_tags,
            confidence_score=confidence_score
        )

    async def _analyze_prep_requirements(self, recipe_data: Dict[str, Any]) -> PrepAnalysis:
        """
        Use LLM to analyze recipe instructions for preparation time requirements.
        Focuses on detecting overnight and extended prep needs.
        """
        instructions = self._extract_instructions_text(recipe_data)
        if not instructions:
            return PrepAnalysis(
                requires_overnight_prep=False,
                prep_duration_hours=None,
                prep_type=None,
                confidence=0.0,
                reasoning="No instructions found",
                detected_patterns=[]
            )

        # Use centralized prompt from prompts.py
        prompt = PREP_ANALYSIS_PROMPT.format(
            recipe_name=recipe_data.get('name', 'Unknown'),
            instructions=instructions
        )

        try:
            # Use structured output with JSON schema to prevent thinking text
            # schema = self._get_prep_analysis_schema()
            # response = await self._call_llm(prompt, response_format=schema)
            response = await self._call_llm(prompt, response_format=None)

            # Debug: Show what we got from LLM
            logger.debug(f"🔍 LLM Response: '{response[:200]}{'...' if len(response) > 200 else ''}'")

            # Extract JSON from response (handles markdown code fences, thinking text, etc.)
            result = self._extract_json_from_response(response)

            return PrepAnalysis(
                requires_overnight_prep=result.get('requires_overnight_prep', False),
                prep_duration_hours=result.get('prep_duration_hours'),
                prep_type=result.get('prep_type'),
                confidence=result.get('confidence', 0.0),
                reasoning=result.get('reasoning', ''),
                detected_patterns=result.get('detected_patterns', [])
            )

        except (json.JSONDecodeError, Exception) as e:
            # CRITICAL FAILURE: Structured output should prevent this, but keep fallback
            logger.critical(f"❌ CRITICAL: Structured prep analysis failed: {e}")
            logger.critical(f"❌ Response preview: '{response[:200] if 'response' in locals() else 'No response'}...'")
            logger.critical("❌ Structured outputs should prevent JSON parsing errors - check LM Studio configuration")
            raise RuntimeError(f"Prep analysis failed: {e}. Structured outputs should prevent this error.") from e

    async def _analyze_cuisine(self, recipe_data: Dict[str, Any]) -> CuisineClassification:
        """
        Multi-layer cuisine analysis using LLM with comprehensive taxonomy knowledge.
        """
        # Extract relevant text for analysis
        name = recipe_data.get('name', '')
        description = recipe_data.get('description', '')
        ingredients = self._extract_ingredients_text(recipe_data)
        instructions = self._extract_instructions_text(recipe_data)

        analysis_text = f"""
Recipe Name: {name}
Description: {description}
Ingredients: {ingredients[:1000]}...  # Truncate for LLM context
Instructions: {instructions[:200]}...  # Truncate for LLM context
"""

        # Get comprehensive taxonomy list for the prompt
        taxonomy_list = self._get_taxonomy_list()

        # Use centralized prompt from prompts.py
        prompt = CUISINE_ANALYSIS_PROMPT.format(
            analysis_text=analysis_text,
            taxonomy_list=taxonomy_list
        )

        # CRITICAL: LLM analysis is mandatory - no fallback allowed
        try:
            # Use LM Studio's official structured output support (verified working in Context7 docs)
            allowed = self._get_taxonomy_enum()
            cuisine_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "cuisine_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "primary_cuisine": {
                                "type": "string",
                                "enum": allowed,
                                "description": "Primary cuisine type (must be exactly one of the allowed taxonomy values)"
                            },
                            "secondary_cuisines": {
                                "type": "array",
                                "items": {"type": "string", "enum": allowed},
                                "description": "Secondary cuisine influences from the taxonomy list"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence score from 0.0 to 1.0"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of the analysis"
                            },
                            "detection_sources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "What parts of the recipe influenced this classification"
                            }
                        },
                        "required": ["primary_cuisine", "secondary_cuisines", "confidence", "reasoning", "detection_sources"],
                        "additionalProperties": False
                    }
                }
            }

            # Use structured output - disable reasoning for strict JSON compliance
            response = await self._call_llm(prompt, response_format=cuisine_schema, reasoning={"effort": "none"})

            # With structured output, this should be pure JSON, but keep extraction as a defensive guard.
            result = self._extract_json_from_response(response)

            # QUALITY FIRST: Validate cuisine names - reject empty, malformed, and unknown cuisines
            primary_cuisine = result.get('primary_cuisine')
            if not primary_cuisine or not TagFormatter.validate_cuisine_name(primary_cuisine):
                raise ValueError(f"Structured output returned invalid primary_cuisine: '{primary_cuisine}'. "
                               "Cuisine names must be valid strings without colons, null, or empty values. "
                               "Expected: 'Italian', 'Sichuan', 'Northern Italian'. "
                               f"Rejected: '{primary_cuisine}'")

            # STRICT VALIDATION: Reject unknown cuisines that aren't in our taxonomy
            if not self._is_known_cuisine(primary_cuisine):
                raise ValueError(f"LLM returned unknown cuisine: '{primary_cuisine}'. "
                               "Cuisine must be a recognized type from our taxonomy. "
                               f"Rejected unknown cuisine: '{primary_cuisine}'")

            # Filter and validate secondary cuisines - don't fail on invalid ones, just skip them
            secondary_cuisines_raw = result.get('secondary_cuisines', [])
            secondary_cuisines = []
            
            # Dietary restrictions that LLM sometimes incorrectly returns as cuisines
            dietary_restrictions = {'vegan', 'vegetarian', 'gluten-free', 'dairy-free', 'low-carb', 'keto', 'paleo'}
            
            for cuisine in secondary_cuisines_raw:
                # Skip empty or invalid format
                if not cuisine or not TagFormatter.validate_cuisine_name(cuisine):
                    logger.debug(f"⏭️  Skipping malformed secondary cuisine: '{cuisine}'")
                    continue
                    
                # Skip dietary restrictions incorrectly returned as cuisines
                if cuisine.lower().strip() in dietary_restrictions:
                    logger.debug(f"⏭️  Skipping dietary restriction returned as cuisine: '{cuisine}'")
                    continue
                    
                # Skip unknown cuisines with warning
                if not self._is_known_cuisine(cuisine):
                    logger.warning(f"⚠️  Skipping unknown secondary cuisine: '{cuisine}'")
                    continue
                    
                secondary_cuisines.append(cuisine)

            return CuisineClassification(
                primary_cuisine=primary_cuisine,
                secondary_cuisines=secondary_cuisines,
                region_specific=None,
                confidence=result.get('confidence', 0.0),
                reasoning=result.get('reasoning', ''),
                detection_sources=result.get('detection_sources', [])
            )

        except ValueError as e:
            # Validation errors (empty/malformed cuisines) - fail fast
            logger.error(f"❌ VALIDATION FAILURE: Cuisine analysis validation failed: {e}")
            logger.error("❌ FAIL FAST: Invalid cuisine data - analysis cannot continue")
            raise RuntimeError(f"Cuisine validation failed: {e}") from e
        except Exception as e:
            # Infrastructure errors (network, LLM unavailable) - graceful degradation
            logger.critical(f"❌ CRITICAL: LLM cuisine analysis infrastructure failed: {e}")
            logger.warning("⚠️  GRACEFUL DEGRADATION: Returning None for cuisine analysis - recipe will be skipped")
            return None

    async def _call_llm(self, prompt: str, response_format: Optional[dict] = None, reasoning: Optional[dict] = None) -> str:
        """
        Call local LLM for analysis with caching and structured outputs.
        Uses official LM Studio SDK if available, falls back to HTTP.
        """
        cache = await get_llm_cache()
        return await cache.call_llm(
            prompt=prompt,
            model=None,  # Uses default from SDK or config
            temperature=0.0,  # Zero temperature for maximum determinism
            max_tokens=1000,  # Enough for JSON, less chance of cutoff
            response_format=response_format,  # Enable structured outputs
            reasoning=reasoning  # Pass through reasoning config
        )



    def _extract_instructions_text(self, recipe_data: Dict[str, Any]) -> str:
        """Extract recipe instructions as plain text."""
        instructions = recipe_data.get('recipeInstructions', [])
        if isinstance(instructions, list):
            return ' '.join(str(step) for step in instructions)
        return str(instructions)

    def _extract_ingredients_text(self, recipe_data: Dict[str, Any]) -> str:
        """Extract ingredients as clean, human-readable text, filtering out malformed entries."""
        ingredients = recipe_data.get('recipeIngredient', [])
        if isinstance(ingredients, list):
            # Use the 'display' field for clean ingredient text instead of raw JSON
            ingredient_texts = []
            for ing in ingredients:
                if isinstance(ing, dict):
                    display = ing.get('display', '').strip()

                    # Skip malformed ingredients that are just preparation instructions
                    if not display or display in ['for dusting', 'for sprinkling', 'to taste']:
                        continue

                    # Skip ingredients with no food item (just quantities/units)
                    food = ing.get('food', {})
                    if isinstance(food, dict) and not food.get('name'):
                        # If display looks like just a quantity/unit without food, skip
                        if any(display.startswith(prefix) for prefix in [
                            'for ', 'to ', 'roughly ', 'finely ', 'thinly ', 'thickly '
                        ]) or len(display.split()) <= 2:
                            continue

                    ingredient_texts.append(display)
                else:
                    # Fallback for non-dict ingredients
                    ing_str = str(ing).strip()
                    if ing_str and ing_str not in ['for dusting', 'for sprinkling', 'to taste']:
                        ingredient_texts.append(ing_str)

            # If we filtered out too many ingredients, fall back to basic extraction
            if len(ingredient_texts) < len(ingredients) * 0.5 and ingredient_texts:
                # We filtered out more than half - use original approach but clean it
                basic_texts = []
                for ing in ingredients:
                    if isinstance(ing, dict) and 'display' in ing:
                        display = ing['display'].strip()
                        if display and display not in ['for dusting', 'for sprinkling', 'to taste']:
                            basic_texts.append(display)
                    elif isinstance(ing, str) and ing.strip():
                        basic_texts.append(ing.strip())
                return ', '.join(basic_texts)

            return ', '.join(ingredient_texts)
        return str(ingredients)

    def _extract_json_from_response(self, raw_content: str) -> dict:
        """
        Extract JSON from LLM response (handle thinking/reasoning text).
        More aggressive extraction - finds JSON anywhere in response.
        """
        content = raw_content.strip()

        # First try: Look for complete JSON object anywhere in the text
        import re

        # Find all JSON-like objects (balanced braces)
        json_pattern = r'\{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*\}'
        json_matches = re.findall(json_pattern, content, re.DOTALL)

        if json_matches:
            # Return the first complete JSON object found
            return json.loads(json_matches[0])

        # Fallback: Look for content starting with {
        json_start = content.find('{')
        if json_start >= 0:
            # Extract from { to end and try to balance braces
            potential_json = content[json_start:]
            brace_count = 0
            end_pos = 0

            for i, char in enumerate(potential_json):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break

            if end_pos > 0:
                return json.loads(potential_json[:end_pos])

        # Last resort: If no JSON found, try to parse the whole thing
        # This will cause JSON parsing to fail with a more informative error
        return json.loads(content)

    def _generate_recommended_tags(self, prep_analysis: PrepAnalysis,
                                 cuisine_analysis: CuisineClassification) -> List[str]:
        """Generate recommended Mealie tags based on analysis, using only existing tags."""
        tags = []

        # Get available tags from Mealie (cached for performance)
        available_tags = self._get_available_tags()
        available_lower = {t.lower(): t for t in available_tags}

        # Prep tags
        if prep_analysis.requires_overnight_prep:
            prep_tag = TagFormatter.create_prep_tag()
            if prep_tag in available_tags:
                tags.append(prep_tag)

        # Cuisine tags - only use tags that exist in Mealie
        if cuisine_analysis.primary_cuisine:
            # Skip overly generic cuisines that Mealie rejects
            generic_cuisines_to_skip = {"International", "Generic", "World", "Global"}
            if cuisine_analysis.primary_cuisine.lower() not in [g.lower() for g in generic_cuisines_to_skip]:
                cuisine_tag = TagFormatter.create_cuisine_tag(cuisine_analysis.primary_cuisine)
                # Case-insensitive existence match to avoid casing drift
                if cuisine_tag.lower() in available_lower:
                    tags.append(available_lower[cuisine_tag.lower()])

        for secondary in cuisine_analysis.secondary_cuisines:
            secondary_tag = self._get_specific_cuisine_tag(secondary)
            if secondary_tag and secondary_tag.lower() in available_lower:
                tags.append(available_lower[secondary_tag.lower()])

        return list(set(tags))  # Remove duplicates

    def _get_available_tags(self) -> Set[str]:
        """Get set of available tag names from Mealie (cached, thread-safe)."""
        with self._tag_cache_lock:
            if not hasattr(self, '_available_tags_cache') or self._available_tags_cache is None:
                try:
                    existing_tags = self.client.get_all_tags()
                    self._available_tags_cache = {tag.get('name') for tag in existing_tags if tag.get('name')}
                    logger.debug(f"📋 Loaded {len(self._available_tags_cache)} available tags from Mealie")
                except MealieClientError as e:
                    logger.warning(f"⚠️  Failed to get available tags from Mealie: {e}")
                    self._available_tags_cache = set()
            return self._available_tags_cache

    def _get_available_categories(self) -> Dict[str, Dict]:
        """Get dict of available categories from Mealie (cached, thread-safe).
        
        Returns dict mapping lowercase name -> category object with id/name/slug.
        Only 1 API call per process lifetime (categories rarely change).
        """
        with self._category_cache_lock:
            if not hasattr(self, '_available_categories_cache') or self._available_categories_cache is None:
                try:
                    items = self.client.get_all_categories()
                    
                    # Build lookup dict: lowercase name -> full category object
                    self._available_categories_cache = {}
                    for cat in items:
                        name = cat.get('name', '')
                        if name:
                            self._available_categories_cache[name.lower()] = {
                                "id": cat["id"],
                                "name": cat["name"],
                                "slug": cat.get("slug")
                            }
                    
                    logger.info(f"📋 Loaded {len(self._available_categories_cache)} categories from Mealie")
                except MealieClientError as e:
                    logger.warning(f"⚠️  Failed to get categories from Mealie: {e}")
                    self._available_categories_cache = {}
            return self._available_categories_cache

    def _get_specific_cuisine_tag(self, cuisine_name: str) -> str:
        """
        Convert generic cuisine names to specific variants that work with Mealie API.
        NOTE: We no longer force fake regional specificity (e.g., Italian -> Northern),
        because it creates incorrect tags and cultural assumptions.
        """
        # QUALITY FIRST: Validate cuisine_name format - reject malformed LLM output
        if not TagFormatter.validate_cuisine_name(cuisine_name):
            raise ValueError(f"LLM returned malformed cuisine name: '{cuisine_name}'. "
                           "Expected clean format like 'Italian' or 'Northern Italian', "
                           f"never with colons like 'Italian: Northern'")

        # Skip overly generic cuisines that cause 500 errors in Mealie
        generic_cuisines_to_skip = {"International", "Generic", "World", "Global"}
        if cuisine_name.lower() in [g.lower() for g in generic_cuisines_to_skip]:
            return None  # Skip this cuisine tag

        return TagFormatter.create_cuisine_tag(cuisine_name)

    def required_cuisine_tags_to_precreate(self) -> List[str]:
        """
        Return the canonical tag names that should be pre-created in Mealie.
        This project intentionally does NOT auto-create tags to avoid drift.
        """
        return [TagFormatter.create_cuisine_tag(c) for c in self._get_taxonomy_enum()]

    # Mapping from specific cuisines to broad category
    CUISINE_TO_CATEGORY = {
        # Asian
        "cantonese": "Asian", "sichuan": "Asian", "hunan": "Asian", "shanghai": "Asian",
        "beijing": "Asian", "dongbei": "Asian", "fujian": "Asian", "hakka": "Asian",
        "hong kong": "Asian", "taiwanese": "Asian", "chinese": "Asian",
        "american chinese": "Asian", "british chinese": "Asian", "indo-chinese": "Fusion",
        "japanese": "Asian", "korean": "Asian",
        "thai": "Asian", "vietnamese": "Asian", "malaysian": "Asian", 
        "indonesian": "Asian", "singaporean": "Asian", "filipino": "Asian",
        "indian": "Asian", "punjabi": "Asian",
        "peranakan": "Fusion", "macanese": "Fusion",
        
        # European
        "italian": "European", "northern italian": "European", "southern italian": "European",
        "french": "European", "provençal": "European", "provencal": "European",
        "spanish": "European", "greek": "European", "british": "European",
        "german": "European", "bavarian": "European", "austrian": "European", "swiss": "European",
        "polish": "European", "hungarian": "European", "russian": "European",
        "scandinavian": "European", "portuguese": "European", "irish": "European",
        "ukrainian": "European", "eastern european": "European",
        
        # Generic/International (no specific category - skip)
        "generic": None, "international": None, "western": None, "westernized chinese": "Asian",
        
        # American
        "american": "American", "cajun": "American", "southern us": "American",
        "tex-mex": "American", "canadian": "American",
        
        # Caribbean
        "jamaican": "American", "caribbean": "American",
        
        # Oceania (map to Fusion as we don't have Oceania category)
        "australian": "Fusion", "new zealand": "Fusion",
        
        # Middle Eastern
        "lebanese": "Middle Eastern", "turkish": "Middle Eastern", 
        "persian": "Middle Eastern", "moroccan": "Middle Eastern",
        "middle eastern": "Middle Eastern",
        
        # Latin American
        "mexican": "Latin American", "brazilian": "Latin American", "peruvian": "Latin American",
        
        # Fusion
        "fusion": "Fusion", "pan-asian": "Fusion",
    }

    def _get_category_for_cuisine(self, cuisine_name: str) -> Optional[str]:
        """Map a cuisine to its broad category."""
        if not cuisine_name:
            return None
        return self.CUISINE_TO_CATEGORY.get(cuisine_name.lower())

    def _ensure_category_exists(self, category_name: str) -> Optional[Dict]:
        """Ensure a category exists in Mealie, return the category object.
        
        Uses cached category lookup to avoid redundant API calls.
        Only 1 API call per process lifetime instead of 1 per recipe.
        """
        if not category_name:
            return None
        
        # Use cached categories (loaded once per process)
        categories = self._get_available_categories()
        
        # Case-insensitive lookup
        category_obj = categories.get(category_name.lower())
        
        if category_obj:
            return category_obj
        
        # Category doesn't exist - this shouldn't happen if we seeded properly
        logger.warning(f"Category '{category_name}' not found in Mealie cache")
        return None

    def apply_tags_to_mealie(self, recipe_id: str, analysis: Optional[TaggingAnalysis],
                           dry_run: bool = False, recipe_data: dict = None) -> Dict[str, Any]:
        """
        Apply recommended tags to a recipe in Mealie.
        Respects manual tags and only adds automatic ones.

        Args:
            recipe_id: Mealie recipe ID
            analysis: Complete tagging analysis
            dry_run: If True, return what would be done without making changes
            recipe_data: Optional pre-fetched recipe data (avoids redundant API call,
                        critical for freshly imported recipes not yet in DB cache)

        Returns:
            Dict with operation results
        """
        result = {
            "recipe_id": recipe_id,
            "tags_added": [],
            "tags_skipped": [],
            "categories_added": [],
            "errors": []
        }

        # Handle failed analysis
        if analysis is None:
            result["errors"].append("Recipe analysis failed - cannot apply tags")
            # CRITICAL: Do not create any tags if analysis fails - maintain data quality
            raise RuntimeError(f"Recipe analysis failed for {recipe_id} - refusing to create invalid tags")

        if dry_run:
            result["would_add"] = analysis.recommended_tags
            return result

        try:
            # Get current recipe data (need slug for PATCH API)
            # Use provided recipe_data if available (critical for freshly imported recipes
            # that may not yet be in the local DB cache)
            if recipe_data is not None:
                current_recipe = recipe_data
            else:
                current_recipe = self.client.get_recipe_by_id(recipe_id)

            # CRITICAL: Get slug from local DB instead of Mealie's GET-by-ID response
            # Mealie has a bug where the slug field in the recipe record can become
            # inconsistent with the slug routing table (especially after renames).
            # Our local DB has the correct slug (synced from list API).
            recipe_slug = self._get_correct_slug_from_local_db(recipe_id)
            if not recipe_slug:
                # Fallback to Mealie's response if not in local DB
                recipe_slug = current_recipe.get('slug')
            if not recipe_slug:
                result["errors"].append("Could not get recipe slug")
                return result

            # Get existing tags
            existing_tags = []
            if 'tags' in current_recipe:
                for tag in current_recipe['tags']:
                    if isinstance(tag, dict):
                        existing_tags.append(tag.get('name', ''))
                    else:
                        existing_tags.append(str(tag))

            # Check for manual "Overnight Preparation" tag or new format
            has_manual_overnight = ("Overnight Preparation" in existing_tags or
                                  any("Prep" in tag for tag in existing_tags))

            # Filter tags to add (respect manual tags)
            tags_to_add = []
            for tag in analysis.recommended_tags:
                if tag not in existing_tags:
                    # Don't add automatic prep tags if manual prep tags exist
                    if "Prep" in tag and has_manual_overnight:
                        result["tags_skipped"].append(f"{tag} (manual prep tag exists)")
                        continue
                    tags_to_add.append(tag)

            logger.info(f"🏷️  Recipe {recipe_id}: existing_tags={existing_tags}, recommended={analysis.recommended_tags}, adding={tags_to_add}")

            # CRITICAL: Validate and create tags that don't exist - system fails if any tag creation fails
            tag_objects = []
            valid_tags_to_add = []

            # VALIDATE ALL TAGS BEFORE ATTEMPTING CREATION
            for tag_name in tags_to_add:
                if TagFormatter.validate_tag_format(tag_name):
                    valid_tags_to_add.append(tag_name)
                else:
                    logger.warning(f"❌ BLOCKED: Skipping invalid tag format: '{tag_name}'")
                    result["tags_skipped"].append(f"{tag_name} (invalid format)")
                    continue

            # ONLY CREATE VALID TAGS
            for tag_name in valid_tags_to_add:
                tag_obj = self._ensure_tag_exists_full(tag_name)
                # _ensure_tag_exists_full now returns None on failure (graceful degradation)
                if tag_obj is not None:
                    tag_objects.append(tag_obj)
                    result["tags_added"].append(tag_name)
                else:
                    # Tag creation failed - skip this tag but continue with others
                    result["tags_skipped"].append(f"{tag_name} (creation failed)")
                    logger.warning(f"⚠️  WARNING: Tag creation failed for: {tag_name} - continuing with other tags")

            # Update recipe with new tags using PATCH /api/recipes/{slug}
            if tag_objects:
                # Combine existing tags with new ones
                existing_tag_objects = current_recipe.get('tags', [])
                all_tag_objects = existing_tag_objects + tag_objects

                update_data = {"tags": all_tag_objects}
                try:
                    self.client.update_recipe(recipe_slug, update_data)
                    logger.info(f"✅ Successfully updated recipe {recipe_slug} with {len(tag_objects)} new tags")
                    # Update local database cache with cuisine information
                    self._update_local_database_cuisine(recipe_id, result["tags_added"])
                except MealieClientError as e:
                    result["errors"].append(f"Recipe update failed: {e}")

            # CATEGORY ASSIGNMENT: Map cuisine to broad category
            if analysis.cuisine_analysis and analysis.cuisine_analysis.primary_cuisine:
                category_name = self._get_category_for_cuisine(analysis.cuisine_analysis.primary_cuisine)
                if category_name:
                    # Check if recipe already has this category
                    existing_categories = current_recipe.get('recipeCategory', [])
                    existing_cat_names = [c.get('name', '') if isinstance(c, dict) else str(c) for c in existing_categories]
                    
                    if category_name not in existing_cat_names:
                        category_obj = self._ensure_category_exists(category_name)
                        if category_obj:
                            # Add category to recipe
                            all_categories = existing_categories + [category_obj]
                            cat_update_data = {"recipeCategory": all_categories}
                            try:
                                self.client.update_recipe(recipe_slug, cat_update_data)
                                result["categories_added"].append(category_name)
                                logger.info(f"✅ Added category '{category_name}' to recipe {recipe_slug}")
                            except MealieClientError as e:
                                logger.warning(f"⚠️  Failed to add category: {e}")

        except Exception as e:
            logger.error(f"❌ CRITICAL FAILURE: Failed to apply tags: {str(e)}")
            logger.warning(f"⚠️  CONTINUING: Partial tag application allowed - recipe may have some tags applied")
            result["errors"].append(f"Failed to apply tags: {str(e)}")
            # Don't raise exception - allow partial success
            return result

        return result

    def _get_correct_slug_from_local_db(self, recipe_id: str) -> Optional[str]:
        """
        Get the correct slug from our local DB.
        
        Mealie has a bug where GET /api/recipes/{id} can return a slug that
        doesn't match the slug routing table (especially after renames).
        Our local DB is synced from the list API which has correct slugs.
        
        Args:
            recipe_id: Mealie recipe ID (UUID)
            
        Returns:
            Correct slug from local DB, or None if not found
        """
        import sqlite3
        try:
            db_path = str(DATA_DIR / "recipe_index.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT slug FROM recipes WHERE id = ?", (recipe_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    return row[0]
        except Exception as e:
            logger.warning(f"Could not get slug from local DB for {recipe_id}: {e}")
        return None

    def _update_local_database_cuisine(self, recipe_id: str, tags_added: List[str]) -> None:
        """
        Update the local database cache with cuisine information after successful tagging.

        Args:
            recipe_id: Mealie recipe ID
            tags_added: List of tags that were successfully added
        """
        try:
            logger.info(f"🔄 Updating local database for recipe {recipe_id} with tags: {tags_added}")

            # Extract cuisine from the tags (look for "X Cuisine" pattern)
            cuisine_primary = None
            for tag in tags_added:
                if tag.endswith(" Cuisine"):
                    # Remove " Cuisine" suffix to get the cuisine name
                    cuisine_primary = tag[:-8]  # Remove " Cuisine"
                    break

            if cuisine_primary:
                # Update the local database (must match RecipeRAG db_path)
                conn = sqlite3.connect(str(DATA_DIR / "recipe_index.db"))
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE recipes
                    SET cuisine_primary = ?
                    WHERE id = ?
                """, (cuisine_primary, recipe_id))

                conn.commit()
                conn.close()

                logger.info(f"✅ Updated local database: recipe {recipe_id} cuisine set to '{cuisine_primary}'")
            else:
                logger.info(f"ℹ️  No cuisine tag found in added tags for recipe {recipe_id}: {tags_added}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to update local database for recipe {recipe_id}: {e}")
            # Don't raise exception - local database sync is not critical

    def _sanitize_tag_name(self, tag_name: str) -> str:
        """
        Sanitize tag names to prevent Mealie API rejections.
        Removes or replaces characters that might cause issues.
        """
        # Remove any leading/trailing whitespace
        sanitized = tag_name.strip()

        # Replace problematic characters that might cause API issues
        # Based on Mealie API documentation, tag names should be simple strings
        replacements = {
            '"': '',  # Remove quotes
            "'": '',  # Remove apostrophes
            '&': 'and',  # Replace ampersand
            '/': '-',  # Replace slashes with dashes
            '\\': '-',  # Replace backslashes with dashes
        }

        for old_char, new_char in replacements.items():
            sanitized = sanitized.replace(old_char, new_char)

        # Ensure the name is not empty after sanitization
        if not sanitized:
            sanitized = "Unnamed"

        # Limit length to prevent issues (Mealie might have limits)
        if len(sanitized) > 50:
            sanitized = sanitized[:47] + "..."

        return sanitized

    def _ensure_tag_exists_full(self, tag_name: str) -> Optional[Dict[str, Any]]:
        """
        Ensure a tag exists in Mealie. Only returns existing tags - does not create new ones.
        If tag creation API is broken, we gracefully skip creating new tags.

        Returns:
            Full tag object if it exists, None if tag doesn't exist or API fails
        """
        # Sanitize tag name to prevent Mealie API rejections
        sanitized_name = self._sanitize_tag_name(tag_name)

        try:
            existing_tags = self.client.get_all_tags()
            for tag in existing_tags:
                if tag.get('name') == sanitized_name:
                    return tag

            # Tag doesn't exist - log and skip (don't try to create due to API issues)
            logger.info(f"ℹ️  Tag '{sanitized_name}' doesn't exist in Mealie - skipping (tag creation API has issues)")
            return None

        except MealieClientError as e:
            logger.warning(f"⚠️  Failed to check for existing tag '{sanitized_name}': {e}")
            return None

    def _get_tag_color(self, tag_name: str) -> str:
        """Get appropriate color for tag type."""
        tag_type = TagFormatter.get_tag_type(tag_name)
        if tag_type == "prep":
            return "#FF6B6B"  # Red for prep
        elif tag_type == "cuisine":
            return "#4ECDC4"  # Teal for cuisine
        elif tag_type == "region":
            return "#45B7D1"  # Blue for region
        else:
            return "#96CEB4"  # Green default

    async def batch_analyze_recipes(self, recipe_ids: List[str], batch_size: int = None) -> Dict[str, TaggingAnalysis]:
        """
        Analyze multiple recipes in parallel batches optimized for LM Studio.
        Uses concurrent processing to achieve significant speedup.

        Args:
            recipe_ids: List of Mealie recipe IDs
            batch_size: Number of recipes to process concurrently (None = use config default)

        Returns:
            Dict mapping recipe_id to analysis results
        """
        # Use centralized configuration if no explicit batch_size provided
        if batch_size is None:
            config = get_bulk_operation_config_safe('tag', fallback_batch_size=3, fallback_concurrent=2)
            batch_size = config['default_batch_size']
        results = {}

        # Process recipes in concurrent batches
        for i in range(0, len(recipe_ids), batch_size):
            batch = recipe_ids[i:i + batch_size]
            logger.info(f"🚀 Processing batch {i//batch_size + 1}: recipes {i+1}-{min(i+batch_size, len(recipe_ids))} (concurrent)")

            # Create concurrent tasks for this batch
            async def analyze_single_recipe(recipe_id: str):
                try:
                    # Fetch recipe data
                    recipe_data = self.client.get_recipe_by_id(recipe_id)

                    # Analyze recipe
                    analysis = await self.analyze_recipe(recipe_data)
                    return recipe_id, analysis

                except MealieClientError as e:
                    logger.error(f"❌ Failed to analyze recipe {recipe_id}: {e}")
                    return recipe_id, None
                except Exception as e:
                    logger.error(f"❌ Failed to analyze recipe {recipe_id}: {e}")
                    return recipe_id, None

            # Process batch concurrently
            tasks = [analyze_single_recipe(recipe_id) for recipe_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"❌ Batch processing error: {result}")
                    continue
                recipe_id, analysis = result
                if analysis:
                    results[recipe_id] = analysis

        return results


# Utility functions for external use
async def analyze_single_recipe(recipe_id: str, apply_tags: bool = False, client: Optional[MealieClient] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze a single recipe.

    Args:
        recipe_id: Mealie recipe ID
        apply_tags: Whether to apply the tags to Mealie
        client: Optional MealieClient instance. If None, creates a new one.

    Returns:
        Analysis results and tag application status
    """
    if client is None:
        client = MealieClient()
        client_owned = True
    else:
        client_owned = False
    
    try:
        tagger = AutomaticTagger(client=client)
        
        # Fetch recipe
        recipe_data = client.get_recipe_by_id(recipe_id)

        # Analyze
        analysis = await tagger.analyze_recipe(recipe_data)

        result = {
            "recipe_id": recipe_id,
            "recipe_name": recipe_data.get('name', 'Unknown'),
            "analysis": {
                "prep_required": analysis.prep_analysis.requires_overnight_prep,
                "prep_type": analysis.prep_analysis.prep_type,
                "primary_cuisine": analysis.cuisine_analysis.primary_cuisine,
                "confidence": analysis.confidence_score,
                "recommended_tags": analysis.recommended_tags
            }
        }

        # Apply tags if requested
        if apply_tags:
            tag_result = tagger.apply_tags_to_mealie(recipe_id, analysis)
            result["tag_application"] = tag_result

        return result

    except MealieClientError as e:
        return {
            "recipe_id": recipe_id,
            "error": str(e),
            "analysis": None
        }
    except Exception as e:
        return {
            "recipe_id": recipe_id,
            "error": str(e),
            "analysis": None
        }
    finally:
        if client_owned:
            client.close()


def sync_local_database_from_mealie(client: Optional[MealieClient] = None):
    """
    Sync the local database cuisine information from Mealie tags.
    This fixes the issue where bulk tagging updated Mealie but not the local cache.
    
    Args:
        client: Optional MealieClient instance. If None, creates a new one.
    """
    import sqlite3
    import re

    if client is None:
        client = MealieClient()
        client_owned = True
    else:
        client_owned = False

    try:
        print("🔄 Syncing local database cuisine information from Mealie...")

        # Get all recipes from Mealie
        recipes = client.get_all_recipes()
        mealie_recipes = {}

        for recipe in recipes:
            recipe_id = recipe['id']
            tags = recipe.get('tags', [])
            cuisine_tags = [tag for tag in tags if isinstance(tag, dict) and tag.get('name', '').endswith(' Cuisine')]
            if cuisine_tags:
                # Extract cuisine name from first cuisine tag (remove " Cuisine" suffix)
                cuisine_name = cuisine_tags[0]['name'][:-8]  # Remove " Cuisine"
                mealie_recipes[recipe_id] = cuisine_name

        print(f"📊 Found {len(mealie_recipes)} recipes with cuisine tags in Mealie")

        # Update local database
        conn = sqlite3.connect(str(DATA_DIR / "recipe_index.db"))
        cursor = conn.cursor()

        updated_count = 0
        for recipe_id, cuisine in mealie_recipes.items():
            cursor.execute("""
                UPDATE recipes
                SET cuisine_primary = ?
                WHERE id = ?
            """, (cuisine, recipe_id))
            if cursor.rowcount > 0:
                updated_count += 1

        conn.commit()
        conn.close()

        print(f"✅ Updated {updated_count} recipes in local database")
        print("🔄 Local database sync complete!")
    finally:
        if client_owned:
            client.close()


def diagnose_pipeline(recipe_slug: str, client: Optional[MealieClient] = None):
    """
    Comprehensive pipeline diagnostic to identify all failure points.
    Tests each stage: data extraction, LLM analysis, tag generation, tag application.
    
    Args:
        recipe_slug: Recipe slug to diagnose
        client: Optional MealieClient instance. If None, creates a new one.
    """
    if client is None:
        client = MealieClient()
        client_owned = True
    else:
        client_owned = False

    try:
        print("=" * 80)
        print(f"🔬 PIPELINE DIAGNOSTIC: {recipe_slug}")
        print("=" * 80)

        # STAGE 1: Raw Data Extraction
        print("\n📥 STAGE 1: RAW DATA EXTRACTION")
        try:
            recipe_data = client.get_recipe(recipe_slug)
        except MealieClientError as e:
            print(f"❌ FAILED: Could not fetch recipe: {e}")
            return
        print(f"✅ Recipe fetched: {recipe_data.get('name', 'Unknown')}")

        # Extract components
        ingredients = recipe_data.get('recipeIngredient', [])
        instructions = recipe_data.get('recipeInstructions', [])
        name = recipe_data.get('name', '')
        description = recipe_data.get('description', '')

        print(f"📊 Raw data stats:")
        print(f"  - Ingredients: {len(ingredients)}")
        print(f"  - Instructions: {len(instructions)} steps")

        # STAGE 2: Ingredient Processing
        print("\n🧄 STAGE 2: INGREDIENT PROCESSING")
        tagger = AutomaticTagger()

        processed_ingredients = tagger._extract_ingredients_text(recipe_data)
        print(f"✅ Processed ingredients: {processed_ingredients}")

        # Analyze ingredient quality
        real_ingredients = []
        malformed_ingredients = []

        for ing in ingredients:
            if isinstance(ing, dict):
                food = ing.get('food')
                display = ing.get('display', '')
                if food and isinstance(food, dict) and food.get('name'):
                    real_ingredients.append(display)
                else:
                    malformed_ingredients.append(display)

        print("📈 Ingredient quality analysis:")
        print(f"  - Real ingredients: {len(real_ingredients)}")
        print(f"  - Malformed ingredients: {len(malformed_ingredients)}")
        print(f"  - Quality ratio: {len(real_ingredients)/len(ingredients)*100:.1f}%" if ingredients else "N/A")

        if malformed_ingredients:
            print("🚨 MALFORMED INGREDIENTS:")
            for ing in malformed_ingredients[:5]:  # Show first 5
                print(f"    - '{ing}'")
            if len(malformed_ingredients) > 5:
                print(f"    ... and {len(malformed_ingredients)-5} more")

        # STAGE 3: LLM Analysis Simulation
        print("\n🤖 STAGE 3: LLM ANALYSIS SIMULATION")

        # Mock cuisine analysis with the processed data
        analysis_text = f"""
Recipe Name: {name}
Description: {description}
Ingredients: {processed_ingredients[:1000]}...
Instructions: {' '.join(str(step) for step in instructions[:3])[:500]}...
"""

        print("📝 Data being sent to LLM:")
        print(f"  Length: {len(analysis_text)} characters")
        print(f"  Sample: {analysis_text[:300]}...")

        # Check if this would produce meaningful results
        if len(real_ingredients) < 2:
            print("🚨 CRITICAL: Insufficient real ingredients for reliable cuisine classification")
            print("   This will likely result in random/incorrect cuisine assignments")

        # STAGE 4: Tag Generation Analysis
        print("\n🏷️  STAGE 4: TAG GENERATION ANALYSIS")

        # Simulate what tags would be generated
        print("🎯 Expected tag patterns based on ingredient quality:")

        if len(real_ingredients) == 0:
            print("🚨 FATAL: No real ingredients = random tag generation")
            print("   LLM will hallucinate cuisines based on recipe name/description only")
        elif len(real_ingredients) < 3:
            print("⚠️  WARNING: Very few ingredients = unreliable classification")
            print("   Tags will be based on minimal data, likely incorrect")
        else:
            print("✅ Sufficient ingredients for reliable classification")

        # Check existing tags on recipe
        existing_tags = recipe_data.get('tags', [])
        existing_tag_names = []
        malformed_tags = []

        for tag in existing_tags:
            if isinstance(tag, dict):
                tag_name = tag.get('name', '')
                existing_tag_names.append(tag_name)

                # Check for malformed tags
                if 'null' in tag_name or ':' in tag_name or not tag_name.strip():
                    malformed_tags.append(tag_name)

        print("📋 Existing tags on recipe:")
        for tag in existing_tag_names[:10]:  # Show first 10
            print(f"  - '{tag}'")
        if len(existing_tag_names) > 10:
            print(f"  ... and {len(existing_tag_names)-10} more")

        if malformed_tags:
            print("🚨 MALFORMED EXISTING TAGS:")
            for tag in malformed_tags:
                print(f"  ❌ '{tag}'")

        # STAGE 5: Overall Assessment
        print("\n🎯 STAGE 5: OVERALL PIPELINE ASSESSMENT")

        issues = []

        if len(malformed_ingredients) > 0:
            issues.append(f"DATA QUALITY: {len(malformed_ingredients)}/{len(ingredients)} malformed ingredients")

        if len(real_ingredients) < 2:
            issues.append("CLASSIFICATION: Insufficient ingredients for reliable cuisine detection")

        if malformed_tags:
            issues.append(f"TAG QUALITY: {len(malformed_tags)} malformed existing tags")

        if len(existing_tag_names) > 10:
            issues.append(f"DUPLICATION: {len(existing_tag_names)} tags (excessive duplication)")

        if issues:
            print("🚨 CRITICAL ISSUES IDENTIFIED:")
            for issue in issues:
                print(f"  ❌ {issue}")
        else:
            print("✅ No major issues detected")

        print("\n" + "=" * 80)
        print("🔬 DIAGNOSTIC COMPLETE")
        print("=" * 80)
    finally:
        if client_owned:
            client.close()


async def main():
    """Main async execution function."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python automatic_tagger.py <recipe_id> [--apply] | --print-required-cuisine-tags")
        sys.exit(1)

    recipe_id = sys.argv[1]
    apply_tags = "--apply" in sys.argv

    # Create client once at start
    client = MealieClient()

    try:
        if recipe_id == "--print-required-cuisine-tags":
            tagger = AutomaticTagger(client=client)
            tags = tagger.required_cuisine_tags_to_precreate()
            print("\n".join(tags))
            return

        if recipe_id == "--sync-db":
            # Special command to sync local database from Mealie
            sync_local_database_from_mealie(client)
            return

        if recipe_id.startswith("--diagnose="):
            # Diagnostic mode
            recipe_slug = recipe_id.split("--diagnose=")[1]
            diagnose_pipeline(recipe_slug, client)
            return

        print(f"🔍 Analyzing recipe {recipe_id}...")
        result = await analyze_single_recipe(recipe_id, apply_tags, client)

        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            sys.exit(1)

        print("✅ Analysis complete!")
        print(f"Recipe: {result['recipe_name']}")
        print(f"Prep Required: {result['analysis']['prep_required']}")
        print(f"Prep Type: {result['analysis']['prep_type']}")
        print(f"Primary Cuisine: {result['analysis']['primary_cuisine']}")
        print(f"Confidence: {result['analysis']['confidence']:.2f}")
        print(f"Recommended Tags: {', '.join(result['analysis']['recommended_tags'])}")

        if apply_tags and "tag_application" in result:
            tag_result = result["tag_application"]
            if tag_result["tags_added"]:
                print(f"✅ Added tags: {', '.join(tag_result['tags_added'])}")
            if tag_result["tags_skipped"]:
                print(f"⏭️  Skipped tags: {', '.join(tag_result['tags_skipped'])}")
            if tag_result["errors"]:
                print(f"❌ Tag errors: {', '.join(tag_result['errors'])}")
    finally:
        client.close()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
