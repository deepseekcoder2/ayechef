#!/usr/bin/env python3
"""
Recipe RAG (Retrieval Augmented Generation) System
==================================================

Semantic recipe search using SQLite database + LLM embeddings for the menu-first
meal planning system. Replaces bulk recipe loading with intelligent semantic matching.

Features:
- SQLite database for scalable recipe storage (handles 10k+ recipes)
- LLM-generated embeddings for semantic similarity search
- Hybrid search combining vector similarity + keyword matching
- Efficient indexing and querying for menu concept matching

Usage:
    from recipe_rag import RecipeRAG

    rag = RecipeRAG()
    rag.index_all_recipes()  # One-time setup

    # Find recipes for menu concept
    matches = rag.find_recipes_for_concept("Cantonese beef stir-fry", top_k=10)
"""

import sqlite3
import numpy as np
import json
import os
import re
from typing import List, Dict, Tuple, Optional, Any
from config import get_bulk_operation_config_safe, USER_CONFIG, DATA_DIR
from batch_llm_processor import get_llm_cache
from tools.logging_utils import get_logger
from utils.url_utils import normalize_url

# Initialize logger for this module
logger = get_logger(__name__)


def _check_ingredients_parsed(recipe: dict) -> bool:
    """
    Check if a recipe has parsed ingredients.
    
    A recipe is considered "parsed" if any ingredient has a `food` object.
    This is the inverse of mealie_parse.is_recipe_actually_unparsed().
    
    NOTE: This function intentionally duplicates logic from mealie_parse.py
    to avoid circular imports. Both functions check for food object presence
    as the definitive indicator of parsing success. If modifying this logic,
    ensure mealie_parse.is_recipe_actually_unparsed() stays in sync.
    
    Args:
        recipe: Recipe data dict from Mealie API
        
    Returns:
        True if at least one ingredient has a food object, False otherwise
    """
    ingredients = recipe.get("recipeIngredient", [])
    for ing in ingredients:
        if isinstance(ing, dict) and ing.get("food") is not None:
            return True  # Has at least one parsed ingredient
    return False  # No parsed ingredients

class RecipeRAG:
    """
    Recipe Retrieval Augmented Generation system using SQLite + LLM embeddings.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(DATA_DIR / "recipe_index.db")
        """
        Initialize the Recipe RAG system.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Embedding model loaded from config.yaml
        self.embedding_model = USER_CONFIG["llm"]["embedding_model"]
        # IMPORTANT: embedding_dim is HARDCODED at 4096 - do NOT make configurable
        # Changing this after data exists corrupts the SQLite database and ANN index
        self.embedding_dim = 4096

        # Initialize database
        self._init_database()

        # Add ANN index initialization
        try:
            from recipe_ann_index import RecipeANNIndex
            self.ann_index = RecipeANNIndex(dimension=4096)

            # Try loading existing index, build if not found
            try:
                self.ann_index.load()
                logger.info(f"✅ ANN index loaded with {len(self.ann_index.recipe_id_map)} recipes")
            except FileNotFoundError:
                logger.warning("⚠️ No ANN index found. Run migration to build index.")
                # Keep the index for future use
        except ImportError as e:
            logger.warning(f"⚠️ ANN index not available: {e}")
            self.ann_index = None

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Main recipes table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recipes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    slug TEXT,
                    org_url TEXT,  -- Original source URL for duplicate checking
                    description TEXT,
                    searchable_text TEXT,
                    tags TEXT,  -- JSON array
                    categories TEXT,  -- JSON array
                    ingredients TEXT,  -- JSON array of ingredient summaries
                    cuisine_primary TEXT,  -- Primary cuisine tag
                    cuisine_secondary TEXT,  -- Secondary cuisine tags (JSON)
                    cuisine_region TEXT,  -- Regional specifier
                    cuisine_confidence REAL,  -- Cuisine classification confidence
                    embedding BLOB,  -- Numpy array stored as bytes
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add org_url column if it doesn't exist (migration for existing DBs)
            try:
                conn.execute('ALTER TABLE recipes ADD COLUMN org_url TEXT')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Add mealie_updated_at column for incremental sync (migration for existing DBs)
            # This stores Mealie's updatedAt timestamp so we can detect changes
            try:
                conn.execute('ALTER TABLE recipes ADD COLUMN mealie_updated_at TEXT')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Add ingredients_parsed column for tracking ingredient parsing status
            # Values: NULL = unknown, 0 = unparsed, 1 = parsed
            try:
                conn.execute('ALTER TABLE recipes ADD COLUMN ingredients_parsed INTEGER DEFAULT NULL')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Index for fast duplicate checking by URL
            conn.execute('CREATE INDEX IF NOT EXISTS idx_org_url ON recipes(org_url)')

            # Full-text search table for keyword fallback
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS recipes_fts
                USING fts5(id, name, description, searchable_text, tags, ingredients)
            ''')

            # Indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_name ON recipes(name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tags ON recipes(tags)')

    def _generate_embedding(self, text: str, is_query: bool = False) -> Optional[np.ndarray]:
        """
        Generate embedding vector for text using LLM through batch processor.

        This method uses the async batch_llm_processor infrastructure for proper
        caching, concurrency, and error handling as documented in PRD Section 3.1.2.

        Args:
            text: Text to embed
            is_query: If True, use query encoding (for search queries).
                      If False, use document encoding (for indexing recipes).
                      Nemotron model requires this distinction for optimal retrieval.

        Returns:
            Numpy array of embedding, or None if failed
        """
        try:
            import asyncio
            from batch_llm_processor import get_llm_cache

            async def generate_embedding_async():
                cache = await get_llm_cache()
                return await cache.call_embedding(text, model=self.embedding_model, is_query=is_query)

            # Run async function - handle existing event loop gracefully
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an async context, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, generate_embedding_async())
                    embedding_result = future.result(timeout=60)
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                embedding_result = asyncio.run(generate_embedding_async())

            if embedding_result:
                return np.array(embedding_result, dtype=np.float32)
            return None

        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            return None

    def _embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for storage."""
        return embedding.tobytes()

    def _bytes_to_embedding(self, data: bytes) -> np.ndarray:
        """Convert bytes back to numpy array."""
        return np.frombuffer(data, dtype=np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0.0

    def index_recipe(self, recipe: Dict[str, Any], force: bool = False, auto_save: bool = True, ingredients_parsed: Optional[bool] = None) -> bool:
        """
        Index a single recipe in the RAG system (generates embedding internally).

        Args:
            recipe: Full recipe data from Mealie API
            force: If False (default), skip recipes that are already indexed
            auto_save: If True, save index after adding (default for backward compatibility)
            ingredients_parsed: Whether recipe has parsed ingredients. If None, auto-computed.

        Returns:
            True if successfully indexed, False otherwise
        """
        try:
            # Skip if already indexed (unless force=True)
            if not force and self._is_recipe_indexed(recipe.get("id", "")):
                return False

            # Generate embedding
            searchable_text = self._create_searchable_text(recipe)
            embedding = self._generate_embedding(searchable_text)
            if embedding is None:
                logger.warning(f"⚠️  Skipping recipe {recipe.get('name', 'unknown')} - embedding generation failed")
                return False

            # Store with precomputed embedding
            return self.store_with_precomputed_embedding(recipe, embedding, force=force, auto_save=auto_save, ingredients_parsed=ingredients_parsed)

        except Exception as e:
            logger.error(f"❌ Failed to index recipe {recipe.get('name', 'unknown')}: {e}")
            return False

    def index_recipes_batch(self, recipes: List[dict], force: bool = False) -> int:
        """
        Index multiple recipes in one operation with batched embedding generation.
        
        This is significantly faster than calling index_recipe() in a loop because:
        - Generates all embeddings in a single GPU/MPS call
        - Updates SQLite in fewer transactions
        - Saves USearch index once at the end
        
        Args:
            recipes: List of full recipe data dicts from Mealie API
            force: If True, re-index even if already indexed
            
        Returns:
            Count of successfully indexed recipes
        """
        if not recipes:
            return 0
        
        # Filter to recipes that need indexing (unless force=True)
        if force:
            to_index = recipes
        else:
            to_index = [r for r in recipes if not self._is_recipe_indexed(r.get("id", ""))]
        
        if not to_index:
            logger.info("All recipes already indexed, skipping batch")
            return 0
        
        logger.info(f"📚 Batch indexing {len(to_index)} recipes...")
        
        # Generate searchable texts for all recipes
        texts = [self._create_searchable_text(r) for r in to_index]
        
        # Batch embedding generation (single GPU/MPS call)
        try:
            from batch_llm_processor import get_embed_model
            from config import get_embedding_batch_size
            import numpy as np
            
            model = get_embed_model()
            if model is None:
                logger.error("❌ Embedding model not available")
                return 0
            
            # Use encode_document for consistency with existing embeddings
            # This ensures search works correctly (documents and queries use same method)
            batch_size = get_embedding_batch_size()
            logger.info(f"   Generating {len(texts)} embeddings (batch_size={batch_size})...")
            embeddings = model.encode_document(texts, batch_size=batch_size)
            
            # Convert to numpy if needed
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype=np.float32)
            
            logger.info(f"   ✅ Embeddings generated")
            
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {e}")
            return 0
        
        # Store all recipes with precomputed embeddings
        success_count = 0
        for recipe, embedding in zip(to_index, embeddings):
            try:
                # Compute ingredients_parsed for this recipe
                ingredients_parsed = _check_ingredients_parsed(recipe)
                
                if self.store_with_precomputed_embedding(
                    recipe, 
                    embedding, 
                    force=True,  # Already filtered above
                    auto_save=False,  # Save once at end
                    ingredients_parsed=ingredients_parsed
                ):
                    success_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Failed to store recipe {recipe.get('name', 'unknown')}: {e}")
        
        # Save USearch index once at the end
        if self.ann_index is not None and success_count > 0:
            try:
                self.ann_index.save()
                logger.info(f"💾 Saved ANN index with {success_count} new recipes")
            except Exception as e:
                logger.error(f"❌ Failed to save ANN index: {e}")
        
        logger.info(f"✅ Batch indexed {success_count}/{len(to_index)} recipes")
        return success_count

    def store_with_precomputed_embedding(
        self,
        recipe: Dict[str, Any],
        embedding: np.ndarray,
        force: bool = False,
        auto_save: bool = True,
        mealie_updated_at: Optional[str] = None,
        ingredients_parsed: Optional[bool] = None
    ) -> bool:
        """
        Store recipe with pre-computed embedding.

        Use this when embedding generation is done separately (e.g., in parallel).
        Caller is responsible for thread-safety of ANN index operations.

        Args:
            recipe: Full recipe data dict from Mealie
            embedding: Pre-computed embedding vector (numpy array)
            force: If True, re-index even if already indexed
            auto_save: If True, save ANN index after adding
            mealie_updated_at: Mealie's updatedAt timestamp for incremental sync
            ingredients_parsed: Whether recipe has parsed ingredients. If None, auto-computed.

        Returns:
            True if successfully indexed, False otherwise
        """
        try:
            # Extract key information
            recipe_id = recipe.get("id")
            if not recipe_id:
                logger.warning("⚠️  Skipping recipe - missing 'id' field")
                return False

            name = recipe.get("name", "")
            slug = recipe.get("slug", "")
            description = recipe.get("description", "")

            # Skip if already indexed (unless force=True)
            if not force and self._is_recipe_indexed(recipe_id):
                return False

            # Validate embedding
            if embedding is None or not isinstance(embedding, np.ndarray):
                logger.warning(f"⚠️  Skipping recipe {name} - invalid embedding")
                return False

            # Validate embedding dimension BEFORE attempting to store
            if embedding.shape[0] != self.embedding_dim:
                logger.error(
                    f"❌ Embedding dimension mismatch for recipe {name}: "
                    f"got {embedding.shape[0]}, expected {self.embedding_dim}"
                )
                return False

            # Create searchable text summary
            searchable_text = self._create_searchable_text(recipe)

            # Extract tags and categories
            tags = json.dumps(recipe.get("tags", []))
            categories = json.dumps(recipe.get("recipeCategory", []))

            # Extract cuisine information from tags for analysis
            cuisine_primary, cuisine_secondary, cuisine_region, cuisine_confidence = self._extract_cuisine_from_tags(recipe.get("tags", []))

            # Summarize ingredients
            ingredients = self._summarize_ingredients(recipe.get("recipeIngredient", []))

            # Ensure ingredients is a list of strings (defensive programming)
            if not isinstance(ingredients, list):
                ingredients = []
            ingredients = [str(ing) for ing in ingredients if ing is not None]

            ingredients_json = json.dumps(ingredients)
            
            # Extract and normalize original URL for consistent duplicate checking
            raw_org_url = recipe.get('orgURL', '') or ''
            org_url = normalize_url(raw_org_url) if raw_org_url else ''
            
            # Get Mealie's updatedAt for incremental sync (from parameter or recipe data)
            if mealie_updated_at is None:
                mealie_updated_at = recipe.get('updatedAt') or recipe.get('dateUpdated') or ''

            # Compute ingredients_parsed if not provided
            if ingredients_parsed is None:
                ingredients_parsed = _check_ingredients_parsed(recipe)
            # Convert bool to int for SQLite storage (1 = parsed, 0 = unparsed)
            ingredients_parsed_value = 1 if ingredients_parsed else 0

            embedding_bytes = self._embedding_to_bytes(embedding)

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                # Insert/update main table
                conn.execute('''
                    INSERT OR REPLACE INTO recipes
                    (id, name, slug, org_url, description, searchable_text, tags, categories, ingredients, cuisine_primary, cuisine_secondary, cuisine_region, cuisine_confidence, embedding, updated_at, mealie_updated_at, ingredients_parsed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                ''', (recipe_id, name, slug, org_url, description, searchable_text, tags, categories, ingredients_json, cuisine_primary, cuisine_secondary, cuisine_region, cuisine_confidence, embedding_bytes, mealie_updated_at, ingredients_parsed_value))

                # Update FTS table - convert complex data to searchable strings
                ingredients_text = " ".join(ingredients) if ingredients else ""

                # Convert tags JSON to searchable text
                try:
                    tags_list = json.loads(tags) if tags else []
                    tags_text = " ".join([tag.get("name", "") for tag in tags_list if isinstance(tag, dict)])
                except:
                    tags_text = ""

                conn.execute('''
                    INSERT OR REPLACE INTO recipes_fts
                    (id, name, description, searchable_text, tags, ingredients)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (recipe_id, name, description, searchable_text, tags_text, ingredients_text))

                # Add to USearch index (incremental update <1ms)
                if self.ann_index is not None:
                    self.ann_index.add(recipe_id, embedding)
                    if auto_save:
                        self.ann_index.save()  # Persist immediately (can be disabled for batch operations)

            return True

        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to store recipe {recipe.get('name', 'unknown')}: {e}")
            logger.debug("Full traceback:")
            traceback.print_exc()
            return False

    def _create_searchable_text(self, recipe: Dict[str, Any]) -> str:
        """
        Create a comprehensive searchable text representation of the recipe.

        Args:
            recipe: Full recipe data

        Returns:
            Searchable text string
        """
        parts = []

        # Basic info
        if recipe.get("name"):
            parts.append(f"Recipe: {recipe['name']}")

        if recipe.get("description"):
            parts.append(f"Description: {recipe['description']}")

        # Ingredients summary
        ingredients = recipe.get("recipeIngredient", [])
        if ingredients:
            ingredient_names = []
            for ing in ingredients[:10]:  # Limit to first 10 ingredients
                if isinstance(ing, dict):
                    # Handle case where food field is explicitly null vs missing
                    food_obj = ing.get("food")
                    if food_obj and isinstance(food_obj, dict):
                        food = food_obj.get("name", "")
                        if food:
                            ingredient_names.append(food)
                elif isinstance(ing, str) and ing.strip():
                    # Handle unparsed ingredients
                    ingredient_names.append(ing.strip())

            if ingredient_names:
                parts.append(f"Ingredients: {', '.join(ingredient_names)}")

        # Tags and categories
        tags = recipe.get("tags", [])
        categories = recipe.get("recipeCategory", [])
        if tags or categories:
            # Extract tag/category names from dict objects
            tag_names = []
            for tag in tags + categories:
                if isinstance(tag, dict):
                    name = tag.get("name", "")
                    if name:
                        tag_names.append(name)
                elif isinstance(tag, str):
                    tag_names.append(tag)

            if tag_names:
                parts.append(f"Tags: {', '.join(tag_names)}")

        # Instructions summary (first few steps)
        instructions = recipe.get("recipeInstructions", [])
        if instructions:
            # Extract text from instruction dicts
            instruction_texts = []
            for inst in instructions[:3]:  # First 3 instructions
                if isinstance(inst, dict):
                    text = inst.get("text", "")
                    if text:
                        instruction_texts.append(text)
                elif isinstance(inst, str):
                    instruction_texts.append(inst)

            if instruction_texts:
                instruction_text = " ".join(instruction_texts)
                if len(instruction_text) > 200:
                    instruction_text = instruction_text[:200] + "..."
                parts.append(f"Method: {instruction_text}")

        return " | ".join(parts)

    def _summarize_ingredients(self, ingredients: List[Dict]) -> List[str]:
        """
        Summarize ingredients into key food names.

        Args:
            ingredients: List of ingredient objects

        Returns:
            List of key ingredient names
        """
        key_ingredients = []

        for ing in ingredients:
            if isinstance(ing, dict):
                # Parsed ingredient - handle case where food field is explicitly null
                food_obj = ing.get("food")
                if food_obj and isinstance(food_obj, dict):
                    food = food_obj.get("name", "")
                    if food:
                        key_ingredients.append(food.lower())
            elif isinstance(ing, str):
                # Unparsed ingredient - extract likely food names
                # Simple heuristic: take words that look like food
                words = ing.lower().split()
                for word in words:
                    word = word.strip('.,()')
                    if len(word) > 2 and word not in ['the', 'and', 'for', 'with', 'into', 'over']:
                        key_ingredients.append(word)
                        break  # Just take first likely food word

        # Remove duplicates and common words
        common_words = {'salt', 'pepper', 'oil', 'water', 'sugar', 'flour', 'butter', 'milk', 'egg', 'eggs'}
        unique_ingredients = list(set(key_ingredients) - common_words)

        return unique_ingredients[:15]  # Limit to 15 key ingredients

    def _extract_cuisine_from_tags(self, tags: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
        """
        Extract cuisine information from Mealie tags for analysis.

        Args:
            tags: List of tag dictionaries from Mealie API

        Returns:
            Tuple of (primary_cuisine, secondary_cuisines_json, region, confidence)
        """
        primary_cuisine = None
        secondary_cuisines = []
        region = None
        confidence = 0.0

        for tag in tags:
            if not isinstance(tag, dict):
                continue

            tag_name = tag.get("name", "")
            if not tag_name:
                continue

            # Parse cuisine tags (format: "Cuisine: Italian - Northern" or "Italian Cuisine")
            if "Cuisine:" in tag_name or " Cuisine" in tag_name:
                confidence = 1.0  # Tagged cuisines are authoritative

                # Extract cuisine name
                if "Cuisine:" in tag_name:
                    # Format: "Cuisine: Italian - Northern"
                    cuisine_part = tag_name.split("Cuisine:")[1].strip()
                else:
                    # Format: "Italian Cuisine"
                    cuisine_part = tag_name.split(" Cuisine")[0].strip()

                # Handle regional variants (Cuisine - Region)
                if " - " in cuisine_part:
                    cuisine_name, region_name = cuisine_part.split(" - ", 1)
                    cuisine_name = cuisine_name.strip()
                    region_name = region_name.strip()

                    if not primary_cuisine:
                        primary_cuisine = cuisine_name
                        region = region_name
                    else:
                        secondary_cuisines.append(cuisine_name)
                else:
                    # Simple cuisine name
                    if not primary_cuisine:
                        primary_cuisine = cuisine_part
                    else:
                        secondary_cuisines.append(cuisine_part)

            # Also check for region tags
            elif "Region:" in tag_name:
                if not region:
                    region = tag_name.split("Region:")[1].strip()

        # Convert secondary cuisines to JSON string
        secondary_cuisines_json = json.dumps(secondary_cuisines) if secondary_cuisines else None

        return primary_cuisine, secondary_cuisines_json, region, confidence

    def find_recipes_for_concept(self, menu_concept: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find recipes that best match a menu concept using hybrid search.

        Args:
            menu_concept: Menu concept (e.g., "Cantonese beef stir-fry")
            top_k: Number of top matches to return

        Returns:
            List of matching recipes with scores
        """
        # Generate embedding for menu concept (FAIL FAST if embedding infra is broken)
        # Use is_query=True for search queries (Nemotron adds instruction prefix for retrieval)
        concept_embedding = self._generate_embedding(menu_concept, is_query=True)
        if concept_embedding is None:
            raise RuntimeError(
                f"Embedding generation failed for concept '{menu_concept}'. "
                "Cannot continue with degraded (keyword-only) search."
            )

        # Perform hybrid search
        vector_matches = self._vector_search(concept_embedding, top_k * 2)
        keyword_matches = self._keyword_search(menu_concept, top_k * 2)

        # Combine and rerank results
        combined_results = self._combine_results(vector_matches, keyword_matches, top_k)

        # Fetch full recipe details for top matches
        final_results = []
        for recipe_id, score in combined_results:
            recipe = self.get_recipe_by_id(recipe_id)
            if recipe:
                recipe["relevance_score"] = score
                final_results.append(recipe)

        return final_results

    @staticmethod
    def _fts_sanitize_query(raw_query: str) -> str:
        """
        Convert arbitrary user/menu text into a safe FTS5 MATCH query.

        FTS5 has its own query language; characters like '-' and '+' can change parsing.
        We avoid operator grammar entirely by extracting alphanumeric tokens and joining
        with AND (fail fast if we cannot produce any tokens).
        """
        if not raw_query or not raw_query.strip():
            raise ValueError("FTS query is empty")

        tokens = re.findall(r"[0-9A-Za-z]+", raw_query.lower())
        if not tokens:
            raise ValueError(f"FTS query has no searchable tokens: {raw_query!r}")

        # Join tokens with AND so all must appear; avoids FTS operator pitfalls.
        return " AND ".join(tokens)

    def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Perform vector similarity search.

        BEFORE: Linear scan O(n) - 3-5 seconds at 10k recipes
        AFTER: USearch ANN O(log n) - 5-8ms at 10k recipes

        Args:
            query_embedding: Embedding vector for query
            top_k: Number of top results

        Returns:
            List of (recipe_id, similarity_score) tuples
        """
        # Use USearch if available, fallback to linear scan
        if self.ann_index is not None:
            distances, recipe_ids = self.ann_index.search(query_embedding, k=top_k)

            # Return as (recipe_id, similarity) tuples
            return list(zip(recipe_ids, distances))
        else:
            # Fallback to original linear scan
            return self._vector_search_linear(query_embedding, top_k)

    def _vector_search_linear(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Original linear scan implementation (fallback).

        Args:
            query_embedding: Embedding vector for query
            top_k: Number of top results

        Returns:
            List of (recipe_id, similarity_score) tuples
        """
        results = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, embedding FROM recipes WHERE embedding IS NOT NULL')

            for recipe_id, embedding_bytes in cursor:
                if embedding_bytes:
                    recipe_embedding = self._bytes_to_embedding(embedding_bytes)
                    similarity = self._cosine_similarity(query_embedding, recipe_embedding)
                    results.append((recipe_id, similarity))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _ensure_fts_integrity(self):
        """Ensure FTS table exists and is populated. Rebuild if corrupted."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if FTS table exists
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recipes_fts'")
                if not cursor.fetchone():
                    logger.warning("FTS table missing, recreating...")
                    self._create_fts_table(conn)
                    return

                # Check if FTS is populated
                cursor = conn.execute("SELECT COUNT(*) FROM recipes_fts")
                fts_count = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM recipes")
                main_count = cursor.fetchone()[0]

                if fts_count < main_count * 0.9:  # 10% tolerance
                    logger.warning(f"FTS out of sync ({fts_count} vs {main_count} recipes), rebuilding...")
                    self._rebuild_fts_index(conn)

        except Exception as e:
            logger.critical(f"❌ CRITICAL FAILURE: FTS integrity check failed: {e}")
            raise RuntimeError(f"FTS integrity check failed: {e}. Cannot continue with broken search infrastructure.") from e

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Perform keyword-based search using FTS.

        Args:
            query: Search query
            top_k: Number of top results

        Returns:
            List of (recipe_id, relevance_score) tuples
        """
        # Ensure FTS integrity before searching
        self._ensure_fts_integrity()

        results = []

        with sqlite3.connect(self.db_path) as conn:
            try:
                # Use FTS to find matches - MUST sanitize to avoid FTS query-language pitfalls
                safe_query = self._fts_sanitize_query(query)

                cursor = conn.execute('''
                    SELECT id, bm25(recipes_fts) as score
                    FROM recipes_fts
                    WHERE recipes_fts MATCH ?
                    ORDER BY bm25(recipes_fts)
                    LIMIT ?
                ''', (safe_query, top_k * 2))

                for recipe_id, score in cursor:
                    # Convert BM25 score to 0-1 range (lower BM25 = better match)
                    normalized_score = 1.0 / (1.0 + abs(score)) if score < 0 else 0.5
                    results.append((recipe_id, normalized_score))

            except sqlite3.OperationalError as e:
                # NO FALLBACK - FTS is critical for quality search
                logger.critical(f"❌ CRITICAL FAILURE: FTS search failed ({e})")
                logger.critical(f"❌ Query: {query}")
                logger.critical(f"❌ This breaks semantic recipe matching - fix FTS instead of falling back to primitive LIKE")
                raise RuntimeError(f"FTS search failed for query '{query}': {e}. Cannot continue with degraded search.") from e

        return results[:top_k]

    def _combine_results(self, vector_results: List[Tuple[str, float]],
                        keyword_results: List[Tuple[str, float]],
                        top_k: int) -> List[Tuple[str, float]]:
        """
        Combine vector and keyword search results with reranking.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            top_k: Final number of results

        Returns:
            Combined and reranked results
        """
        # Create score dictionaries
        vector_scores = dict(vector_results)
        keyword_scores = dict(keyword_results)

        # Combine scores: 70% vector similarity, 30% keyword relevance
        combined_scores = {}

        all_recipe_ids = set(vector_scores.keys()) | set(keyword_scores.keys())

        for recipe_id in all_recipe_ids:
            vector_score = vector_scores.get(recipe_id, 0.0)
            keyword_score = keyword_scores.get(recipe_id, 0.0)

            combined_score = (vector_score * 0.7) + (keyword_score * 0.3)
            combined_scores[recipe_id] = combined_score

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:top_k]

    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full recipe data by ID.

        Args:
            recipe_id: Recipe ID

        Returns:
            Recipe data dictionary or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM recipes WHERE id = ?', (recipe_id,))
                row = cursor.fetchone()

                if row:
                    # Convert back to dictionary
                    columns = [desc[0] for desc in cursor.description]
                    recipe = dict(zip(columns, row))

                    # Parse JSON fields
                    if recipe.get('tags'):
                        recipe['tags'] = json.loads(recipe['tags'])
                    if recipe.get('categories'):
                        recipe['categories'] = json.loads(recipe['categories'])
                    if recipe.get('ingredients'):
                        recipe['ingredients'] = json.loads(recipe['ingredients'])

                    return recipe

        except Exception as e:
            logger.error(f"❌ Error fetching recipe {recipe_id}: {e}")

        return None

    def get_recipe_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Get recipe data by slug.

        Args:
            slug: Recipe slug

        Returns:
            Recipe data dictionary with id, name, slug, or None if not found
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.execute(
                    'SELECT id, name, slug FROM recipes WHERE slug = ? LIMIT 1',
                    (slug,)
                )
                row = cursor.fetchone()
                if row:
                    return {"id": row[0], "name": row[1], "slug": row[2]}
        except Exception as e:
            logger.error(f"❌ Error fetching recipe by slug {slug}: {e}")
        return None

    def get_total_recipes(self) -> int:
        """Get total number of indexed recipes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM recipes')
            return cursor.fetchone()[0]
    
    def _is_recipe_indexed(self, recipe_id: str) -> bool:
        """Check if a recipe is already indexed."""
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.execute('SELECT 1 FROM recipes WHERE id = ? LIMIT 1', (recipe_id,))
            return cursor.fetchone() is not None
    
    def get_mealie_updated_at(self, recipe_id: str) -> Optional[str]:
        """Get the stored Mealie updatedAt timestamp for a recipe."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT mealie_updated_at FROM recipes WHERE id = ? LIMIT 1', (recipe_id,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def get_all_recipe_timestamps(self) -> Dict[str, str]:
        """
        Get all recipe IDs and their stored Mealie timestamps.
        Returns dict of {recipe_id: mealie_updated_at}.
        Used for efficient bulk comparison during sync.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, mealie_updated_at FROM recipes')
            return {row[0]: row[1] for row in cursor.fetchall()}
    
    def needs_reindex(self, recipe_id: str, mealie_updated_at: str) -> bool:
        """
        Check if a recipe needs re-indexing based on timestamp comparison.
        
        Returns True if:
        - Recipe is not indexed yet
        - Stored timestamp is NULL (never synced with timestamp tracking)
        - Mealie's timestamp is newer than stored timestamp
        """
        stored_ts = self.get_mealie_updated_at(recipe_id)
        
        # Not indexed or no stored timestamp = needs indexing
        if stored_ts is None or stored_ts == '':
            return True
        
        # No Mealie timestamp provided = can't compare, assume needs update
        if not mealie_updated_at:
            return True
        
        # Compare timestamps (ISO format strings compare correctly)
        # Mealie: "2026-01-07T13:25:14.227981+00:00"
        # Stored: "2026-01-07T13:25:14.227981+00:00" (same format)
        return mealie_updated_at > stored_ts
    
    def get_existing_urls(self) -> set:
        """Get all existing recipe URLs from local cache for fast duplicate checking.
        
        Returns normalized URLs for consistent comparison.
        """
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.execute('SELECT org_url FROM recipes WHERE org_url IS NOT NULL AND org_url != ""')
            # Normalize URLs for consistent comparison (handles legacy non-normalized data)
            return {normalize_url(row[0]) for row in cursor.fetchall() if row[0]}
    
    def has_url(self, url: str) -> bool:
        """Check if a URL already exists in local cache.
        
        Handles both normalized URLs (new data) and legacy non-normalized URLs.
        """
        if not url:
            return False
        normalized = normalize_url(url)
        if not normalized:
            return False
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            # Check for normalized URL (new format) or original URL (legacy)
            # This handles both old data (non-normalized) and new data (normalized)
            cursor = conn.execute(
                'SELECT 1 FROM recipes WHERE org_url = ? OR org_url = ? LIMIT 1', 
                (normalized, url)
            )
            return cursor.fetchone() is not None

    def get_recipes_by_parsed_status(self, parsed: bool) -> List[str]:
        """
        Get recipe slugs by ingredient parsing status.

        Args:
            parsed: If True, return recipes where ingredients_parsed = 1.
                    If False, return recipes where ingredients_parsed = 0.

        Returns:
            List of recipe slugs matching the specified parsing status.
        """
        status_value = 1 if parsed else 0
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.execute(
                'SELECT slug FROM recipes WHERE ingredients_parsed = ?',
                (status_value,)
            )
            return [row[0] for row in cursor.fetchall() if row[0]]

    def get_recipes_with_unknown_parsed_status(self) -> List[str]:
        """
        Get recipe slugs where ingredient parsing status is unknown.

        Returns:
            List of recipe slugs where ingredients_parsed IS NULL.
        """
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.execute(
                'SELECT slug FROM recipes WHERE ingredients_parsed IS NULL'
            )
            return [row[0] for row in cursor.fetchall() if row[0]]

    def update_parsed_status(self, recipe_id: str, is_parsed: bool) -> None:
        """
        Update the ingredient parsing status for a recipe.

        Uses BEGIN IMMEDIATE to acquire write lock immediately, preventing
        race conditions when multiple processes update parsing status.

        Args:
            recipe_id: The recipe ID (UUID) to update.
            is_parsed: True to mark as parsed (1), False to mark as unparsed (0).
        """
        status_value = 1 if is_parsed else 0
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute('BEGIN IMMEDIATE')  # Acquire write lock immediately
            conn.execute(
                'UPDATE recipes SET ingredients_parsed = ? WHERE id = ?',
                (status_value, recipe_id)
            )
            conn.commit()

    def update_parsed_status_by_slug(self, slug: str, is_parsed: bool) -> None:
        """
        Update the ingredient parsing status for a recipe by slug.

        Args:
            slug: The recipe slug to update.
            is_parsed: True to mark as parsed (1), False to mark as unparsed (0).
        """
        status_value = 1 if is_parsed else 0
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute(
                'UPDATE recipes SET ingredients_parsed = ? WHERE slug = ?',
                (status_value, slug)
            )
            conn.commit()

    def batch_update_parsed_status(self, slugs: List[str], is_parsed: bool) -> int:
        """
        Batch update ingredient parsing status for multiple recipes.

        Uses a single transaction for all updates, much faster than individual updates.

        Args:
            slugs: List of recipe slugs to update.
            is_parsed: True to mark as parsed (1), False to mark as unparsed (0).

        Returns:
            Number of recipes updated.
        """
        if not slugs:
            return 0
        
        status_value = 1 if is_parsed else 0
        with sqlite3.connect(self.db_path, timeout=60) as conn:
            conn.execute('BEGIN IMMEDIATE')
            # Use executemany for batch updates
            conn.executemany(
                'UPDATE recipes SET ingredients_parsed = ? WHERE slug = ?',
                [(status_value, slug) for slug in slugs]
            )
            conn.commit()
        return len(slugs)

    def get_last_sync_timestamp(self) -> Optional['datetime']:
        """
        Get the most recent updated_at timestamp from the recipes table.

        This is useful for health check warnings to detect stale data.

        Returns:
            The most recent updated_at datetime, or None if no recipes exist.
        """
        from datetime import datetime
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            cursor = conn.execute(
                'SELECT MAX(updated_at) FROM recipes'
            )
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    # SQLite stores timestamps as strings in ISO format
                    return datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    return None
            return None

    def clear_index(self):
        """Clear all indexed recipes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM recipes')
            conn.execute('DELETE FROM recipes_fts')

    def rebuild_fts_index(self):
        """Rebuild the full-text search index."""
        with sqlite3.connect(self.db_path) as conn:
            # Re-populate FTS from main table
            conn.execute('DELETE FROM recipes_fts')
            conn.execute('''
                INSERT INTO recipes_fts (id, name, description, searchable_text, tags, ingredients)
                SELECT id, name, description, searchable_text, tags, ingredients FROM recipes
            ''')


    def analyze_menu_history(self, history_recipes: List[str]) -> Dict[str, List[str]]:
        """
        Analyze recent meal history to extract variety constraints for menu planning.

        Args:
            history_recipes: List of recently served recipe names

        Returns:
            Dictionary with variety constraints for LLM prompts
        """
        if not history_recipes:
            return {
                "recent_proteins": [],
                "recent_cuisines": [],
                "recent_dish_types": []
            }

        # Extract variety patterns from recipe names
        proteins = []
        cuisines = []
        dish_types = []

        for recipe_name in history_recipes:
            recipe_lower = recipe_name.lower()

            # Extract protein families
            if any(word in recipe_lower for word in ['fish', 'salmon', 'cod', 'sea bass', 'snapper']):
                proteins.append('fish')
            elif any(word in recipe_lower for word in ['chicken', 'turkey']):
                proteins.append('chicken')
            elif any(word in recipe_lower for word in ['pork', 'bacon', 'ham']):
                proteins.append('pork')
            elif any(word in recipe_lower for word in ['beef', 'steak', 'burger']):
                proteins.append('beef')
            elif any(word in recipe_lower for word in ['shrimp', 'prawn', 'seafood', 'crab', 'lobster']):
                proteins.append('seafood')
            elif any(word in recipe_lower for word in ['tofu', 'bean', 'lentil', 'vegetable', 'mushroom']):
                proteins.append('vegetarian')
            elif any(word in recipe_lower for word in ['egg', 'omelette']):
                proteins.append('eggs')

            # Extract cuisine types
            if any(word in recipe_lower for word in ['cantonese', 'chinese', 'wok', 'stir fry', 'dim sum']):
                cuisines.append('cantonese')
            elif any(word in recipe_lower for word in ['japanese', 'sushi', 'tempura', 'ramen', 'miso']):
                cuisines.append('japanese')
            elif any(word in recipe_lower for word in ['thai', 'curry', 'pad thai', 'tom yum']):
                cuisines.append('thai')
            elif any(word in recipe_lower for word in ['italian', 'pasta', 'pizza', 'risotto']):
                cuisines.append('italian')
            elif any(word in recipe_lower for word in ['american', 'burger', 'bbq', 'grill']):
                cuisines.append('american')
            elif any(word in recipe_lower for word in ['indian', 'masala', 'biryani']):
                cuisines.append('indian')
            else:
                cuisines.append('fusion')  # Default for unrecognized

            # Extract dish types
            if any(word in recipe_lower for word in ['stir fry', 'wok', 'fry']):
                dish_types.append('stir_fry')
            elif any(word in recipe_lower for word in ['grill', 'bbq', 'barbecue']):
                dish_types.append('grilled')
            elif any(word in recipe_lower for word in ['curry', 'stew', 'braise']):
                dish_types.append('braised')
            elif any(word in recipe_lower for word in ['steam', 'dumpling']):
                dish_types.append('steamed')
            elif any(word in recipe_lower for word in ['bake', 'roast', 'oven']):
                dish_types.append('baked')
            elif any(word in recipe_lower for word in ['soup', 'broth']):
                dish_types.append('soup')
            elif any(word in recipe_lower for word in ['pasta', 'noodle']):
                dish_types.append('pasta')
            elif any(word in recipe_lower for word in ['rice', 'pilaf']):
                dish_types.append('rice')
            elif any(word in recipe_lower for word in ['salad', 'vegetable']):
                dish_types.append('vegetable')
            else:
                dish_types.append('other')

        # Get most recent patterns (last 10 meals for context)
        recent_limit = min(10, len(proteins))

        return {
            "recent_proteins": proteins[-recent_limit:],
            "recent_cuisines": cuisines[-recent_limit:],
            "recent_dish_types": dish_types[-recent_limit:]
        }

    def analyze_recipe_database(self) -> Dict[str, Any]:
        """
        Analyze the entire recipe database for menu planning constraints.

        Returns:
            Dictionary with database statistics and constraints for LLM prompts
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get total recipes
            cursor.execute("SELECT COUNT(*) FROM recipes")
            total_recipes = cursor.fetchone()[0]

            # Get recipes with embeddings
            cursor.execute("SELECT COUNT(*) FROM recipes WHERE embedding IS NOT NULL")
            recipes_with_embeddings = cursor.fetchone()[0]

            # Get cuisine distribution
            cursor.execute("""
                SELECT cuisine_primary, COUNT(*) as count
                FROM recipes
                WHERE cuisine_primary IS NOT NULL
                GROUP BY cuisine_primary
                ORDER BY count DESC
            """)
            cuisine_counts = dict(cursor.fetchall())

            # Get protein distribution from searchable text
            protein_families = {
                "chicken": 0, "beef": 0, "pork": 0, "fish": 0, "seafood": 0,
                "lamb": 0, "turkey": 0, "duck": 0, "tofu": 0, "eggs": 0, "vegetarian": 0
            }

            cursor.execute("SELECT searchable_text FROM recipes WHERE searchable_text IS NOT NULL")
            for row in cursor.fetchall():
                text = row[0].lower()
                for protein in protein_families.keys():
                    if protein in text:
                        protein_families[protein] += 1

            # Get dish type distribution
            dish_types = {
                "stir_fry": 0, "grilled": 0, "braised": 0, "steamed": 0, "baked": 0,
                "soup": 0, "pasta": 0, "rice": 0, "salad": 0, "other": 0
            }

            cursor.execute("SELECT searchable_text FROM recipes WHERE searchable_text IS NOT NULL")
            for row in cursor.fetchall():
                text = row[0].lower()
                for dish_type in dish_types.keys():
                    if dish_type.replace('_', ' ') in text:
                        dish_types[dish_type] += 1

            conn.close()

            return {
                "total_recipes": total_recipes,
                "recipes_with_embeddings": recipes_with_embeddings,
                "embedding_coverage": recipes_with_embeddings / max(total_recipes, 1),
                "cuisine_distribution": cuisine_counts,
                "unique_cuisines": len(cuisine_counts),
                "protein_families": protein_families,
                "dish_types": dish_types,
                "available_cuisines": list(cuisine_counts.keys()),
                "available_proteins": [p for p, count in protein_families.items() if count > 0]
            }

        except Exception as e:
            logger.error(f"Error analyzing recipe database: {e}")
            return {
                "total_recipes": 0,
                "recipes_with_embeddings": 0,
                "embedding_coverage": 0.0,
                "cuisine_distribution": {},
                "unique_cuisines": 0,
                "protein_families": {},
                "dish_types": {},
                "available_cuisines": [],
                "available_proteins": []
            }


# Utility functions for external use
def index_all_mealie_recipes(rag: RecipeRAG, batch_size: int = None) -> int:
    """
    Index all recipes from Mealie into the RAG system.

    Args:
        rag: RecipeRAG instance
        batch_size: Number of recipes to process at once

    Returns:
        Number of recipes indexed
    """
    from mealie_client import MealieClient

    # Use centralized configuration if no explicit batch_size provided
    if batch_size is None:
        config = get_bulk_operation_config_safe('index', fallback_batch_size=50, fallback_concurrent=5)
        batch_size = config['default_batch_size']

    indexed_count = 0
    client = MealieClient()
    
    try:
        logger.info("📚 Indexing recipes from Mealie...")
        
        # Fetch all recipe summaries (MealieClient handles pagination internally)
        all_recipes = client.get_all_recipes()
        
        if not all_recipes:
            logger.info("No recipes found in Mealie")
            return 0
        
        logger.info(f"Found {len(all_recipes)} recipes to process")
        
        # Process recipes in batches for memory efficiency
        for i in range(0, len(all_recipes), batch_size):
            batch = all_recipes[i:i + batch_size]
            page = (i // batch_size) + 1
            
            # Index each recipe - fetch full details first
            for recipe_summary in batch:
                try:
                    recipe_id = recipe_summary.get("id")
                    slug = recipe_summary.get("slug")
                    
                    if not slug:
                        logger.warning(f"⚠️  Skipping recipe {recipe_summary.get('name', 'unknown')} - no slug")
                        continue
                    
                    # Skip if already indexed (huge performance gain - avoids fetching details)
                    if rag._is_recipe_indexed(recipe_id):
                        continue

                    # Fetch full recipe details using MealieClient
                    try:
                        recipe_detail = client.get_recipe(slug)
                        
                        # Use force=True since we already checked above
                        # Disable auto_save to avoid excessive disk writes (save once per page instead)
                        if rag.index_recipe(recipe_detail, force=True, auto_save=False):
                            indexed_count += 1
                        else:
                            logger.warning(f"⚠️  Failed to index: {recipe_summary.get('name', 'unknown')}")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to fetch details for {recipe_summary.get('name', 'unknown')}: {e}")

                except Exception as e:
                    logger.warning(f"⚠️  Error processing recipe {recipe_summary.get('name', 'unknown')}: {e}")

            # Save index once per page (instead of after every recipe)
            if rag.ann_index is not None and indexed_count > 0:
                try:
                    rag.ann_index.save()
                except Exception as e:
                    logger.error(f"❌ Failed to save ANN index after page {page}: {e}")
                    logger.warning(f"⚠️  Page {page} indexed in memory but not persisted")
                    # Continue processing - will retry save on next page
            
            logger.info(f"✅ Indexed {indexed_count} recipes (page {page})")

    except Exception as e:
        logger.error(f"❌ Error indexing recipes: {e}")
    finally:
        client.close()

    logger.info(f"🎉 Successfully indexed {indexed_count} recipes")
    return indexed_count


if __name__ == "__main__":
    # Test the RAG system
    rag = RecipeRAG()

    # Index a few recipes for testing
    indexed = index_all_mealie_recipes(rag, batch_size=10)
    print(f"Total recipes in index: {rag.get_total_recipes()}")

    # Test search
    if indexed > 0:
        results = rag.find_recipes_for_concept("chicken curry", top_k=3)
        print(f"\nSearch results for 'chicken curry':")
        for recipe in results:
            print(f"  - {recipe['name']} (score: {recipe.get('relevance_score', 0):.3f})")
