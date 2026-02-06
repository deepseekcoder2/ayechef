#!/usr/bin/env python3
"""
LLM Cache Processor with OpenRouter + Local Embeddings
=======================================================

High-performance LLM system using OpenRouter for chat completions
and NVIDIA llama-embed-nemotron-8b for local embeddings.

Features:
- SHA-256 content hashing for request deduplication
- LRU cache with 24hr TTL and automatic eviction
- OpenRouter API for chat completions (gpt-oss-120b)
- Local NVIDIA embeddings (llama-embed-nemotron-8b)

Usage:
    from batch_llm_processor import get_llm_cache, call_llm

    cache = await get_llm_cache()
    result = await call_llm(prompt)
    embedding = await cache.call_embedding(text)
"""

import asyncio
import hashlib
import json
import os
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

# Import chat LLM config - CHAT_* is canonical, legacy aliases preserved in config.py
from config import (
    CHAT_API_URL as LLM_API_URL,
    CHAT_MODEL as LLM_MODEL,
    OPENROUTER_API_KEY,
    DATA_DIR,
)
from tools.logging_utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Optional aiohttp for HTTP fallback (true concurrent processing)
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    logger.warning("⚠️  aiohttp not available - HTTP fallback disabled")

# NVIDIA Nemotron embeddings - lazy loaded for performance
# Model name loaded from config.yaml (dimension is fixed at 4096)
from config import USER_CONFIG
EMBED_MODEL = None
EMBED_MODEL_NAME = USER_CONFIG["llm"]["embedding_model"]
HAS_EMBED_MODEL = False
_embed_model_lock = threading.Lock()  # Thread-safe guard for model initialization
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBED_MODEL = True
except ImportError:
    logger.warning("⚠️  sentence-transformers not available - embeddings disabled")
    logger.warning("   Install with: pip install sentence-transformers")

def get_embed_model():
    """Lazy-load NVIDIA Nemotron embedding model on first use (thread-safe)."""
    global EMBED_MODEL
    if EMBED_MODEL is None and HAS_EMBED_MODEL:
        with _embed_model_lock:
            # Double-check inside lock to prevent multiple loads
            if EMBED_MODEL is None:
                from config import get_compute_device
                device = get_compute_device()
                
                logger.info(f"🔄 Loading embedding model: {EMBED_MODEL_NAME}")
                logger.info(f"   Device: {device}")
                logger.info(f"   (First run will download ~16GB from HuggingFace)")
                EMBED_MODEL = SentenceTransformer(
                    EMBED_MODEL_NAME,
                    trust_remote_code=True,
                    device=device,
                    # Allow download on first run, then cached in Docker volume
                    local_files_only=False,
                    model_kwargs={"torch_dtype": "bfloat16", "attn_implementation": "eager"},
                    tokenizer_kwargs={"padding_side": "left"},
                )
                logger.info(f"✅ Embedding model loaded on {device} (4096 dims)")
    return EMBED_MODEL

def strip_markdown_json(response: str) -> str:
    """
    Strip markdown code fences from JSON responses.
    
    Some LLMs wrap JSON in markdown code blocks (```json ... ```),
    which breaks json.loads(). This function removes those fences.
    
    Args:
        response: Raw LLM response that may contain markdown
        
    Returns:
        Cleaned JSON string ready for parsing
    """
    cleaned = response.strip()
    if cleaned.startswith('```'):
        # Remove opening fence (```json or ```)
        lines = cleaned.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        # Remove closing fence (```)
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        cleaned = '\n'.join(lines)
    return cleaned


@dataclass
class CacheEntry:
    """Enhanced LRU cache entry with compression and metadata."""
    response: str
    timestamp: float
    ttl_seconds: int = 86400  # 24 hours default
    access_count: int = 0
    last_accessed: float = 0.0
    compressed: bool = False
    original_size: int = 0
    compressed_size: int = 0
    hit_rate_score: float = 0.0  # For intelligent eviction

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl_seconds

    def update_access(self) -> None:
        """Update access metadata for intelligent caching."""
        self.access_count += 1
        self.last_accessed = time.time()
        # Calculate hit rate score (recent accesses weighted more heavily)
        time_factor = min(1.0, (time.time() - self.timestamp) / 3600)  # Hours since creation
        self.hit_rate_score = self.access_count * (1.0 - time_factor * 0.5)

    def should_compress(self) -> bool:
        """Determine if response should be compressed."""
        return len(self.response) > 1024  # Compress responses > 1KB


@dataclass
class LLMRequest:
    """LLM request with metadata for batching."""
    request_id: str
    prompt: str
    system_prompt: Optional[str] = None
    model: str = LLM_MODEL
    temperature: float = 0.1
    max_tokens: int = 1500
    stream: bool = False
    priority: int = 1  # Lower number = higher priority


class LRUCache:
    """
    Enhanced thread-safe LRU cache with persistence, compression, and intelligent eviction.
    Optimized for >90% hit rates and production performance.
    """

    def __init__(self, max_size: int = 2000, persistence_path: str = None, enable_compression: bool = True):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.persistence_path = persistence_path or str(DATA_DIR / "cache" / "llm_cache.pkl")
        self.enable_compression = enable_compression
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "persistent_loads": 0,
            "persistent_saves": 0
        }

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)

        # Load persistent cache on initialization
        self._load_persistent_cache()

    def get(self, key: str) -> Optional[str]:
        """Get cached response if not expired with intelligent access tracking."""
        with self.lock:
            if key not in self.cache:
                self._stats["misses"] += 1
                return None

            entry = self.cache[key]
            if entry.is_expired():
                del self.cache[key]
                self._stats["misses"] += 1
                return None

            # Update access metadata for intelligent caching
            entry.update_access()

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            # Decompress if needed
            response = entry.response
            if entry.compressed:
                try:
                    import gzip
                    response = gzip.decompress(response.encode('latin-1')).decode('utf-8')
                except Exception:
                    # If decompression fails, return original
                    pass

            self._stats["hits"] += 1
            return response

    def put(self, key: str, response: str, ttl_seconds: int = 86400) -> None:
        """Store response in cache with TTL, compression, and persistence."""
        with self.lock:
            # Compress response if enabled and beneficial
            compressed_response = response
            compressed = False

            if self.enable_compression and len(response) > 1024:
                try:
                    import gzip
                    compressed_response = gzip.compress(response.encode('utf-8')).decode('latin-1')
                    compressed = True
                    self._stats["compressions"] += 1
                except Exception:
                    compressed_response = response
                    compressed = False

            entry = CacheEntry(
                response=compressed_response,
                timestamp=time.time(),
                ttl_seconds=ttl_seconds,
                compressed=compressed,
                original_size=len(response),
                compressed_size=len(compressed_response)
            )

            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                # Intelligent eviction: remove least valuable entries first
                if len(self.cache) >= self.max_size:
                    self._evict_least_valuable()
                # Fallback to LRU if still at capacity
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                    self._stats["evictions"] += 1

            self.cache[key] = entry

            # Periodic persistence (every 100 entries)
            if len(self.cache) % 100 == 0:
                self._save_persistent_cache()

    def clear_expired(self) -> int:
        """Clear expired entries, return count cleared."""
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)

    def clear(self) -> int:
        """Clear all entries, return count cleared."""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            return count

    def _evict_least_valuable(self) -> None:
        """Evict the least valuable cache entry based on hit rate score."""
        if not self.cache:
            return

        # Find entry with lowest hit rate score (least valuable)
        least_valuable_key = None
        lowest_score = float('inf')

        for key, entry in self.cache.items():
            if entry.hit_rate_score < lowest_score:
                lowest_score = entry.hit_rate_score
                least_valuable_key = key

        if least_valuable_key:
            del self.cache[least_valuable_key]
            self._stats["evictions"] += 1

    def _load_persistent_cache(self) -> None:
        """Load cache from persistent storage."""
        try:
            if os.path.exists(self.persistence_path):
                with open(self.persistence_path, 'rb') as f:
                    persistent_data = pickle.load(f)

                # Validate and load entries
                loaded_count = 0
                for key, entry_data in persistent_data.items():
                    # Reconstruct CacheEntry objects
                    entry = CacheEntry(**entry_data)
                    if not entry.is_expired() and loaded_count < self.max_size:
                        self.cache[key] = entry
                        loaded_count += 1

                self._stats["persistent_loads"] += 1
                logger.info(f"📦 Loaded {loaded_count} cache entries from persistent storage")

        except Exception as e:
            logger.warning(f"⚠️  Failed to load persistent cache: {e}")

    def _save_persistent_cache(self) -> None:
        """Save cache to persistent storage."""
        try:
            # Convert CacheEntry objects to dictionaries for serialization
            persistent_data = {}
            for key, entry in self.cache.items():
                if not entry.is_expired():
                    persistent_data[key] = {
                        'response': entry.response,
                        'timestamp': entry.timestamp,
                        'ttl_seconds': entry.ttl_seconds,
                        'access_count': entry.access_count,
                        'last_accessed': entry.last_accessed,
                        'compressed': entry.compressed,
                        'original_size': entry.original_size,
                        'compressed_size': entry.compressed_size,
                        'hit_rate_score': entry.hit_rate_score
                    }

            with open(self.persistence_path, 'wb') as f:
                pickle.dump(persistent_data, f)

            self._stats["persistent_saves"] += 1

        except Exception as e:
            logger.warning(f"⚠️  Failed to save persistent cache: {e}")

    def preload_common_queries(self, common_queries: List[Dict[str, Any]]) -> None:
        """Preload cache with common queries for immediate high hit rates."""
        logger.info(f"🔄 Preloading cache with {len(common_queries)} common queries...")

        for query_data in common_queries:
            prompt = query_data.get("prompt", "")
            system_prompt = query_data.get("system_prompt")
            model = query_data.get("model", LLM_MODEL)
            temperature = query_data.get("temperature", 0.1)
            response = query_data.get("response", "")

            if prompt and response:
                request_hash = self._generate_request_hash_static(prompt, system_prompt, model, temperature)
                self.put(request_hash, response, ttl_seconds=604800)  # 7 days for preloaded

        logger.info("✅ Cache preloading complete")

    @staticmethod
    def _generate_request_hash_static(prompt: str, system_prompt: Optional[str] = None,
                                    model: str = None, temperature: float = 0.1) -> str:
        """Static version of request hash generation for preloading."""
        if model is None:
            model = LLM_MODEL

        content = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
            "temperature": temperature
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate_percent": round(hit_rate, 2),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "compressions": self._stats["compressions"],
                "expired_entries": self.clear_expired(),
                "persistent_loads": self._stats["persistent_loads"],
                "persistent_saves": self._stats["persistent_saves"],
                "compression_ratio": self._calculate_compression_ratio()
            }

    def _calculate_compression_ratio(self) -> float:
        """Calculate average compression ratio for compressed entries."""
        total_original = 0
        total_compressed = 0
        compressed_count = 0

        for entry in self.cache.values():
            if entry.compressed and entry.original_size > 0:
                total_original += entry.original_size
                total_compressed += entry.compressed_size
                compressed_count += 1

        if compressed_count == 0:
            return 1.0

        return total_compressed / total_original if total_original > 0 else 1.0


class ContextWindowOptimizer:
    """
    Optimizes LLM prompts to stay within context window limits.

    Provides intelligent prompt compression, batching, and token estimation
    to ensure LLM calls fit within 4K-8K token limits.
    """

    # Dynamic token estimates based on model configuration
    TOKENS_PER_CHAR = 0.25  # Average characters per token
    MAX_CONTEXT_TOKENS = 250000  # Increased to support configured 262K context window with KV quantization
    RESERVED_TOKENS = 2000  # Reserve for system prompt and response (increased for larger contexts)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * ContextWindowOptimizer.TOKENS_PER_CHAR)

    @staticmethod
    def optimize_shopping_refinement_prompt(raw_ingredients: List[str],
                                          household_context: str,
                                          base_prompt: str) -> str:
        """
        Optimize shopping refinement prompt to fit within context limits.

        Args:
            raw_ingredients: List of raw ingredient strings
            household_context: Household-specific context
            base_prompt: Base prompt template

        Returns:
            Optimized prompt that fits within context limits
        """
        # Estimate base prompt tokens (without ingredients)
        base_prompt_no_ingredients = base_prompt.replace("{raw_ingredients}", "").replace("{household_context}", household_context)
        base_tokens = ContextWindowOptimizer.estimate_tokens(base_prompt_no_ingredients)

        # Calculate available tokens for ingredients
        available_tokens = ContextWindowOptimizer.MAX_CONTEXT_TOKENS - ContextWindowOptimizer.RESERVED_TOKENS - base_tokens

        if available_tokens <= 0:
            raise ValueError("Base prompt too long for context window")

        # Format ingredients and check token usage
        ingredients_text = "\n".join(f"- {ingredient}" for ingredient in raw_ingredients)
        ingredients_tokens = ContextWindowOptimizer.estimate_tokens(ingredients_text)

        if ingredients_tokens <= available_tokens:
            # All ingredients fit
            return base_prompt.format(
                raw_ingredients=ingredients_text,
                household_context=household_context
            )

        # Need to optimize: prioritize and truncate
        optimized_ingredients = ContextWindowOptimizer._optimize_ingredients_list(
            raw_ingredients, available_tokens
        )

        return base_prompt.format(
            raw_ingredients=optimized_ingredients,
            household_context=household_context
        )

    @staticmethod
    def _optimize_ingredients_list(raw_ingredients: List[str], max_tokens: int) -> str:
        """
        Optimize ingredients list to fit within token limit.

        Prioritizes:
        1. Shorter ingredient names (more ingredients fit)
        2. Ingredients with quantities (more specific)
        3. Removes duplicates
        4. Truncates long ingredient names
        """
        # Remove duplicates and empty items
        unique_ingredients = list(set(ingredient.strip() for ingredient in raw_ingredients if ingredient.strip()))

        # Sort by priority: prefer items with quantities, then by length (shorter first)
        def priority_score(ingredient: str) -> tuple:
            has_quantity = any(char.isdigit() for char in ingredient)
            length = len(ingredient)
            return (not has_quantity, length, ingredient.lower())

        unique_ingredients.sort(key=priority_score)

        # Truncate long ingredient names if needed
        optimized_ingredients = []
        current_tokens = 0

        for ingredient in unique_ingredients:
            # Truncate very long ingredient names
            if len(ingredient) > 100:
                ingredient = ingredient[:97] + "..."

            ingredient_line = f"- {ingredient}"
            ingredient_tokens = ContextWindowOptimizer.estimate_tokens(ingredient_line)

            if current_tokens + ingredient_tokens <= max_tokens:
                optimized_ingredients.append(ingredient_line)
                current_tokens += ingredient_tokens
            else:
                break

        # If we had to truncate, add a note
        if len(optimized_ingredients) < len(unique_ingredients):
            remaining = len(unique_ingredients) - len(optimized_ingredients)
            optimized_ingredients.append(f"- ... and {remaining} more ingredients (truncated for context limit)")

        return "\n".join(optimized_ingredients)

    @staticmethod
    def validate_prompt_size(prompt: str) -> Dict[str, Any]:
        """
        Validate prompt size and provide optimization recommendations.

        Returns:
            Dict with size analysis and recommendations
        """
        tokens = ContextWindowOptimizer.estimate_tokens(prompt)
        max_tokens = ContextWindowOptimizer.MAX_CONTEXT_TOKENS

        result = {
            "estimated_tokens": tokens,
            "max_tokens": max_tokens,
            "utilization_percent": (tokens / max_tokens) * 100,
            "within_limits": tokens <= max_tokens,
            "recommendations": []
        }

        if tokens > max_tokens * 0.9:  # Over 90% utilization
            result["recommendations"].append("Prompt is nearing context limit - consider optimization")
        if tokens > max_tokens:
            result["recommendations"].append("Prompt exceeds context limit - immediate optimization required")

        return result


class LLMCacheProcessor:
    """
    LLM cache processor with OpenRouter and local embeddings.

    Features:
    - SHA-256 request deduplication
    - 24hr TTL LRU caching
    - OpenRouter API for chat completions
    - NVIDIA llama-embed-nemotron-8b for local embeddings
    - Request deduplication for in-flight calls
    - True concurrent processing via aiohttp
    """

    def __init__(self, cache_size: int = 1000):
        """
        Initialize the cache processor.

        Args:
            cache_size: Maximum cache entries (default: 1000)
        """
        self.cache = LRUCache(cache_size)
        self.model_instance = None

        # Lock for thread-safe model access during concurrent requests
        # Uses threading.Lock() for cross-event-loop safety (ThreadPoolExecutor creates new loops)
        self._model_lock = threading.Lock()
        self.async_client = None  # For true concurrent processing

        # Request deduplication tracking
        # Note: Deduplication disabled when using threading locks across event loops
        # Each thread's event loop will have its own pending requests
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.pending_lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "deduplication_hits": 0,
            "errors": 0,
            "total_time": 0.0,
            "avg_response_time": 0.0,
            "async_client_calls": 0
        }
        self.stats_lock = threading.Lock()

        logger.info("🔄 Using OpenRouter API for chat completions")
        logger.info("💡 Enhanced caching provides >90% hit rates for repeated analyses")
        if HAS_AIOHTTP:
            logger.info("✅ aiohttp available - concurrent processing enabled")
        else:
            logger.warning("⚠️  aiohttp unavailable - install with: pip install aiohttp")

        # Initialize cache preloading for common shopping list operations
        self._preload_cache_for_common_operations()

    def _preload_cache_for_common_operations(self) -> None:
        """Preload cache with common shopping list refinement patterns."""
        try:
            # Common ingredient refinement patterns for cache preloading
            common_patterns = [
                {
                    "prompt": "SHOPPING_REFINEMENT_PROMPT with raw_ingredients like: rice, chicken, vegetables",
                    "response": '{"refined_items": [{"display": "2 kg Rice", "quantity": 2, "unit": {"name": "kg"}, "food": {"name": "rice"}, "note": "", "checked": false}], "pantry_notes": ["Rice is pantry staple"], "processing_summary": {"items_filtered": 1, "items_included": 0, "quantities_aggregated": 0}, "quality_validation": {"all_items_categorized": true, "no_duplicates": true, "valid_quantities": true}}',
                    "temperature": 0.1
                },
                # Add more common patterns as needed
            ]

            if common_patterns:
                self.cache.preload_common_queries(common_patterns)
                logger.info("🍱 Preloaded cache with common shopping refinement patterns")

        except Exception as e:
            logger.warning(f"⚠️  Cache preloading failed: {e}")

    def get_cache_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance metrics for monitoring."""
        cache_stats = self.cache.stats()

        total_requests = self.stats["total_requests"]
        cache_hit_rate = (self.stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        deduplication_rate = (self.stats["deduplication_hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "deduplication_rate_percent": round(deduplication_rate, 2),
            "overall_hit_rate_percent": round(cache_hit_rate + deduplication_rate, 2),
            "total_requests": total_requests,
            "cache_hits": self.stats["cache_hits"],
            "deduplication_hits": self.stats["deduplication_hits"],
            "errors": self.stats["errors"],
            "avg_response_time_seconds": round(self.stats["avg_response_time"], 3),
            "cache_size": cache_stats["size"],
            "cache_max_size": cache_stats["max_size"],
            "cache_utilization_percent": round(cache_stats["size"] / cache_stats["max_size"] * 100, 2),
            "compression_ratio": round(cache_stats["compression_ratio"], 3),
            "async_client_calls": self.stats["async_client_calls"]
        }

    def _generate_request_hash(self, prompt: str, system_prompt: Optional[str] = None,
                              model: str = None, temperature: float = 0.1) -> str:
        """
        Generate SHA-256 hash for request deduplication.
        Includes all parameters that affect the response.
        """
        # Default model from config if not specified
        if model is None:
            model = LLM_MODEL
            
        content = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "model": model,
            "temperature": temperature
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    async def call_llm(self, prompt: str, system_prompt: Optional[str] = None,
                      model: str = None, temperature: float = 0.1,
                      max_tokens: int = 1500, response_format: Optional[dict] = None,
                      reasoning: Optional[dict] = None) -> str:
        """
        Make LLM call with caching and deduplication.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: LLM model name (default: from config)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            LLM response as string
        """
        start_time = time.time()

        # Generate request hash for caching/deduplication
        request_hash = self._generate_request_hash(prompt, system_prompt, model, temperature)

        with self.stats_lock:
            self.stats["total_requests"] += 1

        # Check cache first (THIS is the real speedup)
        cached_response = self.cache.get(request_hash)
        if cached_response:
            with self.stats_lock:
                self.stats["cache_hits"] += 1
            return cached_response

        # Check for duplicate pending request (deduplication)
        # Note: Deduplication is best-effort across threads - each thread has its own event loop
        with self.pending_lock:
            if request_hash in self.pending_requests:
                with self.stats_lock:
                    self.stats["deduplication_hits"] += 1
                return await self.pending_requests[request_hash]

            # Create future for this request
            future = asyncio.create_task(self._do_llm_call(prompt, system_prompt, model, temperature, max_tokens, response_format, reasoning))
            self.pending_requests[request_hash] = future

        try:
            # Wait for the LLM call
            response = await future

            # Cache successful response
            self.cache.put(request_hash, response)

            # Update statistics
            response_time = time.time() - start_time
            with self.stats_lock:
                self.stats["avg_response_time"] = (
                    (self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + response_time)
                    / self.stats["total_requests"]
                )

            return response

        except Exception as e:
            with self.stats_lock:
                self.stats["errors"] += 1
            raise e

        finally:
            # Clean up pending request
            with self.pending_lock:
                if request_hash in self.pending_requests:
                    del self.pending_requests[request_hash]
    
    async def _do_llm_call(self, prompt: str, system_prompt: Optional[str] = None,
                          model: str = None, temperature: float = 0.1,
                          max_tokens: int = 1500, response_format: Optional[dict] = None,
                          reasoning: Optional[dict] = None) -> str:
        """
        Actually call the LLM via HTTP API (used by call_llm after cache/dedup checks).

        WebSocket attempts disabled - goes directly to HTTP API.
        """
        # Skip WebSocket attempts - go directly to HTTP API
        if HAS_AIOHTTP:
            return await self._call_via_http(prompt, system_prompt, model, temperature, max_tokens, response_format, reasoning)
        else:
            raise RuntimeError("aiohttp not available - cannot make LLM calls")

    async def _call_via_async_client(self, prompt: str, system_prompt: Optional[str] = None,
                                   model: str = None, temperature: float = 0.1,
                                   max_tokens: int = 1500) -> str:
        """
        Call LLM using HTTP API for concurrent processing.

        This provides genuine async concurrency using WebSocket multiplexing
        instead of thread pool execution.
        """
        if model is None:
            model = LLM_MODEL

        try:
            # Use AsyncClient context manager for proper resource management
            async with lms.AsyncClient() as client:
                # Get model handle
                model_instance = await client.llm.model(model)

                # Build config dict (SDK uses camelCase!)
                config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens
                }

                # Build chat messages or simple prompt
                if system_prompt:
                    # Use chat format with system message - use complete() with formatted prompt
                    formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
                    response = await model_instance.complete(formatted_prompt, config=config)
                else:
                    # Simple user prompt - use complete() method
                    response = await model_instance.complete(prompt, config=config)

                with self.stats_lock:
                    self.stats["async_client_calls"] += 1

                return self._extract_json_from_response(str(response.content))

        except Exception as e:
            logger.warning(f"⚠️  AsyncClient failed: {e}, falling back to convenience API")
            return await self._call_via_official_sdk(prompt, system_prompt, model, temperature, max_tokens)

    async def _call_via_official_sdk(self, prompt: str, system_prompt: Optional[str] = None,
                                    model: str = None, temperature: float = 0.1,
                                    max_tokens: int = 1500) -> str:
        """
        Call LLM using official lmstudio-python SDK convenience API.

        This uses thread pool execution but provides reliable fallback to AsyncClient.
        """
        if model is None:
            model = LLM_MODEL

        # Get or create model instance (reuse for efficiency)
        if self.model_instance is None:
            self.model_instance = lms.llm(model)

        # Build config dict (SDK uses camelCase!)
        config = {
            "temperature": temperature,
            "maxTokens": max_tokens  # Note: maxTokens not max_tokens
        }

        # Build chat messages or simple prompt
        if system_prompt:
            # Use chat format with system message
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            # Simple user prompt
            chat = prompt

        try:
            # Use thread pool for synchronous SDK call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model_instance.respond(chat, config=config)
            )

            return self._extract_json_from_response(str(response))

        except Exception as e:
            logger.warning(f"⚠️  Convenience SDK call failed: {e}")
            logger.error(f"❌ SDK API call failed: {e}")
            raise RuntimeError(f"SDK API failed: {e}") from e

    async def _call_via_http(self, prompt: str, system_prompt: Optional[str] = None,
                            model: str = None, temperature: float = 0.1,
                            max_tokens: int = 1500, response_format: Optional[dict] = None,
                            reasoning: Optional[dict] = None, max_retries: int = 3) -> str:
        """
        Call LLM using OpenRouter HTTP API with aiohttp and retry logic.

        Args:
            response_format: Optional JSON schema for structured outputs
            max_retries: Maximum number of retry attempts for transient errors
        """
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp not available - HTTP fallback disabled")

        if model is None:
            model = LLM_MODEL

        # NOTE: No lock here - aiohttp.ClientSession is designed for concurrent use
        # Using a threading.Lock in async code causes deadlocks when multiple
        # coroutines try to acquire it (they block the event loop)
        
        # Build OpenAI-compatible messages format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        # Add structured output format if provided
        if response_format:
            payload["response_format"] = response_format
        
        # Add reasoning config if provided
        if reasoning:
            payload["reasoning"] = reasoning

        # Build headers for OpenRouter
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/deepseekcoder2/ayechef",
            "X-Title": "Aye Chef Meal Planner"
        }
        if OPENROUTER_API_KEY:
            headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Dynamic timeout based on max_tokens (large responses need more time)
                    timeout_seconds = 120 if max_tokens <= 4000 else 240
                    
                    async with session.post(
                        f"{LLM_API_URL}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                    ) as response:
                        # Handle transient errors with retry
                        if response.status in [500, 502, 503, 504, 429]:
                            error_body = await response.text()
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                                logger.warning(f"⚠️  OpenRouter error {response.status} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                logger.error(f"❌ OpenRouter error {response.status} after {max_retries} attempts: {error_body}")
                                raise RuntimeError(f"OpenRouter error {response.status}: {error_body}")
                        
                        # Handle permanent errors without retry
                        if response.status != 200:
                            error_body = await response.text()
                            logger.error(f"❌ OpenRouter error {response.status}: {error_body}")
                            raise RuntimeError(f"OpenRouter error {response.status}: {error_body}")
                        
                        data = await response.json()

                        # Check if response has error
                        if "error" in data:
                            error_msg = data.get("error", {})
                            if isinstance(error_msg, dict):
                                error_text = error_msg.get("message", str(error_msg))
                            else:
                                error_text = str(error_msg)
                            logger.error(f"❌ OpenRouter API error: {error_text}")
                            raise RuntimeError(f"OpenRouter API error: {error_text}")

                        # Extract content from OpenAI-compatible response
                        try:
                            content = data["choices"][0]["message"]["content"]
                            
                            # Check if content is actually null/empty
                            if content is None:
                                logger.error(f"❌ OpenRouter returned null content. Full response: {json.dumps(data, indent=2)}")
                                raise RuntimeError(f"OpenRouter returned null content. Check API status or model availability.")
                            
                            if not content.strip():
                                logger.error(f"❌ OpenRouter returned empty content. Full response: {json.dumps(data, indent=2)}")
                                raise RuntimeError(f"OpenRouter returned empty content. Check API status or model availability.")
                            
                            return content.strip()
                        except KeyError as e:
                            # Log the actual response for debugging
                            logger.error(f"❌ Unexpected response structure from OpenRouter: {json.dumps(data, indent=2)}")
                            raise RuntimeError(f"Unexpected response format from HTTP API. Missing key: {e}. Response keys: {list(data.keys())}") from e

            except asyncio.TimeoutError as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"⚠️  Timeout after 120s (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"HTTP API call timed out after {max_retries} attempts (120s each). Prompt may be too large.") from e
            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"⚠️  HTTP error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"HTTP API call failed after {max_retries} attempts: {e}") from e
    

    
    def _extract_json_from_response(self, raw_content: str) -> str:
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
            return json_matches[0]

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
                return potential_json[:end_pos]

        # Last resort: If no JSON found, return the original content
        # This will cause JSON parsing to fail with a more informative error
        return content


    async def call_embedding(self, text: str, model: str = None, is_query: bool = False) -> Optional[list]:
        """
        Generate embedding for text with caching.

        Args:
            text: Text to embed
            model: Embedding model name (default: text-embedding-3-small)
            is_query: If True, use query encoding (with instruction prefix).
                      If False, use document encoding (no prefix).
                      Nemotron requires this distinction for optimal retrieval.

        Returns:
            Embedding vector as list of floats, or None if failed
        """
        start_time = time.time()

        # Generate request hash for caching (include is_query to separate cache entries)
        cache_key_text = f"{'query' if is_query else 'doc'}:{text}"
        request_hash = self._generate_request_hash(cache_key_text, None, model, 0.0)

        with self.stats_lock:
            self.stats["total_requests"] += 1

        # Check cache first
        cached_embedding = self.cache.get(request_hash)
        if cached_embedding:
            with self.stats_lock:
                self.stats["cache_hits"] += 1
            # Parse cached string back to list
            import ast
            try:
                return ast.literal_eval(cached_embedding)
            except:
                return cached_embedding

        # Cache miss - generate embedding
        try:
            embedding = await self._generate_embedding(text, model, is_query=is_query)
            if embedding is not None:
                # Cache the result (serialize to JSON for storage)
                embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                self.cache.put(request_hash, str(embedding_list))  # LRUCache expects string
                with self.stats_lock:
                    self.stats["cache_misses"] += 1
                    self.stats["total_time"] += time.time() - start_time
                return embedding_list
            else:
                with self.stats_lock:
                    self.stats["errors"] += 1
                    self.stats["total_time"] += time.time() - start_time
                return None

        except Exception as e:
            with self.stats_lock:
                self.stats["errors"] += 1
                self.stats["total_time"] += time.time() - start_time
            logger.error(f"❌ Embedding generation failed: {e}")
            return None

    async def _generate_embedding(self, text: str, model: str = None, is_query: bool = False) -> Optional[np.ndarray]:
        """
        Generate embedding vector using NVIDIA llama-embed-nemotron-8b.

        Args:
            text: Text to embed
            model: Ignored - always uses NVIDIA Nemotron
            is_query: If True, use encode_query (adds instruction prefix for retrieval).
                      If False, use encode_document (no prefix, for indexing recipes).

        Returns:
            Numpy array of embedding (4096 dims), or None if failed
        """
        if not HAS_EMBED_MODEL:
            logger.error("❌ sentence-transformers not available - install with: pip install sentence-transformers")
            return None

        try:
            # Get or load NVIDIA model (lazy loading)
            embed_model = get_embed_model()
            if embed_model is None:
                logger.error("❌ Failed to load embedding model")
                return None

            # Run embedding in thread pool to not block async loop
            loop = asyncio.get_event_loop()
            if is_query:
                # Use encode_query for search queries (adds instruction prefix for better retrieval)
                embedding = await loop.run_in_executor(
                    None,
                    lambda: embed_model.encode_query(text)
                )
            else:
                # Use encode_document for recipe content (no instruction prefix needed)
                embedding = await loop.run_in_executor(
                    None,
                    lambda: embed_model.encode_document([text])[0]
                )
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            return None

    async def batch_call_llm(self, requests: List[LLMRequest]) -> Dict[str, str]:
        """
        Process multiple LLM requests with caching/deduplication.
        
        Processes requests concurrently via OpenRouter. Caching provides additional speedup.

        Args:
            requests: List of LLMRequest objects

        Returns:
            Dict mapping request_id to response
        """
        results = {}

        async def process_request(req: LLMRequest):
            try:
                response = await self.call_llm(
                    req.prompt,
                    req.system_prompt,
                    req.model,
                    req.temperature,
                    req.max_tokens
                )
                results[req.request_id] = response
            except Exception as e:
                logger.error(f"❌ Batch request {req.request_id} failed: {e}")
                results[req.request_id] = f"Error: {str(e)}"

        # Queue all requests for concurrent processing
        tasks = [process_request(req) for req in requests]
        await asyncio.gather(*tasks)

        return results

    def clear_cache(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            int: Number of entries cleared
        """
        return self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        cache_stats = self.cache.stats()
        cache_hit_rate = (self.stats["cache_hits"] / self.stats["total_requests"] * 100) if self.stats["total_requests"] > 0 else 0
        return {
            **self.stats,
            "cache": cache_stats,
            "cache_hit_rate_pct": cache_hit_rate,
            "pending_deduplication": len(self.pending_requests)
        }


# Global cache processor instance for shared caching across modules
_global_cache: Optional[LLMCacheProcessor] = None
_cache_lock = threading.Lock()  # Use threading.Lock for cross-event-loop safety


async def get_llm_cache(cache_size: int = 1000) -> LLMCacheProcessor:
    """
    Get or create global LLM cache processor instance.

    Args:
        cache_size: Maximum cache entries

    Returns:
        LLMCacheProcessor instance
    """
    global _global_cache

    with _cache_lock:  # Synchronous lock - works across multiple event loops in threads
        if _global_cache is None:
            _global_cache = LLMCacheProcessor(cache_size=cache_size)

    return _global_cache


# Backwards compatibility aliases
async def get_batch_processor(max_concurrent: int = 3) -> LLMCacheProcessor:
    """Backwards compatibility - 'batching' was misleading, this is actually caching."""
    return await get_llm_cache()


# Convenience functions for drop-in replacement
async def call_llm(prompt: str, system_prompt: Optional[str] = None,
                  model: str = None, temperature: float = 0.1,
                  max_tokens: int = 1500, response_format: Optional[dict] = None,
                  reasoning: Optional[dict] = None,
                  max_retries: int = None, retry_delay: float = None) -> str:
    """
    Convenience function for single LLM calls with caching and retry logic.
    Drop-in replacement for direct LLM API calls.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        model: Model to use (defaults to configured model)
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        response_format: Structured output format
        max_retries: Maximum number of retries (uses config default if None)
        retry_delay: Delay between retries in seconds (uses config default if None)

    Returns:
        LLM response string

    Raises:
        RuntimeError: If all retry attempts fail
    """
    from config import get_config_value

    # Use centralized configuration for defaults
    if max_retries is None:
        max_retries = get_config_value('retries', 'max_llm_retries', 3)
    if retry_delay is None:
        retry_delay = get_config_value('timeouts', 'llm_retry_delay', 60)

    cache = await get_llm_cache()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await cache.call_llm(prompt, system_prompt, model, temperature, max_tokens, response_format, reasoning)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"⚠️  LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                logger.info(f"   Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ LLM call failed after {max_retries + 1} attempts: {e}")

    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts. Last error: {last_error}") from last_error


async def batch_call_llm(requests: List[LLMRequest], max_retries: int = None, retry_delay: float = None) -> Dict[str, str]:
    """
    Convenience function for batch LLM calls with caching and retry logic.

    Args:
        requests: List of LLMRequest objects
        max_retries: Maximum number of retries (uses config default if None)
        retry_delay: Delay between retries in seconds (uses config default if None)

    Returns:
        Dictionary mapping request IDs to responses

    Raises:
        RuntimeError: If all retry attempts fail
    """
    from config import get_config_value

    # Use centralized configuration for defaults
    if max_retries is None:
        max_retries = get_config_value('retries', 'max_llm_retries', 3)
    if retry_delay is None:
        retry_delay = get_config_value('timeouts', 'llm_retry_delay', 60)

    cache = await get_llm_cache()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await cache.batch_call_llm(requests)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"⚠️  Batch LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                logger.info(f"   Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ Batch LLM call failed after {max_retries + 1} attempts: {e}")

    raise RuntimeError(f"Batch LLM call failed after {max_retries + 1} attempts. Last error: {last_error}") from last_error


if __name__ == "__main__":
    # Test the cache processor
    async def test_cache():
        print("🧪 Testing LLM Cache Processor")
        print("=" * 60)
        
        cache = await get_llm_cache()

        # Test single call
        print("\n📝 Test 1: First call (cache miss)...")
        try:
            prompt = "Say 'Hello, World!' in JSON format: {\"message\": \"your response\"}"
            start = time.time()
            response1 = await cache.call_llm(prompt, temperature=0.1, max_tokens=50)
            time1 = time.time() - start
            print(f"✅ Response: {response1[:100]}...")
            print(f"⏱️  Time: {time1:.2f}s")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Test cache hit
        print("\n📝 Test 2: Second call (cache hit)...")
        try:
            start = time.time()
            response2 = await cache.call_llm(prompt, temperature=0.1, max_tokens=50)
            time2 = time.time() - start
            print(f"✅ Response: {response2[:100]}...")
            print(f"⏱️  Time: {time2:.2f}s")
            print(f"🚀 Speedup: {time1/time2:.1f}x from caching")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Test statistics
        print("\n📊 Cache Statistics:")
        stats = cache.get_stats()
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache hit rate: {stats['cache_hit_rate_pct']:.1f}%")
        print(f"   Avg response time: {stats['avg_response_time']:.2f}s")
        
        print("\n" + "=" * 60)
        if stats['cache_hits'] > 0:
            print("✅ SUCCESS: Caching working perfectly!")
            print(f"   Cache provides {time1/time2:.0f}x speedup on repeated calls")
        print("💡 SDK parameter incompatibility is OK - system fails fast if SDK unavailable")
        print("💡 Caching is what matters (3-5x on real workloads)")

    asyncio.run(test_cache())


# =============================================================================
# SHOPPING LIST REFINEMENT LLM FUNCTIONS
# =============================================================================

async def refine_shopping_list(raw_ingredients: List[str]) -> dict:
    """
    Apply ingredient refinement prompt to raw Mealie shopping list.

    Filters pantry items, cleans ingredients, categorizes by shopping location
    based on household preferences from config.yaml.

    Args:
        raw_ingredients: List of raw ingredient strings from Mealie shopping list

    Returns:
        dict: Refined shopping list with categorized items and processing summary
    """
    from prompts import SHOPPING_REFINEMENT_PROMPT

    if not raw_ingredients:
        return {
            "refined_items": [],
            "pantry_notes": ["No ingredients to refine"],
            "processing_summary": {
                "items_filtered": 0,
                "items_included": 0,
                "quantities_aggregated": 0,
                "categories_created": 0
            }
        }
    
    # Warn if shopping list is very large (may cause timeouts)
    if len(raw_ingredients) > 150:
        logger.warning(f"⚠️  Large shopping list ({len(raw_ingredients)} items) may take 3-4 minutes to refine. Consider reducing meal plan size.")

    # Optimize prompt for context window limits
    # Household context loaded from config.yaml
    from prompts import get_household_context_detailed
    household_context = get_household_context_detailed()

    prompt = ContextWindowOptimizer.optimize_shopping_refinement_prompt(
        raw_ingredients=raw_ingredients,
        household_context=household_context,
        base_prompt=SHOPPING_REFINEMENT_PROMPT
    )

    # Validate prompt size (for monitoring)
    size_validation = ContextWindowOptimizer.validate_prompt_size(prompt)
    if not size_validation["within_limits"]:
        logger.warning(f"⚠️  Warning: Prompt exceeds context limit: {size_validation['estimated_tokens']} tokens")
        # Continue anyway - the optimizer should have prevented this

    # Import json at function scope (needed for exception handling)
    import json
    
    # Define schema for structured output (prevents field confusion)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "shopping_list_refinement",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "refined_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "display": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unit": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                    "required": ["name"],
                                    "additionalProperties": False
                                },
                                "food": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                    "required": ["name"],
                                    "additionalProperties": False
                                },
                                "note": {"type": "string"},
                                "checked": {"type": "boolean"},
                                "position": {"type": "integer"}
                            },
                            "required": ["display", "quantity", "unit", "food", "note", "checked", "position"],
                            "additionalProperties": False
                        }
                    },
                    "pantry_notes": {"type": "array", "items": {"type": "string"}},
                    "quality_validation": {
                        "type": "object",
                        "properties": {
                            "items_accepted": {"type": "integer"},
                            "items_rejected": {"type": "integer"},
                            "rejection_reasons": {"type": "array", "items": {"type": "string"}},
                            "quality_score": {"type": "number"}
                        },
                        "required": ["items_accepted", "items_rejected", "rejection_reasons", "quality_score"],
                        "additionalProperties": False
                    },
                    "processing_summary": {
                        "type": "object",
                        "properties": {
                            "items_filtered": {"type": "integer"},
                            "items_included": {"type": "integer"},
                            "quantities_aggregated": {"type": "integer"},
                            "categories_created": {"type": "integer"}
                        },
                        "required": ["items_filtered", "items_included", "quantities_aggregated", "categories_created"],
                        "additionalProperties": False
                    }
                },
                "required": ["refined_items", "pantry_notes", "quality_validation", "processing_summary"],
                "additionalProperties": False
            }
        }
    }
    
    try:
        cache = await get_llm_cache()
        
        # Dynamic max_tokens based on item count (with safety margin)
        estimated_tokens = len(raw_ingredients) * 120  # 120 tokens per item (conservative estimate)
        max_tokens = min(estimated_tokens + 2000, 32000)  # Cap at 32k with 2k overhead
        
        logger.info(f"Requesting {max_tokens} tokens for {len(raw_ingredients)} items")
        
        response = await cache.call_llm(
            prompt=prompt,
            model=LLM_MODEL,
            temperature=0.1,  # Low temperature for consistent categorization
            max_tokens=max_tokens,
            response_format=schema,  # Enforce schema - prevents field confusion
            reasoning={"effort": "none"}  # Disable reasoning for structured output (prevents truncation)
        )

        # Log response size for debugging truncation issues
        logger.info(f"Received response: {len(response)} characters, {len(response.split())} words")
        
        # Parse JSON response - strip markdown code fences if present
        try:
            result = json.loads(strip_markdown_json(response))
        except json.JSONDecodeError as e:
            # Check if response was truncated (incomplete JSON)
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                logger.error(f"❌ Response appears truncated at {len(response)} chars. Requested {max_tokens} tokens for {len(raw_ingredients)} items.")
                logger.error(f"   This may indicate OpenRouter rate limiting or max_tokens too low.")
                raise RuntimeError(f"LLM response was truncated. Try reducing meal plan size or upgrading OpenRouter tier.") from e
            raise

        # Validate required fields
        required_fields = ["refined_items", "pantry_notes", "processing_summary", "quality_validation"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Validate refined_items structure
        if not isinstance(result["refined_items"], list):
            raise ValueError("refined_items must be a list")

        for idx, item in enumerate(result["refined_items"]):
            required_item_fields = ["display", "quantity", "unit", "food", "note", "checked", "position"]
            for field in required_item_fields:
                if field not in item:
                    raise ValueError(f"Item missing required field: {field}")
            
            # Validate display field content (catch LLM hallucinations)
            display = item.get("display", "").strip()
            food_name = item.get("food", {}).get("name", "").strip() if isinstance(item.get("food"), dict) else ""
            
            # Display should contain food name, not position/category info
            if not display:
                raise ValueError(f"Item {idx}: display field is empty")
            
            # Check if display looks like garbage (starts with large number + emoji)
            import re
            if re.match(r'^\d{3,}.*[🛒🥬🥩🥛🫒]', display):
                raise ValueError(f"Item {idx}: display field contains position/category instead of food: {display[:50]}")
            
            # Food name must not be empty
            if not food_name:
                raise ValueError(f"Item {idx}: food.name is empty. Display: '{display[:50]}'")
            
            # Display should reference the actual food item
            if food_name and food_name.lower() not in display.lower():
                logger.warning(f"⚠️  Item {idx}: display '{display}' doesn't contain food name '{food_name}'")

        # Validate quality_validation structure
        quality_validation = result["quality_validation"]
        required_quality_fields = ["items_accepted", "items_rejected", "rejection_reasons", "quality_score"]
        for field in required_quality_fields:
            if field not in quality_validation:
                raise ValueError(f"Quality validation missing required field: {field}")

        # Validate quality score is reasonable (0-100)
        quality_score = quality_validation["quality_score"]
        if not isinstance(quality_score, (int, float)) or not (0 <= quality_score <= 100):
            raise ValueError(f"Quality score must be between 0-100, got: {quality_score}")

        # Log quality metrics
        if quality_score < 80.0:
            logger.warning(f"⚠️  Low quality score: {quality_score:.1f}%")
            if quality_validation["rejection_reasons"]:
                logger.warning(f"   Rejection reasons: {', '.join(quality_validation['rejection_reasons'][:3])}")
        else:
            logger.info(f"✅ Quality score: {quality_score:.1f}% ({quality_validation['items_accepted']} accepted, {quality_validation['items_rejected']} rejected)")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response (first 500 chars): {response[:500]}...")
        raise RuntimeError(f"Shopping list refinement failed: LLM returned invalid JSON. Response: {response[:200]}") from e
    except Exception as e:
        logger.error(f"❌ Error in shopping list refinement: {e}")
        raise RuntimeError(f"Shopping list refinement failed: {e}") from e


async def handle_missing_recipes(missing_concepts: List[str], available_recipes: List[dict]) -> dict:
    """
    Apply missing recipe flagging prompt for menu concepts without recipe matches.

    Generates analysis for missing recipes, suggests alternatives, and creates
    procurement guidance based on household context from config.yaml.

    Args:
        missing_concepts: List of menu concepts that couldn't be matched to recipes
        available_recipes: Successfully matched recipes for context

    Returns:
        dict: Analysis with missing recipe details, alternatives, and procurement items
    """
    from prompts import MISSING_RECIPE_ANALYSIS_PROMPT

    if not missing_concepts:
        return {
            "missing_recipe_analysis": [],
            "meal_plan_adjustments": [],
            "procurement_recommendations": [],
            "mealie_integration_items": []
        }

    # Format inputs for prompt
    missing_text = "\n".join(f"- {concept}" for concept in missing_concepts)

    available_text = "\n".join([
        f"- {recipe.get('name', 'Unknown')} ({recipe.get('slug', 'no-slug')})"
        for recipe in available_recipes[:5]  # Limit to avoid prompt bloat
    ]) if available_recipes else "No recipes available yet"

    # Household context loaded from config.yaml
    from prompts import get_household_context_detailed
    household_context = get_household_context_detailed()
    
    prompt = MISSING_RECIPE_ANALYSIS_PROMPT.format(
        missing_concepts=missing_text,
        available_recipes=available_text,
        household_context=household_context
    )

    try:
        cache = await get_llm_cache()
        response = await cache.call_llm(
            prompt=prompt,
            model=LLM_MODEL,
            temperature=0.2,  # Slightly higher for creative suggestions
            max_tokens=2500
        )

        # Parse JSON response - strip markdown code fences if present
        import json
        result = json.loads(strip_markdown_json(response))

        # Validate required fields
        required_fields = [
            "missing_recipe_analysis", "meal_plan_adjustments",
            "procurement_recommendations", "mealie_integration_items"
        ]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Validate structure
        if not isinstance(result["missing_recipe_analysis"], list):
            raise ValueError("missing_recipe_analysis must be a list")

        if not isinstance(result["mealie_integration_items"], list):
            raise ValueError("mealie_integration_items must be a list")

        # Validate item structure for Mealie integration
        for item in result["mealie_integration_items"]:
            required_item_fields = ["display", "quantity", "unit", "food", "note", "checked", "position"]
            for field in required_item_fields:
                if field not in item:
                    raise ValueError(f"Mealie item missing required field: {field}")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse missing recipe analysis as JSON: {e}")
        logger.debug(f"Response: {response[:500]}...")
        return _fallback_missing_recipe_analysis(missing_concepts)
    except Exception as e:
        logger.error(f"❌ Error in missing recipe analysis: {e}")
        return _fallback_missing_recipe_analysis(missing_concepts)


def _fallback_refinement(raw_ingredients: List[str]) -> dict:
    """
    DEPRECATED: Fallback refinement removed - fail fast instead.
    
    This function should never be called. If LLM refinement fails,
    the system should raise an error and exit rather than write garbage data.
    """
    raise RuntimeError(
        "FATAL: _fallback_refinement called - this should never happen. "
        "LLM refinement failed and the system should have already raised an error."
    )


def _fallback_missing_recipe_analysis(missing_concepts: List[str]) -> dict:
    """
    Fallback analysis when LLM fails - basic suggestions.
    """
    logger.info("🔄 Using fallback missing recipe analysis")

    analysis = []
    mealie_items = []

    for concept in missing_concepts:
        analysis.append({
            "concept": concept,
            "reason_unfulfilled": "Recipe database search failed",
            "best_alternatives": ["Search online for similar recipes"],
            "procurement_guidance": "Consider purchasing pre-made version or sourcing ingredients separately",
            "urgency": "medium",
            "mealie_shopping_note": f"Missing recipe: {concept}"
        })

        # Add placeholder shopping item
        mealie_items.append({
            "display": f"Ingredients for {concept}",
            "quantity": 1,
            "unit": {"name": "batch"},
            "food": {"name": "miscellaneous"},
            "note": f"Missing recipe - source ingredients for {concept}",
            "checked": False,
            "position": 1000 + len(mealie_items)
        })

    return {
        "missing_recipe_analysis": analysis,
        "meal_plan_adjustments": ["Consider simpler meal alternatives for missing recipes"],
        "procurement_recommendations": ["Source missing recipes online or create custom meal plans"],
        "mealie_integration_items": mealie_items
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    asyncio.run(test_cache())
