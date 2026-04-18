"""
Performance Optimization & Caching Layer
Simple in-memory cache with TTL for predictions, analytics, and computations
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable
import hashlib
import json
from backend.utils.logger import ai_logger as logger


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 300
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    def update_hit_count(self):
        """Increment cache hit count"""
        self.hit_count += 1


class CacheManager:
    def __init__(self, default_ttl_seconds: int = 300):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl_seconds
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        self.stats["total_requests"] += 1
        
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        if entry.is_expired():
            del self.cache[key]
            self.stats["misses"] += 1
            return None
        
        entry.update_hit_count()
        self.stats["hits"] += 1
        logger.debug(f"Cache hit: {key}")
        
        return entry.value
    
    async def set(self, key: str, value: Any, ttl_seconds: int = None):
        """Store value in cache with TTL"""
        ttl = ttl_seconds or self.default_ttl
        
        entry = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl
        )
        
        self.cache[key] = entry
        logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Cache delete: {key}")
            return True
        return False
    
    async def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    async def cleanup_expired(self):
        """Remove expired entries from cache"""
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from prefix and kwargs"""
        # Sort kwargs to ensure consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        key_str = f"{prefix}:{json.dumps(sorted_kwargs)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get_or_compute(self, key: str, compute_func: Callable,
                            ttl_seconds: int = None) -> Any:
        """Get from cache or compute and cache value"""
        
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        value = await compute_func()
        await self.set(key, value, ttl_seconds)
        
        return value
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all cache keys matching pattern"""
        matching_keys = [k for k in self.cache.keys() if pattern in k]
        
        for key in matching_keys:
            await self.delete(key)
        
        return len(matching_keys)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.stats["total_requests"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            "total_requests": total,
            "cache_hits": self.stats["hits"],
            "cache_misses": self.stats["misses"],
            "hit_rate_percent": round(hit_rate, 2),
            "entries_in_cache": len(self.cache),
            "total_cache_size_mb": self._estimate_cache_size() / (1024 * 1024)
        }
    
    def _estimate_cache_size(self) -> int:
        """Estimate total cache size in bytes"""
        total_size = 0
        for entry in self.cache.values():
            if isinstance(entry.value, (dict, list)):
                total_size += len(json.dumps(entry.value))
            elif isinstance(entry.value, str):
                total_size += len(entry.value)
            else:
                total_size += 100  # Rough estimate
        return total_size


class QueryOptimizer:
    """Database query optimization with result caching and pagination"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.default_page_size = 100
    
    async def get_paginated_results(self, query_id: str, total_items: int,
                                    page: int = 1, page_size: int = None,
                                    fetch_func: Callable = None) -> Dict:
        """Get paginated results with caching"""
        
        page_size = page_size or self.default_page_size
        cache_key = f"paginated:{query_id}:page_{page}:size_{page_size}"
        
        # Try cache first
        cached_page = await self.cache.get(cache_key)
        if cached_page:
            return cached_page
        
        # Compute results
        if fetch_func:
            items = await fetch_func(page, page_size)
        else:
            items = []
        
        total_pages = (total_items + page_size - 1) // page_size
        
        result = {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "items": items,
            "has_next": page < total_pages,
            "has_previous": page > 1
        }
        
        # Cache page results (5 minute TTL)
        await self.cache.set(cache_key, result, ttl_seconds=300)
        
        return result


class ComputationCache:
    """Specialized cache for expensive computations"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    async def cache_prediction(self, machine_id: str, prediction: Dict,
                              ttl_seconds: int = 600):
        """Cache prediction results"""
        key = f"prediction:{machine_id}"
        await self.cache.set(key, prediction, ttl_seconds=ttl_seconds)
    
    async def get_prediction(self, machine_id: str) -> Optional[Dict]:
        """Get cached prediction"""
        key = f"prediction:{machine_id}"
        return await self.cache.get(key)
    
    async def cache_analytics(self, machine_id: str, analytics_type: str,
                             data: Dict, ttl_seconds: int = 900):
        """Cache analytics computation"""
        key = f"analytics:{machine_id}:{analytics_type}"
        await self.cache.set(key, data, ttl_seconds=ttl_seconds)
    
    async def get_analytics(self, machine_id: str, analytics_type: str) -> Optional[Dict]:
        """Get cached analytics"""
        key = f"analytics:{machine_id}:{analytics_type}"
        return await self.cache.get(key)
    
    async def invalidate_machine_cache(self, machine_id: str):
        """Invalidate all cache for a machine"""
        count = await self.cache.invalidate_pattern(f":{machine_id}:")
        logger.info(f"Invalidated {count} cache entries for {machine_id}")
        return count
