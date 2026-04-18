"""
API Routes for Performance Optimization & Cache Management
"""

from fastapi import APIRouter, Depends, HTTPException
from backend.ai_engine.cache_manager import CacheManager, ComputationCache
from backend.api.middleware.auth import require_admin

router = APIRouter()

# Global cache manager instance
cache_manager: CacheManager = None


def get_cache_manager() -> CacheManager:
    """Get or create cache manager instance"""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
    return cache_manager


@router.get("/cache/stats", tags=["Performance"])
async def get_cache_statistics(
    cache: CacheManager = Depends(get_cache_manager),
    user = Depends(require_admin)
):
    """Get cache performance statistics"""
    try:
        stats = cache.get_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear", tags=["Performance"])
async def clear_cache(
    cache: CacheManager = Depends(get_cache_manager),
    user = Depends(require_admin)
):
    """Clear all cache entries"""
    try:
        await cache.clear()
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/cleanup", tags=["Performance"])
async def cleanup_expired_cache(
    cache: CacheManager = Depends(get_cache_manager),
    user = Depends(require_admin)
):
    """Remove expired cache entries"""
    try:
        removed_count = await cache.cleanup_expired()
        return {
            "status": "success",
            "message": f"Removed {removed_count} expired entries"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/metrics", tags=["Performance"])
async def get_performance_metrics(
    cache: CacheManager = Depends(get_cache_manager),
    user = Depends(require_admin)
):
    """Get system performance metrics"""
    try:
        stats = cache.get_stats()
        
        return {
            "status": "success",
            "data": {
                "cache_statistics": stats,
                "optimization_score": min(100, stats["hit_rate_percent"] * 1.5),
                "api_performance": "optimal" if stats["hit_rate_percent"] > 50 else "good" if stats["hit_rate_percent"] > 25 else "needs_improvement"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
