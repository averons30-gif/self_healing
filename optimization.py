# Digital Twin AI System — Database Optimization & Caching

"""
Advanced database optimization and caching layer for improved performance.
Includes connection pooling, query optimization, and Redis caching.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict

import aiosqlite
from backend.utils.logger import app_logger


class QueryOptimizer:
    """Optimizes database queries and provides performance insights."""

    def __init__(self):
        self.query_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0,
            "avg_time": 0,
            "slow_queries": []
        })
        self.slow_query_threshold = 0.1  # 100ms

    def track_query(self, query: str, execution_time: float):
        """Track query performance."""
        stats = self.query_stats[query]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["count"]

        if execution_time > self.slow_query_threshold:
            stats["slow_queries"].append({
                "time": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            })
            # Keep only last 10 slow queries
            stats["slow_queries"] = stats["slow_queries"][-10:]

    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        return dict(self.query_stats)

    def get_slowest_queries(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get the slowest queries."""
        query_avgs = [(query, stats["avg_time"]) for query, stats in self.query_stats.items()]
        return sorted(query_avgs, key=lambda x: x[1], reverse=True)[:limit]


class ConnectionPool:
    """Connection pool for SQLite database with performance optimizations."""

    def __init__(self, database_path: str, pool_size: int = 10):
        self.database_path = database_path
        self.pool_size = pool_size
        self.connections = asyncio.Queue(maxsize=pool_size)
        self._initialized = False

    async def initialize(self):
        """Initialize the connection pool."""
        if self._initialized:
            return

        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.database_path)
            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            await conn.execute("PRAGMA foreign_keys=ON")
            # Optimize for performance
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA temp_store=MEMORY")

            await self.connections.put(conn)

        self._initialized = True
        app_logger.info(f"Database connection pool initialized with {self.pool_size} connections")

    async def get_connection(self) -> aiosqlite.Connection:
        """Get a connection from the pool."""
        if not self._initialized:
            await self.initialize()

        return await self.connections.get()

    async def return_connection(self, conn: aiosqlite.Connection):
        """Return a connection to the pool."""
        if conn:
            await self.connections.put(conn)

    async def close_all(self):
        """Close all connections in the pool."""
        while not self.connections.empty():
            conn = await self.connections.get()
            await conn.close()


class CacheManager:
    """Redis-based caching layer for frequently accessed data."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url
        self.cache = {}  # Fallback in-memory cache
        self.ttl_cache = {}  # TTL tracking
        self.use_redis = redis_url is not None

        if self.use_redis:
            try:
                import redis.asyncio as redis
                self.redis_client = redis.from_url(redis_url)
            except ImportError:
                app_logger.warning("Redis not available, using in-memory cache")
                self.use_redis = False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.use_redis:
            try:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                app_logger.error(f"Redis get error: {e}")

        # Fallback to memory cache
        if key in self.cache:
            if time.time() < self.ttl_cache.get(key, 0):
                return self.cache[key]
            else:
                # Expired, remove
                del self.cache[key]
                del self.ttl_cache[key]

        return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set value in cache with TTL."""
        serialized_value = json.dumps(value)

        if self.use_redis:
            try:
                await self.redis_client.setex(key, ttl_seconds, serialized_value)
                return
            except Exception as e:
                app_logger.error(f"Redis set error: {e}")

        # Fallback to memory cache
        self.cache[key] = value
        self.ttl_cache[key] = time.time() + ttl_seconds

    async def delete(self, key: str):
        """Delete value from cache."""
        if self.use_redis:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                app_logger.error(f"Redis delete error: {e}")

        # Clean memory cache
        self.cache.pop(key, None)
        self.ttl_cache.pop(key, None)

    async def clear(self):
        """Clear all cache."""
        if self.use_redis:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                app_logger.error(f"Redis clear error: {e}")

        self.cache.clear()
        self.ttl_cache.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache_size": len(self.cache),
            "using_redis": self.use_redis
        }

        if self.use_redis:
            try:
                info = await self.redis_client.info()
                stats.update({
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "redis_used_memory": info.get("used_memory_human", "unknown"),
                    "redis_total_keys": await self.redis_client.dbsize()
                })
            except Exception as e:
                stats["redis_error"] = str(e)

        return stats


class OptimizedDatabaseManager:
    """Enhanced database manager with connection pooling and caching."""

    def __init__(self, database_path: str, cache_manager: CacheManager = None):
        self.database_path = database_path
        self.connection_pool = ConnectionPool(database_path)
        self.cache = cache_manager or CacheManager()
        self.query_optimizer = QueryOptimizer()
        self._initialized = False

    async def initialize(self):
        """Initialize database and connection pool."""
        if self._initialized:
            return

        await self.connection_pool.initialize()

        # Create optimized indexes
        await self._create_indexes()

        self._initialized = True
        app_logger.info("Optimized database manager initialized")

    async def _create_indexes(self):
        """Create performance indexes."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sensor_readings_machine_timestamp ON sensor_readings(machine_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp ON sensor_readings(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_machine_timestamp ON alerts(machine_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp)"
        ]

        conn = await self.connection_pool.get_connection()
        try:
            for index_sql in indexes:
                await conn.execute(index_sql)
            await conn.commit()
            app_logger.info("Database indexes created/verified")
        finally:
            await self.connection_pool.return_connection(conn)

    async def execute_query(self, query: str, params: tuple = None, cache_key: str = None,
                          cache_ttl: int = 300) -> List[Dict]:
        """
        Execute a SELECT query with optional caching.

        Args:
            query: SQL query string
            params: Query parameters
            cache_key: Cache key for result caching
            cache_ttl: Cache TTL in seconds

        Returns:
            List of result dictionaries
        """
        # Check cache first
        if cache_key:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        start_time = time.time()
        conn = await self.connection_pool.get_connection()

        try:
            cursor = await conn.execute(query, params or ())
            columns = [desc[0] for desc in cursor.description or []]
            rows = await cursor.fetchall()

            # Convert to dictionaries
            results = [dict(zip(columns, row)) for row in rows]

            # Track query performance
            execution_time = time.time() - start_time
            self.query_optimizer.track_query(query, execution_time)

            # Cache result if cache_key provided
            if cache_key:
                await self.cache.set(cache_key, results, cache_ttl)

            return results

        finally:
            await self.connection_pool.return_connection(conn)

    async def execute_write(self, query: str, params: tuple = None,
                          invalidate_cache: List[str] = None) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query.

        Args:
            query: SQL query string
            params: Query parameters
            invalidate_cache: List of cache keys to invalidate

        Returns:
            Number of affected rows
        """
        conn = await self.connection_pool.get_connection()

        try:
            cursor = await conn.execute(query, params or ())
            await conn.commit()

            affected_rows = cursor.rowcount

            # Invalidate cache if specified
            if invalidate_cache:
                for key in invalidate_cache:
                    await self.cache.delete(key)

            return affected_rows

        finally:
            await self.connection_pool.return_connection(conn)

    async def get_machine_data_cached(self, machine_id: str, hours: int = 24) -> List[Dict]:
        """Get machine sensor data with caching."""
        cache_key = f"machine_data:{machine_id}:{hours}h"
        query = """
            SELECT * FROM sensor_readings
            WHERE machine_id = ? AND timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        """.format(hours)

        return await self.execute_query(query, (machine_id,), cache_key, 300)  # 5 min cache

    async def get_alerts_cached(self, machine_id: str = None, status: str = None) -> List[Dict]:
        """Get alerts with caching."""
        cache_key = f"alerts:{machine_id or 'all'}:{status or 'all'}"

        query = "SELECT * FROM alerts WHERE 1=1"
        params = []

        if machine_id:
            query += " AND machine_id = ?"
            params.append(machine_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC LIMIT 100"

        return await self.execute_query(query, tuple(params), cache_key, 60)  # 1 min cache

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        return {
            "query_stats": self.query_optimizer.get_query_stats(),
            "slowest_queries": self.query_optimizer.get_slowest_queries(),
            "cache_stats": await self.cache.get_stats(),
            "connection_pool_size": self.connection_pool.pool_size
        }

    async def cleanup_expired_data(self, days_to_keep: int = 90):
        """Clean up old data to maintain performance."""
        cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()

        # Archive old sensor readings (keep last 90 days)
        await self.execute_write(
            "DELETE FROM sensor_readings WHERE timestamp < ?",
            (cutoff_date,),
            ["machine_data:*"]  # Invalidate all machine data cache
        )

        # Archive old resolved alerts (keep last 30 days for resolved)
        await self.execute_write(
            "DELETE FROM alerts WHERE status = 'resolved' AND timestamp < ?",
            ((datetime.utcnow() - timedelta(days=30)).isoformat(),),
            ["alerts:*"]  # Invalidate alerts cache
        )

        app_logger.info(f"Cleaned up data older than {days_to_keep} days")

    async def optimize_database(self):
        """Run database optimization commands."""
        conn = await self.connection_pool.get_connection()

        try:
            # Vacuum database to reclaim space
            await conn.execute("VACUUM")

            # Analyze tables for query optimization
            await conn.execute("ANALYZE")

            # Rebuild indexes
            await conn.execute("REINDEX")

            await conn.commit()
            app_logger.info("Database optimization completed")

        finally:
            await self.connection_pool.return_connection(conn)

    async def close(self):
        """Close database connections."""
        await self.connection_pool.close_all()
        if hasattr(self.cache, 'close'):
            await self.cache.close()


# Global instances
cache_manager = CacheManager()
optimized_db = None


async def init_database_optimization(database_path: str):
    """Initialize the optimized database system."""
    global optimized_db

    optimized_db = OptimizedDatabaseManager(database_path, cache_manager)
    await optimized_db.initialize()

    # Schedule periodic maintenance
    asyncio.create_task(periodic_maintenance())

    app_logger.info("Database optimization system initialized")


async def periodic_maintenance():
    """Run periodic database maintenance tasks."""
    while True:
        try:
            if optimized_db:
                # Clean up old data weekly
                await optimized_db.cleanup_expired_data()

                # Optimize database monthly (every 30 days)
                await optimized_db.optimize_database()

        except Exception as e:
            app_logger.error(f"Database maintenance error: {e}")

        # Run weekly
        await asyncio.sleep(7 * 24 * 60 * 60)  # 7 days