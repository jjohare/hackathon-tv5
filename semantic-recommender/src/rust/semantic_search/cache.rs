// Query Result Caching
// High-performance LRU cache for recommendation queries

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use anyhow::Result;
use tokio::sync::RwLock;

/// Cache entry with TTL
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub created_at: u64,
    pub access_count: u64,
    pub ttl_seconds: u64,
}

impl CacheEntry {
    fn is_expired(&self, current_time: u64) -> bool {
        current_time - self.created_at > self.ttl_seconds
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub default_ttl_seconds: u64,
    pub enable_lru: bool,
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            default_ttl_seconds: 3600,
            enable_lru: true,
            enable_stats: true,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub expirations: u64,
    pub total_size_bytes: u64,
}

impl CacheStats {
    fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

/// Main query cache
pub struct QueryCache {
    config: CacheConfig,
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    stats: Arc<RwLock<CacheStats>>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl QueryCache {
    pub fn new(max_entries: usize, default_ttl_seconds: u64) -> Result<Self> {
        let config = CacheConfig {
            max_entries,
            default_ttl_seconds,
            ..Default::default()
        };

        Ok(Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Get a value from cache
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let current_time = self.current_timestamp();
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.get_mut(key) {
            if entry.is_expired(current_time) {
                // Remove expired entry
                entries.remove(key);
                if self.config.enable_stats {
                    let mut stats = self.stats.write().await;
                    stats.expirations += 1;
                }
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Update access count
            entry.access_count += 1;

            if self.config.enable_stats {
                self.hits.fetch_add(1, Ordering::Relaxed);
                let mut stats = self.stats.write().await;
                stats.hits += 1;
            }

            Some(entry.value.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            if self.config.enable_stats {
                let mut stats = self.stats.write().await;
                stats.misses += 1;
            }
            None
        }
    }

    /// Put a value into cache
    pub async fn put(&self, key: String, value: Vec<u8>) -> Result<()> {
        let current_time = self.current_timestamp();
        let mut entries = self.entries.write().await;

        // Check if we need to evict
        if entries.len() >= self.config.max_entries && !entries.contains_key(&key) {
            self.evict_lru(&mut entries, current_time).await?;
        }

        let entry = CacheEntry {
            key: key.clone(),
            value: value.clone(),
            created_at: current_time,
            access_count: 0,
            ttl_seconds: self.config.default_ttl_seconds,
        };

        let value_size = value.len() as u64;
        entries.insert(key, entry);

        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            stats.total_size_bytes += value_size;
        }

        Ok(())
    }

    /// Put with custom TTL
    pub async fn put_with_ttl(
        &self,
        key: String,
        value: Vec<u8>,
        ttl_seconds: u64,
    ) -> Result<()> {
        let current_time = self.current_timestamp();
        let mut entries = self.entries.write().await;

        if entries.len() >= self.config.max_entries && !entries.contains_key(&key) {
            self.evict_lru(&mut entries, current_time).await?;
        }

        let entry = CacheEntry {
            key: key.clone(),
            value: value.clone(),
            created_at: current_time,
            access_count: 0,
            ttl_seconds,
        };

        let value_size = value.len() as u64;
        entries.insert(key, entry);

        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            stats.total_size_bytes += value_size;
        }

        Ok(())
    }

    /// Remove a key from cache
    pub async fn remove(&self, key: &str) -> Result<()> {
        let mut entries = self.entries.write().await;

        if let Some(entry) = entries.remove(key) {
            if self.config.enable_stats {
                let mut stats = self.stats.write().await;
                stats.total_size_bytes -= entry.value.len() as u64;
            }
        }

        Ok(())
    }

    /// Clear entire cache
    pub async fn clear(&self) -> Result<()> {
        let mut entries = self.entries.write().await;
        entries.clear();

        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            *stats = CacheStats::default();
        }

        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Get current cache size
    pub async fn size(&self) -> usize {
        let entries = self.entries.read().await;
        entries.len()
    }

    /// Get hit rate
    pub async fn hit_rate(&self) -> f32 {
        let stats = self.stats.read().await;
        stats.hit_rate()
    }

    /// Remove expired entries
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let current_time = self.current_timestamp();
        let mut entries = self.entries.write().await;

        let initial_count = entries.len();

        entries.retain(|_, entry| {
            let expired = entry.is_expired(current_time);
            if expired && self.config.enable_stats {
                // Stats will be updated after retain
            }
            !expired
        });

        let removed = initial_count - entries.len();

        if self.config.enable_stats && removed > 0 {
            let mut stats = self.stats.write().await;
            stats.expirations += removed as u64;
        }

        Ok(removed)
    }

    // Private helper methods

    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    async fn evict_lru(
        &self,
        entries: &mut HashMap<String, CacheEntry>,
        current_time: u64,
    ) -> Result<()> {
        if !self.config.enable_lru {
            // Random eviction
            if let Some(key) = entries.keys().next().cloned() {
                entries.remove(&key);
            }
            return Ok(());
        }

        // First try to evict expired entries
        let expired_keys: Vec<String> = entries
            .iter()
            .filter(|(_, entry)| entry.is_expired(current_time))
            .map(|(key, _)| key.clone())
            .collect();

        if !expired_keys.is_empty() {
            for key in expired_keys {
                entries.remove(&key);
            }
            return Ok(());
        }

        // LRU eviction: find entry with lowest access count
        if let Some((lru_key, _)) = entries
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
        {
            let key_to_remove = lru_key.clone();
            entries.remove(&key_to_remove);

            if self.config.enable_stats {
                let mut stats = self.stats.write().await;
                stats.evictions += 1;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_creation() {
        let cache = QueryCache::new(100, 3600);
        assert!(cache.is_ok());
    }

    #[tokio::test]
    async fn test_cache_put_get() {
        let cache = QueryCache::new(100, 3600).unwrap();

        cache.put("key1".to_string(), b"value1".to_vec()).await.unwrap();
        let result = cache.get("key1").await;

        assert!(result.is_some());
        assert_eq!(result.unwrap(), b"value1");
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let cache = QueryCache::new(100, 1).unwrap();

        cache.put("key1".to_string(), b"value1".to_vec()).await.unwrap();

        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        let result = cache.get("key1").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = QueryCache::new(2, 3600).unwrap();

        cache.put("key1".to_string(), b"value1".to_vec()).await.unwrap();
        cache.put("key2".to_string(), b"value2".to_vec()).await.unwrap();
        cache.put("key3".to_string(), b"value3".to_vec()).await.unwrap();

        let size = cache.size().await;
        assert_eq!(size, 2);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = QueryCache::new(100, 3600).unwrap();

        cache.put("key1".to_string(), b"value1".to_vec()).await.unwrap();
        cache.get("key1").await;
        cache.get("key2").await;

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = QueryCache::new(100, 3600).unwrap();

        cache.put("key1".to_string(), b"value1".to_vec()).await.unwrap();
        cache.put("key2".to_string(), b"value2".to_vec()).await.unwrap();

        cache.clear().await.unwrap();

        let size = cache.size().await;
        assert_eq!(size, 0);
    }
}
