//! Result caching for tool output.
//!
//! Large tool results are stored out-of-context and replaced with a compact
//! summary + preview. Agents retrieve slices on demand via a `result_cache`
//! tool, keeping the context window small while preserving full data access.
//!
//! # Architecture
//!
//! ```text
//! ToolResultProcessor → ResultCache::store() → CacheBackend (Text, …)
//!                                  ↕
//!                     result_cache tool → CacheBackend::execute()
//! ```
//!
//! The [`CacheBackend`] trait is the extension point. Each backend knows how
//! to store one kind of data and expose operations on it. The [`ResultCache`]
//! manages entries, expiration, and disk budget.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod text;

pub use text::TextBackend;

// ── Backend trait ────────────────────────────────────────────────────

/// The kind of backend storing a cached result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Plain-text file with line-oriented operations.
    Text,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
        }
    }
}

/// Operations that can be performed on a cached result.
#[derive(Debug, Clone)]
pub enum CacheOp {
    /// Read a range of lines (1-indexed, inclusive).
    Read {
        /// First line to read (1-indexed).
        start: usize,
        /// Last line to read (1-indexed, inclusive).
        end: usize,
    },
    /// Search for a regex pattern, returning matches with context.
    Grep {
        /// Regex pattern to search for.
        pattern: String,
        /// Number of context lines around each match.
        context_lines: usize,
    },
    /// Return the first N lines.
    Head {
        /// Number of lines to return.
        lines: usize,
    },
    /// Return the last N lines.
    Tail {
        /// Number of lines to return.
        lines: usize,
    },
    /// Return statistics about the cached data.
    Stats,
}

/// Statistics about a cached result.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of lines (for text) or rows (for tabular data).
    pub line_count: usize,
    /// Size on disk in bytes.
    pub disk_bytes: u64,
    /// Human-readable summary (e.g. "1,234 lines, 56 KB").
    pub summary: String,
}

/// Errors from cache operations.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// An I/O error occurred reading or writing cache data.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A grep pattern failed to compile as a regex.
    #[error("invalid regex pattern: {0}")]
    InvalidPattern(String),

    /// The requested cache entry does not exist.
    #[error("cache entry not found: {ref_id}")]
    NotFound {
        /// The reference ID that was not found.
        ref_id: String,
    },

    /// The requested cache entry has expired.
    #[error("cache entry expired: {ref_id}")]
    Expired {
        /// The reference ID that expired.
        ref_id: String,
    },

    /// The requested line range is outside the cached data.
    #[error("line range out of bounds: requested {start}..{end}, have {total} lines")]
    OutOfBounds {
        /// Requested start line.
        start: usize,
        /// Requested end line.
        end: usize,
        /// Actual line count.
        total: usize,
    },
}

/// A backend that stores and operates on one cached result.
///
/// Implementations are created by [`ResultCache::store`] and live for the
/// lifetime of the cache entry. Each backend owns its backing storage
/// (file, buffer, etc.).
///
/// # Extending
///
/// To add a new backend (e.g. Parquet/Arrow for tabular data):
///
/// 1. Implement `CacheBackend` with the new storage format
/// 2. Add a variant to [`BackendKind`]
/// 3. Update [`ResultCache::store`] to select the new backend
pub trait CacheBackend: Send + Sync {
    /// What kind of backend this is.
    fn kind(&self) -> BackendKind;

    /// Execute an operation on the cached data.
    fn execute(&self, op: CacheOp) -> Result<String, CacheError>;

    /// Statistics about the cached data.
    fn stats(&self) -> Result<CacheStats, CacheError>;

    /// A short preview of the data (first N lines/rows).
    fn preview(&self, max_lines: usize) -> Result<String, CacheError>;

    /// Size on disk in bytes.
    fn disk_bytes(&self) -> Result<u64, CacheError>;
}

// ── Cache entry ──────────────────────────────────────────────────────

/// A single cached tool result.
pub struct CacheEntry {
    /// Unique reference ID (e.g. `ref_0001`).
    pub ref_id: String,
    /// The backend that stores and operates on this result.
    pub backend: Box<dyn CacheBackend>,
    /// Which tool produced this result.
    pub tool_name: String,
    /// When the entry was created.
    pub created_at: Instant,
    /// When the entry expires and can be evicted.
    pub expires_at: Instant,
}

impl fmt::Debug for CacheEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CacheEntry")
            .field("ref_id", &self.ref_id)
            .field("tool_name", &self.tool_name)
            .field("kind", &self.backend.kind())
            .finish_non_exhaustive()
    }
}

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for the result cache.
#[derive(Debug, Clone)]
pub struct ResultCacheConfig {
    /// How long entries survive before expiration.
    pub ttl: Duration,
    /// Maximum total disk usage across all entries.
    pub max_disk_bytes: u64,
    /// Number of preview lines to include in the context summary.
    pub preview_lines: usize,
}

impl Default for ResultCacheConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(30 * 60), // 30 minutes
            max_disk_bytes: 100 * 1024 * 1024, // 100 MB
            preview_lines: 20,
        }
    }
}

// ── ResultCache ──────────────────────────────────────────────────────

/// Manages cached tool results with expiration and disk budgets.
///
/// The cache stores large tool outputs on disk and provides agents with
/// random-access operations (read, grep, head, tail, stats) via
/// [`CacheBackend`] implementations.
pub struct ResultCache {
    entries: HashMap<String, CacheEntry>,
    config: ResultCacheConfig,
    base_dir: PathBuf,
    next_id: u64,
}

impl fmt::Debug for ResultCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResultCache")
            .field("entries", &self.entries.len())
            .field("base_dir", &self.base_dir)
            .finish_non_exhaustive()
    }
}

impl ResultCache {
    /// Create a new cache rooted at `base_dir`.
    ///
    /// The directory is created if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns `CacheError::Io` if the directory can't be created.
    pub fn new(
        base_dir: impl Into<PathBuf>,
        config: ResultCacheConfig,
    ) -> Result<Self, CacheError> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self {
            entries: HashMap::new(),
            config,
            base_dir,
            next_id: 0,
        })
    }

    /// Store a tool result, returning the reference ID.
    ///
    /// The `backend_kind` determines which backend stores the data.
    /// Currently only [`BackendKind::Text`] is supported.
    ///
    /// # Errors
    ///
    /// Returns `CacheError::Io` if writing to disk fails.
    pub fn store(
        &mut self,
        tool_name: &str,
        content: &str,
        backend_kind: BackendKind,
    ) -> Result<String, CacheError> {
        let ref_id = self.generate_ref_id();
        let now = Instant::now();

        let backend: Box<dyn CacheBackend> = match backend_kind {
            BackendKind::Text => {
                let path = self.base_dir.join(format!("{ref_id}.txt"));
                Box::new(TextBackend::store(content, &path)?)
            }
        };

        let entry = CacheEntry {
            ref_id: ref_id.clone(),
            backend,
            tool_name: tool_name.to_string(),
            created_at: now,
            expires_at: now + self.config.ttl,
        };

        self.entries.insert(ref_id.clone(), entry);
        Ok(ref_id)
    }

    /// Get a cache entry by reference ID.
    pub fn get(&self, ref_id: &str) -> Result<&CacheEntry, CacheError> {
        let entry = self
            .entries
            .get(ref_id)
            .ok_or_else(|| CacheError::NotFound {
                ref_id: ref_id.to_string(),
            })?;

        if Instant::now() >= entry.expires_at {
            return Err(CacheError::Expired {
                ref_id: ref_id.to_string(),
            });
        }

        Ok(entry)
    }

    /// Execute an operation on a cached entry.
    pub fn execute_op(&self, ref_id: &str, op: CacheOp) -> Result<String, CacheError> {
        let entry = self.get(ref_id)?;
        entry.backend.execute(op)
    }

    /// Remove all expired entries, returning the count removed.
    pub fn evict_expired(&mut self) -> usize {
        let now = Instant::now();
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| now >= e.expires_at)
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired.len();
        for ref_id in &expired {
            if let Some(entry) = self.entries.remove(ref_id) {
                // Best-effort cleanup of backing files
                self.cleanup_entry(&entry);
            }
        }
        count
    }

    /// Total disk bytes across all entries.
    pub fn total_bytes(&self) -> u64 {
        self.entries
            .values()
            .filter_map(|e| e.backend.disk_bytes().ok())
            .sum()
    }

    /// Number of active entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates over all cache entries.
    ///
    /// Returns `(ref_id, entry)` pairs in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &CacheEntry)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The configured number of preview lines.
    pub fn preview_lines(&self) -> usize {
        self.config.preview_lines
    }

    /// The base directory for cache files.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    fn generate_ref_id(&mut self) -> String {
        self.next_id += 1;
        format!("ref_{:04x}", self.next_id)
    }

    fn cleanup_entry(&self, entry: &CacheEntry) {
        match entry.backend.kind() {
            BackendKind::Text => {
                let path = self.base_dir.join(format!("{}.txt", entry.ref_id));
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cache(dir: &Path) -> ResultCache {
        ResultCache::new(dir, ResultCacheConfig::default()).unwrap()
    }

    #[test]
    fn test_store_and_get() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = test_cache(dir.path());

        let ref_id = cache
            .store("db_sql", "line1\nline2\nline3", BackendKind::Text)
            .unwrap();
        assert!(ref_id.starts_with("ref_"));

        let entry = cache.get(&ref_id).unwrap();
        assert_eq!(entry.tool_name, "db_sql");
        assert_eq!(entry.backend.kind(), BackendKind::Text);
    }

    #[test]
    fn test_get_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let cache = test_cache(dir.path());

        let result = cache.get("ref_nonexistent");
        assert!(matches!(result, Err(CacheError::NotFound { .. })));
    }

    #[test]
    fn test_execute_op() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = test_cache(dir.path());

        let ref_id = cache
            .store("test", "alpha\nbeta\ngamma\ndelta", BackendKind::Text)
            .unwrap();

        let result = cache
            .execute_op(&ref_id, CacheOp::Head { lines: 2 })
            .unwrap();
        assert_eq!(result, "alpha\nbeta");
    }

    #[test]
    fn test_evict_expired() {
        let dir = tempfile::tempdir().unwrap();
        let config = ResultCacheConfig {
            ttl: Duration::from_millis(1),
            ..Default::default()
        };
        let mut cache = ResultCache::new(dir.path(), config).unwrap();

        cache.store("test", "data", BackendKind::Text).unwrap();
        assert_eq!(cache.len(), 1);

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(10));

        let evicted = cache.evict_expired();
        assert_eq!(evicted, 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_expired_entry_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let config = ResultCacheConfig {
            ttl: Duration::from_millis(1),
            ..Default::default()
        };
        let mut cache = ResultCache::new(dir.path(), config).unwrap();

        let ref_id = cache.store("test", "data", BackendKind::Text).unwrap();

        std::thread::sleep(Duration::from_millis(10));

        assert!(matches!(
            cache.get(&ref_id),
            Err(CacheError::Expired { .. })
        ));
    }

    #[test]
    fn test_total_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = test_cache(dir.path());

        cache
            .store("test", "hello world", BackendKind::Text)
            .unwrap();
        assert!(cache.total_bytes() > 0);
    }

    #[test]
    fn test_multiple_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = test_cache(dir.path());

        let r1 = cache.store("t1", "data1", BackendKind::Text).unwrap();
        let r2 = cache.store("t2", "data2", BackendKind::Text).unwrap();

        assert_ne!(r1, r2);
        assert_eq!(cache.len(), 2);

        assert_eq!(cache.get(&r1).unwrap().tool_name, "t1");
        assert_eq!(cache.get(&r2).unwrap().tool_name, "t2");
    }

    #[test]
    fn test_iter_returns_all_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = test_cache(dir.path());

        let r1 = cache
            .store("db_sql", "SELECT 1", BackendKind::Text)
            .unwrap();
        let r2 = cache
            .store("web_search", "results here", BackendKind::Text)
            .unwrap();
        let r3 = cache
            .store("db_sql", "SELECT 2", BackendKind::Text)
            .unwrap();

        let entries: HashMap<String, String> = cache
            .iter()
            .map(|(ref_id, entry)| (ref_id.to_string(), entry.tool_name.clone()))
            .collect();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[&r1], "db_sql");
        assert_eq!(entries[&r2], "web_search");
        assert_eq!(entries[&r3], "db_sql");
    }

    #[test]
    fn test_backend_kind_display() {
        assert_eq!(format!("{}", BackendKind::Text), "text");
    }

    #[test]
    fn test_cache_debug() {
        let dir = tempfile::tempdir().unwrap();
        let cache = test_cache(dir.path());
        let debug = format!("{cache:?}");
        assert!(debug.contains("ResultCache"));
    }

    #[test]
    fn test_entry_debug() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = test_cache(dir.path());
        let ref_id = cache.store("tool", "data", BackendKind::Text).unwrap();
        let entry = cache.get(&ref_id).unwrap();
        let debug = format!("{entry:?}");
        assert!(debug.contains("CacheEntry"));
        assert!(debug.contains(&ref_id));
    }

    #[test]
    fn test_config_default() {
        let config = ResultCacheConfig::default();
        assert_eq!(config.ttl, Duration::from_secs(30 * 60));
        assert_eq!(config.max_disk_bytes, 100 * 1024 * 1024);
        assert_eq!(config.preview_lines, 20);
    }

    #[test]
    fn test_evict_cleans_files() {
        let dir = tempfile::tempdir().unwrap();
        let config = ResultCacheConfig {
            ttl: Duration::from_millis(1),
            ..Default::default()
        };
        let mut cache = ResultCache::new(dir.path(), config).unwrap();

        let ref_id = cache.store("test", "data", BackendKind::Text).unwrap();
        let file_path = dir.path().join(format!("{ref_id}.txt"));
        assert!(file_path.exists());

        std::thread::sleep(Duration::from_millis(10));
        cache.evict_expired();

        assert!(!file_path.exists());
    }
}
