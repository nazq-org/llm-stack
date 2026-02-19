//! Text file-backed cache backend.
//!
//! Stores tool output as a plain text file with line-oriented operations.
//! Uses buffered reads for efficient access to large files.

use super::{BackendKind, CacheError, CacheOp, CacheStats};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// A cache backend backed by a plain text file.
///
/// Supports line-oriented operations: read ranges, grep, head, tail, stats.
/// The file is written once at creation and read on each operation.
#[derive(Debug)]
pub struct TextBackend {
    /// Path to the backing file.
    path: PathBuf,
    /// Cached line count (computed on first access or at store time).
    line_count: usize,
}

impl TextBackend {
    /// Write content to a file and create a backend for it.
    ///
    /// # Errors
    ///
    /// Returns `CacheError::Io` if the file can't be written.
    pub fn store(content: &str, path: &Path) -> Result<Self, CacheError> {
        fs::write(path, content)?;
        let line_count = content.lines().count();
        Ok(Self {
            path: path.to_path_buf(),
            line_count,
        })
    }

    /// Open an existing text cache file.
    ///
    /// # Errors
    ///
    /// Returns `CacheError::Io` if the file can't be read.
    pub fn open(path: &Path) -> Result<Self, CacheError> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let line_count = reader.lines().count();
        Ok(Self {
            path: path.to_path_buf(),
            line_count,
        })
    }

    fn read_lines(&self) -> Result<Vec<String>, CacheError> {
        let file = fs::File::open(&self.path)?;
        let reader = BufReader::new(file);
        reader
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .map_err(CacheError::from)
    }

    fn execute_read(&self, start: usize, end: usize) -> Result<String, CacheError> {
        if start == 0 || end == 0 || start > end {
            return Err(CacheError::OutOfBounds {
                start,
                end,
                total: self.line_count,
            });
        }
        if start > self.line_count {
            return Err(CacheError::OutOfBounds {
                start,
                end,
                total: self.line_count,
            });
        }

        let lines = self.read_lines()?;
        let clamped_end = end.min(self.line_count);
        let selected: Vec<&str> = lines[start - 1..clamped_end]
            .iter()
            .map(String::as_str)
            .collect();
        Ok(selected.join("\n"))
    }

    fn execute_grep(&self, pattern: &str, context_lines: usize) -> Result<String, CacheError> {
        let re =
            regex::Regex::new(pattern).map_err(|e| CacheError::InvalidPattern(e.to_string()))?;

        let lines = self.read_lines()?;
        let total = lines.len();

        // Find matching line indices
        let matches: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| re.is_match(line))
            .map(|(i, _)| i)
            .collect();

        if matches.is_empty() {
            return Ok("(no matches)".to_string());
        }

        // Expand matches with context, merge overlapping ranges
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for &m in &matches {
            let start = m.saturating_sub(context_lines);
            let end = (m + context_lines).min(total.saturating_sub(1));
            if let Some(last) = ranges.last_mut() {
                if start <= last.1 + 1 {
                    last.1 = end;
                    continue;
                }
            }
            ranges.push((start, end));
        }

        // Format output with line numbers
        let mut output = Vec::new();
        for (range_idx, &(start, end)) in ranges.iter().enumerate() {
            if range_idx > 0 {
                output.push("---".to_string());
            }
            for (i, line) in lines.iter().enumerate().take(end + 1).skip(start) {
                let marker = if matches.contains(&i) { ">" } else { " " };
                output.push(format!("{marker}{:>4}: {line}", i + 1));
            }
        }

        output.push(format!("\n({} matches)", matches.len()));
        Ok(output.join("\n"))
    }

    fn execute_head(&self, n: usize) -> Result<String, CacheError> {
        let lines = self.read_lines()?;
        let take = n.min(lines.len());
        Ok(lines[..take].join("\n"))
    }

    fn execute_tail(&self, n: usize) -> Result<String, CacheError> {
        let lines = self.read_lines()?;
        let skip = lines.len().saturating_sub(n);
        Ok(lines[skip..].join("\n"))
    }
}

impl super::CacheBackend for TextBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Text
    }

    fn execute(&self, op: CacheOp) -> Result<String, CacheError> {
        match op {
            CacheOp::Read { start, end } => self.execute_read(start, end),
            CacheOp::Grep {
                pattern,
                context_lines,
            } => self.execute_grep(&pattern, context_lines),
            CacheOp::Head { lines } => self.execute_head(lines),
            CacheOp::Tail { lines } => self.execute_tail(lines),
            CacheOp::Stats => {
                let stats = self.stats()?;
                Ok(stats.summary)
            }
        }
    }

    fn stats(&self) -> Result<CacheStats, CacheError> {
        let meta = fs::metadata(&self.path)?;
        let disk_bytes = meta.len();
        let summary = format!("{} lines, {}", self.line_count, format_bytes(disk_bytes));
        Ok(CacheStats {
            line_count: self.line_count,
            disk_bytes,
            summary,
        })
    }

    fn preview(&self, max_lines: usize) -> Result<String, CacheError> {
        self.execute_head(max_lines)
    }

    fn disk_bytes(&self) -> Result<u64, CacheError> {
        Ok(fs::metadata(&self.path)?.len())
    }
}

/// Format bytes as a human-readable string.
#[allow(clippy::cast_precision_loss)] // Acceptable: display-only, not used for math
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;

    if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::cache::CacheBackend;
    use tempfile::tempdir;

    fn sample_content() -> String {
        (1..=100)
            .map(|i| format!("line {i}: content for testing"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn create_backend(dir: &Path, content: &str) -> TextBackend {
        let path = dir.join("test.txt");
        TextBackend::store(content, &path).unwrap()
    }

    // ── Store / open ────────────────────────────────────────────────

    #[test]
    fn test_store_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        let backend = TextBackend::store("hello\nworld", &path).unwrap();
        assert_eq!(backend.line_count, 2);
        assert!(path.exists());
    }

    #[test]
    fn test_open_existing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "a\nb\nc").unwrap();
        let backend = TextBackend::open(&path).unwrap();
        assert_eq!(backend.line_count, 3);
    }

    // ── Head / tail ─────────────────────────────────────────────────

    #[test]
    fn test_head() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), &sample_content());
        let result = backend.execute_head(3).unwrap();
        assert_eq!(
            result,
            "line 1: content for testing\nline 2: content for testing\nline 3: content for testing"
        );
    }

    #[test]
    fn test_head_more_than_available() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "a\nb");
        let result = backend.execute_head(10).unwrap();
        assert_eq!(result, "a\nb");
    }

    #[test]
    fn test_tail() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), &sample_content());
        let result = backend.execute_tail(2).unwrap();
        assert!(result.contains("line 99:"));
        assert!(result.contains("line 100:"));
    }

    #[test]
    fn test_tail_more_than_available() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "a\nb");
        let result = backend.execute_tail(10).unwrap();
        assert_eq!(result, "a\nb");
    }

    // ── Read range ──────────────────────────────────────────────────

    #[test]
    fn test_read_range() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), &sample_content());
        let result = backend.execute_read(5, 7).unwrap();
        assert!(result.contains("line 5:"));
        assert!(result.contains("line 6:"));
        assert!(result.contains("line 7:"));
        assert!(!result.contains("line 4:"));
        assert!(!result.contains("line 8:"));
    }

    #[test]
    fn test_read_range_clamps_end() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "a\nb\nc");
        let result = backend.execute_read(2, 100).unwrap();
        assert_eq!(result, "b\nc");
    }

    #[test]
    fn test_read_range_invalid_zero() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "a\nb");
        assert!(matches!(
            backend.execute_read(0, 1),
            Err(CacheError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn test_read_range_start_past_end() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "a\nb");
        assert!(matches!(
            backend.execute_read(5, 3),
            Err(CacheError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn test_read_range_start_past_total() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "a\nb");
        assert!(matches!(
            backend.execute_read(10, 20),
            Err(CacheError::OutOfBounds { .. })
        ));
    }

    // ── Grep ────────────────────────────────────────────────────────

    #[test]
    fn test_grep_finds_matches() {
        let dir = tempdir().unwrap();
        let content = "apple\nbanana\napricot\nblueberry\navocado";
        let backend = create_backend(dir.path(), content);
        let result = backend.execute_grep("^a", 0).unwrap();
        assert!(result.contains("apple"));
        assert!(result.contains("apricot"));
        assert!(result.contains("avocado"));
        assert!(result.contains("(3 matches)"));
    }

    #[test]
    fn test_grep_with_context() {
        let dir = tempdir().unwrap();
        let content = "aaa\nbbb\nccc\nddd\neee";
        let backend = create_backend(dir.path(), content);
        let result = backend.execute_grep("ccc", 1).unwrap();
        // Should include bbb (context before) and ddd (context after)
        assert!(result.contains("bbb"));
        assert!(result.contains("ccc"));
        assert!(result.contains("ddd"));
        assert!(result.contains("(1 matches)"));
    }

    #[test]
    fn test_grep_no_matches() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "hello\nworld");
        let result = backend.execute_grep("zzz", 0).unwrap();
        assert_eq!(result, "(no matches)");
    }

    #[test]
    fn test_grep_invalid_pattern() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "test");
        let result = backend.execute_grep("[invalid", 0);
        assert!(matches!(result, Err(CacheError::InvalidPattern(_))));
    }

    #[test]
    fn test_grep_marks_matching_lines() {
        let dir = tempdir().unwrap();
        let content = "aaa\nbbb\nccc";
        let backend = create_backend(dir.path(), content);
        let result = backend.execute_grep("bbb", 1).unwrap();
        // Matching line gets ">" prefix, context lines get " " prefix
        assert!(result.contains(">   2: bbb"));
        assert!(result.contains("    1: aaa"));
    }

    // ── Stats / preview / disk_bytes ────────────────────────────────

    #[test]
    fn test_stats() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), &sample_content());
        let stats = backend.stats().unwrap();
        assert_eq!(stats.line_count, 100);
        assert!(stats.disk_bytes > 0);
        assert!(stats.summary.contains("100 lines"));
    }

    #[test]
    fn test_preview() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), &sample_content());
        let preview = backend.preview(5).unwrap();
        let lines: Vec<&str> = preview.lines().collect();
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn test_disk_bytes() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "hello world");
        let bytes = backend.disk_bytes().unwrap();
        assert_eq!(bytes, 11); // "hello world" is 11 bytes
    }

    // ── CacheBackend trait dispatch ─────────────────────────────────

    #[test]
    fn test_trait_dispatch_head() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "a\nb\nc");
        let result = backend.execute(CacheOp::Head { lines: 2 }).unwrap();
        assert_eq!(result, "a\nb");
    }

    #[test]
    fn test_trait_dispatch_tail() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "a\nb\nc");
        let result = backend.execute(CacheOp::Tail { lines: 2 }).unwrap();
        assert_eq!(result, "b\nc");
    }

    #[test]
    fn test_trait_dispatch_stats() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "a\nb\nc");
        let result = backend.execute(CacheOp::Stats).unwrap();
        assert!(result.contains("3 lines"));
    }

    #[test]
    fn test_kind() {
        let dir = tempdir().unwrap();
        let backend = create_backend(dir.path(), "test");
        assert_eq!(backend.kind(), BackendKind::Text);
    }

    // ── format_bytes ────────────────────────────────────────────────

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(2_621_440), "2.5 MB");
    }
}
