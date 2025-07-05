//! Common utilities for RAN Intelligence Platform Foundation Services

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read};

use crate::error::{PlatformError, PlatformResult};
use crate::constants::*;

/// File utilities
pub mod file {
    use super::*;
    
    /// Check if file exists
    pub fn exists(path: &Path) -> bool {
        path.exists() && path.is_file()
    }
    
    /// Check if directory exists
    pub fn dir_exists(path: &Path) -> bool {
        path.exists() && path.is_dir()
    }
    
    /// Ensure directory exists, create if not
    pub fn ensure_dir(path: &Path) -> PlatformResult<()> {
        if !path.exists() {
            fs::create_dir_all(path)
                .map_err(|e| PlatformError::io_error(format!("Failed to create directory {:?}: {}", path, e)))?;
        }
        Ok(())
    }
    
    /// Get file size in bytes
    pub fn size(path: &Path) -> PlatformResult<u64> {
        let metadata = fs::metadata(path)
            .map_err(|e| PlatformError::io_error(format!("Failed to get file metadata for {:?}: {}", path, e)))?;
        Ok(metadata.len())
    }
    
    /// Get file extension
    pub fn extension(path: &Path) -> Option<String> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase())
    }
    
    /// Check if file format is supported
    pub fn is_supported_format(path: &Path, supported_formats: &[&str]) -> bool {
        extension(path)
            .map(|ext| supported_formats.contains(&ext.as_str()))
            .unwrap_or(false)
    }
    
    /// Calculate file checksum (SHA256)
    pub fn checksum(path: &Path) -> PlatformResult<String> {
        let mut file = fs::File::open(path)
            .map_err(|e| PlatformError::io_error(format!("Failed to open file {:?}: {}", path, e)))?;
        
        let mut hasher = Sha256::new();
        let mut buffer = [0; 8192];
        
        loop {
            let bytes_read = file.read(&mut buffer)
                .map_err(|e| PlatformError::io_error(format!("Failed to read file {:?}: {}", path, e)))?;
            
            if bytes_read == 0 {
                break;
            }
            
            hasher.update(&buffer[..bytes_read]);
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Copy file with progress callback
    pub fn copy_with_progress<F>(
        from: &Path,
        to: &Path,
        mut progress_callback: F,
    ) -> PlatformResult<u64>
    where
        F: FnMut(u64, u64),
    {
        let from_size = size(from)?;
        let mut from_file = fs::File::open(from)
            .map_err(|e| PlatformError::io_error(format!("Failed to open source file {:?}: {}", from, e)))?;
        
        let mut to_file = fs::File::create(to)
            .map_err(|e| PlatformError::io_error(format!("Failed to create destination file {:?}: {}", to, e)))?;
        
        let mut buffer = [0; 64 * 1024]; // 64KB buffer
        let mut total_copied = 0u64;
        
        loop {
            let bytes_read = from_file.read(&mut buffer)
                .map_err(|e| PlatformError::io_error(format!("Failed to read from source file: {}", e)))?;
            
            if bytes_read == 0 {
                break;
            }
            
            use std::io::Write;
            to_file.write_all(&buffer[..bytes_read])
                .map_err(|e| PlatformError::io_error(format!("Failed to write to destination file: {}", e)))?;
            
            total_copied += bytes_read as u64;
            progress_callback(total_copied, from_size);
        }
        
        Ok(total_copied)
    }
    
    /// Delete file safely
    pub fn delete(path: &Path) -> PlatformResult<()> {
        if exists(path) {
            fs::remove_file(path)
                .map_err(|e| PlatformError::io_error(format!("Failed to delete file {:?}: {}", path, e)))?;
        }
        Ok(())
    }
    
    /// List files in directory with optional extension filter
    pub fn list_files(dir: &Path, extension: Option<&str>) -> PlatformResult<Vec<PathBuf>> {
        let entries = fs::read_dir(dir)
            .map_err(|e| PlatformError::io_error(format!("Failed to read directory {:?}: {}", dir, e)))?;
        
        let mut files = Vec::new();
        
        for entry in entries {
            let entry = entry
                .map_err(|e| PlatformError::io_error(format!("Failed to read directory entry: {}", e)))?;
            
            let path = entry.path();
            
            if path.is_file() {
                let should_include = if let Some(ext) = extension {
                    self::extension(&path)
                        .map(|file_ext| file_ext == ext)
                        .unwrap_or(false)
                } else {
                    true
                };
                
                if should_include {
                    files.push(path);
                }
            }
        }
        
        files.sort();
        Ok(files)
    }
}

/// Time utilities
pub mod time {
    use super::*;
    
    /// Get current Unix timestamp in seconds
    pub fn unix_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Get current Unix timestamp in milliseconds
    pub fn unix_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
    
    /// Convert Duration to human-readable string
    pub fn duration_to_string(duration: Duration) -> String {
        let total_secs = duration.as_secs();
        
        if total_secs < 60 {
            format!("{:.1}s", duration.as_secs_f64())
        } else if total_secs < 3600 {
            let mins = total_secs / 60;
            let secs = total_secs % 60;
            format!("{}m {}s", mins, secs)
        } else if total_secs < 86400 {
            let hours = total_secs / 3600;
            let mins = (total_secs % 3600) / 60;
            format!("{}h {}m", hours, mins)
        } else {
            let days = total_secs / 86400;
            let hours = (total_secs % 86400) / 3600;
            format!("{}d {}h", days, hours)
        }
    }
    
    /// Measure execution time of a closure
    pub fn measure<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
    
    /// Measure execution time of an async closure
    pub async fn measure_async<F, Fut, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = Instant::now();
        let result = f().await;
        let duration = start.elapsed();
        (result, duration)
    }
}

/// Memory utilities
pub mod memory {
    use super::*;
    
    /// Format bytes as human-readable string
    pub fn bytes_to_string(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
        const THRESHOLD: f64 = 1024.0;
        
        if bytes == 0 {
            return "0 B".to_string();
        }
        
        let bytes_f = bytes as f64;
        let unit_index = (bytes_f.log(THRESHOLD).floor() as usize).min(UNITS.len() - 1);
        let value = bytes_f / THRESHOLD.powi(unit_index as i32);
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.1} {}", value, UNITS[unit_index])
        }
    }
    
    /// Parse human-readable bytes string to u64
    pub fn string_to_bytes(s: &str) -> PlatformResult<u64> {
        let s = s.trim().to_uppercase();
        
        let (number_part, unit_part) = if let Some(pos) = s.find(char::is_alphabetic) {
            (&s[..pos], &s[pos..])
        } else {
            (s.as_str(), "B")
        };
        
        let number: f64 = number_part.trim().parse()
            .map_err(|e| PlatformError::validation(format!("Invalid number in bytes string '{}': {}", s, e)))?;
        
        let multiplier = match unit_part.trim() {
            "B" => 1,
            "KB" | "K" => 1024,
            "MB" | "M" => 1024 * 1024,
            "GB" | "G" => 1024 * 1024 * 1024,
            "TB" | "T" => 1024u64.pow(4),
            "PB" | "P" => 1024u64.pow(5),
            _ => return Err(PlatformError::validation(format!("Unknown unit in bytes string: {}", unit_part))),
        };
        
        Ok((number * multiplier as f64) as u64)
    }
    
    /// Get current process memory usage in bytes
    #[cfg(target_os = "linux")]
    pub fn current_usage() -> PlatformResult<u64> {
        let statm = fs::read_to_string("/proc/self/statm")
            .map_err(|e| PlatformError::resource(format!("Failed to read /proc/self/statm: {}", e)))?;
        
        let parts: Vec<&str> = statm.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(PlatformError::resource("Invalid /proc/self/statm format".to_string()));
        }
        
        let rss_pages: u64 = parts[1].parse()
            .map_err(|e| PlatformError::resource(format!("Failed to parse RSS pages: {}", e)))?;
        
        // Convert pages to bytes (page size is typically 4096 bytes)
        Ok(rss_pages * 4096)
    }
    
    #[cfg(not(target_os = "linux"))]
    pub fn current_usage() -> PlatformResult<u64> {
        // Fallback implementation for non-Linux systems
        // This is a rough estimate and not accurate
        Ok(0)
    }
}

/// String utilities
pub mod string {
    use super::*;
    
    /// Truncate string to specified length with ellipsis
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else if max_len <= 3 {
            "...".to_string()
        } else {
            format!("{}...", &s[..max_len - 3])
        }
    }
    
    /// Convert string to snake_case
    pub fn to_snake_case(s: &str) -> String {
        let mut result = String::new();
        let mut prev_is_lower = false;
        
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() {
                if i > 0 && (prev_is_lower || s.chars().nth(i + 1).map_or(false, |next| next.is_lowercase())) {
                    result.push('_');
                }
                result.push(c.to_lowercase().next().unwrap());
                prev_is_lower = false;
            } else if c.is_alphanumeric() {
                result.push(c);
                prev_is_lower = c.is_lowercase();
            } else {
                result.push('_');
                prev_is_lower = false;
            }
        }
        
        result
    }
    
    /// Convert string to kebab-case
    pub fn to_kebab_case(s: &str) -> String {
        to_snake_case(s).replace('_', "-")
    }
    
    /// Generate random string of specified length
    pub fn random(length: usize) -> String {
        use uuid::Uuid;
        
        let mut result = String::new();
        while result.len() < length {
            let uuid_str = Uuid::new_v4().to_string().replace('-', "");
            result.push_str(&uuid_str);
        }
        
        result.truncate(length);
        result
    }
    
    /// Sanitize string for use as filename
    pub fn sanitize_filename(s: &str) -> String {
        s.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }
}

/// Math utilities
pub mod math {
    use super::*;
    
    /// Calculate percentage
    pub fn percentage(value: f64, total: f64) -> f64 {
        if total == 0.0 {
            0.0
        } else {
            (value / total) * 100.0
        }
    }
    
    /// Round to specified decimal places
    pub fn round_to_places(value: f64, places: u32) -> f64 {
        let multiplier = 10f64.powi(places as i32);
        (value * multiplier).round() / multiplier
    }
    
    /// Calculate mean of slice
    pub fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }
    
    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = self::mean(values);
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    /// Calculate median
    pub fn median(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = values.len();
        
        if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        }
    }
    
    /// Clamp value between min and max
    pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }
}

/// Validation utilities
pub mod validation {
    use super::*;
    
    /// Validate email format (basic)
    pub fn is_valid_email(email: &str) -> bool {
        email.contains('@') && email.contains('.') && email.len() > 5
    }
    
    /// Validate UUID format
    pub fn is_valid_uuid(uuid: &str) -> bool {
        uuid::Uuid::parse_str(uuid).is_ok()
    }
    
    /// Validate port number
    pub fn is_valid_port(port: u16) -> bool {
        port > 0
    }
    
    /// Validate file path
    pub fn is_valid_path(path: &str) -> bool {
        !path.is_empty() && !path.contains('\0')
    }
    
    /// Validate service name
    pub fn is_valid_service_name(name: &str) -> bool {
        !name.is_empty() 
            && name.len() <= 64 
            && name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    }
    
    /// Validate version string (semantic versioning)
    pub fn is_valid_version(version: &str) -> bool {
        let parts: Vec<&str> = version.split('.').collect();
        parts.len() >= 2 
            && parts.len() <= 3 
            && parts.iter().all(|part| part.parse::<u32>().is_ok())
    }
}

/// Configuration utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Whether to use jitter
    pub use_jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            use_jitter: true,
        }
    }
}

/// Retry utility with exponential backoff
pub async fn retry_async<F, Fut, T, E>(
    config: &RetryConfig,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut delay = config.initial_delay;
    let mut attempts = 0;
    
    loop {
        attempts += 1;
        
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempts >= config.max_attempts {
                    return Err(e);
                }
                
                // Calculate next delay
                let next_delay = std::cmp::min(
                    Duration::from_millis(
                        (delay.as_millis() as f64 * config.backoff_multiplier) as u64
                    ),
                    config.max_delay,
                );
                
                // Add jitter if configured
                let actual_delay = if config.use_jitter {
                    let jitter = (next_delay.as_millis() as f64 * 0.1) as u64;
                    Duration::from_millis(
                        next_delay.as_millis() as u64 + (jitter / 2)
                    )
                } else {
                    next_delay
                };
                
                tokio::time::sleep(actual_delay).await;
                delay = next_delay;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_file_utilities() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        
        // Test file doesn't exist initially
        assert!(!file::exists(&test_file));
        
        // Create file
        fs::write(&test_file, "test content").unwrap();
        assert!(file::exists(&test_file));
        
        // Test file size
        assert_eq!(file::size(&test_file).unwrap(), 12);
        
        // Test extension
        assert_eq!(file::extension(&test_file), Some("txt".to_string()));
        
        // Test supported format
        assert!(file::is_supported_format(&test_file, &["txt", "csv"]));
        assert!(!file::is_supported_format(&test_file, &["json", "csv"]));
        
        // Test checksum
        let checksum = file::checksum(&test_file).unwrap();
        assert!(!checksum.is_empty());
        
        // Test delete
        file::delete(&test_file).unwrap();
        assert!(!file::exists(&test_file));
    }
    
    #[test]
    fn test_time_utilities() {
        let timestamp = time::unix_timestamp();
        assert!(timestamp > 0);
        
        let timestamp_ms = time::unix_timestamp_ms();
        assert!(timestamp_ms > timestamp * 1000);
        
        // Test duration formatting
        assert_eq!(time::duration_to_string(Duration::from_secs(30)), "30.0s");
        assert_eq!(time::duration_to_string(Duration::from_secs(90)), "1m 30s");
        assert_eq!(time::duration_to_string(Duration::from_secs(3660)), "1h 1m");
        
        // Test measure
        let (result, duration) = time::measure(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
    }
    
    #[test]
    fn test_memory_utilities() {
        assert_eq!(memory::bytes_to_string(0), "0 B");
        assert_eq!(memory::bytes_to_string(1024), "1.0 KB");
        assert_eq!(memory::bytes_to_string(1024 * 1024), "1.0 MB");
        
        assert_eq!(memory::string_to_bytes("1024").unwrap(), 1024);
        assert_eq!(memory::string_to_bytes("1 KB").unwrap(), 1024);
        assert_eq!(memory::string_to_bytes("1 MB").unwrap(), 1024 * 1024);
        
        // Test error cases
        assert!(memory::string_to_bytes("invalid").is_err());
        assert!(memory::string_to_bytes("1 XB").is_err());
    }
    
    #[test]
    fn test_string_utilities() {
        assert_eq!(string::truncate("hello world", 5), "he...");
        assert_eq!(string::truncate("hi", 10), "hi");
        
        assert_eq!(string::to_snake_case("CamelCase"), "camel_case");
        assert_eq!(string::to_snake_case("HTTPSConnection"), "https_connection");
        
        assert_eq!(string::to_kebab_case("CamelCase"), "camel-case");
        
        let random_str = string::random(10);
        assert_eq!(random_str.len(), 10);
        
        assert_eq!(string::sanitize_filename("hello/world:test"), "hello_world_test");
    }
    
    #[test]
    fn test_math_utilities() {
        assert_eq!(math::percentage(50.0, 100.0), 50.0);
        assert_eq!(math::percentage(0.0, 0.0), 0.0);
        
        assert_eq!(math::round_to_places(3.14159, 2), 3.14);
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(math::mean(&values), 3.0);
        
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(math::median(&mut values), 3.0);
        
        assert_eq!(math::clamp(5, 1, 10), 5);
        assert_eq!(math::clamp(0, 1, 10), 1);
        assert_eq!(math::clamp(15, 1, 10), 10);
    }
    
    #[test]
    fn test_validation_utilities() {
        assert!(validation::is_valid_email("test@example.com"));
        assert!(!validation::is_valid_email("invalid"));
        
        assert!(validation::is_valid_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!validation::is_valid_uuid("invalid-uuid"));
        
        assert!(validation::is_valid_port(8080));
        assert!(!validation::is_valid_port(0));
        
        assert!(validation::is_valid_path("/valid/path"));
        assert!(!validation::is_valid_path(""));
        
        assert!(validation::is_valid_service_name("my-service"));
        assert!(!validation::is_valid_service_name(""));
        assert!(!validation::is_valid_service_name("invalid/service"));
        
        assert!(validation::is_valid_version("1.0.0"));
        assert!(validation::is_valid_version("2.1"));
        assert!(!validation::is_valid_version("invalid"));
    }
    
    #[tokio::test]
    async fn test_retry_async() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
            use_jitter: false,
        };
        
        let mut attempt_count = 0;
        let result = retry_async(&config, || {
            attempt_count += 1;
            async move {
                if attempt_count < 3 {
                    Err("fail")
                } else {
                    Ok("success")
                }
            }
        }).await;
        
        assert_eq!(result, Ok("success"));
        assert_eq!(attempt_count, 3);
        
        // Test max attempts exceeded
        let mut attempt_count = 0;
        let result = retry_async(&config, || {
            attempt_count += 1;
            async move {
                Err::<&str, &str>("always fail")
            }
        }).await;
        
        assert_eq!(result, Err("always fail"));
        assert_eq!(attempt_count, 3);
    }
}