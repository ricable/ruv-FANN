//! Common configuration management for RAN Intelligence Platform Foundation Services

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

use crate::error::{PlatformError, PlatformResult};
use crate::constants::*;

/// Base configuration trait that all services implement
pub trait ServiceConfig {
    /// Validate the configuration
    fn validate(&self) -> PlatformResult<()>;
    
    /// Get the service name
    fn service_name(&self) -> &str;
    
    /// Get the service version
    fn service_version(&self) -> &str;
}

/// Common configuration shared across all services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonConfig {
    /// Service name
    pub service_name: String,
    
    /// Service version
    pub service_version: String,
    
    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,
    
    /// Enable metrics collection
    pub enable_metrics: bool,
    
    /// Metrics port
    pub metrics_port: u16,
    
    /// Health check port
    pub health_port: u16,
    
    /// Maximum concurrent operations
    pub max_concurrent_operations: usize,
    
    /// Default timeout for operations
    pub default_timeout: Duration,
    
    /// Environment (dev, staging, prod)
    pub environment: String,
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            service_name: "platform-foundation".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            log_level: "info".to_string(),
            enable_metrics: true,
            metrics_port: 9090,
            health_port: 8080,
            max_concurrent_operations: DEFAULT_MAX_CONCURRENT,
            default_timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            environment: "dev".to_string(),
        }
    }
}

/// gRPC server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcConfig {
    /// Server host
    pub host: String,
    
    /// Server port
    pub port: u16,
    
    /// Maximum message size in bytes
    pub max_message_size: usize,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Keep alive timeout
    pub keep_alive_timeout: Duration,
    
    /// Enable TLS
    pub enable_tls: bool,
    
    /// TLS certificate path
    pub tls_cert_path: Option<PathBuf>,
    
    /// TLS private key path
    pub tls_key_path: Option<PathBuf>,
    
    /// Enable gRPC reflection
    pub enable_reflection: bool,
    
    /// Enable gRPC-Web
    pub enable_grpc_web: bool,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: DEFAULT_GRPC_PORT,
            max_message_size: DEFAULT_GRPC_MAX_MESSAGE_SIZE,
            request_timeout: Duration::from_secs(DEFAULT_GRPC_TIMEOUT_SECS),
            keep_alive_timeout: Duration::from_secs(60),
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
            enable_reflection: true,
            enable_grpc_web: false,
        }
    }
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database URL
    pub url: String,
    
    /// Maximum number of connections in the pool
    pub max_connections: u32,
    
    /// Minimum number of connections in the pool
    pub min_connections: u32,
    
    /// Connection timeout
    pub connection_timeout: Duration,
    
    /// Idle timeout
    pub idle_timeout: Duration,
    
    /// Maximum lifetime of a connection
    pub max_lifetime: Duration,
    
    /// Enable SQL logging
    pub enable_logging: bool,
    
    /// Run migrations on startup
    pub run_migrations: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgresql://localhost:5432/ran_intelligence".to_string(),
            max_connections: 10,
            min_connections: 1,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(1800),
            enable_logging: false,
            run_migrations: true,
        }
    }
}

/// File storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Base directory for file storage
    pub base_path: PathBuf,
    
    /// Maximum file size in bytes
    pub max_file_size: u64,
    
    /// Enable file compression
    pub enable_compression: bool,
    
    /// Compression algorithm (gzip, snappy, lz4)
    pub compression_algorithm: String,
    
    /// Enable file checksums
    pub enable_checksums: bool,
    
    /// Checksum algorithm (md5, sha256)
    pub checksum_algorithm: String,
    
    /// Auto-cleanup old files
    pub auto_cleanup: bool,
    
    /// Maximum age for files before cleanup
    pub max_file_age: Duration,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("/tmp/ran-intelligence"),
            max_file_size: 1024 * 1024 * 1024, // 1GB
            enable_compression: true,
            compression_algorithm: "snappy".to_string(),
            enable_checksums: true,
            checksum_algorithm: "sha256".to_string(),
            auto_cleanup: true,
            max_file_age: Duration::from_secs(86400 * 7), // 7 days
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    
    /// Maximum processing time in seconds
    pub max_processing_time_secs: u64,
    
    /// Minimum throughput (operations per second)
    pub min_throughput_ops_per_sec: u64,
    
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    
    /// Performance sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    
    /// Enable adaptive performance tuning
    pub enable_adaptive_tuning: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: MAX_MEMORY_USAGE_MB,
            max_processing_time_secs: MAX_PROCESSING_TIME_SECS,
            min_throughput_ops_per_sec: MIN_THROUGHPUT_ROWS_PER_SEC,
            enable_monitoring: true,
            sampling_rate: 0.1,
            enable_adaptive_tuning: false,
        }
    }
}

/// Load configuration from file
pub fn load_config<T>(path: &str) -> PlatformResult<T>
where
    T: serde::de::DeserializeOwned,
{
    let content = std::fs::read_to_string(path)
        .map_err(|e| PlatformError::config(format!("Failed to read config file {}: {}", path, e)))?;
    
    let config: T = if path.ends_with(".toml") {
        toml::from_str(&content)
            .map_err(|e| PlatformError::config(format!("Failed to parse TOML config: {}", e)))?
    } else if path.ends_with(".json") {
        serde_json::from_str(&content)
            .map_err(|e| PlatformError::config(format!("Failed to parse JSON config: {}", e)))?
    } else {
        return Err(PlatformError::config(format!(
            "Unsupported config file format: {}",
            path
        )));
    };
    
    Ok(config)
}

/// Save configuration to file
pub fn save_config<T>(config: &T, path: &str) -> PlatformResult<()>
where
    T: serde::Serialize,
{
    let content = if path.ends_with(".toml") {
        toml::to_string_pretty(config)
            .map_err(|e| PlatformError::config(format!("Failed to serialize TOML config: {}", e)))?
    } else if path.ends_with(".json") {
        serde_json::to_string_pretty(config)
            .map_err(|e| PlatformError::config(format!("Failed to serialize JSON config: {}", e)))?
    } else {
        return Err(PlatformError::config(format!(
            "Unsupported config file format: {}",
            path
        )));
    };
    
    std::fs::write(path, content)
        .map_err(|e| PlatformError::config(format!("Failed to write config file {}: {}", path, e)))?;
    
    Ok(())
}

/// Environment variable configuration loader
pub struct EnvConfigLoader {
    prefix: String,
}

impl EnvConfigLoader {
    /// Create a new environment config loader with the given prefix
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_uppercase(),
        }
    }
    
    /// Get an environment variable with the prefix
    pub fn get_env(&self, key: &str) -> Option<String> {
        std::env::var(format!("{}_{}", self.prefix, key.to_uppercase())).ok()
    }
    
    /// Get an environment variable with the prefix and parse it
    pub fn get_env_parsed<T>(&self, key: &str) -> PlatformResult<Option<T>>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        match self.get_env(key) {
            Some(value) => {
                let parsed = value.parse::<T>()
                    .map_err(|e| PlatformError::config(format!(
                        "Failed to parse environment variable {}_{}: {}",
                        self.prefix, key.to_uppercase(), e
                    )))?;
                Ok(Some(parsed))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_common_config_default() {
        let config = CommonConfig::default();
        assert_eq!(config.service_name, "platform-foundation");
        assert_eq!(config.log_level, "info");
        assert!(config.enable_metrics);
        assert_eq!(config.max_concurrent_operations, DEFAULT_MAX_CONCURRENT);
        assert_eq!(config.environment, "dev");
    }
    
    #[test]
    fn test_grpc_config_default() {
        let config = GrpcConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, DEFAULT_GRPC_PORT);
        assert_eq!(config.max_message_size, DEFAULT_GRPC_MAX_MESSAGE_SIZE);
        assert!(!config.enable_tls);
        assert!(config.enable_reflection);
    }
    
    #[test]
    fn test_database_config_default() {
        let config = DatabaseConfig::default();
        assert!(config.url.contains("postgresql"));
        assert_eq!(config.max_connections, 10);
        assert_eq!(config.min_connections, 1);
        assert!(config.run_migrations);
    }
    
    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert!(config.base_path.to_string_lossy().contains("ran-intelligence"));
        assert!(config.enable_compression);
        assert_eq!(config.compression_algorithm, "snappy");
        assert!(config.enable_checksums);
        assert_eq!(config.checksum_algorithm, "sha256");
    }
    
    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert_eq!(config.max_memory_mb, MAX_MEMORY_USAGE_MB);
        assert_eq!(config.max_processing_time_secs, MAX_PROCESSING_TIME_SECS);
        assert_eq!(config.min_throughput_ops_per_sec, MIN_THROUGHPUT_ROWS_PER_SEC);
        assert!(config.enable_monitoring);
        assert_eq!(config.sampling_rate, 0.1);
    }
    
    #[test]
    fn test_load_save_config() {
        let config = CommonConfig::default();
        
        // Test TOML
        let toml_file = NamedTempFile::new().unwrap();
        let toml_path = toml_file.path().with_extension("toml");
        
        assert!(save_config(&config, toml_path.to_str().unwrap()).is_ok());
        let loaded_config: CommonConfig = load_config(toml_path.to_str().unwrap()).unwrap();
        assert_eq!(config.service_name, loaded_config.service_name);
        
        // Test JSON
        let json_file = NamedTempFile::new().unwrap();
        let json_path = json_file.path().with_extension("json");
        
        assert!(save_config(&config, json_path.to_str().unwrap()).is_ok());
        let loaded_config: CommonConfig = load_config(json_path.to_str().unwrap()).unwrap();
        assert_eq!(config.service_name, loaded_config.service_name);
    }
    
    #[test]
    fn test_env_config_loader() {
        let loader = EnvConfigLoader::new("TEST");
        
        // Set a test environment variable
        std::env::set_var("TEST_FOO", "bar");
        
        assert_eq!(loader.get_env("FOO"), Some("bar".to_string()));
        assert_eq!(loader.get_env("NONEXISTENT"), None);
        
        // Test parsing
        std::env::set_var("TEST_NUMBER", "42");
        let parsed: Option<i32> = loader.get_env_parsed("NUMBER").unwrap();
        assert_eq!(parsed, Some(42));
        
        // Cleanup
        std::env::remove_var("TEST_FOO");
        std::env::remove_var("TEST_NUMBER");
    }
}