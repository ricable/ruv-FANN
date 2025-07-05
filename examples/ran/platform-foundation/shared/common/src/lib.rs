//! Common utilities for RAN Intelligence Platform Foundation Services
//!
//! This crate provides shared functionality across all foundation services:
//! - Error handling and result types
//! - Logging and tracing utilities
//! - Configuration management
//! - Metrics and monitoring
//! - Common data types and constants

#![warn(missing_docs, rust_2018_idioms)]
#![deny(unsafe_code)]

pub mod error;
pub mod config;
pub mod metrics;
pub mod logging;
pub mod types;
pub mod utils;

pub use error::*;
pub use config::*;
pub use metrics::*;
pub use logging::*;
pub use types::*;
pub use utils::*;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SERVICE_NAME: &str = "platform-foundation-common";

/// Common constants for RAN Intelligence Platform
pub mod constants {
    /// Default batch size for data processing
    pub const DEFAULT_BATCH_SIZE: usize = 10_000;
    
    /// Default maximum concurrent operations
    pub const DEFAULT_MAX_CONCURRENT: usize = 4;
    
    /// Default maximum error rate (1%)
    pub const DEFAULT_MAX_ERROR_RATE: f64 = 0.01;
    
    /// Default timeout for operations (30 seconds)
    pub const DEFAULT_TIMEOUT_SECS: u64 = 30;
    
    /// Standard RAN data columns
    pub const TIMESTAMP_COLUMN: &str = "timestamp";
    pub const CELL_ID_COLUMN: &str = "cell_id";
    pub const KPI_NAME_COLUMN: &str = "kpi_name";
    pub const KPI_VALUE_COLUMN: &str = "kpi_value";
    pub const UE_ID_COLUMN: &str = "ue_id";
    pub const SECTOR_ID_COLUMN: &str = "sector_id";
    
    /// Standard feature columns
    pub const FEATURE_PREFIX: &str = "feat_";
    pub const LAG_FEATURE_PREFIX: &str = "lag_";
    pub const ROLLING_FEATURE_PREFIX: &str = "rolling_";
    pub const TIME_FEATURE_PREFIX: &str = "time_";
    
    /// Model registry constants
    pub const MODEL_VERSION_DELIMITER: &str = "_v";
    pub const MODEL_METADATA_SUFFIX: &str = ".metadata.json";
    pub const MODEL_BINARY_SUFFIX: &str = ".model.bin";
    
    /// gRPC configuration
    pub const DEFAULT_GRPC_PORT: u16 = 50051;
    pub const DEFAULT_GRPC_MAX_MESSAGE_SIZE: usize = 4 * 1024 * 1024; // 4MB
    pub const DEFAULT_GRPC_TIMEOUT_SECS: u64 = 30;
    
    /// File processing
    pub const SUPPORTED_INPUT_FORMATS: &[&str] = &["csv", "json", "parquet"];
    pub const SUPPORTED_OUTPUT_FORMATS: &[&str] = &["parquet", "csv"];
    pub const DEFAULT_PARQUET_COMPRESSION: &str = "snappy";
    pub const DEFAULT_ROW_GROUP_SIZE: usize = 1_000_000;
    
    /// Performance thresholds
    pub const MAX_MEMORY_USAGE_MB: u64 = 4096; // 4GB
    pub const MAX_PROCESSING_TIME_SECS: u64 = 3600; // 1 hour
    pub const MIN_THROUGHPUT_ROWS_PER_SEC: u64 = 1000;
}

/// Initialize the common library
pub fn init() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize metrics registry
    metrics::init_metrics()?;
    
    // Set up panic hook for better error reporting
    std::panic::set_hook(Box::new(|panic_info| {
        tracing::error!(
            "Panic occurred: {}",
            panic_info.to_string()
        );
    }));
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::constants::*;
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert_eq!(SERVICE_NAME, "platform-foundation-common");
    }
    
    #[test]
    fn test_constants() {
        assert!(DEFAULT_BATCH_SIZE > 0);
        assert!(DEFAULT_MAX_CONCURRENT > 0);
        assert!(DEFAULT_MAX_ERROR_RATE < 1.0);
        assert!(DEFAULT_TIMEOUT_SECS > 0);
        
        // Test column names
        assert_eq!(TIMESTAMP_COLUMN, "timestamp");
        assert_eq!(CELL_ID_COLUMN, "cell_id");
        assert_eq!(KPI_NAME_COLUMN, "kpi_name");
        assert_eq!(KPI_VALUE_COLUMN, "kpi_value");
        
        // Test supported formats
        assert!(SUPPORTED_INPUT_FORMATS.contains(&"csv"));
        assert!(SUPPORTED_INPUT_FORMATS.contains(&"json"));
        assert!(SUPPORTED_INPUT_FORMATS.contains(&"parquet"));
        
        assert!(SUPPORTED_OUTPUT_FORMATS.contains(&"parquet"));
        assert!(SUPPORTED_OUTPUT_FORMATS.contains(&"csv"));
    }
    
    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }
}