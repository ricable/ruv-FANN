//! PFS-DATA: File-based Data Ingestion Service for RAN Intelligence Platform
//!
//! This service provides high-performance batch data ingestion capabilities for processing
//! CSV and JSON files into normalized Parquet format with standardized schema for RAN
//! intelligence applications.
//!
//! ## Features
//!
//! - High-performance file processing with configurable concurrency
//! - Schema normalization with standard RAN data model
//! - Error handling with configurable error rate thresholds
//! - Real-time monitoring and metrics
//! - gRPC API for ingestion control
//! - Directory watching with automatic file discovery
//! - Support for 100GB+ datasets with <0.01% error rate
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   File Watch    │───▶│  Ingestion Queue │───▶│  Processing     │
//! │   & Discovery   │    │  & Scheduling    │    │  Workers        │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//!           │                       │                       │
//!           ▼                       ▼                       ▼
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   File System   │    │     Metrics      │    │   Parquet       │
//! │   Monitoring    │    │   & Monitoring   │    │   Output        │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//! ```

#![warn(missing_docs, rust_2018_idioms)]
#![deny(unsafe_code)]

pub mod config;
pub mod engine;
pub mod error;
pub mod generated;
pub mod metrics;
pub mod schema;
pub mod service;
pub mod storage;
pub mod watcher;

use platform_foundation_common as common;

pub use config::*;
pub use engine::*;
pub use error::*;
pub use metrics::*;
pub use schema::*;
pub use service::*;
pub use storage::*;
pub use watcher::*;

// Re-export generated protobuf types
pub use generated::pfs::data::v1::*;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SERVICE_NAME: &str = "pfs-data";

/// Default configuration values
pub const DEFAULT_BATCH_SIZE: usize = 10_000;
pub const DEFAULT_MAX_CONCURRENT_FILES: usize = 4;
pub const DEFAULT_MAX_ERROR_RATE: f64 = 0.01; // 1%
pub const DEFAULT_ROW_GROUP_SIZE: usize = 1_000_000; // 1M rows
pub const DEFAULT_COMPRESSION: &str = "snappy";

/// Standard RAN data columns
pub const TIMESTAMP_COLUMN: &str = "timestamp";
pub const CELL_ID_COLUMN: &str = "cell_id";
pub const KPI_NAME_COLUMN: &str = "kpi_name";
pub const KPI_VALUE_COLUMN: &str = "kpi_value";
pub const UE_ID_COLUMN: &str = "ue_id";
pub const SECTOR_ID_COLUMN: &str = "sector_id";

/// Initialize the data ingestion service
pub async fn init_service(config: DataIngestionConfig) -> DataIngestionResult<DataIngestionService> {
    // Initialize logging
    common::init_logging(
        SERVICE_NAME,
        &config.common.log_level,
        config.common.environment == "prod",
    )
    .map_err(|e| DataIngestionError::config(format!("Failed to initialize logging: {}", e)))?;
    
    // Initialize metrics
    common::init_metrics()
        .map_err(|e| DataIngestionError::config(format!("Failed to initialize metrics: {}", e)))?;
    
    // Create ingestion engine
    let engine = IngestionEngine::new(config.clone()).await?;
    
    // Create service
    let service = DataIngestionService::new(config, engine).await?;
    
    tracing::info!(
        version = VERSION,
        service = SERVICE_NAME,
        "Data ingestion service initialized successfully"
    );
    
    Ok(service)
}

/// Health check for the service
pub async fn health_check() -> common::HealthCheck {
    let mut health = common::HealthCheck::new(VERSION.to_string(), 0); // TODO: track uptime
    
    // Check memory usage
    let memory_usage = common::memory::current_usage().unwrap_or(0);
    let memory_health = if memory_usage > 4 * 1024 * 1024 * 1024 {
        // > 4GB
        common::ComponentHealth::unhealthy(
            "memory".to_string(),
            format!("High memory usage: {}", common::memory::bytes_to_string(memory_usage)),
        )
    } else if memory_usage > 2 * 1024 * 1024 * 1024 {
        // > 2GB
        common::ComponentHealth::degraded(
            "memory".to_string(),
            format!("Elevated memory usage: {}", common::memory::bytes_to_string(memory_usage)),
        )
    } else {
        common::ComponentHealth::healthy("memory".to_string())
    }
    .with_metric("usage_bytes".to_string(), memory_usage as f64);
    
    health = health.add_component(memory_health);
    
    // Check disk space (simplified)
    let disk_health = common::ComponentHealth::healthy("disk".to_string())
        .with_metric("available_gb".to_string(), 100.0); // TODO: actual disk check
    
    health = health.add_component(disk_health);
    
    // Check processing capacity
    let processing_health = common::ComponentHealth::healthy("processing".to_string())
        .with_metric("active_workers".to_string(), 0.0); // TODO: actual worker count
    
    health = health.add_component(processing_health);
    
    health
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert_eq!(SERVICE_NAME, "pfs-data");
    }
    
    #[test]
    fn test_constants() {
        assert!(DEFAULT_BATCH_SIZE > 0);
        assert!(DEFAULT_MAX_CONCURRENT_FILES > 0);
        assert!(DEFAULT_MAX_ERROR_RATE < 1.0);
        assert!(DEFAULT_ROW_GROUP_SIZE > 0);
        assert!(!DEFAULT_COMPRESSION.is_empty());
        
        // Test column names
        assert_eq!(TIMESTAMP_COLUMN, "timestamp");
        assert_eq!(CELL_ID_COLUMN, "cell_id");
        assert_eq!(KPI_NAME_COLUMN, "kpi_name");
        assert_eq!(KPI_VALUE_COLUMN, "kpi_value");
        assert_eq!(UE_ID_COLUMN, "ue_id");
        assert_eq!(SECTOR_ID_COLUMN, "sector_id");
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let health = health_check().await;
        assert!(!health.components.is_empty());
        assert!(health.components.contains_key("memory"));
        assert!(health.components.contains_key("disk"));
        assert!(health.components.contains_key("processing"));
    }
}