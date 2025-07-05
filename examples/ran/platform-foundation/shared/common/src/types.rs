//! Common data types for RAN Intelligence Platform Foundation Services

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Unique identifier type
pub type Id = Uuid;

/// Timestamp type
pub type Timestamp = DateTime<Utc>;

/// Generate a new unique identifier
pub fn new_id() -> Id {
    Uuid::new_v4()
}

/// Get current timestamp
pub fn now() -> Timestamp {
    Utc::now()
}

/// Service health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Service is healthy and operational
    Healthy,
    /// Service is degraded but operational
    Degraded,
    /// Service is unhealthy
    Unhealthy,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::Healthy
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

/// Health check result for a service component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    /// Component health status
    pub status: HealthStatus,
    /// Additional details about the component
    pub details: Option<String>,
    /// Component metrics
    pub metrics: HashMap<String, f64>,
    /// Last check timestamp
    pub last_check: Timestamp,
}

impl ComponentHealth {
    /// Create a new healthy component
    pub fn healthy(name: String) -> Self {
        Self {
            name,
            status: HealthStatus::Healthy,
            details: None,
            metrics: HashMap::new(),
            last_check: now(),
        }
    }
    
    /// Create a degraded component
    pub fn degraded(name: String, details: String) -> Self {
        Self {
            name,
            status: HealthStatus::Degraded,
            details: Some(details),
            metrics: HashMap::new(),
            last_check: now(),
        }
    }
    
    /// Create an unhealthy component
    pub fn unhealthy(name: String, details: String) -> Self {
        Self {
            name,
            status: HealthStatus::Unhealthy,
            details: Some(details),
            metrics: HashMap::new(),
            last_check: now(),
        }
    }
    
    /// Add a metric to the component
    pub fn with_metric(mut self, key: String, value: f64) -> Self {
        self.metrics.insert(key, value);
        self
    }
}

/// Overall service health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Overall health status
    pub status: HealthStatus,
    /// Component-specific health checks
    pub components: HashMap<String, ComponentHealth>,
    /// Timestamp of the health check
    pub timestamp: Timestamp,
    /// Service version
    pub version: String,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

impl HealthCheck {
    /// Create a new health check
    pub fn new(version: String, uptime_seconds: u64) -> Self {
        Self {
            status: HealthStatus::Healthy,
            components: HashMap::new(),
            timestamp: now(),
            version,
            uptime_seconds,
        }
    }
    
    /// Add a component health check
    pub fn add_component(mut self, component: ComponentHealth) -> Self {
        // Update overall status based on component status
        match component.status {
            HealthStatus::Unhealthy => self.status = HealthStatus::Unhealthy,
            HealthStatus::Degraded if self.status == HealthStatus::Healthy => {
                self.status = HealthStatus::Degraded;
            }
            _ => {}
        }
        
        self.components.insert(component.name.clone(), component);
        self
    }
    
    /// Check if the service is healthy
    pub fn is_healthy(&self) -> bool {
        self.status == HealthStatus::Healthy
    }
}

/// Processing status for operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingStatus {
    /// Operation is pending
    Pending,
    /// Operation is in progress
    InProgress,
    /// Operation completed successfully
    Completed,
    /// Operation failed
    Failed,
    /// Operation was cancelled
    Cancelled,
}

impl Default for ProcessingStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl std::fmt::Display for ProcessingStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::InProgress => write!(f, "in_progress"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Operation result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationResult<T> {
    /// Unique operation ID
    pub id: Id,
    /// Operation status
    pub status: ProcessingStatus,
    /// Result data (if successful)
    pub data: Option<T>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Operation start time
    pub started_at: Timestamp,
    /// Operation completion time
    pub completed_at: Option<Timestamp>,
    /// Processing duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl<T> OperationResult<T> {
    /// Create a new pending operation
    pub fn pending() -> Self {
        Self {
            id: new_id(),
            status: ProcessingStatus::Pending,
            data: None,
            error: None,
            started_at: now(),
            completed_at: None,
            duration_ms: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Mark operation as in progress
    pub fn in_progress(mut self) -> Self {
        self.status = ProcessingStatus::InProgress;
        self
    }
    
    /// Complete operation successfully
    pub fn complete(mut self, data: T) -> Self {
        let now = now();
        self.status = ProcessingStatus::Completed;
        self.data = Some(data);
        self.completed_at = Some(now);
        self.duration_ms = Some(
            (now - self.started_at).num_milliseconds() as u64
        );
        self
    }
    
    /// Fail operation with error
    pub fn fail(mut self, error: String) -> Self {
        let now = now();
        self.status = ProcessingStatus::Failed;
        self.error = Some(error);
        self.completed_at = Some(now);
        self.duration_ms = Some(
            (now - self.started_at).num_milliseconds() as u64
        );
        self
    }
    
    /// Cancel operation
    pub fn cancel(mut self) -> Self {
        let now = now();
        self.status = ProcessingStatus::Cancelled;
        self.completed_at = Some(now);
        self.duration_ms = Some(
            (now - self.started_at).num_milliseconds() as u64
        );
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Check if operation is finished
    pub fn is_finished(&self) -> bool {
        matches!(self.status, ProcessingStatus::Completed | ProcessingStatus::Failed | ProcessingStatus::Cancelled)
    }
    
    /// Check if operation was successful
    pub fn is_successful(&self) -> bool {
        self.status == ProcessingStatus::Completed
    }
}

/// File processing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size_bytes: u64,
    /// File format (csv, json, parquet, etc.)
    pub format: String,
    /// File creation time
    pub created_at: Timestamp,
    /// File modification time
    pub modified_at: Timestamp,
    /// File checksum (optional)
    pub checksum: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl FileInfo {
    /// Create file info from path
    pub fn from_path(path: PathBuf) -> std::io::Result<Self> {
        let metadata = std::fs::metadata(&path)?;
        let size_bytes = metadata.len();
        
        let format = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("unknown")
            .to_lowercase();
        
        // Try to get timestamps
        let created_at = metadata.created()
            .map(|time| DateTime::from(time))
            .unwrap_or_else(|_| now());
        
        let modified_at = metadata.modified()
            .map(|time| DateTime::from(time))
            .unwrap_or_else(|_| now());
        
        Ok(Self {
            path,
            size_bytes,
            format,
            created_at,
            modified_at,
            checksum: None,
            metadata: HashMap::new(),
        })
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Set checksum
    pub fn with_checksum(mut self, checksum: String) -> Self {
        self.checksum = Some(checksum);
        self
    }
}

/// Data processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Number of input rows
    pub input_rows: u64,
    /// Number of output rows
    pub output_rows: u64,
    /// Number of processed bytes
    pub processed_bytes: u64,
    /// Number of errors encountered
    pub error_count: u64,
    /// Processing duration in milliseconds
    pub duration_ms: u64,
    /// Memory usage in MB
    pub memory_usage_mb: u64,
    /// Throughput (rows per second)
    pub throughput_rows_per_sec: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
}

impl ProcessingMetrics {
    /// Create new processing metrics
    pub fn new(
        input_rows: u64,
        output_rows: u64,
        processed_bytes: u64,
        error_count: u64,
        duration_ms: u64,
        memory_usage_mb: u64,
    ) -> Self {
        let throughput_rows_per_sec = if duration_ms > 0 {
            (output_rows as f64) / (duration_ms as f64 / 1000.0)
        } else {
            0.0
        };
        
        let error_rate = if input_rows > 0 {
            error_count as f64 / input_rows as f64
        } else {
            0.0
        };
        
        Self {
            input_rows,
            output_rows,
            processed_bytes,
            error_count,
            duration_ms,
            memory_usage_mb,
            throughput_rows_per_sec,
            error_rate,
        }
    }
}

/// ML model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: Id,
    /// Model name
    pub name: String,
    /// Model type (classifier, regressor, etc.)
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Model accuracy (0.0 to 1.0)
    pub accuracy: Option<f64>,
    /// Training timestamp
    pub trained_at: Timestamp,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Feature names
    pub features: Vec<String>,
    /// Target variable name
    pub target: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ModelInfo {
    /// Create new model info
    pub fn new(
        name: String,
        model_type: String,
        version: String,
        target: String,
    ) -> Self {
        Self {
            id: new_id(),
            name,
            model_type,
            version,
            accuracy: None,
            trained_at: now(),
            size_bytes: 0,
            features: Vec::new(),
            target,
            metadata: HashMap::new(),
        }
    }
    
    /// Set accuracy
    pub fn with_accuracy(mut self, accuracy: f64) -> Self {
        self.accuracy = Some(accuracy);
        self
    }
    
    /// Set features
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }
    
    /// Set size
    pub fn with_size(mut self, size_bytes: u64) -> Self {
        self.size_bytes = size_bytes;
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Service configuration info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    /// Service name
    pub name: String,
    /// Service version
    pub version: String,
    /// Service description
    pub description: String,
    /// Service start time
    pub started_at: Timestamp,
    /// Service endpoints
    pub endpoints: Vec<String>,
    /// Service dependencies
    pub dependencies: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ServiceInfo {
    /// Create new service info
    pub fn new(
        name: String,
        version: String,
        description: String,
    ) -> Self {
        Self {
            name,
            version,
            description,
            started_at: now(),
            endpoints: Vec::new(),
            dependencies: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add endpoint
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoints.push(endpoint);
        self
    }
    
    /// Add dependency
    pub fn with_dependency(mut self, dependency: String) -> Self {
        self.dependencies.push(dependency);
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        (now() - self.started_at).num_seconds() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_id_generation() {
        let id1 = new_id();
        let id2 = new_id();
        assert_ne!(id1, id2);
    }
    
    #[test]
    fn test_timestamp() {
        let ts1 = now();
        thread::sleep(Duration::from_millis(10));
        let ts2 = now();
        assert!(ts2 > ts1);
    }
    
    #[test]
    fn test_health_status() {
        assert_eq!(HealthStatus::default(), HealthStatus::Healthy);
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
    }
    
    #[test]
    fn test_component_health() {
        let healthy = ComponentHealth::healthy("test".to_string());
        assert_eq!(healthy.status, HealthStatus::Healthy);
        assert!(healthy.details.is_none());
        
        let degraded = ComponentHealth::degraded("test".to_string(), "issue".to_string());
        assert_eq!(degraded.status, HealthStatus::Degraded);
        assert_eq!(degraded.details, Some("issue".to_string()));
        
        let unhealthy = ComponentHealth::unhealthy("test".to_string(), "error".to_string());
        assert_eq!(unhealthy.status, HealthStatus::Unhealthy);
        
        let with_metric = healthy.with_metric("cpu".to_string(), 50.0);
        assert_eq!(with_metric.metrics.get("cpu"), Some(&50.0));
    }
    
    #[test]
    fn test_health_check() {
        let mut health = HealthCheck::new("1.0.0".to_string(), 3600);
        assert_eq!(health.status, HealthStatus::Healthy);
        assert!(health.is_healthy());
        
        health = health.add_component(ComponentHealth::healthy("db".to_string()));
        assert_eq!(health.status, HealthStatus::Healthy);
        
        health = health.add_component(ComponentHealth::degraded("cache".to_string(), "slow".to_string()));
        assert_eq!(health.status, HealthStatus::Degraded);
        assert!(!health.is_healthy());
        
        health = health.add_component(ComponentHealth::unhealthy("api".to_string(), "down".to_string()));
        assert_eq!(health.status, HealthStatus::Unhealthy);
        assert!(!health.is_healthy());
    }
    
    #[test]
    fn test_processing_status() {
        assert_eq!(ProcessingStatus::default(), ProcessingStatus::Pending);
        assert_eq!(ProcessingStatus::Pending.to_string(), "pending");
        assert_eq!(ProcessingStatus::InProgress.to_string(), "in_progress");
        assert_eq!(ProcessingStatus::Completed.to_string(), "completed");
        assert_eq!(ProcessingStatus::Failed.to_string(), "failed");
        assert_eq!(ProcessingStatus::Cancelled.to_string(), "cancelled");
    }
    
    #[test]
    fn test_operation_result() {
        let mut result = OperationResult::<String>::pending();
        assert_eq!(result.status, ProcessingStatus::Pending);
        assert!(!result.is_finished());
        assert!(!result.is_successful());
        
        result = result.in_progress();
        assert_eq!(result.status, ProcessingStatus::InProgress);
        
        result = result.complete("success".to_string());
        assert_eq!(result.status, ProcessingStatus::Completed);
        assert!(result.is_finished());
        assert!(result.is_successful());
        assert_eq!(result.data, Some("success".to_string()));
        assert!(result.duration_ms.is_some());
        
        let mut failed_result = OperationResult::<String>::pending();
        failed_result = failed_result.fail("error".to_string());
        assert_eq!(failed_result.status, ProcessingStatus::Failed);
        assert!(failed_result.is_finished());
        assert!(!failed_result.is_successful());
        assert_eq!(failed_result.error, Some("error".to_string()));
        
        let mut cancelled_result = OperationResult::<String>::pending();
        cancelled_result = cancelled_result.cancel();
        assert_eq!(cancelled_result.status, ProcessingStatus::Cancelled);
        assert!(cancelled_result.is_finished());
        assert!(!cancelled_result.is_successful());
    }
    
    #[test]
    fn test_processing_metrics() {
        let metrics = ProcessingMetrics::new(1000, 950, 1024, 50, 5000, 256);
        assert_eq!(metrics.input_rows, 1000);
        assert_eq!(metrics.output_rows, 950);
        assert_eq!(metrics.error_count, 50);
        assert_eq!(metrics.error_rate, 0.05);
        assert!(metrics.throughput_rows_per_sec > 0.0);
    }
    
    #[test]
    fn test_model_info() {
        let model = ModelInfo::new(
            "test_model".to_string(),
            "classifier".to_string(),
            "1.0".to_string(),
            "target".to_string(),
        )
        .with_accuracy(0.95)
        .with_features(vec!["feature1".to_string(), "feature2".to_string()])
        .with_size(1024)
        .with_metadata("author".to_string(), "test".to_string());
        
        assert_eq!(model.name, "test_model");
        assert_eq!(model.accuracy, Some(0.95));
        assert_eq!(model.features.len(), 2);
        assert_eq!(model.size_bytes, 1024);
        assert_eq!(model.metadata.get("author"), Some(&"test".to_string()));
    }
    
    #[test]
    fn test_service_info() {
        let service = ServiceInfo::new(
            "test_service".to_string(),
            "1.0.0".to_string(),
            "Test service".to_string(),
        )
        .with_endpoint("http://localhost:8080".to_string())
        .with_dependency("database".to_string())
        .with_metadata("env".to_string(), "test".to_string());
        
        assert_eq!(service.name, "test_service");
        assert_eq!(service.endpoints.len(), 1);
        assert_eq!(service.dependencies.len(), 1);
        assert!(service.uptime_seconds() >= 0);
    }
}