//! Metrics and monitoring utilities for RAN Intelligence Platform Foundation Services

use once_cell::sync::Lazy;
use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec, IntCounter, IntCounterVec,
    IntGauge, IntGaugeVec, Registry, Opts, HistogramOpts,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{PlatformError, PlatformResult};

/// Global metrics registry
static METRICS_REGISTRY: Lazy<Arc<Mutex<MetricsRegistry>>> = Lazy::new(|| {
    Arc::new(Mutex::new(MetricsRegistry::new()))
});

/// Central metrics registry for all foundation services
pub struct MetricsRegistry {
    registry: Registry,
    
    // Common metrics
    pub requests_total: IntCounterVec,
    pub request_duration: HistogramVec,
    pub active_connections: IntGauge,
    pub memory_usage: Gauge,
    pub cpu_usage: Gauge,
    pub errors_total: IntCounterVec,
    pub operations_total: IntCounterVec,
    pub operations_duration: HistogramVec,
    
    // Data processing metrics
    pub data_processed_bytes: IntCounter,
    pub data_processed_rows: IntCounter,
    pub data_processing_errors: IntCounter,
    pub batch_processing_duration: Histogram,
    pub concurrent_operations: IntGauge,
    
    // ML model metrics
    pub model_predictions: IntCounterVec,
    pub model_training_duration: Histogram,
    pub model_accuracy: GaugeVec,
    pub model_memory_usage: GaugeVec,
    
    // gRPC metrics
    pub grpc_requests_total: IntCounterVec,
    pub grpc_request_duration: HistogramVec,
    pub grpc_errors_total: IntCounterVec,
    
    // Storage metrics
    pub storage_operations: IntCounterVec,
    pub storage_size_bytes: IntGauge,
    pub storage_errors: IntCounterVec,
    
    // Service-specific metrics
    service_metrics: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        let registry = Registry::new();
        
        // Common metrics
        let requests_total = IntCounterVec::new(
            Opts::new("requests_total", "Total number of requests"),
            &["service", "method", "status"],
        ).unwrap();
        
        let request_duration = HistogramVec::new(
            HistogramOpts::new("request_duration_seconds", "Request duration in seconds"),
            &["service", "method"],
        ).unwrap();
        
        let active_connections = IntGauge::new(
            "active_connections",
            "Number of active connections",
        ).unwrap();
        
        let memory_usage = Gauge::new(
            "memory_usage_bytes",
            "Memory usage in bytes",
        ).unwrap();
        
        let cpu_usage = Gauge::new(
            "cpu_usage_percent",
            "CPU usage percentage",
        ).unwrap();
        
        let errors_total = IntCounterVec::new(
            Opts::new("errors_total", "Total number of errors"),
            &["service", "type"],
        ).unwrap();
        
        let operations_total = IntCounterVec::new(
            Opts::new("operations_total", "Total number of operations"),
            &["service", "operation", "status"],
        ).unwrap();
        
        let operations_duration = HistogramVec::new(
            HistogramOpts::new("operations_duration_seconds", "Operation duration in seconds"),
            &["service", "operation"],
        ).unwrap();
        
        // Data processing metrics
        let data_processed_bytes = IntCounter::new(
            "data_processed_bytes_total",
            "Total bytes of data processed",
        ).unwrap();
        
        let data_processed_rows = IntCounter::new(
            "data_processed_rows_total",
            "Total rows of data processed",
        ).unwrap();
        
        let data_processing_errors = IntCounter::new(
            "data_processing_errors_total",
            "Total data processing errors",
        ).unwrap();
        
        let batch_processing_duration = Histogram::new(
            "batch_processing_duration_seconds",
            "Batch processing duration in seconds",
        ).unwrap();
        
        let concurrent_operations = IntGauge::new(
            "concurrent_operations",
            "Number of concurrent operations",
        ).unwrap();
        
        // ML model metrics
        let model_predictions = IntCounterVec::new(
            Opts::new("model_predictions_total", "Total number of model predictions"),
            &["model_id", "model_type"],
        ).unwrap();
        
        let model_training_duration = Histogram::new(
            "model_training_duration_seconds",
            "Model training duration in seconds",
        ).unwrap();
        
        let model_accuracy = GaugeVec::new(
            Opts::new("model_accuracy", "Model accuracy"),
            &["model_id", "model_type"],
        ).unwrap();
        
        let model_memory_usage = GaugeVec::new(
            Opts::new("model_memory_usage_bytes", "Model memory usage in bytes"),
            &["model_id", "model_type"],
        ).unwrap();
        
        // gRPC metrics
        let grpc_requests_total = IntCounterVec::new(
            Opts::new("grpc_requests_total", "Total number of gRPC requests"),
            &["service", "method", "status"],
        ).unwrap();
        
        let grpc_request_duration = HistogramVec::new(
            HistogramOpts::new("grpc_request_duration_seconds", "gRPC request duration in seconds"),
            &["service", "method"],
        ).unwrap();
        
        let grpc_errors_total = IntCounterVec::new(
            Opts::new("grpc_errors_total", "Total number of gRPC errors"),
            &["service", "method", "code"],
        ).unwrap();
        
        // Storage metrics
        let storage_operations = IntCounterVec::new(
            Opts::new("storage_operations_total", "Total number of storage operations"),
            &["operation", "status"],
        ).unwrap();
        
        let storage_size_bytes = IntGauge::new(
            "storage_size_bytes",
            "Storage size in bytes",
        ).unwrap();
        
        let storage_errors = IntCounterVec::new(
            Opts::new("storage_errors_total", "Total number of storage errors"),
            &["operation", "error_type"],
        ).unwrap();
        
        // Register all metrics
        registry.register(Box::new(requests_total.clone())).unwrap();
        registry.register(Box::new(request_duration.clone())).unwrap();
        registry.register(Box::new(active_connections.clone())).unwrap();
        registry.register(Box::new(memory_usage.clone())).unwrap();
        registry.register(Box::new(cpu_usage.clone())).unwrap();
        registry.register(Box::new(errors_total.clone())).unwrap();
        registry.register(Box::new(operations_total.clone())).unwrap();
        registry.register(Box::new(operations_duration.clone())).unwrap();
        registry.register(Box::new(data_processed_bytes.clone())).unwrap();
        registry.register(Box::new(data_processed_rows.clone())).unwrap();
        registry.register(Box::new(data_processing_errors.clone())).unwrap();
        registry.register(Box::new(batch_processing_duration.clone())).unwrap();
        registry.register(Box::new(concurrent_operations.clone())).unwrap();
        registry.register(Box::new(model_predictions.clone())).unwrap();
        registry.register(Box::new(model_training_duration.clone())).unwrap();
        registry.register(Box::new(model_accuracy.clone())).unwrap();
        registry.register(Box::new(model_memory_usage.clone())).unwrap();
        registry.register(Box::new(grpc_requests_total.clone())).unwrap();
        registry.register(Box::new(grpc_request_duration.clone())).unwrap();
        registry.register(Box::new(grpc_errors_total.clone())).unwrap();
        registry.register(Box::new(storage_operations.clone())).unwrap();
        registry.register(Box::new(storage_size_bytes.clone())).unwrap();
        registry.register(Box::new(storage_errors.clone())).unwrap();
        
        Self {
            registry,
            requests_total,
            request_duration,
            active_connections,
            memory_usage,
            cpu_usage,
            errors_total,
            operations_total,
            operations_duration,
            data_processed_bytes,
            data_processed_rows,
            data_processing_errors,
            batch_processing_duration,
            concurrent_operations,
            model_predictions,
            model_training_duration,
            model_accuracy,
            model_memory_usage,
            grpc_requests_total,
            grpc_request_duration,
            grpc_errors_total,
            storage_operations,
            storage_size_bytes,
            storage_errors,
            service_metrics: HashMap::new(),
        }
    }
    
    /// Get the Prometheus registry
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
    
    /// Register a custom metric
    pub fn register_metric<T: prometheus::core::Collector + 'static>(&mut self, metric: T) -> PlatformResult<()> {
        self.registry.register(Box::new(metric))
            .map_err(|e| PlatformError::internal(format!("Failed to register metric: {}", e)))
    }
    
    /// Add service-specific metrics
    pub fn add_service_metrics<T: 'static + Send + Sync>(&mut self, service_name: String, metrics: T) {
        self.service_metrics.insert(service_name, Box::new(metrics));
    }
    
    /// Get service-specific metrics
    pub fn get_service_metrics<T: 'static>(&self, service_name: &str) -> Option<&T> {
        self.service_metrics.get(service_name)
            .and_then(|metrics| metrics.downcast_ref::<T>())
    }
}

/// Initialize the global metrics registry
pub fn init_metrics() -> PlatformResult<()> {
    let _registry = METRICS_REGISTRY.lock()
        .map_err(|e| PlatformError::internal(format!("Failed to initialize metrics registry: {}", e)))?;
    
    tracing::info!("Metrics registry initialized");
    Ok(())
}

/// Get the global metrics registry
pub fn get_metrics() -> Arc<Mutex<MetricsRegistry>> {
    METRICS_REGISTRY.clone()
}

/// Record a request metric
pub fn record_request(service: &str, method: &str, status: &str) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.requests_total
            .with_label_values(&[service, method, status])
            .inc();
    }
}

/// Record request duration
pub fn record_request_duration(service: &str, method: &str, duration: f64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.request_duration
            .with_label_values(&[service, method])
            .observe(duration);
    }
}

/// Record an error
pub fn record_error(service: &str, error_type: &str) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.errors_total
            .with_label_values(&[service, error_type])
            .inc();
    }
}

/// Record an operation
pub fn record_operation(service: &str, operation: &str, status: &str) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.operations_total
            .with_label_values(&[service, operation, status])
            .inc();
    }
}

/// Record operation duration
pub fn record_operation_duration(service: &str, operation: &str, duration: f64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.operations_duration
            .with_label_values(&[service, operation])
            .observe(duration);
    }
}

/// Record data processing
pub fn record_data_processed(bytes: u64, rows: u64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.data_processed_bytes.inc_by(bytes);
        registry.data_processed_rows.inc_by(rows);
    }
}

/// Record data processing error
pub fn record_data_processing_error() {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.data_processing_errors.inc();
    }
}

/// Record batch processing duration
pub fn record_batch_processing_duration(duration: f64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.batch_processing_duration.observe(duration);
    }
}

/// Set concurrent operations count
pub fn set_concurrent_operations(count: i64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.concurrent_operations.set(count);
    }
}

/// Record model prediction
pub fn record_model_prediction(model_id: &str, model_type: &str) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.model_predictions
            .with_label_values(&[model_id, model_type])
            .inc();
    }
}

/// Record model training duration
pub fn record_model_training_duration(duration: f64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.model_training_duration.observe(duration);
    }
}

/// Set model accuracy
pub fn set_model_accuracy(model_id: &str, model_type: &str, accuracy: f64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.model_accuracy
            .with_label_values(&[model_id, model_type])
            .set(accuracy);
    }
}

/// Set model memory usage
pub fn set_model_memory_usage(model_id: &str, model_type: &str, memory_bytes: f64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.model_memory_usage
            .with_label_values(&[model_id, model_type])
            .set(memory_bytes);
    }
}

/// Record gRPC request
pub fn record_grpc_request(service: &str, method: &str, status: &str) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.grpc_requests_total
            .with_label_values(&[service, method, status])
            .inc();
    }
}

/// Record gRPC request duration
pub fn record_grpc_request_duration(service: &str, method: &str, duration: f64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.grpc_request_duration
            .with_label_values(&[service, method])
            .observe(duration);
    }
}

/// Record gRPC error
pub fn record_grpc_error(service: &str, method: &str, code: &str) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.grpc_errors_total
            .with_label_values(&[service, method, code])
            .inc();
    }
}

/// Record storage operation
pub fn record_storage_operation(operation: &str, status: &str) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.storage_operations
            .with_label_values(&[operation, status])
            .inc();
    }
}

/// Set storage size
pub fn set_storage_size(size_bytes: i64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.storage_size_bytes.set(size_bytes);
    }
}

/// Record storage error
pub fn record_storage_error(operation: &str, error_type: &str) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.storage_errors
            .with_label_values(&[operation, error_type])
            .inc();
    }
}

/// Set memory usage
pub fn set_memory_usage(bytes: f64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.memory_usage.set(bytes);
    }
}

/// Set CPU usage
pub fn set_cpu_usage(percent: f64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.cpu_usage.set(percent);
    }
}

/// Set active connections
pub fn set_active_connections(count: i64) {
    if let Ok(registry) = METRICS_REGISTRY.lock() {
        registry.active_connections.set(count);
    }
}

/// Timer for measuring operation duration
pub struct Timer {
    start: std::time::Instant,
}

impl Timer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }
    
    /// Get elapsed time in seconds
    pub fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
    
    /// Stop the timer and record to metrics
    pub fn stop_and_record<F>(self, record_fn: F)
    where
        F: FnOnce(f64),
    {
        let duration = self.elapsed();
        record_fn(duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_registry_creation() {
        let registry = MetricsRegistry::new();
        assert!(registry.registry().gather().len() > 0);
    }
    
    #[test]
    fn test_init_metrics() {
        assert!(init_metrics().is_ok());
    }
    
    #[test]
    fn test_record_functions() {
        let _ = init_metrics();
        
        record_request("test", "GET", "200");
        record_request_duration("test", "GET", 0.5);
        record_error("test", "validation");
        record_operation("test", "process", "success");
        record_operation_duration("test", "process", 1.0);
        record_data_processed(1000, 100);
        record_data_processing_error();
        record_batch_processing_duration(5.0);
        set_concurrent_operations(5);
        record_model_prediction("model1", "classifier");
        record_model_training_duration(60.0);
        set_model_accuracy("model1", "classifier", 0.95);
        set_model_memory_usage("model1", "classifier", 1024.0);
        record_grpc_request("test", "Predict", "OK");
        record_grpc_request_duration("test", "Predict", 0.1);
        record_grpc_error("test", "Predict", "InvalidArgument");
        record_storage_operation("read", "success");
        set_storage_size(1000000);
        record_storage_error("write", "permission");
        set_memory_usage(1024.0);
        set_cpu_usage(50.0);
        set_active_connections(10);
        
        // Should not panic
    }
    
    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let duration = timer.elapsed();
        assert!(duration >= 0.01);
    }
    
    #[test]
    fn test_timer_with_record() {
        let timer = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        timer.stop_and_record(|duration| {
            assert!(duration >= 0.01);
        });
    }
}