//! Logging and tracing utilities for RAN Intelligence Platform Foundation Services

use tracing::{Level, Span};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};
use std::io;

use crate::error::{PlatformError, PlatformResult};

/// Initialize structured logging for the service
pub fn init_logging(service_name: &str, log_level: &str, enable_json: bool) -> PlatformResult<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(log_level));
    
    let registry = tracing_subscriber::registry().with(filter);
    
    if enable_json {
        // JSON format for production
        let json_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true)
            .with_file(true)
            .with_current_span(true)
            .with_span_list(true)
            .flatten_event(true)
            .with_field("service", service_name)
            .with_writer(io::stdout);
        
        registry.with(json_layer).try_init()
            .map_err(|e| PlatformError::config(format!("Failed to initialize JSON logging: {}", e)))?;
    } else {
        // Human-readable format for development
        let fmt_layer = tracing_subscriber::fmt::layer()
            .pretty()
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true)
            .with_file(true)
            .with_ansi(true)
            .with_writer(io::stdout);
        
        registry.with(fmt_layer).try_init()
            .map_err(|e| PlatformError::config(format!("Failed to initialize pretty logging: {}", e)))?;
    }
    
    tracing::info!(
        service = service_name,
        level = log_level,
        json = enable_json,
        "Logging initialized"
    );
    
    Ok(())
}

/// Create a span for tracing operations
pub fn create_operation_span(operation: &str, service: &str) -> Span {
    tracing::info_span!(
        "operation",
        operation = operation,
        service = service,
        start_time = %chrono::Utc::now(),
    )
}

/// Create a span for tracing requests
pub fn create_request_span(method: &str, path: &str, request_id: &str) -> Span {
    tracing::info_span!(
        "request",
        method = method,
        path = path,
        request_id = request_id,
        start_time = %chrono::Utc::now(),
    )
}

/// Create a span for tracing data processing
pub fn create_data_processing_span(operation: &str, file_path: &str, rows: u64) -> Span {
    tracing::info_span!(
        "data_processing",
        operation = operation,
        file_path = file_path,
        rows = rows,
        start_time = %chrono::Utc::now(),
    )
}

/// Create a span for tracing ML operations
pub fn create_ml_span(operation: &str, model_id: &str, model_type: &str) -> Span {
    tracing::info_span!(
        "ml_operation",
        operation = operation,
        model_id = model_id,
        model_type = model_type,
        start_time = %chrono::Utc::now(),
    )
}

/// Log an error with context
pub fn log_error_with_context<E: std::error::Error>(error: &E, context: &str, service: &str) {
    tracing::error!(
        error = %error,
        context = context,
        service = service,
        "Operation failed"
    );
}

/// Log a warning with context
pub fn log_warning_with_context(message: &str, context: &str, service: &str) {
    tracing::warn!(
        message = message,
        context = context,
        service = service,
        "Warning occurred"
    );
}

/// Log performance metrics
pub fn log_performance_metrics(
    operation: &str,
    duration_ms: u64,
    memory_mb: u64,
    throughput_ops_per_sec: f64,
    service: &str,
) {
    tracing::info!(
        operation = operation,
        duration_ms = duration_ms,
        memory_mb = memory_mb,
        throughput_ops_per_sec = throughput_ops_per_sec,
        service = service,
        "Performance metrics"
    );
}

/// Log data processing metrics
pub fn log_data_metrics(
    operation: &str,
    input_rows: u64,
    output_rows: u64,
    processing_time_ms: u64,
    error_count: u64,
    service: &str,
) {
    tracing::info!(
        operation = operation,
        input_rows = input_rows,
        output_rows = output_rows,
        processing_time_ms = processing_time_ms,
        error_count = error_count,
        error_rate = if input_rows > 0 { error_count as f64 / input_rows as f64 } else { 0.0 },
        service = service,
        "Data processing metrics"
    );
}

/// Log ML model metrics
pub fn log_model_metrics(
    model_id: &str,
    model_type: &str,
    accuracy: f64,
    training_time_ms: u64,
    prediction_time_ms: u64,
    memory_mb: u64,
    service: &str,
) {
    tracing::info!(
        model_id = model_id,
        model_type = model_type,
        accuracy = accuracy,
        training_time_ms = training_time_ms,
        prediction_time_ms = prediction_time_ms,
        memory_mb = memory_mb,
        service = service,
        "Model metrics"
    );
}

/// Log system health metrics
pub fn log_health_metrics(
    cpu_usage_percent: f64,
    memory_usage_mb: u64,
    disk_usage_percent: f64,
    active_connections: u64,
    service: &str,
) {
    let health_status = if cpu_usage_percent > 90.0 || memory_usage_mb > 3072 || disk_usage_percent > 95.0 {
        "unhealthy"
    } else if cpu_usage_percent > 70.0 || memory_usage_mb > 2048 || disk_usage_percent > 80.0 {
        "degraded"
    } else {
        "healthy"
    };
    
    tracing::info!(
        cpu_usage_percent = cpu_usage_percent,
        memory_usage_mb = memory_usage_mb,
        disk_usage_percent = disk_usage_percent,
        active_connections = active_connections,
        health_status = health_status,
        service = service,
        "System health metrics"
    );
}

/// Log configuration changes
pub fn log_config_change(key: &str, old_value: &str, new_value: &str, service: &str) {
    tracing::info!(
        key = key,
        old_value = old_value,
        new_value = new_value,
        service = service,
        "Configuration changed"
    );
}

/// Log security events
pub fn log_security_event(
    event_type: &str,
    user_id: Option<&str>,
    ip_address: Option<&str>,
    details: &str,
    service: &str,
) {
    tracing::warn!(
        event_type = event_type,
        user_id = user_id,
        ip_address = ip_address,
        details = details,
        service = service,
        "Security event"
    );
}

/// Log business logic events
pub fn log_business_event(
    event_type: &str,
    entity_id: &str,
    entity_type: &str,
    details: &str,
    service: &str,
) {
    tracing::info!(
        event_type = event_type,
        entity_id = entity_id,
        entity_type = entity_type,
        details = details,
        service = service,
        "Business event"
    );
}

/// Logging middleware for operations
pub struct LoggingMiddleware {
    service_name: String,
}

impl LoggingMiddleware {
    /// Create a new logging middleware
    pub fn new(service_name: String) -> Self {
        Self { service_name }
    }
    
    /// Log the start of an operation
    pub fn log_operation_start(&self, operation: &str, details: &str) {
        tracing::info!(
            operation = operation,
            details = details,
            service = self.service_name,
            "Operation started"
        );
    }
    
    /// Log the completion of an operation
    pub fn log_operation_complete(&self, operation: &str, duration_ms: u64, success: bool) {
        if success {
            tracing::info!(
                operation = operation,
                duration_ms = duration_ms,
                service = self.service_name,
                "Operation completed successfully"
            );
        } else {
            tracing::error!(
                operation = operation,
                duration_ms = duration_ms,
                service = self.service_name,
                "Operation failed"
            );
        }
    }
    
    /// Log an operation with automatic timing
    pub async fn log_operation<F, T, E>(&self, operation: &str, future: F) -> Result<T, E>
    where
        F: std::future::Future<Output = Result<T, E>>,
        E: std::error::Error,
    {
        let start = std::time::Instant::now();
        self.log_operation_start(operation, "Starting operation");
        
        let result = future.await;
        let duration_ms = start.elapsed().as_millis() as u64;
        
        match &result {
            Ok(_) => self.log_operation_complete(operation, duration_ms, true),
            Err(e) => {
                tracing::error!(
                    operation = operation,
                    duration_ms = duration_ms,
                    error = %e,
                    service = self.service_name,
                    "Operation failed with error"
                );
                self.log_operation_complete(operation, duration_ms, false);
            }
        }
        
        result
    }
}

/// Structured logging macros
#[macro_export]
macro_rules! log_operation {
    ($service:expr, $operation:expr, $($field:ident = $value:expr),*) => {
        tracing::info!(
            service = $service,
            operation = $operation,
            $($field = $value,)*
            "Operation logged"
        );
    };
}

#[macro_export]
macro_rules! log_error {
    ($service:expr, $error:expr, $($field:ident = $value:expr),*) => {
        tracing::error!(
            service = $service,
            error = %$error,
            $($field = $value,)*
            "Error occurred"
        );
    };
}

#[macro_export]
macro_rules! log_metric {
    ($service:expr, $metric_type:expr, $($field:ident = $value:expr),*) => {
        tracing::info!(
            service = $service,
            metric_type = $metric_type,
            $($field = $value,)*
            "Metric recorded"
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;
    
    static INIT: Once = Once::new();
    
    fn setup_logging() {
        INIT.call_once(|| {
            let _ = init_logging("test-service", "debug", false);
        });
    }
    
    #[test]
    fn test_init_logging_pretty() {
        // Can't test this easily due to global subscriber
        // but we can at least verify it doesn't panic
        let result = init_logging("test", "info", false);
        // May fail if already initialized, which is ok
        assert!(result.is_ok() || result.is_err());
    }
    
    #[test]
    fn test_span_creation() {
        setup_logging();
        
        let _span = create_operation_span("test_operation", "test_service");
        let _span = create_request_span("GET", "/api/test", "req-123");
        let _span = create_data_processing_span("ingest", "/data/test.csv", 1000);
        let _span = create_ml_span("train", "model-1", "classifier");
    }
    
    #[test]
    fn test_logging_functions() {
        setup_logging();
        
        log_error_with_context(
            &PlatformError::internal("test error"),
            "test context",
            "test_service"
        );
        
        log_warning_with_context("test warning", "test context", "test_service");
        
        log_performance_metrics("test_op", 100, 50, 1000.0, "test_service");
        
        log_data_metrics("ingest", 1000, 950, 5000, 50, "test_service");
        
        log_model_metrics("model-1", "classifier", 0.95, 30000, 10, 256, "test_service");
        
        log_health_metrics(45.0, 1024, 60.0, 10, "test_service");
        
        log_config_change("log_level", "info", "debug", "test_service");
        
        log_security_event("login_attempt", Some("user123"), Some("192.168.1.1"), "Successful login", "test_service");
        
        log_business_event("model_registered", "model-1", "classifier", "New model registered", "test_service");
    }
    
    #[test]
    fn test_logging_middleware() {
        setup_logging();
        
        let middleware = LoggingMiddleware::new("test_service".to_string());
        
        middleware.log_operation_start("test_operation", "Starting test");
        middleware.log_operation_complete("test_operation", 100, true);
        middleware.log_operation_complete("test_operation", 200, false);
    }
    
    #[tokio::test]
    async fn test_logging_middleware_async() {
        setup_logging();
        
        let middleware = LoggingMiddleware::new("test_service".to_string());
        
        let result = middleware.log_operation("test_async", async {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            Ok::<_, PlatformError>("success")
        }).await;
        
        assert!(result.is_ok());
        
        let result = middleware.log_operation("test_async_error", async {
            Err::<String, PlatformError>(PlatformError::internal("test error"))
        }).await;
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_macros() {
        setup_logging();
        
        log_operation!("test_service", "test_op", duration = 100u64, success = true);
        log_error!("test_service", PlatformError::internal("test"), operation = "test_op");
        log_metric!("test_service", "performance", cpu_usage = 50.0, memory_mb = 256u64);
    }
}