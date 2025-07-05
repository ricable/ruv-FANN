//! Common error handling for RAN Intelligence Platform Foundation Services

use std::fmt;
use thiserror::Error;

/// Result type for platform operations
pub type PlatformResult<T> = Result<T, PlatformError>;

/// Common error types across all foundation services
#[derive(Debug, Error)]
pub enum PlatformError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },
    
    /// Data processing errors
    #[error("Data processing error: {message}")]
    DataProcessing { message: String },
    
    /// I/O errors
    #[error("I/O error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },
    
    /// Serialization errors
    #[error("Serialization error: {source}")]
    Serialization {
        #[from]
        source: serde_json::Error,
    },
    
    /// Database errors
    #[error("Database error: {message}")]
    Database { message: String },
    
    /// gRPC errors
    #[error("gRPC error: {message}")]
    Grpc { message: String },
    
    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },
    
    /// Timeout errors
    #[error("Operation timed out after {timeout_secs} seconds")]
    Timeout { timeout_secs: u64 },
    
    /// Resource errors (memory, disk, etc.)
    #[error("Resource error: {message}")]
    Resource { message: String },
    
    /// ML model errors
    #[error("ML model error: {message}")]
    Model { message: String },
    
    /// Feature engineering errors
    #[error("Feature engineering error: {message}")]
    FeatureEngineering { message: String },
    
    /// Data ingestion errors
    #[error("Data ingestion error: {message}")]
    DataIngestion { message: String },
    
    /// Model registry errors
    #[error("Model registry error: {message}")]
    ModelRegistry { message: String },
    
    /// Generic internal errors
    #[error("Internal error: {message}")]
    Internal { message: String },
    
    /// External service errors
    #[error("External service error: {service}: {message}")]
    ExternalService { service: String, message: String },
    
    /// Authentication/authorization errors
    #[error("Authentication error: {message}")]
    Auth { message: String },
    
    /// Rate limiting errors
    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },
    
    /// Concurrent access errors
    #[error("Concurrent access error: {message}")]
    ConcurrentAccess { message: String },
    
    /// Generic errors with context
    #[error("Error in {context}: {source}")]
    WithContext {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl PlatformError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }
    
    /// Create a data processing error
    pub fn data_processing<S: Into<String>>(message: S) -> Self {
        Self::DataProcessing {
            message: message.into(),
        }
    }
    
    /// Create a database error
    pub fn database<S: Into<String>>(message: S) -> Self {
        Self::Database {
            message: message.into(),
        }
    }
    
    /// Create a gRPC error
    pub fn grpc<S: Into<String>>(message: S) -> Self {
        Self::Grpc {
            message: message.into(),
        }
    }
    
    /// Create a validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }
    
    /// Create a timeout error
    pub fn timeout(timeout_secs: u64) -> Self {
        Self::Timeout { timeout_secs }
    }
    
    /// Create a resource error
    pub fn resource<S: Into<String>>(message: S) -> Self {
        Self::Resource {
            message: message.into(),
        }
    }
    
    /// Create a model error
    pub fn model<S: Into<String>>(message: S) -> Self {
        Self::Model {
            message: message.into(),
        }
    }
    
    /// Create a feature engineering error
    pub fn feature_engineering<S: Into<String>>(message: S) -> Self {
        Self::FeatureEngineering {
            message: message.into(),
        }
    }
    
    /// Create a data ingestion error
    pub fn data_ingestion<S: Into<String>>(message: S) -> Self {
        Self::DataIngestion {
            message: message.into(),
        }
    }
    
    /// Create a model registry error
    pub fn model_registry<S: Into<String>>(message: S) -> Self {
        Self::ModelRegistry {
            message: message.into(),
        }
    }
    
    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
    
    /// Create an external service error
    pub fn external_service<S: Into<String>>(service: S, message: S) -> Self {
        Self::ExternalService {
            service: service.into(),
            message: message.into(),
        }
    }
    
    /// Create an auth error
    pub fn auth<S: Into<String>>(message: S) -> Self {
        Self::Auth {
            message: message.into(),
        }
    }
    
    /// Create a rate limit error
    pub fn rate_limit<S: Into<String>>(message: S) -> Self {
        Self::RateLimit {
            message: message.into(),
        }
    }
    
    /// Create a concurrent access error
    pub fn concurrent_access<S: Into<String>>(message: S) -> Self {
        Self::ConcurrentAccess {
            message: message.into(),
        }
    }
    
    /// Add context to an error
    pub fn with_context<S: Into<String>>(context: S, source: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::WithContext {
            context: context.into(),
            source,
        }
    }
    
    /// Check if this is a retryable error
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            PlatformError::Timeout { .. }
                | PlatformError::Resource { .. }
                | PlatformError::ExternalService { .. }
                | PlatformError::RateLimit { .. }
                | PlatformError::ConcurrentAccess { .. }
        )
    }
    
    /// Check if this is a client error (4xx equivalent)
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            PlatformError::Validation { .. }
                | PlatformError::Auth { .. }
                | PlatformError::Config { .. }
        )
    }
    
    /// Check if this is a server error (5xx equivalent)
    pub fn is_server_error(&self) -> bool {
        matches!(
            self,
            PlatformError::Internal { .. }
                | PlatformError::Database { .. }
                | PlatformError::Resource { .. }
                | PlatformError::ExternalService { .. }
        )
    }
}

/// Extension trait for Result to add context
pub trait ResultExt<T> {
    /// Add context to an error
    fn with_context<S: Into<String>>(self, context: S) -> PlatformResult<T>;
    
    /// Add context using a closure
    fn with_context_fn<F, S>(self, f: F) -> PlatformResult<T>
    where
        F: FnOnce() -> S,
        S: Into<String>;
}

impl<T, E> ResultExt<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_context<S: Into<String>>(self, context: S) -> PlatformResult<T> {
        self.map_err(|e| PlatformError::with_context(context, Box::new(e)))
    }
    
    fn with_context_fn<F, S>(self, f: F) -> PlatformResult<T>
    where
        F: FnOnce() -> S,
        S: Into<String>,
    {
        self.map_err(|e| PlatformError::with_context(f(), Box::new(e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = PlatformError::config("test config error");
        assert!(matches!(err, PlatformError::Config { .. }));
        assert_eq!(err.to_string(), "Configuration error: test config error");
    }
    
    #[test]
    fn test_error_retryable() {
        let timeout_err = PlatformError::timeout(30);
        assert!(timeout_err.is_retryable());
        
        let config_err = PlatformError::config("test");
        assert!(!config_err.is_retryable());
    }
    
    #[test]
    fn test_error_classification() {
        let validation_err = PlatformError::validation("test");
        assert!(validation_err.is_client_error());
        assert!(!validation_err.is_server_error());
        
        let internal_err = PlatformError::internal("test");
        assert!(!internal_err.is_client_error());
        assert!(internal_err.is_server_error());
    }
    
    #[test]
    fn test_result_ext() {
        let result: Result<i32, std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        
        let platform_result = result.with_context("reading config file");
        assert!(platform_result.is_err());
        
        let err = platform_result.unwrap_err();
        assert!(matches!(err, PlatformError::WithContext { .. }));
        assert!(err.to_string().contains("reading config file"));
    }
}