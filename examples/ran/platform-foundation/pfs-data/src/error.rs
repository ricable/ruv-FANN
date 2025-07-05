//! Error handling for PFS-DATA service

use thiserror::Error;
use platform_foundation_common::PlatformError;

/// Result type for data ingestion operations
pub type DataIngestionResult<T> = Result<T, DataIngestionError>;

/// Error types for data ingestion service
#[derive(Debug, Error)]
pub enum DataIngestionError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },
    
    /// File processing errors
    #[error("File processing error: {message}")]
    FileProcessing { message: String },
    
    /// Schema validation errors
    #[error("Schema validation error: {message}")]
    SchemaValidation { message: String },
    
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
    
    /// CSV parsing errors
    #[error("CSV parsing error: {source}")]
    CsvParsing {
        #[from]
        source: csv::Error,
    },
    
    /// Parquet errors
    #[error("Parquet error: {source}")]
    Parquet {
        #[from]
        source: parquet::errors::ParquetError,
    },
    
    /// Arrow errors
    #[error("Arrow error: {source}")]
    Arrow {
        #[from]
        source: arrow::error::ArrowError,
    },
    
    /// Polars errors
    #[error("Polars error: {source}")]
    Polars {
        #[from]
        source: polars::error::PolarsError,
    },
    
    /// File watching errors
    #[error("File watching error: {source}")]
    FileWatching {
        #[from]
        source: notify::Error,
    },
    
    /// gRPC errors
    #[error("gRPC error: {source}")]
    Grpc {
        #[from]
        source: tonic::Status,
    },
    
    /// Task joining errors
    #[error("Task join error: {source}")]
    TaskJoin {
        #[from]
        source: tokio::task::JoinError,
    },
    
    /// Timeout errors
    #[error("Operation timed out after {timeout_secs} seconds")]
    Timeout { timeout_secs: u64 },
    
    /// Resource errors (memory, disk, etc.)
    #[error("Resource error: {message}")]
    Resource { message: String },
    
    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },
    
    /// Ingestion engine errors
    #[error("Ingestion engine error: {message}")]
    IngestionEngine { message: String },
    
    /// File format errors
    #[error("Unsupported file format: {format}")]
    UnsupportedFormat { format: String },
    
    /// Data quality errors
    #[error("Data quality error: {message}, error_rate: {error_rate:.2}%")]
    DataQuality { message: String, error_rate: f64 },
    
    /// Concurrent access errors
    #[error("Concurrent access error: {message}")]
    ConcurrentAccess { message: String },
    
    /// Generic internal errors
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl DataIngestionError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }
    
    /// Create a file processing error
    pub fn file_processing<S: Into<String>>(message: S) -> Self {
        Self::FileProcessing {
            message: message.into(),
        }
    }
    
    /// Create a schema validation error
    pub fn schema_validation<S: Into<String>>(message: S) -> Self {
        Self::SchemaValidation {
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
    
    /// Create a validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }
    
    /// Create an ingestion engine error
    pub fn ingestion_engine<S: Into<String>>(message: S) -> Self {
        Self::IngestionEngine {
            message: message.into(),
        }
    }
    
    /// Create an unsupported format error
    pub fn unsupported_format<S: Into<String>>(format: S) -> Self {
        Self::UnsupportedFormat {
            format: format.into(),
        }
    }
    
    /// Create a data quality error
    pub fn data_quality<S: Into<String>>(message: S, error_rate: f64) -> Self {
        Self::DataQuality {
            message: message.into(),
            error_rate,
        }
    }
    
    /// Create a concurrent access error
    pub fn concurrent_access<S: Into<String>>(message: S) -> Self {
        Self::ConcurrentAccess {
            message: message.into(),
        }
    }
    
    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
    
    /// Check if this is a retryable error
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            DataIngestionError::Timeout { .. }
                | DataIngestionError::Resource { .. }
                | DataIngestionError::ConcurrentAccess { .. }
                | DataIngestionError::Io { .. }
        )
    }
    
    /// Check if this is a client error (4xx equivalent)
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            DataIngestionError::Validation { .. }
                | DataIngestionError::SchemaValidation { .. }
                | DataIngestionError::UnsupportedFormat { .. }
                | DataIngestionError::Config { .. }
        )
    }
    
    /// Check if this is a server error (5xx equivalent)
    pub fn is_server_error(&self) -> bool {
        matches!(
            self,
            DataIngestionError::Internal { .. }
                | DataIngestionError::Resource { .. }
                | DataIngestionError::IngestionEngine { .. }
        )
    }
    
    /// Convert to tonic::Status for gRPC responses
    pub fn to_grpc_status(&self) -> tonic::Status {
        let code = if self.is_client_error() {
            tonic::Code::InvalidArgument
        } else if self.is_server_error() {
            tonic::Code::Internal
        } else {
            tonic::Code::Unknown
        };
        
        tonic::Status::new(code, self.to_string())
    }
}

/// Convert from PlatformError
impl From<PlatformError> for DataIngestionError {
    fn from(err: PlatformError) -> Self {
        match err {
            PlatformError::Config { message } => DataIngestionError::config(message),
            PlatformError::DataProcessing { message } => DataIngestionError::file_processing(message),
            PlatformError::Validation { message } => DataIngestionError::validation(message),
            PlatformError::Timeout { timeout_secs } => DataIngestionError::timeout(timeout_secs),
            PlatformError::Resource { message } => DataIngestionError::resource(message),
            PlatformError::DataIngestion { message } => DataIngestionError::ingestion_engine(message),
            PlatformError::ConcurrentAccess { message } => DataIngestionError::concurrent_access(message),
            PlatformError::Internal { message } => DataIngestionError::internal(message),
            PlatformError::Io { source } => DataIngestionError::Io { source },
            PlatformError::Serialization { source } => DataIngestionError::Serialization { source },
            _ => DataIngestionError::internal(err.to_string()),
        }
    }
}

/// Convert to PlatformError
impl From<DataIngestionError> for PlatformError {
    fn from(err: DataIngestionError) -> Self {
        match err {
            DataIngestionError::Config { message } => PlatformError::config(message),
            DataIngestionError::FileProcessing { message } => PlatformError::data_processing(message),
            DataIngestionError::Validation { message } => PlatformError::validation(message),
            DataIngestionError::Timeout { timeout_secs } => PlatformError::timeout(timeout_secs),
            DataIngestionError::Resource { message } => PlatformError::resource(message),
            DataIngestionError::IngestionEngine { message } => PlatformError::data_ingestion(message),
            DataIngestionError::ConcurrentAccess { message } => PlatformError::concurrent_access(message),
            DataIngestionError::Internal { message } => PlatformError::internal(message),
            DataIngestionError::Io { source } => PlatformError::Io { source },
            DataIngestionError::Serialization { source } => PlatformError::Serialization { source },
            _ => PlatformError::internal(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = DataIngestionError::config("test config error");
        assert!(matches!(err, DataIngestionError::Config { .. }));
        assert_eq!(err.to_string(), "Configuration error: test config error");
    }
    
    #[test]
    fn test_error_retryable() {
        let timeout_err = DataIngestionError::timeout(30);
        assert!(timeout_err.is_retryable());
        
        let config_err = DataIngestionError::config("test");
        assert!(!config_err.is_retryable());
    }
    
    #[test]
    fn test_error_classification() {
        let validation_err = DataIngestionError::validation("test");
        assert!(validation_err.is_client_error());
        assert!(!validation_err.is_server_error());
        
        let internal_err = DataIngestionError::internal("test");
        assert!(!internal_err.is_client_error());
        assert!(internal_err.is_server_error());
    }
    
    #[test]
    fn test_grpc_status_conversion() {
        let validation_err = DataIngestionError::validation("test");
        let status = validation_err.to_grpc_status();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        
        let internal_err = DataIngestionError::internal("test");
        let status = internal_err.to_grpc_status();
        assert_eq!(status.code(), tonic::Code::Internal);
    }
    
    #[test]
    fn test_platform_error_conversion() {
        let platform_err = PlatformError::config("test");
        let ingestion_err: DataIngestionError = platform_err.into();
        assert!(matches!(ingestion_err, DataIngestionError::Config { .. }));
        
        let ingestion_err = DataIngestionError::config("test");
        let platform_err: PlatformError = ingestion_err.into();
        assert!(matches!(platform_err, PlatformError::Config { .. }));
    }
}