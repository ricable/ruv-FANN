//! Generated protobuf definitions for PFS-DATA service

#[rustfmt::skip]
#[allow(unused_imports, missing_docs, clippy::all)]
pub mod pfs {
    pub mod data {
        pub mod v1 {
            include!("data_ingestion.rs");
        }
    }
}

// Re-export commonly used types
pub use pfs::data::v1::*;