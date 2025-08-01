use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    
    // Compile the proto files
    tonic_build::configure()
        .file_descriptor_set_path(out_dir.join("interference_classifier_descriptor.bin"))
        .compile(
            &["proto/interference_classifier.proto"],
            &["proto", "../../shared/proto/proto"],
        )?;
    
    Ok(())
}