fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/generated")
        .compile(
            &["proto/data_ingestion.proto"],
            &["proto"]
        )?;
    
    println!("cargo:rerun-if-changed=proto/data_ingestion.proto");
    
    Ok(())
}