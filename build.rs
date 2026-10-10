use std::io::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    prost_build::compile_protos(&["proto/pcs.proto"], &["proto/"])?;

    let include_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("include");
    std::fs::create_dir_all(include_dir.join("pcs"))?;
    std::fs::copy("proto/pcs.proto", include_dir.join("pcs/pcs.proto"))?;
    println!("cargo::metadata=proto_dir={}", include_dir.display());

    Ok(())
}
