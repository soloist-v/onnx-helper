use std::io::Result;
use protoc_prebuilt::init;
use std::env::set_var;

fn main() -> Result<()> {
    let (protoc_bin, _) = init("22.0").unwrap();
    set_var("PROTOC", protoc_bin);
    prost_build::compile_protos(&["src/onnx.proto3"], &["src/"])?;
    Ok(())
}