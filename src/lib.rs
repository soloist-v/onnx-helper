mod error;
mod helper;

pub use error::{Error, Result};
pub use helper::{OnnxHelper, InputInfo, OutputInfo};

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
