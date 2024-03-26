use std::path::Path;
use prost::{Message};
use crate::onnx;
use crate::onnx::{OperatorSetIdProto, StringStringEntryProto, tensor_shape_proto};
use crate::onnx::tensor_shape_proto::dimension;
use crate::onnx::type_proto::Value;
use crate::Result;

pub struct OnnxHelper {
    model: onnx::ModelProto,
}

#[derive(Debug)]
pub struct InputInfo {
    pub name: String,
    pub shape: Vec<i64>,
    pub elem_type: i32,
}

#[derive(Debug)]
pub struct OutputInfo {
    pub name: String,
    pub shape: Vec<i64>,
    pub elem_type: i32,
}

fn proto_dims_to_vec(dims: &[tensor_shape_proto::Dimension]) -> Option<Vec<i64>> {
    let mut res = vec![];
    for x in dims {
        let d = match x.value.as_ref()? {
            dimension::Value::DimValue(a) => { *a }
            dimension::Value::DimParam(_) => { -1 }
        };
        res.push(d);
    }
    Some(res)
}

fn get_shape(val: &Value) -> Option<(Vec<i64>, i32)> {
    match val {
        Value::TensorType(a) => {
            Some((proto_dims_to_vec(a.shape.as_ref()?.dim.as_slice())?, a.elem_type))
        }
        Value::SparseTensorType(a) => {
            Some((proto_dims_to_vec(a.shape.as_ref()?.dim.as_slice())?, a.elem_type))
        }
        _ => { None }
    }
}

impl OnnxHelper {
    pub fn new(data: &[u8]) -> Result<Self> {
        Ok(Self {
            model: onnx::ModelProto::decode(data)?,
        })
    }

    pub fn with_path(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read(path)?;
        Self::new(data.as_slice())
    }
}

impl OnnxHelper {
    pub fn model_version(&self) -> i64 {
        self.model.model_version
    }

    pub fn metadata_props(&self) -> &[StringStringEntryProto] {
        self.model.metadata_props.as_slice()
    }

    pub fn producer_name(&self) -> &str {
        self.model.producer_name.as_str()
    }

    pub fn producer_version(&self) -> &str {
        self.model.producer_version.as_str()
    }

    pub fn opset_import(&self) -> &[OperatorSetIdProto] {
        self.model.opset_import.as_slice()
    }

    pub fn input_shape(&self, name: &str) -> Option<Vec<i64>> {
        for (i, info) in self.model.graph.as_ref()?.input.iter().enumerate() {
            if info.name == name {
                return self.input_shape_with_idx(i);
            }
        }
        None
    }

    pub fn input_shape_with_idx(&self, idx: usize) -> Option<Vec<i64>> {
        let info = self.model.graph.as_ref()?.input.get(idx)?;
        Some(get_shape(info.r#type.as_ref()?.value.as_ref()?)?.0)
    }

    pub fn input_shapes(&self) -> Option<Vec<Vec<i64>>> {
        let len = self.model.graph.as_ref()?.input.len();
        let mut res = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.input_shape_with_idx(i)?;
            res.push(a);
        }
        Some(res)
    }

    pub fn output_shape(&self, name: &str) -> Option<Vec<i64>> {
        for (i, info) in self.model.graph.as_ref()?.output.iter().enumerate() {
            if info.name == name {
                return self.output_shape_with_idx(i);
            }
        }
        None
    }

    pub fn output_shape_with_idx(&self, idx: usize) -> Option<Vec<i64>> {
        let info = self.model.graph.as_ref()?.output.get(idx)?;
        Some(get_shape(info.r#type.as_ref()?.value.as_ref()?)?.0)
    }

    pub fn output_shapes(&self) -> Option<Vec<Vec<i64>>> {
        let len = self.model.graph.as_ref()?.output.len();
        let mut res = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.output_shape_with_idx(i)?;
            res.push(a);
        }
        Some(res)
    }

    pub fn inputs(&self) -> Option<Vec<InputInfo>> {
        let mut res = vec![];
        for info in self.model.graph.as_ref()?.input.iter() {
            let (shape, elem_type) = get_shape(info.r#type.as_ref()?.value.as_ref()?)?;
            res.push(InputInfo {
                name: info.name.clone(),
                shape,
                elem_type,
            });
        }
        Some(res)
    }

    pub fn outputs(&self) -> Option<Vec<OutputInfo>> {
        let mut res = vec![];
        for info in self.model.graph.as_ref()?.output.iter() {
            let (shape, elem_type) = get_shape(info.r#type.as_ref()?.value.as_ref()?)?;
            res.push(OutputInfo {
                name: info.name.clone(),
                shape,
                elem_type,
            });
        }
        Some(res)
    }
}

#[test]
pub fn test() {
    let path = r"D:\Workspace\Rust\binary\WZSB-Nitro\data\model\jzq.onnx";
    let model = OnnxHelper::with_path(path).unwrap();
    println!("shape: {:?}", model.input_shape_with_idx(0));
    println!("shape: {:?}", model.output_shape_with_idx(0));
    println!("shape: {:?}", model.input_shapes());
    println!("shape: {:?}", model.output_shapes());
}