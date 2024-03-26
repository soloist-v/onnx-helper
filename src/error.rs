#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error(transparent)]
    EncodeError(#[from] prost::EncodeError),
    #[error(transparent)]
    DecodeError(#[from] prost::DecodeError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;