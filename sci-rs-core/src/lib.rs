//! Core library for sci-rs.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::format;

use core::{error, fmt};

pub type Result<T> = core::result::Result<T, Error>;

/// Errors raised whilst running sci-rs.
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// Argument parsed into function were invalid.
    #[cfg(feature = "alloc")]
    InvalidArg {
        /// The invalid arg
        arg: alloc::string::String,
        /// Explaining why arg is invalid.
        reason: alloc::string::String,
    },
    /// Argument parsed into function were invalid.
    #[cfg(not(feature = "alloc"))]
    InvalidArg,
    /// Two or more optional arguments passed into functions conflict.
    #[cfg(feature = "alloc")]
    ConflictArg {
        /// Explaining what arg is invalid.
        reason: alloc::string::String,
    },
    /// Two or more optional arguments passed into functions conflict.
    #[cfg(not(feature = "alloc"))]
    ConflictArg,
    /// Errors raised by [ndarray_conv::Error]
    #[cfg(feature = "alloc")]
    Conv { reason: alloc::string::String },
    /// Errors raised by [ndarray_conv::Error]
    #[cfg(not(feature = "alloc"))]
    Conv,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                #[cfg(feature = "alloc")]
                Error::InvalidArg { arg, reason } =>
                    format!("Invalid Argument on arg = {} with reason = {}", arg, reason),
                #[cfg(not(feature = "alloc"))]
                Error::InvalidArg =>
                    "There were invalid arguments. Reasons not shown without `alloc` feature.",
                #[cfg(feature = "alloc")]
                Error::ConflictArg { reason } =>
                    format!("Conflicting Arguments with reason = {}", reason),
                #[cfg(not(feature = "alloc"))]
                Error::ConflictArg =>
                    "There were conflicting arguments. Reasons not shown without `alloc` feature.",
                #[cfg(feature = "alloc")]
                Error::Conv { reason } => format!(
                    "An error occurred during the convolution from ndarray_conv with reason {}.",
                    reason
                ),
                #[cfg(not(feature = "alloc"))]
                Error::Conv => "An error occurred during the convolution from ndarray_conv. Reasons not shown without `alloc` feature.",
            }
        )
    }
}

impl error::Error for Error {}

/// Collection of numpy-like functions for use by sci-rs.
/// Provide behaviour parity against Numpy, even if the types are not identical.
pub mod num_rs;
