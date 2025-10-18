//! Functions for acting on a axis of an array.
//!
//! Designed for ndarrays; with scipy's internal nomenclature.

use ndarray::{ArrayBase, Axis, Data, Dim, Dimension, IntoDimension, Ix, RemoveAxis};
use sci_rs_core::{Error, Result};

/// Internal function for obtaining length of all axis as array from input from input.
///
/// This is almost the same as `a.shape()`, but is a array `[T; N]` instead of a slice `&[T]`.
///
/// # Parameters
/// `a`: Array whose shape is needed as a slice.
pub(crate) fn ndarray_shape_as_array_st<'a, S, T, const N: usize>(
    a: &ArrayBase<S, Dim<[Ix; N]>>,
) -> [Ix; N]
where
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    S: Data<Elem = T> + 'a,
{
    a.shape().try_into().expect("Could not cast shape to array")
}
