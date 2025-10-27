//! Functions for acting on a axis of an array.
//!
//! Designed for ndarrays; with scipy's internal nomenclature.

use ndarray::{ArrayBase, Axis, Data, Dim, Dimension, IntoDimension, Ix, RemoveAxis};
use sci_rs_core::{Error, Result};

/// Internal function for casting into [Axis] and appropriate usize from isize.
///
/// # Parameters
/// axis: The user-specificed axis which filter is to be applied on.
/// x: The input-data whose axis object that will be manipulated against.
///
/// # Notes
/// Const nature of this function means error has to be manually created.
#[inline]
pub(crate) const fn check_and_get_axis_st<'a, T, S, const N: usize>(
    axis: Option<isize>,
    x: &ArrayBase<S, Dim<[Ix; N]>>,
) -> core::result::Result<usize, ()>
where
    S: Data<Elem = T> + 'a,
{
    // Before we convert into the appropriate axis object, we have to check at runtime that the
    // axis value specified is within -N <= axis < N.
    match axis {
        None => (),
        Some(axis) if axis.is_negative() => {
            if axis.unsigned_abs() > N {
                return Err(());
            }
        }
        Some(axis) => {
            if axis.unsigned_abs() >= N {
                return Err(());
            }
        }
    }

    // We make a best effort to convert into appropriate axis object.
    let axis_inner: isize = match axis {
        Some(axis) => axis,
        None => -1,
    };
    if axis_inner >= 0 {
        Ok(axis_inner.unsigned_abs())
    } else {
        let axis_inner = N
            .checked_add_signed(axis_inner)
            .expect("Invalid add to `axis` option");
        Ok(axis_inner)
    }
}

/// Internal function for casting into [Axis] and appropriate usize from isize.
/// [check_and_get_axis_st] but without const, especially for IxDyn arrays.
///
/// # Parameters
/// axis: The user-specificed axis which filter is to be applied on.
/// x: The input-data whose axis object that will be manipulated against.
#[inline]
pub(crate) fn check_and_get_axis_dyn<'a, T, S, D>(
    axis: Option<isize>,
    x: &ArrayBase<S, D>,
) -> Result<usize>
where
    D: Dimension,
    S: Data<Elem = T> + 'a,
{
    let ndim = D::NDIM.unwrap_or(x.ndim());
    // Before we convert into the appropriate axis object, we have to check at runtime that the
    // axis value specified is within -N <= axis < N.
    if axis.is_some_and(|axis| {
        !(if axis < 0 {
            axis.unsigned_abs() <= ndim
        } else {
            axis.unsigned_abs() < ndim
        })
    }) {
        return Err(Error::InvalidArg {
            arg: "axis".into(),
            reason: "index out of range.".into(),
        });
    }

    // We make a best effort to convert into appropriate axis object.
    let axis_inner: isize = axis.unwrap_or(-1);
    if axis_inner >= 0 {
        Ok(axis_inner.unsigned_abs())
    } else {
        let axis_inner = ndim
            .checked_add_signed(axis_inner)
            .expect("Invalid add to `axis` option");
        Ok(axis_inner)
    }
}

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
