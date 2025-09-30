//! Functions for acting on a axis of an array.
//!
//! Designed for ndarrays; with scipy's internal nomenclature.

use alloc::{vec, vec::Vec};
use ndarray::{
    ArrayBase, ArrayView, Axis, Data, Dim, Dimension, IntoDimension, Ix, RemoveAxis, SliceArg,
    SliceInfo, SliceInfoElem,
};
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

/// Takes a slice along `axis` from `a`.
///
/// # Parameters
/// * `a`: Array being sliced from.
/// * `start`: `Option<isize>`. None defaults to 0.
/// * `end`: `Option<isize>`.
/// * `step`: `Option<isize>`. None default to 1.
/// * `axis`: `Option<isize>`. None defaults to -1.
///
/// # Errors
/// - Axis is out of bounds.
/// - Start/stop elements are out of bounds.
///
pub fn axis_slice<A, S, D>(
    a: &ArrayBase<S, D>,
    start: Option<isize>,
    end: Option<isize>,
    step: Option<isize>,
    axis: Option<isize>,
) -> Result<ArrayView<'_, A, D>>
where
    S: Data<Elem = A>,
    D: Dimension,
    SliceInfo<Vec<SliceInfoElem>, D, D>: SliceArg<D, OutDim = D>,
{
    if D::NDIM.is_none() {
        return Err(Error::InvalidArg {
            arg: 'a'.into(),
            reason: "IxDyn array is not supported".into(),
        });
    }
    #[allow(non_snake_case)]
    let N = D::NDIM.unwrap();

    // Axis object and its corresponding usize internal.
    let (axis, axis_inner) = {
        if axis.is_some_and(|axis| {
            !(if axis < 0 {
                axis.unsigned_abs() <= D::NDIM.unwrap()
            } else {
                axis.unsigned_abs() < D::NDIM.unwrap()
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
            (Axis(axis_inner as usize), axis_inner.unsigned_abs())
        } else {
            let axis_inner = a
                .ndim()
                .checked_add_signed(axis_inner)
                .expect("Invalid add to `axis` option");
            (Axis(axis_inner), axis_inner)
        }
    };

    let sl = SliceInfo::<_, D, D>::try_from({
        let mut tmp = vec![
            SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            };
            N
        ];
        tmp[axis_inner] = SliceInfoElem::Slice {
            start: start.unwrap_or(0),
            end,
            step: step.unwrap_or(1),
        };

        tmp
    })
    .unwrap();

    Ok(a.slice(&sl))
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    fn axis_slice_doc() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        assert_eq!(
            axis_slice(&a, Some(0), Some(1), Some(1), Some(1)).unwrap(),
            array![[1], [4], [7]]
        );
    }
}
