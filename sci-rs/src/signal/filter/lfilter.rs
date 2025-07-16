use alloc::vec::Vec;
use core::marker::Copy;
use ndarray::{
    Array, Array1, ArrayBase, ArrayView, ArrayView1, ArrayViewMut1, Axis, Data, Dim, IntoDimension,
    Ix, IxDyn, RemoveAxis, SliceInfo, SliceInfoElem,
};
use num_traits::{FromPrimitive, Num, NumAssign};
use sci_rs_core::{Error, Result};

/// /// Internal function for obtaining length of all axis as array from input from input.
///
/// This is almost the same as `a.shape()`, but is a array [T; N] instead of a Vec<T>.
///
/// # Parameters
/// `a`: Array whose shape is needed as a slice.
fn ndarray_ndim_as_array<'a, S, T, const N: usize>(a: &ArrayBase<S, Dim<[Ix; N]>>) -> [Ix; N]
where
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    T: FromPrimitive,
    S: Data<Elem = T> + 'a,
{
    let mut tmp = [0; N];
    (0..N).for_each(|axis| tmp[axis] = a.len_of(Axis(axis)));
    tmp
}

/// Internal function for casting into [Axis] and appropriate usize from isize.
///
/// # Parameters
/// axis: The user-specificed axis which filter is to be applied on.
/// x: The input-data whose axis object that will be manipulated against.
fn check_and_get_axis<'a, T, S, const N: usize>(
    axis: Option<isize>,
    x: &ArrayBase<S, Dim<[Ix; N]>>,
) -> Result<(Axis, usize)>
where
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    T: NumAssign + FromPrimitive + Copy + 'a,
    S: Data<Elem = T> + 'a,
{
    // Before we convert into the appropriate axis object, we have to check at runtime that the
    // axis value specified is within -N <= axis < N.
    if axis.is_some_and(|axis| {
        !(if axis < 0 {
            axis.unsigned_abs() <= N
        } else {
            axis.unsigned_abs() < N
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
        Ok((Axis(axis_inner as usize), axis_inner.unsigned_abs()))
    } else {
        let axis_inner = x
            .ndim()
            .checked_add_signed(axis_inner)
            .expect("Invalid add to `axis` option");
        Ok((Axis(axis_inner), axis_inner))
    }
}

/// Filter data along one-dimension with an IIR or FIR filter.
///
/// Filter a data sequence, `x`, using a digital filter.  This works for many
/// fundamental data types (including Object type).  The filter is a direct
/// form II transposed implementation of the standard difference equation
/// (see Notes).
///
/// The function [super::sosfilt] (and filter design using ``output='sos'``) should be
/// preferred over `lfilter` for most filtering tasks, as second-order sections
/// have fewer numerical problems.
///
/// ## Parameters
/// * `b` : array_like  
///   The numerator coefficient vector in a 1-D sequence.
/// * `a` : array_like  
///   The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
///   is not 1, then both `a` and `b` are normalized by ``a[0]``.
/// * `x` : array_like  
///   An N-dimensional input array.
/// * `axis`: Option<isize>
///   Default to `-1` if `None`.  
///   Panics in accordance with [ndarray::ArrayBase::axis_iter].
/// * `zi`: array_like  
///   Currently not implemented.  
///   Initial conditions for filter delays. It is a vector
///   (or array of vectors for an N-dimensional input) of length
///   ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
///   initial rest is assumed.  See `lfiltic` and [super::lfilter_zi] for more information.
///
/// ## Returns
/// * `y` : array  
///   The output of the digital filter.
/// * `zf` : array, optional  
///   If `zi` is None, this is not returned, otherwise, `zf` holds the
///   final filter delay values.
///
/// # See Also
/// * [super::lfilter_zi]  
///
/// # Notes
///
/// # Examples
/// On a 1-dimensional signal:
/// ```
/// use ndarray::{array, ArrayBase, Dim, Ix, OwnedRepr};
/// use sci_rs::signal::filter::lfilter;
///
/// let b = array![5., 4., 1., 2.];
/// let a = array![1.];
/// let x = array![1., 2., 3., 4., 3., 5., 6.];
/// let expected = array![5., 14., 24., 36., 38., 47., 61.];
/// let (result, _) = lfilter((&b).into(), (&a).into(), x, None, None).unwrap();
///
/// assert_eq!(result.len(), expected.len());
/// result.into_iter().zip(expected).for_each(|(r, e)| {
///     assert_eq!(r, e);
/// })
/// ```
///
/// # Panics
/// Currently yet to implement for `zi = Some(...)`, nor for `a.len() > 1`.
/// Panics if axis is out or range.
// NOTE: zi's TypeSig inherits from lfilter's output, in accordance with examples section of
// documentation, both lfilter_zi and this should eventually support NDArray.
pub fn lfilter<'a, T, S, const N: usize>(
    b: ArrayView1<'a, T>,
    a: ArrayView1<'a, T>,
    x: ArrayBase<S, Dim<[Ix; N]>>,
    axis: Option<isize>,
    zi: Option<Vec<T>>,
) -> Result<(Array<T, Dim<[Ix; N]>>, Option<Vec<T>>)>
where
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    T: NumAssign + FromPrimitive + Copy + 'a,
    S: Data<Elem = T> + 'a,
{
    if N == 0 {
        // `_validate_x` condition - ndarray allows for 0-dimensional arrays
        return Err(Error::InvalidArg {
            arg: "x".into(),
            reason: "Linear filter requires at least 1-dimensional `x`.".into(),
        });
    }

    if a.len() > 1 {
        unimplemented!()
    };

    let (axis, axis_inner) = check_and_get_axis(axis, &x)?;

    if a.is_empty() {
        return Err(Error::InvalidArg {
            arg: "a".into(),
            reason:
                "Empty 1D array will result in inf/nan result. Consider setting to `array![1.]`."
                    .into(),
        });
    } else if a.first().unwrap().is_zero() {
        return Err(Error::InvalidArg {
            arg: "a".into(),
            reason: "First element of a found to be zero.".into(),
        });
    }
    let b: Array1<T> = b.mapv(|bi| bi / a[0]);

    let (out_dim, out_dim_inner): (Dim<_>, [Ix; N]) = {
        let mut tmp: [Ix; N] = ndarray_ndim_as_array(&x);
        (IntoDimension::into_dimension(tmp), tmp)
    };
    let mut out = ArrayBase::zeros(out_dim);

    out.lanes_mut(axis)
        .into_iter()
        .zip(x.lanes(axis)) // Almost basically np.apply_along_axis
        .for_each(|(mut out_slice, y)| {
            // np.convolve uses full mode, but is eventually slices out with
            // ```py
            // ind = out_full.ndim * [slice(None)] # creates the "[:, :, ..., :]" slicer
            // ind[axis] = slice(out_full.shape[axis] - len(b) + 1) # [:out_full.shape[..] - len(b) + 1]
            // ```
            use sci_rs_core::num_rs::{convolve, ConvolveMode};
            let out_full = convolve(y, (&b).into(), ConvolveMode::Full).unwrap();
            out_full
                .slice(
                    SliceInfo::try_from([SliceInfoElem::Slice {
                        start: 0,
                        end: Some(out_dim_inner[axis_inner] as isize),
                        step: 1,
                    }])
                    .unwrap(),
                )
                .assign_to(&mut out_slice);
        });

    Ok((out, None))
}

#[cfg(test)]
mod test {
    use super::*;
    use alloc::vec;
    use approx::assert_relative_eq;
    use ndarray::{array, ArrayBase, Dim, Ix, OwnedRepr};

    // Tests that have a = [1.] with zi = None on input x with dim = 1.
    #[test]
    fn one_dim_fir_no_zi() {
        {
            // Tests for b.sum() > 1.
            let b = array![5., 4., 1., 2.];
            let a = array![1.];
            let x = array![1., 2., 3., 4., 3., 5., 6.];
            let expected = array![5., 14., 24., 36., 38., 47., 61.];

            let Ok((result, None)) = lfilter((&b).into(), (&a).into(), x, None, None) else {
                panic!("Should not have errored")
            };

            assert_eq!(result.len(), expected.len());
            result.into_iter().zip(expected).for_each(|(r, e)| {
                assert_eq!(r, e);
            })
        }
        {
            // Tests for b[i] < 0 for some i, such that b.sum() = 1.
            let b = array![0.7, -0.3, 0.6];
            let a = array![1.];
            let x = array![1., 2., 3., 4., 3., 5., 6.];
            let expected = array![0.7, 1.1, 2.1, 3.1, 2.7, 5., 4.5];

            let Ok((result, None)) = lfilter((&b).into(), (&a).into(), x, None, None) else {
                panic!("Should not have errored")
            };

            assert_eq!(result.len(), expected.len());
            result.into_iter().zip(expected).for_each(|(r, e)| {
                assert_relative_eq!(r, e, max_relative = 1e-6);
            })
        }
    }

    #[test]
    fn invalid_axis() {
        let b = array![5., 4., 1., 2.];
        let a = array![1.];
        let x = array![1., 2., 3., 4., 3., 5., 6.];

        let result = lfilter((&b).into(), (&a).into(), x.clone(), Some(2), None);
        assert!(result.is_err());

        let result = lfilter((&b).into(), (&a).into(), x.clone(), Some(1), None);
        assert!(result.is_err());

        let result = lfilter((&b).into(), (&a).into(), x.clone(), Some(0), None);
        assert!(result.is_ok());

        let result = lfilter((&b).into(), (&a).into(), x.clone(), Some(-1), None);
        assert!(result.is_ok());

        let result = lfilter((&b).into(), (&a).into(), x, Some(-2), None);
        assert!(result.is_err());
    }
}
