use alloc::vec::Vec;
use core::marker::Copy;
use ndarray::{
    Array, Array1, ArrayBase, ArrayView, ArrayView1, ArrayViewMut1, Axis, Data, Dim, IntoDimension,
    Ix, IxDyn, RemoveAxis, SliceInfo, SliceInfoElem,
};
use num_traits::{FromPrimitive, Num, NumAssign};

/// Internal function for de
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
/// let (result, _) = lfilter((&b).into(), (&a).into(), x, None, None);
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
#[cfg(feature = "alloc")]
pub fn lfilter<'a, T, S, const N: usize>(
    b: ArrayView1<'a, T>,
    a: ArrayView1<'a, T>,
    x: ArrayBase<S, Dim<[Ix; N]>>,
    axis: Option<isize>,
    zi: Option<Vec<T>>,
) -> (Array<T, Dim<[Ix; N]>>, Option<Vec<T>>)
where
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    T: NumAssign + FromPrimitive + Copy + core::fmt::Debug + 'a,
    S: Data<Elem = T> + 'a,
{
    if a.len() > 1 {
        unimplemented!()
    };
    if zi.is_some() {
        unimplemented!()
    };

    let mut x: ArrayBase<S, _> = x.into();
    // We make a best effort to convert into appropriate axis object.
    let (axis, axis_inner): (Axis, usize) = {
        let axis_inner: isize = axis.unwrap_or(-1).into();
        if axis_inner >= 0 {
            (Axis(axis_inner as usize), axis_inner as usize)
        } else {
            let axis_inner = (x.ndim() as isize + axis_inner) as usize;
            (Axis(axis_inner), axis_inner)
        }
    };

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
            // ind[axis] = slice(out_full.shape[axis] - len(b) + 1, None)
            // # [out_full.shape[..] - len(b) + 1 : None]
            // ```
            use sci_rs_core::num_rs::{convolve, ConvolveMode};
            let out_full = convolve(y, (&b).into(), ConvolveMode::Full).unwrap();
            let out_full_slice: ArrayView1<T> = out_full
                .slice(
                    SliceInfo::try_from([SliceInfoElem::Slice {
                        start: (out_dim_inner[axis_inner] - b.len()) as isize,
                        end: None,
                        step: 1,
                    }])
                    .unwrap(),
                )
                .reborrow();
            out_full_slice.assign_to(&mut out_slice);
        });

    (out, None)
}

#[cfg(test)]
mod test {
    use super::*;
    use alloc::vec;

    #[test]
    fn one_dim_no_zi() {
        use ndarray::{array, ArrayBase, Dim, Ix, OwnedRepr};
        let b = array![5., 4., 1., 2.];
        let a = array![1.];
        let x = array![1., 2., 3., 4., 3., 5., 6.];
        let expected = array![5., 14., 24., 36., 38., 47., 61.];

        let (result, _) = lfilter((&b).into(), (&a).into(), x, None, None);

        assert_eq!(result.len(), expected.len());
        result.into_iter().zip(expected).for_each(|(r, e)| {
            assert_eq!(r, e);
        })
    }
}
