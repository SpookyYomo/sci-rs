use alloc::vec::Vec;
use core::marker::Copy;
use ndarray::{
    Array, Array1, ArrayBase, ArrayView, ArrayView1, ArrayViewMut1, Axis, Data, Dim, IntoDimension,
    Ix, IxDyn, RemoveAxis, ShapeBuilder, SliceInfo, SliceInfoElem,
};
use num_traits::{FromPrimitive, Num, NumAssign};
use sci_rs_core::{Error, Result};

/// Internal function for obtaining length of all axis as array from input from input.
///
/// This is almost the same as `a.shape()`, but is a array [T; N] instead of a Vec<T>.
///
/// # Parameters
/// `a`: Array whose shape is needed as a slice.
fn ndarray_shape_as_array<'a, S, T, const N: usize>(a: &ArrayBase<S, Dim<[Ix; N]>>) -> [Ix; N]
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

/// Implement lfilter for fixed dimension of input array `x`.
///
/// Valid only from 1 to 6 dimensional arrays.
pub trait LFilter<S, const N: usize>
where
    S: ndarray::RawData,
{
    /// Filter data `x` along one-dimension with an IIR or FIR filter.
    ///
    /// Filter a data sequence, `x`, using a digital filter.  This works for many
    /// fundamental data types (including Object type).  The filter is a direct
    /// form II transposed implementation of the standard difference equation
    /// (see Notes).
    ///
    /// The function [super::sosfilt_dyn] (and filter design using ``output='sos'``) should be
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
    /// * `axis`: `Option<isize>`
    ///   Default to `-1` if `None`.  
    ///   Panics in accordance with [ndarray::ArrayBase::axis_iter].
    /// * `zi`: array_like  
    ///   Currently not implemented.  
    ///   Initial conditions for filter delays. It is a vector
    ///   (or array of vectors for an N-dimensional input) of length
    ///   ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
    ///   initial rest is assumed.  See `lfiltic` and [super::lfilter_zi_dyn] for more information.
    ///
    /// ## Returns
    /// * `y` : array  
    ///   The output of the digital filter.
    /// * `zf` : array, optional  
    ///   If `zi` is None, this is not returned, otherwise, `zf` holds the
    ///   final filter delay values.
    ///
    /// # See Also
    /// * [super::lfilter_zi_dyn]  
    ///
    /// # Notes
    /// For compile time reasons, lfilter is implemented per ArrayN at the moment.
    ///
    /// # Examples
    /// On a 1-dimensional signal:
    /// ```
    /// use ndarray::{array, ArrayBase, Array1, ArrayView1, Dim, Ix, OwnedRepr};
    /// use sci_rs::signal::filter::LFilter;
    ///
    /// let b = array![5., 4., 1., 2.];
    /// let a = array![1.];
    /// let x = array![1., 2., 3., 4., 3., 5., 6.];
    /// let expected = array![5., 14., 24., 36., 38., 47., 61.];
    /// let (result, _) = ArrayView1::lfilter((&b).into(), (&a).into(), (&x).into(), None, None).unwrap(); // By ref
    ///
    /// assert_eq!(result.len(), expected.len());
    /// result.into_iter().zip(expected).for_each(|(r, e)| {
    ///     assert_eq!(r, e);
    /// });
    ///
    /// let (result, _) = Array1::lfilter((&b).into(), (&a).into(), x, None, None).unwrap(); // By value
    /// ```
    ///
    /// # Panics
    /// Currently yet to implement for `a.len() > 1`.
    // NOTE: zi's TypeSig inherits from lfilter's output, in accordance with examples section of
    // documentation, both lfilter_zi and this should eventually support NDArray.
    fn lfilter<'a, T>(
        b: ArrayView1<'a, T>,
        a: ArrayView1<'a, T>,
        x: Self,
        axis: Option<isize>,
        zi: Option<ArrayView<T, Dim<[Ix; N]>>>,
    ) -> Result<(Array<T, Dim<[Ix; N]>>, Option<Array<T, Dim<[Ix; N]>>>)>
    where
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
        Dim<[Ix; N]>: RemoveAxis,
        T: NumAssign + FromPrimitive + Copy + 'a,
        S: Data<Elem = T> + 'a;
}

macro_rules! lfilter_for_dim {
    ($N:literal) => {
        impl<S> LFilter<S, $N> for ArrayBase<S, Dim<[Ix; $N]>>
        where
            S: ndarray::RawData,
        {
            fn lfilter<'a, T>(
                b: ArrayView1<'a, T>,
                a: ArrayView1<'a, T>,
                x: Self,
                axis: Option<isize>,
                zi: Option<ArrayView<T, Dim<[Ix; $N]>>>,
            ) -> Result<(Array<T, Dim<[Ix; $N]>>, Option<Array<T, Dim<[Ix; $N]>>>)>
            where
                [Ix; $N]: IntoDimension<Dim = Dim<[Ix; $N]>>,
                Dim<[Ix; $N]>: RemoveAxis,
                T: NumAssign + FromPrimitive + Copy + 'a,
                S: Data<Elem = T> + 'a,
            {
                if a.len() > 1 {
                    return linear_filter(b, a, x, axis, zi);
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
                let b: Array1<T> = b.mapv(|bi| bi / a[0]); // b /= a[0]

                if let Some(zi) = zi {
                    // Use a separate branch to avoid unnecessary heap allocation of `out_full` in `zi` = None
                    // case.
                    let mut zi = zi.to_owned();

                    // if zi.ndim != x.ndim { return Err(...) } is signature asserted.

                    let mut expected_shape: [usize; $N] = x.shape().try_into().unwrap();
                    *expected_shape // expected_shape[axis] = b.shape[0] - 1
                        .get_mut(axis_inner)
                        .expect("invalid axis_inner") = b
                        .shape()
                        .first()
                        .expect("Could not get 0th axis len of b")
                        .checked_sub(1)
                        .expect("underflowing subtract");

                    if *zi.shape() != expected_shape {
                        let strides: [Ix; $N] = {
                            let zi_shape = zi.shape();
                            let zi_strides = zi.strides();

                            // Waiting for try_collect() from nightly... we use this Vec<Result<>> -> Result<Vec<>> method..
                            let tmp_heap: Vec<Result<_>> = (0..$N)
                                .map(|k| {
                                    if zi_shape[k] == expected_shape[k] {
                                        zi_strides[k].try_into().map_err(|_| Error::InvalidArg {
                                            arg: "zi".into(),
                                            reason: "zi found with negative stride".into(),
                                        })
                                    } else if k != axis_inner && zi_shape[k] == 1 {
                                        Ok(0)
                                    } else {
                                        Err(Error::InvalidArg {
                                            arg: "zi".into(),
                                            reason: "Unexpected shape for parameter zi".into(),
                                        })
                                    }
                                })
                                .collect();
                            let tmp_heap: Result<Vec<Ix>> = tmp_heap.into_iter().collect();

                            tmp_heap?.try_into().unwrap()
                        };

                        zi = ArrayView::from_shape(expected_shape.strides(strides), zi.as_slice().unwrap())
                            .unwrap()
                            .to_owned();
                    };

                    let (out_full_dim, out_full_dim_inner): (Dim<_>, [Ix; $N]) = {
                        let mut tmp: [Ix; $N] = ndarray_shape_as_array(&x);
                        tmp[axis_inner] += b.len_of(Axis(0)) - 1; // From np.convolve(..., 'full')
                        (IntoDimension::into_dimension(tmp), tmp)
                    };

                    let mut out_full: Array<T, Dim<[Ix; $N]>> = ArrayBase::zeros(out_full_dim);
                    out_full
                        .lanes_mut(axis)
                        .into_iter()
                        .zip(x.lanes(axis)) // Almost basically np.apply_along_axis
                        .try_for_each(|(mut out_full_slice, y)| {
                            // np.convolve uses full mode by default
                            // ```py
                            // out_full = np.apply_along_axis(lambda y: np.convolve(b, y), axis, x)
                            // ```
                            use sci_rs_core::num_rs::{convolve, ConvolveMode};
                            convolve(y, (&b).into(), ConvolveMode::Full)?
                                .assign_to(&mut out_full_slice);
                            Ok(())
                        })?;

                    // ```py
                    // ind[axis] = slice(zi.shape[axis])
                    // out_full[tuple(ind)] += zi
                    // ```
                    {
                        let slice_info: SliceInfo<_, Dim<[Ix; $N]>, Dim<[Ix; $N]>> = {
                            let t = zi.shape()[axis_inner];
                            let mut tmp = [SliceInfoElem::from(..); $N];
                            tmp[axis_inner] = SliceInfoElem::Slice {
                                start: 0,
                                end: Some(t as isize),
                                step: 1,
                            };

                            SliceInfo::try_from(tmp).unwrap()
                        }; // Does not work because unless N: N<=6 cannot be bounded on type_sig
                        let mut s = out_full.slice_mut(&slice_info);
                        s += &zi;
                    }

                    let (out_dim, out_dim_inner) = {
                        let tmp: [Ix; $N] = ndarray_shape_as_array(&x);
                        (IntoDimension::into_dimension(tmp), tmp)
                    };
                    let mut out = ArrayBase::zeros(out_dim);
                    out.lanes_mut(axis)
                        .into_iter()
                        .zip(out_full.lanes(axis))
                        .for_each(|(mut out_slice, out_full_slice)| {
                            // ```py
                            // # Create the [...; :out_full.shape[axis] - len(b) + 1; ...] at index=axis
                            // ind[axis] = slice(out_full.shape[axis] - len(b) + 1)
                            // out = out_full[tuple(ind)]
                            // ```
                            out_full_slice
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

                    // ```py
                    // ind[axis] = slice(out_full.shape[axis] - len(b) + 1, None)
                    // zf = out_full[tuple(ind)]
                    // ```
                    let zf = {
                        let slice_info: SliceInfo<_, Dim<[Ix; $N]>, Dim<[Ix; $N]>> = {
                            let t = out_full.shape()[axis_inner]
                                .checked_add(1)
                                .unwrap()
                                .checked_sub(b.len())
                                .unwrap();
                            let mut tmp = [SliceInfoElem::from(..); $N];
                            tmp[axis_inner] = SliceInfoElem::Slice {
                                start: t as isize,
                                end: None,
                                step: 1,
                            };

                            SliceInfo::try_from(tmp).unwrap()
                        };
                        out_full.slice(slice_info).to_owned()
                    };

                    Ok((out, Some(zf)))
                } else {
                    // In contrast to the case where zi.is_some(), we can inline a slicing operation to reduce
                    // one extra heap allocation.

                    let (out_dim, out_dim_inner): (Dim<_>, [Ix; $N]) = {
                        let mut tmp: [Ix; $N] = ndarray_shape_as_array(&x);
                        (IntoDimension::into_dimension(tmp), tmp)
                    };
                    let mut out = ArrayBase::zeros(out_dim);

                    out.lanes_mut(axis)
                        .into_iter()
                        .zip(x.lanes(axis)) // Almost basically np.apply_along_axis
                        .try_for_each(|(mut out_slice, y)| {
                            // np.convolve uses full mode, but is eventually slices out with
                            // ```py
                            // ind = out_full.ndim * [slice(None)] # creates the "[:, :, ..., :]" slice r
                            // ind[axis] = slice(out_full.shape[axis] - len(b) + 1) # [:out_full.shape[ ..] - len(b) + 1]
                            // ```
                            use sci_rs_core::num_rs::{convolve, ConvolveMode};
                            let out_full = convolve(y, (&b).into(), ConvolveMode::Full)?;
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
                            Ok(())
                        })?;

                    Ok((out, None))
                }
            }
        }
    };
}

/// Internal function called by [lfilter] for situation a.len() > 1.
fn linear_filter<'a, T, S, const N: usize>(
    b: ArrayView1<'a, T>,
    a: ArrayView1<'a, T>,
    x: ArrayBase<S, Dim<[Ix; N]>>,
    axis: Option<isize>,
    zi: Option<ArrayView<T, Dim<[Ix; N]>>>,
) -> Result<(Array<T, Dim<[Ix; N]>>, Option<Array<T, Dim<[Ix; N]>>>)>
where
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: RemoveAxis,
    T: NumAssign + FromPrimitive + Copy + 'a,
    S: Data<Elem = T> + 'a,
{
    todo!()
}

lfilter_for_dim!(1);
lfilter_for_dim!(2);
lfilter_for_dim!(3);
lfilter_for_dim!(4);
lfilter_for_dim!(5);
lfilter_for_dim!(6);

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

            let Ok((result, None)) = Array1::lfilter((&b).into(), (&a).into(), x, None, None)
            else {
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

            let Ok((result, None)) = Array1::lfilter((&b).into(), (&a).into(), x, None, None)
            else {
                panic!("Should not have errored")
            };

            assert_eq!(result.len(), expected.len());
            result.into_iter().zip(expected).for_each(|(r, e)| {
                assert_relative_eq!(r, e, max_relative = 1e-6);
            })
        }
    }

    #[test]
    fn one_dim_fir_with_zi() {
        {
            // Case which does not falls into zi.shape() != expected_shape branch
            let b = array![0.5, 0.4];
            let a = array![1.];
            let x = array![
                [-4., -3., -1., -2., 1., 2., -3., 4., 3., 5., 6., 7., -8., 1.],
                [-4., -3., -1., -2., 1., 2., -3., 4., 3., 5., 6., 7., -8., 1.],
            ];
            let zi = array![[-1.6], [1.4]];
            let expected = array![
                [-3.6, -3.1, -1.7, -1.4, -0.3, 1.4, -0.7, 0.8, 3.1, 3.7, 5., 5.9, -1.2, -2.7],
                [-0.6, -3.1, -1.7, -1.4, -0.3, 1.4, -0.7, 0.8, 3.1, 3.7, 5., 5.9, -1.2, -2.7]
            ];
            let expected_zi = array![[0.4], [0.4]];

            let Ok((result, Some(r_zi))) = Array::<_, Dim<[Ix; 2]>>::lfilter(
                (&b).into(),
                (&a).into(),
                x,
                None,
                Some((&zi).into()),
            ) else {
                panic!("Should not have errored")
            };

            assert_eq!(result.len(), expected.len());
            result.into_iter().zip(expected).for_each(|(r, e)| {
                assert_relative_eq!(r, e, max_relative = 1e-6);
            });
            assert_eq!(r_zi.len(), expected_zi.len());
            r_zi.into_iter().zip(expected_zi).for_each(|(r, e)| {
                assert_relative_eq!(r, e, max_relative = 1e-6);
            })
        }
        {
            // Case which does falls into zi.shape() != expected_shape branch
            let b = array![5., 0.4, 1., -2.];
            let a = array![1.];
            let x = array![[1., 2., 3., 4., 3., 5., 6.], [8., 0., 1., 0., 3., 7., 6.]];
            let zi = array![[0.4], [0.45], [0.05]];
            let expected = array![
                [5.4, 10.4, 15.4, 20.4, 15.4, 25.4, 30.4],
                [40.85, 1.25, 6.65, 2.05, 16.65, 37.45, 32.85],
            ];
            let expected_zi = array![
                [4.25, 2.05, 3.45, 4.05, 4.25, 7.85, 8.45],
                [6., -4., -5., -8., -3., -3., -6.],
                [-16., 0., -2., 0., -6., -14., -12.],
            ];

            let Ok((result, Some(r_zi))) = Array::<_, Dim<[Ix; 2]>>::lfilter(
                (&b).into(),
                (&a).into(),
                x,
                Some(0),
                Some((&zi).into()),
            ) else {
                panic!("Should not have errored")
            };

            assert_eq!(result.len(), expected.len());
            result.into_iter().zip(expected).for_each(|(r, e)| {
                assert_relative_eq!(r, e, max_relative = 1e-6);
            });
            assert_eq!(r_zi.len(), expected_zi.len());
            r_zi.into_iter().zip(expected_zi).for_each(|(r, e)| {
                assert_relative_eq!(r, e, max_relative = 1e-6);
            })
        }
        {
            // Case which does falls into zi.shape() != expected_shape branch for 3D input
            let b = array![5., 0.4, 1., -2.];
            let a = array![1.];
            let x = array![
                [[0.2, 2., 3., 4., 3., 5., 6.], [8., 0., 1., 0., 3., 7., 6.]],
                [[1., 2., 3., 4., 3., 5., 6.], [8., 0., 1., 0., 3., 7., 6.]]
            ];
            let zi = array![[[0.4], [0.45], [0.05]], [[0.6], [0.15], [0.25]]];
            let expected = array![
                [
                    [1.4, 10.4, 15.4, 20.4, 15.4, 25.4, 30.4],
                    [40.53, 1.25, 6.65, 2.05, 16.65, 37.45, 32.85]
                ],
                [
                    [5.6, 10.6, 15.6, 20.6, 15.6, 25.6, 30.6],
                    [40.55, 0.95, 6.35, 1.75, 16.35, 37.15, 32.55]
                ]
            ];
            let expected_zi = array![
                [
                    [3.45, 2.05, 3.45, 4.05, 4.25, 7.85, 8.45],
                    [7.6, -4., -5., -8., -3., -3., -6.],
                    [-16., 0., -2., 0., -6., -14., -12.]
                ],
                [
                    [4.45, 2.25, 3.65, 4.25, 4.45, 8.05, 8.65],
                    [6., -4., -5., -8., -3., -3., -6.],
                    [-16., 0., -2., 0., -6., -14., -12.]
                ]
            ];

            let Ok((result, Some(r_zi))) = Array::<_, Dim<[Ix; 3]>>::lfilter(
                (&b).into(),
                (&a).into(),
                x,
                Some(1),
                Some((&zi).into()),
            ) else {
                panic!("Should not have errored")
            };

            assert_eq!(result.len(), expected.len());
            result.into_iter().zip(expected).for_each(|(r, e)| {
                assert_relative_eq!(r, e, max_relative = 1e-6);
            });
            assert_eq!(r_zi.len(), expected_zi.len());
            r_zi.into_iter().zip(expected_zi).for_each(|(r, e)| {
                assert_relative_eq!(r, e, max_relative = 1e-6);
            })
        }
    }

    #[test]
    fn invalid_axis() {
        let b = array![5., 4., 1., 2.];
        let a = array![1.];
        let x = array![1., 2., 3., 4., 3., 5., 6.];

        let result = ArrayView1::lfilter((&b).into(), (&a).into(), (&x).into(), Some(2), None);
        assert!(result.is_err());

        let result = Array1::lfilter((&b).into(), (&a).into(), x.clone(), Some(1), None);
        assert!(result.is_err());

        let result = Array1::lfilter((&b).into(), (&a).into(), x.clone(), Some(0), None);
        assert!(result.is_ok());

        let result = Array1::lfilter((&b).into(), (&a).into(), x.clone(), Some(-1), None);
        assert!(result.is_ok());

        let result = Array1::lfilter((&b).into(), (&a).into(), x, Some(-2), None);
        assert!(result.is_err());
    }
}
