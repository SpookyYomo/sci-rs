use super::arraytools::{
    axis_reverse_unsafe, axis_slice_unsafe, check_and_get_axis_dyn, ndarray_shape_as_array_st,
};
use super::lfilter::LFilter;
use super::lfilter_zi::lfilter_zi_dyn;
use alloc::{vec, vec::Vec};
use core::ops::{Add, Sub};
use ndarray::{
    Array, ArrayBase, ArrayView, ArrayView1, Axis, CowArray, Data, Dim, Dimension, Ix, RawData,
    RemoveAxis, SliceArg, SliceInfo, SliceInfoElem,
};
use sci_rs_core::{Error, Result};

/// Padding utilised in [FiltFilt::filtfilt].
// WARN: Related/Duplicate: [super::Pad].
#[derive(Copy, Clone, Default)]
pub enum FiltFiltPadType {
    /// Odd extensions
    #[default]
    Odd,
    /// Even extensions
    Even,
    /// Constant extensions
    Const,
}

impl FiltFiltPadType {
    /// Extensions on ndarrays.
    ///
    /// # Parameters
    /// `self`: Type of extension.
    /// `x`: Array to extend on.
    /// `n`: The number of elements by which to extend `x` at each end of the axis.
    /// `axis`: The axis along which to extend `x`.
    ///
    /// ## Type of extension
    /// * odd: Odd extension at the boundaries of an array, generating a new ndarray by making an
    ///   odd extension of `x` along the specified axis.
    /// * even: Even extension at the boundaries of an array, generating a new ndarray by making an
    ///   even extension of `x` along the specified axis.
    /// * const: Constant extension at the boundaries of an array, generating a new ndarray by
    ///   making an constant extension of `x` along the specified axis.
    fn ext<T, S, D>(&self, x: ArrayBase<S, D>, n: usize, axis: Option<isize>) -> Result<Array<T, D>>
    where
        T: Clone + Add<T, Output = T> + Sub<T, Output = T> + num_traits::One,
        S: Data<Elem = T>,
        D: Dimension + RemoveAxis,
        SliceInfo<Vec<SliceInfoElem>, D, D>: SliceArg<D, OutDim = D>,
    {
        if n < 1 {
            return Ok(x.to_owned());
        }

        let ndim = D::NDIM.unwrap_or(x.ndim());

        let axis = check_and_get_axis_dyn(axis, &x).map_err(|_| Error::InvalidArg {
            arg: "axis".into(),
            reason: "index out of range.".into(),
        })?;

        {
            let axis_len = x.shape()[axis];
            if n >= axis_len {
                return Err(Error::InvalidArg {
                    arg: "n".into(),
                    reason: "Extension of array cannot be longer than array in specified axis."
                        .into(),
                });
            }
        }

        match self {
            FiltFiltPadType::Odd => {
                let left_end =
                    unsafe { axis_slice_unsafe(&x, Some(0), Some(1), None, axis, ndim) }?;
                let left_ext = unsafe {
                    axis_slice_unsafe(&x, Some(n as isize), Some(0), Some(-1), axis, ndim)
                }?;
                let right_end = unsafe { axis_slice_unsafe(&x, Some(-1), None, None, axis, ndim) }?;
                let right_ext = unsafe {
                    axis_slice_unsafe(&x, Some(-2), Some(-2 - (n as isize)), Some(-1), axis, ndim)
                }?;

                let ll = left_end.to_owned().add(left_end).sub(left_ext);
                let rr = right_end.to_owned().add(right_end).sub(right_ext);

                ndarray::concatenate(Axis(axis), &[ll.view(), x.view(), rr.view()]).map_err(|_| {
                    Error::InvalidArg {
                        arg: "x".into(),
                        reason: "Shape Error".into(),
                    }
                })
            }
            FiltFiltPadType::Even => {
                let left_ext = unsafe {
                    axis_slice_unsafe(&x, Some(n as isize), Some(0), Some(-1), axis, ndim)
                }?;
                let right_ext = unsafe {
                    axis_slice_unsafe(&x, Some(-2), Some(-2 - (n as isize)), Some(-1), axis, ndim)
                }?;

                ndarray::concatenate(Axis(axis), &[left_ext.view(), x.view(), right_ext.view()])
                    .map_err(|_| Error::InvalidArg {
                        arg: "x".into(),
                        reason: "Shape Error".into(),
                    })
            }
            FiltFiltPadType::Const => {
                let ones: Array<T, D> = Array::ones({
                    let mut t = vec![1; ndim];
                    t[axis] = n;
                    ndarray::IxDyn(&t)
                })
                .into_dimensionality() // This is needed for IxDyn -> IxN
                .map_err(|_| Error::InvalidArg {
                    arg: "x".into(),
                    reason: "Coercing into identical dimensionality had issue".into(),
                })?;

                let left_ext = {
                    let left_end =
                        unsafe { axis_slice_unsafe(&x, Some(0), Some(1), None, axis, ndim) }?;
                    ones.clone() * left_end
                };

                let right_ext = {
                    let right_end =
                        unsafe { axis_slice_unsafe(&x, Some(-1), None, None, axis, ndim) }?;
                    ones * right_end
                };

                ndarray::concatenate(Axis(axis), &[left_ext.view(), x.view(), right_ext.view()])
                    .map_err(|_| Error::InvalidArg {
                        arg: "x".into(),
                        reason: "Shape Error".into(),
                    })
            }
        }
    }
}

/// Arguments for [FiltFilt::filtfilt].
#[derive(Copy, Clone, Default)]
pub struct FiltFiltPad {
    /// Padding type.
    pub pad_type: FiltFiltPadType,
    /// Length of padding.
    pub len: Option<usize>,
}

/// Helper for validating padding of filtfilt.
///
/// # Parameters
/// `pad`: `Option<FiltFiltPad>`
///   A none value from the user specifies no padding, which implies that the pad len is also 0.
///   Otherwise, the user specifies a specific padding and pad_len.
/// `x`: NDArray
///   Array that is being filtered.
/// `axis`: usize
///   Axis of `x` which is being filtered.
/// `ntaps`: usize
///   This simply is `max(a.len(), b.len())`.
//   In an ideal world, this would be shoe-horned into FiltFiltPad's len parameter.
///
/// # Panics
/// `axis` is as acting on `x` is assumed to be valid, otherwise panics.
fn validate_pad<T, D>(
    pad: Option<FiltFiltPad>,
    x: ArrayView<T, D>,
    axis: usize,
    ntaps: usize,
) -> Result<(usize, CowArray<T, D>)>
where
    T: Clone + Add<T, Output = T> + Sub<T, Output = T> + num_traits::One,
    D: Dimension + RemoveAxis,
    SliceInfo<Vec<SliceInfoElem>, D, D>: SliceArg<D, OutDim = D>,
{
    let edge = match pad {
        None => 0,
        Some(FiltFiltPad { len, .. }) => len.unwrap_or(ntaps * 3),
    };

    {
        x.shape().get(axis).ok_or(Error::InvalidArg {
            arg: "axis".into(),
            reason: "The length of the input vector x must be greater than padlen.".into(),
        })?;
    }

    let ext = if let Some(FiltFiltPad { pad_type, .. }) = pad {
        CowArray::from(pad_type.ext(x, edge, Some(axis as _))?)
    } else {
        CowArray::from(x)
    };

    Ok((edge, ext))
}

/// Implement filtfilt for fixed dimension of input array `x`.
///
/// Valid only from 1 to 6 dimensional arrays.
///
/// Note: FiltFilt gust is a separate function not yet implemented.
// Note: Usage of trait and macro for implementation is an inherited from LFilter.
// LFilter for supertrait?
pub trait FiltFilt<S, T, const N: usize>
where
    S: Data<Elem = T>,
{
    /// Apply a digital filter forward and backward to a signal.
    ///
    /// This function applies a linear digital filter twice, once forward and
    /// once backwards.  The combined filter has zero phase and a filter order
    /// twice that of the original.
    ///
    /// The function provides options for handling the edges of the signal.
    ///
    /// The function `sosfiltfilt` (and filter design using ``output='sos'``)
    /// should be preferred over `filtfilt` for most filtering tasks, as
    /// second-order sections have fewer numerical problems.
    ///
    /// # Parameters
    /// * `b`: (N,) array_like  
    ///   The numerator coefficient vector of the filter.
    /// * `a`: (N,) array_like  
    ///   The denominator coefficient vector of the filter.  If ``a[0]``
    ///   is not 1, then both `a` and `b` are normalized b:y ``a[0]``.
    /// * `x`: array_like  
    ///   The array of data to be filtered.
    /// * `axis`: int, optional  
    ///   The axis of `x` to which the filter is applied.  
    ///   Default is -1.
    /// * `pad`
    ///   [Option::None] here denotes a deliberate absence of padding.
    ///   * `padtype` [FiltFiltPadType]
    ///     Must be 'odd', 'even', 'constant', or None.  
    ///     This determines the type of extension to use for the padded signal to which the filter is applied.  
    ///     The default is 'odd'.
    ///   * `padlen` int or None, optional  
    ///     The number of elements by which to extend `x` at both ends of `axis` before applying
    ///     the filter.  
    ///     This value must be less than ``x.shape[axis] - 1``.  ``padlen=0`` implies no padding. [Option::None] here denotes the default value.  
    ///     The default value is ``3 * max(len(a), len(b))``.
    ///
    /// # Returns
    /// * y : `Array`
    ///   The filtered output with the same shape as `x`.
    ///
    /// # Example
    /// The following examples shows how to use an arbitrary FIR filter on a 2-dimensional input
    /// `x`.
    /// ```
    /// use sci_rs::signal::filter::{FiltFilt, FiltFiltPad};
    /// use ndarray::{array, Array2, ArrayView2};
    ///     let x = array![
    ///         [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
    ///         [0., 1., 4., 9., 16., 25., 36., 49., 64., 81.]
    ///     ];
    /// let b = array![0.5, 0.4, 0.1];
    /// let a = array![1.];
    /// let _ = ArrayView2::filtfilt(
    ///     b.view(),
    ///     a.view(),
    ///     x.view(), // Pass x by reference
    ///     Some(1),
    ///     Some(FiltFiltPad::default())).unwrap();
    /// let result = Array2::filtfilt(
    ///     b.view(),
    ///     a.view(),
    ///     x, // Pass x by value
    ///     Some(1),
    ///     Some(FiltFiltPad::default())).unwrap();
    ///
    /// use approx::assert_relative_eq;
    /// use ndarray::Zip;
    /// let expected = array![
    ///     [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
    ///     [0., 1.78, 4.88, 9.88, 16.88, 25.88, 36.88, 49.88, 64.78, 81.]
    /// ];
    /// Zip::from(&result).and(&expected)
    ///     .for_each(|&r, &e| assert_relative_eq!(r, e, max_relative = 1e-6));
    /// ```
    ///
    /// # See Also
    /// sosfiltfilt, lfilter_zi, lfilter, lfiltic, savgol_filter, sosfilt, filtfilt_gust
    ///
    /// # Notes
    /// When `method` is "pad", the function pads the data along the given axis in one of three
    /// ways: odd, even or constant.  The odd and even extensions have the corresponding symmetry
    /// about the end point of the data.  The constant extension extends the data with the values
    /// at the end points. On both the forward and backward passes, the initial condition of the
    /// filter is found by using `lfilter_zi` and scaling it by the end point of the extended data.
    fn filtfilt<'a>(
        b: ArrayView1<'a, T>,
        a: ArrayView1<'a, T>,
        x: Self,
        axis: Option<isize>,
        padding: Option<FiltFiltPad>,
    ) -> Result<Array<T, Dim<[Ix; N]>>>
    where
        T: Clone + Add<T, Output = T> + Sub<T, Output = T> + num_traits::One,
        Dim<[Ix; N]>: Dimension,
        T: nalgebra::RealField + Copy + core::iter::Sum; // From lfilter_zi_dyn

    /// Forward-back IIR filter that uses Gustafsson's method.
    ///
    /// Not yet implemented.
    fn filtfilt_gust<'a>(
        b: ArrayView1<'a, T>,
        a: ArrayView1<'a, T>,
        x: Self,
        axis: Option<isize>,
        irlen: Option<usize>,
    ) -> Result<Array<T, Dim<[Ix; N]>>>
    where
        Self: Sized,
    {
        todo!("Gust method of FiltFilt is not yet implemented.");
    }
}

impl<S, T> FiltFilt<S, T, 2> for ArrayBase<S, Dim<[Ix; 2]>>
where
    S: Data<Elem = T>,
{
    fn filtfilt<'a>(
        b: ArrayView1<'a, T>,
        a: ArrayView1<'a, T>,
        x: Self,
        axis: Option<isize>,
        padding: Option<FiltFiltPad>,
    ) -> Result<Array<T, Dim<[Ix; 2]>>>
    where
        T: Clone + Add<T, Output = T> + Sub<T, Output = T> + num_traits::One,
        Dim<[Ix; 2]>: Dimension,
        T: nalgebra::RealField + Copy + core::iter::Sum, // From lfilter_zi_dyn
    {
        let axis = {
            if axis.is_some_and(|axis| {
                !(if axis < 0 {
                    axis.unsigned_abs() <= 2
                } else {
                    axis.unsigned_abs() < 2
                })
            }) {
                return Err(Error::InvalidArg {
                    arg: "axis".into(),
                    reason: "index out of range.".into(),
                });
            }

            // We make a best effort to convert into appropriate usize.
            let axis: isize = axis.unwrap_or(-1);
            if axis >= 0 {
                axis.unsigned_abs()
            } else {
                a.ndim()
                    .checked_add_signed(axis)
                    .expect("Invalid add to `axis` option")
            }
        };
        let (edge, ext) = validate_pad(padding, x.view(), axis, a.len().max(b.len()))?;

        let zi: Array<T, Dim<[Ix; 2]>> = {
            let mut zi = lfilter_zi_dyn(b.as_slice().unwrap(), a.as_slice().unwrap());
            let mut sh = [1; 2];
            sh[axis] = zi.len(); // .size()?

            zi.into_shape_with_order(sh)
                .map_err(|_| Error::InvalidArg {
                    arg: "b/a".into(),
                    reason: "Generated lfilter_zi from given b or a resulted in an error.".into(),
                })?
        };
        let (y, _) = {
            let x0 = axis_slice_unsafe(&ext, None, Some(1), None, axis, ext.ndim())?;
            let zi_arg = zi.clone() * x0; // Is it possible to not need to clone?
            ArrayBase::<_, Dim<[Ix; 2]>>::lfilter(
                b.view(),
                a.view(),
                ext,
                Some(axis as _),
                Some(zi_arg.view()),
            )?
        };

        let (y, _) = {
            let y0 = axis_slice_unsafe(&y, Some(-1), None, None, axis, y.ndim())?;
            let zi_arg = zi * y0; // originally zi * y0
            ArrayView::<T, Dim<[Ix; 2]>>::lfilter(
                b.view(),
                a.view(),
                unsafe { axis_reverse_unsafe(&y, axis, 2) },
                Some(axis as _),
                Some(zi_arg.view()),
            )?
        };

        let y = unsafe { axis_reverse_unsafe(&y, axis, 2) };

        if edge > 0 {
            let y = unsafe {
                axis_slice_unsafe(&y, Some(edge as _), Some(-(edge as isize)), None, axis, 2)
            }?;
            Ok(y.to_owned())
        } else {
            Ok(y.to_owned())
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use alloc::vec;
    use approx::assert_relative_eq;
    use ndarray::{array, Zip};

    /// Test odd_ext as from documentation.
    #[test]
    fn odd_ext_doc() {
        let odd = FiltFiltPadType::Odd;
        let a = array![[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]];

        let result = odd.ext(a.view(), 2, None).expect("Could not get odd_ext.");
        let expected = array![
            [-1, 0, 1, 2, 3, 4, 5, 6, 7],
            [-4, -1, 0, 1, 4, 9, 16, 23, 28]
        ];

        ndarray::Zip::from(&result)
            .and(&expected)
            .for_each(|&r, &e| assert_eq!(r, e));

        let result = odd
            .ext(a.into_dyn(), 2, None)
            .expect("Could not get odd_ext.");
        ndarray::Zip::from(&result)
            .and(&expected.into_dyn())
            .for_each(|&r, &e| assert_eq!(r, e));
    }

    /// Test odd_ext's limits.
    #[test]
    fn odd_ext_limits() {
        let odd = FiltFiltPadType::Odd;
        let a = array![[1, 2, 3, 4], [0, 1, 4, 9]];

        let result = odd.ext(a.view(), 3, None);
        assert!(result.is_ok());
        let result = odd.ext(a, 4, None);
        assert!(result.is_err());
    }

    /// Test odd_ext as from documentation.
    #[test]
    fn even_ext_doc() {
        let even = FiltFiltPadType::Even;
        let a = array![[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]];

        let result = even
            .ext(a.view(), 2, None)
            .expect("Could not get even_ext.");
        let expected = array![[3, 2, 1, 2, 3, 4, 5, 4, 3], [4, 1, 0, 1, 4, 9, 16, 9, 4]];

        ndarray::Zip::from(&result)
            .and(&expected)
            .for_each(|&r, &e| assert_eq!(r, e));

        let result = even
            .ext(a.into_dyn(), 2, None)
            .expect("Could not get even_ext.");
        ndarray::Zip::from(&result)
            .and(&expected.into_dyn())
            .for_each(|&r, &e| assert_eq!(r, e));
    }

    /// Test even_ext's limits.
    #[test]
    fn even_ext_limits() {
        let even = FiltFiltPadType::Even;
        let a = array![[1, 2, 3, 4], [0, 1, 4, 9]];

        let result = even.ext(a.view(), 3, None);
        assert!(result.is_ok());
        let result = even.ext(a, 4, None);
        assert!(result.is_err());
    }

    /// Test const_ext as from documentation.
    #[test]
    fn const_ext_doc() {
        let const_ext = FiltFiltPadType::Const;
        let a = array![[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]];

        let result = const_ext
            .ext(a.view(), 2, None)
            .expect("Could not get even_ext.");
        let expected = array![[1, 1, 1, 2, 3, 4, 5, 5, 5], [0, 0, 0, 1, 4, 9, 16, 16, 16]];

        ndarray::Zip::from(&result)
            .and(&expected)
            .for_each(|&r, &e| assert_eq!(r, e));

        let result = const_ext
            .ext(a.into_dyn(), 2, None)
            .expect("Could not get even_ext.");
        ndarray::Zip::from(&result)
            .and(&expected.into_dyn())
            .for_each(|&r, &e| assert_eq!(r, e));
    }

    /// Test const_ext's limits.
    #[test]
    fn const_ext_limits() {
        let const_ext = FiltFiltPadType::Const;
        let a = array![[1, 2, 3, 4], [0, 1, 4, 9]];

        let result = const_ext.ext(a.view(), 3, None);
        assert!(result.is_ok());
        let result = const_ext.ext(a, 4, None);
        assert!(result.is_err());
    }

    /// Tests for when there is no padding.
    #[test]
    fn pad_none() {
        let p = None;
        let x = array![1];
        let result = validate_pad(p, x.view(), 0, 0).expect("Could not pad with none.");

        assert_eq!(result.0, 0);
        assert_eq!(result.1, x);
    }

    /// Tests for when there is even padding.
    #[test]
    fn pad_even() {
        let p = Some(FiltFiltPad {
            pad_type: FiltFiltPadType::Even,
            len: Some(2),
        });
        let x = array![[1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 4, 9, 16, 25, 36, 49]];
        let (result_edge, result) =
            validate_pad(p, x.view(), 1, 2).expect("Could not pad with even.");
        let expected_edge = 2;
        let expected = array![
            [3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6],
            [4, 1, 0, 1, 4, 9, 16, 25, 36, 49, 36, 25]
        ];

        assert_eq!(result_edge, expected_edge);
        assert_eq!(result, expected);
        ndarray::Zip::from(&result)
            .and(&expected)
            .for_each(|r, e| assert_eq!(r, e));
    }

    /// Tests for when there is even padding.
    #[test]
    fn pad_odd() {
        let p = Some(FiltFiltPad {
            pad_type: FiltFiltPadType::Odd,
            len: Some(2),
        });
        let x = array![[1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 4, 9, 16, 25, 36, 49]];
        let (result_edge, result) =
            validate_pad(p, x.view(), 1, 2).expect("Could not pad with odd.");
        let expected_edge = 2;
        let expected = array![
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [-4, -1, 0, 1, 4, 9, 16, 25, 36, 49, 62, 73]
        ];

        assert_eq!(result_edge, expected_edge);
        assert_eq!(result, expected);
    }

    /// Tests for when there is const padding.
    #[test]
    fn pad_const() {
        let p = Some(FiltFiltPad {
            pad_type: FiltFiltPadType::Const,
            len: Some(2),
        });
        let x = array![[1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 4, 9, 16, 25, 36, 49]];
        let (result_edge, result) =
            validate_pad(p, x.view(), 1, 2).expect("Could not pad with odd.");
        let expected_edge = 2;
        let expected = array![
            [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8],
            [0, 0, 0, 1, 4, 9, 16, 25, 36, 49, 49, 49]
        ];

        assert_eq!(result_edge, expected_edge);
        assert_eq!(result, expected);
    }

    /// Tests that filtfilt works with no padding with a FIR filter.
    #[test]
    fn filtfilt_2d_fir_none_pad() {
        let b = array![0.1, 0.2, 0.1, -0.3, 0.2, 0.4, 0.2, 0.1];
        let a = array![1.];
        let x = {
            let rows_n = 40;
            let mut x = Array::zeros((2, rows_n));
            x.row_mut(0)
                .assign(&Array::linspace(1 as _, rows_n as _, rows_n));
            x.row_mut(1)
                .assign(&Array::from_iter((0..rows_n).map(|i| (i as f64).powi(2))));

            x
        };
        let result = Array::<_, Dim<[_; 2]>>::filtfilt(b.view(), a.view(), x, Some(1), None)
            .expect("Could not filtfilt none_pad");
        let expected = array![
            [
                2.14, 2.84, 3.67, 4.43, 5.2, 6.06, 7.01, 8., 9., 10., 11., 12., 13., 14., 15., 16.,
                17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
                33., 33.9, 34.6, 34.9, 35., 35.4, 35.7, 35.8
            ],
            [
                5.56, 8.54, 13.05, 19.15, 26.78, 36.04, 47.11, 60.12, 75.12, 92.12, 111.12, 132.12,
                155.12, 180.12, 207.12, 236.12, 267.12, 300.12, 335.12, 372.12, 411.12, 452.12,
                495.12, 540.12, 587.12, 636.12, 687.12, 740.12, 795.12, 852.12, 911.12, 972.12,
                1035.12, 1093.06, 1138.68, 1157.46, 1162.72, 1189.36, 1209.74, 1216.6
            ]
        ];

        Zip::from(&result)
            .and(&expected)
            .for_each(|&r, &e| assert_relative_eq!(r, e, max_relative = 1e-6));
    }

    /// Tests that filtfilt works with some padding with a FIR filter.
    #[test]
    fn filtfilt_2d_fir_some_pad() {
        let b = array![0.1, 0.2, 0.1, -0.3, 0.2, 0.4, 0.2, 0.1];
        let a = array![1.];
        let x = {
            let rows_n = 40;
            let mut x = Array::zeros((2, rows_n));
            x.row_mut(0)
                .assign(&Array::linspace(1 as _, rows_n as _, rows_n));
            x.row_mut(1)
                .assign(&Array::from_iter((0..rows_n).map(|i| (i as f64).powi(2))));

            x
        };
        let pad_arg = FiltFiltPad {
            pad_type: FiltFiltPadType::default(),
            len: Some(4),
        };
        let result =
            Array::<_, Dim<[_; 2]>>::filtfilt(b.view(), a.view(), x, Some(1), Some(pad_arg))
                .expect("Could not filtfilt none_pad");
        let expected = array![
            [
                1.2, 2.06, 3.01, 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
                34., 35., 36., 37., 37.9, 38.6, 38.9
            ],
            [
                1.94, 5.52, 11.07, 18.18, 26.44, 35.96, 47.1, 60.12, 75.12, 92.12, 111.12, 132.12,
                155.12, 180.12, 207.12, 236.12, 267.12, 300.12, 335.12, 372.12, 411.12, 452.12,
                495.12, 540.12, 587.12, 636.12, 687.12, 740.12, 795.12, 852.12, 911.12, 972.12,
                1035.12, 1100.1, 1166.96, 1235.44, 1305.18, 1368.54, 1418.2, 1439.28
            ]
        ];

        Zip::from(&result)
            .and(&expected)
            .for_each(|&r, &e| assert_relative_eq!(r, e, max_relative = 1e-6));
    }

    /// Tests that filtfilt works with default padding with a FIR filter.
    #[test]
    fn filtfilt_2d_fir_default_pad() {
        let b = array![0.1, 0.2, 0.1, -0.3, 0.2, 0.4, 0.2, 0.1];
        let a = array![1.];
        let x = {
            let rows_n = 40;
            let mut x = Array::zeros((2, rows_n));
            x.row_mut(0)
                .assign(&Array::linspace(1 as _, rows_n as _, rows_n));
            x.row_mut(1)
                .assign(&Array::from_iter((0..rows_n).map(|i| (i as f64).powi(2))));

            x
        };
        let result = Array::<_, Dim<[_; 2]>>::filtfilt(
            b.view(),
            a.view(),
            x,
            Some(1),
            Some(FiltFiltPad::default()),
        )
        .expect("Could not filtfilt none_pad");
        let expected = array![
            [
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.,
                35., 36., 37., 38., 39., 40.
            ],
            [
                0., 4.96, 10.98, 18.18, 26.44, 35.96, 47.1, 60.12, 75.12, 92.12, 111.12, 132.12,
                155.12, 180.12, 207.12, 236.12, 267.12, 300.12, 335.12, 372.12, 411.12, 452.12,
                495.12, 540.12, 587.12, 636.12, 687.12, 740.12, 795.12, 852.12, 911.12, 972.12,
                1035.12, 1100.1, 1166.96, 1235.44, 1305.18, 1375.98, 1447.96, 1521.
            ]
        ];

        Zip::from(&result)
            .and(&expected)
            .for_each(|&r, &e| assert_relative_eq!(r, e, max_relative = 1e-6, epsilon = 1e-10));
    }

    /// Tests that is an error if the specified padding is a lot longer than the array.
    #[test]
    fn filfilt_2d_fir_limit() {
        let b = array![0.1, 0.2, 0.1, -0.3, 0.2, 0.4, 0.2, 0.1];
        let a = array![1.];
        let x = {
            let rows_n = 4;
            let mut x = Array::zeros((2, rows_n));
            x.row_mut(0)
                .assign(&Array::linspace(1 as _, rows_n as _, rows_n));
            x.row_mut(1)
                .assign(&Array::from_iter((0..rows_n).map(|i| (i as f64).powi(2))));

            x
        };
        let result = Array::<_, Dim<[_; 2]>>::filtfilt(
            b.view(),
            a.view(),
            x,
            Some(1),
            Some(FiltFiltPad::default()),
        );

        assert!(result.is_err());
    }
}
