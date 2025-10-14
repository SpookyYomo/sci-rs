use super::{axis_slice_unsafe, check_and_get_axis_dyn};
use alloc::{vec, vec::Vec};
use core::ops::{Add, Sub};
use ndarray::{
    Array, ArrayBase, ArrayView, ArrayView1, Axis, Data, Dim, Dimension, Ix, RawData, RemoveAxis,
    SliceArg, SliceInfo, SliceInfoElem,
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
    fn ext<'a, T, S, D>(
        &'a self,
        x: ArrayBase<S, D>,
        n: usize,
        axis: Option<isize>,
    ) -> Result<Array<T, D>>
    where
        T: Clone + Add<T, Output = T> + Sub<T, Output = T> + num_traits::One + 'a,
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

#[cfg(test)]
mod test {
    use super::*;
    use alloc::vec;
    use ndarray::array;

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
}
