use super::{axis_slice_unsafe, check_and_get_axis_dyn};
use alloc::vec::Vec;
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
    fn ext<'a, T, S, D>(
        &'a self,
        x: ArrayBase<S, D>,
        n: usize,
        axis: Option<isize>,
    ) -> Result<Array<T, D>>
    where
        T: Clone + Add<T, Output = T> + Sub<T, Output = T> + 'a,
        S: Data<Elem = T>,
        D: Dimension + RemoveAxis,
        SliceInfo<Vec<SliceInfoElem>, D, D>: SliceArg<D, OutDim = D>,
    {
        if D::NDIM.is_none() {
            return Err(Error::InvalidArg {
                arg: "x".into(),
                reason: "IxDyn is not supported".into(),
            });
        }

        let ndim = D::NDIM.unwrap();

        let axis = check_and_get_axis_dyn(axis, &x).map_err(|_| Error::InvalidArg {
            arg: "axis".into(),
            reason: "index out of range.".into(),
        })?;

        match self {
            FiltFiltPadType::Odd => {
                if n < 1 {
                    return Ok(x.to_owned());
                }

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
            FiltFiltPadType::Even => todo!(),
            FiltFiltPadType::Const => todo!(),
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

        let result = odd.ext(a, 2, None).expect("Could not get odd_ext.");
        let expected = array![
            [-1, 0, 1, 2, 3, 4, 5, 6, 7],
            [-4, -1, 0, 1, 4, 9, 16, 23, 28]
        ];

        ndarray::Zip::from(&result)
            .and(&expected)
            .for_each(|&r, &e| assert_eq!(r, e));
    }
}
