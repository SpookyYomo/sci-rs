mod ndarray_conv_binds;

use crate::{Error, Result};
use alloc::string::ToString;
use ndarray::{Array1, ArrayView1};
use ndarray_conv::{ConvExt, PaddingMode};

/// Convolution mode determines behavior near edges and output size
pub enum ConvolveMode {
    /// Full convolution, output size is `in1.len() + in2.len() - 1`
    Full,
    /// Valid convolution, output size is `max(in1.len(), in2.len()) - min(in1.len(), in2.len()) + 1`
    Valid,
    /// Same convolution, output size is `in1.len()`
    Same,
}

/// Best effort parallel behaviour with numpy's convolve method. We take `v` as the convolution
/// kernel.
///
/// Returns the discrete, linear convolution of two one-dimensional sequences.
///
/// # Parameters
/// * `a` : (N,) [[array_like]]([ndarray::Array1])  
///   Signal to be (linearly) convolved.
/// * `v` : (M,) [[array_like]]([ndarray::Array1])  
///   Second one-dimensional input array.
/// * `mode` : [ConvolveMode]  
///   [ConvolveMode::Full]:  
///   By default, mode is 'full'.  This returns the convolution at each point of overlap, with an
///   output shape of (N+M-1,). At the end-points of the convolution, the signals do not overlap
///   completely, and boundary effects may be seen.
///
///   [ConvolveMode::Same]:  
///   Mode 'same' returns output of length ``max(M, N)``.  Boundary effects are still visible.
///
///   [ConvolveMode::Valid]:  
///   Mode 'valid' returns output of length ``max(M, N) - min(M, N) + 1``.  The convolution
///   product is only given for points where the signals overlap completely.  Values outside the
///   signal boundary have no effect.
///
/// # Panics
/// We assume that `v` is shorter than `a`.
///
/// # Examples
/// With [ConvolveMode::Full]:
/// ```
/// use ndarray::array;
/// use sci_rs_core::num_rs::{ConvolveMode, convolve};
///
/// let a = array![1., 2., 3.];
/// let v = array![0., 1., 0.5];
///
/// let expected = array![0., 1., 2.5, 4., 1.5];
/// let result = convolve((&a).into(), (&v).into(), ConvolveMode::Full).unwrap();
/// assert_eq!(result, expected);
/// ```
/// With [ConvolveMode::Same]:
/// ```
/// use ndarray::array;
/// use sci_rs_core::num_rs::{ConvolveMode, convolve};
///
/// let a = array![1., 2., 3.];
/// let v = array![0., 1., 0.5];
///
/// let expected = array![1., 2.5, 4.];
/// let result = convolve((&a).into(), (&v).into(), ConvolveMode::Same).unwrap();
/// assert_eq!(result, expected);
/// ```
/// With [ConvolveMode::Same]:
/// ```
/// use ndarray::array;
/// use sci_rs_core::num_rs::{ConvolveMode, convolve};
///
/// let a = array![1., 2., 3.];
/// let v = array![0., 1., 0.5];
///
/// let expected = array![2.5];
/// let result = convolve((&a).into(), (&v).into(), ConvolveMode::Valid).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn convolve<T>(a: ArrayView1<T>, v: ArrayView1<T>, mode: ConvolveMode) -> Result<Array1<T>>
where
    // ? Debug for ndarray_conv::ConvExt::conv
    T: num_traits::NumAssign + core::marker::Copy + core::fmt::Debug,
{
    // Flip the convolution kernel (see [ndarray_conv#6](https://github.com/TYPEmber/ndarray-conv/issues/6))
    // waiting for ndarray_conv v0.4.2 to not require for us to flip
    let v: Array1<_> = {
        let mut v = v.to_vec();
        v.reverse();
        v.into()
    };

    // Convolve
    let result = a.conv(&v, mode.into(), PaddingMode::Zeros);
    #[cfg(feature = "alloc")]
    {
        result.map_err(|e| Error::Conv {
            reason: e.to_string(),
        })
    }
    #[cfg(not(feature = "alloc"))]
    {
        result.map_err({ Error::Conv })
    }
}

#[cfg(test)]
mod linear_convolve {
    use super::*;
    use alloc::vec;
    use ndarray::array;

    #[test]
    fn full() {
        let a = array![1., 2., 3.];
        let v = array![0., 1., 0.5];

        let expected = array![0., 1., 2.5, 4., 1.5];
        let result = convolve((&a).into(), (&v).into(), ConvolveMode::Full).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn same() {
        let a = array![1., 2., 3.];
        let v = array![0., 1., 0.5];

        let expected = array![1., 2.5, 4.];
        let result = convolve((&a).into(), (&v).into(), ConvolveMode::Same).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn valid() {
        let a = array![1., 2., 3.];
        let v = array![0., 1., 0.5];

        let expected = array![2.5];
        let result = convolve((&a).into(), (&v).into(), ConvolveMode::Valid).unwrap();
        assert_eq!(result, expected);
    }
}
