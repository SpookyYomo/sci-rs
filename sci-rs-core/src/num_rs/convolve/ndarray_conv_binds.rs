use super::ConvolveMode;
use ndarray_conv::ConvMode;

impl<const N: usize> From<ConvolveMode> for ConvMode<N> {
    fn from(value: ConvolveMode) -> Self {
        match value {
            ConvolveMode::Full => ConvMode::Full,
            ConvolveMode::Same => ConvMode::Same,
            ConvolveMode::Valid => ConvMode::Valid,
        }
    }
}
