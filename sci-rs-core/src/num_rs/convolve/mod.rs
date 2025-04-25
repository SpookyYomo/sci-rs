mod ndarray_conv_binds;

/// Convolution mode determines behavior near edges and output size
pub enum ConvolveMode {
    /// Full convolution, output size is `in1.len() + in2.len() - 1`
    Full,
    /// Valid convolution, output size is `max(in1.len(), in2.len()) - min(in1.len(), in2.len()) + 1`
    Valid,
    /// Same convolution, output size is `in1.len()`
    Same,
}
