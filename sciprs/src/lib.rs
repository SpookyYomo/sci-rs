use pyo3::prelude::*;
use sci_rs::signal::convolve::{fftconvolve, ConvolveMode};

#[pyfunction]
#[pyo3(name = "fftconvolve")]
fn fftconvolve_py(in1: Vec<f64>, in2: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(fftconvolve(&in1, &in2, ConvolveMode::Full))
}

/// A Python module implemented in Rust.
#[pymodule]
fn sciprs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fftconvolve_py, m)?)?;
    Ok(())
}
