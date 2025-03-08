use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

#[pyfunction]
#[pyo3(name = "i0")]
fn i0_py<'py>(py: Python<'py>, inp: PyReadonlyArrayDyn<'py, f64>) -> Bound<'py, PyArrayDyn<f64>> {
    let inp = inp.as_array();
    use sci_rs::special::Bessel;
    let result = inp.map(|x| x.i0());
    result.into_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn sciprs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(i0_py, m)?)?;
    Ok(())
}
