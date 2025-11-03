use pyo3::prelude::*;

mod kabsch;
mod wrapping;
mod sasa;

use kabsch::kabsch_align;
use wrapping::unwrap_system;
use sasa::{calculate_sasa, calculate_residue_sasa, calculate_total_sasa};


/// A Python module implemented in Rust for fast Kabsch alignment and trajectory processing
#[pymodule]
fn rust_simulation_tools(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kabsch_align, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_system, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_residue_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_total_sasa, m)?)?;
    Ok(())
}
