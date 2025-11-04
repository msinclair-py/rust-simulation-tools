use pyo3::prelude::*;

mod kabsch;
mod wrapping;
mod sasa;

/// A Python module implemented in Rust for fast Kabsch alignment and trajectory processing
#[pymodule]
fn rust_simulation_tools(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kabsch::kabsch_align, m)?)?;
    m.add_function(wrap_pyfunction!(wrapping::unwrap_system, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::calculate_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::calculate_sasa_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::calculate_residue_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::calculate_total_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::get_radii_array, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::get_vdw_radius, m)?)?;
    Ok(())
}
