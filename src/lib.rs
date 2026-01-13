// Allow complex types in PyO3 return signatures (necessary for Python bindings)
#![allow(clippy::type_complexity)]
// PyResult conversions are required by PyO3, not useless
#![allow(clippy::useless_conversion)]
// Some FFI functions require many arguments
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

mod amber;
mod fingerprint;
mod kabsch;
mod sasa;
mod trajectory;
mod util;
mod wrapping;

/// A Python module implemented in Rust for fast Kabsch alignment and trajectory processing
#[pymodule]
fn rust_simulation_tools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Alignment
    m.add_function(wrap_pyfunction!(kabsch::kabsch_align, m)?)?;
    // Trajectory processing
    m.add_function(wrap_pyfunction!(wrapping::unwrap_system, m)?)?;
    // SASA calculations
    m.add_function(wrap_pyfunction!(sasa::calculate_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::calculate_sasa_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::calculate_residue_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::calculate_total_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::get_radii_array, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::get_vdw_radius, m)?)?;
    // Interaction energy fingerprinting
    m.add_function(wrap_pyfunction!(fingerprint::compute_fingerprints_py, m)?)?;
    // AMBER file readers
    m.add_class::<amber::prmtop::PyAmberTopology>()?;
    m.add_function(wrap_pyfunction!(amber::prmtop::read_prmtop_py, m)?)?;
    m.add_function(wrap_pyfunction!(amber::inpcrd::read_inpcrd_py, m)?)?;
    // Trajectory readers
    m.add_class::<trajectory::dcd::PyDcdReader>()?;
    m.add_function(wrap_pyfunction!(trajectory::dcd::read_dcd_header_py, m)?)?;
    Ok(())
}
