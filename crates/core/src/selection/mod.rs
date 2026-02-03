//! Atom selection language for VMD/MDAnalysis-style queries.
//!
//! # Examples
//!
//! ```ignore
//! use rst_core::selection::{select, select_full};
//!
//! // Index-based selection (backward compatible)
//! let indices = select(&topology, "name CA and resid 1-50")?;
//! let near = select_with_coordinates(&topology, "within 5.0 of resname LIG", &coords)?;
//!
//! // Object-oriented selection with property access
//! let sel = select_full(&topology, "protein and name CA")?;
//! println!("Selected {} atoms from {} residues", sel.n_atoms, sel.n_residues);
//! println!("Total mass: {:.2}", sel.total_mass());
//! ```

pub mod ast;
pub mod error;
pub mod eval;
pub mod keywords;
pub mod parser;
pub mod selection;
pub mod token;

pub use error::SelectionError;
pub use eval::SelectionContext;
pub use selection::Selection;

use crate::amber::prmtop::AmberTopology;

/// Select atoms using a VMD-style selection expression (topology only).
/// Returns a vector of atom indices.
pub fn select(topology: &AmberTopology, expression: &str) -> Result<Vec<usize>, SelectionError> {
    let ctx = SelectionContext::new(topology);
    ctx.eval_str(expression)
}

/// Select atoms using a VMD-style selection expression with coordinates
/// (required for `within` selections). Returns a vector of atom indices.
pub fn select_with_coordinates(
    topology: &AmberTopology,
    expression: &str,
    coordinates: &[[f64; 3]],
) -> Result<Vec<usize>, SelectionError> {
    let ctx = SelectionContext::with_coordinates(topology, coordinates);
    ctx.eval_str(expression)
}

/// Select atoms and return a Selection object with direct property access.
///
/// This is the object-oriented alternative to `select()` that provides
/// direct access to atom properties like names, masses, and charges.
///
/// # Arguments
/// * `topology` - The AmberTopology to select from
/// * `expression` - VMD-style selection expression
///
/// # Returns
/// A `Selection` containing the selected atoms with their properties
pub fn select_full(
    topology: &AmberTopology,
    expression: &str,
) -> Result<Selection, SelectionError> {
    let indices = select(topology, expression)?;
    Ok(Selection::from_indices(topology, indices))
}

/// Select atoms with coordinates and return a Selection object.
///
/// This is the object-oriented alternative to `select_with_coordinates()`
/// required for `within` selections. The resulting Selection will include
/// the positions of selected atoms.
///
/// # Arguments
/// * `topology` - The AmberTopology to select from
/// * `expression` - VMD-style selection expression
/// * `coordinates` - Atom coordinates for spatial queries (also stored in Selection)
///
/// # Returns
/// A `Selection` containing the selected atoms with their properties and positions
pub fn select_full_with_coordinates(
    topology: &AmberTopology,
    expression: &str,
    coordinates: &[[f64; 3]],
) -> Result<Selection, SelectionError> {
    let indices = select_with_coordinates(topology, expression, coordinates)?;
    Ok(Selection::from_indices_with_coordinates(
        topology,
        indices,
        coordinates,
    ))
}
