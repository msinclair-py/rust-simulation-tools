//! Atom selection language for VMD/MDAnalysis-style queries.
//!
//! # Examples
//!
//! ```ignore
//! use rst_core::selection::select;
//!
//! let indices = select(&topology, "name CA and resid 1-50")?;
//! let near = select_with_coordinates(&topology, "within 5.0 of resname LIG", &coords)?;
//! ```

pub mod ast;
pub mod error;
pub mod eval;
pub mod keywords;
pub mod parser;
pub mod token;

pub use error::SelectionError;
pub use eval::SelectionContext;

use crate::amber::prmtop::AmberTopology;

/// Select atoms using a VMD-style selection expression (topology only).
pub fn select(topology: &AmberTopology, expression: &str) -> Result<Vec<usize>, SelectionError> {
    let ctx = SelectionContext::new(topology);
    ctx.eval_str(expression)
}

/// Select atoms using a VMD-style selection expression with coordinates
/// (required for `within` selections).
pub fn select_with_coordinates(
    topology: &AmberTopology,
    expression: &str,
    coordinates: &[[f64; 3]],
) -> Result<Vec<usize>, SelectionError> {
    let ctx = SelectionContext::with_coordinates(topology, coordinates);
    ctx.eval_str(expression)
}
