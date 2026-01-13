//! AMBER inpcrd/rst7 coordinate file parser.
//!
//! Parses AMBER coordinate/restart files to extract atomic positions.
//! Coordinates are converted from Angstrom to nm.

use numpy::ndarray::Array2;
use numpy::PyArray2;
use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Angstrom to nm conversion factor
const ANGSTROM_TO_NM: f64 = 0.1;

/// Parsed AMBER coordinate data.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AmberCoordinates {
    /// Number of atoms
    pub n_atoms: usize,
    /// Atomic coordinates in nm, shape (n_atoms, 3)
    pub positions: Vec<[f64; 3]>,
    /// Box dimensions in nm (if present)
    pub box_dimensions: Option<[f64; 3]>,
    /// Box angles in degrees (if present)
    pub box_angles: Option<[f64; 3]>,
}

/// Parse an AMBER inpcrd/rst7 coordinate file.
///
/// # Format
/// Line 1: Title
/// Line 2: Number of atoms (and optionally time)
/// Lines 3+: Coordinates in 12.7f format, 6 values per line
/// Optional last line: Box dimensions
///
/// # Arguments
/// * `path` - Path to the inpcrd file
///
/// # Returns
/// * `Ok(AmberCoordinates)` - Parsed coordinates in nm
/// * `Err(String)` - Error message if parsing fails
pub fn parse_inpcrd<P: AsRef<Path>>(path: P) -> Result<AmberCoordinates, String> {
    let file =
        File::open(path.as_ref()).map_err(|e| format!("Failed to open inpcrd file: {}", e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Line 1: Title (skip)
    let _title = lines
        .next()
        .ok_or("Empty file")?
        .map_err(|e| format!("Failed to read title: {}", e))?;

    // Line 2: Number of atoms
    let header = lines
        .next()
        .ok_or("Missing atom count line")?
        .map_err(|e| format!("Failed to read header: {}", e))?;

    let n_atoms: usize = header
        .split_whitespace()
        .next()
        .ok_or("No atom count found")?
        .parse()
        .map_err(|e| format!("Failed to parse atom count: {}", e))?;

    // Read coordinate values
    // Format: 12.7f, 6 values per line
    let n_values_needed = n_atoms * 3;
    let mut values: Vec<f64> = Vec::with_capacity(n_values_needed);
    let mut pending_box_line: Option<String> = None;

    while let Some(line_result) = lines.next() {
        let line = line_result.map_err(|e| format!("Failed to read line: {}", e))?;

        // Parse fixed-width format (12 characters per value)
        let mut pos = 0;
        while pos + 12 <= line.len() && values.len() < n_values_needed {
            let val_str = line[pos..pos + 12].trim();
            if !val_str.is_empty() {
                let val: f64 = val_str
                    .parse()
                    .map_err(|e| format!("Failed to parse coordinate '{}': {}", val_str, e))?;
                values.push(val * ANGSTROM_TO_NM); // Convert to nm
            }
            pos += 12;
        }

        // Also try whitespace-separated for compatibility
        if pos < line.len() && values.len() < n_values_needed {
            for word in line[pos..].split_whitespace() {
                if values.len() >= n_values_needed {
                    break;
                }
                let val: f64 = word
                    .parse()
                    .map_err(|e| format!("Failed to parse coordinate: {}", e))?;
                values.push(val * ANGSTROM_TO_NM);
            }
        }

        if values.len() >= n_values_needed {
            // Check for box dimensions in next line
            if let Some(box_line_result) = lines.next() {
                if let Ok(box_line) = box_line_result {
                    pending_box_line = Some(box_line);
                }
            }
            break;
        }
    }

    // Parse box dimensions if present
    if let Some(box_line) = pending_box_line {
        let box_vals: Vec<f64> = box_line
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();

        if box_vals.len() >= 3 {
            let box_dimensions = Some([
                box_vals[0] * ANGSTROM_TO_NM,
                box_vals[1] * ANGSTROM_TO_NM,
                box_vals[2] * ANGSTROM_TO_NM,
            ]);
            let box_angles = if box_vals.len() >= 6 {
                Some([box_vals[3], box_vals[4], box_vals[5]])
            } else {
                Some([90.0, 90.0, 90.0])
            };

            let positions: Vec<[f64; 3]> = values
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();

            return Ok(AmberCoordinates {
                n_atoms,
                positions,
                box_dimensions,
                box_angles,
            });
        }
    }

    if values.len() < n_values_needed {
        return Err(format!(
            "Expected {} coordinate values, found {}",
            n_values_needed,
            values.len()
        ));
    }

    // Build positions array
    let positions: Vec<[f64; 3]> = values
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    Ok(AmberCoordinates {
        n_atoms,
        positions,
        box_dimensions: None,
        box_angles: None,
    })
}

// ============================================================================
// Python Interface
// ============================================================================

/// Read an AMBER inpcrd/rst7 coordinate file.
///
/// Parameters
/// ----------
/// path : str
///     Path to the inpcrd/rst7 file.
///
/// Returns
/// -------
/// tuple
///     (positions, box_dimensions) where:
///     - positions: ndarray of shape (n_atoms, 3) in nm
///     - box_dimensions: ndarray of shape (3,) in nm, or None if not present
#[pyfunction]
#[pyo3(name = "read_inpcrd")]
pub fn read_inpcrd_py<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Option<Vec<f64>>)> {
    let coords =
        parse_inpcrd(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

    // Convert to 2D array
    let n_atoms = coords.n_atoms;
    let mut flat: Vec<f64> = Vec::with_capacity(n_atoms * 3);
    for pos in &coords.positions {
        flat.extend_from_slice(pos);
    }

    let arr = Array2::from_shape_vec((n_atoms, 3), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;

    let positions = PyArray2::from_owned_array_bound(py, arr);
    let box_dims = coords.box_dimensions.map(|b| b.to_vec());

    Ok((positions, box_dims))
}
