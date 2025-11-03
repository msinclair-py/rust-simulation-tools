// src/sasa.rs
// SASA calculator module for rust-simulation-tools
// Implements the Shrake-Rupley algorithm for calculating Solvent Accessible Surface Area

use ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::HashMap;

/// Generate Fibonacci sphere points for uniform sampling
fn generate_fibonacci_sphere(n_points: usize) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(n_points);
    let phi = std::f64::consts::PI * (3.0 - (5.0_f64).sqrt()); // golden angle

    for i in 0..n_points {
        let y = 1.0 - (i as f64 / (n_points - 1) as f64) * 2.0;
        let radius = (1.0 - y * y).sqrt();
        let theta = phi * i as f64;

        let x = theta.cos() * radius;
        let z = theta.sin() * radius;

        points.push([x, y, z]);
    }

    points
}

/// Core SASA calculation engine
struct SASAEngine {
    coords: Vec<[f64; 3]>,
    radii: Vec<f64>,
    residue_indices: Vec<usize>,
    probe_radius: f64,
    sphere_points: Vec<[f64; 3]>,
    n_sphere_points: usize,
}

impl SASAEngine {
    fn new(
        coords: ArrayView2<f64>,
        radii: ArrayView1<f64>,
        residue_indices: ArrayView1<usize>,
        probe_radius: f64,
        n_sphere_points: usize,
    ) -> Self {
        let n_atoms = coords.shape()[0];

        // Convert to internal format for faster access
        let coords: Vec<[f64; 3]> = (0..n_atoms)
            .map(|i| [coords[[i, 0]], coords[[i, 1]], coords[[i, 2]]])
            .collect();

        let radii: Vec<f64> = radii.iter().copied().collect();
        let residue_indices: Vec<usize> = residue_indices.iter().copied().collect();

        let sphere_points = generate_fibonacci_sphere(n_sphere_points);

        SASAEngine {
            coords,
            radii,
            residue_indices,
            probe_radius,
            sphere_points,
            n_sphere_points,
        }
    }

    #[inline]
    fn distance_squared(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        dx * dx + dy * dy + dz * dz
    }

    /// Calculate SASA for a single atom
    fn calculate_atom_sasa(&self, atom_idx: usize) -> f64 {
        let atom_coord = &self.coords[atom_idx];
        let extended_radius = self.radii[atom_idx] + self.probe_radius;
        let extended_radius_sq = extended_radius * extended_radius;

        // Count accessible points in parallel
        let accessible_points: usize = self
            .sphere_points
            .par_iter()
            .map(|point| {
                // Scale point to atom's extended radius
                let test_point = [
                    atom_coord[0] + point[0] * extended_radius,
                    atom_coord[1] + point[1] * extended_radius,
                    atom_coord[2] + point[2] * extended_radius,
                ];

                // Check if this point is buried by any other atom
                let is_accessible = self.coords.iter().enumerate().all(|(j, other_coord)| {
                    if j == atom_idx {
                        return true;
                    }

                    let other_extended_radius = self.radii[j] + self.probe_radius;
                    let dist_sq = Self::distance_squared(&test_point, other_coord);
                    dist_sq > other_extended_radius * other_extended_radius
                });

                if is_accessible {
                    1
                } else {
                    0
                }
            })
            .sum();

        // Calculate surface area
        let fraction_accessible = accessible_points as f64 / self.n_sphere_points as f64;
        let sphere_area = 4.0 * std::f64::consts::PI * extended_radius_sq;

        fraction_accessible * sphere_area
    }

    /// Calculate per-atom SASA in parallel
    fn calculate_per_atom_sasa(&self) -> Vec<f64> {
        (0..self.coords.len())
            .into_par_iter()
            .map(|i| self.calculate_atom_sasa(i))
            .collect()
    }

    /// Calculate per-residue SASA
    fn calculate_per_residue_sasa(&self) -> HashMap<usize, f64> {
        let atom_sasa = self.calculate_per_atom_sasa();

        let mut residue_sasa: HashMap<usize, f64> = HashMap::new();
        for (atom_idx, &sasa) in atom_sasa.iter().enumerate() {
            let residue_idx = self.residue_indices[atom_idx];
            *residue_sasa.entry(residue_idx).or_insert(0.0) += sasa;
        }

        residue_sasa
    }

    /// Calculate total SASA
    fn calculate_total_sasa(&self) -> f64 {
        self.calculate_per_atom_sasa().iter().sum()
    }
}

/// Calculate SASA with full results (per-atom, per-residue, and total)
///
/// Parameters
/// ----------
/// coordinates : np.ndarray, shape (n_atoms, 3)
///     Atomic coordinates in Angstroms
/// radii : np.ndarray, shape (n_atoms,)
///     Atomic radii in Angstroms
/// residue_indices : np.ndarray, shape (n_atoms,)
///     Residue index for each atom (0-indexed)
/// probe_radius : float, optional
///     Probe radius in Angstroms (default: 1.4 for water)
/// n_sphere_points : int, optional
///     Number of points on test sphere (default: 960)
///
/// Returns
/// -------
/// dict
///     Dictionary with keys 'per_atom', 'per_residue', and 'total'
#[pyfunction]
#[pyo3(signature = (coordinates, radii, residue_indices, probe_radius=1.4, n_sphere_points=960))]
pub fn calculate_sasa(
    coordinates: PyReadonlyArray2<f64>,
    radii: PyReadonlyArray1<f64>,
    residue_indices: PyReadonlyArray1<usize>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<HashMap<String, PyObject>> {
    let coords = coordinates.as_array();
    let radii_arr = radii.as_array();
    let res_indices = residue_indices.as_array();

    // Validate inputs
    let n_atoms = coords.shape()[0];
    if coords.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "Coordinates must have shape (n_atoms, 3)",
        ));
    }
    if radii_arr.len() != n_atoms {
        return Err(PyValueError::new_err(
            "Radii array must match number of atoms",
        ));
    }
    if res_indices.len() != n_atoms {
        return Err(PyValueError::new_err(
            "Residue indices must match number of atoms",
        ));
    }

    // Create engine and calculate
    let engine = SASAEngine::new(coords, radii_arr, res_indices, probe_radius, n_sphere_points);

    let per_atom = engine.calculate_per_atom_sasa();
    let per_residue = engine.calculate_per_residue_sasa();
    let total = per_atom.iter().sum::<f64>();

    // Convert to Python objects
    Python::with_gil(|py| {
        let mut result = HashMap::new();

        // Per-atom SASA as numpy array
        let per_atom_py = PyArray1::from_vec(py, per_atom);
        result.insert("per_atom".to_string(), per_atom_py.into_py(py));

        // Per-residue SASA as dict
        let per_residue_dict = PyDict::new(py);
        for (residue_idx, sasa) in per_residue {
            per_residue_dict.set_item(residue_idx, sasa)?;
        }
        result.insert("per_residue".to_string(), per_residue_dict.into_py(py));

        // Total SASA
        result.insert("total".to_string(), total.into_py(py));

        Ok(result)
    })
}

/// Calculate only per-residue SASA (faster if you don't need per-atom)
///
/// Parameters
/// ----------
/// coordinates : np.ndarray, shape (n_atoms, 3)
///     Atomic coordinates in Angstroms
/// radii : np.ndarray, shape (n_atoms,)
///     Atomic radii in Angstroms
/// residue_indices : np.ndarray, shape (n_atoms,)
///     Residue index for each atom (0-indexed)
/// probe_radius : float, optional
///     Probe radius in Angstroms (default: 1.4)
/// n_sphere_points : int, optional
///     Number of points on test sphere (default: 960)
///
/// Returns
/// -------
/// dict
///     Dictionary mapping residue index to SASA value
#[pyfunction]
#[pyo3(signature = (coordinates, radii, residue_indices, probe_radius=1.4, n_sphere_points=960))]
pub fn calculate_residue_sasa(
    coordinates: PyReadonlyArray2<f64>,
    radii: PyReadonlyArray1<f64>,
    residue_indices: PyReadonlyArray1<usize>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<HashMap<usize, f64>> {
    let coords = coordinates.as_array();
    let radii_arr = radii.as_array();
    let res_indices = residue_indices.as_array();

    // Validate inputs
    let n_atoms = coords.shape()[0];
    if coords.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "Coordinates must have shape (n_atoms, 3)",
        ));
    }
    if radii_arr.len() != n_atoms {
        return Err(PyValueError::new_err(
            "Radii array must match number of atoms",
        ));
    }
    if res_indices.len() != n_atoms {
        return Err(PyValueError::new_err(
            "Residue indices must match number of atoms",
        ));
    }

    let engine = SASAEngine::new(coords, radii_arr, res_indices, probe_radius, n_sphere_points);
    Ok(engine.calculate_per_residue_sasa())
}

/// Calculate only total SASA (fastest option)
///
/// Parameters
/// ----------
/// coordinates : np.ndarray, shape (n_atoms, 3)
///     Atomic coordinates in Angstroms
/// radii : np.ndarray, shape (n_atoms,)
///     Atomic radii in Angstroms
/// probe_radius : float, optional
///     Probe radius in Angstroms (default: 1.4)
/// n_sphere_points : int, optional
///     Number of points on test sphere (default: 960)
///
/// Returns
/// -------
/// float
///     Total SASA in Ų
#[pyfunction]
#[pyo3(signature = (coordinates, radii, probe_radius=1.4, n_sphere_points=960))]
pub fn calculate_total_sasa(
    coordinates: PyReadonlyArray2<f64>,
    radii: PyReadonlyArray1<f64>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<f64> {
    let coords = coordinates.as_array();
    let radii_arr = radii.as_array();

    // Validate inputs
    let n_atoms = coords.shape()[0];
    if coords.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "Coordinates must have shape (n_atoms, 3)",
        ));
    }
    if radii_arr.len() != n_atoms {
        return Err(PyValueError::new_err(
            "Radii array must match number of atoms",
        ));
    }

    // Create dummy residue indices (not used for total calculation)
    let dummy_indices: Vec<usize> = (0..n_atoms).collect();
    let dummy_view = ArrayView1::from(&dummy_indices);

    let engine = SASAEngine::new(coords, radii_arr, dummy_view, probe_radius, n_sphere_points);
    Ok(engine.calculate_total_sasa())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_single_atom() {
        let coords = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let radii = ndarray::arr1(&[1.5]);
        let res_indices = ndarray::arr1(&[0]);

        let engine = SASAEngine::new(
            coords.view(),
            radii.view(),
            res_indices.view(),
            1.4,
            1000,
        );
        let total = engine.calculate_total_sasa();

        let expected = 4.0 * std::f64::consts::PI * (1.5 + 1.4).powi(2);
        let error = (total - expected).abs() / expected;
        assert!(error < 0.05, "Error: {:.2}%", error * 100.0);
    }

    #[test]
    fn test_two_distant_atoms() {
        let coords = Array2::from_shape_vec(
            (2, 3),
            vec![0.0, 0.0, 0.0, 100.0, 0.0, 0.0],
        ).unwrap();
        let radii = ndarray::arr1(&[1.5, 1.5]);
        let res_indices = ndarray::arr1(&[0, 1]);

        let engine = SASAEngine::new(
            coords.view(),
            radii.view(),
            res_indices.view(),
            1.4,
            1000,
        );
        let total = engine.calculate_total_sasa();

        let expected = 2.0 * 4.0 * std::f64::consts::PI * (1.5 + 1.4).powi(2);
        let error = (total - expected).abs() / expected;
        assert!(error < 0.05);
    }

    #[test]
    fn test_per_residue_aggregation() {
        let coords = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 100.0, 0.0, 0.0],
        ).unwrap();
        let radii = ndarray::arr1(&[1.5, 1.0, 1.5]);
        let res_indices = ndarray::arr1(&[0, 0, 1]);

        let engine = SASAEngine::new(
            coords.view(),
            radii.view(),
            res_indices.view(),
            1.4,
            1000,
        );
        let per_residue = engine.calculate_per_residue_sasa();

        assert_eq!(per_residue.len(), 2);
        assert!(per_residue.contains_key(&0));
        assert!(per_residue.contains_key(&1));
    }
}
