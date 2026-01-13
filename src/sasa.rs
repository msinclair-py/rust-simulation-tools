// src/sasa.rs
// SASA calculator module for rust-simulation-tools
// Implements the Shrake-Rupley algorithm for calculating Solvent Accessible Surface Area
// with KD-tree optimization for efficient neighbor searching

use crate::util::distance_squared;
use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::HashMap;

/// KD-Tree node for efficient 3D spatial queries
#[derive(Debug, Clone)]
struct KDNode {
    point_idx: usize,
    split_dim: usize,
    split_value: f64,
    left: Option<Box<KDNode>>,
    right: Option<Box<KDNode>>,
}

/// KD-Tree for fast spatial neighbor queries
pub struct KDTree {
    root: Option<Box<KDNode>>,
    coords: Vec<[f64; 3]>,
}

impl KDTree {
    /// Build a KD-tree from 3D coordinates
    fn new(coords: &[[f64; 3]]) -> Self {
        let n_points = coords.len();
        let mut indices: Vec<usize> = (0..n_points).collect();

        let root = if !indices.is_empty() {
            Self::build_tree(coords, &mut indices, 0)
        } else {
            None
        };

        KDTree {
            root,
            coords: coords.to_vec(),
        }
    }

    /// Recursively build the KD-tree
    fn build_tree(coords: &[[f64; 3]], indices: &mut [usize], depth: usize) -> Option<Box<KDNode>> {
        if indices.is_empty() {
            return None;
        }

        if indices.len() == 1 {
            let idx = indices[0];
            let split_dim = depth % 3;
            return Some(Box::new(KDNode {
                point_idx: idx,
                split_dim,
                split_value: coords[idx][split_dim],
                left: None,
                right: None,
            }));
        }

        let split_dim = depth % 3;

        // Sort indices by the splitting dimension
        indices.sort_unstable_by(|&a, &b| {
            coords[a][split_dim]
                .partial_cmp(&coords[b][split_dim])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let median = indices.len() / 2;
        let median_idx = indices[median];
        let split_value = coords[median_idx][split_dim];

        // Split indices
        let (left_indices, right_and_median) = indices.split_at_mut(median);
        let right_indices = &mut right_and_median[1..];

        Some(Box::new(KDNode {
            point_idx: median_idx,
            split_dim,
            split_value,
            left: Self::build_tree(coords, left_indices, depth + 1),
            right: Self::build_tree(coords, right_indices, depth + 1),
        }))
    }

    /// Query all points within a given radius, reusing provided buffer
    #[inline]
    fn query_radius_into(&self, target: &[f64; 3], radius: f64, result: &mut Vec<usize>) {
        result.clear();
        let radius_sq = radius * radius;

        if let Some(ref root) = self.root {
            self.query_radius_recursive(root, target, radius_sq, result);
        }
    }

    /// Recursive radius query
    #[inline]
    fn query_radius_recursive(
        &self,
        node: &KDNode,
        target: &[f64; 3],
        radius_sq: f64,
        result: &mut Vec<usize>,
    ) {
        let point = &self.coords[node.point_idx];

        // Check if current point is within radius
        let dist_sq = distance_squared(point, target);

        if dist_sq <= radius_sq {
            result.push(node.point_idx);
        }

        // Determine which subtrees to search
        let diff = target[node.split_dim] - node.split_value;
        let diff_sq = diff * diff;

        // Search near side first
        if diff < 0.0 {
            if let Some(ref left) = node.left {
                self.query_radius_recursive(left, target, radius_sq, result);
            }
            // Check if we need to search far side
            if diff_sq <= radius_sq {
                if let Some(ref right) = node.right {
                    self.query_radius_recursive(right, target, radius_sq, result);
                }
            }
        } else {
            if let Some(ref right) = node.right {
                self.query_radius_recursive(right, target, radius_sq, result);
            }
            // Check if we need to search far side
            if diff_sq <= radius_sq {
                if let Some(ref left) = node.left {
                    self.query_radius_recursive(left, target, radius_sq, result);
                }
            }
        }
    }
}

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

/// Core SASA calculation engine with KD-tree optimization
/// Optimized to avoid nested parallelism and cache max_radius
struct SASAEngine {
    coords: Vec<[f64; 3]>,
    radii: Vec<f64>,
    residue_indices: Vec<usize>,
    probe_radius: f64,
    sphere_points: Vec<[f64; 3]>,
    n_sphere_points: usize,
    kdtree: KDTree,
    // Cached values to avoid recomputation
    max_extended_radius: f64,
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

        // Build KD-tree for efficient neighbor queries
        let kdtree = KDTree::new(&coords);

        // Cache maximum extended radius (computed once, not per atom)
        let max_extended_radius =
            radii.iter().copied().fold(0.0f64, |a, b| a.max(b)) + probe_radius;

        SASAEngine {
            coords,
            radii,
            residue_indices,
            probe_radius,
            sphere_points,
            n_sphere_points,
            kdtree,
            max_extended_radius,
        }
    }

    /// Calculate SASA for a single atom (sequential over sphere points)
    /// Uses provided neighbor buffer to avoid allocation
    #[inline]
    fn calculate_atom_sasa(&self, atom_idx: usize, neighbors: &mut Vec<usize>) -> f64 {
        let atom_coord = &self.coords[atom_idx];
        let extended_radius = self.radii[atom_idx] + self.probe_radius;
        let extended_radius_sq = extended_radius * extended_radius;

        // Search radius uses cached max_extended_radius
        let search_radius = extended_radius + self.max_extended_radius;

        // Get potential neighbors using KD-tree (reuses buffer)
        self.kdtree
            .query_radius_into(atom_coord, search_radius, neighbors);

        // Count accessible points (sequential - no nested parallelism)
        // This is faster for typical sphere point counts (< 1000)
        let mut accessible_points = 0usize;

        for point in &self.sphere_points {
            // Scale point to atom's extended radius
            let test_point = [
                atom_coord[0] + point[0] * extended_radius,
                atom_coord[1] + point[1] * extended_radius,
                atom_coord[2] + point[2] * extended_radius,
            ];

            // Check if this point is buried by any neighbor
            let is_accessible = neighbors.iter().all(|&j| {
                if j == atom_idx {
                    return true;
                }

                let other_coord = &self.coords[j];
                let other_extended_radius = self.radii[j] + self.probe_radius;
                let dist_sq = distance_squared(&test_point, other_coord);
                dist_sq > other_extended_radius * other_extended_radius
            });

            if is_accessible {
                accessible_points += 1;
            }
        }

        // Calculate surface area
        let fraction_accessible = accessible_points as f64 / self.n_sphere_points as f64;
        let sphere_area = 4.0 * std::f64::consts::PI * extended_radius_sq;

        fraction_accessible * sphere_area
    }

    /// Calculate per-atom SASA in parallel (only one level of parallelism)
    fn calculate_per_atom_sasa(&self) -> Vec<f64> {
        // Parallel over atoms only - each thread gets its own neighbor buffer
        (0..self.coords.len())
            .into_par_iter()
            .map_init(
                || Vec::with_capacity(100), // Thread-local neighbor buffer
                |neighbors, i| self.calculate_atom_sasa(i, neighbors),
            )
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
///     Residue index for each atom (0-indexed, will accept any integer type)
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
    residue_indices: PyReadonlyArray1<i64>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<HashMap<String, PyObject>> {
    let coords = coordinates.as_array();
    let radii_arr = radii.as_array();
    let res_indices_i64 = residue_indices.as_array();

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
    if res_indices_i64.len() != n_atoms {
        return Err(PyValueError::new_err(
            "Residue indices must match number of atoms",
        ));
    }

    // Convert i64 to usize, checking for negative values
    let res_indices_vec: Result<Vec<usize>, _> = res_indices_i64
        .iter()
        .map(|&idx| {
            if idx < 0 {
                Err(PyValueError::new_err(format!(
                    "Residue index cannot be negative: {}",
                    idx
                )))
            } else {
                Ok(idx as usize)
            }
        })
        .collect();
    let res_indices_vec = res_indices_vec?;
    let res_indices = numpy::ndarray::ArrayView1::from(&res_indices_vec);

    // Create engine and calculate
    let engine = SASAEngine::new(
        coords,
        radii_arr,
        res_indices,
        probe_radius,
        n_sphere_points,
    );

    let per_atom = engine.calculate_per_atom_sasa();
    let per_residue = engine.calculate_per_residue_sasa();
    let total = per_atom.iter().sum::<f64>();

    // Convert to Python objects
    Python::with_gil(|py| {
        let mut result = HashMap::new();

        // Per-atom SASA as numpy array
        let per_atom_py = PyArray1::from_vec_bound(py, per_atom);
        result.insert("per_atom".to_string(), per_atom_py.into_py(py));

        // Per-residue SASA as dict
        let per_residue_dict = PyDict::new_bound(py);
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
///     Residue index for each atom (0-indexed, will accept any integer type)
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
    residue_indices: PyReadonlyArray1<i64>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<HashMap<usize, f64>> {
    let coords = coordinates.as_array();
    let radii_arr = radii.as_array();
    let res_indices_i64 = residue_indices.as_array();

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
    if res_indices_i64.len() != n_atoms {
        return Err(PyValueError::new_err(
            "Residue indices must match number of atoms",
        ));
    }

    // Convert i64 to usize, checking for negative values
    let res_indices_vec: Result<Vec<usize>, _> = res_indices_i64
        .iter()
        .map(|&idx| {
            if idx < 0 {
                Err(PyValueError::new_err(format!(
                    "Residue index cannot be negative: {}",
                    idx
                )))
            } else {
                Ok(idx as usize)
            }
        })
        .collect();
    let res_indices_vec = res_indices_vec?;
    let res_indices = numpy::ndarray::ArrayView1::from(&res_indices_vec);

    let engine = SASAEngine::new(
        coords,
        radii_arr,
        res_indices,
        probe_radius,
        n_sphere_points,
    );
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

/// Calculate SASA for an entire trajectory
///
/// Parameters
/// ----------
/// trajectory : np.ndarray, shape (n_frames, n_atoms, 3)
///     Atomic coordinates for all frames in Angstroms
/// radii : np.ndarray, shape (n_atoms,)
///     Atomic radii in Angstroms
/// residue_indices : np.ndarray, shape (n_atoms,)
///     Residue index for each atom (0-indexed, will accept any integer type)
/// probe_radius : float, optional
///     Probe radius in Angstroms (default: 1.4)
/// n_sphere_points : int, optional
///     Number of points on test sphere (default: 960)
///
/// Returns
/// -------
/// dict
///     Dictionary with keys:
///     - 'per_atom': np.ndarray, shape (n_frames, n_atoms)
///     - 'per_residue': list of dicts (one per frame)
///     - 'total': np.ndarray, shape (n_frames,)
#[pyfunction]
#[pyo3(signature = (trajectory, radii, residue_indices, probe_radius=1.4, n_sphere_points=960))]
pub fn calculate_sasa_trajectory<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray3<f64>,
    radii: PyReadonlyArray1<f64>,
    residue_indices: PyReadonlyArray1<i64>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<HashMap<String, PyObject>> {
    let traj = trajectory.as_array();
    let radii_arr = radii.as_array();
    let res_indices_i64 = residue_indices.as_array();

    let shape = traj.shape();
    let n_frames = shape[0];
    let n_atoms = shape[1];

    // Validate inputs
    if shape[2] != 3 {
        return Err(PyValueError::new_err(
            "Trajectory must have shape (n_frames, n_atoms, 3)",
        ));
    }
    if radii_arr.len() != n_atoms {
        return Err(PyValueError::new_err(format!(
            "Radii array length ({}) must match number of atoms ({})",
            radii_arr.len(),
            n_atoms
        )));
    }
    if res_indices_i64.len() != n_atoms {
        return Err(PyValueError::new_err(format!(
            "Residue indices length ({}) must match number of atoms ({})",
            res_indices_i64.len(),
            n_atoms
        )));
    }

    // Convert i64 to usize, checking for negative values
    let res_indices_vec: Result<Vec<usize>, _> = res_indices_i64
        .iter()
        .map(|&idx| {
            if idx < 0 {
                Err(PyValueError::new_err(format!(
                    "Residue index cannot be negative: {}",
                    idx
                )))
            } else {
                Ok(idx as usize)
            }
        })
        .collect();
    let res_indices_vec = res_indices_vec?;
    let res_indices = numpy::ndarray::ArrayView1::from(&res_indices_vec);

    // Pre-generate sphere points once (shared across all frames)
    let _sphere_points = generate_fibonacci_sphere(n_sphere_points);
    let _max_radius = radii_arr.iter().copied().fold(0.0f64, |a, b| a.max(b)) + probe_radius;

    // Process each frame in parallel
    let frame_results: Vec<_> = (0..n_frames)
        .into_par_iter()
        .map(|frame_idx| {
            // Extract frame coordinates directly from 3D array
            let frame_coords = traj.slice(numpy::ndarray::s![frame_idx, .., ..]);

            let engine = SASAEngine::new(
                frame_coords,
                radii_arr,
                res_indices,
                probe_radius,
                n_sphere_points,
            );

            let per_atom = engine.calculate_per_atom_sasa();
            let per_residue = engine.calculate_per_residue_sasa();
            let total: f64 = per_atom.iter().sum();

            (per_atom, per_residue, total)
        })
        .collect();

    // Convert to Python objects
    let mut result = HashMap::new();

    // Per-atom SASA as 2D numpy array (n_frames, n_atoms)
    let mut per_atom_vec = Vec::with_capacity(n_frames * n_atoms);
    for (per_atom, _, _) in &frame_results {
        per_atom_vec.extend_from_slice(per_atom);
    }
    let per_atom_array = PyArray1::from_vec_bound(py, per_atom_vec);
    result.insert("per_atom".to_string(), per_atom_array.into_py(py));

    // Per-residue SASA as list of dicts
    let per_residue_list = pyo3::types::PyList::empty_bound(py);
    for (_, per_residue, _) in &frame_results {
        let per_residue_dict = pyo3::types::PyDict::new_bound(py);
        for (&residue_idx, &sasa) in per_residue {
            per_residue_dict.set_item(residue_idx, sasa)?;
        }
        per_residue_list.append(per_residue_dict)?;
    }
    result.insert("per_residue".to_string(), per_residue_list.into_py(py));

    // Total SASA as 1D array
    let total_vec: Vec<f64> = frame_results.iter().map(|(_, _, total)| *total).collect();
    let total_py = PyArray1::from_vec_bound(py, total_vec);
    result.insert("total".to_string(), total_py.into_py(py));

    Ok(result)
}

/// Get Van der Waals radius for common elements
///
/// Parameters
/// ----------
/// element : str
///     Element symbol (e.g., 'C', 'N', 'O')
///
/// Returns
/// -------
/// float
///     Van der Waals radius in Angstroms
#[pyfunction]
pub fn get_vdw_radius(element: &str) -> f64 {
    match element.to_uppercase().as_str() {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "F" => 1.47,
        "CL" => 1.75,
        "BR" => 1.85,
        "I" => 1.98,
        "SE" => 1.90,
        "B" => 1.92,
        "SI" => 2.10,
        "AS" => 1.85,
        "HG" => 1.55,
        "CU" => 1.40,
        "ZN" => 1.39,
        "MG" => 1.73,
        "CA" => 2.31,
        "FE" => 1.32,
        "MN" => 1.61,
        "K" => 2.75,
        "NA" => 2.27,
        _ => 1.70, // Default to carbon radius
    }
}

/// Get radii array from element symbols
///
/// Parameters
/// ----------
/// elements : list of str
///     List of element symbols
///
/// Returns
/// -------
/// np.ndarray
///     Array of Van der Waals radii
#[pyfunction]
pub fn get_radii_array<'py>(
    py: Python<'py>,
    elements: Vec<String>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let radii: Vec<f64> = elements.iter().map(|e| get_vdw_radius(e)).collect();
    Ok(PyArray1::from_vec_bound(py, radii))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{arr1, Array2};

    #[test]
    fn test_kdtree_construction() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let kdtree = KDTree::new(&coords);
        assert!(kdtree.root.is_some());
        assert_eq!(kdtree.coords.len(), 4);
    }

    #[test]
    fn test_kdtree_query_radius() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ];

        let kdtree = KDTree::new(&coords);
        let mut neighbors = Vec::new();
        kdtree.query_radius_into(&[0.0, 0.0, 0.0], 1.5, &mut neighbors);

        // Should find points 0 and 1 (distances 0.0 and 1.0)
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
    }

    #[test]
    fn test_single_atom() {
        let coords = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let radii = arr1(&[1.5]);
        let res_indices = arr1(&[0]);

        let engine = SASAEngine::new(coords.view(), radii.view(), res_indices.view(), 1.4, 1000);
        let total = engine.calculate_total_sasa();

        let expected = 4.0 * std::f64::consts::PI * (1.5 + 1.4).powi(2);
        let error = (total - expected).abs() / expected;
        assert!(error < 0.05, "Error: {:.2}%", error * 100.0);
    }

    #[test]
    fn test_vdw_radii() {
        assert_eq!(get_vdw_radius("C"), 1.70);
        assert_eq!(get_vdw_radius("c"), 1.70); // Test case insensitivity
        assert_eq!(get_vdw_radius("N"), 1.55);
        assert_eq!(get_vdw_radius("O"), 1.52);
        assert_eq!(get_vdw_radius("H"), 1.20);
        assert_eq!(get_vdw_radius("S"), 1.80);
        assert_eq!(get_vdw_radius("UNKNOWN"), 1.70); // Default
    }

    #[test]
    fn test_fibonacci_sphere() {
        let points = generate_fibonacci_sphere(100);
        assert_eq!(points.len(), 100);

        // Check that points are approximately on unit sphere
        for point in points {
            let radius_sq = point[0].powi(2) + point[1].powi(2) + point[2].powi(2);
            let radius = radius_sq.sqrt();
            assert!(
                (radius - 1.0).abs() < 1e-10,
                "Point not on unit sphere: radius = {}",
                radius
            );
        }
    }
}
