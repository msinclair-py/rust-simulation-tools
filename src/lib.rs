use pyo3::prelude::*;
use numpy::{PyArray3, PyReadonlyArray3, PyReadonlyArray2, PyReadonlyArray1};
use ndarray::{Array2, ArrayView2, s, Axis};

/// Compute the centroid of a set of coordinates
fn centroid(coords: &ArrayView2<f64>) -> Array2<f64> {
    let n = coords.nrows() as f64;
    let sum = coords.sum_axis(Axis(0));
    sum.insert_axis(Axis(0)) / n
}

/// Simple 3x3 matrix determinant
fn det3x3(m: &Array2<f64>) -> f64 {
    m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]]) -
    m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]]) +
    m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]])
}

/// SVD for 3x3 matrices using nalgebra
fn svd_3x3(h: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    use nalgebra::{Matrix3, SVD};
    
    let mut mat = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            mat[(i, j)] = h[[i, j]];
        }
    }
    
    let svd = SVD::new(mat, true, true);
    let u_na = svd.u.unwrap();
    let v_t_na = svd.v_t.unwrap();
    
    // Convert back to ndarray
    let mut u = Array2::zeros((3, 3));
    let mut vt = Array2::zeros((3, 3));
    
    for i in 0..3 {
        for j in 0..3 {
            u[[i, j]] = u_na[(i, j)];
            vt[[i, j]] = v_t_na[(i, j)];
        }
    }
    
    (u, vt)
}

/// Kabsch algorithm to find optimal rotation matrix
/// Aligns mobile (P) coordinates to reference (Q) coordinates
/// Returns rotation matrix R such that P_aligned = (P - P_center) @ R + Q_center
fn kabsch(mobile: &ArrayView2<f64>, reference: &ArrayView2<f64>) -> Array2<f64> {
    // Center both coordinate sets
    let mobile_center = centroid(mobile);
    let ref_center = centroid(reference);
    
    let mobile_centered = mobile.to_owned() - &mobile_center;
    let ref_centered = reference.to_owned() - &ref_center;
    
    // Compute covariance matrix H = mobile^T * reference
    let h = mobile_centered.t().dot(&ref_centered);
    
    // SVD decomposition: H = U * S * V^T
    let (u, vt) = svd_3x3(&h);
    
    // Compute rotation matrix R = V * U^T
    let v = vt.t();
    let mut r = v.dot(&u.t());
    
    // Ensure proper rotation (det(R) = 1, not reflection)
    let det = det3x3(&r);
    
    if det < 0.0 {
        // Flip the sign of the third column of V
        let mut v_corrected = v.to_owned();
        for i in 0..3 {
            v_corrected[[i, 2]] *= -1.0;
        }
        r = v_corrected.dot(&u.t());
    }
    
    r
}

/// Align MD trajectory frames to a reference structure using Kabsch algorithm
/// 
/// Parameters
/// ----------
/// trajectory : ndarray (num_frames, num_atoms, 3)
///     Trajectory coordinates to align
/// reference : ndarray (num_atoms, 3)
///     Reference structure coordinates
/// align_indices : ndarray (num_align_atoms,)
///     Indices of atoms to use for alignment calculation
/// 
/// Returns
/// -------
/// aligned : ndarray (num_frames, num_atoms, 3)
///     Aligned trajectory coordinates
#[pyfunction]
fn kabsch_align<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray3<f64>,
    reference: PyReadonlyArray2<f64>,
    align_indices: PyReadonlyArray1<usize>,
) -> PyResult<&'py PyArray3<f64>> {
    let traj = trajectory.as_array();
    let ref_coords = reference.as_array();
    let indices = align_indices.as_array();
    
    let num_frames = traj.shape()[0];
    let num_atoms = traj.shape()[1];
    let num_align = indices.len();
    
    // Extract alignment atoms from reference
    let mut ref_align = Array2::zeros((num_align, 3));
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..3 {
            ref_align[[i, j]] = ref_coords[[idx, j]];
        }
    }
    
    // Calculate reference centroid once
    let ref_centroid = centroid(&ref_align.view());
    
    // Allocate output array
    let mut aligned_data = Vec::with_capacity(num_frames * num_atoms * 3);
    
    // Process each frame
    for frame_idx in 0..num_frames {
        let frame = traj.slice(s![frame_idx, .., ..]);
        
        // Extract alignment atoms from current frame
        let mut mobile_align = Array2::zeros((num_align, 3));
        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..3 {
                mobile_align[[i, j]] = frame[[idx, j]];
            }
        }
        
        // Calculate mobile centroid
        let mobile_centroid = centroid(&mobile_align.view());
        
        // Compute rotation matrix
        let rotation = kabsch(&mobile_align.view(), &ref_align.view());
        
        // Apply alignment to ALL atoms in the frame
        for atom_idx in 0..num_atoms {
            // Get atom coordinates
            let mut atom_coords = [frame[[atom_idx, 0]], frame[[atom_idx, 1]], frame[[atom_idx, 2]]];
            
            // Step 1: Center on mobile centroid
            for j in 0..3 {
                atom_coords[j] -= mobile_centroid[[0, j]];
            }
            
            // Step 2: Apply rotation (matrix-vector multiplication)
            // rotated = rotation @ atom_coords
            let mut rotated = [0.0, 0.0, 0.0];
            for i in 0..3 {
                for j in 0..3 {
                    rotated[i] += rotation[[i, j]] * atom_coords[j];
                }
            }
            
            // Step 3: Translate to reference centroid
            for j in 0..3 {
                rotated[j] += ref_centroid[[0, j]];
            }
            
            // Store in output
            aligned_data.extend_from_slice(&rotated);
        }
    }
    
    // Convert to ndarray and reshape
    let aligned_array = Array2::from_shape_vec((num_frames * num_atoms, 3), aligned_data)
        .expect("Failed to create array from aligned data");
    let aligned_3d = aligned_array.into_shape((num_frames, num_atoms, 3))
        .expect("Failed to reshape to 3D");
    
    Ok(PyArray3::from_owned_array(py, aligned_3d))
}

/// A Python module implemented in Rust for fast Kabsch alignment
#[pymodule]
fn rust_simulation_tools(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kabsch_align, m)?)?;
    m.add("__version__", "0.2.0")?;  // Changed version to verify rebuild
    Ok(())
}
