use ndarray::{s, Array2, ArrayView2, Axis};
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

/// Align MD trajectory frames to a reference structure using Kabsch algorithm
///
/// Parameters
/// ----------
/// trajectory : ndarray (num_frames, num_atoms, 3)
///     Trajectory coordinates to align (float32 or float64)
/// reference : ndarray (num_atoms, 3)
///     Reference structure coordinates (float32 or float64)
/// align_indices : ndarray (num_align_atoms,)
///     Indices of atoms to use for alignment calculation (any integer type)
///
/// Returns
/// -------
/// aligned : ndarray (num_frames, num_atoms, 3)
///     Aligned trajectory coordinates (same dtype as input)
#[pyfunction]
fn kabsch_align(
    py: Python,
    trajectory: &PyAny,
    reference: &PyAny,
    align_indices: &PyAny,
) -> PyResult<PyObject> {
    // Convert indices to Vec<usize> from any integer array type
    let indices: Vec<usize> = if let Ok(idx_i64) = align_indices.extract::<PyReadonlyArray1<i64>>()
    {
        idx_i64.as_array().iter().map(|&x| x as usize).collect()
    } else if let Ok(idx_i32) = align_indices.extract::<PyReadonlyArray1<i32>>() {
        idx_i32.as_array().iter().map(|&x| x as usize).collect()
    } else if let Ok(idx_u64) = align_indices.extract::<PyReadonlyArray1<u64>>() {
        idx_u64.as_array().iter().map(|&x| x as usize).collect()
    } else if let Ok(idx_usize) = align_indices.extract::<PyReadonlyArray1<usize>>() {
        idx_usize.as_array().to_vec()
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "align_indices must be an integer array",
        ));
    };

    // Check if input is float32 or float64 and dispatch accordingly
    if let Ok(traj_f32) = trajectory.extract::<PyReadonlyArray3<f32>>() {
        let ref_f32 = reference.extract::<PyReadonlyArray2<f32>>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "trajectory and reference must have the same dtype (both float32 or both float64)",
            )
        })?;
        let result = kabsch_align_generic(py, traj_f32, ref_f32, &indices)?;
        Ok(result.to_object(py))
    } else if let Ok(traj_f64) = trajectory.extract::<PyReadonlyArray3<f64>>() {
        let ref_f64 = reference.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "trajectory and reference must have the same dtype (both float32 or both float64)",
            )
        })?;
        let result = kabsch_align_generic(py, traj_f64, ref_f64, &indices)?;
        Ok(result.to_object(py))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "trajectory must be float32 or float64 array",
        ))
    }
}

/// Generic implementation that works with both f32 and f64
fn kabsch_align_generic<'py, T>(
    py: Python<'py>,
    trajectory: PyReadonlyArray3<T>,
    reference: PyReadonlyArray2<T>,
    align_indices: &[usize],
) -> PyResult<&'py PyArray3<T>>
where
    T: numpy::Element + ndarray::NdFloat,
{
    let traj = trajectory.as_array();
    let ref_coords = reference.as_array();

    let num_frames = traj.shape()[0];
    let num_atoms = traj.shape()[1];
    let num_align = align_indices.len();

    // Extract alignment atoms from reference
    let mut ref_align = Array2::<T>::zeros((num_align, 3));
    for (i, &idx) in align_indices.iter().enumerate() {
        for j in 0..3 {
            ref_align[[i, j]] = ref_coords[[idx, j]];
        }
    }

    // Calculate reference centroid once
    let ref_centroid = centroid_generic(&ref_align.view());

    // Allocate output array
    let mut aligned_data = Vec::with_capacity(num_frames * num_atoms * 3);

    // Process each frame
    for frame_idx in 0..num_frames {
        let frame = traj.slice(s![frame_idx, .., ..]);

        // Extract alignment atoms from current frame
        let mut mobile_align = Array2::<T>::zeros((num_align, 3));
        for (i, &idx) in align_indices.iter().enumerate() {
            for j in 0..3 {
                mobile_align[[i, j]] = frame[[idx, j]];
            }
        }

        // Calculate mobile centroid
        let mobile_centroid = centroid_generic(&mobile_align.view());

        // Compute rotation matrix
        let rotation = kabsch_generic(&mobile_align.view(), &ref_align.view());

        // Apply alignment to ALL atoms in the frame
        for atom_idx in 0..num_atoms {
            // Get atom coordinates
            let atom_coords = [
                frame[[atom_idx, 0]],
                frame[[atom_idx, 1]],
                frame[[atom_idx, 2]],
            ];

            // Step 1: Center on mobile centroid
            let mut centered = atom_coords;
            for j in 0..3 {
                centered[j] -= mobile_centroid[[0, j]];
            }

            // Step 2: Apply rotation (matrix-vector multiplication)
            let mut rotated = [T::zero(); 3];
            for i in 0..3 {
                for j in 0..3 {
                    rotated[i] += rotation[[i, j]] * centered[j];
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
    let aligned_3d = aligned_array
        .into_shape((num_frames, num_atoms, 3))
        .expect("Failed to reshape to 3D");

    Ok(PyArray3::from_owned_array(py, aligned_3d))
}

/// Generic centroid function
fn centroid_generic<T>(coords: &ArrayView2<T>) -> Array2<T>
where
    T: ndarray::NdFloat,
{
    let n = T::from(coords.nrows()).unwrap();
    let sum = coords.sum_axis(Axis(0));
    sum.insert_axis(Axis(0)) / n
}

/// Generic Kabsch algorithm
fn kabsch_generic<T>(mobile: &ArrayView2<T>, reference: &ArrayView2<T>) -> Array2<T>
where
    T: ndarray::NdFloat,
{
    // Center both coordinate sets
    let mobile_center = centroid_generic(mobile);
    let ref_center = centroid_generic(reference);

    let mobile_centered = mobile.to_owned() - &mobile_center;
    let ref_centered = reference.to_owned() - &ref_center;

    // Compute covariance matrix H = mobile^T * reference
    let h = mobile_centered.t().dot(&ref_centered);

    // Convert to nalgebra for SVD (always use f64 for numerical stability)
    let mut mat = nalgebra::Matrix3::<f64>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            mat[(i, j)] = h[[i, j]].to_f64().unwrap();
        }
    }

    let svd = nalgebra::SVD::new(mat, true, true);
    let u_na = svd.u.unwrap();
    let v_t_na = svd.v_t.unwrap();

    // Compute rotation matrix R = V * U^T
    let v = v_t_na.transpose();
    let mut r = v * u_na.transpose();

    // Ensure proper rotation (det(R) = 1, not reflection)
    let det = r.determinant();

    if det < 0.0 {
        let mut v_corrected = v;
        for i in 0..3 {
            v_corrected[(i, 2)] *= -1.0;
        }
        r = v_corrected * u_na.transpose();
    }

    // Convert back to ndarray and original type T
    let mut rotation = Array2::<T>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            rotation[[i, j]] = T::from(r[(i, j)]).unwrap();
        }
    }

    rotation
}

