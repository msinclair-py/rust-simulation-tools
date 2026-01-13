use numpy::ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

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
pub fn kabsch_align<'py>(
    py: Python<'py>,
    trajectory: &Bound<'py, PyAny>,
    reference: &Bound<'py, PyAny>,
    align_indices: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    // Convert indices to Vec<usize> from any integer array type
    let indices: Vec<usize> =
        if let Ok(idx_i64) = align_indices.extract::<PyReadonlyArray1<i64>>() {
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
        let result = kabsch_align_parallel(py, traj_f32, ref_f32, &indices)?;
        Ok(result.into_any().unbind())
    } else if let Ok(traj_f64) = trajectory.extract::<PyReadonlyArray3<f64>>() {
        let ref_f64 = reference.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "trajectory and reference must have the same dtype (both float32 or both float64)",
            )
        })?;
        let result = kabsch_align_parallel(py, traj_f64, ref_f64, &indices)?;
        Ok(result.into_any().unbind())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "trajectory must be float32 or float64 array",
        ))
    }
}

/// Parallelized Kabsch alignment implementation
fn kabsch_align_parallel<'py, T>(
    py: Python<'py>,
    trajectory: PyReadonlyArray3<'py, T>,
    reference: PyReadonlyArray2<'py, T>,
    align_indices: &[usize],
) -> PyResult<Bound<'py, PyArray3<T>>>
where
    T: numpy::Element + numpy::ndarray::NdFloat + Send + Sync,
{
    let traj = trajectory.as_array();
    let ref_coords = reference.as_array();

    let num_frames = traj.shape()[0];
    let num_atoms = traj.shape()[1];
    let _num_align = align_indices.len();

    // Pre-extract alignment atoms from reference (done once)
    let ref_align: Vec<[f64; 3]> = align_indices
        .iter()
        .map(|&idx| {
            [
                ref_coords[[idx, 0]].to_f64().unwrap(),
                ref_coords[[idx, 1]].to_f64().unwrap(),
                ref_coords[[idx, 2]].to_f64().unwrap(),
            ]
        })
        .collect();

    // Calculate reference centroid once
    let ref_centroid = compute_centroid(&ref_align);

    // Pre-center reference alignment atoms
    let ref_centered: Vec<[f64; 3]> = ref_align
        .iter()
        .map(|p| {
            [
                p[0] - ref_centroid[0],
                p[1] - ref_centroid[1],
                p[2] - ref_centroid[2],
            ]
        })
        .collect();

    // Process frames in parallel
    let results: Vec<Vec<T>> = (0..num_frames)
        .into_par_iter()
        .map(|frame_idx| {
            // Extract alignment atoms from current frame
            let mobile_align: Vec<[f64; 3]> = align_indices
                .iter()
                .map(|&idx| {
                    [
                        traj[[frame_idx, idx, 0]].to_f64().unwrap(),
                        traj[[frame_idx, idx, 1]].to_f64().unwrap(),
                        traj[[frame_idx, idx, 2]].to_f64().unwrap(),
                    ]
                })
                .collect();

            // Calculate mobile centroid
            let mobile_centroid = compute_centroid(&mobile_align);

            // Compute rotation matrix using Kabsch algorithm
            let rotation = compute_kabsch_rotation(&mobile_align, &mobile_centroid, &ref_centered);

            // Apply alignment to ALL atoms in the frame
            let mut frame_result = Vec::with_capacity(num_atoms * 3);

            for atom_idx in 0..num_atoms {
                // Get atom coordinates and center on mobile centroid
                let centered = [
                    traj[[frame_idx, atom_idx, 0]].to_f64().unwrap() - mobile_centroid[0],
                    traj[[frame_idx, atom_idx, 1]].to_f64().unwrap() - mobile_centroid[1],
                    traj[[frame_idx, atom_idx, 2]].to_f64().unwrap() - mobile_centroid[2],
                ];

                // Apply rotation (manually unrolled 3x3 matrix multiply)
                let rotated = [
                    rotation[0][0] * centered[0]
                        + rotation[0][1] * centered[1]
                        + rotation[0][2] * centered[2],
                    rotation[1][0] * centered[0]
                        + rotation[1][1] * centered[1]
                        + rotation[1][2] * centered[2],
                    rotation[2][0] * centered[0]
                        + rotation[2][1] * centered[1]
                        + rotation[2][2] * centered[2],
                ];

                // Translate to reference centroid and store
                frame_result.push(T::from(rotated[0] + ref_centroid[0]).unwrap());
                frame_result.push(T::from(rotated[1] + ref_centroid[1]).unwrap());
                frame_result.push(T::from(rotated[2] + ref_centroid[2]).unwrap());
            }

            frame_result
        })
        .collect();

    // Flatten results into output array
    let aligned_data: Vec<T> = results.into_iter().flatten().collect();

    let aligned_array = Array1::from_vec(aligned_data)
        .into_shape((num_frames, num_atoms, 3))
        .expect("Failed to reshape to 3D");

    Ok(PyArray3::from_owned_array_bound(py, aligned_array))
}

/// Compute centroid of a set of 3D points
#[inline]
fn compute_centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len() as f64;
    let mut sum = [0.0; 3];
    for p in points {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Compute Kabsch rotation matrix
/// Takes mobile points, mobile centroid, and pre-centered reference points
#[inline]
fn compute_kabsch_rotation(
    mobile: &[[f64; 3]],
    mobile_centroid: &[f64; 3],
    ref_centered: &[[f64; 3]],
) -> [[f64; 3]; 3] {
    // Compute covariance matrix H = mobile_centered^T * ref_centered
    let mut h = [[0.0f64; 3]; 3];
    let n = mobile.len();

    for k in 0..n {
        let m_centered = [
            mobile[k][0] - mobile_centroid[0],
            mobile[k][1] - mobile_centroid[1],
            mobile[k][2] - mobile_centroid[2],
        ];

        // Outer product contribution to H
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += m_centered[i] * ref_centered[k][j];
            }
        }
    }

    // Convert to nalgebra for SVD
    let mat = nalgebra::Matrix3::new(
        h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
    );

    let svd = nalgebra::SVD::new(mat, true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

    // Compute rotation matrix R = V * U^T
    let v = v_t.transpose();
    let mut r = v * u.transpose();

    // Ensure proper rotation (det(R) = 1, not reflection)
    if r.determinant() < 0.0 {
        let mut v_corrected = v;
        for i in 0..3 {
            v_corrected[(i, 2)] *= -1.0;
        }
        r = v_corrected * u.transpose();
    }

    // Convert back to array
    [
        [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
        [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
        [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
    ]
}

/// Generic centroid function for ndarray (kept for compatibility)
#[allow(dead_code)]
fn centroid_generic<T>(coords: &ArrayView2<T>) -> Array2<T>
where
    T: numpy::ndarray::NdFloat,
{
    let n = T::from(coords.nrows()).unwrap();
    let sum = coords.sum_axis(Axis(0));
    sum.insert_axis(Axis(0)) / n
}
