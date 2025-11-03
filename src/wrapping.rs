use ndarray::{s, Array2};
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

/// Unwrap a molecular dynamics trajectory to remove periodic boundary artifacts
///
/// This function ensures molecules remain whole across periodic boundaries by
/// unwrapping coordinates based on fragment (molecule) assignments.
///
/// Parameters
/// ----------
/// trajectory : ndarray (num_frames, num_atoms, 3)
///     Trajectory coordinates (wrapped, with periodic boundaries) (float32 or float64)
/// box_dimensions : ndarray (num_frames, 3) or (num_frames, 6)
///     Box dimensions for each frame (float32 or float64)
/// fragment_indices : ndarray (num_atoms,)
///     Fragment/molecule ID for each atom (any integer type)
///
/// Returns
/// -------
/// unwrapped : ndarray (num_frames, num_atoms, 3)
///     Unwrapped trajectory coordinates (same dtype as input)
#[pyfunction]
pub fn unwrap_system(
    py: Python,
    trajectory: &PyAny,
    box_dimensions: &PyAny,
    fragment_indices: &PyAny,
) -> PyResult<PyObject> {
    // Convert indices to Vec<usize> from any integer array type
    let indices: Vec<usize> =
        if let Ok(idx_i64) = fragment_indices.extract::<PyReadonlyArray1<i64>>() {
            idx_i64.as_array().iter().map(|&x| x as usize).collect()
        } else if let Ok(idx_i32) = fragment_indices.extract::<PyReadonlyArray1<i32>>() {
            idx_i32.as_array().iter().map(|&x| x as usize).collect()
        } else if let Ok(idx_u64) = fragment_indices.extract::<PyReadonlyArray1<u64>>() {
            idx_u64.as_array().iter().map(|&x| x as usize).collect()
        } else if let Ok(idx_usize) = fragment_indices.extract::<PyReadonlyArray1<usize>>() {
            idx_usize.as_array().to_vec()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "fragment_indices must be an integer array",
            ));
        };

    // Check if input is float32 or float64 and dispatch accordingly
    if let Ok(traj_f32) = trajectory.extract::<PyReadonlyArray3<f32>>() {
        let box_f32 = box_dimensions.extract::<PyReadonlyArray2<f32>>()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "trajectory and box_dimensions must have the same dtype (both float32 or both float64)"
            ))?;
        let result = unwrap_system_generic(py, traj_f32, box_f32, &indices)?;
        Ok(result.to_object(py))
    } else if let Ok(traj_f64) = trajectory.extract::<PyReadonlyArray3<f64>>() {
        let box_f64 = box_dimensions.extract::<PyReadonlyArray2<f64>>()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "trajectory and box_dimensions must have the same dtype (both float32 or both float64)"
            ))?;
        let result = unwrap_system_generic(py, traj_f64, box_f64, &indices)?;
        Ok(result.to_object(py))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "trajectory must be float32 or float64 array",
        ))
    }
}

/// Generic unwrap implementation
fn unwrap_system_generic<'py, T>(
    py: Python<'py>,
    trajectory: PyReadonlyArray3<T>,
    box_dimensions: PyReadonlyArray2<T>,
    fragment_indices: &[usize],
) -> PyResult<&'py PyArray3<T>>
where
    T: numpy::Element + ndarray::NdFloat,
{
    let traj = trajectory.as_array();
    let boxes = box_dimensions.as_array();

    let num_frames = traj.shape()[0];
    let num_atoms = traj.shape()[1];

    // Validate inputs
    if boxes.shape()[0] != num_frames {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of box dimensions must match number of frames",
        ));
    }

    if fragment_indices.len() != num_atoms {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Fragment indices length must match number of atoms",
        ));
    }

    // Check box dimensions format (should be 3 or 6 columns)
    let box_cols = boxes.shape()[1];
    if box_cols != 3 && box_cols != 6 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Box dimensions must have 3 (orthogonal) or 6 (triclinic) columns",
        ));
    }

    // For now, only support orthogonal boxes
    if box_cols == 6 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Triclinic boxes not yet supported",
        ));
    }

    // Initialize unwrapped coordinates with first frame
    let mut unwrapped = Array2::<T>::zeros((num_frames * num_atoms, 3));

    // Copy first frame as-is
    for atom_idx in 0..num_atoms {
        for dim in 0..3 {
            unwrapped[[atom_idx, dim]] = traj[[0, atom_idx, dim]];
        }
    }

    // Track cumulative shifts for each atom
    let mut cumulative_shifts: Array2<T> = Array2::zeros((num_atoms, 3));

    // Process each subsequent frame
    for frame_idx in 1..num_frames {
        let prev_frame = traj.slice(s![frame_idx - 1, .., ..]);
        let curr_frame = traj.slice(s![frame_idx, .., ..]);
        let box_dims = [
            boxes[[frame_idx, 0]],
            boxes[[frame_idx, 1]],
            boxes[[frame_idx, 2]],
        ];

        // For each atom, detect and correct for boundary crossings
        for atom_idx in 0..num_atoms {
            for dim in 0..3 {
                let curr_pos = curr_frame[[atom_idx, dim]];
                let prev_pos = prev_frame[[atom_idx, dim]];
                let delta = curr_pos - prev_pos;

                // Use minimum image convention: if displacement > box/2, wrapped
                let half_box = box_dims[dim] / T::from(2.0).unwrap();

                if delta > half_box {
                    // Wrapped backward (right to left)
                    cumulative_shifts[[atom_idx, dim]] -= box_dims[dim];
                } else if delta < -half_box {
                    // Wrapped forward (left to right)
                    cumulative_shifts[[atom_idx, dim]] += box_dims[dim];
                }

                // Store unwrapped coordinate
                let out_idx = frame_idx * num_atoms + atom_idx;
                unwrapped[[out_idx, dim]] = curr_pos + cumulative_shifts[[atom_idx, dim]];
            }
        }
    }

    // Reshape to (num_frames, num_atoms, 3)
    let unwrapped_3d = unwrapped
        .into_shape((num_frames, num_atoms, 3))
        .expect("Failed to reshape to 3D");

    Ok(PyArray3::from_owned_array(py, unwrapped_3d))
}

