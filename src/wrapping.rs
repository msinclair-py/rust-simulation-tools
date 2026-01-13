use numpy::ndarray::{Array1, ArrayView2, ArrayView3};
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Unwrap a molecular dynamics trajectory to remove periodic boundary artifacts
///
/// This function ensures molecules remain whole across periodic boundaries by
/// unwrapping coordinates based on the minimum image convention.
///
/// Parameters
/// ----------
/// trajectory : ndarray (num_frames, num_atoms, 3)
///     Trajectory coordinates (wrapped, with periodic boundaries) (float32 or float64)
/// box_dimensions : ndarray (num_frames, 3) or (num_frames, 6)
///     Box dimensions for each frame (float32 or float64)
/// fragment_indices : ndarray (num_atoms,)
///     Fragment/molecule ID for each atom (any integer type) - reserved for future use
///
/// Returns
/// -------
/// unwrapped : ndarray (num_frames, num_atoms, 3)
///     Unwrapped trajectory coordinates (same dtype as input)
#[pyfunction]
pub fn unwrap_system<'py>(
    py: Python<'py>,
    trajectory: &Bound<'py, PyAny>,
    box_dimensions: &Bound<'py, PyAny>,
    fragment_indices: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    // Convert indices to Vec<usize> from any integer array type
    // Note: fragment_indices is validated but reserved for future molecule-aware unwrapping
    let _indices: Vec<usize> =
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
        let box_f32 = box_dimensions.extract::<PyReadonlyArray2<f32>>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "trajectory and box_dimensions must have the same dtype (both float32 or both float64)",
            )
        })?;
        let result = unwrap_system_optimized(py, traj_f32.as_array(), box_f32.as_array())?;
        Ok(result.into_any().unbind())
    } else if let Ok(traj_f64) = trajectory.extract::<PyReadonlyArray3<f64>>() {
        let box_f64 = box_dimensions.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "trajectory and box_dimensions must have the same dtype (both float32 or both float64)",
            )
        })?;
        let result = unwrap_system_optimized(py, traj_f64.as_array(), box_f64.as_array())?;
        Ok(result.into_any().unbind())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "trajectory must be float32 or float64 array",
        ))
    }
}

/// Optimized unwrap implementation with better cache locality and parallelization
fn unwrap_system_optimized<'py, T>(
    py: Python<'py>,
    trajectory: ArrayView3<'py, T>,
    box_dimensions: ArrayView2<'py, T>,
) -> PyResult<Bound<'py, PyArray3<T>>>
where
    T: numpy::Element + numpy::ndarray::NdFloat + Send + Sync,
{
    let num_frames = trajectory.shape()[0];
    let num_atoms = trajectory.shape()[1];

    // Validate inputs
    if box_dimensions.shape()[0] != num_frames {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of box dimensions must match number of frames",
        ));
    }

    let box_cols = box_dimensions.shape()[1];
    if box_cols != 3 && box_cols != 6 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Box dimensions must have 3 (orthogonal) or 6 (triclinic) columns",
        ));
    }

    if box_cols == 6 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Triclinic boxes not yet supported",
        ));
    }

    // Pre-extract box dimensions for all frames (avoids repeated indexing)
    let two = T::from(2.0).unwrap();
    let box_data: Vec<([T; 3], [T; 3])> = (0..num_frames)
        .map(|f| {
            let bx = box_dimensions[[f, 0]];
            let by = box_dimensions[[f, 1]];
            let bz = box_dimensions[[f, 2]];
            ([bx, by, bz], [bx / two, by / two, bz / two])
        })
        .collect();

    // Allocate output as flat Vec for optimal memory layout
    // Layout: [frame0_atom0_xyz, frame0_atom1_xyz, ..., frame1_atom0_xyz, ...]
    let mut unwrapped = vec![T::zero(); num_frames * num_atoms * 3];

    // Copy first frame directly (no unwrapping needed)
    for atom_idx in 0..num_atoms {
        let base = atom_idx * 3;
        unwrapped[base] = trajectory[[0, atom_idx, 0]];
        unwrapped[base + 1] = trajectory[[0, atom_idx, 1]];
        unwrapped[base + 2] = trajectory[[0, atom_idx, 2]];
    }

    // Use Structure-of-Arrays for cumulative shifts (better cache locality)
    let mut shifts_x = vec![T::zero(); num_atoms];
    let mut shifts_y = vec![T::zero(); num_atoms];
    let mut shifts_z = vec![T::zero(); num_atoms];

    // Process frames sequentially (shift accumulation has frame dependencies)
    for frame_idx in 1..num_frames {
        let (box_dims, half_box) = &box_data[frame_idx];
        let frame_offset = frame_idx * num_atoms * 3;

        // Process atoms - this inner loop could potentially be parallelized
        // but for typical MD systems the overhead isn't worth it
        for atom_idx in 0..num_atoms {
            // Load current and previous positions (all 3 coords together)
            let curr_x = trajectory[[frame_idx, atom_idx, 0]];
            let curr_y = trajectory[[frame_idx, atom_idx, 1]];
            let curr_z = trajectory[[frame_idx, atom_idx, 2]];

            let prev_x = trajectory[[frame_idx - 1, atom_idx, 0]];
            let prev_y = trajectory[[frame_idx - 1, atom_idx, 1]];
            let prev_z = trajectory[[frame_idx - 1, atom_idx, 2]];

            // Compute deltas
            let delta_x = curr_x - prev_x;
            let delta_y = curr_y - prev_y;
            let delta_z = curr_z - prev_z;

            // Update cumulative shifts using minimum image convention
            // Wrapped backward (crossed right boundary)
            if delta_x > half_box[0] {
                shifts_x[atom_idx] -= box_dims[0];
            } else if delta_x < -half_box[0] {
                shifts_x[atom_idx] += box_dims[0];
            }

            if delta_y > half_box[1] {
                shifts_y[atom_idx] -= box_dims[1];
            } else if delta_y < -half_box[1] {
                shifts_y[atom_idx] += box_dims[1];
            }

            if delta_z > half_box[2] {
                shifts_z[atom_idx] -= box_dims[2];
            } else if delta_z < -half_box[2] {
                shifts_z[atom_idx] += box_dims[2];
            }

            // Store unwrapped coordinates (all 3 together for cache efficiency)
            let out_base = frame_offset + atom_idx * 3;
            unwrapped[out_base] = curr_x + shifts_x[atom_idx];
            unwrapped[out_base + 1] = curr_y + shifts_y[atom_idx];
            unwrapped[out_base + 2] = curr_z + shifts_z[atom_idx];
        }
    }

    // Convert to ndarray with proper shape
    let unwrapped_array = Array1::from_vec(unwrapped)
        .into_shape((num_frames, num_atoms, 3))
        .expect("Failed to reshape to 3D");

    Ok(PyArray3::from_owned_array_bound(py, unwrapped_array))
}

/// Parallel unwrap for very large systems (>50k atoms)
/// Uses a two-phase approach: compute shift deltas, then parallel apply
#[allow(dead_code)]
fn unwrap_system_parallel<'py, T>(
    py: Python<'py>,
    trajectory: ArrayView3<'py, T>,
    box_dimensions: ArrayView2<'py, T>,
) -> PyResult<Bound<'py, PyArray3<T>>>
where
    T: numpy::Element + numpy::ndarray::NdFloat + Send + Sync,
{
    let num_frames = trajectory.shape()[0];
    let num_atoms = trajectory.shape()[1];

    // Pre-extract box dimensions
    let two = T::from(2.0).unwrap();
    let box_data: Vec<([T; 3], [T; 3])> = (0..num_frames)
        .map(|f| {
            let bx = box_dimensions[[f, 0]];
            let by = box_dimensions[[f, 1]];
            let bz = box_dimensions[[f, 2]];
            ([bx, by, bz], [bx / two, by / two, bz / two])
        })
        .collect();

    // Phase 1: Compute per-frame shift deltas (what shift changes occurred this frame)
    // This is parallelizable because each frame's delta only depends on that frame and the previous
    let shift_deltas: Vec<Vec<[T; 3]>> = (1..num_frames)
        .into_par_iter()
        .map(|frame_idx| {
            let (box_dims, half_box) = &box_data[frame_idx];
            let mut deltas = vec![[T::zero(); 3]; num_atoms];

            for atom_idx in 0..num_atoms {
                let curr_x = trajectory[[frame_idx, atom_idx, 0]];
                let curr_y = trajectory[[frame_idx, atom_idx, 1]];
                let curr_z = trajectory[[frame_idx, atom_idx, 2]];

                let prev_x = trajectory[[frame_idx - 1, atom_idx, 0]];
                let prev_y = trajectory[[frame_idx - 1, atom_idx, 1]];
                let prev_z = trajectory[[frame_idx - 1, atom_idx, 2]];

                let delta_x = curr_x - prev_x;
                let delta_y = curr_y - prev_y;
                let delta_z = curr_z - prev_z;

                if delta_x > half_box[0] {
                    deltas[atom_idx][0] = -box_dims[0];
                } else if delta_x < -half_box[0] {
                    deltas[atom_idx][0] = box_dims[0];
                }

                if delta_y > half_box[1] {
                    deltas[atom_idx][1] = -box_dims[1];
                } else if delta_y < -half_box[1] {
                    deltas[atom_idx][1] = box_dims[1];
                }

                if delta_z > half_box[2] {
                    deltas[atom_idx][2] = -box_dims[2];
                } else if delta_z < -half_box[2] {
                    deltas[atom_idx][2] = box_dims[2];
                }
            }
            deltas
        })
        .collect();

    // Phase 2: Compute cumulative shifts via prefix sum (sequential, but O(frames))
    let mut cumulative_shifts: Vec<Vec<[T; 3]>> = Vec::with_capacity(num_frames);
    cumulative_shifts.push(vec![[T::zero(); 3]; num_atoms]); // Frame 0 has no shifts

    for frame_idx in 1..num_frames {
        let prev_shifts = &cumulative_shifts[frame_idx - 1];
        let deltas = &shift_deltas[frame_idx - 1];
        let mut curr_shifts = vec![[T::zero(); 3]; num_atoms];

        for atom_idx in 0..num_atoms {
            curr_shifts[atom_idx][0] = prev_shifts[atom_idx][0] + deltas[atom_idx][0];
            curr_shifts[atom_idx][1] = prev_shifts[atom_idx][1] + deltas[atom_idx][1];
            curr_shifts[atom_idx][2] = prev_shifts[atom_idx][2] + deltas[atom_idx][2];
        }
        cumulative_shifts.push(curr_shifts);
    }

    // Phase 3: Apply shifts in parallel across frames
    let unwrapped: Vec<T> = (0..num_frames)
        .into_par_iter()
        .flat_map(|frame_idx| {
            let shifts = &cumulative_shifts[frame_idx];
            let mut frame_data = Vec::with_capacity(num_atoms * 3);

            for atom_idx in 0..num_atoms {
                frame_data.push(trajectory[[frame_idx, atom_idx, 0]] + shifts[atom_idx][0]);
                frame_data.push(trajectory[[frame_idx, atom_idx, 1]] + shifts[atom_idx][1]);
                frame_data.push(trajectory[[frame_idx, atom_idx, 2]] + shifts[atom_idx][2]);
            }
            frame_data
        })
        .collect();

    let unwrapped_array = Array1::from_vec(unwrapped)
        .into_shape((num_frames, num_atoms, 3))
        .expect("Failed to reshape to 3D");

    Ok(PyArray3::from_owned_array_bound(py, unwrapped_array))
}
