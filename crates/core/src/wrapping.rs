//! Periodic boundary unwrapping for trajectory continuity.

/// Unwrap a trajectory to remove periodic boundary artifacts.
///
/// # Arguments
/// * `trajectory` - Frames of [n_atoms] positions
/// * `box_dimensions` - Box [x, y, z] per frame
///
/// # Returns
/// Unwrapped trajectory
pub fn unwrap_system(
    trajectory: &[Vec<[f64; 3]>],
    box_dimensions: &[[f64; 3]],
) -> Result<Vec<Vec<[f64; 3]>>, String> {
    let num_frames = trajectory.len();
    if box_dimensions.len() != num_frames {
        return Err("Number of box dimensions must match number of frames".to_string());
    }
    if num_frames == 0 {
        return Ok(Vec::new());
    }

    let num_atoms = trajectory[0].len();
    let mut result = Vec::with_capacity(num_frames);

    // First frame: copy directly
    result.push(trajectory[0].clone());

    let mut shifts_x = vec![0.0f64; num_atoms];
    let mut shifts_y = vec![0.0f64; num_atoms];
    let mut shifts_z = vec![0.0f64; num_atoms];

    for frame_idx in 1..num_frames {
        let box_dims = &box_dimensions[frame_idx];
        let half_box = [box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0];

        let mut frame = Vec::with_capacity(num_atoms);

        for atom_idx in 0..num_atoms {
            let curr = &trajectory[frame_idx][atom_idx];
            let prev = &trajectory[frame_idx - 1][atom_idx];

            let delta_x = curr[0] - prev[0];
            let delta_y = curr[1] - prev[1];
            let delta_z = curr[2] - prev[2];

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

            frame.push([
                curr[0] + shifts_x[atom_idx],
                curr[1] + shifts_y[atom_idx],
                curr[2] + shifts_z[atom_idx],
            ]);
        }

        result.push(frame);
    }

    Ok(result)
}
