//! Kabsch alignment algorithm for trajectory processing.

use nalgebra;
use rayon::prelude::*;

/// Compute centroid of a set of 3D points.
#[inline]
pub fn compute_centroid(points: &[[f64; 3]]) -> [f64; 3] {
    if points.is_empty() {
        return [0.0; 3];
    }
    let n = points.len() as f64;
    let mut sum = [0.0; 3];
    for p in points {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Compute Kabsch rotation matrix.
/// Takes mobile points, mobile centroid, and pre-centered reference points.
#[inline]
pub fn compute_kabsch_rotation(
    mobile: &[[f64; 3]],
    mobile_centroid: &[f64; 3],
    ref_centered: &[[f64; 3]],
) -> Result<[[f64; 3]; 3], String> {
    let mut h = [[0.0f64; 3]; 3];
    let n = mobile.len();

    for k in 0..n {
        let m_centered = [
            mobile[k][0] - mobile_centroid[0],
            mobile[k][1] - mobile_centroid[1],
            mobile[k][2] - mobile_centroid[2],
        ];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += m_centered[i] * ref_centered[k][j];
            }
        }
    }

    let mat = nalgebra::Matrix3::new(
        h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
    );

    let svd = nalgebra::SVD::new(mat, true, true);
    let u = svd
        .u
        .ok_or_else(|| "SVD decomposition failed: no U matrix".to_string())?;
    let v_t = svd
        .v_t
        .ok_or_else(|| "SVD decomposition failed: no V^T matrix".to_string())?;
    let v = v_t.transpose();
    let mut r = v * u.transpose();

    if r.determinant() < 0.0 {
        let mut v_corrected = v;
        for i in 0..3 {
            v_corrected[(i, 2)] *= -1.0;
        }
        r = v_corrected * u.transpose();
    }

    Ok([
        [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
        [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
        [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
    ])
}

/// Align trajectory frames to a reference structure using Kabsch algorithm.
///
/// # Arguments
/// * `trajectory` - Frames of shape [n_frames][n_atoms] with [f64; 3] positions
/// * `reference` - Reference coordinates [n_atoms]
/// * `align_indices` - Atom indices to use for alignment
///
/// # Returns
/// Aligned trajectory with same shape
pub fn kabsch_align(
    trajectory: &[Vec<[f64; 3]>],
    reference: &[[f64; 3]],
    align_indices: &[usize],
) -> Vec<Vec<[f64; 3]>> {
    let ref_align: Vec<[f64; 3]> = align_indices.iter().map(|&idx| reference[idx]).collect();
    let ref_centroid = compute_centroid(&ref_align);
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

    trajectory
        .par_iter()
        .map(|frame| {
            let num_atoms = frame.len();
            let mobile_align: Vec<[f64; 3]> = align_indices.iter().map(|&idx| frame[idx]).collect();
            let mobile_centroid = compute_centroid(&mobile_align);
            let rotation = compute_kabsch_rotation(&mobile_align, &mobile_centroid, &ref_centered)
                .expect("Kabsch SVD decomposition failed");

            let mut aligned = Vec::with_capacity(num_atoms);
            for atom_idx in 0..num_atoms {
                let centered = [
                    frame[atom_idx][0] - mobile_centroid[0],
                    frame[atom_idx][1] - mobile_centroid[1],
                    frame[atom_idx][2] - mobile_centroid[2],
                ];
                aligned.push([
                    rotation[0][0] * centered[0]
                        + rotation[0][1] * centered[1]
                        + rotation[0][2] * centered[2]
                        + ref_centroid[0],
                    rotation[1][0] * centered[0]
                        + rotation[1][1] * centered[1]
                        + rotation[1][2] * centered[2]
                        + ref_centroid[1],
                    rotation[2][0] * centered[0]
                        + rotation[2][1] * centered[1]
                        + rotation[2][2] * centered[2]
                        + ref_centroid[2],
                ]);
            }
            aligned
        })
        .collect()
}
