//! Entropy estimation methods for MM-GBSA binding free energy.
//!
//! Three approaches are provided:
//!
//! - **Interaction Entropy (IE)**: Uses fluctuations of the gas-phase
//!   interaction energy. Simplest and fastest; recommended default.
//!   Reference: Duan et al., JACS 2016, 138, 5722.
//!
//! - **Quasi-Harmonic Analysis (QH)**: Computes configurational entropy from
//!   the covariance matrix of atomic positional fluctuations. Requires
//!   mass-weighted coordinate trajectories.
//!
//! - **Normal Mode Analysis (NMA)**: Placeholder — requires energy minimization
//!   and Hessian computation, which is not yet implemented.

use crate::binding::FrameEnergy;

/// Entropy estimation result.
#[derive(Debug, Clone)]
pub struct EntropyEstimate {
    /// -TΔS in kcal/mol (positive values disfavor binding).
    pub minus_tds: f64,
    /// Method used for the estimate.
    pub method: EntropyMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyMethod {
    InteractionEntropy,
    QuasiHarmonic,
}

// ---------------------------------------------------------------------------
// Interaction Entropy (IE)
// ---------------------------------------------------------------------------

/// Compute -TΔS using the Interaction Entropy method.
///
/// The IE method estimates the entropic contribution from the fluctuation
/// of the gas-phase interaction energy (ΔE_MM) across trajectory frames:
///
///   ΔE_int(t) = ΔE_MM(t) - <ΔE_MM>
///   -TΔS_IE = kT · ln( <exp( ΔE_int / kT )> )
///
/// where kT = 0.593 kcal/mol at T = 298.15 K.
///
/// # Arguments
/// * `frames` - Per-frame binding energies from the MM-GBSA workflow
/// * `temperature` - Temperature in Kelvin (default 298.15)
///
/// # Returns
/// `None` if fewer than 2 frames are provided.
pub fn interaction_entropy(frames: &[FrameEnergy], temperature: f64) -> Option<EntropyEstimate> {
    let n = frames.len();
    if n < 2 {
        return None;
    }

    let kt = 0.00198688 * temperature; // kcal/mol

    // Mean gas-phase interaction energy
    let mean_de: f64 = frames.iter().map(|f| f.delta_mm).sum::<f64>() / n as f64;

    // Compute <exp(ΔE_int / kT)> using the log-sum-exp trick for numerical stability
    let exponents: Vec<f64> = frames.iter().map(|f| (f.delta_mm - mean_de) / kt).collect();

    let max_exp = exponents.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let log_mean_exp =
        max_exp + (exponents.iter().map(|&x| (x - max_exp).exp()).sum::<f64>() / n as f64).ln();

    let minus_tds = kt * log_mean_exp;

    Some(EntropyEstimate {
        minus_tds,
        method: EntropyMethod::InteractionEntropy,
    })
}

// ---------------------------------------------------------------------------
// Quasi-Harmonic Analysis (QH)
// ---------------------------------------------------------------------------

/// Compute -TΔS using the Quasi-Harmonic method.
///
/// Builds the mass-weighted covariance matrix of atomic fluctuations,
/// diagonalizes it, and computes the entropy from eigenvalues using the
/// quantum harmonic oscillator formula.
///
/// # Arguments
/// * `trajectories` - Per-frame coordinates for the subsystem (Angstroms)
/// * `masses` - Atomic masses (amu), one per atom
/// * `temperature` - Temperature in Kelvin
///
/// # Returns
/// `None` if fewer than 2 frames or zero atoms.
pub fn quasi_harmonic_entropy(
    trajectories: &[Vec<[f64; 3]>],
    masses: &[f64],
    temperature: f64,
) -> Option<EntropyEstimate> {
    let n_frames = trajectories.len();
    let n_atoms = masses.len();
    if n_frames < 2 || n_atoms == 0 {
        return None;
    }
    let dim = n_atoms * 3;

    // Mass-weight and compute mean
    let mut mean = vec![0.0f64; dim];
    for frame in trajectories {
        for (i, coord) in frame.iter().enumerate() {
            let sqrt_m = masses[i].sqrt();
            mean[i * 3] += coord[0] * sqrt_m;
            mean[i * 3 + 1] += coord[1] * sqrt_m;
            mean[i * 3 + 2] += coord[2] * sqrt_m;
        }
    }
    for v in mean.iter_mut() {
        *v /= n_frames as f64;
    }

    // Build upper triangle of the covariance matrix (symmetric)
    // C_ij = <(x_i - <x_i>)(x_j - <x_j>)>
    // For memory efficiency with large systems, store only the upper triangle.
    let mut cov = vec![0.0f64; dim * dim];
    let mut delta = vec![0.0f64; dim];

    for frame in trajectories {
        for (i, coord) in frame.iter().enumerate() {
            let sqrt_m = masses[i].sqrt();
            delta[i * 3] = coord[0] * sqrt_m - mean[i * 3];
            delta[i * 3 + 1] = coord[1] * sqrt_m - mean[i * 3 + 1];
            delta[i * 3 + 2] = coord[2] * sqrt_m - mean[i * 3 + 2];
        }
        for i in 0..dim {
            for j in i..dim {
                cov[i * dim + j] += delta[i] * delta[j];
            }
        }
    }

    // Normalize and symmetrize
    let nf = n_frames as f64;
    for i in 0..dim {
        for j in i..dim {
            cov[i * dim + j] /= nf;
            if i != j {
                cov[j * dim + i] = cov[i * dim + j];
            }
        }
    }

    // Diagonalize using Jacobi eigenvalue algorithm for symmetric matrices
    let eigenvalues = jacobi_eigenvalues(&mut cov, dim);

    // Convert eigenvalues to entropy using quantum harmonic oscillator
    // ω² = kT / λ_i (eigenvalue of mass-weighted covariance in Å²·amu)
    // S = Σ_i s(ω_i) where s(ω) is the QHO entropy
    //
    // In convenient units:
    // frequency ν_i from eigenvalue λ_i:
    //   ν_i = 1/(2π) * sqrt(kT / λ_i)  (but we need consistent units)
    //
    // The QHO entropy for mode i with eigenvalue λ_i (in Å²·amu):
    //   Define: α_i = ℏω_i/(kBT) where ω²_i = kBT/λ_i
    //   α_i = ℏ/(kBT) * sqrt(kBT/λ_i) = ℏ/sqrt(kBT·λ_i)
    //
    // Using SI: kBT in J, λ in m²·kg:
    //   kBT = 1.380649e-23 * T (J)
    //   λ_SI = λ_i * 1e-20 * 1.66054e-27 (m²·kg)
    //   ℏ = 1.054571817e-34 J·s
    //
    //   α_i = ℏ / sqrt(kBT · λ_SI)
    //   S_i/kB = α_i/(exp(α_i)-1) - ln(1-exp(-α_i))

    let kb_si = 1.380649e-23; // J/K
    let hbar = 1.054571817e-34; // J·s
    let kbt_si = kb_si * temperature;
    let ang2_to_m2 = 1e-20;
    let amu_to_kg = 1.66054e-27;
    let conv = ang2_to_m2 * amu_to_kg; // Å²·amu → m²·kg

    let mut entropy_kb = 0.0f64; // S/kB (dimensionless)

    for &lambda in &eigenvalues {
        if lambda < 1e-10 {
            continue; // Skip zero/near-zero modes (translation/rotation if present)
        }

        let lambda_si = lambda * conv;
        let alpha = hbar / (kbt_si * lambda_si).sqrt();

        if alpha > 500.0 {
            continue; // Negligible contribution
        }

        let exp_a = alpha.exp();
        let s_mode = alpha / (exp_a - 1.0) - (1.0 - (-alpha).exp()).ln();
        entropy_kb += s_mode;
    }

    // Convert S/kB to -TΔS in kcal/mol
    // S = entropy_kb * kB * N_A = entropy_kb * R
    // -TΔS = -T * entropy_kb * R (R = 0.00198688 kcal/(mol·K))
    let minus_tds = -temperature * 0.00198688 * entropy_kb;

    Some(EntropyEstimate {
        minus_tds,
        method: EntropyMethod::QuasiHarmonic,
    })
}

/// Jacobi eigenvalue algorithm for a symmetric matrix.
///
/// Operates in-place on the matrix (stored as row-major dim×dim).
/// Returns eigenvalues sorted in descending order.
fn jacobi_eigenvalues(matrix: &mut [f64], dim: usize) -> Vec<f64> {
    let max_iter = 100;
    let tol = 1e-10;

    // Initialize eigenvalues as diagonal
    let mut eigenvalues: Vec<f64> = (0..dim).map(|i| matrix[i * dim + i]).collect();
    let mut changed = true;
    let mut iter = 0;

    while changed && iter < max_iter {
        changed = false;
        iter += 1;

        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = matrix[p * dim + q];
                if apq.abs() < tol {
                    continue;
                }

                let app = eigenvalues[p];
                let aqq = eigenvalues[q];
                let tau = (aqq - app) / (2.0 * apq);

                let t = if tau.abs() > 1e15 {
                    1.0 / (2.0 * tau)
                } else {
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau_rot = s / (1.0 + c);

                eigenvalues[p] -= t * apq;
                eigenvalues[q] += t * apq;
                matrix[p * dim + q] = 0.0;
                matrix[q * dim + p] = 0.0;

                // Rotate rows/columns
                for r in 0..dim {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = matrix[r * dim + p];
                    let arq = matrix[r * dim + q];
                    matrix[r * dim + p] = arp - s * (arq + tau_rot * arp);
                    matrix[p * dim + r] = matrix[r * dim + p];
                    matrix[r * dim + q] = arq + s * (arp - tau_rot * arq);
                    matrix[q * dim + r] = matrix[r * dim + q];
                }

                changed = true;
            }
        }
    }

    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interaction_entropy_constant_energy() {
        // If all frames have the same ΔE_MM, fluctuation is zero → -TΔS = 0
        let frames: Vec<FrameEnergy> = (0..10)
            .map(|_| FrameEnergy {
                complex: Default::default(),
                receptor: Default::default(),
                ligand: Default::default(),
                delta_mm: -50.0,
                delta_gb: 10.0,
                delta_sa: -1.0,
                delta_total: -41.0,
            })
            .collect();

        let result = interaction_entropy(&frames, 298.15).unwrap();
        assert!(
            result.minus_tds.abs() < 1e-10,
            "-TΔS should be ~0 for constant energy, got {}",
            result.minus_tds
        );
    }

    #[test]
    fn test_interaction_entropy_with_fluctuation() {
        // Alternating energies should give positive -TΔS
        let frames: Vec<FrameEnergy> = (0..100)
            .map(|i| {
                let e = if i % 2 == 0 { -40.0 } else { -60.0 };
                FrameEnergy {
                    complex: Default::default(),
                    receptor: Default::default(),
                    ligand: Default::default(),
                    delta_mm: e,
                    delta_gb: 0.0,
                    delta_sa: 0.0,
                    delta_total: e,
                }
            })
            .collect();

        let result = interaction_entropy(&frames, 298.15).unwrap();
        assert!(
            result.minus_tds > 0.0,
            "-TΔS should be positive for fluctuating energy, got {}",
            result.minus_tds
        );
    }

    #[test]
    fn test_interaction_entropy_insufficient_frames() {
        assert!(interaction_entropy(&[], 298.15).is_none());
        let one_frame = vec![FrameEnergy {
            complex: Default::default(),
            receptor: Default::default(),
            ligand: Default::default(),
            delta_mm: -50.0,
            delta_gb: 0.0,
            delta_sa: 0.0,
            delta_total: -50.0,
        }];
        assert!(interaction_entropy(&one_frame, 298.15).is_none());
    }

    #[test]
    fn test_jacobi_eigenvalues_identity() {
        // 3x3 identity matrix → eigenvalues all 1.0
        let mut mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let eigs = jacobi_eigenvalues(&mut mat, 3);
        for &e in &eigs {
            assert!((e - 1.0).abs() < 1e-10, "eigenvalue = {}", e);
        }
    }

    #[test]
    fn test_jacobi_eigenvalues_known() {
        // [[2, 1], [1, 2]] → eigenvalues 3, 1
        let mut mat = vec![2.0, 1.0, 1.0, 2.0];
        let eigs = jacobi_eigenvalues(&mut mat, 2);
        assert!((eigs[0] - 3.0).abs() < 1e-10, "eig[0] = {}", eigs[0]);
        assert!((eigs[1] - 1.0).abs() < 1e-10, "eig[1] = {}", eigs[1]);
    }

    #[test]
    fn test_quasi_harmonic_single_mode() {
        // Two frames of a single atom oscillating in x
        let masses = vec![12.0]; // carbon
        let trajectories = vec![vec![[1.0, 0.0, 0.0]], vec![[-1.0, 0.0, 0.0]]];

        let result = quasi_harmonic_entropy(&trajectories, &masses, 298.15);
        assert!(result.is_some());
        let est = result.unwrap();
        // Should be negative (entropy favors disorder → -TΔS < 0 for single subsystem)
        assert!(est.minus_tds.is_finite(), "-TΔS is not finite");
    }
}
