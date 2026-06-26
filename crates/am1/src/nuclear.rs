//! Core-core repulsion energy for AM1.
//!
//! E_AB = Z'_A * Z'_B * gamma_ss(R) * [1 + f(R)] + Gaussian corrections
//!
//! The Gaussian corrections use the (K, L, M) parameters per atom.

use crate::molecule::Am1Molecule;
use crate::params::Am1Element;
use crate::two_electron::gamma_ss;

/// Compute total core-core repulsion energy in eV.
pub fn compute_nuclear_repulsion(mol: &Am1Molecule) -> f64 {
    let mut energy = 0.0;

    for i in 0..mol.atoms.len() {
        for j in (i + 1)..mol.atoms.len() {
            energy += core_core_pair(
                mol.atoms[i].params,
                mol.atoms[j].params,
                mol.distance(i, j),
            );
        }
    }

    energy
}

/// Core-core repulsion between a pair of atoms.
///
/// E_AB = Z_A * Z_B * gamma_ss(R_AB) * alpha_correction(R_AB)
///      + Z_A * Z_B * gaussian_correction(R_AB)
fn core_core_pair(pa: &Am1Element, pb: &Am1Element, r_ang: f64) -> f64 {
    if r_ang < 1.0e-10 {
        return 0.0;
    }

    let za = pa.core_charge as f64;
    let zb = pb.core_charge as f64;

    let gam = gamma_ss(pa, pb, r_ang);

    // Alpha correction factor: for most atom pairs
    // f = 1 + exp(-alpha_A * R) + exp(-alpha_B * R)
    // Special handling for N-H and O-H pairs
    let alpha_term = compute_alpha_term(pa, pb, r_ang);

    // Base core-core repulsion
    let e_base = za * zb * gam * alpha_term;

    // Gaussian corrections
    let e_gauss = gaussian_correction(pa, pb, za, zb, r_ang);

    e_base + e_gauss
}

/// Compute the alpha correction factor.
///
/// Standard (MNDO): scale = 1 + exp(-alpha_A * R) + exp(-alpha_B * R)
///
/// For N-H and O-H pairs (AM1): the heavy atom's exponential is modified:
///   scale = 1 + exp(-alpha_H * R) + R * exp(-alpha_X * R)
/// where X is the heavy atom (N or O). This originates from the SQM formula:
///   scale = 1 + exp(-alpha_H * R) + exp(-alpha_X * R) + (R - 1) * exp(-alpha_X * R)
/// Reference: qm2_core_core_repulsion.F90 in SQM.
fn compute_alpha_term(pa: &Am1Element, pb: &Am1Element, r_ang: f64) -> f64 {
    // Check for special N-H or O-H pairs
    let is_nh = (pa.atomic_number == 7 && pb.atomic_number == 1)
        || (pa.atomic_number == 1 && pb.atomic_number == 7);
    let is_oh = (pa.atomic_number == 8 && pb.atomic_number == 1)
        || (pa.atomic_number == 1 && pb.atomic_number == 8);

    if is_nh || is_oh {
        // Identify which atom is hydrogen and which is the heavy atom (N/O)
        let (h_params, x_params) = if pa.atomic_number == 1 {
            (pa, pb)
        } else {
            (pb, pa)
        };

        // SQM formula: scale = 1 + exp(-alpha_H * R) + R * exp(-alpha_X * R)
        // The heavy atom contribution is multiplied by R (from the (R-1) correction).
        let exp_h = (-h_params.alpha * r_ang).exp();
        let exp_x = (-x_params.alpha * r_ang).exp();
        1.0 + exp_h + r_ang * exp_x
    } else {
        1.0 + (-pa.alpha * r_ang).exp() + (-pb.alpha * r_ang).exp()
    }
}

/// Gaussian correction to core-core repulsion.
///
/// Sum over Gaussian terms: Z_A * Z_B / R * sum_k K_k * exp(-L_k * (R - M_k)^2)
fn gaussian_correction(pa: &Am1Element, pb: &Am1Element, za: f64, zb: f64, r_ang: f64) -> f64 {
    let mut correction = 0.0;

    // Gaussian terms from atom A
    for &(k, l, m) in pa.gaussians {
        correction += za * zb / r_ang * k * (-l * (r_ang - m).powi(2)).exp();
    }

    // Gaussian terms from atom B
    for &(k, l, m) in pb.gaussians {
        correction += za * zb / r_ang * k * (-l * (r_ang - m).powi(2)).exp();
    }

    correction
}
