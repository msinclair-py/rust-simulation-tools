//! Fock matrix construction for AM1.
//!
//! F = H_core + G(P)
//! where G has one-center and two-center electron-electron repulsion terms.

use crate::molecule::Am1Molecule;
use crate::two_electron::{get_tei_global, TwoElectronBlock};
use nalgebra::DMatrix;

/// Build the Fock matrix from core Hamiltonian and density matrix.
///
/// F(mu,nu) = H(mu,nu) + G(mu,nu)
///
/// One-center terms (mu, nu on same atom A):
///   G_one(mu,mu) = sum_lambda_on_A P(lambda,lambda) * [(mu mu|lambda lambda) - 0.5*(mu lambda|mu lambda)]
///   G_one(mu,nu) = P(mu,nu) * [3*(mu nu|mu nu) - (mu mu|nu nu)] * 0.5  (off-diagonal)
///
/// Two-center terms (mu on A, nu on B):
///   G_two(mu,nu) = sum over density on B * exchange integrals
pub fn build_fock_matrix(
    mol: &Am1Molecule,
    h_core: &DMatrix<f64>,
    density: &DMatrix<f64>,
    tei_blocks: &[Vec<Option<TwoElectronBlock>>],
) -> DMatrix<f64> {
    let mut f = h_core.clone();

    // One-center terms: Coulomb and exchange within each atom
    // Uses the general integral function for all cases (diagonal and off-diagonal)
    // to correctly handle all NDDO selection rules.
    for (_i, atom) in mol.atoms.iter().enumerate() {
        let pa = atom.params;
        let off = atom.basis_offset;
        let norb = pa.n_orbitals as usize;

        let gss = pa.gss;
        let gsp = pa.gsp;
        let gpp = pa.gpp;
        let gp2 = pa.gp2;
        let hsp = pa.hsp;

        for mu in 0..norb {
            for nu in mu..norb {
                let gmu = off + mu;
                let gnu = off + nu;

                let mut g_val = 0.0;

                for lam in 0..norb {
                    for sig in 0..norb {
                        let glam = off + lam;
                        let gsig = off + sig;
                        let p_ls = density[(glam, gsig)];

                        // Coulomb: (mu nu | lam sig)
                        let j = one_center_general(mu, nu, lam, sig, gss, gsp, gpp, gp2, hsp);
                        // Exchange: (mu lam | nu sig)
                        let k = one_center_general(mu, lam, nu, sig, gss, gsp, gpp, gp2, hsp);

                        g_val += p_ls * (j - 0.5 * k);
                    }
                }

                f[(gmu, gnu)] += g_val;
                if mu != nu {
                    f[(gnu, gmu)] += g_val;
                }
            }
        }
    }

    // Two-center terms
    for i in 0..mol.atoms.len() {
        let ai = &mol.atoms[i];
        let pi = ai.params;
        let offi = ai.basis_offset;
        let ni = pi.n_orbitals as usize;

        for j in 0..mol.atoms.len() {
            if i == j {
                continue;
            }
            let aj = &mol.atoms[j];
            let pj = aj.params;
            let offj = aj.basis_offset;
            let nj = pj.n_orbitals as usize;

            let dc = mol.direction_cosines(i, j);

            if let Some(block) = &tei_blocks[i][j] {
                // Two-center Coulomb: contribution to F(mu_A, nu_A) from density on B
                for mu in 0..ni {
                    for nu in mu..ni {
                        let gmu = offi + mu;
                        let gnu = offi + nu;
                        let mut g_val = 0.0;

                        for lam in 0..nj {
                            for sig in 0..nj {
                                let glam = offj + lam;
                                let gsig = offj + sig;
                                let p_ls = density[(glam, gsig)];

                                // (mu_A nu_A | lam_B sig_B) - Coulomb
                                let j_int = get_tei_global(
                                    block, pi, pj, &dc, mu, nu, lam, sig,
                                );
                                g_val += p_ls * j_int;
                            }
                        }

                        f[(gmu, gnu)] += g_val;
                        if mu != nu {
                            f[(gnu, gmu)] += g_val;
                        }
                    }
                }

                // Two-center exchange: contribution to F(mu_A, lam_B)
                for mu in 0..ni {
                    for lam in 0..nj {
                        let gmu = offi + mu;
                        let glam = offj + lam;

                        if gmu >= glam {
                            continue; // Only compute upper triangle
                        }

                        let mut g_val = 0.0;

                        for nu in 0..ni {
                            for sig in 0..nj {
                                let gnu = offi + nu;
                                let gsig = offj + sig;
                                let p_ns = density[(gnu, gsig)];

                                // -0.5 * (mu_A nu_A | lam_B sig_B) with indices swapped
                                // Exchange: -(mu_A lam_B | nu_A sig_B) * P(nu_A, sig_B)
                                // But in NDDO, only (mu_A nu_A | lam_B sig_B) is nonzero
                                // The exchange contribution to F(mu_A, lam_B) is:
                                // -0.5 * sum_{nu,sig} P(nu_A, sig_B) * (mu_A sig_B | nu_A lam_B)
                                // = -0.5 * sum_{nu,sig} P(nu,sig) * (mu nu | sig lam) [rearranged]
                                // In NDDO this becomes:
                                // For exchange between atoms: -P(nu_A, sig_B) * (mu_A nu_A | sig_B lam_B)
                                // Wait, NDDO zero differential overlap means:
                                // (mu_A lam_B | nu_A sig_B) = 0 unless pairs are on same atom
                                // So the exchange integral (mu_A sig_B | nu_A lam_B) requires
                                // mu,sig on same atom AND nu,lam on same atom
                                // => mu on A, sig on B, nu on A, lam on B
                                // => (mu_A sig_B | nu_A lam_B) is a valid two-center integral!
                                // = (mu_A nu_A | sig_B lam_B) by Mulliken notation
                                let k_int = get_tei_global(
                                    block, pi, pj, &dc, mu, nu, lam, sig,
                                );
                                g_val -= 0.5 * p_ns * k_int;
                            }
                        }

                        f[(gmu, glam)] += g_val;
                        f[(glam, gmu)] += g_val;
                    }
                }
            }
        }
    }

    f
}

// ============================================================================
// One-center two-electron integrals from AM1 parameters
// ============================================================================

/// General one-center two-electron integral (mu nu | lam sig).
///
/// In NDDO, many of these are zero. The nonzero ones are:
/// (ss|ss) = GSS
/// (ss|pp) = GSP
/// (pp|pp) = GPP (same p)
/// (pp|p'p') = GP2 (different p)
/// (sp|sp) = HSP
/// (pp'|pp') = 0.5*(GPP-GP2)
fn one_center_general(mu: usize, nu: usize, lam: usize, sig: usize, gss: f64, gsp: f64, gpp: f64, gp2: f64, hsp: f64) -> f64 {
    // Sort pairs for lookup
    let (a, b) = if mu <= nu { (mu, nu) } else { (nu, mu) };
    let (c, d) = if lam <= sig { (lam, sig) } else { (sig, lam) };

    match ((a, b), (c, d)) {
        // (ss|ss) = GSS
        ((0, 0), (0, 0)) => gss,
        // (ss|pp) = GSP
        ((0, 0), (p, p2)) if p >= 1 && p == p2 => gsp,
        ((p, p2), (0, 0)) if p >= 1 && p == p2 => gsp,
        // (pp|pp) = GPP (same p orbital)
        ((p1, p1b), (p2, p2b)) if p1 >= 1 && p1 == p1b && p2 >= 1 && p2 == p2b && p1 == p2 => gpp,
        // (pp|p'p') = GP2 (different p orbitals)
        ((p1, p1b), (p2, p2b)) if p1 >= 1 && p1 == p1b && p2 >= 1 && p2 == p2b && p1 != p2 => gp2,
        // (sp|sp) = HSP
        ((0, p1), (0, p2)) if p1 >= 1 && p2 >= 1 && p1 == p2 => hsp,
        // (pp'|pp') = 0.5*(GPP-GP2) (different p orbitals, exchange type)
        ((p1, p2), (p3, p4)) if p1 >= 1 && p2 >= 1 && p3 >= 1 && p4 >= 1
            && p1 != p2 && p3 != p4 && p1 == p3 && p2 == p4 => 0.5 * (gpp - gp2),
        // All other combinations are zero in NDDO
        _ => 0.0,
    }
}
