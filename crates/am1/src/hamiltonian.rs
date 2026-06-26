//! Core Hamiltonian construction for AM1.
//!
//! The core Hamiltonian H has:
//! - Diagonal: U_ss/U_pp + core-electron attraction from other atoms
//! - Off-diagonal same-atom: core-electron attraction terms
//! - Off-diagonal two-center: resonance integrals (0.5 * (beta_mu + beta_nu) * S_mu_nu)

use crate::molecule::Am1Molecule;
use crate::two_electron::TwoElectronBlock;
use nalgebra::DMatrix;

/// Build the core Hamiltonian matrix.
///
/// H_core(mu,mu) = U_mu + sum_B V_AB(mu,mu)
/// H_core(mu,nu) same atom = V_AB(mu,nu) terms
/// H_core(mu,nu) different atoms = 0.5*(beta_mu + beta_nu)*S(mu,nu)
///
/// The core-electron attraction term for each orbital pair (mu_A, nu_A)
/// from atom B's nuclear charge is:
///
///   V_AB(mu, nu) = -Z_B * (mu_A nu_A | s_B s_B)
///
/// This uses the properly rotated two-electron integrals, NOT just gamma_ss.
/// For p-orbitals, the (pp|ss) integral differs from (ss|ss) due to the
/// directional character of the p-orbital charge distribution.
pub fn build_core_hamiltonian(
    mol: &Am1Molecule,
    overlap: &DMatrix<f64>,
    tei_blocks: &[Vec<Option<TwoElectronBlock>>],
) -> DMatrix<f64> {
    let n = mol.n_basis;
    let mut h = DMatrix::zeros(n, n);

    // Diagonal one-center terms: orbital energies U_ss, U_pp
    for (_i, atom_a) in mol.atoms.iter().enumerate() {
        let pa = atom_a.params;
        let offset = atom_a.basis_offset;

        // s orbital diagonal: USS
        h[(offset, offset)] = pa.uss;

        // p orbital diagonals: UPP
        if pa.n_orbitals >= 4 {
            for k in 1..4 {
                h[(offset + k, offset + k)] = pa.upp;
            }
        }
    }

    // Core-electron attraction from other atoms.
    // V_AB(mu, nu) = -Z_B * (mu_A nu_A | s_B s_B)
    //
    // This uses the rotated two-electron integrals to correctly account
    // for the fact that different orbital pairs (ss|ss), (pp_sigma|ss),
    // (pp_pi|ss) have different values due to the spatial distribution
    // of the charge.
    for (i, atom_a) in mol.atoms.iter().enumerate() {
        let pa = atom_a.params;
        let offset = atom_a.basis_offset;
        let na = pa.n_orbitals as usize;

        for (j, atom_b) in mol.atoms.iter().enumerate() {
            if i == j {
                continue;
            }
            let pb = atom_b.params;
            let zb = pb.core_charge as f64;

            if let Some(block) = &tei_blocks[i][j] {
                let dc = mol.direction_cosines(i, j);

                // All same-atom pairs (mu, nu) on atom A, interacting with B's core
                for mu in 0..na {
                    for nu in mu..na {
                        // V_AB(mu, nu) = -Z_B * (mu_A nu_A | s_B s_B)
                        let integral = crate::two_electron::get_tei_global(
                            block, pa, pb, &dc, mu, nu, 0, 0,
                        );
                        h[(offset + mu, offset + nu)] -= zb * integral;
                        if mu != nu {
                            h[(offset + nu, offset + mu)] -= zb * integral;
                        }
                    }
                }
            }
        }
    }

    // Off-diagonal two-center elements: resonance integrals
    for i in 0..mol.atoms.len() {
        for j in (i + 1)..mol.atoms.len() {
            let ai = &mol.atoms[i];
            let aj = &mol.atoms[j];

            let ni = ai.params.n_orbitals as usize;
            let nj = aj.params.n_orbitals as usize;

            for mu in 0..ni {
                for nu in 0..nj {
                    let gi = ai.basis_offset + mu;
                    let gj = aj.basis_offset + nu;

                    // beta value for orbital mu on atom i
                    let beta_mu = if mu == 0 {
                        ai.params.beta_s
                    } else {
                        ai.params.beta_p
                    };

                    // beta value for orbital nu on atom j
                    let beta_nu = if nu == 0 {
                        aj.params.beta_s
                    } else {
                        aj.params.beta_p
                    };

                    // Resonance integral: H(mu,nu) = 0.5*(beta_mu + beta_nu)*S(mu,nu)
                    let s_val = overlap[(gi, gj)];
                    let h_val = 0.5 * (beta_mu + beta_nu) * s_val;

                    h[(gi, gj)] = h_val;
                    h[(gj, gi)] = h_val;
                }
            }
        }
    }

    h
}
