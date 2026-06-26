//! Mulliken charge analysis for AM1.
//!
//! In NDDO methods, the overlap matrix is the identity on-site,
//! so Mulliken charges simplify to: q_A = Z_A - sum(P_mu_mu for mu on A).

use crate::molecule::Am1Molecule;
use nalgebra::DMatrix;

/// Compute Mulliken charges from the density matrix.
///
/// q_A = Z'_A - sum_{mu on A} P(mu, mu)
///
/// where Z'_A is the core charge and P is the density matrix.
pub fn compute_mulliken_charges(mol: &Am1Molecule, density: &DMatrix<f64>) -> Vec<f64> {
    let mut charges = Vec::with_capacity(mol.atoms.len());

    for atom in &mol.atoms {
        let z = atom.params.core_charge as f64;
        let mut electron_pop = 0.0;

        for mu in 0..atom.params.n_orbitals as usize {
            let idx = atom.basis_offset + mu;
            electron_pop += density[(idx, idx)];
        }

        charges.push(z - electron_pop);
    }

    charges
}
