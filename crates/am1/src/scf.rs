//! Self-consistent field (SCF) procedure for AM1.
//!
//! Iterative solution: build Fock matrix, diagonalize, fill orbitals,
//! build new density matrix, check convergence. Uses simple damping
//! for initial iterations, then switches to DIIS.

use crate::fock::build_fock_matrix;
use crate::hamiltonian::build_core_hamiltonian;
use crate::molecule::Am1Molecule;
use crate::nuclear::compute_nuclear_repulsion;
use crate::overlap::compute_overlap_matrix;
use crate::two_electron::compute_two_electron_integrals;
use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// SCF configuration.
#[derive(Debug, Clone)]
pub struct ScfConfig {
    /// Maximum number of SCF iterations.
    pub max_iter: usize,
    /// RMS density change convergence threshold.
    pub convergence: f64,
    /// Number of initial iterations to use simple damping.
    pub damping_iters: usize,
    /// Damping factor for initial iterations (0 = no damping, 1 = full damping).
    pub damping_factor: f64,
    /// Maximum number of DIIS vectors to store.
    pub diis_size: usize,
}

impl Default for ScfConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            convergence: 1.0e-8,
            damping_iters: 3,
            damping_factor: 0.5,
            diis_size: 6,
        }
    }
}

/// Result of an AM1 calculation.
#[derive(Debug, Clone)]
pub struct Am1Result {
    /// Total energy in eV.
    pub total_energy: f64,
    /// Electronic energy in eV.
    pub electronic_energy: f64,
    /// Nuclear repulsion energy in eV.
    pub nuclear_repulsion: f64,
    /// Heat of formation in kcal/mol.
    pub heat_of_formation: f64,
    /// Mulliken charges per atom.
    pub charges: Vec<f64>,
    /// Orbital energies in eV.
    pub orbital_energies: Vec<f64>,
    /// Whether SCF converged.
    pub converged: bool,
    /// Number of SCF iterations.
    pub n_iterations: usize,
    /// Final RMS density change.
    pub rms_density_change: f64,
    /// Density matrix.
    pub density: DMatrix<f64>,
}

/// Run the SCF procedure.
pub fn run_scf(mol: &Am1Molecule, config: &ScfConfig) -> Result<Am1Result, String> {
    let n = mol.n_basis;
    let n_occ = mol.n_occupied();

    if n == 0 {
        return Err("No basis functions".into());
    }
    if n_occ == 0 {
        return Err("No occupied orbitals".into());
    }

    // Compute overlap matrix
    let overlap = compute_overlap_matrix(mol);

    // Compute two-electron integrals
    let tei = compute_two_electron_integrals(mol);

    // Build core Hamiltonian
    let h_core = build_core_hamiltonian(mol, &overlap, &tei);

    // Initial guess: diagonalize H_core
    let h_sym = symmetrize(&h_core);
    let eigen = SymmetricEigen::new(h_sym);
    let mut coeffs = sort_eigenvectors(&eigen.eigenvalues, &eigen.eigenvectors);

    // Build initial density matrix
    let mut density = build_density(&coeffs, n_occ, n);
    let mut old_density = density.clone();

    // DIIS storage
    let mut diis_focks: Vec<DMatrix<f64>> = Vec::new();
    let mut diis_errors: Vec<DMatrix<f64>> = Vec::new();

    let mut converged = false;
    let mut n_iter = 0;
    let mut rms_change = f64::MAX;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Build Fock matrix
        let mut fock = build_fock_matrix(mol, &h_core, &density, &tei);

        // Compute error vector for DIIS: e = F*D - D*F (commutator)
        let error = &fock * &density - &density * &fock;

        // Apply damping for initial iterations
        if iter < config.damping_iters {
            let alpha = config.damping_factor;
            for i in 0..n {
                for j in 0..n {
                    density[(i, j)] = (1.0 - alpha) * density[(i, j)] + alpha * old_density[(i, j)];
                }
            }
            // Rebuild Fock with damped density
            fock = build_fock_matrix(mol, &h_core, &density, &tei);
        } else {
            // DIIS extrapolation
            diis_focks.push(fock.clone());
            diis_errors.push(error);

            if diis_focks.len() > config.diis_size {
                diis_focks.remove(0);
                diis_errors.remove(0);
            }

            if diis_focks.len() >= 2 {
                if let Some(extrapolated) = diis_extrapolate(&diis_focks, &diis_errors) {
                    fock = extrapolated;
                }
            }
        }

        // Diagonalize Fock matrix
        let f_sym = symmetrize(&fock);
        let eigen = SymmetricEigen::new(f_sym);
        coeffs = sort_eigenvectors(&eigen.eigenvalues, &eigen.eigenvectors);

        // Build new density matrix
        old_density = density.clone();
        density = build_density(&coeffs, n_occ, n);

        // Check convergence
        let mut rms = 0.0;
        for i in 0..n {
            for j in 0..n {
                let diff = density[(i, j)] - old_density[(i, j)];
                rms += diff * diff;
            }
        }
        rms_change = (rms / (n * n) as f64).sqrt();

        if rms_change < config.convergence {
            converged = true;
            break;
        }
    }

    // Compute final energies
    let fock_final = build_fock_matrix(mol, &h_core, &density, &tei);
    let electronic_energy = compute_electronic_energy(&density, &h_core, &fock_final);
    let nuclear_repulsion = compute_nuclear_repulsion(mol);
    let total_energy = electronic_energy + nuclear_repulsion;

    // Heat of formation
    let heat_of_formation = compute_heat_of_formation(mol, total_energy);

    // Mulliken charges (trivial in NDDO since S=I on-site)
    let charges = crate::charges::compute_mulliken_charges(mol, &density);

    // Orbital energies from final diagonalization
    let f_sym = symmetrize(&fock_final);
    let eigen = SymmetricEigen::new(f_sym);
    let (sorted_evals, _) = sort_eigen(&eigen.eigenvalues, &eigen.eigenvectors);

    Ok(Am1Result {
        total_energy,
        electronic_energy,
        nuclear_repulsion,
        heat_of_formation,
        charges,
        orbital_energies: sorted_evals,
        converged,
        n_iterations: n_iter,
        rms_density_change: rms_change,
        density,
    })
}

/// Build density matrix from MO coefficients.
fn build_density(coeffs: &DMatrix<f64>, n_occ: usize, n: usize) -> DMatrix<f64> {
    let mut p = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n_occ {
                sum += coeffs[(i, k)] * coeffs[(j, k)];
            }
            p[(i, j)] = 2.0 * sum; // Factor of 2 for closed-shell
        }
    }
    p
}

/// Compute electronic energy: E_elec = 0.5 * Tr[P * (H + F)]
fn compute_electronic_energy(p: &DMatrix<f64>, h: &DMatrix<f64>, f: &DMatrix<f64>) -> f64 {
    let n = p.nrows();
    let mut energy = 0.0;
    for i in 0..n {
        for j in 0..n {
            energy += p[(i, j)] * (h[(i, j)] + f[(i, j)]);
        }
    }
    0.5 * energy
}

/// Compute heat of formation from total energy.
fn compute_heat_of_formation(mol: &Am1Molecule, total_energy: f64) -> f64 {
    // Heat of formation = total_energy - sum(atomic_electronic_energies) + sum(atomic_heats_of_formation)
    let mut atomic_energy = 0.0;
    let mut atomic_hof = 0.0;

    for atom in &mol.atoms {
        atomic_energy += compute_isolated_atom_energy(atom.params);
        atomic_hof += atom.params.heat_of_form;
    }

    // Convert total energy from eV to kcal/mol for comparison
    let total_kcal = total_energy * crate::params::EV_TO_KCAL;
    let atomic_kcal = atomic_energy * crate::params::EV_TO_KCAL;

    (total_kcal - atomic_kcal) + atomic_hof
}

/// Compute the electronic energy of an isolated atom.
///
/// This matches the SQM/MOPAC EISOL formula exactly:
///
///   EISOL = USS*IOS + UPP*IOP + GSS*GSSC + GSP*GSPC + GPP*GPPC + GP2*GP2C + HSP*HSPC
///
/// where IOS = number of s electrons, IOP = number of p electrons, and the
/// coefficients are:
///   GSSC = max(IOS - 1, 0)
///   GSPC = IOS * IOP
///   GP2C = IOP*(IOP-1)/2 + 0.5 * min(IOP, 6-IOP) * (min(IOP, 6-IOP) - 1) / 2
///   GPPC = -0.5 * min(IOP, 6-IOP) * (min(IOP, 6-IOP) - 1) / 2
///   HSPC = -IOP
///
/// Reference: qm2_load_params_and_allocate.F90 in SQM.
fn compute_isolated_atom_energy(p: &crate::params::Am1Element) -> f64 {
    // Determine electron configuration: IOS = s electrons, IOP = p electrons
    let (ios, iop) = match p.core_charge {
        1 => (1_i32, 0_i32), // H: 1 s-electron
        _ => {
            // For sp atoms with core_charge >= 2: IOS=2, IOP = core_charge - 2
            (2_i32, p.core_charge as i32 - 2)
        }
    };

    // One-electron part
    let e_one = (ios as f64) * p.uss + (iop as f64) * p.upp;

    // Two-electron coefficients (exact SQM formula)
    let gssc = (ios - 1).max(0) as f64;
    let gspc = (ios * iop) as f64;
    let min_iop = iop.min(6 - iop);
    let gppc = -0.5 * (min_iop * (min_iop - 1) / 2) as f64;
    let gp2c = (iop * (iop - 1) / 2) as f64
        + 0.5 * (min_iop * (min_iop - 1) / 2) as f64;
    let hspc = -(iop as f64);

    let e_two = p.gss * gssc + p.gsp * gspc + p.gpp * gppc + p.gp2 * gp2c + p.hsp * hspc;

    e_one + e_two
}

/// Symmetrize a matrix: M = 0.5 * (M + M^T).
fn symmetrize(m: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (m + m.transpose())
}

/// Sort eigenvectors by eigenvalue (ascending).
fn sort_eigenvectors(
    eigenvalues: &DVector<f64>,
    eigenvectors: &DMatrix<f64>,
) -> DMatrix<f64> {
    let (_, sorted_vecs) = sort_eigen(eigenvalues, eigenvectors);
    sorted_vecs
}

/// Sort eigenvalues and eigenvectors together.
fn sort_eigen(
    eigenvalues: &DVector<f64>,
    eigenvectors: &DMatrix<f64>,
) -> (Vec<f64>, DMatrix<f64>) {
    let n = eigenvalues.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    let mut sorted_vals = Vec::with_capacity(n);
    let mut sorted_vecs = DMatrix::zeros(eigenvectors.nrows(), n);

    for (new_col, &old_col) in indices.iter().enumerate() {
        sorted_vals.push(eigenvalues[old_col]);
        for row in 0..eigenvectors.nrows() {
            sorted_vecs[(row, new_col)] = eigenvectors[(row, old_col)];
        }
    }

    (sorted_vals, sorted_vecs)
}

/// DIIS extrapolation.
///
/// Minimizes the error by finding optimal linear combination of previous Fock matrices.
fn diis_extrapolate(
    fock_list: &[DMatrix<f64>],
    error_list: &[DMatrix<f64>],
) -> Option<DMatrix<f64>> {
    let m = fock_list.len();
    if m < 2 {
        return None;
    }

    // Build B matrix: B(i,j) = Tr(e_i * e_j)
    let dim = m + 1;
    let mut b_mat = DMatrix::zeros(dim, dim);

    for i in 0..m {
        for j in 0..m {
            let mut trace = 0.0;
            let ei = &error_list[i];
            let ej = &error_list[j];
            for r in 0..ei.nrows() {
                for c in 0..ei.ncols() {
                    trace += ei[(r, c)] * ej[(r, c)];
                }
            }
            b_mat[(i, j)] = trace;
        }
    }

    // Lagrange multiplier row/column
    for i in 0..m {
        b_mat[(m, i)] = -1.0;
        b_mat[(i, m)] = -1.0;
    }
    b_mat[(m, m)] = 0.0;

    // RHS: [0, 0, ..., 0, -1]
    let mut rhs = DVector::zeros(dim);
    rhs[m] = -1.0;

    // Solve B * c = rhs
    // Use simple LU-like approach: we have a small system
    let b_inv = match b_mat.clone().try_inverse() {
        Some(inv) => inv,
        None => return None,
    };

    let coeffs = &b_inv * &rhs;

    // Build extrapolated Fock matrix
    let n = fock_list[0].nrows();
    let mut f_new = DMatrix::zeros(n, n);
    for i in 0..m {
        f_new += coeffs[i] * &fock_list[i];
    }

    Some(f_new)
}
