//! Interaction energy fingerprinting with cell list acceleration.
//!
//! Computes per-residue Lennard-Jones and electrostatic interaction energies
//! between target and binder selections using spatial hashing for O(n) complexity.

use crate::util::distance_squared;
use numpy::ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::cell::RefCell;

// ============================================================================
// Physical Constants
// ============================================================================

/// Coulomb constant in J*m/C^2
const COULOMB_CONSTANT: f64 = 8.988e9;
/// Elementary charge in Coulombs
const ELEMENTARY_CHARGE: f64 = 1.602e-19;
/// Avogadro's number
const AVOGADRO: f64 = 6.022e23;
/// Solvent dielectric constant (water at 298K)
const SOLVENT_DIELECTRIC: f64 = 78.5;
/// Electrostatic cutoff in nm
const ES_CUTOFF_NM: f64 = 1.0;
/// Lennard-Jones cutoff in nm
const LJ_CUTOFF_NM: f64 = 1.2;
/// Cell size for spatial hashing (slightly larger than max cutoff)
const CELL_SIZE: f64 = 1.2;

// Pre-computed squared cutoffs for early rejection
const ES_CUTOFF_SQ: f64 = ES_CUTOFF_NM * ES_CUTOFF_NM;
const LJ_CUTOFF_SQ: f64 = LJ_CUTOFF_NM * LJ_CUTOFF_NM;

// ============================================================================
// Reaction Field Parameters (computed once, cached)
// ============================================================================

/// Pre-computed reaction field constants for electrostatic calculations.
struct ReactionFieldParams {
    k_rf: f64,
    c_rf: f64,
    prefactor: f64,
}

impl ReactionFieldParams {
    fn new() -> Self {
        let eps = SOLVENT_DIELECTRIC;
        let r_c = ES_CUTOFF_NM * 1e-9; // Convert to meters

        let k_rf = (eps - 1.0) / ((2.0 * eps + 1.0) * r_c.powi(3));
        let c_rf = (3.0 * eps) / ((2.0 * eps + 1.0) * r_c);

        // Pre-compute the outer constant term (without charges)
        let prefactor = COULOMB_CONSTANT * ELEMENTARY_CHARGE.powi(2) * AVOGADRO / 1000.0; // J to kJ

        Self {
            k_rf,
            c_rf,
            prefactor,
        }
    }
}

// Global cached reaction field params (computed once)
thread_local! {
    static RF_PARAMS: ReactionFieldParams = ReactionFieldParams::new();
    static NEIGHBOR_BUFFER: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(500));
}

// ============================================================================
// Cell List for Spatial Hashing (using FxHashMap for faster hashing)
// ============================================================================

/// Cell list for efficient neighbor finding via spatial hashing.
/// Uses FxHashMap which is ~2x faster than std HashMap for integer keys.
struct CellList {
    cells: FxHashMap<(i32, i32, i32), Vec<usize>>,
}

impl CellList {
    /// Build a cell list from atom positions.
    fn new(positions: &[[f64; 3]]) -> Self {
        let mut cells: FxHashMap<(i32, i32, i32), Vec<usize>> = FxHashMap::default();

        for (idx, pos) in positions.iter().enumerate() {
            let cell_idx = Self::position_to_cell(pos);
            cells.entry(cell_idx).or_default().push(idx);
        }

        Self { cells }
    }

    #[inline(always)]
    fn position_to_cell(pos: &[f64; 3]) -> (i32, i32, i32) {
        (
            (pos[0] / CELL_SIZE).floor() as i32,
            (pos[1] / CELL_SIZE).floor() as i32,
            (pos[2] / CELL_SIZE).floor() as i32,
        )
    }

    /// Get neighbor indices from the 27-cell cube around a position.
    /// Reuses the provided buffer to avoid allocation.
    #[inline]
    fn get_neighbors_into(&self, pos: &[f64; 3], buffer: &mut Vec<usize>) {
        buffer.clear();
        let (cx, cy, cz) = Self::position_to_cell(pos);

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(atoms) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        buffer.extend_from_slice(atoms);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Pre-extracted Data Structures
// ============================================================================

/// Pre-extracted partner data with cell list for efficient lookup.
/// "Partner" refers to the selection we're computing interactions against
/// (build cell list from these atoms).
pub struct PartnerData {
    positions: Vec<[f64; 3]>,
    charges: Vec<f64>,
    sigmas: Vec<f64>,
    epsilons: Vec<f64>,
    cell_list: CellList,
}

impl PartnerData {
    /// Create partner data from global arrays and atom indices.
    pub fn new(
        all_positions: ArrayView2<f64>,
        all_charges: ArrayView1<f64>,
        all_sigmas: ArrayView1<f64>,
        all_epsilons: ArrayView1<f64>,
        partner_indices: ArrayView1<i64>,
    ) -> Self {
        let positions: Vec<[f64; 3]> = partner_indices
            .iter()
            .map(|&idx| {
                let i = idx as usize;
                [
                    all_positions[[i, 0]],
                    all_positions[[i, 1]],
                    all_positions[[i, 2]],
                ]
            })
            .collect();

        let charges: Vec<f64> = partner_indices
            .iter()
            .map(|&idx| all_charges[idx as usize])
            .collect();

        let sigmas: Vec<f64> = partner_indices
            .iter()
            .map(|&idx| all_sigmas[idx as usize])
            .collect();

        let epsilons: Vec<f64> = partner_indices
            .iter()
            .map(|&idx| all_epsilons[idx as usize])
            .collect();

        let cell_list = CellList::new(&positions);

        Self {
            positions,
            charges,
            sigmas,
            epsilons,
            cell_list,
        }
    }

    /// Create partner data from pre-extracted vectors (for FingerprintSession).
    pub fn from_vecs(
        positions: Vec<[f64; 3]>,
        charges: Vec<f64>,
        sigmas: Vec<f64>,
        epsilons: Vec<f64>,
    ) -> Self {
        let cell_list = CellList::new(&positions);
        Self {
            positions,
            charges,
            sigmas,
            epsilons,
            cell_list,
        }
    }
}

/// Pre-extracted residue data for efficient parallel processing.
/// Extracting once before parallelization avoids repeated work in each thread.
pub struct ResidueData {
    positions: Vec<[f64; 3]>,
    charges: Vec<f64>,
    sigmas: Vec<f64>,
    epsilons: Vec<f64>,
}

impl ResidueData {
    /// Create residue data from global arrays using resmap indices.
    pub fn new(
        positions: ArrayView2<f64>,
        charges: ArrayView1<f64>,
        sigmas: ArrayView1<f64>,
        epsilons: ArrayView1<f64>,
        resmap_indices: ArrayView1<i64>,
        start: usize,
        end: usize,
    ) -> Self {
        let n_atoms = end - start;
        let mut res_positions = Vec::with_capacity(n_atoms);
        let mut res_charges = Vec::with_capacity(n_atoms);
        let mut res_sigmas = Vec::with_capacity(n_atoms);
        let mut res_epsilons = Vec::with_capacity(n_atoms);

        for k in start..end {
            let idx = resmap_indices[k] as usize;
            res_positions.push([
                positions[[idx, 0]],
                positions[[idx, 1]],
                positions[[idx, 2]],
            ]);
            res_charges.push(charges[idx]);
            res_sigmas.push(sigmas[idx]);
            res_epsilons.push(epsilons[idx]);
        }

        Self {
            positions: res_positions,
            charges: res_charges,
            sigmas: res_sigmas,
            epsilons: res_epsilons,
        }
    }

    /// Create residue data from pre-extracted vectors (for FingerprintSession).
    /// atom_indices are global atom indices within the positions array.
    pub fn from_global_indices(
        positions: &[[f64; 3]],
        charges: &[f64],
        sigmas: &[f64],
        epsilons: &[f64],
        atom_indices: &[usize],
    ) -> Self {
        let mut res_positions = Vec::with_capacity(atom_indices.len());
        let mut res_charges = Vec::with_capacity(atom_indices.len());
        let mut res_sigmas = Vec::with_capacity(atom_indices.len());
        let mut res_epsilons = Vec::with_capacity(atom_indices.len());

        for &idx in atom_indices {
            res_positions.push(positions[idx]);
            res_charges.push(charges[idx]);
            res_sigmas.push(sigmas[idx]);
            res_epsilons.push(epsilons[idx]);
        }

        Self {
            positions: res_positions,
            charges: res_charges,
            sigmas: res_sigmas,
            epsilons: res_epsilons,
        }
    }
}

// ============================================================================
// Energy Computation
// ============================================================================

/// Compute LJ and ES energy for a single residue using cell list neighbor lookup.
#[inline]
fn compute_residue_energy(
    residue: &ResidueData,
    partner: &PartnerData,
    k_rf: f64,
    c_rf: f64,
    prefactor: f64,
) -> (f64, f64) {
    let mut lj_energy = 0.0;
    let mut es_energy = 0.0;

    // Use thread-local buffer to avoid allocation
    NEIGHBOR_BUFFER.with(|buf| {
        let mut neighbor_buffer = buf.borrow_mut();

        for i in 0..residue.positions.len() {
            let pos_i = &residue.positions[i];
            let qi = residue.charges[i];
            let sig_i = residue.sigmas[i];
            let eps_i = residue.epsilons[i];

            // Get only partner atoms in neighboring cells
            partner
                .cell_list
                .get_neighbors_into(pos_i, &mut neighbor_buffer);

            for &j in neighbor_buffer.iter() {
                let pos_j = &partner.positions[j];
                let dist_sq = distance_squared(pos_i, pos_j);

                if dist_sq < 1e-20 {
                    continue;
                }

                // Lennard-Jones (cutoff 1.2 nm)
                if dist_sq <= LJ_CUTOFF_SQ {
                    let sig_j = partner.sigmas[j];
                    let eps_j = partner.epsilons[j];

                    let sigma_ij = 0.5 * (sig_i + sig_j);
                    let epsilon_ij = (eps_i * eps_j).sqrt();

                    let sigma_sq = sigma_ij * sigma_ij;
                    let sigma_r_sq = sigma_sq / dist_sq;
                    let sigma_r_6 = sigma_r_sq * sigma_r_sq * sigma_r_sq;
                    let sigma_r_12 = sigma_r_6 * sigma_r_6;

                    lj_energy += 4.0 * epsilon_ij * (sigma_r_12 - sigma_r_6);
                }

                // Electrostatic (cutoff 1.0 nm)
                if dist_sq <= ES_CUTOFF_SQ {
                    let qj = partner.charges[j];
                    let distance = dist_sq.sqrt();
                    let r = distance * 1e-9; // nm to m

                    es_energy += prefactor * qi * qj * (1.0 / r + k_rf * r * r - c_rf);
                }
            }
        }
    });

    (lj_energy, es_energy)
}

/// Generic fingerprint computation - can fingerprint either selection.
///
/// This is the core computation function that computes per-residue interaction
/// energies between a "primary" selection (what we're fingerprinting) and a
/// "partner" selection (what we're interacting with).
///
/// # Arguments
/// * `positions` - All atom coordinates (n_atoms, 3) in nm
/// * `charges` - All atom partial charges in elementary charge units
/// * `sigmas` - All atom LJ sigma parameters in nm
/// * `epsilons` - All atom LJ epsilon parameters in kJ/mol
/// * `primary_resmap_indices` - Atom indices for each primary residue (flattened)
/// * `primary_resmap_offsets` - Start offset for each primary residue (length n_residues + 1)
/// * `partner_indices` - Atom indices for the partner selection
///
/// # Returns
/// (lj_fingerprint, es_fingerprint) - Per-residue LJ and ES energies in kJ/mol
pub fn compute_fingerprints_generic(
    positions: ArrayView2<f64>,
    charges: ArrayView1<f64>,
    sigmas: ArrayView1<f64>,
    epsilons: ArrayView1<f64>,
    primary_resmap_indices: ArrayView1<i64>,
    primary_resmap_offsets: ArrayView1<i64>,
    partner_indices: ArrayView1<i64>,
) -> (Array1<f64>, Array1<f64>) {
    let n_residues = primary_resmap_offsets.len() - 1;

    // Get cached reaction field parameters
    let (k_rf, c_rf, prefactor) =
        RF_PARAMS.with(|params| (params.k_rf, params.c_rf, params.prefactor));

    // Build partner data with cell list (done once)
    let partner = PartnerData::new(positions, charges, sigmas, epsilons, partner_indices);

    // Pre-extract all residue data before parallel processing
    let residue_data: Vec<ResidueData> = (0..n_residues)
        .map(|res_idx| {
            let start = primary_resmap_offsets[res_idx] as usize;
            let end = primary_resmap_offsets[res_idx + 1] as usize;
            ResidueData::new(
                positions,
                charges,
                sigmas,
                epsilons,
                primary_resmap_indices,
                start,
                end,
            )
        })
        .collect();

    // Parallel computation over residues using pre-extracted data
    let results: Vec<(f64, f64)> = residue_data
        .par_iter()
        .map(|residue| compute_residue_energy(residue, &partner, k_rf, c_rf, prefactor))
        .collect();

    let mut lj_fingerprint = Array1::<f64>::zeros(n_residues);
    let mut es_fingerprint = Array1::<f64>::zeros(n_residues);

    for (i, (lj, es)) in results.into_iter().enumerate() {
        lj_fingerprint[i] = lj;
        es_fingerprint[i] = es;
    }

    (lj_fingerprint, es_fingerprint)
}

/// Compute fingerprints from pre-extracted residue and partner data.
///
/// This variant is used by FingerprintSession for efficient frame-by-frame
/// computation where we don't need to re-extract FF parameters each frame.
pub fn compute_fingerprints_from_residues(
    residue_data: &[ResidueData],
    partner: &PartnerData,
) -> (Array1<f64>, Array1<f64>) {
    let n_residues = residue_data.len();

    // Get cached reaction field parameters
    let (k_rf, c_rf, prefactor) =
        RF_PARAMS.with(|params| (params.k_rf, params.c_rf, params.prefactor));

    // Parallel computation over residues
    let results: Vec<(f64, f64)> = residue_data
        .par_iter()
        .map(|residue| compute_residue_energy(residue, partner, k_rf, c_rf, prefactor))
        .collect();

    let mut lj_fingerprint = Array1::<f64>::zeros(n_residues);
    let mut es_fingerprint = Array1::<f64>::zeros(n_residues);

    for (i, (lj, es)) in results.into_iter().enumerate() {
        lj_fingerprint[i] = lj;
        es_fingerprint[i] = es;
    }

    (lj_fingerprint, es_fingerprint)
}

/// Core fingerprint computation (internal) - backward compatible wrapper.
fn compute_fingerprints_internal(
    positions: ArrayView2<f64>,
    charges: ArrayView1<f64>,
    sigmas: ArrayView1<f64>,
    epsilons: ArrayView1<f64>,
    resmap_indices: ArrayView1<i64>,
    resmap_offsets: ArrayView1<i64>,
    binder_indices: ArrayView1<i64>,
) -> (Array1<f64>, Array1<f64>) {
    // Delegate to the generic function with target as primary, binder as partner
    compute_fingerprints_generic(
        positions,
        charges,
        sigmas,
        epsilons,
        resmap_indices,
        resmap_offsets,
        binder_indices,
    )
}

// ============================================================================
// Python Interface
// ============================================================================

/// Compute per-residue interaction energy fingerprints.
///
/// Uses cell list (spatial hashing) for efficient neighbor finding,
/// providing ~3800x speedup over naive implementations.
///
/// Parameters
/// ----------
/// positions : ndarray of shape (n_atoms, 3)
///     Atom coordinates in nm.
/// charges : ndarray of shape (n_atoms,)
///     Partial charges in elementary charge units.
/// sigmas : ndarray of shape (n_atoms,)
///     Lennard-Jones sigma parameters in nm.
/// epsilons : ndarray of shape (n_atoms,)
///     Lennard-Jones epsilon parameters in kJ/mol.
/// resmap_indices : ndarray of int64
///     Flattened atom indices for all residues.
/// resmap_offsets : ndarray of int64
///     Start offset for each residue (length n_residues + 1).
/// binder_indices : ndarray of int64
///     Atom indices for the binder.
///
/// Returns
/// -------
/// tuple of (lj_fingerprint, es_fingerprint)
///     Per-residue LJ and electrostatic energies in kJ/mol.
///
/// Notes
/// -----
/// - Electrostatic: Reaction field method, 10 Å cutoff, ε=78.5
/// - Lennard-Jones: 12 Å cutoff, Lorentz-Berthelot combining rules
/// - Parallelized across residues using rayon
#[pyfunction]
#[pyo3(name = "compute_fingerprints")]
pub fn compute_fingerprints_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<f64>,
    charges: PyReadonlyArray1<f64>,
    sigmas: PyReadonlyArray1<f64>,
    epsilons: PyReadonlyArray1<f64>,
    resmap_indices: PyReadonlyArray1<i64>,
    resmap_offsets: PyReadonlyArray1<i64>,
    binder_indices: PyReadonlyArray1<i64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let positions = positions.as_array();
    let charges = charges.as_array();
    let sigmas = sigmas.as_array();
    let epsilons = epsilons.as_array();
    let resmap_indices = resmap_indices.as_array();
    let resmap_offsets = resmap_offsets.as_array();
    let binder_indices = binder_indices.as_array();

    let (lj_fp, es_fp) = compute_fingerprints_internal(
        positions,
        charges,
        sigmas,
        epsilons,
        resmap_indices,
        resmap_offsets,
        binder_indices,
    );

    Ok((
        PyArray1::from_owned_array_bound(py, lj_fp),
        PyArray1::from_owned_array_bound(py, es_fp),
    ))
}
