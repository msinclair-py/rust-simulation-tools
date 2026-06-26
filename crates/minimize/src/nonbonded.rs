//! Direct-space non-bonded force evaluation (LJ + Ewald direct).
//!
//! Computes Lennard-Jones (van der Waals) and direct-space Ewald
//! electrostatic forces for all pairs in a [`NeighborList`], plus
//! scaled 1-4 interactions from the dihedral topology.
//!
//! Energy is returned in kcal/mol and forces in kcal/(mol*A).
//! Positions and box dimensions are in Angstroms.
//!
//! # AMBER conventions
//!
//! * LJ parameters use the A/B coefficient form:
//!   `E_lj = A_ij / r^12 - B_ij / r^6`
//!
//! * Charges are stored in AMBER internal units (`q_e * 18.2223`).
//!   The Coulomb energy in kcal/mol is simply
//!   `E_elec = q_i_amber * q_j_amber / r`
//!   because `18.2223^2 = 332.0637` (the Coulomb constant in AMBER units).
//!
//! * 1-4 interactions are scaled by `1/SCEE` (electrostatics) and
//!   `1/SCNB` (LJ), with defaults 1.2 and 2.0 respectively.

use crate::neighbor_list::NeighborList;
use rst_core::amber::prmtop::AmberTopology;

// ---------------------------------------------------------------------------
// PairBitmap -- identical to the one in mmpbsa/mm_energy.rs
// ---------------------------------------------------------------------------

/// Flat triangular bitmap for O(1) pair lookups without hashing.
///
/// For a pair (i, j) where i < j, the bit index is `j*(j-1)/2 + i`.
pub struct PairBitmap {
    data: Vec<u64>,
    _n_atoms: usize,
}

impl PairBitmap {
    /// Create a new empty bitmap for `n_atoms` atoms.
    pub fn new(n_atoms: usize) -> Self {
        let n_bits = if n_atoms >= 2 {
            n_atoms * (n_atoms - 1) / 2
        } else {
            0
        };
        let n_words = if n_bits > 0 { n_bits.div_ceil(64) } else { 1 };
        Self {
            data: vec![0u64; n_words],
            _n_atoms: n_atoms,
        }
    }

    /// Set the bit for pair (i, j) where i < j.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize) {
        debug_assert!(i < j);
        let idx = j * (j - 1) / 2 + i;
        self.data[idx / 64] |= 1u64 << (idx % 64);
    }

    /// Test whether pair (i, j) is set, where i < j.
    #[inline]
    pub fn contains(&self, i: usize, j: usize) -> bool {
        debug_assert!(i < j);
        let idx = j * (j - 1) / 2 + i;
        (self.data[idx / 64] >> (idx % 64)) & 1 != 0
    }
}

// ---------------------------------------------------------------------------
// NonbondedParams
// ---------------------------------------------------------------------------

/// Precomputed non-bonded parameters for fast lookup during force evaluation.
///
/// Encapsulates the exclusion list (1-2 and 1-3 pairs), the 1-4 pair bitmap,
/// and the scaling factors read from the topology.
pub struct NonbondedParams {
    /// Exclusion bitmap (1-2 and 1-3 pairs, from topology).
    pub excluded: PairBitmap,
    /// 1-4 pair bitmap (from dihedral topology).
    pub pairs_14: PairBitmap,
    /// SCEE scale factor for 1-4 electrostatic interactions.
    pub scee_14: f64,
    /// SCNB scale factor for 1-4 LJ interactions.
    pub scnb_14: f64,
}

impl NonbondedParams {
    /// Build non-bonded parameters from an AMBER topology.
    ///
    /// Extracts the exclusion list, identifies 1-4 pairs from the dihedral
    /// table, and reads SCEE / SCNB scaling factors.
    pub fn from_topology(topology: &AmberTopology) -> Self {
        let excluded = build_exclusion_bitmap(topology);
        let pairs_14 = build_14_bitmap(topology);
        Self {
            excluded,
            pairs_14,
            scee_14: topology.scee_scale_factor,
            scnb_14: topology.scnb_scale_factor,
        }
    }
}

// ---------------------------------------------------------------------------
// Exclusion / 1-4 bitmap builders
// ---------------------------------------------------------------------------

/// Build exclusion bitmap from AMBER topology exclusion lists.
fn build_exclusion_bitmap(topology: &AmberTopology) -> PairBitmap {
    let mut bitmap = PairBitmap::new(topology.n_atoms.max(2));
    let mut offset = 0usize;
    for i in 0..topology.n_atoms {
        let count = if i < topology.num_excluded_atoms.len() {
            topology.num_excluded_atoms[i]
        } else {
            0
        };
        for k in 0..count {
            if offset + k < topology.excluded_atoms_list.len() {
                let j = topology.excluded_atoms_list[offset + k];
                // AMBER uses usize::MAX (converted from 0) as placeholder
                if j < topology.n_atoms {
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    bitmap.set(lo, hi);
                }
            }
        }
        offset += count;
    }
    bitmap
}

/// Build 1-4 pair bitmap from AMBER dihedral table.
///
/// Only dihedrals without the `ignore_14` flag contribute pairs.
fn build_14_bitmap(topology: &AmberTopology) -> PairBitmap {
    let mut bitmap = PairBitmap::new(topology.n_atoms.max(2));
    for &(i, _j, _k, l, _type_idx, ignore_14) in &topology.dihedrals {
        if !ignore_14 {
            let (lo, hi) = if i < l { (i, l) } else { (l, i) };
            bitmap.set(lo, hi);
        }
    }
    bitmap
}

// ---------------------------------------------------------------------------
// LJ parameter lookup
// ---------------------------------------------------------------------------

/// Look up LJ A and B coefficients for atom pair (i, j).
#[inline]
fn lj_ab(topology: &AmberTopology, i: usize, j: usize) -> (f64, f64) {
    let ti = topology.atom_type_indices[i];
    let tj = topology.atom_type_indices[j];
    let idx = topology.nb_parm_index[topology.n_types * ti + tj];
    if idx < 1 {
        return (0.0, 0.0);
    }
    let idx = (idx - 1) as usize;
    (topology.lj_acoef[idx], topology.lj_bcoef[idx])
}

// ---------------------------------------------------------------------------
// Minimum-image helper
// ---------------------------------------------------------------------------

/// Compute the minimum-image displacement vector and squared distance.
///
/// Returns `(dx, dy, dz, r2)` where `dx, dy, dz` point from atom `j`
/// to atom `i` (i.e. `pos_i - pos_j`) after applying minimum image.
#[inline]
fn min_image_dr(
    pos_i: &[f64; 3],
    pos_j: &[f64; 3],
    box_dims: &[f64; 3],
) -> (f64, f64, f64, f64) {
    let mut dx = pos_i[0] - pos_j[0];
    let mut dy = pos_i[1] - pos_j[1];
    let mut dz = pos_i[2] - pos_j[2];
    dx -= (dx / box_dims[0]).round() * box_dims[0];
    dy -= (dy / box_dims[1]).round() * box_dims[1];
    dz -= (dz / box_dims[2]).round() * box_dims[2];
    let r2 = dx * dx + dy * dy + dz * dz;
    (dx, dy, dz, r2)
}

// ---------------------------------------------------------------------------
// erfc approximation
// ---------------------------------------------------------------------------

/// Fast rational approximation of the complementary error function.
///
/// Abramowitz & Stegun formula 7.1.26, max error ~ 1.5e-7.
/// This is sufficient for molecular simulation energy/force accuracy.
#[inline]
fn erfc_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let poly = 0.254_829_592 * t
        - 0.284_496_736 * t2
        + 1.421_413_741 * t3
        - 1.453_152_027 * t4
        + 1.061_405_429 * t5;
    poly * (-x * x).exp()
}

// ---------------------------------------------------------------------------
// Main force routine
// ---------------------------------------------------------------------------

/// Compute direct-space non-bonded forces (LJ + direct-space Ewald electrostatics).
///
/// Also evaluates 1-4 interactions with appropriate scaling.
///
/// Forces are **added** to the `forces` array (not overwritten) so that
/// bonded contributions can be accumulated in the same array beforehand.
///
/// # Returns
///
/// A tuple `(vdw_energy, elec_direct_energy, vdw_14_energy, elec_14_energy)`
/// with all components in kcal/mol.
///
/// # Arguments
///
/// * `topology`      - AMBER topology with LJ params and charges.
/// * `positions`     - Atom positions in Angstroms.
/// * `forces`        - Mutable force array; contributions are added in-place.
/// * `neighbor_list` - Pre-built cell-list neighbor list.
/// * `nb_params`     - Precomputed exclusion / 1-4 bitmaps and scale factors.
/// * `box_dims`      - Orthorhombic box dimensions in Angstroms.
/// * `ewald_alpha`   - Ewald splitting parameter (1/Angstrom).
pub fn compute_nonbonded_forces(
    topology: &AmberTopology,
    positions: &[[f64; 3]],
    forces: &mut [[f64; 3]],
    neighbor_list: &NeighborList,
    nb_params: &NonbondedParams,
    box_dims: &[f64; 3],
    ewald_alpha: f64,
) -> (f64, f64, f64, f64) {
    let cutoff_sq = neighbor_list.cutoff * neighbor_list.cutoff;
    let two_alpha_over_sqrt_pi =
        2.0 * ewald_alpha * std::f64::consts::FRAC_2_SQRT_PI * 0.5;
    // Note: FRAC_2_SQRT_PI = 2/sqrt(pi), so 2*alpha/sqrt(pi) = alpha * FRAC_2_SQRT_PI
    let alpha_sq = ewald_alpha * ewald_alpha;

    let mut vdw_energy = 0.0f64;
    let mut elec_energy = 0.0f64;
    let mut vdw_14_energy = 0.0f64;
    let mut elec_14_energy = 0.0f64;

    // ------------------------------------------------------------------
    // Process neighbor-list pairs (non-bonded + 1-4 that happen to be
    // in range, though 1-4 pairs may also be beyond cutoff -- we handle
    // them separately below).
    // ------------------------------------------------------------------
    for &(i, j) in &neighbor_list.pairs {
        let (dx, dy, dz, r2) = min_image_dr(&positions[i], &positions[j], box_dims);

        // Skip excluded pairs (1-2 and 1-3 bonds).
        if nb_params.excluded.contains(i, j) {
            // But check if this is also a 1-4 pair.  In AMBER, 1-4 pairs
            // appear in the exclusion list as well, so we need to handle
            // them here.
            if nb_params.pairs_14.contains(i, j) && r2 < cutoff_sq && r2 > 1.0e-10 {
                let r = r2.sqrt();
                let inv_r = 1.0 / r;
                let r6 = r2 * r2 * r2;
                let r12 = r6 * r6;
                let inv_r6 = 1.0 / r6;
                let inv_r12 = 1.0 / r12;

                // Scaled LJ 1-4
                let (a, b) = lj_ab(topology, i, j);
                let e_lj = (a * inv_r12 - b * inv_r6) / nb_params.scnb_14;
                vdw_14_energy += e_lj;

                // Force magnitude from LJ: F/r = (12*A/r^14 - 6*B/r^8) / scnb
                let f_lj = (12.0 * a * inv_r12 - 6.0 * b * inv_r6)
                    * inv_r * inv_r / nb_params.scnb_14;

                // Scaled Coulomb 1-4 (full Coulomb, no Ewald splitting for 1-4).
                let qi = topology.charges_amber[i];
                let qj = topology.charges_amber[j];
                let e_elec = qi * qj * inv_r / nb_params.scee_14;
                elec_14_energy += e_elec;

                // Force magnitude from Coulomb: F/r = q_i*q_j / (r^3 * scee)
                let f_elec = qi * qj * inv_r * inv_r * inv_r / nb_params.scee_14;

                let f_total = f_lj + f_elec;
                forces[i][0] += f_total * dx;
                forces[i][1] += f_total * dy;
                forces[i][2] += f_total * dz;
                forces[j][0] -= f_total * dx;
                forces[j][1] -= f_total * dy;
                forces[j][2] -= f_total * dz;
            }
            continue;
        }

        // Regular non-bonded pair.
        if r2 >= cutoff_sq || r2 < 1.0e-10 {
            continue;
        }

        let r = r2.sqrt();
        let inv_r = 1.0 / r;
        let r6 = r2 * r2 * r2;
        let r12 = r6 * r6;
        let inv_r6 = 1.0 / r6;
        let inv_r12 = 1.0 / r12;

        // ----- LJ -----
        let (a, b) = lj_ab(topology, i, j);
        let e_lj = a * inv_r12 - b * inv_r6;
        vdw_energy += e_lj;

        // LJ force magnitude (divided by r for convenience):
        // F_lj = (12*A/r^13 - 6*B/r^7) directed along r_ij
        // F_lj / r = 12*A/r^14 - 6*B/r^8
        let f_lj_over_r =
            (12.0 * a * inv_r12 - 6.0 * b * inv_r6) * inv_r * inv_r;

        // ----- Direct-space Ewald electrostatics -----
        let qi = topology.charges_amber[i];
        let qj = topology.charges_amber[j];
        let alpha_r = ewald_alpha * r;
        let erfc_val = erfc_approx(alpha_r);
        let exp_val = (-alpha_sq * r2).exp();

        // E_direct = q_i * q_j * erfc(alpha*r) / r
        let e_elec = qi * qj * erfc_val * inv_r;
        elec_energy += e_elec;

        // F_direct / r = q_i * q_j * [erfc(alpha*r)/r^2 + 2*alpha/sqrt(pi) * exp(-alpha^2*r^2)/r] / r
        //             = q_i * q_j * [erfc(alpha*r)/r^3 + 2*alpha/sqrt(pi) * exp(-alpha^2*r^2)/r^2]
        let f_elec_over_r = qi * qj
            * (erfc_val * inv_r * inv_r * inv_r
                + two_alpha_over_sqrt_pi * exp_val * inv_r * inv_r);

        let f_total_over_r = f_lj_over_r + f_elec_over_r;

        // Apply Newton's third law.
        forces[i][0] += f_total_over_r * dx;
        forces[i][1] += f_total_over_r * dy;
        forces[i][2] += f_total_over_r * dz;
        forces[j][0] -= f_total_over_r * dx;
        forces[j][1] -= f_total_over_r * dy;
        forces[j][2] -= f_total_over_r * dz;
    }

    // ------------------------------------------------------------------
    // Handle 1-4 pairs that were NOT in the neighbor list (because the
    // atoms were beyond cutoff + skin).  This is rare but necessary for
    // correctness -- 1-4 interactions are evaluated regardless of cutoff.
    // ------------------------------------------------------------------
    for &(i, _j, _k, l, _type_idx, ignore_14) in &topology.dihedrals {
        if ignore_14 {
            continue;
        }
        let (lo, hi) = if i < l { (i, l) } else { (l, i) };

        // If this pair was already handled in the neighbor-list loop
        // above (i.e. it appeared in the exclusion bitmap AND was in
        // the neighbor list AND within cutoff), skip it.
        // We detect "already handled" by checking whether the pair was
        // in the neighbor list; if not, we evaluate it here.
        // A simple approach: check if the pair's minimum-image distance
        // is beyond the list radius.  If it is within, the neighbor list
        // should contain it and we handled it above.
        let (dx, dy, dz, r2) = min_image_dr(&positions[lo], &positions[hi], box_dims);
        let r_list = neighbor_list.cutoff + neighbor_list.skin;
        if r2 <= r_list * r_list {
            // Was in the neighbor list -- already processed.
            continue;
        }
        if r2 < 1.0e-10 {
            continue;
        }

        let r = r2.sqrt();
        let inv_r = 1.0 / r;
        let r6 = r2 * r2 * r2;
        let r12 = r6 * r6;
        let inv_r6 = 1.0 / r6;
        let inv_r12 = 1.0 / r12;

        let (a, b) = lj_ab(topology, lo, hi);
        let e_lj = (a * inv_r12 - b * inv_r6) / nb_params.scnb_14;
        vdw_14_energy += e_lj;

        let f_lj = (12.0 * a * inv_r12 - 6.0 * b * inv_r6)
            * inv_r * inv_r / nb_params.scnb_14;

        let qi = topology.charges_amber[lo];
        let qj = topology.charges_amber[hi];
        let e_elec = qi * qj * inv_r / nb_params.scee_14;
        elec_14_energy += e_elec;

        let f_elec = qi * qj * inv_r * inv_r * inv_r / nb_params.scee_14;

        let f_total = f_lj + f_elec;
        forces[lo][0] += f_total * dx;
        forces[lo][1] += f_total * dy;
        forces[lo][2] += f_total * dz;
        forces[hi][0] -= f_total * dx;
        forces[hi][1] -= f_total * dy;
        forces[hi][2] -= f_total * dz;
    }

    (vdw_energy, elec_energy, vdw_14_energy, elec_14_energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Helper to create a minimal topology for testing.
    fn minimal_topology(n_atoms: usize) -> AmberTopology {
        AmberTopology {
            n_atoms,
            n_residues: 1,
            n_types: 1,
            atom_names: vec!["C".to_string(); n_atoms],
            atom_type_indices: vec![0; n_atoms],
            charges: vec![0.0; n_atoms],
            charges_amber: vec![0.0; n_atoms],
            residue_labels: vec!["RES".to_string()],
            residue_pointers: vec![0],
            lj_sigma: Arc::new(vec![0.0]),
            lj_epsilon: Arc::new(vec![0.0]),
            atom_sigmas: vec![0.0; n_atoms],
            atom_epsilons: vec![0.0; n_atoms],
            bonds: vec![],
            bond_types: vec![],
            masses: vec![12.0; n_atoms],
            radii: vec![1.7; n_atoms],
            screen: vec![0.72; n_atoms],
            bond_force_constants: Arc::new(vec![]),
            bond_equil_values: Arc::new(vec![]),
            angle_force_constants: Arc::new(vec![]),
            angle_equil_values: Arc::new(vec![]),
            dihedral_force_constants: Arc::new(vec![]),
            dihedral_periodicities: Arc::new(vec![]),
            dihedral_phases: Arc::new(vec![]),
            angles: vec![],
            dihedrals: vec![],
            num_excluded_atoms: vec![0; n_atoms],
            excluded_atoms_list: vec![],
            scee_scale_factor: 1.2,
            scnb_scale_factor: 2.0,
            lj_acoef: Arc::new(vec![0.0]),
            lj_bcoef: Arc::new(vec![0.0]),
            nb_parm_index: Arc::new(vec![1]),
        }
    }

    #[test]
    fn pair_bitmap_basic() {
        let mut bm = PairBitmap::new(10);
        assert!(!bm.contains(2, 5));
        bm.set(2, 5);
        assert!(bm.contains(2, 5));
        assert!(!bm.contains(1, 5));
    }

    #[test]
    fn nonbonded_params_from_empty_topology() {
        let top = minimal_topology(4);
        let params = NonbondedParams::from_topology(&top);
        // No exclusions and no 1-4 pairs in an empty topology.
        assert!(!params.excluded.contains(0, 1));
        assert!(!params.pairs_14.contains(0, 1));
        assert!((params.scee_14 - 1.2).abs() < f64::EPSILON);
        assert!((params.scnb_14 - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn lj_force_repulsive_at_short_range() {
        // Two neutral atoms with LJ parameters close together.
        // At short range, the repulsive A/r^12 term dominates, pushing apart.
        let mut top = minimal_topology(2);
        // A = 1.0e6, B = 1.0e3 (arbitrary but gives a nice potential)
        top.lj_acoef = Arc::new(vec![1.0e6]);
        top.lj_bcoef = Arc::new(vec![1.0e3]);
        top.nb_parm_index = Arc::new(vec![1]); // index 1 -> acoef/bcoef[0]
        top.charges_amber = vec![0.0, 0.0]; // no electrostatics

        let positions = [[10.0, 15.0, 15.0], [12.0, 15.0, 15.0]]; // 2 A apart
        let box_dims = [30.0, 30.0, 30.0];

        let mut nlist = NeighborList::new(14.0, 0.0);
        nlist.build(&positions, &box_dims);

        let params = NonbondedParams::from_topology(&top);
        let mut forces = [[0.0f64; 3]; 2];

        let (vdw, elec, vdw_14, elec_14) = compute_nonbonded_forces(
            &top, &positions, &mut forces, &nlist, &params, &box_dims, 0.0,
        );

        // VDW energy should be large and positive (repulsive regime).
        assert!(vdw > 0.0, "vdw energy = {} should be > 0", vdw);
        // No charges, so electrostatic should be zero.
        assert!(
            elec.abs() < 1e-15,
            "elec energy = {} should be ~0",
            elec
        );
        assert!(vdw_14.abs() < 1e-15);
        assert!(elec_14.abs() < 1e-15);

        // Force on atom 0 should be in -x direction (pushed left by atom 1
        // which is to the right), and force on atom 1 in +x.
        // Wait -- the displacement vector is pos[0] - pos[1] = (-2, 0, 0).
        // For repulsive force: atom 0 is pushed in the -x direction? No:
        // The displacement from j to i is pos_i - pos_j. If LJ is repulsive,
        // force on i is along +displacement (away from j), so atom 0 gets
        // pushed in -x direction and atom 1 in +x direction.
        // Actually: dx = pos[0] - pos[1] = -2. Force = f_over_r * dx.
        // For repulsion at short range, f_over_r > 0 (12A/r^14 dominates).
        // So force on i: f_over_r * (-2) < 0 => atom 0 pushed in -x. Good.
        assert!(
            forces[0][0] < 0.0,
            "atom 0 should be pushed in -x, got {}",
            forces[0][0]
        );
        assert!(
            forces[1][0] > 0.0,
            "atom 1 should be pushed in +x, got {}",
            forces[1][0]
        );

        // Newton's 3rd law.
        assert!(
            (forces[0][0] + forces[1][0]).abs() < 1e-10,
            "Newton's 3rd law violated: {} + {} != 0",
            forces[0][0],
            forces[1][0]
        );
    }

    #[test]
    fn coulomb_force_between_like_charges() {
        // Two atoms with positive AMBER charges, no LJ.
        let mut top = minimal_topology(2);
        top.lj_acoef = Arc::new(vec![0.0]);
        top.lj_bcoef = Arc::new(vec![0.0]);
        top.nb_parm_index = Arc::new(vec![1]);
        // charge = +1e => amber charge = 18.2223
        top.charges_amber = vec![18.2223, 18.2223];
        top.charges = vec![1.0, 1.0];

        let positions = [[10.0, 15.0, 15.0], [15.0, 15.0, 15.0]]; // 5 A apart
        let box_dims = [30.0, 30.0, 30.0];

        let mut nlist = NeighborList::new(14.0, 0.0);
        nlist.build(&positions, &box_dims);

        let params = NonbondedParams::from_topology(&top);
        let mut forces = [[0.0f64; 3]; 2];

        // alpha = 0 => erfc(0) = 1 => full Coulomb (no splitting).
        let (vdw, elec, _, _) = compute_nonbonded_forces(
            &top, &positions, &mut forces, &nlist, &params, &box_dims, 0.0,
        );

        // E = q_i_amber * q_j_amber / r = 18.2223^2 / 5 = 332.0637 / 5 = 66.41274
        let expected_elec = 18.2223 * 18.2223 / 5.0;
        assert!(
            (elec - expected_elec).abs() < 0.01,
            "elec = {}, expected {}",
            elec,
            expected_elec
        );
        assert!(vdw.abs() < 1e-15);

        // Like charges repel: atom 0 pushed in -x direction.
        assert!(
            forces[0][0] < 0.0,
            "atom 0 force_x = {} should be < 0 (repulsive)",
            forces[0][0]
        );
    }

    #[test]
    fn excluded_pair_skipped() {
        let mut top = minimal_topology(2);
        top.lj_acoef = Arc::new(vec![1.0e6]);
        top.lj_bcoef = Arc::new(vec![1.0e3]);
        top.nb_parm_index = Arc::new(vec![1]);
        top.charges_amber = vec![18.2223, 18.2223];
        // Mark pair (0,1) as excluded (1-2 bond).
        top.num_excluded_atoms = vec![1, 0];
        top.excluded_atoms_list = vec![1]; // 0-based, atom 0 excludes atom 1

        let positions = [[10.0, 15.0, 15.0], [12.0, 15.0, 15.0]];
        let box_dims = [30.0, 30.0, 30.0];

        let mut nlist = NeighborList::new(14.0, 0.0);
        nlist.build(&positions, &box_dims);

        let params = NonbondedParams::from_topology(&top);
        let mut forces = [[0.0f64; 3]; 2];

        let (vdw, elec, vdw_14, elec_14) = compute_nonbonded_forces(
            &top, &positions, &mut forces, &nlist, &params, &box_dims, 0.0,
        );

        // Excluded pair: all energies and forces should be zero.
        assert!(vdw.abs() < 1e-15, "vdw should be 0 for excluded pair");
        assert!(elec.abs() < 1e-15, "elec should be 0 for excluded pair");
        assert!(vdw_14.abs() < 1e-15);
        assert!(elec_14.abs() < 1e-15);
        assert!(forces[0][0].abs() < 1e-15);
        assert!(forces[1][0].abs() < 1e-15);
    }

    #[test]
    fn one_four_pair_scaled() {
        // Set up a 4-atom chain with a dihedral that defines a 1-4 pair.
        let mut top = minimal_topology(4);
        top.lj_acoef = Arc::new(vec![1.0e5]);
        top.lj_bcoef = Arc::new(vec![1.0e2]);
        top.nb_parm_index = Arc::new(vec![1]);
        top.charges_amber = vec![18.2223, 0.0, 0.0, 18.2223];
        top.charges = vec![1.0, 0.0, 0.0, 1.0];

        // Bonds: 0-1, 1-2, 2-3
        top.bonds = vec![(0, 1), (1, 2), (2, 3)];
        top.bond_types = vec![0, 0, 0];

        // Dihedral: 0-1-2-3, not ignore_14
        top.dihedrals = vec![(0, 1, 2, 3, 0, false)];
        top.dihedral_force_constants = Arc::new(vec![1.0]);
        top.dihedral_periodicities = Arc::new(vec![2.0]);
        top.dihedral_phases = Arc::new(vec![std::f64::consts::PI]);

        // Exclusion list: pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) all excluded.
        // In AMBER, 1-4 pairs are in the exclusion list; the dihedral table
        // is used to identify them for scaled interactions.
        top.num_excluded_atoms = vec![3, 2, 1, 0];
        top.excluded_atoms_list = vec![1, 2, 3, 2, 3, 3];

        let positions = [
            [10.0, 15.0, 15.0], // atom 0
            [11.5, 15.0, 15.0], // atom 1
            [13.0, 15.0, 15.0], // atom 2
            [14.5, 15.0, 15.0], // atom 3
        ];
        let box_dims = [30.0, 30.0, 30.0];

        let mut nlist = NeighborList::new(14.0, 0.0);
        nlist.build(&positions, &box_dims);

        let params = NonbondedParams::from_topology(&top);
        let mut forces = [[0.0f64; 3]; 4];

        let (vdw, elec, vdw_14, elec_14) = compute_nonbonded_forces(
            &top, &positions, &mut forces, &nlist, &params, &box_dims, 0.0,
        );

        // Regular VDW and elec should be zero (all pairs excluded).
        assert!(
            vdw.abs() < 1e-15,
            "vdw should be 0 (all excluded), got {}",
            vdw
        );
        assert!(
            elec.abs() < 1e-15,
            "elec should be 0 (all excluded), got {}",
            elec
        );

        // 1-4 VDW and elec should be non-zero (scaled).
        let r: f64 = 4.5; // distance between atoms 0 and 3
        let r6 = r.powi(6);
        let r12 = r.powi(12);
        let expected_vdw_14 = (1.0e5 / r12 - 1.0e2 / r6) / 2.0;
        let expected_elec_14 = 18.2223 * 18.2223 / r / 1.2;

        assert!(
            (vdw_14 - expected_vdw_14).abs() < 1e-6,
            "vdw_14 = {}, expected {}",
            vdw_14,
            expected_vdw_14
        );
        assert!(
            (elec_14 - expected_elec_14).abs() < 0.01,
            "elec_14 = {}, expected {}",
            elec_14,
            expected_elec_14
        );
    }

    #[test]
    fn ewald_splitting_reduces_energy() {
        // With alpha > 0, erfc(alpha*r) < 1, so direct-space energy should
        // be smaller than the full Coulomb energy.
        let mut top = minimal_topology(2);
        top.lj_acoef = Arc::new(vec![0.0]);
        top.lj_bcoef = Arc::new(vec![0.0]);
        top.nb_parm_index = Arc::new(vec![1]);
        top.charges_amber = vec![18.2223, 18.2223];

        let positions = [[10.0, 15.0, 15.0], [15.0, 15.0, 15.0]];
        let box_dims = [30.0, 30.0, 30.0];

        let mut nlist = NeighborList::new(14.0, 0.0);
        nlist.build(&positions, &box_dims);
        let params = NonbondedParams::from_topology(&top);

        // Full Coulomb (alpha=0).
        let mut forces_full = [[0.0f64; 3]; 2];
        let (_, elec_full, _, _) = compute_nonbonded_forces(
            &top, &positions, &mut forces_full, &nlist, &params, &box_dims, 0.0,
        );

        // Ewald split (alpha = 0.3).
        let mut forces_ewald = [[0.0f64; 3]; 2];
        let (_, elec_ewald, _, _) = compute_nonbonded_forces(
            &top, &positions, &mut forces_ewald, &nlist, &params, &box_dims, 0.3,
        );

        assert!(
            elec_ewald < elec_full,
            "Ewald direct ({}) should be < full Coulomb ({})",
            elec_ewald,
            elec_full
        );
        assert!(elec_ewald > 0.0, "Ewald direct should still be positive");
    }

    #[test]
    fn newton_third_law_multi_atom() {
        // Total force on the system should be zero.
        let mut top = minimal_topology(3);
        top.lj_acoef = Arc::new(vec![1.0e5]);
        top.lj_bcoef = Arc::new(vec![1.0e2]);
        top.nb_parm_index = Arc::new(vec![1]);
        top.charges_amber = vec![18.2223, -9.111, 4.556];

        let positions = [
            [10.0, 15.0, 15.0],
            [13.0, 17.0, 14.0],
            [11.0, 12.0, 18.0],
        ];
        let box_dims = [30.0, 30.0, 30.0];

        let mut nlist = NeighborList::new(14.0, 0.0);
        nlist.build(&positions, &box_dims);
        let params = NonbondedParams::from_topology(&top);
        let mut forces = [[0.0f64; 3]; 3];

        compute_nonbonded_forces(
            &top, &positions, &mut forces, &nlist, &params, &box_dims, 0.2,
        );

        let total_x: f64 = forces.iter().map(|f| f[0]).sum();
        let total_y: f64 = forces.iter().map(|f| f[1]).sum();
        let total_z: f64 = forces.iter().map(|f| f[2]).sum();

        assert!(
            total_x.abs() < 1e-8,
            "total force x = {} (should be ~0)",
            total_x
        );
        assert!(
            total_y.abs() < 1e-8,
            "total force y = {} (should be ~0)",
            total_y
        );
        assert!(
            total_z.abs() < 1e-8,
            "total force z = {} (should be ~0)",
            total_z
        );
    }

    #[test]
    fn erfc_approx_accuracy() {
        // Compare our erfc approximation against known values.
        let test_cases: [(f64, f64); 5] = [
            (0.0, 1.0),
            (0.5, 0.4795),
            (1.0, 0.1573),
            (2.0, 0.00468),
            (3.0, 0.0000221),
        ];
        for (x, expected) in test_cases {
            let result = erfc_approx(x);
            let rel_err = if expected.abs() > 1e-10 {
                ((result - expected) / expected).abs()
            } else {
                (result - expected).abs()
            };
            assert!(
                rel_err < 0.01,
                "erfc({}) = {}, expected ~{}, rel_err = {}",
                x,
                result,
                expected,
                rel_err
            );
        }
    }

    #[test]
    fn zero_alpha_gives_full_coulomb() {
        // With alpha = 0, erfc(0) = 1, so we should get full Coulomb.
        let mut top = minimal_topology(2);
        top.lj_acoef = Arc::new(vec![0.0]);
        top.lj_bcoef = Arc::new(vec![0.0]);
        top.nb_parm_index = Arc::new(vec![1]);
        top.charges_amber = vec![18.2223, -18.2223]; // +1e and -1e

        let positions = [[10.0, 15.0, 15.0], [14.0, 15.0, 15.0]]; // 4 A apart
        let box_dims = [30.0, 30.0, 30.0];

        let mut nlist = NeighborList::new(14.0, 0.0);
        nlist.build(&positions, &box_dims);
        let params = NonbondedParams::from_topology(&top);
        let mut forces = [[0.0f64; 3]; 2];

        let (_, elec, _, _) = compute_nonbonded_forces(
            &top, &positions, &mut forces, &nlist, &params, &box_dims, 0.0,
        );

        // E = qi*qj/r = 18.2223 * (-18.2223) / 4 = -332.0637 / 4 = -83.016
        let expected = 18.2223 * (-18.2223) / 4.0;
        assert!(
            (elec - expected).abs() < 0.01,
            "elec = {}, expected {}",
            elec,
            expected
        );
    }
}
