//! Bonded force evaluation for AMBER force field energy minimization.
//!
//! Computes forces (negative gradients) for bonds, angles, proper dihedrals,
//! and improper dihedrals using parameters from a parsed AMBER prmtop topology.
//!
//! All energies are in kcal/mol and coordinates are in Angstroms, consistent
//! with AMBER internal units.
//!
//! # Force conventions
//!
//! Forces are the negative gradient of the potential energy:
//! **F** = -dE/d**r**. Forces are *accumulated* into the output array so that
//! multiple force contributions (bonded, nonbonded, restraints) can be summed
//! without intermediate zeroing.

use rst_core::amber::prmtop::AmberTopology;

// ============================================================================
// Vector helpers (inlined for hot-path performance)
// ============================================================================

/// Compute the difference vector a - b.
#[inline]
fn vec_sub(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product of two 3-vectors.
#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors.
#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Squared norm of a 3-vector.
#[inline]
fn norm_sq(v: &[f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// Scale a 3-vector by a scalar.
#[inline]
fn scale(v: &[f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Add vector b into vector a in place.
#[inline]
fn add_to(a: &mut [f64; 3], b: &[f64; 3]) {
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
}

/// Subtract vector b from vector a in place.
#[inline]
fn sub_from(a: &mut [f64; 3], b: &[f64; 3]) {
    a[0] -= b[0];
    a[1] -= b[1];
    a[2] -= b[2];
}

// ============================================================================
// Bond forces
// ============================================================================

/// Compute bond stretch forces and return the total bond energy.
///
/// AMBER potential: E = K * (r - r_eq)^2
///
/// The force on atom i along the bond vector is:
///   F_i = -dE/dr * (r_ij / |r_ij|) = -2K * (r - r_eq) * r_hat
/// where r_ij = pos_i - pos_j, r = |r_ij|, r_hat = r_ij / r.
///
/// Force on atom j is the negative of force on atom i (Newton's third law).
#[inline]
fn compute_bond_forces(
    topology: &AmberTopology,
    positions: &[[f64; 3]],
    forces: &mut [[f64; 3]],
) -> f64 {
    let mut energy = 0.0;

    for (bond_idx, &(ai, aj)) in topology.bonds.iter().enumerate() {
        let type_idx = topology.bond_types[bond_idx];
        let k = topology.bond_force_constants[type_idx];
        let r_eq = topology.bond_equil_values[type_idx];

        let rij = vec_sub(&positions[ai], &positions[aj]);
        let r = norm_sq(&rij).sqrt();

        // Avoid division by zero for overlapping atoms.
        if r < 1e-15 {
            continue;
        }

        let dr = r - r_eq;
        energy += k * dr * dr;

        // dE/dr = 2 * k * (r - r_eq)
        // F_i = -dE/dr * r_hat = -2k * dr / r * rij
        let mag = -2.0 * k * dr / r;
        let f = [rij[0] * mag, rij[1] * mag, rij[2] * mag];

        add_to(&mut forces[ai], &f);
        sub_from(&mut forces[aj], &f);
    }

    energy
}

// ============================================================================
// Angle forces
// ============================================================================

/// Compute angle bending forces and return the total angle energy.
///
/// AMBER potential: E = K * (theta - theta_eq)^2
///
/// For angle i-j-k (j is the central/vertex atom):
///   Let r_ji = pos_i - pos_j, r_jk = pos_k - pos_j
///   cos(theta) = (r_ji . r_jk) / (|r_ji| * |r_jk|)
///   dE/dtheta = 2K * (theta - theta_eq)
///
/// The forces on atoms i, j, k are derived from the chain rule:
///   F_i = -(dE/dtheta) * d(theta)/d(pos_i)
///   F_k = -(dE/dtheta) * d(theta)/d(pos_k)
///   F_j = -(F_i + F_k)   (translational invariance)
///
/// The gradient of theta with respect to pos_i is:
///   d(theta)/d(pos_i) = (cos(theta) * r_ji / |r_ji|^2 - r_jk / (|r_ji|*|r_jk|)) / sin(theta)
/// and similarly for pos_k by symmetry.
#[inline]
fn compute_angle_forces(
    topology: &AmberTopology,
    positions: &[[f64; 3]],
    forces: &mut [[f64; 3]],
) -> f64 {
    let mut energy = 0.0;

    for &(ai, aj, ak, type_idx) in &topology.angles {
        let ka = topology.angle_force_constants[type_idx];
        let theta_eq = topology.angle_equil_values[type_idx];

        // Vectors from central atom j to end atoms i and k
        let rji = vec_sub(&positions[ai], &positions[aj]);
        let rjk = vec_sub(&positions[ak], &positions[aj]);

        let rji_sq = norm_sq(&rji);
        let rjk_sq = norm_sq(&rjk);
        let rji_len = rji_sq.sqrt();
        let rjk_len = rjk_sq.sqrt();

        if rji_len < 1e-15 || rjk_len < 1e-15 {
            continue;
        }

        let cos_theta = (dot(&rji, &rjk) / (rji_len * rjk_len)).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let dtheta = theta - theta_eq;
        energy += ka * dtheta * dtheta;

        // dE/dtheta = 2 * ka * (theta - theta_eq)
        let de_dtheta = 2.0 * ka * dtheta;

        // sin(theta) with guard against linear/degenerate angles
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        if sin_theta < 1e-15 {
            continue;
        }

        // Prefactor: -dE/dtheta / sin(theta)
        // The negative sign converts gradient to force, and the 1/sin(theta)
        // comes from d(acos(x))/dx = -1/sqrt(1-x^2).
        let prefactor = -de_dtheta / sin_theta;

        // d(theta)/d(pos_i) = (cos(theta)*rji/|rji|^2 - rjk/(|rji|*|rjk|)) / sin(theta)
        // F_i = prefactor * (cos(theta)*rji/|rji|^2 - rjk/(|rji|*|rjk|))
        let inv_rji_sq = 1.0 / rji_sq;
        let inv_rjk_sq = 1.0 / rjk_sq;
        let inv_rji_rjk = 1.0 / (rji_len * rjk_len);

        let fi = [
            prefactor * (cos_theta * rji[0] * inv_rji_sq - rjk[0] * inv_rji_rjk),
            prefactor * (cos_theta * rji[1] * inv_rji_sq - rjk[1] * inv_rji_rjk),
            prefactor * (cos_theta * rji[2] * inv_rji_sq - rjk[2] * inv_rji_rjk),
        ];

        // d(theta)/d(pos_k) by symmetry: swap rji <-> rjk
        let fk = [
            prefactor * (cos_theta * rjk[0] * inv_rjk_sq - rji[0] * inv_rji_rjk),
            prefactor * (cos_theta * rjk[1] * inv_rjk_sq - rji[1] * inv_rji_rjk),
            prefactor * (cos_theta * rjk[2] * inv_rjk_sq - rji[2] * inv_rji_rjk),
        ];

        add_to(&mut forces[ai], &fi);
        add_to(&mut forces[ak], &fk);
        // Force on central atom j from translational invariance:
        // F_j = -(F_i + F_k)
        forces[aj][0] -= fi[0] + fk[0];
        forces[aj][1] -= fi[1] + fk[1];
        forces[aj][2] -= fi[2] + fk[2];
    }

    energy
}

// ============================================================================
// Dihedral / improper forces
// ============================================================================

/// Compute dihedral (torsion) and improper dihedral forces and return the
/// total dihedral energy.
///
/// AMBER potential: E = pk * (1 + cos(n * phi - gamma))
///
/// where pk = Vn/2 is the half-barrier height stored in
/// `DIHEDRAL_FORCE_CONSTANT`, n is the periodicity, and gamma is the phase.
///
/// The force computation uses the standard torsion gradient formulation
/// based on Blondel and Karplus (J. Comput. Chem. 17, 1132-1141, 1996).
///
/// For dihedral i-j-k-l:
///   b1 = pos_j - pos_i   (bond vector i->j)
///   b2 = pos_k - pos_j   (bond vector j->k, the central bond)
///   b3 = pos_l - pos_k   (bond vector k->l)
///   m  = b1 x b2         (normal to plane i-j-k)
///   n  = b2 x b3         (normal to plane j-k-l)
///   phi = atan2(b2_hat . (m x n), m . n)
///
/// The dihedral gradient can be decomposed into forces on the four atoms
/// using the fact that the torsion angle depends on positions through the
/// two plane normals m and n.
#[inline]
fn compute_dihedral_forces(
    topology: &AmberTopology,
    positions: &[[f64; 3]],
    forces: &mut [[f64; 3]],
) -> f64 {
    let mut energy = 0.0;

    for &(ai, aj, ak, al, type_idx, _ignore_14) in &topology.dihedrals {
        let pk = topology.dihedral_force_constants[type_idx];
        let pn = topology.dihedral_periodicities[type_idx];
        let gamma = topology.dihedral_phases[type_idx];

        // Bond vectors
        let b1 = vec_sub(&positions[aj], &positions[ai]);
        let b2 = vec_sub(&positions[ak], &positions[aj]);
        let b3 = vec_sub(&positions[al], &positions[ak]);

        // Plane normals
        let m = cross(&b1, &b2); // b1 x b2
        let n = cross(&b2, &b3); // b2 x b3

        let m_sq = norm_sq(&m);
        let n_sq = norm_sq(&n);

        // Guard against degenerate (linear) configurations
        if m_sq < 1e-30 || n_sq < 1e-30 {
            continue;
        }

        let b2_len_sq = norm_sq(&b2);
        let b2_len = b2_len_sq.sqrt();

        if b2_len < 1e-15 {
            continue;
        }

        let m_len = m_sq.sqrt();
        let n_len = n_sq.sqrt();

        // Compute dihedral angle using atan2 for correct quadrant.
        //   cos(phi) = (m . n) / (|m| |n|)
        //   sin(phi) = ((m x n) . b2) / (|m| |n| |b2|)
        let cos_phi = dot(&m, &n) / (m_len * n_len);
        let mn_cross = cross(&m, &n);
        let sin_phi = dot(&mn_cross, &b2) / (m_len * n_len * b2_len);
        let phi = sin_phi.atan2(cos_phi);

        // Energy: E = pk * (1 + cos(n*phi - gamma))
        energy += pk * (1.0 + (pn * phi - gamma).cos());

        // dE/dphi = -pk * n * sin(n*phi - gamma)
        let de_dphi = -pk * pn * (pn * phi - gamma).sin();

        // ----------------------------------------------------------
        // Forces on all four atoms.
        //
        // The standard result (used in NAMD, LAMMPS, CHARMM):
        //
        // F_i =  (dE/dphi) * |b2| / |m|^2 * m
        // F_l = -(dE/dphi) * |b2| / |n|^2 * n
        //
        // These are the forces on the terminal atoms. The forces on
        // the central atoms j and k are obtained by distributing the
        // terminal forces along the bond vectors:
        //
        //   F_j = (r_ij . r_kj / |r_kj|^2 - 1) * F_i
        //       - (r_kl . r_kj / |r_kj|^2)     * F_l
        //
        //   F_k = (r_kl . r_kj / |r_kj|^2 - 1) * F_l
        //       - (r_ij . r_kj / |r_kj|^2)     * F_i
        //
        // where r_ij = r_j - r_i = b1, r_kj = r_j - r_k = -b2,
        //       r_kl = r_l - r_k = b3.
        //
        // Substituting:
        //   r_ij . r_kj = b1 . (-b2) = -b1_dot_b2
        //   r_kl . r_kj = b3 . (-b2) = -b3_dot_b2
        //   |r_kj|^2 = |b2|^2
        //
        // So:
        //   F_j = (-b1_dot_b2 / b2_sq - 1) * F_i - (-b3_dot_b2 / b2_sq) * F_l
        //       = -(b1_dot_b2 / b2_sq + 1) * F_i + (b3_dot_b2 / b2_sq) * F_l
        //
        //   F_k = (-b3_dot_b2 / b2_sq - 1) * F_l - (-b1_dot_b2 / b2_sq) * F_i
        //       = -(b3_dot_b2 / b2_sq + 1) * F_l + (b1_dot_b2 / b2_sq) * F_i
        //
        // Note: F_i + F_j + F_k + F_l = F_i * [1 - (c1+1) + c1] + F_l * [c3 - (c3+1) + 1] = 0.

        let b1_dot_b2 = dot(&b1, &b2);
        let b3_dot_b2 = dot(&b3, &b2);
        let inv_b2_sq = 1.0 / b2_len_sq;
        let c1 = b1_dot_b2 * inv_b2_sq;
        let c3 = b3_dot_b2 * inv_b2_sq;

        let fi = scale(&m, de_dphi * b2_len / m_sq);
        let fl = scale(&n, -de_dphi * b2_len / n_sq);

        let fj = [
            -(c1 + 1.0) * fi[0] + c3 * fl[0],
            -(c1 + 1.0) * fi[1] + c3 * fl[1],
            -(c1 + 1.0) * fi[2] + c3 * fl[2],
        ];

        let fk = [
            -(c3 + 1.0) * fl[0] + c1 * fi[0],
            -(c3 + 1.0) * fl[1] + c1 * fi[1],
            -(c3 + 1.0) * fl[2] + c1 * fi[2],
        ];

        add_to(&mut forces[ai], &fi);
        add_to(&mut forces[aj], &fj);
        add_to(&mut forces[ak], &fk);
        add_to(&mut forces[al], &fl);
    }

    energy
}

// ============================================================================
// Public API
// ============================================================================

/// Compute all bonded forces and return the total bonded energy.
///
/// Evaluates bond stretching, angle bending, and dihedral (proper and improper)
/// torsion contributions. Forces are **added** to the `forces` array -- the
/// caller is responsible for zeroing forces before the first call if needed.
///
/// # Arguments
///
/// * `topology` - Parsed AMBER topology containing bonded parameters.
/// * `positions` - Cartesian coordinates for every atom (Angstroms).
/// * `forces` - Mutable force accumulator for every atom (kcal/mol/A).
///
/// # Returns
///
/// Total bonded potential energy in kcal/mol.
///
/// # Panics
///
/// Panics if `positions` or `forces` have fewer elements than the number of
/// atoms referenced by the topology bond/angle/dihedral lists.
pub fn compute_bonded_forces(
    topology: &AmberTopology,
    positions: &[[f64; 3]],
    forces: &mut [[f64; 3]],
) -> f64 {
    let e_bond = compute_bond_forces(topology, positions, forces);
    let e_angle = compute_angle_forces(topology, positions, forces);
    let e_dihedral = compute_dihedral_forces(topology, positions, forces);

    e_bond + e_angle + e_dihedral
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Build a minimal empty topology suitable for populating in tests.
    fn minimal_topology() -> AmberTopology {
        AmberTopology {
            n_atoms: 0,
            n_residues: 0,
            n_types: 0,
            atom_names: vec![],
            atom_type_indices: vec![],
            charges: vec![],
            charges_amber: vec![],
            residue_labels: vec![],
            residue_pointers: vec![],
            lj_sigma: Arc::new(vec![]),
            lj_epsilon: Arc::new(vec![]),
            atom_sigmas: vec![],
            atom_epsilons: vec![],
            bonds: vec![],
            bond_types: vec![],
            masses: vec![],
            radii: vec![],
            screen: vec![],
            bond_force_constants: Arc::new(vec![]),
            bond_equil_values: Arc::new(vec![]),
            angle_force_constants: Arc::new(vec![]),
            angle_equil_values: Arc::new(vec![]),
            dihedral_force_constants: Arc::new(vec![]),
            dihedral_periodicities: Arc::new(vec![]),
            dihedral_phases: Arc::new(vec![]),
            angles: vec![],
            dihedrals: vec![],
            num_excluded_atoms: vec![],
            excluded_atoms_list: vec![],
            scee_scale_factor: 1.2,
            scnb_scale_factor: 2.0,
            lj_acoef: Arc::new(vec![]),
            lj_bcoef: Arc::new(vec![]),
            nb_parm_index: Arc::new(vec![]),
        }
    }

    /// Numerically differentiate energy with respect to a single coordinate
    /// using central differences. Returns the negative gradient (i.e. force).
    fn numerical_force(
        topology: &AmberTopology,
        positions: &[[f64; 3]],
        atom: usize,
        dim: usize,
        energy_fn: fn(&AmberTopology, &[[f64; 3]], &mut [[f64; 3]]) -> f64,
    ) -> f64 {
        let h = 1e-6;
        let n = positions.len();

        let mut pos_plus = positions.to_vec();
        let mut pos_minus = positions.to_vec();
        pos_plus[atom][dim] += h;
        pos_minus[atom][dim] -= h;

        let mut f_dummy = vec![[0.0; 3]; n];
        let e_plus = energy_fn(topology, &pos_plus, &mut f_dummy);
        f_dummy.fill([0.0; 3]);
        let e_minus = energy_fn(topology, &pos_minus, &mut f_dummy);

        // Force = -dE/dx, central difference: dE/dx ~ (E(x+h) - E(x-h)) / (2h)
        -(e_plus - e_minus) / (2.0 * h)
    }

    // -----------------------------------------------------------------------
    // Bond tests
    // -----------------------------------------------------------------------

    #[test]
    fn bond_energy_stretched() {
        let mut top = minimal_topology();
        top.n_atoms = 2;
        top.bonds = vec![(0, 1)];
        top.bond_types = vec![0];
        top.bond_force_constants = Arc::new(vec![300.0]);
        top.bond_equil_values = Arc::new(vec![1.5]);

        let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let mut forces = [[0.0; 3]; 2];
        let energy = compute_bond_forces(&top, &positions, &mut forces);

        // E = 300 * (2.0 - 1.5)^2 = 300 * 0.25 = 75
        assert!((energy - 75.0).abs() < 1e-10, "bond energy = {energy}");

        // rij = pos[0] - pos[1] = (-2, 0, 0), r = 2, dr = 0.5
        // mag = -2 * 300 * 0.5 / 2 = -150
        // F_0 = rij * mag = (-2, 0, 0) * (-150) = (300, 0, 0)
        // Atom 0 is pulled toward atom 1 (positive x direction).
        assert!((forces[0][0] - 300.0).abs() < 1e-10);
        assert!(forces[0][1].abs() < 1e-10);
        assert!(forces[0][2].abs() < 1e-10);

        // Newton's third law: F_1 = -F_0
        assert!((forces[1][0] - (-300.0)).abs() < 1e-10);
        assert!(forces[1][1].abs() < 1e-10);
        assert!(forces[1][2].abs() < 1e-10);
    }

    #[test]
    fn bond_energy_at_equilibrium() {
        let mut top = minimal_topology();
        top.n_atoms = 2;
        top.bonds = vec![(0, 1)];
        top.bond_types = vec![0];
        top.bond_force_constants = Arc::new(vec![300.0]);
        top.bond_equil_values = Arc::new(vec![1.5]);

        let positions = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
        let mut forces = [[0.0; 3]; 2];
        let energy = compute_bond_forces(&top, &positions, &mut forces);

        assert!(energy.abs() < 1e-14, "bond energy at equilibrium = {energy}");
        assert!(forces[0][0].abs() < 1e-14);
        assert!(forces[1][0].abs() < 1e-14);
    }

    #[test]
    fn bond_force_numerical_gradient() {
        let mut top = minimal_topology();
        top.n_atoms = 2;
        top.bonds = vec![(0, 1)];
        top.bond_types = vec![0];
        top.bond_force_constants = Arc::new(vec![317.0]);
        top.bond_equil_values = Arc::new(vec![1.522]);

        // Off-axis bond
        let positions = [[0.3, 0.7, -0.2], [1.1, -0.4, 0.9]];
        let mut forces = [[0.0; 3]; 2];
        compute_bond_forces(&top, &positions, &mut forces);

        for atom in 0..2 {
            for dim in 0..3 {
                let f_num = numerical_force(&top, &positions, atom, dim, compute_bond_forces);
                let f_ana = forces[atom][dim];
                let err = (f_ana - f_num).abs();
                assert!(
                    err < 1e-5,
                    "bond force mismatch: atom={atom} dim={dim} analytical={f_ana} numerical={f_num} err={err}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Angle tests
    // -----------------------------------------------------------------------

    #[test]
    fn angle_energy_right_angle() {
        let mut top = minimal_topology();
        top.n_atoms = 3;
        top.angles = vec![(0, 1, 2, 0)];
        top.angle_force_constants = Arc::new(vec![50.0]);
        top.angle_equil_values = Arc::new(vec![std::f64::consts::PI]); // eq = 180 degrees

        // 90-degree angle
        let positions = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut forces = [[0.0; 3]; 3];
        let energy = compute_angle_forces(&top, &positions, &mut forces);

        let expected = 50.0 * std::f64::consts::FRAC_PI_2 * std::f64::consts::FRAC_PI_2;
        assert!(
            (energy - expected).abs() < 1e-10,
            "angle energy = {energy}, expected = {expected}"
        );
    }

    #[test]
    fn angle_force_numerical_gradient() {
        let mut top = minimal_topology();
        top.n_atoms = 3;
        top.angles = vec![(0, 1, 2, 0)];
        top.angle_force_constants = Arc::new(vec![80.0]);
        top.angle_equil_values = Arc::new(vec![1.9]); // ~109 degrees

        let positions = [
            [1.0, 0.2, 0.1],
            [0.0, 0.0, 0.0],
            [-0.3, 0.8, 0.5],
        ];
        let mut forces = [[0.0; 3]; 3];
        compute_angle_forces(&top, &positions, &mut forces);

        for atom in 0..3 {
            for dim in 0..3 {
                let f_num = numerical_force(&top, &positions, atom, dim, compute_angle_forces);
                let f_ana = forces[atom][dim];
                let err = (f_ana - f_num).abs();
                assert!(
                    err < 1e-5,
                    "angle force mismatch: atom={atom} dim={dim} analytical={f_ana} numerical={f_num} err={err}"
                );
            }
        }
    }

    #[test]
    fn angle_force_zero_at_equilibrium() {
        let mut top = minimal_topology();
        top.n_atoms = 3;
        top.angles = vec![(0, 1, 2, 0)];
        top.angle_force_constants = Arc::new(vec![80.0]);
        // Equilibrium at 90 degrees
        top.angle_equil_values = Arc::new(vec![std::f64::consts::FRAC_PI_2]);

        // Exact 90-degree angle
        let positions = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut forces = [[0.0; 3]; 3];
        let energy = compute_angle_forces(&top, &positions, &mut forces);

        assert!(energy.abs() < 1e-14, "angle energy at eq = {energy}");
        for atom in 0..3 {
            for dim in 0..3 {
                assert!(
                    forces[atom][dim].abs() < 1e-12,
                    "angle force not zero at equilibrium: atom={atom} dim={dim} f={}",
                    forces[atom][dim]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Dihedral tests
    // -----------------------------------------------------------------------

    #[test]
    fn dihedral_force_numerical_gradient() {
        let mut top = minimal_topology();
        top.n_atoms = 4;
        top.dihedrals = vec![(0, 1, 2, 3, 0, false)];
        top.dihedral_force_constants = Arc::new(vec![2.5]);
        top.dihedral_periodicities = Arc::new(vec![2.0]);
        top.dihedral_phases = Arc::new(vec![std::f64::consts::PI]);

        // Non-trivial geometry with b1.b2 != 0 and b3.b2 != 0
        let positions = [
            [1.0, 0.5, 0.3],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.5],
            [-0.5, 1.0, 1.5],
        ];
        let mut forces = [[0.0; 3]; 4];
        compute_dihedral_forces(&top, &positions, &mut forces);

        for atom in 0..4 {
            for dim in 0..3 {
                let f_num =
                    numerical_force(&top, &positions, atom, dim, compute_dihedral_forces);
                let f_ana = forces[atom][dim];
                let err = (f_ana - f_num).abs();
                assert!(
                    err < 1e-4,
                    "dihedral force mismatch: atom={atom} dim={dim} analytical={f_ana} numerical={f_num} err={err}"
                );
            }
        }
    }

    #[test]
    fn dihedral_energy_trans() {
        let mut top = minimal_topology();
        top.n_atoms = 4;
        top.dihedrals = vec![(0, 1, 2, 3, 0, false)];
        top.dihedral_force_constants = Arc::new(vec![1.0]);
        top.dihedral_periodicities = Arc::new(vec![1.0]);
        top.dihedral_phases = Arc::new(vec![0.0]);

        // Trans configuration: phi = PI
        // E = 1.0 * (1 + cos(1*PI - 0)) = 1 + (-1) = 0
        let positions = [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let mut forces = [[0.0; 3]; 4];
        let energy = compute_dihedral_forces(&top, &positions, &mut forces);
        assert!(
            energy.abs() < 0.1,
            "trans dihedral energy should be near 0, got {energy}"
        );
    }

    #[test]
    fn dihedral_multiple_periodicities() {
        let positions = [
            [1.0, 0.3, -0.2],
            [0.0, 0.0, 0.0],
            [0.4, 0.1, 1.5],
            [-0.3, 1.2, 2.1],
        ];

        // Test term 1 alone (n=2)
        let mut top1 = minimal_topology();
        top1.n_atoms = 4;
        top1.dihedrals = vec![(0, 1, 2, 3, 0, true)];
        top1.dihedral_force_constants = Arc::new(vec![1.0]);
        top1.dihedral_periodicities = Arc::new(vec![2.0]);
        top1.dihedral_phases = Arc::new(vec![std::f64::consts::PI]);

        let mut f1 = [[0.0; 3]; 4];
        compute_dihedral_forces(&top1, &positions, &mut f1);
        for atom in 0..4 {
            for dim in 0..3 {
                let f_num =
                    numerical_force(&top1, &positions, atom, dim, compute_dihedral_forces);
                let err = (f1[atom][dim] - f_num).abs();
                assert!(
                    err < 1e-4,
                    "term1 force mismatch: atom={atom} dim={dim} ana={} num={f_num} err={err}",
                    f1[atom][dim]
                );
            }
        }

        // Test term 2 alone (n=3)
        let mut top2 = minimal_topology();
        top2.n_atoms = 4;
        top2.dihedrals = vec![(0, 1, 2, 3, 0, false)];
        top2.dihedral_force_constants = Arc::new(vec![0.5]);
        top2.dihedral_periodicities = Arc::new(vec![3.0]);
        top2.dihedral_phases = Arc::new(vec![0.0]);

        let mut f2 = [[0.0; 3]; 4];
        compute_dihedral_forces(&top2, &positions, &mut f2);
        for atom in 0..4 {
            for dim in 0..3 {
                let f_num =
                    numerical_force(&top2, &positions, atom, dim, compute_dihedral_forces);
                let err = (f2[atom][dim] - f_num).abs();
                assert!(
                    err < 1e-4,
                    "term2 force mismatch: atom={atom} dim={dim} ana={} num={f_num} err={err}",
                    f2[atom][dim]
                );
            }
        }

        // Test both terms combined
        let mut top = minimal_topology();
        top.n_atoms = 4;
        top.dihedrals = vec![
            (0, 1, 2, 3, 0, true),
            (0, 1, 2, 3, 1, false),
        ];
        top.dihedral_force_constants = Arc::new(vec![1.0, 0.5]);
        top.dihedral_periodicities = Arc::new(vec![2.0, 3.0]);
        top.dihedral_phases = Arc::new(vec![std::f64::consts::PI, 0.0]);

        let mut forces = [[0.0; 3]; 4];
        compute_dihedral_forces(&top, &positions, &mut forces);

        for atom in 0..4 {
            for dim in 0..3 {
                let f_num =
                    numerical_force(&top, &positions, atom, dim, compute_dihedral_forces);
                let f_ana = forces[atom][dim];
                let f_sum = f1[atom][dim] + f2[atom][dim];
                let err = (f_ana - f_num).abs();
                assert!(
                    err < 1e-4,
                    "combined force mismatch: atom={atom} dim={dim} analytical={f_ana} numerical={f_num} sum_of_parts={f_sum} err={err}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Combined bonded force tests
    // -----------------------------------------------------------------------

    #[test]
    fn compute_bonded_forces_accumulates() {
        let mut top = minimal_topology();
        top.n_atoms = 2;
        top.bonds = vec![(0, 1)];
        top.bond_types = vec![0];
        top.bond_force_constants = Arc::new(vec![300.0]);
        top.bond_equil_values = Arc::new(vec![1.5]);

        let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];

        // Pre-set forces to a non-zero value to confirm accumulation
        let mut forces = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]];
        let energy = compute_bonded_forces(&top, &positions, &mut forces);

        assert!((energy - 75.0).abs() < 1e-10);
        // Bond force on atom 0 is (+300, 0, 0); accumulated onto (10, 20, 30)
        assert!((forces[0][0] - (10.0 + 300.0)).abs() < 1e-10);
        assert!((forces[0][1] - 20.0).abs() < 1e-10);
        assert!((forces[0][2] - 30.0).abs() < 1e-10);
        // Bond force on atom 1 is (-300, 0, 0); accumulated onto (40, 50, 60)
        assert!((forces[1][0] - (40.0 - 300.0)).abs() < 1e-10);
        assert!((forces[1][1] - 50.0).abs() < 1e-10);
        assert!((forces[1][2] - 60.0).abs() < 1e-10);
    }

    #[test]
    fn bonded_force_translational_invariance() {
        // Total bonded force on the system should be zero (no net translation)
        let mut top = minimal_topology();
        top.n_atoms = 4;
        top.bonds = vec![(0, 1), (1, 2), (2, 3)];
        top.bond_types = vec![0, 0, 0];
        top.bond_force_constants = Arc::new(vec![300.0]);
        top.bond_equil_values = Arc::new(vec![1.5]);
        top.angles = vec![(0, 1, 2, 0), (1, 2, 3, 0)];
        top.angle_force_constants = Arc::new(vec![80.0]);
        top.angle_equil_values = Arc::new(vec![1.91]); // ~109.5 degrees
        top.dihedrals = vec![(0, 1, 2, 3, 0, false)];
        top.dihedral_force_constants = Arc::new(vec![2.0]);
        top.dihedral_periodicities = Arc::new(vec![3.0]);
        top.dihedral_phases = Arc::new(vec![0.0]);

        let positions = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.4, 0.0],
            [3.5, 1.4, 0.5],
        ];
        let mut forces = [[0.0; 3]; 4];
        compute_bonded_forces(&top, &positions, &mut forces);

        // Sum of all forces should be zero
        let total = [
            forces[0][0] + forces[1][0] + forces[2][0] + forces[3][0],
            forces[0][1] + forces[1][1] + forces[2][1] + forces[3][1],
            forces[0][2] + forces[1][2] + forces[2][2] + forces[3][2],
        ];
        for dim in 0..3 {
            assert!(
                total[dim].abs() < 1e-10,
                "net force in dim {dim} = {} (should be zero)",
                total[dim]
            );
        }
    }

    #[test]
    fn empty_topology_returns_zero() {
        let top = minimal_topology();
        let positions: &[[f64; 3]] = &[];
        let mut forces: Vec<[f64; 3]> = vec![];
        let energy = compute_bonded_forces(&top, positions, &mut forces);
        assert!(energy.abs() < 1e-15);
    }
}
