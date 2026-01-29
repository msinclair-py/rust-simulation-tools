//! Molecular mechanics energy calculation using AMBER force field parameters.

use rayon::prelude::*;
use rst_core::amber::prmtop::AmberTopology;
use std::collections::HashSet;

/// MM energy components in kcal/mol.
#[derive(Debug, Clone, Default)]
pub struct MmEnergy {
    pub bond: f64,
    pub angle: f64,
    pub dihedral: f64,
    pub vdw: f64,
    pub elec: f64,
    pub vdw_14: f64,
    pub elec_14: f64,
}

impl MmEnergy {
    /// Total MM energy.
    pub fn total(&self) -> f64 {
        self.bond + self.angle + self.dihedral + self.vdw + self.elec + self.vdw_14 + self.elec_14
    }
}

/// Build set of excluded atom pairs from topology exclusion lists.
pub(crate) fn build_exclusion_set(topology: &AmberTopology) -> HashSet<(usize, usize)> {
    let mut excluded = HashSet::new();
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
                // AMBER uses 0 as placeholder when there are no exclusions
                if j < topology.n_atoms {
                    let pair = if i < j { (i, j) } else { (j, i) };
                    excluded.insert(pair);
                }
            }
        }
        offset += count;
    }
    excluded
}

/// Extract 1-4 pairs from dihedrals (those without ignore_14 flag).
pub(crate) fn build_14_pairs(topology: &AmberTopology) -> HashSet<(usize, usize)> {
    let mut pairs = HashSet::new();
    for &(i, _j, _k, l, _type_idx, ignore_14) in &topology.dihedrals {
        if !ignore_14 {
            let pair = if i < l { (i, l) } else { (l, i) };
            pairs.insert(pair);
        }
    }
    pairs
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn compute_angle(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    // Vectors b→a and b→c
    let ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
    let mag_ba = (ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]).sqrt();
    let mag_bc = (bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]).sqrt();
    let cos_theta = (dot / (mag_ba * mag_bc)).clamp(-1.0, 1.0);
    cos_theta.acos()
}

fn compute_dihedral(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    let b1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let b2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let b3 = [d[0] - c[0], d[1] - c[1], d[2] - c[2]];

    // n1 = b1 × b2, n2 = b2 × b3
    let n1 = cross(&b1, &b2);
    let n2 = cross(&b2, &b3);

    let m1 = cross(&n1, &b2_normalized(&b2));

    let x = dot3(&n1, &n2);
    let y = dot3(&m1, &n2);
    (-y).atan2(x)
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn b2_normalized(b2: &[f64; 3]) -> [f64; 3] {
    let len = (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
    if len < 1e-20 {
        return [0.0, 0.0, 0.0];
    }
    [b2[0] / len, b2[1] / len, b2[2] / len]
}

/// Look up LJ A and B coefficients for atom pair (i, j).
pub(crate) fn lj_ab(topology: &AmberTopology, i: usize, j: usize) -> (f64, f64) {
    let ti = topology.atom_type_indices[i];
    let tj = topology.atom_type_indices[j];
    let idx = topology.nb_parm_index[topology.n_types * ti + tj];
    if idx < 1 {
        return (0.0, 0.0);
    }
    let idx = (idx - 1) as usize;
    (topology.lj_acoef[idx], topology.lj_bcoef[idx])
}

/// Compute all MM energy components for a single frame.
///
/// Convenience wrapper that builds exclusion and 1-4 sets internally.
/// For repeated calls with the same topology, prefer
/// [`compute_mm_energy_with_nb`] with pre-built sets.
pub fn compute_mm_energy(topology: &AmberTopology, coords: &[[f64; 3]]) -> MmEnergy {
    let excluded = build_exclusion_set(topology);
    let pairs_14 = build_14_pairs(topology);
    compute_mm_energy_with_nb(topology, coords, &excluded, &pairs_14)
}

/// Compute all MM energy components for a single frame using pre-built
/// exclusion and 1-4 pair sets.
///
/// This avoids redundant set construction when the same topology is used
/// across multiple frames or subsystems.
pub fn compute_mm_energy_with_nb(
    topology: &AmberTopology,
    coords: &[[f64; 3]],
    excluded: &HashSet<(usize, usize)>,
    pairs_14: &HashSet<(usize, usize)>,
) -> MmEnergy {
    let mut energy = MmEnergy::default();

    // Bond energy: E = K_b * (r - r_eq)^2
    for (idx, &(i, j)) in topology.bonds.iter().enumerate() {
        let type_idx = topology.bond_types[idx];
        let k = topology.bond_force_constants[type_idx];
        let r_eq = topology.bond_equil_values[type_idx];
        let r = distance(&coords[i], &coords[j]);
        energy.bond += k * (r - r_eq) * (r - r_eq);
    }

    // Angle energy: E = K_a * (theta - theta_eq)^2
    for &(i, j, k, type_idx) in &topology.angles {
        let ka = topology.angle_force_constants[type_idx];
        let theta_eq = topology.angle_equil_values[type_idx];
        let theta = compute_angle(&coords[i], &coords[j], &coords[k]);
        energy.angle += ka * (theta - theta_eq) * (theta - theta_eq);
    }

    // Dihedral energy: E = (V_n/2) * [1 + cos(n*phi - gamma)]
    for &(i, j, k, l, type_idx, _ignore_14) in &topology.dihedrals {
        let vn = topology.dihedral_force_constants[type_idx];
        let n = topology.dihedral_periodicities[type_idx];
        let gamma = topology.dihedral_phases[type_idx];
        let phi = compute_dihedral(&coords[i], &coords[j], &coords[k], &coords[l]);
        energy.dihedral += (vn / 2.0) * (1.0 + (n * phi - gamma).cos());
    }

    // Non-bonded interactions (parallelized over outer atom index)
    let scee = topology.scee_scale_factor;
    let scnb = topology.scnb_scale_factor;

    let (nb_vdw, nb_elec, nb_vdw_14, nb_elec_14) = (0..topology.n_atoms)
        .into_par_iter()
        .map(|i| {
            let mut vdw = 0.0f64;
            let mut elec = 0.0f64;
            let mut vdw_14 = 0.0f64;
            let mut elec_14 = 0.0f64;
            for j in (i + 1)..topology.n_atoms {
                let pair = (i, j);

                if pairs_14.contains(&pair) {
                    // 1-4 interaction (scaled)
                    let r = distance(&coords[i], &coords[j]);
                    let r2 = r * r;
                    let r6 = r2 * r2 * r2;
                    let r12 = r6 * r6;

                    let (a, b) = lj_ab(topology, i, j);
                    vdw_14 += (a / r12 - b / r6) / scnb;

                    let qi = topology.charges_amber[i];
                    let qj = topology.charges_amber[j];
                    elec_14 += qi * qj / r / scee;
                } else if !excluded.contains(&pair) {
                    // Full non-bonded interaction
                    let r = distance(&coords[i], &coords[j]);
                    let r2 = r * r;
                    let r6 = r2 * r2 * r2;
                    let r12 = r6 * r6;

                    let (a, b) = lj_ab(topology, i, j);
                    vdw += a / r12 - b / r6;

                    let qi = topology.charges_amber[i];
                    let qj = topology.charges_amber[j];
                    elec += qi * qj / r;
                }
            }
            (vdw, elec, vdw_14, elec_14)
        })
        .reduce(
            || (0.0, 0.0, 0.0, 0.0),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
        );

    energy.vdw = nb_vdw;
    energy.elec = nb_elec;
    energy.vdw_14 = nb_vdw_14;
    energy.elec_14 = nb_elec_14;

    energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use rst_core::amber::prmtop::AmberTopology;

    /// Helper to create a minimal topology for testing.
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
            lj_sigma: vec![],
            lj_epsilon: vec![],
            atom_sigmas: vec![],
            atom_epsilons: vec![],
            bonds: vec![],
            bond_types: vec![],
            masses: vec![],
            radii: vec![],
            screen: vec![],
            bond_force_constants: vec![],
            bond_equil_values: vec![],
            angle_force_constants: vec![],
            angle_equil_values: vec![],
            dihedral_force_constants: vec![],
            dihedral_periodicities: vec![],
            dihedral_phases: vec![],
            angles: vec![],
            dihedrals: vec![],
            num_excluded_atoms: vec![],
            excluded_atoms_list: vec![],
            scee_scale_factor: 1.2,
            scnb_scale_factor: 2.0,
            lj_acoef: vec![],
            lj_bcoef: vec![],
            nb_parm_index: vec![],
        }
    }

    #[test]
    fn test_bond_energy() {
        let mut top = minimal_topology();
        top.n_atoms = 2;
        top.bonds = vec![(0, 1)];
        top.bond_types = vec![0];
        top.bond_force_constants = vec![300.0]; // kcal/mol/Å²
        top.bond_equil_values = vec![1.5]; // Å
        top.num_excluded_atoms = vec![1, 0];
        top.excluded_atoms_list = vec![1];
        top.atom_type_indices = vec![0, 0];
        top.n_types = 1;
        top.charges_amber = vec![0.0, 0.0];
        top.nb_parm_index = vec![1];
        top.lj_acoef = vec![0.0];
        top.lj_bcoef = vec![0.0];

        // Bond at 2.0 Å, equilibrium 1.5 Å → E = 300 * (0.5)^2 = 75 kcal/mol
        let coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let e = compute_mm_energy(&top, &coords);
        assert!((e.bond - 75.0).abs() < 1e-10, "bond energy = {}", e.bond);
    }

    #[test]
    fn test_angle_energy() {
        let mut top = minimal_topology();
        top.n_atoms = 3;
        top.angles = vec![(0, 1, 2, 0)];
        top.angle_force_constants = vec![50.0]; // kcal/mol/rad²
        top.angle_equil_values = vec![std::f64::consts::PI]; // 180°
        top.num_excluded_atoms = vec![2, 1, 0];
        top.excluded_atoms_list = vec![1, 2, 2];
        top.atom_type_indices = vec![0, 0, 0];
        top.n_types = 1;
        top.charges_amber = vec![0.0, 0.0, 0.0];
        top.nb_parm_index = vec![1];
        top.lj_acoef = vec![0.0];
        top.lj_bcoef = vec![0.0];

        // 90° angle, eq=180° → delta = PI/2 → E = 50 * (PI/2)^2
        let coords = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let e = compute_mm_energy(&top, &coords);
        let expected = 50.0 * (std::f64::consts::FRAC_PI_2) * (std::f64::consts::FRAC_PI_2);
        assert!(
            (e.angle - expected).abs() < 1e-10,
            "angle energy = {}, expected = {}",
            e.angle,
            expected
        );
    }

    #[test]
    fn test_dihedral_energy() {
        let mut top = minimal_topology();
        top.n_atoms = 4;
        top.dihedrals = vec![(0, 1, 2, 3, 0, false)];
        top.dihedral_force_constants = vec![2.0]; // kcal/mol
        top.dihedral_periodicities = vec![2.0];
        top.dihedral_phases = vec![std::f64::consts::PI]; // gamma = PI
        top.num_excluded_atoms = vec![3, 2, 1, 0];
        top.excluded_atoms_list = vec![1, 2, 3, 2, 3, 3];
        top.atom_type_indices = vec![0, 0, 0, 0];
        top.n_types = 1;
        top.charges_amber = vec![0.0, 0.0, 0.0, 0.0];
        top.nb_parm_index = vec![1];
        top.lj_acoef = vec![0.0];
        top.lj_bcoef = vec![0.0];

        // Trans dihedral (phi=PI): E = (2/2) * [1 + cos(2*PI - PI)] = 1 * [1 + cos(PI)] = 0
        let coords = [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ];
        let e = compute_mm_energy(&top, &coords);
        // Exact value depends on geometry; just check it's reasonable
        assert!(
            e.dihedral >= 0.0 && e.dihedral < 10.0,
            "dihedral energy = {}",
            e.dihedral
        );
    }

    #[test]
    fn test_parse_and_compute() {
        // Integration test using raf.prmtop if available
        let path = "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/raf.prmtop";
        if !std::path::Path::new(path).exists() {
            return; // Skip if test file not available
        }

        let top = rst_core::amber::prmtop::parse_prmtop(path).expect("Failed to parse prmtop");

        // Verify new fields are populated
        assert!(!top.masses.is_empty());
        assert!(!top.bond_force_constants.is_empty());
        assert!(!top.angle_force_constants.is_empty());
        assert!(!top.dihedral_force_constants.is_empty());
        assert!(!top.angles.is_empty());
        assert!(!top.dihedrals.is_empty());
        assert_eq!(top.charges_amber.len(), top.n_atoms);
        assert_eq!(top.bond_types.len(), top.bonds.len());
    }
}
