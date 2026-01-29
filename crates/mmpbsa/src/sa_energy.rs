//! Non-polar solvation energy via the SA (solvent-accessible surface area) term.
//!
//! ΔG_nonpolar = γ · SASA + β
//!
//! Uses the Shrake-Rupley SASA engine from `rst_core::sasa`.

use rst_core::amber::prmtop::AmberTopology;
use rst_core::sasa::SASAEngine;

/// Parameters for the SA non-polar solvation calculation.
#[derive(Debug, Clone)]
pub struct SaParams {
    /// Surface tension coefficient γ in kcal/(mol·Å²). Default: 0.0072.
    pub surface_tension: f64,
    /// Offset β in kcal/mol. Default: 0.0.
    pub offset: f64,
    /// Solvent probe radius in Å. Default: 1.4.
    pub probe_radius: f64,
    /// Number of sphere points for SASA calculation. Default: 960.
    pub n_sphere_points: usize,
}

impl Default for SaParams {
    fn default() -> Self {
        Self {
            surface_tension: 0.0072,
            offset: 0.0,
            probe_radius: 1.4,
            n_sphere_points: 960,
        }
    }
}

/// Result of SA energy calculation.
#[derive(Debug, Clone)]
pub struct SaEnergy {
    /// Total non-polar solvation energy in kcal/mol.
    pub total: f64,
    /// Total SASA in Å².
    pub total_sasa: f64,
    /// Per-atom SASA in Å².
    pub per_atom_sasa: Vec<f64>,
}

/// Compute the non-polar solvation energy from SASA.
///
/// # Arguments
/// * `topology` - AMBER topology (used for radii and residue mapping)
/// * `coords` - Atomic coordinates in Angstroms
/// * `params` - SA calculation parameters
pub fn compute_sa_energy(
    topology: &AmberTopology,
    coords: &[[f64; 3]],
    params: &SaParams,
) -> SaEnergy {
    let residue_indices = topology.atom_residue_indices();

    let engine = SASAEngine::new(
        coords,
        &topology.radii,
        &residue_indices,
        params.probe_radius,
        params.n_sphere_points,
    );

    let per_atom_sasa = engine.calculate_per_atom_sasa();
    let total_sasa: f64 = per_atom_sasa.iter().sum();
    let total = params.surface_tension * total_sasa + params.offset;

    SaEnergy {
        total,
        total_sasa,
        per_atom_sasa,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_atom_sa() {
        // Single atom: SASA = 4π(r + probe)²
        let top = AmberTopology {
            n_atoms: 1,
            n_residues: 1,
            n_types: 1,
            atom_names: vec!["C".to_string()],
            atom_type_indices: vec![0],
            charges: vec![0.0],
            charges_amber: vec![0.0],
            residue_labels: vec!["ALA".to_string()],
            residue_pointers: vec![0],
            lj_sigma: vec![],
            lj_epsilon: vec![],
            atom_sigmas: vec![],
            atom_epsilons: vec![],
            bonds: vec![],
            bond_types: vec![],
            masses: vec![12.0],
            radii: vec![1.7],
            screen: vec![0.72],
            bond_force_constants: vec![],
            bond_equil_values: vec![],
            angle_force_constants: vec![],
            angle_equil_values: vec![],
            dihedral_force_constants: vec![],
            dihedral_periodicities: vec![],
            dihedral_phases: vec![],
            angles: vec![],
            dihedrals: vec![],
            num_excluded_atoms: vec![0],
            excluded_atoms_list: vec![],
            scee_scale_factor: 1.2,
            scnb_scale_factor: 2.0,
            lj_acoef: vec![],
            lj_bcoef: vec![],
            nb_parm_index: vec![],
        };

        let coords = [[0.0, 0.0, 0.0]];
        let params = SaParams::default();
        let result = compute_sa_energy(&top, &coords, &params);

        let expected_sasa = 4.0 * std::f64::consts::PI * (1.7 + 1.4_f64).powi(2);
        assert!(
            (result.total_sasa - expected_sasa).abs() / expected_sasa < 0.02,
            "SASA = {}, expected ~{}",
            result.total_sasa,
            expected_sasa
        );

        let expected_energy = 0.0072 * expected_sasa;
        assert!(
            (result.total - expected_energy).abs() < 0.1,
            "SA energy = {}, expected ~{}",
            result.total,
            expected_energy
        );
    }

    #[test]
    fn test_sa_with_reference_topology() {
        let path = "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/ras-raf.prmtop";
        if !std::path::Path::new(path).exists() {
            return;
        }

        let top = rst_core::amber::prmtop::parse_prmtop(path).expect("Failed to parse prmtop");
        assert_eq!(top.radii.len(), top.n_atoms);
    }
}
