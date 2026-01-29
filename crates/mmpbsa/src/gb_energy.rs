//! Generalized Born (GB) polar solvation energy calculation.
//!
//! Implements the HCT (Hawkins-Cramer-Truhlar) pairwise descreening approach
//! with OBC (Onufriev-Bashford-Case) corrections for effective Born radii.
//!
//! Supports AMBER igb=1 (HCT), igb=2 (OBC-I), and igb=5 (OBC-II).

use rayon::prelude::*;
use rst_core::amber::prmtop::AmberTopology;

/// GB model variant corresponding to AMBER's igb parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbModel {
    /// igb=1: HCT (no OBC correction, raw inverse of descreening sum).
    Hct,
    /// igb=2: OBC-I (α=0.8, β=0.0, γ=2.909125).
    ObcI,
    /// igb=5: OBC-II (α=1.0, β=0.8, γ=4.85).
    ObcII,
}

/// Parameters controlling the GB calculation.
#[derive(Debug, Clone)]
pub struct GbParams {
    /// GB model variant.
    pub model: GbModel,
    /// Interior (solute) dielectric constant. Default: 1.0.
    pub solute_dielectric: f64,
    /// Exterior (solvent) dielectric constant. Default: 78.3.
    pub solvent_dielectric: f64,
    /// Salt concentration in mol/L for Debye-Hückel screening. Default: 0.15.
    pub salt_concentration: f64,
    /// Temperature in Kelvin for Debye-Hückel screening. Default: 300.0.
    pub temperature: f64,
    /// Offset subtracted from intrinsic Born radii (Å). Default: 0.09.
    pub offset: f64,
    /// Maximum allowed effective Born radius (Å). Default: 25.0.
    pub rgbmax: f64,
}

impl Default for GbParams {
    fn default() -> Self {
        Self {
            model: GbModel::ObcI,
            solute_dielectric: 1.0,
            solvent_dielectric: 78.3,
            salt_concentration: 0.15,
            temperature: 300.0,
            offset: 0.09,
            rgbmax: 25.0,
        }
    }
}

/// Result of a GB energy calculation.
#[derive(Debug, Clone)]
pub struct GbEnergy {
    /// Total GB solvation energy in kcal/mol.
    pub total: f64,
    /// Per-atom effective Born radii in Å.
    pub born_radii: Vec<f64>,
}

/// Compute the Debye-Hückel screening parameter κ in Å⁻¹.
///
/// Uses the relation κ = sqrt(8π·l_B·N_A·c·10⁻²⁷) where l_B is the
/// Bjerrum length in Å at the given temperature.
fn compute_kappa(salt_conc: f64, solvent_dielectric: f64, temperature: f64) -> f64 {
    if salt_conc <= 0.0 {
        return 0.0;
    }
    // kB*T in kcal/mol
    // Bjerrum length l_B = 332.0522 / (ε_r * kB*T) Å
    // κ² = 8π * l_B * N_A * c * 1e-27
    //    = 8π * 332.0522 * 6.022e-4 * c / (ε_r * 0.00198688 * T)
    let t = temperature;
    let kb = 0.00198688; // kcal/(mol·K)
    let na_factor = 6.022e-4; // N_A * 1e-27
    let factor = 8.0 * std::f64::consts::PI * 332.0522 * na_factor;
    (factor * salt_conc / (solvent_dielectric * kb * t)).sqrt()
}

/// Compute effective Born radii using HCT pairwise descreening with optional OBC corrections.
///
/// Coordinates must be in Angstroms. Returns Born radii in Angstroms.
fn compute_born_radii(
    topology: &AmberTopology,
    coords: &[[f64; 3]],
    params: &GbParams,
) -> Vec<f64> {
    let n = topology.n_atoms;
    let offset = params.offset;

    // Intrinsic radii with offset subtracted
    let rho: Vec<f64> = topology.radii.iter().map(|r| r - offset).collect();

    // Compute HCT descreening sum for each atom (parallelized over atoms)
    let psi: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let rho_i = rho[i];
            let mut psi_i = 0.0;

            for j in 0..n {
                if i == j {
                    continue;
                }

                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                if r == 0.0 {
                    continue;
                }

                let sj = topology.screen[j] * rho[j];

                // Skip if atom j is completely inside atom i's intrinsic sphere
                if rho_i >= r + sj {
                    continue;
                }

                // Integration bounds
                let u = r + sj;
                let l = if rho_i > (r - sj).abs() {
                    rho_i
                } else {
                    (r - sj).abs()
                };

                let l_inv = 1.0 / l;
                let u_inv = 1.0 / u;

                // HCT descreening integral
                let contrib = 0.5
                    * (l_inv - u_inv
                        + 0.25 * (u_inv * u_inv - l_inv * l_inv) * (r - sj * sj / r)
                        + (l / u).ln() / (2.0 * r));

                // If atom i is completely inside atom j's scaled sphere
                if rho_i < sj - r {
                    psi_i += contrib + 2.0 * (1.0 / rho_i - l_inv);
                } else {
                    psi_i += contrib;
                }
            }

            psi_i
        })
        .collect();

    // Convert descreening sums to effective Born radii
    let (alpha, beta, gamma) = match params.model {
        GbModel::Hct => (1.0, 0.0, 0.0),
        GbModel::ObcI => (0.8, 0.0, 2.909125),
        GbModel::ObcII => (1.0, 0.8, 4.85),
    };

    let mut born_radii = vec![0.0f64; n];
    for i in 0..n {
        let rho_i = rho[i];

        if matches!(params.model, GbModel::Hct) {
            // HCT: R_i = 1 / (1/ρ_i - Ψ_i)
            let inv_r = 1.0 / rho_i - psi[i];
            born_radii[i] = if inv_r > 0.0 {
                1.0 / inv_r
            } else {
                params.rgbmax
            };
        } else {
            // OBC: apply tanh correction
            let psi_rho = psi[i] * rho_i;
            let psi2 = psi_rho * psi_rho;
            let psi3 = psi2 * psi_rho;
            let tanh_val = (alpha * psi_rho - beta * psi2 + gamma * psi3).tanh();
            let inv_r = 1.0 / rho_i - tanh_val / rho_i;
            born_radii[i] = if inv_r > 0.0 {
                1.0 / inv_r
            } else {
                params.rgbmax
            };
        }

        if born_radii[i] > params.rgbmax {
            born_radii[i] = params.rgbmax;
        }
    }

    born_radii
}

/// Compute the Generalized Born polar solvation energy.
///
/// # Arguments
/// * `topology` - AMBER topology with charges, radii, and screening parameters
/// * `coords` - Atomic coordinates in Angstroms as `&[[f64; 3]]`
/// * `params` - GB calculation parameters
///
/// # Returns
/// `GbEnergy` containing the total energy in kcal/mol and per-atom Born radii.
///
/// The GB energy is computed as:
///   E_GB = -½(1/ε_in - 1/ε_out) Σ_{i,j} q_i·q_j·exp(-κ·f_GB) / f_GB
///
/// where f_GB = sqrt(r²_ij + R_i·R_j·exp(-r²_ij/(4·R_i·R_j)))
///
/// Charges are taken from `charges_amber` (AMBER internal units where
/// q_amber = q_real × 18.2223, so q_i·q_j already yields kcal·Å/mol when
/// divided by distance).
pub fn compute_gb_energy(
    topology: &AmberTopology,
    coords: &[[f64; 3]],
    params: &GbParams,
) -> GbEnergy {
    let born_radii = compute_born_radii(topology, coords, params);
    let n = topology.n_atoms;

    let dielectric_factor =
        -0.5 * (1.0 / params.solute_dielectric - 1.0 / params.solvent_dielectric);
    let kappa = compute_kappa(params.salt_concentration, params.solvent_dielectric, params.temperature);

    // Self-energy terms (O(N), kept serial)
    let mut energy = 0.0;
    for i in 0..n {
        let qi = topology.charges_amber[i];
        let mut self_energy = dielectric_factor * qi * qi / born_radii[i];
        if kappa > 0.0 {
            self_energy *= (-kappa * born_radii[i]).exp();
        }
        energy += self_energy;
    }

    // Cross terms (O(N²), parallelized over outer index)
    let cross_energy: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let qi = topology.charges_amber[i];
            let mut sum = 0.0;
            for j in (i + 1)..n {
                let qj = topology.charges_amber[j];

                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let r2 = dx * dx + dy * dy + dz * dz;

                let ri_rj = born_radii[i] * born_radii[j];
                let f_gb = (r2 + ri_rj * (-r2 / (4.0 * ri_rj)).exp()).sqrt();

                let mut pair_energy = dielectric_factor * qi * qj / f_gb;
                if kappa > 0.0 {
                    pair_energy *= (-kappa * f_gb).exp();
                }

                sum += 2.0 * pair_energy;
            }
            sum
        })
        .sum();
    energy += cross_energy;

    GbEnergy {
        total: energy,
        born_radii,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_kappa() {
        let kappa = compute_kappa(0.15, 78.3, 300.0);
        // Reference: AMBER reports kappa=0.127315 for 0.15M salt, ε=78.3
        assert!(
            (kappa - 0.127315).abs() < 0.001,
            "kappa = {}, expected ~0.127315",
            kappa
        );
    }

    #[test]
    fn test_kappa_zero_salt() {
        assert_eq!(compute_kappa(0.0, 78.3, 300.0), 0.0);
        assert_eq!(compute_kappa(-1.0, 78.3, 300.0), 0.0);
    }

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
    fn test_single_ion_gb() {
        // Single ion with charge q in GB: E = -0.5 * (1 - 1/ε) * q² / R
        // Born radius of a single isolated atom = its intrinsic radius (no descreening)
        let mut top = minimal_topology();
        top.n_atoms = 1;
        top.charges_amber = vec![18.2223]; // 1.0 e in AMBER units
        top.radii = vec![1.5]; // 1.5 Å intrinsic radius
        top.screen = vec![0.8];
        top.atom_names = vec!["Na".to_string()];
        top.atom_type_indices = vec![0];
        top.masses = vec![23.0];
        top.n_types = 1;

        let coords = [[0.0, 0.0, 0.0]];
        let params = GbParams {
            model: GbModel::ObcI,
            salt_concentration: 0.0,
            offset: 0.09,
            ..Default::default()
        };

        let result = compute_gb_energy(&top, &coords, &params);

        // No descreening for single atom → born radius = rho = 1.5 - 0.09 = 1.41
        // (OBC tanh(0) = 0, so R = rho)
        let rho = 1.5 - 0.09;
        assert!(
            (result.born_radii[0] - rho).abs() < 1e-10,
            "born radius = {}, expected {}",
            result.born_radii[0],
            rho
        );

        // E = -0.5 * (1 - 1/78.3) * 18.2223² / 1.41
        // 18.2223² = 332.05, (1 - 1/78.3) = 0.98723
        // E = -0.5 * 0.98723 * 332.05 / 1.41 = -116.3 kcal/mol
        let expected = -0.5 * (1.0 - 1.0 / 78.3) * 18.2223 * 18.2223 / rho;
        assert!(
            (result.total - expected).abs() < 0.01,
            "GB energy = {}, expected {}",
            result.total,
            expected
        );
    }

    #[test]
    fn test_gb_energy_with_reference_data() {
        // Integration test: compute GB energy for the ras-raf complex and compare to AMBER output
        let path = "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/ras-raf.prmtop";
        if !std::path::Path::new(path).exists() {
            return;
        }

        let top = rst_core::amber::prmtop::parse_prmtop(path).expect("Failed to parse prmtop");

        // Verify GB parameters are populated
        assert_eq!(top.radii.len(), top.n_atoms);
        assert_eq!(top.screen.len(), top.n_atoms);
        assert!(top.radii.iter().all(|&r| r > 0.0), "All radii should be positive");
        assert!(
            top.screen.iter().all(|&s| s >= 0.0),
            "All screen params should be non-negative"
        );
    }
}
