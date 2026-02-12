//! Poisson-Boltzmann polar solvation energy calculation.
//!
//! Performs two finite-difference PB solves (solvated and vacuum/reference)
//! and returns the reaction field energy ΔG_PB = E_solv - E_ref.

use rayon;

use crate::pb_grid::{
    assign_dielectrics, assign_kappa, auto_grid, map_charges, topology_charges, topology_radii,
    PbGrid,
};
use crate::pb_solver::{
    compute_elec_energy, interpolated_boundary, solve_lpbe_multigrid, BoundaryCondition,
    PbSolveResult,
};
use crate::subsystem::{extract_coords, extract_subtopology};
use rst_core::amber::prmtop::AmberTopology;

/// Parameters for PB solvation energy calculation.
#[derive(Debug, Clone)]
pub struct PbParams {
    /// Grid spacing in Angstroms. Default: 0.5.
    pub grid_spacing: f64,
    /// Buffer around molecule in Angstroms. Default: 10.0.
    pub grid_buffer: f64,
    /// Interior (solute) dielectric constant. Default: 1.0.
    pub solute_dielectric: f64,
    /// Exterior (solvent) dielectric constant. Default: 80.0.
    pub solvent_dielectric: f64,
    /// Salt concentration in mol/L. Default: 0.15.
    pub salt_concentration: f64,
    /// Temperature in Kelvin. Default: 300.0.
    pub temperature: f64,
    /// Probe radius for molecular surface in Angstroms. Default: 1.4.
    pub probe_radius: f64,
    /// Ion exclusion radius in Angstroms. Default: 2.0.
    pub ion_radius: f64,
    /// SOR convergence tolerance. Default: 1e-6.
    pub tolerance: f64,
    /// Maximum SOR iterations. Default: 10000.
    pub max_iterations: usize,
    /// Ratio of coarse grid extent to molecule extent for focusing. Default: 4.0.
    /// Set to 0.0 to disable focusing (not recommended for large molecules).
    pub fillratio: f64,
    /// Ratio of coarse grid spacing to fine grid spacing. Default: 8.
    pub fscale: usize,
}

impl Default for PbParams {
    fn default() -> Self {
        Self {
            grid_spacing: 0.5,
            grid_buffer: 10.0,
            solute_dielectric: 1.0,
            solvent_dielectric: 80.0,
            salt_concentration: 0.15,
            temperature: 300.0,
            probe_radius: 1.4,
            ion_radius: 2.0,
            tolerance: 1e-6,
            max_iterations: 10000,
            fillratio: 4.0,
            fscale: 8,
        }
    }
}

/// Result of a PB energy calculation.
#[derive(Debug, Clone)]
pub struct PbEnergy {
    /// Reaction field energy (ΔG_PB) in kcal/mol.
    pub total: f64,
}

/// Compute the Debye-Hückel screening parameter κ in Å⁻¹.
///
/// Reuses the same formula as `gb_energy::compute_kappa`.
pub(crate) fn compute_kappa(salt_conc: f64, solvent_dielectric: f64, temperature: f64) -> f64 {
    if salt_conc <= 0.0 {
        return 0.0;
    }
    let kb = 0.00198688; // kcal/(mol·K)
    let na_factor = 6.022e-4; // N_A * 1e-27
    let factor = 8.0 * std::f64::consts::PI * 332.0522 * na_factor;
    (factor * salt_conc / (solvent_dielectric * kb * temperature)).sqrt()
}

/// Build a coarse grid for focusing.
///
/// The coarse grid covers `fillratio` × the molecule extent, with spacing
/// equal to `fscale` × the fine grid spacing.
fn build_coarse_grid(
    coords: &[[f64; 3]],
    fine_spacing: f64,
    fillratio: f64,
    fscale: usize,
) -> PbGrid {
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for c in coords {
        for d in 0..3 {
            if c[d] < min[d] {
                min[d] = c[d];
            }
            if c[d] > max[d] {
                max[d] = c[d];
            }
        }
    }

    let coarse_spacing = fine_spacing * fscale as f64;
    let mut dims = [0usize; 3];
    let mut origin = [0.0f64; 3];
    for d in 0..3 {
        let mol_extent = max[d] - min[d];
        let grid_extent = mol_extent * fillratio;
        let n = (grid_extent / coarse_spacing).ceil() as usize + 1;
        dims[d] = if n.is_multiple_of(2) { n + 1 } else { n };
        let actual_extent = (dims[d] - 1) as f64 * coarse_spacing;
        let center = 0.5 * (min[d] + max[d]);
        origin[d] = center - 0.5 * actual_extent;
    }

    PbGrid::new(dims, [coarse_spacing; 3], origin)
}

/// Perform a single PB solve (solvated or reference) with optional focusing.
///
/// If `coarse_grid` is provided, first solves on the coarse grid, then
/// interpolates boundary conditions for the fine grid.
fn pb_solve_with_focusing(
    fine_grid: &PbGrid,
    coarse_grid: Option<&PbGrid>,
    coords: &[[f64; 3]],
    charges: &[f64],
    radii: &[f64],
    params: &PbParams,
    eps_in: f64,
    eps_out: f64,
    kappa: f64,
    use_ionic: bool,
) -> (PbSolveResult, f64) {
    // Determine DH BC parameters for the initial (or only) solve
    let bc_kappa = if use_ionic { kappa } else { 0.0 };

    match coarse_grid {
        Some(cgrid) => {
            // === Two-level focusing ===
            // 1. Solve on coarse grid with DH/Coulomb BCs
            let coarse_charges = map_charges(cgrid, coords, charges);
            let coarse_diel =
                assign_dielectrics(cgrid, coords, radii, params.probe_radius, eps_in, eps_out);
            let coarse_kappa = if use_ionic {
                assign_kappa(
                    cgrid,
                    coords,
                    radii,
                    params.ion_radius,
                    kappa,
                    params.solvent_dielectric,
                )
            } else {
                vec![0.0; cgrid.len()]
            };

            let coarse_result = solve_lpbe_multigrid(
                cgrid,
                &coarse_charges,
                &coarse_diel,
                &coarse_kappa,
                BoundaryCondition::DebyeHuckel,
                coords,
                charges,
                bc_kappa,
                eps_out,
                params.tolerance * 10.0, // coarser tolerance for speed
                params.max_iterations,
            );

            // 2. Interpolate coarse solution as BCs for fine grid
            let fine_bc = interpolated_boundary(fine_grid, cgrid, &coarse_result.potential);

            // 3. Solve on fine grid with interpolated BCs
            let fine_charges = map_charges(fine_grid, coords, charges);
            let fine_diel = assign_dielectrics(
                fine_grid,
                coords,
                radii,
                params.probe_radius,
                eps_in,
                eps_out,
            );
            let fine_kappa = if use_ionic {
                assign_kappa(
                    fine_grid,
                    coords,
                    radii,
                    params.ion_radius,
                    kappa,
                    params.solvent_dielectric,
                )
            } else {
                vec![0.0; fine_grid.len()]
            };

            let result = solve_lpbe_multigrid(
                fine_grid,
                &fine_charges,
                &fine_diel,
                &fine_kappa,
                BoundaryCondition::Interpolated(fine_bc),
                coords,
                charges,
                bc_kappa,
                eps_out,
                params.tolerance,
                params.max_iterations,
            );
            let energy = compute_elec_energy(fine_grid, &result.potential, coords, charges);
            (result, energy)
        }
        None => {
            // === Single-level solve ===
            let charge_map = map_charges(fine_grid, coords, charges);
            let diel = assign_dielectrics(
                fine_grid,
                coords,
                radii,
                params.probe_radius,
                eps_in,
                eps_out,
            );
            let kappa_map = if use_ionic {
                assign_kappa(
                    fine_grid,
                    coords,
                    radii,
                    params.ion_radius,
                    kappa,
                    params.solvent_dielectric,
                )
            } else {
                vec![0.0; fine_grid.len()]
            };

            let result = solve_lpbe_multigrid(
                fine_grid,
                &charge_map,
                &diel,
                &kappa_map,
                BoundaryCondition::DebyeHuckel,
                coords,
                charges,
                bc_kappa,
                eps_out,
                params.tolerance,
                params.max_iterations,
            );
            let energy = compute_elec_energy(fine_grid, &result.potential, coords, charges);
            (result, energy)
        }
    }
}

/// Compute PB polar solvation energy for a molecular system.
///
/// Uses two-level focusing (when `fillratio > 0`):
/// 1. Coarse solve on a large grid with DH/Coulomb boundary conditions
/// 2. Fine solve with boundary conditions interpolated from the coarse solution
///
/// Performs paired solves (solvated + reference) and returns
/// ΔG_PB = E_solvated - E_reference (reaction field energy).
pub fn compute_pb_energy(
    topology: &AmberTopology,
    coords: &[[f64; 3]],
    params: &PbParams,
) -> PbEnergy {
    let charges = topology_charges(topology);
    let radii = topology_radii(topology);

    let kappa = compute_kappa(
        params.salt_concentration,
        params.solvent_dielectric,
        params.temperature,
    );

    // Build fine grid (centered on molecule with buffer)
    let fine_grid = auto_grid(coords, params.grid_spacing, params.grid_buffer);

    // Build coarse grid for focusing (if enabled)
    let coarse_grid = if params.fillratio > 0.0 && params.fscale > 1 {
        let cg = build_coarse_grid(coords, params.grid_spacing, params.fillratio, params.fscale);
        log::info!(
            "PB focusing: coarse {}x{}x{} (spacing {:.1}), fine {}x{}x{} (spacing {:.1}) for {} atoms",
            cg.dims[0], cg.dims[1], cg.dims[2], cg.spacing[0],
            fine_grid.dims[0], fine_grid.dims[1], fine_grid.dims[2], fine_grid.spacing[0],
            coords.len()
        );
        Some(cg)
    } else {
        log::info!(
            "PB grid: {}x{}x{} ({} points) for {} atoms",
            fine_grid.dims[0],
            fine_grid.dims[1],
            fine_grid.dims[2],
            fine_grid.len(),
            coords.len()
        );
        None
    };

    // Run solvated and reference solves in parallel
    let ((result_solv, e_solv), (result_ref, e_ref)) = rayon::join(
        || {
            pb_solve_with_focusing(
                &fine_grid,
                coarse_grid.as_ref(),
                coords,
                &charges,
                &radii,
                params,
                params.solute_dielectric,
                params.solvent_dielectric,
                kappa,
                true, // use ionic strength
            )
        },
        || {
            pb_solve_with_focusing(
                &fine_grid,
                coarse_grid.as_ref(),
                coords,
                &charges,
                &radii,
                params,
                params.solute_dielectric,
                params.solute_dielectric, // uniform ε_in for reference
                kappa,
                false, // no ionic strength
            )
        },
    );

    if !result_solv.converged {
        log::warn!(
            "PB solvated solve did not converge after {} iterations (residual: {:.2e})",
            result_solv.iterations,
            result_solv.final_residual
        );
    }

    if !result_ref.converged {
        log::warn!(
            "PB reference solve did not converge after {} iterations (residual: {:.2e})",
            result_ref.iterations,
            result_ref.final_residual
        );
    }

    log::debug!(
        "PB energies: E_solv = {:.4} (iters={}, res={:.2e}), E_ref = {:.4} (iters={}, res={:.2e}), delta = {:.4}",
        e_solv,
        result_solv.iterations,
        result_solv.final_residual,
        e_ref,
        result_ref.iterations,
        result_ref.final_residual,
        e_solv - e_ref
    );

    PbEnergy {
        total: e_solv - e_ref,
    }
}

/// Strip a solvated topology down to specified solute residues and return
/// the extracted sub-topology with matching coordinates.
fn strip_to_solute(
    topology: &AmberTopology,
    coords: &[[f64; 3]],
    solute_residues: &[usize],
) -> Result<(AmberTopology, Vec<[f64; 3]>), String> {
    let selection = topology.build_selection(solute_residues)?;
    let sub_top = extract_subtopology(topology, &selection.atom_indices);
    let sub_coords = extract_coords(coords, &selection.atom_indices);
    log::info!(
        "PB: stripped solvated topology from {} to {} atoms ({} solute residues)",
        topology.n_atoms,
        sub_top.n_atoms,
        solute_residues.len(),
    );
    Ok((sub_top, sub_coords))
}

/// Compute PB polar solvation energy, automatically stripping solvent if
/// `solute_residues` is provided.
///
/// When working with solvated (unstripped) topologies, pass the 0-based
/// residue indices of the solute. The function will extract the sub-topology
/// and slice coordinates before running the PB solver, avoiding the need to
/// write a desolvated topology to disk.
///
/// If `solute_residues` is `None`, the full topology is used as-is (assumes
/// it is already stripped or you want all atoms included).
pub fn compute_pb_energy_solvated(
    topology: &AmberTopology,
    coords: &[[f64; 3]],
    params: &PbParams,
    solute_residues: Option<&[usize]>,
) -> Result<PbEnergy, String> {
    match solute_residues {
        Some(residues) => {
            let (sub_top, sub_coords) = strip_to_solute(topology, coords, residues)?;
            Ok(compute_pb_energy(&sub_top, &sub_coords, params))
        }
        None => Ok(compute_pb_energy(topology, coords, params)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn single_ion_topology(charge: f64, radius: f64) -> AmberTopology {
        AmberTopology {
            n_atoms: 1,
            n_residues: 1,
            n_types: 1,
            atom_names: vec!["Na".to_string()],
            atom_type_indices: vec![0],
            charges: vec![charge],
            charges_amber: vec![charge * 18.2223],
            residue_labels: vec!["Na".to_string()],
            residue_pointers: vec![0],
            lj_sigma: Arc::new(vec![]),
            lj_epsilon: Arc::new(vec![]),
            atom_sigmas: vec![],
            atom_epsilons: vec![],
            bonds: vec![],
            bond_types: vec![],
            masses: vec![23.0],
            radii: vec![radius],
            screen: vec![0.8],
            bond_force_constants: Arc::new(vec![]),
            bond_equil_values: Arc::new(vec![]),
            angle_force_constants: Arc::new(vec![]),
            angle_equil_values: Arc::new(vec![]),
            dihedral_force_constants: Arc::new(vec![]),
            dihedral_periodicities: Arc::new(vec![]),
            dihedral_phases: Arc::new(vec![]),
            angles: vec![],
            dihedrals: vec![],
            num_excluded_atoms: vec![0],
            excluded_atoms_list: vec![],
            scee_scale_factor: 1.2,
            scnb_scale_factor: 2.0,
            lj_acoef: Arc::new(vec![]),
            lj_bcoef: Arc::new(vec![]),
            nb_parm_index: Arc::new(vec![]),
        }
    }

    #[test]
    fn test_born_ion_pb() {
        // Analytical Born energy: ΔG = -0.5 * (1/ε_in - 1/ε_out) * q²/R * 332.0522
        let q = 1.0; // elementary charge
        let r = 2.0; // Angstroms
        let eps_in = 1.0;
        let eps_out = 80.0;

        let analytical = -0.5 * (1.0 / eps_in - 1.0 / eps_out) * q * q / r * 332.0522;

        let top = single_ion_topology(q, r);
        let coords = vec![[0.0, 0.0, 0.0]];

        let params = PbParams {
            grid_spacing: 0.25,
            grid_buffer: 15.0,
            solute_dielectric: eps_in,
            solvent_dielectric: eps_out,
            salt_concentration: 0.0,
            probe_radius: 0.0, // No probe for Born ion test
            ion_radius: 0.0,
            tolerance: 1e-8,
            max_iterations: 20000,
            ..Default::default()
        };

        let result = compute_pb_energy(&top, &coords, &params);

        let error_pct = ((result.total - analytical) / analytical * 100.0).abs();
        println!(
            "Born ion test: PB = {:.4}, analytical = {:.4}, error = {:.2}%",
            result.total, analytical, error_pct
        );
        // Allow up to 15% error — the sharp dielectric boundary at the atomic
        // radius causes grid-discretization error that decreases with finer spacing.
        assert!(
            error_pct < 15.0,
            "PB Born ion energy error too large: {:.2}% (PB={:.4}, analytical={:.4})",
            error_pct,
            result.total,
            analytical,
        );
    }
}
