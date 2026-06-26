//! Steepest descent and conjugate gradient minimization.
//!
//! Implements a two-phase energy minimization algorithm:
//!
//! 1. **Steepest descent** (SD) for the first `config.sd_cycles` cycles,
//!    with an adaptive step size that grows on downhill steps and shrinks
//!    on uphill steps.
//! 2. **Conjugate gradient** (CG) using the Polak-Ribiere formula for
//!    the remaining cycles, which converges faster near a minimum.
//!
//! The minimizer creates its own [`ForceContext`] internally and handles
//! both periodic (PBC with PME) and vacuum (all-pairs Coulomb) systems
//! based on whether box dimensions are provided.

use log::info;

use rst_core::amber::prmtop::AmberTopology;

use crate::config::MinimizeConfig;
use crate::force::{EnergyComponents, ForceContext};
use crate::restraints::PositionalRestraints;

// ============================================================================
// Result type
// ============================================================================

/// Result of a minimization run.
#[derive(Debug, Clone)]
pub struct MinimizeResult {
    /// Final energy in kcal/mol.
    pub final_energy: f64,
    /// Final RMS gradient in kcal/(mol*A).
    pub final_rms: f64,
    /// Number of minimization cycles performed.
    pub cycles: usize,
    /// Whether the minimization converged (RMS gradient <= criterion).
    pub converged: bool,
    /// Energy components at the final step.
    pub energy_components: EnergyComponents,
}

// ============================================================================
// RMS gradient
// ============================================================================

/// Compute the RMS gradient from a force array.
///
/// ```text
/// RMS = sqrt( sum(fx^2 + fy^2 + fz^2) / (3 * n_atoms) )
/// ```
///
/// Forces are the negative gradient, so the RMS of the forces equals the
/// RMS of the gradient.
fn compute_rms_gradient(forces: &[[f64; 3]]) -> f64 {
    if forces.is_empty() {
        return 0.0;
    }
    let n = forces.len();
    let sum_sq: f64 = forces
        .iter()
        .map(|f| f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
        .sum();
    (sum_sq / (3.0 * n as f64)).sqrt()
}

/// Compute the dot product of two force/vector arrays.
fn forces_dot(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(fa, fb)| fa[0] * fb[0] + fa[1] * fb[1] + fa[2] * fb[2])
        .sum()
}

/// Compute the norm of a force/vector array.
fn forces_norm(f: &[[f64; 3]]) -> f64 {
    forces_dot(f, f).sqrt()
}

// ============================================================================
// Minimizer
// ============================================================================

/// Run energy minimization.
///
/// Performs steepest descent for the first `config.sd_cycles` cycles, then
/// switches to conjugate gradient (Polak-Ribiere) for the remainder up to
/// `config.max_cycles`.
///
/// # Arguments
///
/// * `topology` - AMBER topology with force field parameters.
/// * `positions` - Atomic positions in Angstroms.  Modified in-place to
///   the minimized coordinates.
/// * `box_dims` - Periodic box dimensions.  Pass `None` for vacuum
///   simulations (no PBC, no PME, all-pairs Coulomb).
/// * `config` - Minimization parameters (cycles, convergence, step size,
///   restraints, etc.).
///
/// # Returns
///
/// A [`MinimizeResult`] with the final energy, RMS gradient, cycle count,
/// and convergence status.
///
/// # Errors
///
/// Returns `Err(String)` if the system has no atoms or if an internal
/// inconsistency is detected.
#[allow(clippy::ptr_arg)]
pub fn minimize(
    topology: &AmberTopology,
    positions: &mut Vec<[f64; 3]>,
    box_dims: &mut Option<[f64; 3]>,
    config: &MinimizeConfig,
) -> Result<MinimizeResult, String> {
    let n_atoms = positions.len();
    if n_atoms == 0 {
        return Err("Cannot minimize a system with zero atoms".to_string());
    }
    if n_atoms != topology.n_atoms {
        return Err(format!(
            "Position count ({}) does not match topology atom count ({})",
            n_atoms, topology.n_atoms
        ));
    }

    // --- Set up restraints if requested ---
    let restraints = if let Some(ref mask) = config.restraint_mask {
        if config.restraint_weight > 0.0 {
            Some(PositionalRestraints::from_mask(
                mask,
                &topology.atom_names,
                positions,
                config.restraint_weight,
            ))
        } else {
            None
        }
    } else {
        None
    };

    // --- Create force context ---
    let mut ctx = ForceContext::new(
        topology,
        positions,
        *box_dims,
        config.cutoff,
        restraints,
    );

    // --- Allocate working arrays ---
    let mut forces = vec![[0.0f64; 3]; n_atoms];
    let mut new_positions = vec![[0.0f64; 3]; n_atoms];
    let mut new_forces = vec![[0.0f64; 3]; n_atoms];

    // CG-specific arrays.
    let mut search_dir = vec![[0.0f64; 3]; n_atoms];
    let mut prev_forces = vec![[0.0f64; 3]; n_atoms];

    let mut step_size = config.initial_step_size;
    let mut energy_components;
    let mut energy;
    let mut rms;
    let mut cg_initialized = false;

    // --- Initial force evaluation ---
    energy_components = ctx.compute_forces(topology, positions, box_dims, &mut forces);
    energy = energy_components.total();
    rms = compute_rms_gradient(&forces);

    info!(
        "Minimization starting: energy = {:.4} kcal/mol, RMS = {:.6}",
        energy, rms
    );

    if rms <= config.convergence_rms {
        info!("Already converged at start.");
        return Ok(MinimizeResult {
            final_energy: energy,
            final_rms: rms,
            cycles: 0,
            converged: true,
            energy_components,
        });
    }

    let total_cycles = config.max_cycles;
    let sd_cycles = config.sd_cycles.min(total_cycles);
    let mut cycle = 0;

    // ====================================================================
    // Phase 1: Steepest Descent
    // ====================================================================
    while cycle < sd_cycles {
        // Compute step direction (normalized forces).
        let grad_norm = forces_norm(&forces);
        if grad_norm < 1e-30 {
            break;
        }

        let scale = step_size / grad_norm;
        for i in 0..n_atoms {
            new_positions[i][0] = positions[i][0] + scale * forces[i][0];
            new_positions[i][1] = positions[i][1] + scale * forces[i][1];
            new_positions[i][2] = positions[i][2] + scale * forces[i][2];
        }

        // Evaluate energy at new positions.
        let new_components =
            ctx.compute_forces(topology, &new_positions, box_dims, &mut new_forces);
        let new_energy = new_components.total();

        if new_energy < energy {
            // Accept step, grow step size.
            positions[..n_atoms].copy_from_slice(&new_positions[..n_atoms]);
            forces[..n_atoms].copy_from_slice(&new_forces[..n_atoms]);
            energy = new_energy;
            energy_components = new_components;
            step_size *= 1.2;
        } else {
            // Reject step, shrink step size.
            step_size *= 0.5;
            // Do not update positions or forces; try smaller step next cycle.
        }

        rms = compute_rms_gradient(&forces);
        cycle += 1;

        if cycle % 100 == 0 || cycle == 1 {
            info!(
                "  SD  {:>6}: energy = {:>14.4}  RMS = {:>12.6}  step = {:.6}",
                cycle, energy, rms, step_size
            );
        }

        if rms <= config.convergence_rms {
            info!(
                "Converged at SD cycle {}: RMS = {:.6} <= {:.6}",
                cycle, rms, config.convergence_rms
            );
            return Ok(MinimizeResult {
                final_energy: energy,
                final_rms: rms,
                cycles: cycle,
                converged: true,
                energy_components,
            });
        }
    }

    if sd_cycles > 0 {
        info!(
            "Switching from SD to CG at cycle {}: energy = {:.4}, RMS = {:.6}",
            cycle, energy, rms
        );
    }

    // ====================================================================
    // Phase 2: Conjugate Gradient (Polak-Ribiere)
    // ====================================================================
    // Re-evaluate forces at current positions to ensure consistency.
    energy_components = ctx.compute_forces(topology, positions, box_dims, &mut forces);
    energy = energy_components.total();

    while cycle < total_cycles {
        rms = compute_rms_gradient(&forces);

        if rms <= config.convergence_rms {
            info!(
                "Converged at CG cycle {}: RMS = {:.6} <= {:.6}",
                cycle, rms, config.convergence_rms
            );
            return Ok(MinimizeResult {
                final_energy: energy,
                final_rms: rms,
                cycles: cycle,
                converged: true,
                energy_components,
            });
        }

        if !cg_initialized {
            // First CG iteration: search direction = forces (= -gradient).
            search_dir[..n_atoms].copy_from_slice(&forces[..n_atoms]);
            prev_forces[..n_atoms].copy_from_slice(&forces[..n_atoms]);
            cg_initialized = true;
        } else {
            // Polak-Ribiere beta:
            // beta = dot(f, f - f_prev) / dot(f_prev, f_prev)
            let prev_dot = forces_dot(&prev_forces, &prev_forces);

            let beta = if prev_dot.abs() < 1e-30 {
                0.0
            } else {
                // Compute dot(forces, forces - prev_forces).
                let mut numerator = 0.0;
                for i in 0..n_atoms {
                    for d in 0..3 {
                        numerator += forces[i][d] * (forces[i][d] - prev_forces[i][d]);
                    }
                }
                (numerator / prev_dot).max(0.0) // Restart if negative.
            };

            // Update search direction: s = f + beta * s_old.
            for i in 0..n_atoms {
                search_dir[i][0] = forces[i][0] + beta * search_dir[i][0];
                search_dir[i][1] = forces[i][1] + beta * search_dir[i][1];
                search_dir[i][2] = forces[i][2] + beta * search_dir[i][2];
            }

            // Save current forces as prev.
            prev_forces[..n_atoms].copy_from_slice(&forces[..n_atoms]);
        }

        // Line search along the search direction.
        let dir_norm = forces_norm(&search_dir);
        if dir_norm < 1e-30 {
            break;
        }

        let scale = step_size / dir_norm;
        for i in 0..n_atoms {
            new_positions[i][0] = positions[i][0] + scale * search_dir[i][0];
            new_positions[i][1] = positions[i][1] + scale * search_dir[i][1];
            new_positions[i][2] = positions[i][2] + scale * search_dir[i][2];
        }

        // Evaluate at new positions.
        let new_components =
            ctx.compute_forces(topology, &new_positions, box_dims, &mut new_forces);
        let new_energy = new_components.total();

        if new_energy < energy {
            // Accept step.
            positions[..n_atoms].copy_from_slice(&new_positions[..n_atoms]);
            forces[..n_atoms].copy_from_slice(&new_forces[..n_atoms]);
            energy = new_energy;
            energy_components = new_components;
            step_size *= 1.2;
        } else {
            // Backtrack: halve step and try again next cycle.
            step_size *= 0.5;
            // Reset CG to restart (losing conjugacy after rejection).
            cg_initialized = false;
        }

        cycle += 1;

        if cycle % 100 == 0 {
            rms = compute_rms_gradient(&forces);
            info!(
                "  CG  {:>6}: energy = {:>14.4}  RMS = {:>12.6}  step = {:.6}",
                cycle, energy, rms, step_size
            );
        }
    }

    // Final state.
    rms = compute_rms_gradient(&forces);
    let converged = rms <= config.convergence_rms;

    if converged {
        info!(
            "Minimization converged at cycle {}: energy = {:.4}, RMS = {:.6}",
            cycle, energy, rms
        );
    } else {
        info!(
            "Minimization did NOT converge after {} cycles: energy = {:.4}, RMS = {:.6}",
            cycle, energy, rms
        );
    }

    Ok(MinimizeResult {
        final_energy: energy,
        final_rms: rms,
        cycles: cycle,
        converged,
        energy_components,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_gradient_zero_forces() {
        let forces = vec![[0.0; 3]; 10];
        assert!((compute_rms_gradient(&forces)).abs() < 1e-15);
    }

    #[test]
    fn rms_gradient_uniform() {
        // All forces = [1, 1, 1].
        // sum = n * 3, rms = sqrt(3n / (3n)) = 1.0.
        let forces = vec![[1.0, 1.0, 1.0]; 5];
        let rms = compute_rms_gradient(&forces);
        assert!((rms - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rms_gradient_single_atom() {
        let forces = vec![[3.0, 4.0, 0.0]];
        // sum_sq = 9 + 16 = 25, rms = sqrt(25 / 3) = sqrt(8.333..)
        let rms = compute_rms_gradient(&forces);
        let expected = (25.0_f64 / 3.0).sqrt();
        assert!((rms - expected).abs() < 1e-10);
    }

    #[test]
    fn rms_gradient_empty() {
        let forces: Vec<[f64; 3]> = vec![];
        assert!((compute_rms_gradient(&forces)).abs() < 1e-15);
    }

    #[test]
    fn forces_dot_orthogonal() {
        let a = vec![[1.0, 0.0, 0.0]];
        let b = vec![[0.0, 1.0, 0.0]];
        assert!((forces_dot(&a, &b)).abs() < 1e-15);
    }

    #[test]
    fn forces_dot_parallel() {
        let a = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        // dot = 1+4+9+16+25+36 = 91
        assert!((forces_dot(&a, &b) - 91.0).abs() < 1e-10);
    }

    #[test]
    fn forces_norm_unit() {
        let f = vec![[1.0, 0.0, 0.0]];
        assert!((forces_norm(&f) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn minimize_result_fields() {
        let result = MinimizeResult {
            final_energy: -100.0,
            final_rms: 0.005,
            cycles: 500,
            converged: true,
            energy_components: EnergyComponents::default(),
        };
        assert!((result.final_energy - (-100.0)).abs() < 1e-10);
        assert_eq!(result.cycles, 500);
        assert!(result.converged);
    }
}
