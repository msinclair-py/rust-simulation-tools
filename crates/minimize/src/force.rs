//! Main force evaluation pipeline.
//!
//! Orchestrates the complete AMBER force field evaluation for energy
//! minimization: bonded terms (bonds, angles, dihedrals), non-bonded
//! van der Waals and direct-space electrostatics, PME reciprocal-space
//! electrostatics, 1-4 interactions, and positional restraints.
//!
//! The [`ForceContext`] struct holds all precomputed data (neighbor lists,
//! PME grids, non-bonded parameters) and provides a single
//! [`compute_forces`](ForceContext::compute_forces) entry point that zeros
//! the force array and accumulates all contributions.

use rst_core::amber::prmtop::AmberTopology;

use crate::bonded::compute_bonded_forces;
use crate::neighbor_list::NeighborList;
use crate::nonbonded::{compute_nonbonded_forces, NonbondedParams};
use crate::pme::PmeCalculator;
use crate::restraints::PositionalRestraints;

// ============================================================================
// Constants
// ============================================================================

/// Neighbor list skin distance in Angstroms.
const NEIGHBOR_LIST_SKIN: f64 = 2.0;

// ============================================================================
// Energy Components
// ============================================================================

/// Energy components from a single force evaluation.
///
/// All values are in kcal/mol.
#[derive(Debug, Clone, Default)]
pub struct EnergyComponents {
    /// Bond stretching energy.
    pub bond: f64,
    /// Angle bending energy.
    pub angle: f64,
    /// Dihedral (proper + improper) torsion energy.
    pub dihedral: f64,
    /// Van der Waals (Lennard-Jones) energy from direct-space pairs.
    pub vdw: f64,
    /// Direct-space electrostatic energy.
    pub elec_direct: f64,
    /// Reciprocal-space electrostatic energy (PME, including self-correction).
    pub elec_recip: f64,
    /// 1-4 van der Waals energy.
    pub vdw_14: f64,
    /// 1-4 electrostatic energy.
    pub elec_14: f64,
    /// Positional restraint energy.
    pub restraint: f64,
}

impl EnergyComponents {
    /// Total potential energy (sum of all components).
    pub fn total(&self) -> f64 {
        self.bond
            + self.angle
            + self.dihedral
            + self.vdw
            + self.elec_direct
            + self.elec_recip
            + self.vdw_14
            + self.elec_14
            + self.restraint
    }
}

impl std::fmt::Display for EnergyComponents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BOND={:.4}  ANGLE={:.4}  DIHED={:.4}  VDW={:.4}  \
             EELEC={:.4}  PME={:.4}  1-4VDW={:.4}  1-4EEL={:.4}  \
             RESTRAINT={:.4}  TOTAL={:.4}",
            self.bond,
            self.angle,
            self.dihedral,
            self.vdw,
            self.elec_direct,
            self.elec_recip,
            self.vdw_14,
            self.elec_14,
            self.restraint,
            self.total()
        )
    }
}

// ============================================================================
// ForceContext
// ============================================================================

/// Complete force evaluation context holding precomputed data.
///
/// Encapsulates everything needed to evaluate the full AMBER force field
/// in a single call: non-bonded parameters, neighbor list, PME calculator,
/// and optional positional restraints.
pub struct ForceContext {
    /// Precomputed non-bonded parameters (exclusion bitmaps, 1-4 pairs, scaling).
    pub nb_params: NonbondedParams,
    /// Cell-list based neighbor list (only used for PBC simulations).
    pub neighbor_list: NeighborList,
    /// PME calculator (only used for PBC simulations).
    pub pme: Option<PmeCalculator>,
    /// Optional positional restraints.
    pub restraints: Option<PositionalRestraints>,
    /// Whether this system uses periodic boundary conditions.
    pub use_pbc: bool,
}

impl ForceContext {
    /// Create a new force evaluation context.
    ///
    /// # Arguments
    ///
    /// * `topology` - AMBER topology with force field parameters.
    /// * `positions` - Initial atomic positions in Angstroms.
    /// * `box_dims` - Orthorhombic box dimensions.  Pass `None` for vacuum
    ///   (no PBC, no PME, all-pairs Coulomb).
    /// * `cutoff` - Non-bonded cutoff in Angstroms.
    /// * `restraints` - Optional positional restraints.
    pub fn new(
        topology: &AmberTopology,
        positions: &[[f64; 3]],
        box_dims: Option<[f64; 3]>,
        cutoff: f64,
        restraints: Option<PositionalRestraints>,
    ) -> Self {
        let nb_params = NonbondedParams::from_topology(topology);

        let (neighbor_list, pme, use_pbc) = if let Some(dims) = box_dims {
            let mut nlist = NeighborList::new(cutoff, NEIGHBOR_LIST_SKIN);
            nlist.build(positions, &dims);
            let pme = PmeCalculator::new(
                dims,
                cutoff,
                topology.n_atoms,
                &topology.charges_amber,
            );
            (nlist, Some(pme), true)
        } else {
            // Vacuum: create a placeholder neighbor list that won't be used.
            let nlist = NeighborList::new(cutoff, NEIGHBOR_LIST_SKIN);
            (nlist, None, false)
        };

        Self {
            nb_params,
            neighbor_list,
            pme,
            restraints,
            use_pbc,
        }
    }

    /// Evaluate all forces and return energy components.
    ///
    /// The `forces` array is zeroed first, then all contributions are
    /// accumulated in the following order:
    ///
    /// 1. Bonded forces (bonds, angles, dihedrals).
    /// 2. Neighbor list update (if displacement criterion is exceeded).
    /// 3. Direct-space non-bonded forces (LJ + Coulomb with Ewald splitting).
    /// 4. PME reciprocal-space forces (PBC only).
    /// 5. Positional restraint forces (if configured).
    ///
    /// # Arguments
    ///
    /// * `topology` - AMBER topology.
    /// * `positions` - Current atomic positions.
    /// * `box_dims` - Current box dimensions (may be `None` for vacuum).
    /// * `forces` - Output force array; will be zeroed and filled.
    ///
    /// # Returns
    ///
    /// The decomposed [`EnergyComponents`] for this evaluation.
    pub fn compute_forces(
        &mut self,
        topology: &AmberTopology,
        positions: &[[f64; 3]],
        box_dims: &Option<[f64; 3]>,
        forces: &mut [[f64; 3]],
    ) -> EnergyComponents {
        let mut components = EnergyComponents::default();

        // 1. Zero the forces array.
        for f in forces.iter_mut() {
            *f = [0.0; 3];
        }

        // 2. Compute bonded forces.
        //    The bonded module returns a single total energy; we store it in
        //    the `bond` field.  A future refinement could return a decomposed
        //    struct, but for now the combined value is sufficient.
        let bonded_energy = compute_bonded_forces(topology, positions, forces);
        components.bond = bonded_energy;

        // 3-4. Non-bonded forces (PBC or vacuum).
        if self.use_pbc {
            if let Some(dims) = box_dims {
                // Update neighbor list if atoms have moved too far.
                self.neighbor_list.update_if_needed(positions, dims);

                // Determine the Ewald splitting parameter from the PME
                // calculator.  If PME is not available (shouldn't happen when
                // use_pbc is true), fall back to 0 (plain Coulomb).
                let ewald_alpha = self.pme.as_ref().map_or(0.0, |p| p.alpha());

                // Direct-space non-bonded.
                let (vdw, elec_direct, vdw_14, elec_14) = compute_nonbonded_forces(
                    topology,
                    positions,
                    forces,
                    &self.neighbor_list,
                    &self.nb_params,
                    dims,
                    ewald_alpha,
                );
                components.vdw = vdw;
                components.elec_direct = elec_direct;
                components.vdw_14 = vdw_14;
                components.elec_14 = elec_14;

                // PME reciprocal-space.
                if let Some(ref mut pme) = self.pme {
                    components.elec_recip = pme.compute_reciprocal_forces(
                        positions,
                        &topology.charges_amber,
                        dims,
                        forces,
                    );
                }
            }
        } else {
            // Vacuum: compute all N*(N-1)/2 pairs with plain Coulomb
            // (ewald_alpha = 0 gives erfc(0) = 1, i.e. full Coulomb).
            // We still need a neighbor list with a very large cutoff, or
            // we build one encompassing all pairs.  For simplicity, we build
            // a neighbor list with a cutoff large enough to include all pairs
            // for small systems, or we just reuse the existing one.

            // For vacuum, build an all-pairs neighbor list by using a huge
            // cutoff.  This is appropriate for gas-phase minimizations which
            // typically have few atoms.
            if self.neighbor_list.pairs.is_empty() && positions.len() > 1 {
                // Estimate max distance and build list.
                let mut max_dist2 = 0.0_f64;
                for i in 0..positions.len() {
                    for j in (i + 1)..positions.len() {
                        let dx = positions[i][0] - positions[j][0];
                        let dy = positions[i][1] - positions[j][1];
                        let dz = positions[i][2] - positions[j][2];
                        max_dist2 = max_dist2.max(dx * dx + dy * dy + dz * dz);
                    }
                }
                let big_cutoff = max_dist2.sqrt() + 10.0;
                self.neighbor_list = NeighborList::new(big_cutoff, 0.0);
                // Use a large dummy box for the cell-list algorithm.
                let big_box = [big_cutoff * 3.0; 3];
                self.neighbor_list.build(positions, &big_box);
            }

            // Use a large dummy box for minimum-image (no wrapping effect).
            let dummy_box = [1.0e10, 1.0e10, 1.0e10];
            let (vdw, elec_direct, vdw_14, elec_14) = compute_nonbonded_forces(
                topology,
                positions,
                forces,
                &self.neighbor_list,
                &self.nb_params,
                &dummy_box,
                0.0, // no Ewald splitting in vacuum
            );
            components.vdw = vdw;
            components.elec_direct = elec_direct;
            components.vdw_14 = vdw_14;
            components.elec_14 = elec_14;
            // No reciprocal space in vacuum.
        }

        // 5. Positional restraints.
        if let Some(ref restraints) = self.restraints {
            components.restraint = restraints.compute_forces(positions, forces);
        }

        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    #[test]
    fn energy_components_default_is_zero() {
        let e = EnergyComponents::default();
        assert!((e.total()).abs() < TOL);
    }

    #[test]
    fn energy_components_total() {
        let e = EnergyComponents {
            bond: 1.0,
            angle: 2.0,
            dihedral: 3.0,
            vdw: 4.0,
            elec_direct: 5.0,
            elec_recip: 6.0,
            vdw_14: 7.0,
            elec_14: 8.0,
            restraint: 9.0,
        };
        assert!((e.total() - 45.0).abs() < TOL);
    }

    #[test]
    fn energy_components_display() {
        let e = EnergyComponents {
            bond: 1.0,
            ..EnergyComponents::default()
        };
        let s = format!("{}", e);
        assert!(s.contains("BOND="));
        assert!(s.contains("TOTAL="));
    }

    #[test]
    fn energy_components_clone() {
        let e = EnergyComponents {
            bond: 42.0,
            ..EnergyComponents::default()
        };
        let e2 = e.clone();
        assert!((e2.bond - 42.0).abs() < TOL);
        assert!((e2.total() - 42.0).abs() < TOL);
    }
}
