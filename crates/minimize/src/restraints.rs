//! Positional restraints for energy minimization.
//!
//! Applies harmonic positional restraints to selected atoms during
//! minimization, pulling them back toward reference positions with a
//! configurable force constant.  This is the standard technique used in
//! AMBER to restrain backbone or other atoms while equilibrating solvent
//! or side chains.
//!
//! # Energy and Force
//!
//! For each restrained atom *i* with current position **r**_i and reference
//! position **r**_ref,i the energy contribution is:
//!
//! ```text
//! E_restraint = k * sum_i |r_i - r_ref,i|^2
//! ```
//!
//! and the force on atom *i* is:
//!
//! ```text
//! F_i = -2k * (r_i - r_ref,i)
//! ```

/// Positional restraints on selected atoms.
///
/// Holds the atom indices, their reference positions, and the harmonic
/// force constant.  Forces are computed relative to the reference
/// positions stored at construction time.
#[derive(Debug, Clone)]
pub struct PositionalRestraints {
    /// Indices of restrained atoms.
    atom_indices: Vec<usize>,
    /// Reference positions `[x, y, z]` for each restrained atom.
    reference_positions: Vec<[f64; 3]>,
    /// Force constant in kcal/(mol*A^2).
    force_constant: f64,
}

impl PositionalRestraints {
    /// Create restraints on specified atoms with the given reference positions.
    ///
    /// # Arguments
    ///
    /// * `atom_indices` - 0-based indices of atoms to restrain.
    /// * `reference_positions` - Reference `[x, y, z]` coordinates for each
    ///   restrained atom.  Must have the same length as `atom_indices`.
    /// * `force_constant` - Harmonic force constant in kcal/(mol*A^2).
    ///
    /// # Panics
    ///
    /// Panics if `atom_indices` and `reference_positions` differ in length.
    pub fn new(
        atom_indices: Vec<usize>,
        reference_positions: Vec<[f64; 3]>,
        force_constant: f64,
    ) -> Self {
        assert_eq!(
            atom_indices.len(),
            reference_positions.len(),
            "atom_indices ({}) and reference_positions ({}) must have the same length",
            atom_indices.len(),
            reference_positions.len()
        );
        Self {
            atom_indices,
            reference_positions,
            force_constant,
        }
    }

    /// Create restraints from an atom mask string.
    ///
    /// Supported masks:
    /// - `"backbone"` -- selects atoms named `"N"`, `"CA"`, `"C"`, `"O"`.
    /// - `"all"` -- selects every atom.
    /// - Comma-separated integers -- selects those 0-based atom indices
    ///   (e.g. `"0,3,7"`).
    ///
    /// # Arguments
    ///
    /// * `mask` - Mask string to parse.
    /// * `atom_names` - Atom name for each atom in the system.
    /// * `positions` - Current `[x, y, z]` positions for all atoms.  The
    ///   selected atoms' positions are stored as reference.
    /// * `force_constant` - Harmonic force constant in kcal/(mol*A^2).
    pub fn from_mask(
        mask: &str,
        atom_names: &[String],
        positions: &[[f64; 3]],
        force_constant: f64,
    ) -> Self {
        let trimmed = mask.trim();
        let atom_indices: Vec<usize> = match trimmed {
            "backbone" => {
                let bb_names = ["N", "CA", "C", "O"];
                atom_names
                    .iter()
                    .enumerate()
                    .filter_map(|(i, name)| {
                        if bb_names.contains(&name.as_str()) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            "all" => (0..atom_names.len()).collect(),
            _ => {
                // Comma-separated indices.
                trimmed
                    .split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect()
            }
        };

        let reference_positions: Vec<[f64; 3]> =
            atom_indices.iter().map(|&i| positions[i]).collect();

        Self {
            atom_indices,
            reference_positions,
            force_constant,
        }
    }

    /// Return the indices of restrained atoms.
    pub fn atom_indices(&self) -> &[usize] {
        &self.atom_indices
    }

    /// Return the reference positions of restrained atoms.
    pub fn reference_positions(&self) -> &[[f64; 3]] {
        &self.reference_positions
    }

    /// Return the force constant.
    pub fn force_constant(&self) -> f64 {
        self.force_constant
    }

    /// Compute restraint forces and energy.
    ///
    /// The energy for each restrained atom *i* is:
    ///
    /// ```text
    /// E_i = k * |r_i - r_ref,i|^2
    /// ```
    ///
    /// The force is:
    ///
    /// ```text
    /// F_i = -2k * (r_i - r_ref,i)
    /// ```
    ///
    /// Forces are **added** to the existing values in `forces`, which allows
    /// callers to accumulate restraint forces alongside other contributions.
    ///
    /// # Arguments
    ///
    /// * `positions` - Current positions for all atoms in the system.
    /// * `forces` - Mutable force array for all atoms.  Restraint forces are
    ///   added to the entries corresponding to restrained atoms.
    ///
    /// # Returns
    ///
    /// The total restraint energy in kcal/mol.
    pub fn compute_forces(
        &self,
        positions: &[[f64; 3]],
        forces: &mut [[f64; 3]],
    ) -> f64 {
        let k = self.force_constant;
        let mut energy = 0.0;

        for (idx_in_list, &atom_idx) in self.atom_indices.iter().enumerate() {
            let ref_pos = &self.reference_positions[idx_in_list];
            let cur_pos = &positions[atom_idx];

            let dx = cur_pos[0] - ref_pos[0];
            let dy = cur_pos[1] - ref_pos[1];
            let dz = cur_pos[2] - ref_pos[2];

            // E_i = k * (dx^2 + dy^2 + dz^2)
            energy += k * (dx * dx + dy * dy + dz * dz);

            // F_i = -2k * (r - r_ref)
            let fscale = -2.0 * k;
            forces[atom_idx][0] += fscale * dx;
            forces[atom_idx][1] += fscale * dy;
            forces[atom_idx][2] += fscale * dz;
        }

        energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn new_basic() {
        let r = PositionalRestraints::new(
            vec![0, 2],
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            10.0,
        );
        assert_eq!(r.atom_indices().len(), 2);
        assert_eq!(r.atom_indices()[0], 0);
        assert_eq!(r.atom_indices()[1], 2);
        assert!((r.force_constant() - 10.0).abs() < TOL);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn new_mismatched_lengths() {
        PositionalRestraints::new(vec![0, 1, 2], vec![[0.0; 3]], 1.0);
    }

    #[test]
    fn from_mask_backbone() {
        let names: Vec<String> = ["N", "CA", "CB", "C", "O", "H"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let positions: Vec<[f64; 3]> = (0..6).map(|i| [i as f64, 0.0, 0.0]).collect();

        let r = PositionalRestraints::from_mask("backbone", &names, &positions, 5.0);
        // Backbone: N(0), CA(1), C(3), O(4)
        assert_eq!(r.atom_indices(), &[0, 1, 3, 4]);
        assert!((r.reference_positions()[0][0] - 0.0).abs() < TOL);
        assert!((r.reference_positions()[1][0] - 1.0).abs() < TOL);
        assert!((r.reference_positions()[2][0] - 3.0).abs() < TOL);
        assert!((r.reference_positions()[3][0] - 4.0).abs() < TOL);
    }

    #[test]
    fn from_mask_all() {
        let names: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let positions = vec![[1.0, 2.0, 3.0]; 3];

        let r = PositionalRestraints::from_mask("all", &names, &positions, 1.0);
        assert_eq!(r.atom_indices(), &[0, 1, 2]);
    }

    #[test]
    fn from_mask_indices() {
        let names: Vec<String> = vec!["X".into(); 10];
        let positions: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();

        let r = PositionalRestraints::from_mask("2, 5, 7", &names, &positions, 3.0);
        assert_eq!(r.atom_indices(), &[2, 5, 7]);
        assert!((r.reference_positions()[0][0] - 2.0).abs() < TOL);
        assert!((r.reference_positions()[1][0] - 5.0).abs() < TOL);
        assert!((r.reference_positions()[2][0] - 7.0).abs() < TOL);
    }

    #[test]
    fn compute_forces_zero_displacement() {
        let r = PositionalRestraints::new(
            vec![0, 1],
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            10.0,
        );
        let positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut forces = vec![[0.0; 3]; 2];

        let energy = r.compute_forces(&positions, &mut forces);
        assert!(energy.abs() < TOL);
        assert!(forces[0][0].abs() < TOL);
        assert!(forces[0][1].abs() < TOL);
        assert!(forces[0][2].abs() < TOL);
        assert!(forces[1][0].abs() < TOL);
        assert!(forces[1][1].abs() < TOL);
        assert!(forces[1][2].abs() < TOL);
    }

    #[test]
    fn compute_forces_with_displacement() {
        // Single atom restrained at origin, displaced to (1, 0, 0).
        // k = 5.0
        // E = k * (1^2 + 0 + 0) = 5.0
        // F_x = -2*k * 1.0 = -10.0
        let r = PositionalRestraints::new(vec![0], vec![[0.0, 0.0, 0.0]], 5.0);
        let positions = vec![[1.0, 0.0, 0.0]];
        let mut forces = vec![[0.0; 3]; 1];

        let energy = r.compute_forces(&positions, &mut forces);
        assert!((energy - 5.0).abs() < TOL);
        assert!((forces[0][0] - (-10.0)).abs() < TOL);
        assert!(forces[0][1].abs() < TOL);
        assert!(forces[0][2].abs() < TOL);
    }

    #[test]
    fn compute_forces_accumulates() {
        // Verify that forces are added, not overwritten.
        let r = PositionalRestraints::new(vec![0], vec![[0.0, 0.0, 0.0]], 1.0);
        let positions = vec![[1.0, 0.0, 0.0]];
        let mut forces = vec![[100.0, 0.0, 0.0]];

        let _energy = r.compute_forces(&positions, &mut forces);
        // Force contribution: -2 * 1.0 * 1.0 = -2.0
        // Total: 100.0 + (-2.0) = 98.0
        assert!((forces[0][0] - 98.0).abs() < TOL);
    }

    #[test]
    fn compute_forces_3d_displacement() {
        // Atom restrained at (1,2,3), current at (3,5,7).
        // k = 2.0
        // dx=2, dy=3, dz=4 => |d|^2 = 4+9+16 = 29
        // E = 2 * 29 = 58.0
        // Fx = -4*2 = -8, Fy = -4*3 = -12, Fz = -4*4 = -16
        let r = PositionalRestraints::new(vec![0], vec![[1.0, 2.0, 3.0]], 2.0);
        let positions = vec![[3.0, 5.0, 7.0]];
        let mut forces = vec![[0.0; 3]; 1];

        let energy = r.compute_forces(&positions, &mut forces);
        assert!((energy - 58.0).abs() < TOL);
        assert!((forces[0][0] - (-8.0)).abs() < TOL);
        assert!((forces[0][1] - (-12.0)).abs() < TOL);
        assert!((forces[0][2] - (-16.0)).abs() < TOL);
    }

    #[test]
    fn compute_forces_only_restrained_atoms_affected() {
        // 4 atoms, only atom 1 restrained.
        let r = PositionalRestraints::new(vec![1], vec![[0.0, 0.0, 0.0]], 1.0);
        let positions = vec![
            [10.0, 10.0, 10.0],
            [1.0, 0.0, 0.0],
            [20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0],
        ];
        let mut forces = vec![[0.0; 3]; 4];

        let _energy = r.compute_forces(&positions, &mut forces);
        // Only atom 1 should have nonzero force.
        assert!(forces[0][0].abs() < TOL);
        assert!(forces[0][1].abs() < TOL);
        assert!(forces[0][2].abs() < TOL);
        assert!((forces[1][0] - (-2.0)).abs() < TOL);
        assert!(forces[2][0].abs() < TOL);
        assert!(forces[3][0].abs() < TOL);
    }

    #[test]
    fn clone_and_debug() {
        let r = PositionalRestraints::new(vec![0], vec![[0.0; 3]], 1.0);
        let r2 = r.clone();
        assert_eq!(r2.atom_indices(), r.atom_indices());
        let _ = format!("{:?}", r);
    }
}
