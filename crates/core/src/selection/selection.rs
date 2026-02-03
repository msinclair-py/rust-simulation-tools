//! Object-oriented Selection API providing MDAnalysis-style access to atom properties.

use crate::amber::prmtop::AmberTopology;
use std::collections::HashSet;

/// A selection of atoms with direct access to their properties.
///
/// Unlike raw index-based selections, a `Selection` owns copies of all relevant
/// atom data, making it suitable for Python bindings and enabling rich property
/// access without repeated topology lookups.
///
/// # Example
/// ```ignore
/// use rst_core::selection::select_full;
///
/// let sel = select_full(&topology, "protein and name CA")?;
/// println!("Selected {} atoms from {} residues", sel.n_atoms, sel.n_residues);
/// println!("Total mass: {:.2}", sel.total_mass());
/// ```
#[derive(Debug, Clone)]
pub struct Selection {
    /// Atom indices (0-based) in the original topology
    pub indices: Vec<usize>,
    /// Atom names for each selected atom
    pub atom_names: Vec<String>,
    /// Residue name for each selected atom
    pub residue_names: Vec<String>,
    /// Residue index (0-based) for each selected atom
    pub residue_indices: Vec<usize>,
    /// Atomic masses for each selected atom
    pub masses: Vec<f64>,
    /// Partial charges for each selected atom
    pub charges: Vec<f64>,
    /// Born radii for each selected atom
    pub radii: Vec<f64>,
    /// Unique residues as (index, name) pairs, sorted by index
    unique_residue_data: Vec<(usize, String)>,
    /// Number of atoms in the selection
    pub n_atoms: usize,
    /// Number of unique residues in the selection
    pub n_residues: usize,
    /// Optional positions for selected atoms (frame-dependent)
    positions: Option<Vec<[f64; 3]>>,
}

impl Selection {
    /// Create a new Selection from atom indices and a topology.
    ///
    /// # Arguments
    /// * `topology` - The AmberTopology containing atom properties
    /// * `indices` - Atom indices to include in the selection (0-based)
    ///
    /// # Returns
    /// A Selection containing copies of the relevant atom properties
    pub fn from_indices(topology: &AmberTopology, indices: Vec<usize>) -> Self {
        let n_atoms = indices.len();

        // Pre-compute residue indices for the entire topology
        let atom_residue_indices = topology.atom_residue_indices();

        // Extract per-atom data
        let mut atom_names = Vec::with_capacity(n_atoms);
        let mut residue_names = Vec::with_capacity(n_atoms);
        let mut residue_indices = Vec::with_capacity(n_atoms);
        let mut masses = Vec::with_capacity(n_atoms);
        let mut charges = Vec::with_capacity(n_atoms);
        let mut radii = Vec::with_capacity(n_atoms);

        // Track unique residues
        let mut seen_residues = HashSet::new();
        let mut unique_residue_data = Vec::new();

        for &idx in &indices {
            atom_names.push(topology.atom_names[idx].clone());

            let res_idx = atom_residue_indices[idx];
            residue_indices.push(res_idx);
            residue_names.push(topology.residue_labels[res_idx].clone());

            masses.push(topology.masses[idx]);
            charges.push(topology.charges[idx]);
            radii.push(topology.radii[idx]);

            if seen_residues.insert(res_idx) {
                unique_residue_data.push((res_idx, topology.residue_labels[res_idx].clone()));
            }
        }

        // Sort unique residues by index
        unique_residue_data.sort_by_key(|(idx, _)| *idx);
        let n_residues = unique_residue_data.len();

        Selection {
            indices,
            atom_names,
            residue_names,
            residue_indices,
            masses,
            charges,
            radii,
            unique_residue_data,
            n_atoms,
            n_residues,
            positions: None,
        }
    }

    /// Create a new Selection from atom indices, topology, and coordinates.
    ///
    /// # Arguments
    /// * `topology` - The AmberTopology containing atom properties
    /// * `indices` - Atom indices to include in the selection (0-based)
    /// * `coordinates` - Full coordinate array for all atoms in the topology
    ///
    /// # Returns
    /// A Selection containing copies of the relevant atom properties and positions
    pub fn from_indices_with_coordinates(
        topology: &AmberTopology,
        indices: Vec<usize>,
        coordinates: &[[f64; 3]],
    ) -> Self {
        let mut sel = Self::from_indices(topology, indices);

        // Extract positions for selected atoms
        let positions: Vec<[f64; 3]> = sel.indices.iter().map(|&idx| coordinates[idx]).collect();
        sel.positions = Some(positions);

        sel
    }

    /// Get the positions of selected atoms, if available.
    ///
    /// Returns `None` if coordinates were not provided during selection.
    pub fn positions(&self) -> Option<&[[f64; 3]]> {
        self.positions.as_deref()
    }

    /// Check if this selection has positions attached.
    pub fn has_positions(&self) -> bool {
        self.positions.is_some()
    }

    /// Set or update the positions for this selection.
    ///
    /// # Arguments
    /// * `coordinates` - Full coordinate array for all atoms in the topology.
    ///   Positions will be extracted for the selected atom indices.
    pub fn set_positions(&mut self, coordinates: &[[f64; 3]]) {
        let positions: Vec<[f64; 3]> = self.indices.iter().map(|&idx| coordinates[idx]).collect();
        self.positions = Some(positions);
    }

    /// Clear the positions from this selection.
    pub fn clear_positions(&mut self) {
        self.positions = None;
    }

    /// Get unique residue names in order of first occurrence by residue index.
    pub fn unique_residue_names(&self) -> Vec<&str> {
        self.unique_residue_data
            .iter()
            .map(|(_, name)| name.as_str())
            .collect()
    }

    /// Get unique residue indices (0-based) in sorted order.
    pub fn unique_residue_indices(&self) -> Vec<usize> {
        self.unique_residue_data
            .iter()
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// Calculate the total mass of all atoms in the selection.
    pub fn total_mass(&self) -> f64 {
        self.masses.iter().sum()
    }

    /// Calculate the total charge of all atoms in the selection.
    pub fn total_charge(&self) -> f64 {
        self.charges.iter().sum()
    }

    /// Compute the union of two selections (atoms in either selection).
    ///
    /// The resulting selection contains all atoms from both selections,
    /// with duplicates removed and indices sorted.
    pub fn union(&self, other: &Selection) -> Selection {
        let mut combined: HashSet<usize> = self.indices.iter().copied().collect();
        combined.extend(other.indices.iter().copied());

        let mut indices: Vec<usize> = combined.into_iter().collect();
        indices.sort_unstable();

        // We need a topology to rebuild the selection properly
        // Since we don't have one, we'll merge the data directly
        self.merge_with_indices(&indices, other)
    }

    /// Compute the intersection of two selections (atoms in both selections).
    pub fn intersection(&self, other: &Selection) -> Selection {
        let other_set: HashSet<usize> = other.indices.iter().copied().collect();
        let indices: Vec<usize> = self
            .indices
            .iter()
            .copied()
            .filter(|idx| other_set.contains(idx))
            .collect();

        // Build from self's data
        self.subset_by_indices(&indices)
    }

    /// Compute the difference of two selections (atoms in self but not in other).
    pub fn difference(&self, other: &Selection) -> Selection {
        let other_set: HashSet<usize> = other.indices.iter().copied().collect();
        let indices: Vec<usize> = self
            .indices
            .iter()
            .copied()
            .filter(|idx| !other_set.contains(idx))
            .collect();

        self.subset_by_indices(&indices)
    }

    /// Create a subset selection using only atoms at the given indices.
    fn subset_by_indices(&self, target_indices: &[usize]) -> Selection {
        let target_set: HashSet<usize> = target_indices.iter().copied().collect();

        let mut indices = Vec::new();
        let mut atom_names = Vec::new();
        let mut residue_names = Vec::new();
        let mut residue_indices = Vec::new();
        let mut masses = Vec::new();
        let mut charges = Vec::new();
        let mut radii = Vec::new();
        let mut positions = if self.positions.is_some() {
            Some(Vec::new())
        } else {
            None
        };
        let mut seen_residues = HashSet::new();
        let mut unique_residue_data = Vec::new();

        for (i, &idx) in self.indices.iter().enumerate() {
            if target_set.contains(&idx) {
                indices.push(idx);
                atom_names.push(self.atom_names[i].clone());
                residue_names.push(self.residue_names[i].clone());
                let res_idx = self.residue_indices[i];
                residue_indices.push(res_idx);
                masses.push(self.masses[i]);
                charges.push(self.charges[i]);
                radii.push(self.radii[i]);

                if let (Some(pos_out), Some(pos_in)) = (&mut positions, &self.positions) {
                    pos_out.push(pos_in[i]);
                }

                if seen_residues.insert(res_idx) {
                    unique_residue_data.push((res_idx, self.residue_names[i].clone()));
                }
            }
        }

        unique_residue_data.sort_by_key(|(idx, _)| *idx);
        let n_atoms = indices.len();
        let n_residues = unique_residue_data.len();

        Selection {
            indices,
            atom_names,
            residue_names,
            residue_indices,
            masses,
            charges,
            radii,
            unique_residue_data,
            n_atoms,
            n_residues,
            positions,
        }
    }

    /// Merge two selections to create a union, preserving data from both.
    /// Positions are only preserved if both selections have positions.
    fn merge_with_indices(&self, target_indices: &[usize], other: &Selection) -> Selection {
        let self_map: std::collections::HashMap<usize, usize> = self
            .indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();
        let other_map: std::collections::HashMap<usize, usize> = other
            .indices
            .iter()
            .enumerate()
            .map(|(i, &idx)| (idx, i))
            .collect();

        let mut indices = Vec::with_capacity(target_indices.len());
        let mut atom_names = Vec::with_capacity(target_indices.len());
        let mut residue_names = Vec::with_capacity(target_indices.len());
        let mut residue_indices = Vec::with_capacity(target_indices.len());
        let mut masses = Vec::with_capacity(target_indices.len());
        let mut charges = Vec::with_capacity(target_indices.len());
        let mut radii = Vec::with_capacity(target_indices.len());
        // Only keep positions if both selections have them
        let mut positions = if self.positions.is_some() && other.positions.is_some() {
            Some(Vec::with_capacity(target_indices.len()))
        } else {
            None
        };
        let mut seen_residues = HashSet::new();
        let mut unique_residue_data = Vec::new();

        for &idx in target_indices {
            indices.push(idx);

            // Prefer data from self, fall back to other
            if let Some(&i) = self_map.get(&idx) {
                atom_names.push(self.atom_names[i].clone());
                residue_names.push(self.residue_names[i].clone());
                let res_idx = self.residue_indices[i];
                residue_indices.push(res_idx);
                masses.push(self.masses[i]);
                charges.push(self.charges[i]);
                radii.push(self.radii[i]);

                if let (Some(pos_out), Some(pos_in)) = (&mut positions, &self.positions) {
                    pos_out.push(pos_in[i]);
                }

                if seen_residues.insert(res_idx) {
                    unique_residue_data.push((res_idx, self.residue_names[i].clone()));
                }
            } else if let Some(&i) = other_map.get(&idx) {
                atom_names.push(other.atom_names[i].clone());
                residue_names.push(other.residue_names[i].clone());
                let res_idx = other.residue_indices[i];
                residue_indices.push(res_idx);
                masses.push(other.masses[i]);
                charges.push(other.charges[i]);
                radii.push(other.radii[i]);

                if let (Some(pos_out), Some(pos_in)) = (&mut positions, &other.positions) {
                    pos_out.push(pos_in[i]);
                }

                if seen_residues.insert(res_idx) {
                    unique_residue_data.push((res_idx, other.residue_names[i].clone()));
                }
            }
        }

        unique_residue_data.sort_by_key(|(idx, _)| *idx);
        let n_atoms = indices.len();
        let n_residues = unique_residue_data.len();

        Selection {
            indices,
            atom_names,
            residue_names,
            residue_indices,
            masses,
            charges,
            radii,
            unique_residue_data,
            n_atoms,
            n_residues,
            positions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_test_topology() -> AmberTopology {
        // Mini topology: 6 atoms, 2 residues (ALA with 3 atoms, WAT with 3 atoms)
        AmberTopology {
            n_atoms: 6,
            n_residues: 2,
            n_types: 2,
            atom_names: vec![
                "N".to_string(),
                "CA".to_string(),
                "C".to_string(),
                "O".to_string(),
                "H1".to_string(),
                "H2".to_string(),
            ],
            atom_type_indices: vec![0, 1, 0, 1, 0, 0],
            charges: vec![0.1, -0.2, 0.3, -0.4, 0.5, 0.5],
            charges_amber: vec![1.82, -3.64, 5.47, -7.29, 9.11, 9.11],
            residue_labels: vec!["ALA".to_string(), "WAT".to_string()],
            residue_pointers: vec![0, 3],
            lj_sigma: Arc::new(vec![0.3, 0.25]),
            lj_epsilon: Arc::new(vec![0.5, 0.3]),
            atom_sigmas: vec![0.3, 0.25, 0.3, 0.25, 0.3, 0.3],
            atom_epsilons: vec![0.5, 0.3, 0.5, 0.3, 0.5, 0.5],
            bonds: vec![],
            bond_types: vec![],
            masses: vec![14.0, 12.0, 12.0, 16.0, 1.008, 1.008],
            radii: vec![1.5, 1.7, 1.7, 1.5, 1.2, 1.2],
            screen: vec![0.0; 6],
            bond_force_constants: Arc::new(vec![]),
            bond_equil_values: Arc::new(vec![]),
            angle_force_constants: Arc::new(vec![]),
            angle_equil_values: Arc::new(vec![]),
            dihedral_force_constants: Arc::new(vec![]),
            dihedral_periodicities: Arc::new(vec![]),
            dihedral_phases: Arc::new(vec![]),
            angles: vec![],
            dihedrals: vec![],
            num_excluded_atoms: vec![0; 6],
            excluded_atoms_list: vec![],
            scee_scale_factor: 1.2,
            scnb_scale_factor: 2.0,
            lj_acoef: Arc::new(vec![]),
            lj_bcoef: Arc::new(vec![]),
            nb_parm_index: Arc::new(vec![]),
        }
    }

    #[test]
    fn test_selection_from_indices() {
        let top = make_test_topology();
        let sel = Selection::from_indices(&top, vec![0, 1, 2]);

        assert_eq!(sel.n_atoms, 3);
        assert_eq!(sel.n_residues, 1);
        assert_eq!(sel.indices, vec![0, 1, 2]);
        assert_eq!(sel.atom_names, vec!["N", "CA", "C"]);
        assert_eq!(sel.residue_names, vec!["ALA", "ALA", "ALA"]);
        assert_eq!(sel.unique_residue_names(), vec!["ALA"]);
        assert_eq!(sel.unique_residue_indices(), vec![0]);
    }

    #[test]
    fn test_selection_multiple_residues() {
        let top = make_test_topology();
        let sel = Selection::from_indices(&top, vec![1, 4]); // CA from ALA, H1 from WAT

        assert_eq!(sel.n_atoms, 2);
        assert_eq!(sel.n_residues, 2);
        assert_eq!(sel.atom_names, vec!["CA", "H1"]);
        assert_eq!(sel.residue_names, vec!["ALA", "WAT"]);
        assert_eq!(sel.unique_residue_names(), vec!["ALA", "WAT"]);
        assert_eq!(sel.unique_residue_indices(), vec![0, 1]);
    }

    #[test]
    fn test_selection_total_mass() {
        let top = make_test_topology();
        let sel = Selection::from_indices(&top, vec![0, 1, 2]); // N, CA, C
        let expected_mass = 14.0 + 12.0 + 12.0;
        assert!((sel.total_mass() - expected_mass).abs() < 1e-10);
    }

    #[test]
    fn test_selection_total_charge() {
        let top = make_test_topology();
        let sel = Selection::from_indices(&top, vec![0, 1, 2]); // charges: 0.1, -0.2, 0.3
        let expected_charge = 0.1 + (-0.2) + 0.3;
        assert!((sel.total_charge() - expected_charge).abs() < 1e-10);
    }

    #[test]
    fn test_selection_union() {
        let top = make_test_topology();
        let sel1 = Selection::from_indices(&top, vec![0, 1]);
        let sel2 = Selection::from_indices(&top, vec![1, 2, 3]);

        let union = sel1.union(&sel2);
        assert_eq!(union.indices, vec![0, 1, 2, 3]);
        assert_eq!(union.n_atoms, 4);
        assert_eq!(union.n_residues, 2); // ALA and WAT
    }

    #[test]
    fn test_selection_intersection() {
        let top = make_test_topology();
        let sel1 = Selection::from_indices(&top, vec![0, 1, 2]);
        let sel2 = Selection::from_indices(&top, vec![1, 2, 3]);

        let inter = sel1.intersection(&sel2);
        assert_eq!(inter.indices, vec![1, 2]);
        assert_eq!(inter.n_atoms, 2);
    }

    #[test]
    fn test_selection_difference() {
        let top = make_test_topology();
        let sel1 = Selection::from_indices(&top, vec![0, 1, 2, 3]);
        let sel2 = Selection::from_indices(&top, vec![2, 3]);

        let diff = sel1.difference(&sel2);
        assert_eq!(diff.indices, vec![0, 1]);
        assert_eq!(diff.n_atoms, 2);
    }

    #[test]
    fn test_empty_selection() {
        let top = make_test_topology();
        let sel = Selection::from_indices(&top, vec![]);

        assert_eq!(sel.n_atoms, 0);
        assert_eq!(sel.n_residues, 0);
        assert_eq!(sel.total_mass(), 0.0);
        assert_eq!(sel.total_charge(), 0.0);
        assert!(sel.unique_residue_names().is_empty());
        assert!(sel.unique_residue_indices().is_empty());
    }

    #[test]
    fn test_selection_preserves_order() {
        let top = make_test_topology();
        // Indices not in order
        let sel = Selection::from_indices(&top, vec![3, 1, 4]);

        assert_eq!(sel.indices, vec![3, 1, 4]);
        assert_eq!(sel.atom_names, vec!["O", "CA", "H1"]);
    }

    #[test]
    fn test_selection_without_positions() {
        let top = make_test_topology();
        let sel = Selection::from_indices(&top, vec![0, 1, 2]);

        assert!(!sel.has_positions());
        assert!(sel.positions().is_none());
    }

    #[test]
    fn test_selection_with_positions() {
        let top = make_test_topology();
        let coords: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ];

        let sel = Selection::from_indices_with_coordinates(&top, vec![0, 1, 2], &coords);

        assert!(sel.has_positions());
        let pos = sel.positions().unwrap();
        assert_eq!(pos.len(), 3);
        assert_eq!(pos[0], [0.0, 0.0, 0.0]);
        assert_eq!(pos[1], [1.0, 0.0, 0.0]);
        assert_eq!(pos[2], [2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_selection_set_positions() {
        let top = make_test_topology();
        let mut sel = Selection::from_indices(&top, vec![0, 1, 2]);
        assert!(!sel.has_positions());

        let coords: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ];

        sel.set_positions(&coords);
        assert!(sel.has_positions());
        let pos = sel.positions().unwrap();
        assert_eq!(pos[0], [0.0, 0.0, 0.0]);
        assert_eq!(pos[1], [1.0, 0.0, 0.0]);
        assert_eq!(pos[2], [2.0, 0.0, 0.0]);

        sel.clear_positions();
        assert!(!sel.has_positions());
    }

    #[test]
    fn test_selection_intersection_preserves_positions() {
        let top = make_test_topology();
        let coords: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ];

        let sel1 = Selection::from_indices_with_coordinates(&top, vec![0, 1, 2], &coords);
        let sel2 = Selection::from_indices(&top, vec![1, 2, 3]);

        // intersection: sel1 has positions, sel2 does not
        let inter = sel1.intersection(&sel2);
        assert!(inter.has_positions());
        let pos = inter.positions().unwrap();
        assert_eq!(pos.len(), 2); // atoms 1 and 2
        assert_eq!(pos[0], [1.0, 0.0, 0.0]);
        assert_eq!(pos[1], [2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_selection_union_positions() {
        let top = make_test_topology();
        let coords: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ];

        let sel1 = Selection::from_indices_with_coordinates(&top, vec![0, 1], &coords);
        let sel2 = Selection::from_indices_with_coordinates(&top, vec![2, 3], &coords);

        // union: both have positions
        let union = sel1.union(&sel2);
        assert!(union.has_positions());
        let pos = union.positions().unwrap();
        assert_eq!(pos.len(), 4);

        // union: only one has positions - should drop positions
        let sel3 = Selection::from_indices(&top, vec![4, 5]);
        let union2 = sel1.union(&sel3);
        assert!(!union2.has_positions());
    }
}
