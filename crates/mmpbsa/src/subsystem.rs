//! Sub-topology extraction for receptor/ligand from a complex topology.
//!
//! Creates a self-consistent `AmberTopology` for a subset of atoms by
//! re-indexing bonds, angles, dihedrals, and exclusion lists.

use rst_core::amber::prmtop::AmberTopology;
use std::collections::HashMap;

/// Extract a sub-topology containing only the specified atom indices.
///
/// All bonded terms (bonds, angles, dihedrals) are filtered to include only
/// those where ALL participating atoms are in the selection. Atom indices in
/// the returned topology are re-numbered starting from 0.
///
/// # Arguments
/// * `topology` - The full complex topology
/// * `atom_indices` - Sorted, 0-based atom indices to include
///
/// # Returns
/// A new `AmberTopology` for the subset of atoms.
pub fn extract_subtopology(topology: &AmberTopology, atom_indices: &[usize]) -> AmberTopology {
    let n_sub = atom_indices.len();

    // Build old→new index mapping
    let mut old_to_new: HashMap<usize, usize> = HashMap::with_capacity(n_sub);
    for (new_idx, &old_idx) in atom_indices.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
    }

    // Subset per-atom arrays
    let atom_names: Vec<String> = atom_indices
        .iter()
        .map(|&i| topology.atom_names[i].clone())
        .collect();
    let atom_type_indices: Vec<usize> = atom_indices
        .iter()
        .map(|&i| topology.atom_type_indices[i])
        .collect();
    let charges: Vec<f64> = atom_indices.iter().map(|&i| topology.charges[i]).collect();
    let charges_amber: Vec<f64> = atom_indices
        .iter()
        .map(|&i| topology.charges_amber[i])
        .collect();
    let masses: Vec<f64> = atom_indices.iter().map(|&i| topology.masses[i]).collect();
    let radii: Vec<f64> = atom_indices.iter().map(|&i| topology.radii[i]).collect();
    let screen: Vec<f64> = atom_indices.iter().map(|&i| topology.screen[i]).collect();
    let atom_sigmas: Vec<f64> = atom_indices
        .iter()
        .map(|&i| topology.atom_sigmas[i])
        .collect();
    let atom_epsilons: Vec<f64> = atom_indices
        .iter()
        .map(|&i| topology.atom_epsilons[i])
        .collect();

    // Build residue information for the subset
    let atom_res = topology.atom_residue_indices();
    let mut sub_residue_labels = Vec::new();
    let mut sub_residue_pointers = Vec::new();
    let mut last_res: Option<usize> = None;
    for (new_idx, &old_idx) in atom_indices.iter().enumerate() {
        let res = atom_res[old_idx];
        if last_res != Some(res) {
            sub_residue_labels.push(topology.residue_labels[res].clone());
            sub_residue_pointers.push(new_idx);
            last_res = Some(res);
        }
    }
    let n_sub_residues = sub_residue_labels.len();

    // Filter and re-index bonds
    let mut sub_bonds = Vec::new();
    let mut sub_bond_types = Vec::new();
    for (idx, &(a, b)) in topology.bonds.iter().enumerate() {
        if let (Some(&na), Some(&nb)) = (old_to_new.get(&a), old_to_new.get(&b)) {
            sub_bonds.push((na, nb));
            sub_bond_types.push(topology.bond_types[idx]);
        }
    }

    // Filter and re-index angles
    let mut sub_angles = Vec::new();
    for &(a, b, c, t) in &topology.angles {
        if let (Some(&na), Some(&nb), Some(&nc)) =
            (old_to_new.get(&a), old_to_new.get(&b), old_to_new.get(&c))
        {
            sub_angles.push((na, nb, nc, t));
        }
    }

    // Filter and re-index dihedrals
    let mut sub_dihedrals = Vec::new();
    for &(a, b, c, d, t, ignore_14) in &topology.dihedrals {
        if let (Some(&na), Some(&nb), Some(&nc), Some(&nd)) = (
            old_to_new.get(&a),
            old_to_new.get(&b),
            old_to_new.get(&c),
            old_to_new.get(&d),
        ) {
            sub_dihedrals.push((na, nb, nc, nd, t, ignore_14));
        }
    }

    // Rebuild exclusion lists
    let mut sub_num_excluded = vec![0usize; n_sub];
    let mut sub_excluded_list = Vec::new();
    let mut offset = 0usize;
    for i in 0..topology.n_atoms {
        let count = if i < topology.num_excluded_atoms.len() {
            topology.num_excluded_atoms[i]
        } else {
            0
        };

        if let Some(&new_i) = old_to_new.get(&i) {
            let mut excl_count = 0;
            for k in 0..count {
                if offset + k < topology.excluded_atoms_list.len() {
                    let j = topology.excluded_atoms_list[offset + k];
                    if let Some(&new_j) = old_to_new.get(&j) {
                        sub_excluded_list.push(new_j);
                        excl_count += 1;
                    }
                }
            }
            sub_num_excluded[new_i] = excl_count;
        }

        offset += count;
    }

    AmberTopology {
        n_atoms: n_sub,
        n_residues: n_sub_residues,
        n_types: topology.n_types,
        atom_names,
        atom_type_indices,
        charges,
        charges_amber,
        residue_labels: sub_residue_labels,
        residue_pointers: sub_residue_pointers,
        lj_sigma: topology.lj_sigma.clone(),
        lj_epsilon: topology.lj_epsilon.clone(),
        atom_sigmas,
        atom_epsilons,
        bonds: sub_bonds,
        bond_types: sub_bond_types,
        masses,
        radii,
        screen,
        bond_force_constants: topology.bond_force_constants.clone(),
        bond_equil_values: topology.bond_equil_values.clone(),
        angle_force_constants: topology.angle_force_constants.clone(),
        angle_equil_values: topology.angle_equil_values.clone(),
        dihedral_force_constants: topology.dihedral_force_constants.clone(),
        dihedral_periodicities: topology.dihedral_periodicities.clone(),
        dihedral_phases: topology.dihedral_phases.clone(),
        angles: sub_angles,
        dihedrals: sub_dihedrals,
        num_excluded_atoms: sub_num_excluded,
        excluded_atoms_list: sub_excluded_list,
        scee_scale_factor: topology.scee_scale_factor,
        scnb_scale_factor: topology.scnb_scale_factor,
        lj_acoef: topology.lj_acoef.clone(),
        lj_bcoef: topology.lj_bcoef.clone(),
        nb_parm_index: topology.nb_parm_index.clone(),
    }
}

/// Extract coordinates for a subset of atoms.
pub fn extract_coords(coords: &[[f64; 3]], atom_indices: &[usize]) -> Vec<[f64; 3]> {
    atom_indices.iter().map(|&i| coords[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_subtopology() {
        let path = "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/ras-raf.prmtop";
        if !std::path::Path::new(path).exists() {
            return;
        }

        let top = rst_core::amber::prmtop::parse_prmtop(path).expect("Failed to parse prmtop");

        // Extract first 10 residues as "receptor"
        let res_indices: Vec<usize> = (0..10).collect();
        let selection = top
            .build_selection(&res_indices)
            .expect("build_selection failed");
        let sub = extract_subtopology(&top, &selection.atom_indices);

        assert_eq!(sub.n_atoms, selection.atom_indices.len());
        assert_eq!(sub.n_residues, 10);
        assert_eq!(sub.charges_amber.len(), sub.n_atoms);
        assert_eq!(sub.radii.len(), sub.n_atoms);

        // All bonds should reference valid atom indices
        for &(a, b) in &sub.bonds {
            assert!(a < sub.n_atoms, "bond atom {} out of range", a);
            assert!(b < sub.n_atoms, "bond atom {} out of range", b);
        }

        // All angles should reference valid indices
        for &(a, b, c, _) in &sub.angles {
            assert!(a < sub.n_atoms);
            assert!(b < sub.n_atoms);
            assert!(c < sub.n_atoms);
        }

        // All dihedrals should reference valid indices
        for &(a, b, c, d, _, _) in &sub.dihedrals {
            assert!(a < sub.n_atoms);
            assert!(b < sub.n_atoms);
            assert!(c < sub.n_atoms);
            assert!(d < sub.n_atoms);
        }
    }

    #[test]
    fn test_subtopology_preserves_atom_count() {
        let path = "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/ras-raf.prmtop";
        if !std::path::Path::new(path).exists() {
            return;
        }

        let top = rst_core::amber::prmtop::parse_prmtop(path).expect("Failed to parse prmtop");

        // Full topology should give back the same atom count
        let all_atoms: Vec<usize> = (0..top.n_atoms).collect();
        let sub = extract_subtopology(&top, &all_atoms);
        assert_eq!(sub.n_atoms, top.n_atoms);
        assert_eq!(sub.bonds.len(), top.bonds.len());
        assert_eq!(sub.angles.len(), top.angles.len());
        assert_eq!(sub.dihedrals.len(), top.dihedrals.len());
    }
}
