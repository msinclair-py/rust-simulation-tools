//! Atomic equivalence detection and charge averaging.
//!
//! Uses Morgan-like extended connectivity to group equivalent atoms,
//! then averages charges within each group for symmetry.

use crate::molecule::AcMolecule;
use std::collections::HashMap;

/// Compute Morgan-like extended connectivity invariants.
///
/// 1. Initial invariant = (atomic_number, degree)
/// 2. Iterate: extend with sorted neighbor invariants
/// 3. Stop when the number of unique classes stabilizes
fn compute_equivalence_classes(mol: &AcMolecule) -> Vec<u64> {
    let n = mol.atoms.len();

    // Initial invariant: hash of (atomic_number, degree, n_hydrogens, ring membership)
    let mut invariants: Vec<u64> = mol
        .atoms
        .iter()
        .map(|a| {
            let mut h = a.atomic_number as u64;
            h = h.wrapping_mul(31).wrapping_add(a.degree as u64);
            h = h.wrapping_mul(31).wrapping_add(a.n_hydrogens as u64);
            h = h
                .wrapping_mul(31)
                .wrapping_add(if a.ring_sizes.is_empty() { 0 } else { 1 });
            h = h
                .wrapping_mul(31)
                .wrapping_add(a.aromatic_type as u64);
            h
        })
        .collect();

    // Iterate until stable
    let max_iter = 20;
    for _ in 0..max_iter {
        let mut new_invariants = vec![0u64; n];

        for i in 0..n {
            let mut neighbor_invs: Vec<u64> = mol.atoms[i]
                .neighbors
                .iter()
                .map(|&j| invariants[j])
                .collect();
            neighbor_invs.sort();

            // Hash current invariant with sorted neighbor invariants
            let mut h = invariants[i];
            for &ni in &neighbor_invs {
                h = h.wrapping_mul(31).wrapping_add(ni);
            }
            new_invariants[i] = h;
        }

        // Check if number of unique classes changed
        let old_unique = unique_count(&invariants);
        let new_unique = unique_count(&new_invariants);

        invariants = new_invariants;

        if new_unique == old_unique {
            break;
        }
    }

    invariants
}

fn unique_count(vals: &[u64]) -> usize {
    let mut set = std::collections::HashSet::new();
    for &v in vals {
        set.insert(v);
    }
    set.len()
}

/// Average charges within equivalent atom groups.
pub fn average_equivalent_charges(mol: &mut AcMolecule) {
    let classes = compute_equivalence_classes(mol);

    // Group atoms by equivalence class
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &class) in classes.iter().enumerate() {
        groups.entry(class).or_default().push(i);
    }

    // Average charges within each group
    for (_, indices) in &groups {
        if indices.len() <= 1 {
            continue;
        }

        let avg_charge: f64 =
            indices.iter().map(|&i| mol.atoms[i].charge).sum::<f64>() / indices.len() as f64;

        for &i in indices {
            mol.atoms[i].charge = avg_charge;
        }
    }
}
