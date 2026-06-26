//! Aromaticity detection and classification.
//!
//! Classifies rings as AR1-AR5 based on planarity, pi electron count,
//! and Huckel's rule (4n+2).

use crate::molecule::{AcMolecule, AromaticType, BondOrder};
use rst_core::graph::Ring;

/// Detect aromaticity for all rings in the molecule.
pub fn detect_aromaticity(
    mol: &mut AcMolecule,
    rings: &[Ring],
    coords: &[[f64; 3]],
) -> Result<(), String> {
    for ring in rings {
        if ring.size() < 3 || ring.size() > 8 {
            continue;
        }

        // Check planarity
        let planar = ring.is_planar(coords, 0.3);
        if !planar {
            continue;
        }

        // Count pi electrons
        let pi_electrons = count_pi_electrons(mol, ring);

        // Apply Huckel's rule: 4n+2
        let is_huckel = pi_electrons > 0 && (pi_electrons - 2) % 4 == 0;

        // Classify aromatic type
        let ar_type = if is_huckel {
            // Check if all atoms are sp2
            let all_sp2 = ring.atoms.iter().all(|&idx| {
                let a = &mol.atoms[idx];
                a.atomic_number == 6 && a.degree <= 3 // Simple sp2 check for carbon
                    || (a.atomic_number == 7 && a.degree <= 3)
                    || (a.atomic_number == 8 && a.degree == 2)
                    || (a.atomic_number == 16 && a.degree == 2)
            });

            if all_sp2 {
                // Check if any atom donates a lone pair
                let has_lone_pair_donor = ring.atoms.iter().any(|&idx| {
                    let a = &mol.atoms[idx];
                    // N with 3 bonds (pyrrole-like), O or S with 2 bonds (furan-like)
                    (a.atomic_number == 7 && a.degree == 3 && pi_count_for_atom(mol, idx, ring) == 2)
                        || (a.atomic_number == 8 && a.degree == 2)
                        || (a.atomic_number == 16 && a.degree == 2)
                });

                if has_lone_pair_donor {
                    AromaticType::Ar3
                } else {
                    AromaticType::Ar1
                }
            } else {
                AromaticType::Ar1
            }
        } else if planar && has_conjugation(mol, ring) {
            AromaticType::Ar4
        } else {
            AromaticType::None
        };

        if ar_type != AromaticType::None {
            // Set aromatic type on atoms and bonds
            for &idx in &ring.atoms {
                if mol.atoms[idx].aromatic_type == AromaticType::None
                    || ar_type == AromaticType::Ar1
                {
                    mol.atoms[idx].aromatic_type = ar_type;
                }
            }

            // Set ring bonds to aromatic
            if ar_type == AromaticType::Ar1 || ar_type == AromaticType::Ar3 {
                for i in 0..ring.atoms.len() {
                    let a1 = ring.atoms[i];
                    let a2 = ring.atoms[(i + 1) % ring.atoms.len()];
                    if let Some(bi) = mol.get_bond_idx(a1, a2) {
                        mol.bonds[bi].order = BondOrder::Aromatic;
                    }
                }
            }
        }
    }

    // Handle fused ring systems: if two AR1 rings share an edge, both should be AR1
    // and the shared atoms in fused systems might need AR2 classification
    classify_fused_rings(mol, rings);

    Ok(())
}

/// Count pi electrons contributed by atoms in a ring.
fn count_pi_electrons(mol: &AcMolecule, ring: &Ring) -> usize {
    let mut pi_e = 0;
    for &idx in &ring.atoms {
        pi_e += pi_count_for_atom(mol, idx, ring);
    }
    pi_e
}

/// Count pi electrons contributed by a single atom in a ring.
fn pi_count_for_atom(mol: &AcMolecule, atom_idx: usize, ring: &Ring) -> usize {
    let a = &mol.atoms[atom_idx];

    match a.atomic_number {
        6 => {
            // Carbon: contributes 1 pi electron if sp2 (degree <= 3)
            if a.degree <= 3 {
                1
            } else {
                0
            }
        }
        7 => {
            // Nitrogen: check if pyridine-like (1 pi electron) or pyrrole-like (2 pi electrons)
            // Pyridine N: =N- in ring (degree 2, or degree 3 with exocyclic bond)
            // Pyrrole N: -NH- in ring (degree 3, all bonds to ring atoms or H)

            // Count bonds within the ring
            let ring_bonds = a
                .neighbors
                .iter()
                .filter(|&&n| ring.contains(n))
                .count();

            if ring_bonds == 2 && a.degree == 2 {
                // Pyridine-like: sp2 N with lone pair in plane, contributes 1 pi e
                1
            } else if ring_bonds == 2 && a.degree == 3 {
                // Could be pyrrole-like (lone pair in pi system) or pyridine-like (with substituent)
                // If no double bonds to ring neighbors: pyrrole-like -> 2 pi electrons
                let has_double_in_ring = a.neighbors.iter().any(|&n| {
                    ring.contains(n)
                        && mol
                            .get_bond(atom_idx, n)
                            .map_or(false, |b| b.order == BondOrder::Double)
                });
                if has_double_in_ring {
                    1 // Pyridine-like with substituent
                } else {
                    2 // Pyrrole-like
                }
            } else if ring_bonds == 3 {
                1 // Part of fused ring
            } else {
                0
            }
        }
        8 => {
            // Oxygen: furan-like, contributes 2 pi electrons via lone pair
            2
        }
        16 => {
            // Sulfur: thiophene-like, contributes 2 pi electrons
            2
        }
        _ => 0,
    }
}

/// Check if a ring has conjugation (alternating single/double bonds).
fn has_conjugation(mol: &AcMolecule, ring: &Ring) -> bool {
    let mut double_count = 0;
    for i in 0..ring.atoms.len() {
        let a1 = ring.atoms[i];
        let a2 = ring.atoms[(i + 1) % ring.atoms.len()];
        if let Some(bond) = mol.get_bond(a1, a2) {
            if bond.order == BondOrder::Double {
                double_count += 1;
            }
        }
    }
    // At least 2 double bonds for conjugation
    double_count >= 2
}

/// Classify fused ring aromatic systems.
fn classify_fused_rings(mol: &mut AcMolecule, rings: &[Ring]) {
    // Find pairs of aromatic rings that share an edge
    for i in 0..rings.len() {
        for j in (i + 1)..rings.len() {
            let shared = rings[i].shared_atoms(&rings[j]);
            if shared.len() >= 2 {
                // These rings are fused
                // If both are aromatic, set shared atoms to AR2
                let i_aromatic = rings[i]
                    .atoms
                    .iter()
                    .any(|&a| mol.atoms[a].aromatic_type == AromaticType::Ar1);
                let j_aromatic = rings[j]
                    .atoms
                    .iter()
                    .any(|&a| mol.atoms[a].aromatic_type == AromaticType::Ar1);

                if i_aromatic && j_aromatic {
                    // The combined fused system is also aromatic (e.g., naphthalene)
                    // Keep AR1 for all atoms in the system
                    for &idx in &rings[i].atoms {
                        if mol.atoms[idx].aromatic_type == AromaticType::None {
                            mol.atoms[idx].aromatic_type = AromaticType::Ar2;
                        }
                    }
                    for &idx in &rings[j].atoms {
                        if mol.atoms[idx].aromatic_type == AromaticType::None {
                            mol.atoms[idx].aromatic_type = AromaticType::Ar2;
                        }
                    }
                }
            }
        }
    }
}
