//! Bond type perception for GAFF2.
//!
//! Assigns bond orders (single/double/triple/aromatic) based on
//! element valence rules and connectivity patterns.

use crate::molecule::{AcMolecule, BondOrder};

/// Expected valence for each element (standard organic chemistry).
fn expected_valence(atomic_number: u8) -> u8 {
    match atomic_number {
        1 => 1,
        5 => 3,
        6 => 4,
        7 => 3,
        8 => 2,
        9 => 1,
        14 => 4,
        15 => 3, // can be 5 with expanded octet
        16 => 2, // can be 4 or 6 with expanded octet
        17 => 1,
        35 => 1,
        53 => 1,
        _ => 4,
    }
}

/// Perceive bond types using a penalty-based approach.
///
/// For each atom, compute the valence deficit between expected connectivity
/// and current bond order sum, then promote bonds to satisfy valences.
pub fn perceive_bond_types(mol: &mut AcMolecule) -> Result<(), String> {
    // If bonds already have types from the input (e.g., SDF), use those.
    // Only re-perceive if we have all single bonds.
    let all_single = mol.bonds.iter().all(|b| b.order == BondOrder::Single);
    if !all_single {
        return Ok(());
    }

    // Compute valence deficits
    let mut deficits: Vec<i32> = mol
        .atoms
        .iter()
        .map(|a| {
            let expected = adjusted_valence(a.atomic_number, a.formal_charge, a.degree);
            let current: i32 = a.degree as i32; // all single bonds = 1 per bond
            (expected as i32) - current
        })
        .collect();

    // Greedy bond promotion: iterate and promote bonds where both atoms have deficits
    let mut changed = true;
    let mut max_passes = 10;

    while changed && max_passes > 0 {
        changed = false;
        max_passes -= 1;

        for bi in 0..mol.bonds.len() {
            let a1 = mol.bonds[bi].atom1;
            let a2 = mol.bonds[bi].atom2;

            if deficits[a1] > 0 && deficits[a2] > 0 {
                // Can promote this bond
                match mol.bonds[bi].order {
                    BondOrder::Single => {
                        mol.bonds[bi].order = BondOrder::Double;
                        deficits[a1] -= 1;
                        deficits[a2] -= 1;
                        changed = true;
                    }
                    BondOrder::Double if deficits[a1] > 0 && deficits[a2] > 0 => {
                        mol.bonds[bi].order = BondOrder::Triple;
                        deficits[a1] -= 1;
                        deficits[a2] -= 1;
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

/// Compute adjusted valence considering formal charge and hypervalency.
fn adjusted_valence(atomic_number: u8, formal_charge: i32, degree: usize) -> u8 {
    let base = expected_valence(atomic_number);
    let adjusted = (base as i32 - formal_charge) as u8;

    // Allow hypervalent states for P, S
    match atomic_number {
        15 if degree > 3 => 5, // Phosphorus can be pentavalent
        16 if degree > 2 => {
            if degree > 4 {
                6 // Sulfone
            } else {
                4 // Sulfoxide
            }
        }
        7 if degree == 4 => 4, // Quaternary nitrogen
        _ => adjusted,
    }
}
