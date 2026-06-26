//! AM1-BCC charge correction.
//!
//! Computes AM1 Mulliken charges, then applies bond charge corrections
//! from BCCPARM.DAT to get AM1-BCC charges.

use crate::data;
use crate::molecule::{AcMolecule, BondOrder};

/// A BCC correction entry.
#[derive(Debug, Clone)]
struct BccEntry {
    /// BCC atom type index for atom 1.
    type1: u32,
    /// BCC atom type index for atom 2.
    type2: u32,
    /// Bond order code (1=single, 2=double, etc.).
    bond_order: u8,
    /// Charge correction value.
    correction: f64,
}

/// Parse BCCPARM.DAT into correction entries.
fn parse_bcc_params() -> Vec<BccEntry> {
    let mut entries = Vec::new();

    for line in data::BCCPARM.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }

        // Format: index  type1  type2  bond_order  correction
        let type1: u32 = match parts[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let type2: u32 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let bond_order: u8 = match parts[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let correction: f64 = match parts[4].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        entries.push(BccEntry {
            type1,
            type2,
            bond_order,
            correction,
        });
    }

    entries
}

/// Look up BCC correction for a bond.
fn lookup_correction(entries: &[BccEntry], type1: u32, type2: u32, bond_order: u8) -> f64 {
    // Try direct match
    if let Some(entry) = entries
        .iter()
        .find(|e| e.type1 == type1 && e.type2 == type2 && e.bond_order == bond_order)
    {
        return entry.correction;
    }

    // Try reversed pair (with negated correction)
    if let Some(entry) = entries
        .iter()
        .find(|e| e.type1 == type2 && e.type2 == type1 && e.bond_order == bond_order)
    {
        return -entry.correction;
    }

    0.0
}

/// Bond order to BCC bond order code.
fn bond_order_to_code(order: &BondOrder) -> u8 {
    match order {
        BondOrder::Single => 1,
        BondOrder::Double => 2,
        BondOrder::Triple => 3,
        BondOrder::Aromatic => 7,
        BondOrder::Amide => 9,
        BondOrder::Deloc => 10,
    }
}

/// Compute AM1-BCC charges.
///
/// 1. Compute AM1 Mulliken charges using rst-am1.
/// 2. Apply bond charge corrections from BCCPARM.DAT.
pub fn compute_am1bcc_charges(mol: &mut AcMolecule, net_charge: i32) -> Result<(), String> {
    // Gather atomic numbers and coordinates
    let atomic_numbers: Vec<u8> = mol.atoms.iter().map(|a| a.atomic_number).collect();
    let coords: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();

    // Compute AM1 Mulliken charges
    let am1_charges = rst_am1::compute_am1_charges(&atomic_numbers, &coords, net_charge)?;

    // Apply charges
    for (i, &q) in am1_charges.iter().enumerate() {
        mol.atoms[i].charge = q;
    }

    // Parse BCC correction parameters
    let bcc_entries = parse_bcc_params();

    // Apply bond charge corrections
    for bond in &mol.bonds {
        let type1 = mol.atoms[bond.atom1].bcc_type;
        let type2 = mol.atoms[bond.atom2].bcc_type;
        let bo_code = bond_order_to_code(&bond.order);

        let correction = lookup_correction(&bcc_entries, type1, type2, bo_code);

        if correction.abs() > 1.0e-10 {
            mol.atoms[bond.atom1].charge += correction;
            mol.atoms[bond.atom2].charge -= correction;
        }
    }

    Ok(())
}
