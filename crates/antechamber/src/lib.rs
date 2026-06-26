#![allow(dead_code)]

//! GAFF2 atom typing, bond typing, ring/aromaticity detection, and charge calculation.
//!
//! Provides a pure-Rust replacement for AmberTools antechamber.
//! Supports GAFF2 atom typing, Gasteiger charges, and AM1-BCC charges.

pub mod aromatic;
pub mod atomtype;
pub mod bcc;
pub mod bondtype;
pub mod data;
pub mod equivalence;
pub mod gasteiger;
pub mod molecule;

use molecule::AcMolecule;

/// Charge calculation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChargeMethod {
    /// AM1-BCC charges (AM1 Mulliken + bond charge corrections).
    Am1Bcc,
    /// Gasteiger electronegativity equalization charges.
    Gasteiger,
}

/// Configuration for antechamber parameterization.
#[derive(Debug, Clone)]
pub struct AntechamberConfig {
    /// Net molecular charge.
    pub net_charge: i32,
    /// Charge calculation method.
    pub charge_method: ChargeMethod,
}

impl Default for AntechamberConfig {
    fn default() -> Self {
        Self {
            net_charge: 0,
            charge_method: ChargeMethod::Am1Bcc,
        }
    }
}

/// Result of antechamber parameterization.
#[derive(Debug, Clone)]
pub struct AntechamberResult {
    /// The parameterized molecule with atom types and charges.
    pub molecule: AcMolecule,
}

/// Run the full antechamber pipeline.
///
/// Pipeline: ring detection -> aromaticity -> bond typing -> atom typing -> charges -> equivalence
pub fn run_antechamber(
    mut mol: AcMolecule,
    config: &AntechamberConfig,
) -> Result<AntechamberResult, String> {
    // Step 1: Ring detection
    let graph = rst_core::graph::MolGraph::new(
        mol.atoms.len(),
        &mol.bonds.iter().map(|b| (b.atom1, b.atom2)).collect::<Vec<_>>(),
    );
    let rings = graph.find_rings(9);

    // Store ring membership
    for ring in &rings {
        for &atom_idx in &ring.atoms {
            if !mol.atoms[atom_idx].ring_sizes.contains(&ring.atoms.len()) {
                mol.atoms[atom_idx].ring_sizes.push(ring.atoms.len());
            }
        }
        for i in 0..ring.atoms.len() {
            let a1 = ring.atoms[i];
            let a2 = ring.atoms[(i + 1) % ring.atoms.len()];
            for bond in &mut mol.bonds {
                if (bond.atom1 == a1 && bond.atom2 == a2)
                    || (bond.atom1 == a2 && bond.atom2 == a1)
                {
                    bond.in_ring = true;
                }
            }
        }
    }

    // Step 2: Bond type perception
    bondtype::perceive_bond_types(&mut mol)?;

    // Step 3: Aromaticity detection
    let coords: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.position).collect();
    aromatic::detect_aromaticity(&mut mol, &rings, &coords)?;

    // Step 4: GAFF2 atom typing
    atomtype::assign_gaff2_types(&mut mol)?;

    // Step 5: BCC atom typing
    atomtype::assign_bcc_types(&mut mol)?;

    // Step 6: Charge calculation
    match config.charge_method {
        ChargeMethod::Am1Bcc => {
            bcc::compute_am1bcc_charges(&mut mol, config.net_charge)?;
        }
        ChargeMethod::Gasteiger => {
            gasteiger::compute_gasteiger_charges(&mut mol)?;
        }
    }

    // Step 7: Equivalence averaging
    equivalence::average_equivalent_charges(&mut mol);

    Ok(AntechamberResult { molecule: mol })
}

/// Convert an antechamber result to a mol2 molecule for output.
pub fn to_mol2(result: &AntechamberResult) -> rst_core::mol2::Mol2Molecule {
    let mol = &result.molecule;

    let atoms: Vec<rst_core::mol2::Mol2Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| rst_core::mol2::Mol2Atom {
            id: i + 1,
            name: a.name.clone(),
            position: a.position,
            atom_type: a.gaff2_type.clone(),
            residue_id: 1,
            residue_name: mol.name.clone(),
            charge: a.charge,
        })
        .collect();

    let bonds: Vec<rst_core::mol2::Mol2Bond> = mol
        .bonds
        .iter()
        .map(|b| {
            let bt = match b.order {
                molecule::BondOrder::Single => "1".to_string(),
                molecule::BondOrder::Double => "2".to_string(),
                molecule::BondOrder::Triple => "3".to_string(),
                molecule::BondOrder::Aromatic => "ar".to_string(),
                molecule::BondOrder::Amide => "am".to_string(),
                molecule::BondOrder::Deloc => "du".to_string(),
            };
            rst_core::mol2::Mol2Bond {
                atom1: b.atom1,
                atom2: b.atom2,
                bond_type: bt,
            }
        })
        .collect();

    rst_core::mol2::Mol2Molecule {
        name: mol.name.clone(),
        atoms,
        bonds,
        substructures: vec![rst_core::mol2::Mol2Substructure {
            id: 1,
            name: mol.name.clone(),
            root_atom: 0,
        }],
    }
}
