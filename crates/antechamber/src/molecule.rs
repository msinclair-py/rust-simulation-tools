//! Working molecule representation for antechamber.
//!
//! Enriched atoms and bonds that accumulate results through the typing pipeline.

/// Aromatic type classification (MOPAC/Antechamber convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AromaticType {
    /// Not aromatic.
    None,
    /// Pure aromatic (benzene-like, 4n+2 in single ring).
    Ar1,
    /// Aromatic in fused system but not individually 4n+2.
    Ar2,
    /// Aromatic via lone pair donation (e.g., pyrrole N, furan O).
    Ar3,
    /// Non-aromatic planar ring with partial conjugation.
    Ar4,
    /// Aromatic with special handling (e.g., charged rings).
    Ar5,
}

/// An atom in the antechamber working molecule.
#[derive(Debug, Clone)]
pub struct AcAtom {
    /// Atom name (e.g., "C1", "N2").
    pub name: String,
    /// Element symbol.
    pub element: String,
    /// Atomic number.
    pub atomic_number: u8,
    /// Cartesian position in Angstroms.
    pub position: [f64; 3],
    /// Indices of bonded atoms.
    pub neighbors: Vec<usize>,
    /// Ring sizes this atom participates in.
    pub ring_sizes: Vec<usize>,
    /// Aromatic classification.
    pub aromatic_type: AromaticType,
    /// GAFF2 atom type.
    pub gaff2_type: String,
    /// BCC atom type index (for AM1-BCC correction lookup).
    pub bcc_type: u32,
    /// Partial charge.
    pub charge: f64,
    /// Formal charge.
    pub formal_charge: i32,
    /// Number of attached hydrogens.
    pub n_hydrogens: usize,
    /// Total number of bonds (degree).
    pub degree: usize,
}

/// Bond order classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    Aromatic,
    Amide,
    Deloc,
}

/// A bond in the antechamber working molecule.
#[derive(Debug, Clone)]
pub struct AcBond {
    /// 0-based index of first atom.
    pub atom1: usize,
    /// 0-based index of second atom.
    pub atom2: usize,
    /// Bond order.
    pub order: BondOrder,
    /// Whether this bond is in a ring.
    pub in_ring: bool,
    /// Original bond type from input (1=single, 2=double, etc.).
    pub original_type: u8,
}

/// A working molecule for the antechamber pipeline.
#[derive(Debug, Clone)]
pub struct AcMolecule {
    /// Molecule name.
    pub name: String,
    /// All atoms.
    pub atoms: Vec<AcAtom>,
    /// All bonds.
    pub bonds: Vec<AcBond>,
}

impl AcMolecule {
    /// Build from an SDF molecule.
    pub fn from_sdf(sdf: &rst_core::sdf::SdfMolecule) -> Self {
        let mut atoms: Vec<AcAtom> = sdf
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let an = rst_core::sdf::SdfMolecule::atomic_number(&a.element);
                AcAtom {
                    name: format!("{}{}", a.element, i + 1),
                    element: a.element.clone(),
                    atomic_number: an,
                    position: a.position,
                    neighbors: Vec::new(),
                    ring_sizes: Vec::new(),
                    aromatic_type: AromaticType::None,
                    gaff2_type: String::new(),
                    bcc_type: 0,
                    charge: 0.0,
                    formal_charge: a.charge,
                    n_hydrogens: 0,
                    degree: 0,
                }
            })
            .collect();

        let bonds: Vec<AcBond> = sdf
            .bonds
            .iter()
            .map(|b| {
                let order = match b.bond_type {
                    1 => BondOrder::Single,
                    2 => BondOrder::Double,
                    3 => BondOrder::Triple,
                    4 => BondOrder::Aromatic,
                    _ => BondOrder::Single,
                };
                AcBond {
                    atom1: b.atom1,
                    atom2: b.atom2,
                    order,
                    in_ring: false,
                    original_type: b.bond_type,
                }
            })
            .collect();

        // Compute neighbor lists
        for bond in &bonds {
            atoms[bond.atom1].neighbors.push(bond.atom2);
            atoms[bond.atom2].neighbors.push(bond.atom1);
        }

        // Compute degree and hydrogen counts
        for i in 0..atoms.len() {
            atoms[i].degree = atoms[i].neighbors.len();
            atoms[i].n_hydrogens = atoms[i]
                .neighbors
                .iter()
                .filter(|&&j| atoms[j].atomic_number == 1)
                .count();
        }

        Self {
            name: sdf.name.clone(),
            atoms,
            bonds,
        }
    }

    /// Build from a mol2 molecule.
    pub fn from_mol2(mol2: &rst_core::mol2::Mol2Molecule) -> Self {
        let element_from_name = |name: &str, atom_type: &str| -> String {
            // Try to extract element from atom type or name
            let s = if !atom_type.is_empty() {
                atom_type
            } else {
                name
            };
            let first = s.chars().next().unwrap_or('X');
            if first.is_ascii_uppercase() {
                let second = s.chars().nth(1);
                match second {
                    Some(c) if c.is_ascii_lowercase() => format!("{}{}", first, c),
                    _ => first.to_string(),
                }
            } else {
                "X".to_string()
            }
        };

        let atomic_number_from_element = |e: &str| -> u8 {
            rst_core::sdf::SdfMolecule::atomic_number(e)
        };

        let mut atoms: Vec<AcAtom> = mol2
            .atoms
            .iter()
            .enumerate()
            .map(|(_i, a)| {
                let elem = element_from_name(&a.name, &a.atom_type);
                let an = atomic_number_from_element(&elem);
                AcAtom {
                    name: a.name.clone(),
                    element: elem,
                    atomic_number: an,
                    position: a.position,
                    neighbors: Vec::new(),
                    ring_sizes: Vec::new(),
                    aromatic_type: AromaticType::None,
                    gaff2_type: a.atom_type.clone(),
                    bcc_type: 0,
                    charge: a.charge,
                    formal_charge: 0,
                    n_hydrogens: 0,
                    degree: 0,
                }
            })
            .collect();

        let bonds: Vec<AcBond> = mol2
            .bonds
            .iter()
            .map(|b| {
                let order = match b.bond_type.as_str() {
                    "1" => BondOrder::Single,
                    "2" => BondOrder::Double,
                    "3" => BondOrder::Triple,
                    "ar" => BondOrder::Aromatic,
                    "am" => BondOrder::Amide,
                    "du" => BondOrder::Deloc,
                    _ => BondOrder::Single,
                };
                let orig = match b.bond_type.as_str() {
                    "1" => 1,
                    "2" => 2,
                    "3" => 3,
                    "ar" => 4,
                    _ => 1,
                };
                AcBond {
                    atom1: b.atom1,
                    atom2: b.atom2,
                    order,
                    in_ring: false,
                    original_type: orig,
                }
            })
            .collect();

        for bond in &bonds {
            atoms[bond.atom1].neighbors.push(bond.atom2);
            atoms[bond.atom2].neighbors.push(bond.atom1);
        }

        for i in 0..atoms.len() {
            atoms[i].degree = atoms[i].neighbors.len();
            atoms[i].n_hydrogens = atoms[i]
                .neighbors
                .iter()
                .filter(|&&j| atoms[j].atomic_number == 1)
                .count();
        }

        Self {
            name: mol2.name.clone(),
            atoms,
            bonds,
        }
    }

    /// Get the bond between two atoms, if it exists.
    pub fn get_bond(&self, a1: usize, a2: usize) -> Option<&AcBond> {
        self.bonds.iter().find(|b| {
            (b.atom1 == a1 && b.atom2 == a2) || (b.atom1 == a2 && b.atom2 == a1)
        })
    }

    /// Get the bond index between two atoms, if it exists.
    pub fn get_bond_idx(&self, a1: usize, a2: usize) -> Option<usize> {
        self.bonds.iter().position(|b| {
            (b.atom1 == a1 && b.atom2 == a2) || (b.atom1 == a2 && b.atom2 == a1)
        })
    }
}
