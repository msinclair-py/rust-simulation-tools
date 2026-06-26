//! Force field parameter data structures.
//!
//! This module defines all parameter types used to represent a molecular mechanics
//! force field in the AMBER family of force fields. The structures here correspond
//! to the parameters found in AMBER parm `.dat` files, `frcmod` files, and related
//! formats.

use std::collections::HashMap;

/// A single atom type definition.
#[derive(Debug, Clone)]
pub struct AtomType {
    /// AMBER atom type name (e.g., "CT", "CA", "N")
    pub name: String,
    /// Element symbol (e.g., "C", "N", "O")
    pub element: String,
    /// Atomic mass in atomic mass units (amu)
    pub mass: f64,
    /// Atomic polarizability in Angstroms^3 (often 0.0)
    pub polarizability: f64,
    /// Hybridization state (e.g., "sp2", "sp3")
    pub hybridization: String,
}

/// Bond stretching parameters: E = k * (r - r_eq)^2
#[derive(Debug, Clone)]
pub struct BondParam {
    /// First atom type
    pub type1: String,
    /// Second atom type
    pub type2: String,
    /// Force constant in kcal/(mol*A^2)
    pub force_constant: f64,
    /// Equilibrium bond length in Angstroms
    pub eq_length: f64,
}

/// Angle bending parameters: E = k * (theta - theta_eq)^2
#[derive(Debug, Clone)]
pub struct AngleParam {
    /// First atom type
    pub type1: String,
    /// Central atom type
    pub type2: String,
    /// Third atom type
    pub type3: String,
    /// Force constant in kcal/(mol*rad^2)
    pub force_constant: f64,
    /// Equilibrium angle in degrees (converted to radians when needed)
    pub eq_angle: f64,
}

/// A single term in a Fourier dihedral expansion.
///
/// Dihedrals may have multiple terms per type combination, each with a
/// different periodicity and phase.
#[derive(Debug, Clone)]
pub struct DihedralTerm {
    /// Barrier height V_n/2 in kcal/mol (the divisor of 2 is already applied
    /// in the `.dat` file, i.e., this is the PK value)
    pub force_constant: f64,
    /// Periodicity n (integer stored as f64 in AMBER format)
    pub periodicity: f64,
    /// Phase angle gamma in degrees
    pub phase: f64,
}

/// Dihedral (torsion) parameters.
///
/// A dihedral may have multiple Fourier terms stored in [`DihedralTerm`].
/// Atom types may be `"X"` to indicate a wildcard match.
#[derive(Debug, Clone)]
pub struct DihedralParam {
    /// First atom type (can be `"X"` for wildcard)
    pub type1: String,
    /// Second atom type (central bond atom)
    pub type2: String,
    /// Third atom type (central bond atom)
    pub type3: String,
    /// Fourth atom type (can be `"X"` for wildcard)
    pub type4: String,
    /// IDIVF: number of paths through the dihedral (force constant is divided by this)
    pub divider: f64,
    /// Fourier terms for this dihedral
    pub terms: Vec<DihedralTerm>,
}

/// Improper dihedral parameters.
///
/// In AMBER convention, `type3` is always the central (sp2) atom.
/// `type1` and `type4` are frequently `"X"` wildcards.
#[derive(Debug, Clone)]
pub struct ImproperParam {
    /// First atom type (can be `"X"` for wildcard)
    pub type1: String,
    /// Second atom type
    pub type2: String,
    /// Central (sp2) atom type -- always position 3 in AMBER convention
    pub type3: String,
    /// Fourth atom type (can be `"X"` for wildcard)
    pub type4: String,
    /// Force constant in kcal/mol
    pub force_constant: f64,
    /// Periodicity
    pub periodicity: f64,
    /// Phase angle in degrees
    pub phase: f64,
}

/// Lennard-Jones (van der Waals) parameters for a single atom type.
///
/// Uses Lorentz-Berthelot combining rules:
///   sigma_ij = (sigma_i + sigma_j) / 2,
///   epsilon_ij = sqrt(eps_i * eps_j)
#[derive(Debug, Clone)]
pub struct LjParam {
    /// Atom type name
    pub atom_type: String,
    /// R* (half the minimum-energy distance) in Angstroms (Rmin/2 in AMBER notation)
    pub radius: f64,
    /// Well depth epsilon in kcal/mol
    pub well_depth: f64,
}

/// Equivalencing of atom types for non-bonded parameters.
///
/// In parm `.dat` files, certain atom types are mapped to share the same
/// van der Waals parameters. The first entry in [`types`] is the "master"
/// type whose LJ parameters are used.
#[derive(Debug, Clone)]
pub struct NbEquiv {
    /// List of atom types sharing the same vdW parameters.
    /// The first entry is the "master" type.
    pub types: Vec<String>,
}

/// Complete set of force field parameters.
///
/// Aggregates all bonded and non-bonded parameter types, and provides
/// lookup methods with wildcard and equivalence support.
#[derive(Debug, Clone, Default)]
pub struct ForceFieldParams {
    /// Atom type definitions keyed by type name
    pub atom_types: HashMap<String, AtomType>,
    /// Bond stretching parameters
    pub bond_params: Vec<BondParam>,
    /// Angle bending parameters
    pub angle_params: Vec<AngleParam>,
    /// Proper dihedral (torsion) parameters
    pub dihedral_params: Vec<DihedralParam>,
    /// Improper dihedral parameters
    pub improper_params: Vec<ImproperParam>,
    /// Lennard-Jones parameters keyed by atom type name
    pub lj_params: HashMap<String, LjParam>,
    /// Non-bonded equivalence groups
    pub nb_equivalences: Vec<NbEquiv>,
    /// 1-4 electrostatic scaling factor (default 1.2, meaning divide by 1.2)
    pub scee: f64,
    /// 1-4 van der Waals scaling factor (default 2.0, meaning divide by 2.0)
    pub scnb: f64,
}

impl ForceFieldParams {
    /// Create a new `ForceFieldParams` with AMBER default scaling factors.
    pub fn new() -> Self {
        Self {
            scee: 1.2,
            scnb: 2.0,
            ..Default::default()
        }
    }

    /// Merge another parameter set into this one (frcmod overlay).
    ///
    /// New parameters override existing ones for matching type combinations.
    /// This is the mechanism by which `frcmod` files modify a base force field.
    pub fn merge(&mut self, other: &ForceFieldParams) {
        // Merge atom types (override existing)
        for (name, at) in &other.atom_types {
            self.atom_types.insert(name.clone(), at.clone());
        }

        // Merge bond params (override matching, append new)
        for bp in &other.bond_params {
            let key = bond_key(&bp.type1, &bp.type2);
            if let Some(existing) = self
                .bond_params
                .iter_mut()
                .find(|p| bond_key(&p.type1, &p.type2) == key)
            {
                *existing = bp.clone();
            } else {
                self.bond_params.push(bp.clone());
            }
        }

        // Merge angle params (override matching, append new)
        for ap in &other.angle_params {
            let key = angle_key(&ap.type1, &ap.type2, &ap.type3);
            if let Some(existing) = self
                .angle_params
                .iter_mut()
                .find(|p| angle_key(&p.type1, &p.type2, &p.type3) == key)
            {
                *existing = ap.clone();
            } else {
                self.angle_params.push(ap.clone());
            }
        }

        // Dihedrals can have multiple entries for the same type combo (multi-term),
        // but frcmod replaces all terms for a given combo
        for dp in &other.dihedral_params {
            self.dihedral_params.push(dp.clone());
        }

        // Improper dihedrals: append
        for ip in &other.improper_params {
            self.improper_params.push(ip.clone());
        }

        // Merge LJ params (override existing)
        for (name, lj) in &other.lj_params {
            self.lj_params.insert(name.clone(), lj.clone());
        }

        // Merge equivalences: append
        for eq in &other.nb_equivalences {
            self.nb_equivalences.push(eq.clone());
        }
    }

    /// Look up bond parameters for two atom types (order-independent).
    pub fn find_bond(&self, t1: &str, t2: &str) -> Option<&BondParam> {
        let key = bond_key(t1, t2);
        self.bond_params
            .iter()
            .find(|p| bond_key(&p.type1, &p.type2) == key)
    }

    /// Look up angle parameters for three atom types.
    ///
    /// The central atom (`t2`) is fixed; `t1` and `t3` may be swapped.
    pub fn find_angle(&self, t1: &str, t2: &str, t3: &str) -> Option<&AngleParam> {
        let key = angle_key(t1, t2, t3);
        self.angle_params
            .iter()
            .find(|p| angle_key(&p.type1, &p.type2, &p.type3) == key)
    }

    /// Look up dihedral parameters with wildcard matching.
    ///
    /// Returns all matching dihedral parameter entries. Both forward and
    /// reverse atom-type orderings are considered.
    pub fn find_dihedrals(&self, t1: &str, t2: &str, t3: &str, t4: &str) -> Vec<&DihedralParam> {
        let mut results = Vec::new();
        for dp in &self.dihedral_params {
            if dihedral_matches(&dp.type1, &dp.type2, &dp.type3, &dp.type4, t1, t2, t3, t4) {
                results.push(dp);
            }
        }
        results
    }

    /// Look up improper dihedral parameters with wildcard matching.
    ///
    /// In AMBER convention, `t3` is the central atom type.
    pub fn find_improper(
        &self,
        t1: &str,
        t2: &str,
        t3: &str,
        t4: &str,
    ) -> Option<&ImproperParam> {
        self.improper_params.iter().find(|p| {
            improper_matches(&p.type1, &p.type2, &p.type3, &p.type4, t1, t2, t3, t4)
        })
    }

    /// Look up Lennard-Jones parameters for an atom type, following equivalences.
    ///
    /// Performs a direct lookup first; if no match is found, searches the
    /// non-bonded equivalence groups and returns parameters for the master type.
    pub fn find_lj(&self, atom_type: &str) -> Option<&LjParam> {
        // Direct lookup first
        if let Some(lj) = self.lj_params.get(atom_type) {
            return Some(lj);
        }
        // Check equivalences
        for eq in &self.nb_equivalences {
            if eq.types.iter().any(|t| t == atom_type) {
                // Use the first type in the equivalence list as the master
                if let Some(lj) = self.lj_params.get(&eq.types[0]) {
                    return Some(lj);
                }
            }
        }
        None
    }
}

/// Canonical key for bond lookup (alphabetically sorted).
fn bond_key(t1: &str, t2: &str) -> (String, String) {
    if t1 <= t2 {
        (t1.to_string(), t2.to_string())
    } else {
        (t2.to_string(), t1.to_string())
    }
}

/// Canonical key for angle lookup.
///
/// The central atom (`t2`) is fixed; `t1` and `t3` are sorted alphabetically
/// so that A-B-C and C-B-A produce the same key.
fn angle_key(t1: &str, t2: &str, t3: &str) -> (String, String, String) {
    if t1 <= t3 {
        (t1.to_string(), t2.to_string(), t3.to_string())
    } else {
        (t3.to_string(), t2.to_string(), t1.to_string())
    }
}

/// Check if a dihedral parameter matches, considering `"X"` wildcards and direction.
///
/// Both forward (1-2-3-4) and reverse (4-3-2-1) orderings are checked.
#[allow(clippy::too_many_arguments)]
fn dihedral_matches(
    p1: &str,
    p2: &str,
    p3: &str,
    p4: &str,
    t1: &str,
    t2: &str,
    t3: &str,
    t4: &str,
) -> bool {
    let fwd = (p1 == t1 || p1 == "X") && (p2 == t2) && (p3 == t3) && (p4 == t4 || p4 == "X");
    let rev = (p1 == t4 || p1 == "X") && (p2 == t3) && (p3 == t2) && (p4 == t1 || p4 == "X");
    fwd || rev
}

/// Check if an improper parameter matches.
///
/// In AMBER convention, `p3`/`t3` is always the central atom and must match
/// exactly. The other three positions (`t1`, `t2`, `t4`) are matched against
/// `p1`, `p2`, `p4` in any order, with `"X"` acting as a wildcard.
#[allow(clippy::too_many_arguments)]
fn improper_matches(
    p1: &str,
    p2: &str,
    p3: &str,
    p4: &str,
    t1: &str,
    t2: &str,
    t3: &str,
    t4: &str,
) -> bool {
    // The central atom (p3/t3) must match exactly
    if p3 != t3 {
        return false;
    }
    // The other three can match in any combination with wildcard support
    let mut targets = vec![t1, t2, t4];
    let patterns = [p1, p2, p4];
    for pat in &patterns {
        if *pat == "X" {
            continue;
        }
        if let Some(pos) = targets.iter().position(|t| *t == *pat) {
            targets.remove(pos);
        } else {
            return false;
        }
    }
    true
}
