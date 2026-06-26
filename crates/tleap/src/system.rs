//! Core data structures for the molecular system builder.
//!
//! This module defines the [`System`] struct, the central intermediate representation
//! used throughout the tleap pipeline. A `System` is populated during structure loading
//! (PDB, mol2, mmCIF), modified during solvation and ion placement, and ultimately
//! serialized to AMBER prmtop/inpcrd files.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Atom
// ---------------------------------------------------------------------------

/// A single atom in the molecular system.
#[derive(Debug, Clone)]
pub struct Atom {
    /// Atom name (e.g. "CA", "N", "H1").
    pub name: String,
    /// AMBER atom type (e.g. "CT", "N", "HC").
    pub atom_type: String,
    /// Element symbol (e.g. "C", "N", "H").
    pub element: String,
    /// Partial charge in elementary charge units.
    pub charge: f64,
    /// Atomic mass in atomic mass units (amu).
    pub mass: f64,
    /// Atomic number (1=H, 6=C, 7=N, 8=O, etc.).
    pub atomic_number: i32,
    /// Cartesian coordinates in Angstroms `[x, y, z]`.
    pub position: [f64; 3],
    /// Index of the residue this atom belongs to.
    pub residue_idx: usize,
    /// Born radius in Angstroms, used for Generalized Born calculations.
    pub born_radius: f64,
    /// Generalized Born screening parameter.
    pub screen: f64,
}

// ---------------------------------------------------------------------------
// Bond
// ---------------------------------------------------------------------------

/// A covalent bond between two atoms, identified by 0-based global atom indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bond {
    /// 0-based index of the first atom.
    pub atom1: usize,
    /// 0-based index of the second atom.
    pub atom2: usize,
}

// ---------------------------------------------------------------------------
// Residue
// ---------------------------------------------------------------------------

/// A residue (amino acid, water molecule, ion, ligand fragment, etc.).
#[derive(Debug, Clone)]
pub struct Residue {
    /// Residue name (e.g. "ALA", "WAT", "Na+").
    pub name: String,
    /// Chain identifier (single character, e.g. 'A').
    pub chain_id: char,
    /// Residue sequence number (from PDB numbering).
    pub seq_num: i32,
    /// Range of atom indices `[start, end)` in the system's atom list.
    pub atom_range: std::ops::Range<usize>,
}

// ---------------------------------------------------------------------------
// Molecule
// ---------------------------------------------------------------------------

/// A molecule (connected component in the bond graph).
///
/// Used for the `ATOMS_PER_MOLECULE` section in prmtop output and for
/// solvation bookkeeping.
#[derive(Debug, Clone)]
pub struct Molecule {
    /// 0-based indices of atoms belonging to this molecule, sorted ascending.
    pub atom_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Union-Find (private helper)
// ---------------------------------------------------------------------------

/// Disjoint-set / union-find data structure for connected component detection.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Create a new union-find structure with `n` disjoint elements.
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the representative of the set containing `x`, with path compression.
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Merge the sets containing `x` and `y` (union by rank).
    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// The complete molecular system being built.
///
/// This is the central data structure for the tleap pipeline. Atoms, bonds, and
/// residues are stored in flat vectors with index-based cross-references. The
/// optional box dimensions describe the periodic unit cell for explicit solvent
/// simulations.
#[derive(Debug, Clone)]
pub struct System {
    /// All atoms in the system, in insertion order.
    pub atoms: Vec<Atom>,
    /// All covalent bonds.
    pub bonds: Vec<Bond>,
    /// All residues, in sequence order.
    pub residues: Vec<Residue>,
    /// Periodic box dimensions `[x, y, z]` in Angstroms, or `None` for vacuum.
    pub box_dimensions: Option<[f64; 3]>,
    /// Periodic box angles `[alpha, beta, gamma]` in degrees, or `None`.
    pub box_angles: Option<[f64; 3]>,
}

impl System {
    /// Create an empty system with no atoms, bonds, or residues.
    pub fn new() -> Self {
        Self {
            atoms: Vec::new(),
            bonds: Vec::new(),
            residues: Vec::new(),
            box_dimensions: None,
            box_angles: None,
        }
    }

    /// Total number of atoms in the system.
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Total number of residues in the system.
    pub fn n_residues(&self) -> usize {
        self.residues.len()
    }

    /// Total charge of the system (sum of all atom partial charges).
    pub fn total_charge(&self) -> f64 {
        self.atoms.iter().map(|a| a.charge).sum()
    }

    /// Add a residue together with its atoms and internal bonds.
    ///
    /// `atoms` are appended as-is (positions are stored verbatim). Each atom's
    /// `residue_idx` will be overwritten to point at the newly created residue.
    ///
    /// `internal_bonds` contains `(i, j)` pairs that are 0-based *relative* to
    /// the supplied `atoms` slice. They will be offset to global indices before
    /// being inserted into the system bond list.
    ///
    /// Returns the index of the newly created residue.
    pub fn add_residue(
        &mut self,
        name: &str,
        chain_id: char,
        seq_num: i32,
        atoms: Vec<Atom>,
        internal_bonds: Vec<(usize, usize)>,
    ) -> usize {
        let residue_idx = self.residues.len();
        let base = self.atoms.len();
        let n_new = atoms.len();

        // Append atoms, fixing up residue_idx.
        for mut atom in atoms {
            atom.residue_idx = residue_idx;
            self.atoms.push(atom);
        }

        // Create residue record.
        self.residues.push(Residue {
            name: name.to_owned(),
            chain_id,
            seq_num,
            atom_range: base..base + n_new,
        });

        // Offset internal bonds to global indices and append.
        for (a, b) in internal_bonds {
            self.bonds.push(Bond {
                atom1: base + a,
                atom2: base + b,
            });
        }

        residue_idx
    }

    /// Add a bond between two global atom indices.
    ///
    /// No duplicate checking is performed; the caller is responsible for
    /// ensuring the bond is not already present if uniqueness matters.
    pub fn add_bond(&mut self, atom1: usize, atom2: usize) {
        self.bonds.push(Bond { atom1, atom2 });
    }

    /// Remove a set of atoms from the system by their 0-based indices.
    ///
    /// `indices` **must** be sorted in ascending order. All bonds that reference
    /// any removed atom are also removed. Residue `atom_range` values are
    /// recomputed, and residues that become empty are dropped entirely. Remaining
    /// atom `residue_idx` fields are updated to reflect the new residue ordering.
    pub fn remove_atoms(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }

        // Build a set for O(1) membership testing.
        let remove_set: std::collections::HashSet<usize> = indices.iter().copied().collect();

        // Build old-to-new index mapping for atoms. Removed atoms map to usize::MAX.
        let n_old = self.atoms.len();
        let mut old_to_new: Vec<usize> = vec![usize::MAX; n_old];
        let mut new_idx: usize = 0;
        for (i, slot) in old_to_new.iter_mut().enumerate() {
            if !remove_set.contains(&i) {
                *slot = new_idx;
                new_idx += 1;
            }
        }

        // Rebuild bonds, discarding any that reference a removed atom.
        let new_bonds: Vec<Bond> = self
            .bonds
            .iter()
            .filter_map(|b| {
                let a1 = old_to_new[b.atom1];
                let a2 = old_to_new[b.atom2];
                if a1 == usize::MAX || a2 == usize::MAX {
                    None
                } else {
                    Some(Bond {
                        atom1: a1,
                        atom2: a2,
                    })
                }
            })
            .collect();

        // Rebuild residues: compute new atom ranges and drop empty residues.
        // For each old residue, count surviving atoms and record the new range.
        let mut new_residues: Vec<Residue> = Vec::with_capacity(self.residues.len());
        let mut residue_old_to_new: Vec<usize> = vec![usize::MAX; self.residues.len()];

        for (old_res_idx, res) in self.residues.iter().enumerate() {
            let _new_start = if res.atom_range.start < n_old {
                old_to_new[res.atom_range.start]
            } else {
                // Degenerate: atom_range.start is past all atoms.
                new_idx
            };

            // Count how many atoms survive in this residue.
            let surviving_count = res
                .atom_range
                .clone()
                .filter(|&i| !remove_set.contains(&i))
                .count();

            if surviving_count == 0 {
                continue;
            }

            // The surviving atoms within this residue are contiguous in the new
            // ordering because we only remove atoms (never reorder). The new
            // start is the mapped index of the first surviving atom.
            let first_surviving = res
                .atom_range
                .clone()
                .find(|i| !remove_set.contains(i))
                .expect("surviving_count > 0 guarantees at least one atom");
            let mapped_start = old_to_new[first_surviving];

            let new_res_idx = new_residues.len();
            residue_old_to_new[old_res_idx] = new_res_idx;

            new_residues.push(Residue {
                name: res.name.clone(),
                chain_id: res.chain_id,
                seq_num: res.seq_num,
                atom_range: mapped_start..mapped_start + surviving_count,
            });
        }

        // Remove atoms (iterate in reverse so indices stay valid).
        // Alternatively, rebuild the vector in one pass.
        let new_atoms: Vec<Atom> = self
            .atoms
            .drain(..)
            .enumerate()
            .filter_map(|(i, mut atom)| {
                if remove_set.contains(&i) {
                    None
                } else {
                    // Update residue_idx to the new residue index.
                    atom.residue_idx = residue_old_to_new[atom.residue_idx];
                    Some(atom)
                }
            })
            .collect();

        self.atoms = new_atoms;
        self.bonds = new_bonds;
        self.residues = new_residues;
    }

    /// Identify connected components (molecules) from the bond graph.
    ///
    /// Uses a union-find algorithm for efficient component detection. Atoms
    /// that are not involved in any bond are returned as singleton molecules.
    ///
    /// The returned [`Molecule`] structs have their `atom_indices` sorted in
    /// ascending order. The list of molecules is also sorted by their first
    /// (smallest) atom index.
    pub fn find_molecules(&self) -> Vec<Molecule> {
        let n = self.atoms.len();
        if n == 0 {
            return Vec::new();
        }

        let mut uf = UnionFind::new(n);
        for bond in &self.bonds {
            uf.union(bond.atom1, bond.atom2);
        }

        // Group atom indices by their root representative.
        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = uf.find(i);
            components.entry(root).or_default().push(i);
        }

        // Build molecule list, sorted by first atom index.
        let mut molecules: Vec<Molecule> = components
            .into_values()
            .map(|mut indices| {
                indices.sort_unstable();
                Molecule {
                    atom_indices: indices,
                }
            })
            .collect();

        molecules.sort_by_key(|m| m.atom_indices[0]);
        molecules
    }

    /// Center the system so that the geometric centroid of all atoms is at the
    /// origin `[0, 0, 0]`.
    ///
    /// This is a no-op on an empty system.
    pub fn center_at_origin(&mut self) {
        let n = self.atoms.len();
        if n == 0 {
            return;
        }

        let inv_n = 1.0 / n as f64;
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let mut cz = 0.0_f64;

        for atom in &self.atoms {
            cx += atom.position[0];
            cy += atom.position[1];
            cz += atom.position[2];
        }

        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;

        for atom in &mut self.atoms {
            atom.position[0] -= cx;
            atom.position[1] -= cy;
            atom.position[2] -= cz;
        }
    }

    /// Compute the axis-aligned bounding box of all atom positions.
    ///
    /// Returns `([min_x, min_y, min_z], [max_x, max_y, max_z])`.
    ///
    /// # Panics
    ///
    /// Panics if the system contains no atoms.
    pub fn bounding_box(&self) -> ([f64; 3], [f64; 3]) {
        assert!(!self.atoms.is_empty(), "bounding_box called on empty system");

        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];

        for atom in &self.atoms {
            for i in 0..3 {
                if atom.position[i] < min[i] {
                    min[i] = atom.position[i];
                }
                if atom.position[i] > max[i] {
                    max[i] = atom.position[i];
                }
            }
        }

        (min, max)
    }

    /// Translate all atom positions by the given displacement vector.
    pub fn translate(&mut self, delta: [f64; 3]) {
        for atom in &mut self.atoms {
            atom.position[0] += delta[0];
            atom.position[1] += delta[1];
            atom.position[2] += delta[2];
        }
    }
}

impl Default for System {
    fn default() -> Self {
        Self::new()
    }
}
