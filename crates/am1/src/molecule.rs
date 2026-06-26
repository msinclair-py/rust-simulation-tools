//! AM1 molecular representation.

use crate::params::{self, Am1Element};

/// An atom in the AM1 calculation.
#[derive(Debug, Clone)]
pub struct Am1Atom {
    /// Atomic number.
    pub atomic_number: u8,
    /// Cartesian position in Angstroms.
    pub position: [f64; 3],
    /// AM1 parameters for this element.
    pub params: &'static Am1Element,
    /// Index of first basis function for this atom.
    pub basis_offset: usize,
}

/// A molecule ready for AM1 calculation.
#[derive(Debug, Clone)]
pub struct Am1Molecule {
    /// Atoms with their parameters.
    pub atoms: Vec<Am1Atom>,
    /// Total number of basis functions.
    pub n_basis: usize,
    /// Total number of electrons.
    pub n_electrons: usize,
    /// Molecular charge.
    pub charge: i32,
}

impl Am1Molecule {
    /// Create a new AM1 molecule.
    ///
    /// # Arguments
    /// * `atomic_numbers` - Atomic numbers for each atom.
    /// * `coords` - Cartesian coordinates in Angstroms, one [x,y,z] per atom.
    /// * `charge` - Net molecular charge.
    pub fn new(
        atomic_numbers: &[u8],
        coords: &[[f64; 3]],
        charge: i32,
    ) -> Result<Self, String> {
        if atomic_numbers.len() != coords.len() {
            return Err(format!(
                "Number of atomic numbers ({}) does not match coordinates ({})",
                atomic_numbers.len(),
                coords.len()
            ));
        }

        let mut atoms = Vec::with_capacity(atomic_numbers.len());
        let mut n_basis = 0;
        let mut total_electrons: i32 = 0;

        for (i, (&z, &pos)) in atomic_numbers.iter().zip(coords.iter()).enumerate() {
            let p = params::get_params(z).ok_or_else(|| {
                format!("No AM1 parameters for element Z={} (atom {})", z, i)
            })?;

            atoms.push(Am1Atom {
                atomic_number: z,
                position: pos,
                params: p,
                basis_offset: n_basis,
            });

            n_basis += p.n_orbitals as usize;
            total_electrons += p.core_charge as i32;
        }

        total_electrons -= charge;
        if total_electrons < 0 {
            return Err(format!(
                "Charge {} results in negative electron count",
                charge
            ));
        }
        if total_electrons % 2 != 0 {
            return Err(format!(
                "AM1 requires even electron count for closed-shell calculation, got {}",
                total_electrons
            ));
        }

        Ok(Self {
            atoms,
            n_basis,
            n_electrons: total_electrons as usize,
            charge,
        })
    }

    /// Number of occupied orbitals (closed-shell).
    pub fn n_occupied(&self) -> usize {
        self.n_electrons / 2
    }

    /// Distance between atoms i and j in Angstroms.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let a = &self.atoms[i].position;
        let b = &self.atoms[j].position;
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Direction cosines from atom i to atom j.
    pub fn direction_cosines(&self, i: usize, j: usize) -> [f64; 3] {
        let a = &self.atoms[i].position;
        let b = &self.atoms[j].position;
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let dz = b[2] - a[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r < 1.0e-10 {
            [0.0, 0.0, 1.0]
        } else {
            [dx / r, dy / r, dz / r]
        }
    }
}
