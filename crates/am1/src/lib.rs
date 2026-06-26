#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

//! Pure Rust AM1 semi-empirical quantum mechanics.
//!
//! Computes single-point AM1 energies and Mulliken charges for closed-shell
//! molecules with elements H, C, N, O, F, P, S, Cl, Br, I.
//!
//! # Example
//!
//! ```
//! use rst_am1::{compute_am1_charges, compute_am1};
//!
//! // Water molecule
//! let atomic_numbers = vec![8, 1, 1];
//! let coords = vec![
//!     [0.0, 0.0, 0.0],
//!     [0.0, 0.757, 0.587],
//!     [0.0, -0.757, 0.587],
//! ];
//!
//! let charges = compute_am1_charges(&atomic_numbers, &coords, 0).unwrap();
//! ```

pub mod charges;
pub mod fock;
pub mod hamiltonian;
pub mod molecule;
pub mod nuclear;
pub mod overlap;
pub mod params;
pub mod scf;
pub mod two_electron;

use molecule::Am1Molecule;
use scf::{Am1Result, ScfConfig};

/// Compute AM1 Mulliken charges for a molecule.
///
/// This is the main entry point for charge calculations.
///
/// # Arguments
/// * `atomic_numbers` - Atomic numbers for each atom.
/// * `coords` - Cartesian coordinates in Angstroms, one [x,y,z] per atom.
/// * `charge` - Net molecular charge.
///
/// # Returns
/// Vector of Mulliken partial charges per atom.
pub fn compute_am1_charges(
    atomic_numbers: &[u8],
    coords: &[[f64; 3]],
    charge: i32,
) -> Result<Vec<f64>, String> {
    let result = compute_am1(atomic_numbers, coords, charge, None)?;
    Ok(result.charges)
}

/// Run a full AM1 calculation.
///
/// Returns detailed results including energy, charges, orbital energies,
/// and convergence information.
///
/// # Arguments
/// * `atomic_numbers` - Atomic numbers for each atom.
/// * `coords` - Cartesian coordinates in Angstroms, one [x,y,z] per atom.
/// * `charge` - Net molecular charge.
/// * `config` - Optional SCF configuration. Uses defaults if None.
pub fn compute_am1(
    atomic_numbers: &[u8],
    coords: &[[f64; 3]],
    charge: i32,
    config: Option<ScfConfig>,
) -> Result<Am1Result, String> {
    let mol = Am1Molecule::new(atomic_numbers, coords, charge)?;
    let cfg = config.unwrap_or_default();
    scf::run_scf(&mol, &cfg)
}
