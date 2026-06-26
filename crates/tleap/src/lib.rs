//! System building library (tleap replacement).
//!
//! Provides a programmatic API for building molecular systems:
//! - Load structures from PDB, mol2, and mmCIF files
//! - Apply AMBER force field parameters (ff19SB, lipid21, OPC, GAFF2)
//! - Solvate in rectangular water boxes
//! - Place counterions
//! - Write AMBER prmtop/inpcrd files

pub mod builder;
pub mod ions;
pub mod parameterize;
pub mod solvate;
pub mod system;

pub use builder::SystemBuilder;
pub use ions::{IonConfig, IonCount};
pub use solvate::SolvateConfig;
pub use system::System;
