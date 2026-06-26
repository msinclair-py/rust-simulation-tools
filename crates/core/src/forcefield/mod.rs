//! Force field data structures and parsers.
//!
//! This module provides the core data structures for representing molecular mechanics
//! force field parameters (bonds, angles, dihedrals, Lennard-Jones, etc.) along with
//! parsers for AMBER-format parameter files.
//!
//! # Sub-modules
//!
//! - [`atom_types`] - Atom type definitions and element lookup utilities
//! - [`parameters`] - Force field parameter structures (`ForceFieldParams`, `BondParam`, etc.)
//! - [`parm_dat`] - Parser for AMBER parm .dat files (parm19.dat, gaff2.dat)
//! - [`frcmod`] - Parser for AMBER frcmod files
//! - [`residue_lib`] - Parser for AMBER .lib/.off residue template files
//! - [`data`] - Embedded force field data files
//! - [`loader`] - Unified force field loading

pub mod atom_types;
pub mod data;
pub mod frcmod;
pub mod loader;
pub mod parameters;
pub mod parm_dat;
pub mod residue_lib;

// Re-export key types for convenience
pub use atom_types::element_from_type;
pub use loader::LoadedForceField;
pub use parameters::{
    AngleParam, AtomType, BondParam, DihedralParam, DihedralTerm, ForceFieldParams,
    ImproperParam, LjParam, NbEquiv,
};
pub use residue_lib::{ResidueLibrary, ResidueTemplate};
