//! Core library for rust-simulation-tools.
//!
//! Pure Rust implementations with no Python dependencies.
//! Provides AMBER file parsers, trajectory readers, SASA calculations,
//! alignment, unwrapping, and fingerprinting engines.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::module_inception)]
#![allow(clippy::type_complexity)]

pub mod amber;
pub mod fingerprint;
pub mod forcefield;
pub mod graph;
pub mod kabsch;
pub mod mmcif;
pub mod mol2;
pub mod pdb;
pub mod pdb_writer;
pub mod sasa;
pub mod sdf;
pub mod selection;
pub mod trajectory;
pub mod util;
pub mod wrapping;
