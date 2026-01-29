//! AMBER file format parsers.
//!
//! Provides native Rust readers for AMBER topology and coordinate files:
//! - PRMTOP: Topology files with force field parameters
//! - INPCRD/RST7: Coordinate/restart files

pub mod inpcrd;
pub mod prmtop;
