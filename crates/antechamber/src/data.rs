//! Embedded antechamber data files.
//!
//! All data files from AmberTools dat/antechamber/ are compiled into the binary
//! using `include_str!()` for zero-dependency deployment.

/// GAFF2 atom type definition rules.
pub const ATOMTYPE_GFF2: &str = include_str!("../data/ATOMTYPE_GFF2.DEF");

/// BCC atom type definition rules (for AM1-BCC correction lookup).
pub const ATOMTYPE_BCC: &str = include_str!("../data/ATOMTYPE_BCC.DEF");

/// AM1-BCC bond charge correction parameters.
pub const BCCPARM: &str = include_str!("../data/BCCPARM.DAT");

/// Gasteiger electronegativity parameters.
pub const GASPARM: &str = include_str!("../data/GASPARM.DAT");

/// VdW radii.
pub const RADIUS: &str = include_str!("../data/RADIUS.DAT");
