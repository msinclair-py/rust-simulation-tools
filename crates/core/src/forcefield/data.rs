//! Embedded force field data files.
//!
//! All standard AMBER force field files are compiled into the binary
//! using `include_str!()` for zero-dependency deployment.

// Parameter files
pub const PARM19_DAT: &str = include_str!("../../data/parm/parm19.dat");
pub const FRCMOD_FF19SB: &str = include_str!("../../data/parm/frcmod.ff19SB");
pub const GAFF2_DAT: &str = include_str!("../../data/parm/gaff2.dat");
pub const LIPID21_DAT: &str = include_str!("../../data/parm/lipid21.dat");
pub const FRCMOD_OPC: &str = include_str!("../../data/parm/frcmod.opc");
pub const FRCMOD_IONS_OPC: &str = include_str!("../../data/parm/frcmod.ionslm_126_opc");

// Library files
pub const AMINO19_LIB: &str = include_str!("../../data/lib/amino19.lib");
pub const AMINOCT12_LIB: &str = include_str!("../../data/lib/aminoct12.lib");
pub const AMINONT12_LIB: &str = include_str!("../../data/lib/aminont12.lib");
pub const LIPID21_LIB: &str = include_str!("../../data/lib/lipid21.lib");
pub const SOLVENTS_LIB: &str = include_str!("../../data/lib/solvents.lib");
pub const ATOMIC_IONS_LIB: &str = include_str!("../../data/lib/atomic_ions.lib");
