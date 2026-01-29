//! Convenience keyword data tables for protein, water, backbone, etc.

/// Standard amino acid residue names (3-letter codes).
pub const PROTEIN_RESIDUES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HID", "HIE", "HIP",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    // N-terminal variants
    "NALA", "NARG", "NASN", "NASP", "NCYS", "NCYX", "NGLN", "NGLU", "NGLY", "NHIS", "NHID", "NHIE",
    "NHIP", "NILE", "NLEU", "NLYS", "NMET", "NPHE", "NPRO", "NSER", "NTHR", "NTRP", "NTYR", "NVAL",
    // C-terminal variants
    "CALA", "CARG", "CASN", "CASP", "CCYS", "CCYX", "CGLN", "CGLU", "CGLY", "CHIS", "CHID", "CHIE",
    "CHIP", "CILE", "CLEU", "CLYS", "CMET", "CPHE", "CPRO", "CSER", "CTHR", "CTRP", "CTYR", "CVAL",
    // ACE/NME caps
    "ACE", "NME", "NHE",
];

/// Water residue names.
pub const WATER_RESIDUES: &[&str] = &[
    "WAT", "HOH", "TIP3", "T3P", "SPC", "TIP4", "TP4", "TIP5", "T5P",
];

/// Backbone atom names.
pub const BACKBONE_ATOMS: &[&str] = &["N", "CA", "C", "O", "H", "HA"];

/// Check if a residue name is a protein residue.
pub fn is_protein_residue(name: &str) -> bool {
    let trimmed = name.trim();
    PROTEIN_RESIDUES
        .iter()
        .any(|&r| r.eq_ignore_ascii_case(trimmed))
}

/// Check if a residue name is a water residue.
pub fn is_water_residue(name: &str) -> bool {
    let trimmed = name.trim();
    WATER_RESIDUES
        .iter()
        .any(|&r| r.eq_ignore_ascii_case(trimmed))
}

/// Check if an atom name is a backbone atom.
pub fn is_backbone_atom(name: &str) -> bool {
    let trimmed = name.trim();
    BACKBONE_ATOMS
        .iter()
        .any(|&a| a.eq_ignore_ascii_case(trimmed))
}
