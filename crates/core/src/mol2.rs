//! TRIPOS mol2 file parser.
//!
//! Parses mol2 files produced by antechamber (GAFF2 atom types + partial charges).
//! Extracts atom coordinates, bond connectivity, and substructure information
//! from the standard TRIPOS mol2 section records.

use std::fs::read_to_string;
use std::path::Path;

// ============================================================================
// Data Structures
// ============================================================================

/// A single atom from a mol2 file.
#[derive(Debug, Clone)]
pub struct Mol2Atom {
    /// 1-based atom ID from the file.
    pub id: usize,
    /// Atom name (e.g. "C1", "N2", "H3").
    pub name: String,
    /// Cartesian coordinates in Angstroms.
    pub position: [f64; 3],
    /// SYBYL/GAFF2 atom type (e.g. "c3", "n", "oh").
    pub atom_type: String,
    /// Residue/substructure ID (1-based in file, stored as-is).
    pub residue_id: usize,
    /// Residue/substructure name.
    pub residue_name: String,
    /// Partial charge (elementary charge units).
    pub charge: f64,
}

/// A bond from a mol2 file.
#[derive(Debug, Clone)]
pub struct Mol2Bond {
    /// 0-based index of first atom.
    pub atom1: usize,
    /// 0-based index of second atom.
    pub atom2: usize,
    /// Bond type string (e.g. "1", "2", "3", "ar", "am").
    pub bond_type: String,
}

/// A substructure entry from a mol2 file.
#[derive(Debug, Clone)]
pub struct Mol2Substructure {
    /// Substructure ID (1-based).
    pub id: usize,
    /// Substructure name.
    pub name: String,
    /// First atom in substructure (1-based in file, stored 0-based).
    pub root_atom: usize,
}

/// A complete mol2 molecule.
#[derive(Debug, Clone)]
pub struct Mol2Molecule {
    /// Molecule name.
    pub name: String,
    /// All atoms.
    pub atoms: Vec<Mol2Atom>,
    /// All bonds.
    pub bonds: Vec<Mol2Bond>,
    /// Substructures.
    pub substructures: Vec<Mol2Substructure>,
}

// ============================================================================
// Section Tags
// ============================================================================

/// Record type indicator prefix used in TRIPOS mol2 files.
const TRIPOS_PREFIX: &str = "@<TRIPOS>";

/// Known section tags we parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Molecule,
    Atom,
    Bond,
    Substructure,
    Unknown,
}

/// Identify the section from a record type indicator line.
fn identify_section(line: &str) -> Option<Section> {
    let trimmed = line.trim();
    if !trimmed.starts_with(TRIPOS_PREFIX) {
        return None;
    }
    let tag = &trimmed[TRIPOS_PREFIX.len()..];
    Some(match tag {
        "MOLECULE" => Section::Molecule,
        "ATOM" => Section::Atom,
        "BOND" => Section::Bond,
        "SUBSTRUCTURE" => Section::Substructure,
        _ => Section::Unknown,
    })
}

// ============================================================================
// Section Parsers
// ============================================================================

/// Parse the MOLECULE section.
///
/// Expected layout (lines after the `@<TRIPOS>MOLECULE` tag):
///   Line 0: molecule name
///   Line 1: num_atoms num_bonds [num_subst [num_feat [num_sets]]]
///   Line 2+: molecule type, charge type, etc. (ignored)
///
/// Returns `(name, num_atoms, num_bonds)`.
fn parse_molecule_section(lines: &[&str]) -> Result<(String, usize, usize), String> {
    if lines.is_empty() {
        return Err("MOLECULE section is empty: expected molecule name".into());
    }

    let name = lines[0].trim().to_string();

    if lines.len() < 2 {
        return Err("MOLECULE section too short: expected atom/bond counts line".into());
    }

    let counts: Vec<&str> = lines[1].split_whitespace().collect();
    if counts.len() < 2 {
        return Err(format!(
            "MOLECULE counts line has {} fields, expected at least 2 (num_atoms num_bonds)",
            counts.len()
        ));
    }

    let num_atoms: usize = counts[0]
        .parse()
        .map_err(|e| format!("Failed to parse atom count '{}': {}", counts[0], e))?;
    let num_bonds: usize = counts[1]
        .parse()
        .map_err(|e| format!("Failed to parse bond count '{}': {}", counts[1], e))?;

    Ok((name, num_atoms, num_bonds))
}

/// Parse a single ATOM line.
///
/// Format: atom_id name x y z atom_type resid resname charge
///
/// Fields are whitespace-separated. The charge field is optional; if missing,
/// it defaults to 0.0.
fn parse_atom_line(line: &str, line_num: usize) -> Result<Mol2Atom, String> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 6 {
        return Err(format!(
            "ATOM line {} has {} fields, expected at least 6: '{}'",
            line_num,
            fields.len(),
            line.trim()
        ));
    }

    let id: usize = fields[0]
        .parse()
        .map_err(|e| format!("ATOM line {}: bad atom ID '{}': {}", line_num, fields[0], e))?;

    let name = fields[1].to_string();

    let x: f64 = fields[2]
        .parse()
        .map_err(|e| format!("ATOM line {}: bad x coordinate '{}': {}", line_num, fields[2], e))?;
    let y: f64 = fields[3]
        .parse()
        .map_err(|e| format!("ATOM line {}: bad y coordinate '{}': {}", line_num, fields[3], e))?;
    let z: f64 = fields[4]
        .parse()
        .map_err(|e| format!("ATOM line {}: bad z coordinate '{}': {}", line_num, fields[4], e))?;

    let atom_type = fields[5].to_string();

    // Residue ID and name are optional in some minimal mol2 files.
    let residue_id: usize = if fields.len() > 6 {
        fields[6]
            .parse()
            .map_err(|e| format!("ATOM line {}: bad residue ID '{}': {}", line_num, fields[6], e))?
    } else {
        1
    };

    let residue_name = if fields.len() > 7 {
        fields[7].to_string()
    } else {
        "UNK".to_string()
    };

    let charge: f64 = if fields.len() > 8 {
        fields[8]
            .parse()
            .map_err(|e| format!("ATOM line {}: bad charge '{}': {}", line_num, fields[8], e))?
    } else {
        0.0
    };

    Ok(Mol2Atom {
        id,
        name,
        position: [x, y, z],
        atom_type,
        residue_id,
        residue_name,
        charge,
    })
}

/// Parse a single BOND line.
///
/// Format: bond_id atom1_id atom2_id bond_type
///
/// Atom IDs are 1-based in the file and converted to 0-based for storage.
fn parse_bond_line(line: &str, line_num: usize) -> Result<Mol2Bond, String> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 4 {
        return Err(format!(
            "BOND line {} has {} fields, expected at least 4: '{}'",
            line_num,
            fields.len(),
            line.trim()
        ));
    }

    // fields[0] is bond_id (unused)

    let atom1_id: usize = fields[1]
        .parse()
        .map_err(|e| format!("BOND line {}: bad atom1 ID '{}': {}", line_num, fields[1], e))?;
    let atom2_id: usize = fields[2]
        .parse()
        .map_err(|e| format!("BOND line {}: bad atom2 ID '{}': {}", line_num, fields[2], e))?;

    if atom1_id == 0 {
        return Err(format!(
            "BOND line {}: atom1 ID is 0 (expected 1-based)",
            line_num
        ));
    }
    if atom2_id == 0 {
        return Err(format!(
            "BOND line {}: atom2 ID is 0 (expected 1-based)",
            line_num
        ));
    }

    let bond_type = fields[3].to_string();

    Ok(Mol2Bond {
        atom1: atom1_id - 1,
        atom2: atom2_id - 1,
        bond_type,
    })
}

/// Parse a single SUBSTRUCTURE line.
///
/// Format: subst_id subst_name root_atom [subst_type [dict_type [chain
///         [sub_type [inter_bonds [status [comment]]]]]]]]
///
/// The root_atom is 1-based in the file and converted to 0-based for storage.
fn parse_substructure_line(line: &str, line_num: usize) -> Result<Mol2Substructure, String> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 3 {
        return Err(format!(
            "SUBSTRUCTURE line {} has {} fields, expected at least 3: '{}'",
            line_num,
            fields.len(),
            line.trim()
        ));
    }

    let id: usize = fields[0].parse().map_err(|e| {
        format!(
            "SUBSTRUCTURE line {}: bad substructure ID '{}': {}",
            line_num, fields[0], e
        )
    })?;

    let name = fields[1].to_string();

    let root_atom_id: usize = fields[2].parse().map_err(|e| {
        format!(
            "SUBSTRUCTURE line {}: bad root atom '{}': {}",
            line_num, fields[2], e
        )
    })?;

    if root_atom_id == 0 {
        return Err(format!(
            "SUBSTRUCTURE line {}: root atom is 0 (expected 1-based)",
            line_num
        ));
    }

    Ok(Mol2Substructure {
        id,
        name,
        root_atom: root_atom_id - 1,
    })
}

// ============================================================================
// Top-Level Parsers
// ============================================================================

/// Parse a mol2 file from its text content.
///
/// Reads the first molecule from the mol2 content, extracting atom coordinates,
/// bond connectivity, and substructure information. If the content contains
/// multiple molecules (separated by `@<TRIPOS>MOLECULE`), only the first one
/// is parsed.
///
/// # Arguments
/// * `content` - The full text content of a mol2 file.
///
/// # Returns
/// * `Ok(Mol2Molecule)` - The parsed molecule.
/// * `Err(String)` - A descriptive error message if parsing fails.
pub fn parse_mol2(content: &str) -> Result<Mol2Molecule, String> {
    // Collect all lines with their original 1-based line numbers for error messages.
    let all_lines: Vec<&str> = content.lines().collect();

    if all_lines.is_empty() {
        return Err("mol2 content is empty".into());
    }

    // Split content into sections. Each section is identified by its
    // @<TRIPOS>TAG header. We collect the body lines that follow each header
    // until the next header (or end of file).
    struct SectionBlock<'a> {
        section: Section,
        lines: Vec<&'a str>,
        first_line_num: usize,
    }

    let mut sections: Vec<SectionBlock> = Vec::new();
    let mut current: Option<SectionBlock> = None;
    let mut seen_first_molecule = false;

    for (idx, &line) in all_lines.iter().enumerate() {
        let line_num = idx + 1;

        if let Some(section) = identify_section(line) {
            // If we encounter a second MOLECULE tag, stop: we only parse the first molecule.
            if section == Section::Molecule {
                if seen_first_molecule {
                    // Save whatever section was in progress and break.
                    if let Some(block) = current.take() {
                        sections.push(block);
                    }
                    break;
                }
                seen_first_molecule = true;
            }

            // Store the previous section block.
            if let Some(block) = current.take() {
                sections.push(block);
            }

            current = Some(SectionBlock {
                section,
                lines: Vec::new(),
                first_line_num: line_num + 1, // body starts on the next line
            });
        } else if let Some(ref mut block) = current {
            // Accumulate body lines for the current section.
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                block.lines.push(line);
            }
        }
    }

    // Don't forget the last section in progress.
    if let Some(block) = current.take() {
        sections.push(block);
    }

    // We must have at least a MOLECULE section.
    if !seen_first_molecule {
        return Err("No @<TRIPOS>MOLECULE section found in mol2 content".into());
    }

    // Parse the MOLECULE section to get name and expected counts.
    let molecule_block = sections
        .iter()
        .find(|b| b.section == Section::Molecule)
        .ok_or("No @<TRIPOS>MOLECULE section found")?;

    let (name, expected_atoms, expected_bonds) = parse_molecule_section(&molecule_block.lines)?;

    // Parse ATOM section.
    let atoms: Vec<Mol2Atom> = if let Some(block) = sections.iter().find(|b| b.section == Section::Atom) {
        let mut atoms = Vec::with_capacity(expected_atoms);
        for (i, &line) in block.lines.iter().enumerate() {
            let line_num = block.first_line_num + i;
            atoms.push(parse_atom_line(line, line_num)?);
        }
        atoms
    } else {
        Vec::new()
    };

    if atoms.len() != expected_atoms {
        return Err(format!(
            "MOLECULE header declared {} atoms but ATOM section contains {}",
            expected_atoms,
            atoms.len()
        ));
    }

    // Parse BOND section.
    let bonds: Vec<Mol2Bond> = if let Some(block) = sections.iter().find(|b| b.section == Section::Bond) {
        let mut bonds = Vec::with_capacity(expected_bonds);
        for (i, &line) in block.lines.iter().enumerate() {
            let line_num = block.first_line_num + i;
            bonds.push(parse_bond_line(line, line_num)?);
        }
        bonds
    } else {
        Vec::new()
    };

    if bonds.len() != expected_bonds {
        return Err(format!(
            "MOLECULE header declared {} bonds but BOND section contains {}",
            expected_bonds,
            bonds.len()
        ));
    }

    // Validate bond atom indices are within range.
    for (i, bond) in bonds.iter().enumerate() {
        if bond.atom1 >= atoms.len() {
            return Err(format!(
                "Bond {} references atom1 index {} but molecule has only {} atoms",
                i + 1,
                bond.atom1,
                atoms.len()
            ));
        }
        if bond.atom2 >= atoms.len() {
            return Err(format!(
                "Bond {} references atom2 index {} but molecule has only {} atoms",
                i + 1,
                bond.atom2,
                atoms.len()
            ));
        }
    }

    // Parse SUBSTRUCTURE section (optional).
    let substructures: Vec<Mol2Substructure> =
        if let Some(block) = sections.iter().find(|b| b.section == Section::Substructure) {
            let mut subs = Vec::new();
            for (i, &line) in block.lines.iter().enumerate() {
                let line_num = block.first_line_num + i;
                subs.push(parse_substructure_line(line, line_num)?);
            }
            subs
        } else {
            Vec::new()
        };

    Ok(Mol2Molecule {
        name,
        atoms,
        bonds,
        substructures,
    })
}

/// Parse a mol2 file from a file path.
///
/// Reads the entire file into memory and delegates to [`parse_mol2`].
///
/// # Arguments
/// * `path` - Path to the mol2 file.
///
/// # Returns
/// * `Ok(Mol2Molecule)` - The parsed molecule.
/// * `Err(String)` - A descriptive error message if reading or parsing fails.
pub fn parse_mol2_file(path: &Path) -> Result<Mol2Molecule, String> {
    let content = read_to_string(path)
        .map_err(|e| format!("Failed to read mol2 file '{}': {}", path.display(), e))?;
    parse_mol2(&content)
}
