//! SDF/MDL MOL file parser (V2000 format).
//!
//! Parses SDF and MOL files containing 2D/3D molecular structures with
//! atom coordinates, bond connectivity, and optional properties.
//! Only V2000 format is supported.

use std::fs::read_to_string;
use std::path::Path;

// ============================================================================
// Data Structures
// ============================================================================

/// A single atom from an SDF/MOL file.
#[derive(Debug, Clone)]
pub struct SdfAtom {
    /// Cartesian coordinates in Angstroms.
    pub position: [f64; 3],
    /// Element symbol (e.g. "C", "N", "O").
    pub element: String,
    /// Formal charge.
    pub charge: i32,
    /// Mass difference from standard isotope (0 = natural).
    pub mass_diff: i32,
    /// Valence (0 = default).
    pub valence: u8,
}

/// A bond from an SDF/MOL file.
#[derive(Debug, Clone)]
pub struct SdfBond {
    /// 0-based index of first atom.
    pub atom1: usize,
    /// 0-based index of second atom.
    pub atom2: usize,
    /// Bond type: 1=single, 2=double, 3=triple, 4=aromatic.
    pub bond_type: u8,
    /// Stereo flag (0=not stereo, 1=up, 4=either, 6=down).
    pub stereo: u8,
}

/// A complete SDF/MOL molecule.
#[derive(Debug, Clone)]
pub struct SdfMolecule {
    /// Molecule name (from header line 1).
    pub name: String,
    /// Comment (from header line 3, if present).
    pub comment: String,
    /// All atoms.
    pub atoms: Vec<SdfAtom>,
    /// All bonds.
    pub bonds: Vec<SdfBond>,
    /// Properties from M  lines and $$$ block.
    pub properties: Vec<(String, String)>,
}

// ============================================================================
// Charge decoding
// ============================================================================

/// Convert the V2000 charge field (column 36-38 of the atom block) to a formal charge.
fn decode_charge_field(val: u8) -> i32 {
    match val {
        0 => 0,
        1 => 3,
        2 => 2,
        3 => 1,
        4 => 0, // doublet radical
        5 => -1,
        6 => -2,
        7 => -3,
        _ => 0,
    }
}

// ============================================================================
// MOL Block Parser
// ============================================================================

/// Parse a single MOL block (V2000 format) from lines.
///
/// Returns the parsed molecule and the number of lines consumed.
fn parse_mol_block(lines: &[&str], start_line: usize) -> Result<(SdfMolecule, usize), String> {
    let n = lines.len();

    if n < 4 {
        return Err(format!(
            "MOL block starting at line {} has only {} lines, need at least 4 (header + counts)",
            start_line, n
        ));
    }

    // Header block: 3 lines
    let name = lines[0].trim().to_string();
    // Line 2 is program/timestamp (ignored)
    let comment = if n > 2 { lines[2].trim().to_string() } else { String::new() };

    // Counts line (line 4 = index 3)
    let counts_line = lines[3];
    if counts_line.len() < 6 {
        return Err(format!(
            "Line {}: counts line too short: '{}'",
            start_line + 4,
            counts_line
        ));
    }

    // V2000 counts line is fixed-format: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    // First 3 chars = num_atoms, next 3 = num_bonds
    let num_atoms: usize = counts_line[0..3]
        .trim()
        .parse()
        .map_err(|e| format!("Line {}: bad atom count '{}': {}", start_line + 4, &counts_line[0..3], e))?;

    let num_bonds: usize = counts_line[3..6]
        .trim()
        .parse()
        .map_err(|e| format!("Line {}: bad bond count '{}': {}", start_line + 4, &counts_line[3..6], e))?;

    // Check for V2000 tag
    if counts_line.contains("V3000") {
        return Err(format!(
            "Line {}: V3000 format not supported, only V2000",
            start_line + 4
        ));
    }

    let header_lines = 4; // 3 header + 1 counts
    let needed = header_lines + num_atoms + num_bonds;
    if n < needed {
        return Err(format!(
            "MOL block starting at line {} needs {} lines (header + {} atoms + {} bonds) but only {} available",
            start_line, needed, num_atoms, num_bonds, n
        ));
    }

    // Parse atom block
    let mut atoms = Vec::with_capacity(num_atoms);
    for i in 0..num_atoms {
        let line_idx = header_lines + i;
        let line = lines[line_idx];
        let line_num = start_line + line_idx + 1;

        // V2000 atom line fixed format:
        // xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaaddcccssshhhbbbvvvHHHrrriiimmmnnneee
        // Columns:  0-9 x, 10-19 y, 20-29 z, 31-33 element, 34-35 mass_diff,
        //           36-38 charge, ...
        if line.len() < 34 {
            // Try whitespace parsing as fallback
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 4 {
                return Err(format!(
                    "Atom line {} too short ({} chars, {} fields): '{}'",
                    line_num, line.len(), fields.len(), line.trim()
                ));
            }
            let x: f64 = fields[0].parse()
                .map_err(|e| format!("Atom line {}: bad x '{}': {}", line_num, fields[0], e))?;
            let y: f64 = fields[1].parse()
                .map_err(|e| format!("Atom line {}: bad y '{}': {}", line_num, fields[1], e))?;
            let z: f64 = fields[2].parse()
                .map_err(|e| format!("Atom line {}: bad z '{}': {}", line_num, fields[2], e))?;
            let element = fields[3].trim().to_string();

            let charge_field: u8 = if fields.len() > 6 {
                fields[6].parse().unwrap_or(0)
            } else {
                0
            };

            atoms.push(SdfAtom {
                position: [x, y, z],
                element,
                charge: decode_charge_field(charge_field),
                mass_diff: 0,
                valence: 0,
            });
            continue;
        }

        let x: f64 = line[0..10].trim().parse()
            .map_err(|e| format!("Atom line {}: bad x '{}': {}", line_num, &line[0..10], e))?;
        let y: f64 = line[10..20].trim().parse()
            .map_err(|e| format!("Atom line {}: bad y '{}': {}", line_num, &line[10..20], e))?;
        let z: f64 = line[20..30].trim().parse()
            .map_err(|e| format!("Atom line {}: bad z '{}': {}", line_num, &line[20..30], e))?;

        let element = line[31..34].trim().to_string();

        let mass_diff: i32 = if line.len() > 36 {
            line[34..36].trim().parse().unwrap_or(0)
        } else {
            0
        };

        let charge_field: u8 = if line.len() > 39 {
            line[36..39].trim().parse().unwrap_or(0)
        } else {
            0
        };

        let valence: u8 = if line.len() > 51 {
            line[48..51].trim().parse().unwrap_or(0)
        } else {
            0
        };

        atoms.push(SdfAtom {
            position: [x, y, z],
            element,
            charge: decode_charge_field(charge_field),
            mass_diff,
            valence,
        });
    }

    // Parse bond block
    let mut bonds = Vec::with_capacity(num_bonds);
    for i in 0..num_bonds {
        let line_idx = header_lines + num_atoms + i;
        let line = lines[line_idx];
        let line_num = start_line + line_idx + 1;

        // V2000 bond line: 111222tttsssxxxrrrccc
        // Columns: 0-2 atom1, 3-5 atom2, 6-8 type, 9-11 stereo
        if line.len() < 9 {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 3 {
                return Err(format!(
                    "Bond line {} too short: '{}'",
                    line_num, line.trim()
                ));
            }
            let a1: usize = fields[0].parse()
                .map_err(|e| format!("Bond line {}: bad atom1 '{}': {}", line_num, fields[0], e))?;
            let a2: usize = fields[1].parse()
                .map_err(|e| format!("Bond line {}: bad atom2 '{}': {}", line_num, fields[1], e))?;
            let bt: u8 = fields[2].parse()
                .map_err(|e| format!("Bond line {}: bad type '{}': {}", line_num, fields[2], e))?;
            let st: u8 = if fields.len() > 3 { fields[3].parse().unwrap_or(0) } else { 0 };

            if a1 == 0 || a2 == 0 {
                return Err(format!("Bond line {}: atom index is 0 (expected 1-based)", line_num));
            }
            bonds.push(SdfBond {
                atom1: a1 - 1,
                atom2: a2 - 1,
                bond_type: bt,
                stereo: st,
            });
            continue;
        }

        let a1: usize = line[0..3].trim().parse()
            .map_err(|e| format!("Bond line {}: bad atom1 '{}': {}", line_num, &line[0..3], e))?;
        let a2: usize = line[3..6].trim().parse()
            .map_err(|e| format!("Bond line {}: bad atom2 '{}': {}", line_num, &line[3..6], e))?;
        let bt: u8 = line[6..9].trim().parse()
            .map_err(|e| format!("Bond line {}: bad type '{}': {}", line_num, &line[6..9], e))?;
        let st: u8 = if line.len() > 12 {
            line[9..12].trim().parse().unwrap_or(0)
        } else {
            0
        };

        if a1 == 0 || a2 == 0 {
            return Err(format!("Bond line {}: atom index is 0 (expected 1-based)", line_num));
        }

        bonds.push(SdfBond {
            atom1: a1 - 1,
            atom2: a2 - 1,
            bond_type: bt,
            stereo: st,
        });
    }

    // Validate bond atom indices
    for (i, bond) in bonds.iter().enumerate() {
        if bond.atom1 >= atoms.len() {
            return Err(format!(
                "Bond {} references atom1 index {} but molecule has only {} atoms",
                i + 1, bond.atom1, atoms.len()
            ));
        }
        if bond.atom2 >= atoms.len() {
            return Err(format!(
                "Bond {} references atom2 index {} but molecule has only {} atoms",
                i + 1, bond.atom2, atoms.len()
            ));
        }
    }

    // Parse properties (M  lines and beyond)
    let mut properties = Vec::new();
    let mut consumed = needed;

    for i in needed..n {
        let line = lines[i].trim();
        consumed = i + 1;

        // Handle M  CHG lines to update formal charges
        if line.starts_with("M  CHG") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let count: usize = parts[2].parse().unwrap_or(0);
                for j in 0..count {
                    let idx_pos = 3 + j * 2;
                    let val_pos = 4 + j * 2;
                    if val_pos < parts.len() {
                        if let (Ok(atom_idx), Ok(charge)) =
                            (parts[idx_pos].parse::<usize>(), parts[val_pos].parse::<i32>())
                        {
                            if atom_idx >= 1 && atom_idx <= atoms.len() {
                                atoms[atom_idx - 1].charge = charge;
                            }
                        }
                    }
                }
            }
        }

        if line == "M  END" {
            break;
        }
    }

    // Parse SDF properties (between M  END and $$$$)
    let mut prop_name = String::new();
    for i in consumed..n {
        let line = lines[i];
        let trimmed = line.trim();
        consumed = i + 1;

        if trimmed == "$$$$" {
            break;
        }

        if trimmed.starts_with("> <") && trimmed.ends_with('>') {
            prop_name = trimmed[3..trimmed.len() - 1].to_string();
        } else if !prop_name.is_empty() && !trimmed.is_empty() {
            properties.push((prop_name.clone(), trimmed.to_string()));
            prop_name.clear();
        }
    }

    Ok((
        SdfMolecule {
            name,
            comment,
            atoms,
            bonds,
            properties,
        },
        consumed,
    ))
}

// ============================================================================
// Top-Level Parsers
// ============================================================================

/// Parse a single MOL block from text content.
pub fn parse_mol(content: &str) -> Result<SdfMolecule, String> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Err("MOL content is empty".into());
    }
    let (mol, _) = parse_mol_block(&lines, 0)?;
    Ok(mol)
}

/// Parse an SDF file from text content.
///
/// Returns all molecules in the file. Molecules are separated by `$$$$` lines.
pub fn parse_sdf(content: &str) -> Result<Vec<SdfMolecule>, String> {
    let all_lines: Vec<&str> = content.lines().collect();
    if all_lines.is_empty() {
        return Err("SDF content is empty".into());
    }

    let mut molecules = Vec::new();
    let mut offset = 0;

    while offset < all_lines.len() {
        // Skip blank lines between molecules
        while offset < all_lines.len() && all_lines[offset].trim().is_empty() {
            offset += 1;
        }
        if offset >= all_lines.len() {
            break;
        }

        let (mol, consumed) = parse_mol_block(&all_lines[offset..], offset)?;
        molecules.push(mol);
        offset += consumed;
    }

    if molecules.is_empty() {
        return Err("No molecules found in SDF content".into());
    }

    Ok(molecules)
}

/// Parse an SDF or MOL file from a file path.
///
/// If the file contains multiple molecules (SDF format), all are returned.
/// For a single MOL file, a single-element vector is returned.
pub fn parse_sdf_file(path: &Path) -> Result<Vec<SdfMolecule>, String> {
    let content = read_to_string(path)
        .map_err(|e| format!("Failed to read SDF file '{}': {}", path.display(), e))?;
    parse_sdf(&content)
}

/// Parse a single molecule from an SDF or MOL file.
///
/// Returns the first molecule found.
pub fn parse_mol_file(path: &Path) -> Result<SdfMolecule, String> {
    let content = read_to_string(path)
        .map_err(|e| format!("Failed to read MOL file '{}': {}", path.display(), e))?;
    parse_mol(&content)
}

// ============================================================================
// Utility
// ============================================================================

impl SdfMolecule {
    /// Get the atomic number for an element symbol.
    pub fn atomic_number(element: &str) -> u8 {
        match element.to_uppercase().trim() {
            "H" => 1,
            "HE" => 2,
            "LI" => 3,
            "BE" => 4,
            "B" => 5,
            "C" => 6,
            "N" => 7,
            "O" => 8,
            "F" => 9,
            "NE" => 10,
            "NA" => 11,
            "MG" => 12,
            "AL" => 13,
            "SI" => 14,
            "P" => 15,
            "S" => 16,
            "CL" => 17,
            "AR" => 18,
            "K" => 19,
            "CA" => 20,
            "BR" => 35,
            "I" => 53,
            _ => 0,
        }
    }

    /// Get atomic numbers for all atoms.
    pub fn atomic_numbers(&self) -> Vec<u8> {
        self.atoms
            .iter()
            .map(|a| Self::atomic_number(&a.element))
            .collect()
    }

    /// Get coordinates as a flat array of [x, y, z] triplets.
    pub fn coordinates(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(|a| a.position).collect()
    }

    /// Compute the total formal charge.
    pub fn total_charge(&self) -> i32 {
        self.atoms.iter().map(|a| a.charge).sum()
    }
}
