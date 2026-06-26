//! PDB (Protein Data Bank) file format writer.
//!
//! Produces valid PDB files with ATOM/HETATM, CONECT, CRYST1, TER, and END
//! records, following the column-based PDB format specification.
//!
//! Coordinates are written with 8.3 formatting, occupancy and B-factor with
//! 6.2 formatting, and all fields respect the strict column alignment required
//! by the PDB standard.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

// ============================================================================
// Data Structures
// ============================================================================

/// Data for a single atom to write.
#[derive(Debug, Clone)]
pub struct PdbWriteAtom {
    /// Serial number (1-based, auto-assigned if 0).
    pub serial: usize,
    /// Atom name (e.g. "CA", "N", "OXT") - will be formatted to PDB column width.
    pub name: String,
    /// Residue name (e.g. "ALA", "WAT").
    pub residue_name: String,
    /// Chain ID.
    pub chain_id: char,
    /// Residue sequence number.
    pub res_seq: i32,
    /// Insertion code (' ' for none).
    pub i_code: char,
    /// Coordinates in Angstroms.
    pub position: [f64; 3],
    /// Occupancy (default 1.0).
    pub occupancy: f64,
    /// B-factor (default 0.0).
    pub temp_factor: f64,
    /// Element symbol (e.g. "C", "N", "O").
    pub element: String,
    /// True for HETATM, false for ATOM.
    pub is_hetatm: bool,
}

/// A bond to write in CONECT records.
#[derive(Debug, Clone)]
pub struct PdbWriteBond {
    /// 1-based serial of first atom.
    pub serial1: usize,
    /// 1-based serial of second atom.
    pub serial2: usize,
}

/// Box dimensions for CRYST1 record.
#[derive(Debug, Clone)]
pub struct PdbCryst1 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

/// Complete data for writing a PDB file.
#[derive(Debug, Clone)]
pub struct PdbWriteData {
    pub atoms: Vec<PdbWriteAtom>,
    pub bonds: Vec<PdbWriteBond>,
    pub cryst1: Option<PdbCryst1>,
}

// ============================================================================
// Atom Name Formatting
// ============================================================================

/// Format an atom name into a 4-character PDB atom name field (columns 13-16).
///
/// The PDB specification requires specific alignment of atom names depending on
/// the element symbol length:
/// - If the atom name is 4 characters long, it is left-justified.
/// - If the atom name starts with a digit, it is left-justified.
/// - Otherwise, the name is padded with a leading space and left-justified in
///   the remaining 3 characters (the standard AMBER convention).
fn format_atom_name(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.len() >= 4 {
        // Truncate or use as-is for 4-character names
        format!("{:<4}", &trimmed[..4.min(trimmed.len())])
    } else if trimmed.is_empty() {
        "    ".to_string()
    } else if trimmed.as_bytes()[0].is_ascii_digit() {
        // Names starting with a digit are left-justified
        format!("{:<4}", trimmed)
    } else {
        // Standard: space prefix, left-justify in remaining 3 columns
        format!(" {:<3}", trimmed)
    }
}

// ============================================================================
// Serial Number Helpers
// ============================================================================

/// Wrap a serial number to stay within the 5-digit PDB field (1..=99999).
/// Returns 1 when the value exceeds 99999.
fn wrap_serial(serial: usize) -> usize {
    if serial > 99999 {
        ((serial - 1) % 99999) + 1
    } else if serial == 0 {
        1
    } else {
        serial
    }
}

/// Wrap a residue sequence number to stay within the 4-digit PDB field
/// (-999..=9999). Values outside this range are wrapped modulo 10000.
fn wrap_res_seq(res_seq: i32) -> i32 {
    if res_seq > 9999 {
        ((res_seq - 1) % 9999) + 1
    } else if res_seq < -999 {
        -((-res_seq - 1) % 999 + 1)
    } else {
        res_seq
    }
}

// ============================================================================
// Record Formatters
// ============================================================================

/// Format a CRYST1 record line.
///
/// ```text
/// CRYST1    a       b       c     alpha  beta   gamma sGroup   Z
/// 1-6    7-15    16-24   25-33  34-40  41-47  48-54  56-66   67-70
/// ```
fn format_cryst1(cryst: &PdbCryst1) -> String {
    format!(
        "CRYST1{:>9.3}{:>9.3}{:>9.3}{:>7.2}{:>7.2}{:>7.2} P 1           1\n",
        cryst.a, cryst.b, cryst.c, cryst.alpha, cryst.beta, cryst.gamma,
    )
}

/// Format a single ATOM or HETATM record line.
///
/// Strict 80-column PDB format with all fields in their correct positions.
#[allow(clippy::too_many_arguments)]
fn format_atom_record(
    record_name: &str,
    serial: usize,
    atom_name: &str,
    residue_name: &str,
    chain_id: char,
    res_seq: i32,
    i_code: char,
    x: f64,
    y: f64,
    z: f64,
    occupancy: f64,
    temp_factor: f64,
    element: &str,
) -> String {
    let serial_w = wrap_serial(serial);
    let res_seq_w = wrap_res_seq(res_seq);
    let atom_field = format_atom_name(atom_name);

    // Right-justify residue name in a 3-character field
    let res_name_field = format!("{:>3}", residue_name);

    // Right-justify element in a 2-character field
    let element_field = format!("{:>2}", element);

    // Build the line strictly by column position:
    //  1- 6: record name (left-justified, 6 chars)
    //  7-11: serial (right-justified, 5 chars)
    // 12:    space
    // 13-16: atom name (4 chars, pre-formatted)
    // 17:    altLoc (space)
    // 18-20: residue name (right-justified, 3 chars)
    // 21:    space
    // 22:    chain ID
    // 23-26: res_seq (right-justified, 4 chars)
    // 27:    iCode
    // 28-30: spaces (3 chars)
    // 31-38: x (8.3f)
    // 39-46: y (8.3f)
    // 47-54: z (8.3f)
    // 55-60: occupancy (6.2f)
    // 61-66: temp_factor (6.2f)
    // 67-76: spaces (10 chars)
    // 77-78: element (right-justified, 2 chars)
    // 79-80: charge (2 spaces)
    format!(
        "{:<6}{:>5} {}{} {:>3} {}{:>4}{}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}  \n",
        record_name,
        serial_w,
        atom_field,
        ' ', // altLoc
        res_name_field,
        chain_id,
        res_seq_w,
        i_code,
        x,
        y,
        z,
        occupancy,
        temp_factor,
        element_field,
    )
}

/// Format a TER record.
///
/// ```text
/// TER   serial      resName chain resSeq iCode
/// 1-6   7-11   12   18-20   22    23-26  27
/// ```
fn format_ter_record(
    serial: usize,
    residue_name: &str,
    chain_id: char,
    res_seq: i32,
    i_code: char,
) -> String {
    let serial_w = wrap_serial(serial);
    let res_seq_w = wrap_res_seq(res_seq);
    format!(
        "TER   {:>5}      {:>3} {}{:>4}{}\n",
        serial_w, residue_name, chain_id, res_seq_w, i_code,
    )
}

/// Format CONECT records for a set of bonds grouped by central atom.
///
/// Each CONECT line can hold up to 4 bonded atoms. If a central atom has more
/// than 4 bonds, multiple CONECT lines are emitted.
fn format_conect_records(bonds: &[PdbWriteBond]) -> String {
    // Group bonds by central atom (serial1)
    let mut bond_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for bond in bonds {
        bond_map
            .entry(bond.serial1)
            .or_default()
            .push(bond.serial2);
        bond_map
            .entry(bond.serial2)
            .or_default()
            .push(bond.serial1);
    }

    // Sort by central atom serial for deterministic output
    let mut central_serials: Vec<usize> = bond_map.keys().copied().collect();
    central_serials.sort_unstable();

    let mut output = String::new();
    for central in central_serials {
        let partners = &bond_map[&central];
        let central_w = wrap_serial(central);

        // Emit CONECT lines in chunks of 4 bonded atoms
        for chunk in partners.chunks(4) {
            let _ = write!(output, "CONECT{:>5}", central_w);
            for &partner in chunk {
                let partner_w = wrap_serial(partner);
                let _ = write!(output, "{:>5}", partner_w);
            }
            output.push('\n');
        }
    }

    output
}

// ============================================================================
// Auto-Assignment
// ============================================================================

/// Assign serial numbers to atoms that have serial == 0, starting from 1.
/// Atoms with non-zero serials keep their original values.
fn assign_serials(atoms: &[PdbWriteAtom]) -> Vec<usize> {
    let all_zero = atoms.iter().all(|a| a.serial == 0);

    if all_zero {
        // Auto-assign sequentially from 1
        (1..=atoms.len()).collect()
    } else {
        // Use provided serials, auto-assigning only where serial == 0
        let mut next_serial = atoms
            .iter()
            .map(|a| a.serial)
            .max()
            .unwrap_or(0)
            + 1;

        atoms
            .iter()
            .map(|a| {
                if a.serial == 0 {
                    let s = next_serial;
                    next_serial += 1;
                    s
                } else {
                    a.serial
                }
            })
            .collect()
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Write a PDB structure to a `String`.
///
/// Produces a complete PDB file with optional CRYST1 record, ATOM/HETATM
/// records, TER records at chain boundaries, CONECT records for explicit bonds,
/// and a terminating END record.
///
/// # Arguments
/// * `data` - The complete PDB write data including atoms, bonds, and optional
///   unit cell.
///
/// # Returns
/// * `Ok(String)` - The formatted PDB file content.
/// * `Err(String)` - A descriptive error if formatting fails.
pub fn write_pdb_string(data: &PdbWriteData) -> Result<String, String> {
    // Pre-allocate a reasonable buffer: ~82 bytes per ATOM line
    let mut output = String::with_capacity(data.atoms.len() * 82 + 256);

    // CRYST1 record (if present)
    if let Some(ref cryst) = data.cryst1 {
        output.push_str(&format_cryst1(cryst));
    }

    // Assign serial numbers
    let serials = assign_serials(&data.atoms);

    // Track chain changes for TER insertion
    let mut prev_chain_id: Option<char> = None;
    // Running serial counter for TER records (TER gets the next serial after
    // the last atom in the chain)
    let mut ter_serial_offset: usize = 0;

    for (i, atom) in data.atoms.iter().enumerate() {
        let serial = serials[i] + ter_serial_offset;

        // Insert TER record when chain_id changes (but not before the first atom)
        if let Some(prev_chain) = prev_chain_id {
            if atom.chain_id != prev_chain {
                // TER record references the previous atom's residue info
                let prev_atom = &data.atoms[i - 1];
                let ter_serial = serials[i - 1] + ter_serial_offset + 1;
                output.push_str(&format_ter_record(
                    ter_serial,
                    &prev_atom.residue_name,
                    prev_atom.chain_id,
                    prev_atom.res_seq,
                    prev_atom.i_code,
                ));
                ter_serial_offset += 1;
            }
        }
        prev_chain_id = Some(atom.chain_id);

        let record_name = if atom.is_hetatm { "HETATM" } else { "ATOM" };

        output.push_str(&format_atom_record(
            record_name,
            serial,
            &atom.name,
            &atom.residue_name,
            atom.chain_id,
            atom.res_seq,
            atom.i_code,
            atom.position[0],
            atom.position[1],
            atom.position[2],
            atom.occupancy,
            atom.temp_factor,
            &atom.element,
        ));
    }

    // Final TER record after the last atom (if there are any atoms)
    if let Some(last_atom) = data.atoms.last() {
        let last_serial = serials[data.atoms.len() - 1] + ter_serial_offset;
        output.push_str(&format_ter_record(
            last_serial + 1,
            &last_atom.residue_name,
            last_atom.chain_id,
            last_atom.res_seq,
            last_atom.i_code,
        ));
    }

    // CONECT records
    if !data.bonds.is_empty() {
        output.push_str(&format_conect_records(&data.bonds));
    }

    // END record
    output.push_str("END\n");

    Ok(output)
}

/// Write a PDB file to disk.
///
/// Formats the PDB data as a string and writes it to the specified path.
///
/// # Arguments
/// * `data` - The complete PDB write data.
/// * `path` - Destination file path.
///
/// # Returns
/// * `Ok(())` - File written successfully.
/// * `Err(String)` - A descriptive error if formatting or I/O fails.
pub fn write_pdb(data: &PdbWriteData, path: &Path) -> Result<(), String> {
    let content = write_pdb_string(data)?;
    fs::write(path, content)
        .map_err(|e| format!("failed to write PDB file '{}': {}", path.display(), e))
}
