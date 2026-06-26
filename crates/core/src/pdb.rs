//! PDB (Protein Data Bank) file format parser.
//!
//! Parses standard PDB files used in structural biology and molecular simulation.
//! Handles ATOM/HETATM, CONECT, CRYST1, TER/END, and MODEL/ENDMDL records with
//! strict column-based field extraction per the PDB format specification.
//!
//! Key features:
//! - Column-based parsing (not whitespace splitting) for ATOM/HETATM records
//! - Alternate conformation resolution (highest occupancy, prefer 'A' on ties)
//! - Insertion code support
//! - Multi-model files: only model 1 is parsed
//! - Tolerant of truncated lines

use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ============================================================================
// Data Structures
// ============================================================================

/// A single atom from a PDB file.
#[derive(Debug, Clone)]
pub struct PdbAtom {
    /// Serial number (1-based)
    pub serial: usize,
    /// Atom name (e.g. "CA", "N", "OXT") - trimmed
    pub name: String,
    /// Alternate location indicator (' ' if none)
    pub alt_loc: char,
    /// Residue name (e.g. "ALA", "WAT")
    pub residue_name: String,
    /// Chain identifier
    pub chain_id: char,
    /// Residue sequence number
    pub res_seq: i32,
    /// Insertion code (' ' if none)
    pub i_code: char,
    /// Cartesian coordinates in Angstroms
    pub position: [f64; 3],
    /// Occupancy factor
    pub occupancy: f64,
    /// Temperature factor (B-factor)
    pub temp_factor: f64,
    /// Element symbol (from columns 77-78 if present)
    pub element: String,
    /// True if this is a HETATM record
    pub is_hetatm: bool,
}

/// A residue grouping.
#[derive(Debug, Clone)]
pub struct PdbResidue {
    /// Residue name
    pub name: String,
    /// Chain ID
    pub chain_id: char,
    /// Residue sequence number
    pub res_seq: i32,
    /// Insertion code
    pub i_code: char,
    /// Indices into the parent PdbStructure's atoms vec (0-based)
    pub atom_indices: Vec<usize>,
}

/// A chain in the PDB structure.
#[derive(Debug, Clone)]
pub struct PdbChain {
    /// Chain identifier
    pub chain_id: char,
    /// Indices into the parent PdbStructure's residues vec
    pub residue_indices: Vec<usize>,
}

/// An explicit bond from CONECT records.
#[derive(Debug, Clone)]
pub struct PdbBond {
    /// 0-based atom index
    pub atom1: usize,
    /// 0-based atom index
    pub atom2: usize,
}

/// Unit cell from CRYST1 record.
#[derive(Debug, Clone)]
pub struct UnitCell {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

/// Complete PDB structure.
#[derive(Debug, Clone)]
pub struct PdbStructure {
    /// All atoms (after alt-loc filtering)
    pub atoms: Vec<PdbAtom>,
    /// Residues (groups of atoms)
    pub residues: Vec<PdbResidue>,
    /// Chains
    pub chains: Vec<PdbChain>,
    /// Explicit bonds from CONECT records
    pub conect_bonds: Vec<PdbBond>,
    /// Unit cell (from CRYST1, if present)
    pub unit_cell: Option<UnitCell>,
}

// ============================================================================
// Column Extraction Helpers
// ============================================================================

/// Safely extract a substring from a line using 0-based byte indices.
/// Returns an empty string if the range is out of bounds.
fn col(line: &str, start: usize, end: usize) -> &str {
    let bytes = line.as_bytes();
    let len = bytes.len();
    if start >= len {
        return "";
    }
    let actual_end = end.min(len);
    // Safety: PDB files are ASCII; slicing on byte boundaries is fine.
    &line[start..actual_end]
}

/// Extract a single character at a 0-based position. Returns ' ' if out of bounds.
fn col_char(line: &str, pos: usize) -> char {
    line.as_bytes()
        .get(pos)
        .map(|&b| b as char)
        .unwrap_or(' ')
}

/// Parse a trimmed column range as an integer. Returns Err on failure.
fn col_i32(line: &str, start: usize, end: usize) -> Result<i32, String> {
    let s = col(line, start, end).trim();
    if s.is_empty() {
        return Err(format!(
            "empty integer field at columns {}-{}",
            start + 1,
            end
        ));
    }
    s.parse::<i32>()
        .map_err(|e| format!("failed to parse integer '{}' at columns {}-{}: {}", s, start + 1, end, e))
}

/// Parse a trimmed column range as a usize. Returns Err on failure.
fn col_usize(line: &str, start: usize, end: usize) -> Result<usize, String> {
    let s = col(line, start, end).trim();
    if s.is_empty() {
        return Err(format!(
            "empty integer field at columns {}-{}",
            start + 1,
            end
        ));
    }
    s.parse::<usize>()
        .map_err(|e| format!("failed to parse integer '{}' at columns {}-{}: {}", s, start + 1, end, e))
}

/// Parse a trimmed column range as f64. Returns Err on failure.
fn col_f64(line: &str, start: usize, end: usize) -> Result<f64, String> {
    let s = col(line, start, end).trim();
    if s.is_empty() {
        return Err(format!(
            "empty float field at columns {}-{}",
            start + 1,
            end
        ));
    }
    s.parse::<f64>()
        .map_err(|e| format!("failed to parse float '{}' at columns {}-{}: {}", s, start + 1, end, e))
}

/// Parse a trimmed column range as f64, returning a default if the field is
/// empty or the line is too short.
fn col_f64_or(line: &str, start: usize, end: usize, default: f64) -> f64 {
    let s = col(line, start, end).trim();
    if s.is_empty() {
        return default;
    }
    s.parse::<f64>().unwrap_or(default)
}

// ============================================================================
// Internal Raw Atom (before alt-loc filtering)
// ============================================================================

/// Raw atom record before alternate-location filtering.
#[derive(Debug, Clone)]
struct RawAtom {
    serial: usize,
    name: String,
    alt_loc: char,
    residue_name: String,
    chain_id: char,
    res_seq: i32,
    i_code: char,
    position: [f64; 3],
    occupancy: f64,
    temp_factor: f64,
    element: String,
    is_hetatm: bool,
}

/// Unique key for identifying atom positions within a residue.
/// Used for alternate-location grouping.
type AltLocKey = (String, char, i32, char, String); // (resname, chain, resseq, icode, atomname)

// ============================================================================
// Record Parsers
// ============================================================================

/// Parse an ATOM or HETATM record line into a RawAtom.
///
/// PDB column layout (1-based, converted to 0-based internally):
///   1- 6  Record type
///   7-11  Serial number
///  12     Space
///  13-16  Atom name
///  17     Alt location
///  18-20  Residue name
///  21     Space
///  22     Chain ID
///  23-26  Residue sequence number
///  27     Insertion code
///  28-30  (blank)
///  31-38  X
///  39-46  Y
///  47-54  Z
///  55-60  Occupancy
///  61-66  Temp factor
///  67-76  (blank)
///  77-78  Element symbol
fn parse_atom_line(line: &str, is_hetatm: bool) -> Result<RawAtom, String> {
    // Serial: columns 7-11 (0-based 6..11)
    let serial = col_usize(line, 6, 11)
        .map_err(|e| format!("ATOM serial: {}", e))?;

    // Atom name: columns 13-16 (0-based 12..16)
    let name = col(line, 12, 16).trim().to_string();

    // Alt loc: column 17 (0-based 16)
    let alt_loc = col_char(line, 16);

    // Residue name: columns 18-20 (0-based 17..20)
    let residue_name = col(line, 17, 20).trim().to_string();

    // Chain ID: column 22 (0-based 21)
    let chain_id = col_char(line, 21);

    // Residue sequence number: columns 23-26 (0-based 22..26)
    let res_seq = col_i32(line, 22, 26)
        .map_err(|e| format!("ATOM res_seq: {}", e))?;

    // Insertion code: column 27 (0-based 26)
    let i_code = col_char(line, 26);

    // Coordinates: columns 31-38, 39-46, 47-54 (0-based 30..38, 38..46, 46..54)
    let x = col_f64(line, 30, 38)
        .map_err(|e| format!("ATOM X coordinate: {}", e))?;
    let y = col_f64(line, 38, 46)
        .map_err(|e| format!("ATOM Y coordinate: {}", e))?;
    let z = col_f64(line, 46, 54)
        .map_err(|e| format!("ATOM Z coordinate: {}", e))?;

    // Occupancy: columns 55-60 (0-based 54..60), default 1.0
    let occupancy = col_f64_or(line, 54, 60, 1.0);

    // Temp factor: columns 61-66 (0-based 60..66), default 0.0
    let temp_factor = col_f64_or(line, 60, 66, 0.0);

    // Element: columns 77-78 (0-based 76..78)
    let element = col(line, 76, 78).trim().to_string();

    Ok(RawAtom {
        serial,
        name,
        alt_loc,
        residue_name,
        chain_id,
        res_seq,
        i_code,
        position: [x, y, z],
        occupancy,
        temp_factor,
        element,
        is_hetatm,
    })
}

/// Parse a CRYST1 record line into a UnitCell.
///
/// CRYST1 column layout (1-based):
///   1- 6  "CRYST1"
///   7-15  a (9.3f)
///  16-24  b (9.3f)
///  25-33  c (9.3f)
///  34-40  alpha (7.2f)
///  41-47  beta  (7.2f)
///  48-54  gamma (7.2f)
fn parse_cryst1_line(line: &str) -> Result<UnitCell, String> {
    let a = col_f64(line, 6, 15).map_err(|e| format!("CRYST1 a: {}", e))?;
    let b = col_f64(line, 15, 24).map_err(|e| format!("CRYST1 b: {}", e))?;
    let c = col_f64(line, 24, 33).map_err(|e| format!("CRYST1 c: {}", e))?;
    let alpha = col_f64(line, 33, 40).map_err(|e| format!("CRYST1 alpha: {}", e))?;
    let beta = col_f64(line, 40, 47).map_err(|e| format!("CRYST1 beta: {}", e))?;
    let gamma = col_f64(line, 47, 54).map_err(|e| format!("CRYST1 gamma: {}", e))?;

    Ok(UnitCell {
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
    })
}

/// Parse a CONECT record line, returning (central_serial, Vec<bonded_serials>).
///
/// CONECT column layout (1-based):
///   1- 6  "CONECT"
///   7-11  Serial of central atom
///  12-16  Bonded atom 1
///  17-21  Bonded atom 2
///  22-26  Bonded atom 3
///  27-31  Bonded atom 4
fn parse_conect_line(line: &str) -> Result<(usize, Vec<usize>), String> {
    let central = col_usize(line, 6, 11)
        .map_err(|e| format!("CONECT central atom: {}", e))?;

    let mut bonded = Vec::new();
    // Up to 4 bonded atom fields at 5-character intervals starting at column 12
    for i in 0..4 {
        let start = 11 + i * 5;
        let end = start + 5;
        let field = col(line, start, end).trim();
        if field.is_empty() {
            continue;
        }
        match field.parse::<usize>() {
            Ok(serial) => bonded.push(serial),
            Err(_) => {
                // Silently skip unparseable bonded atom fields; some PDB files
                // have trailing garbage.
            }
        }
    }

    Ok((central, bonded))
}

// ============================================================================
// Alternate Location Resolution
// ============================================================================

/// Given a set of raw atoms, resolve alternate conformations.
///
/// For each unique (residue_name, chain_id, res_seq, i_code, atom_name) group
/// that has multiple alt-loc entries, keep only the one with the highest
/// occupancy. If occupancies are equal, prefer alt_loc 'A'.
///
/// Atoms with no alt-loc indicator (space) pass through unchanged.
fn resolve_alt_locs(raw_atoms: Vec<RawAtom>) -> Vec<RawAtom> {
    // Group atoms by their alt-loc key. We preserve insertion order by tracking
    // the index of the first occurrence.
    let mut groups: HashMap<AltLocKey, Vec<usize>> = HashMap::new();
    let mut order: Vec<AltLocKey> = Vec::new();

    for (idx, atom) in raw_atoms.iter().enumerate() {
        let key: AltLocKey = (
            atom.residue_name.clone(),
            atom.chain_id,
            atom.res_seq,
            atom.i_code,
            atom.name.clone(),
        );

        let entry = groups.entry(key.clone());
        use std::collections::hash_map::Entry;
        match entry {
            Entry::Vacant(v) => {
                v.insert(vec![idx]);
                order.push(key);
            }
            Entry::Occupied(mut o) => {
                o.get_mut().push(idx);
            }
        }
    }

    // For each group, pick the best atom.
    let mut kept_indices: Vec<usize> = Vec::with_capacity(order.len());
    for key in &order {
        let indices = &groups[key];
        if indices.len() == 1 {
            kept_indices.push(indices[0]);
        } else {
            // Multiple alt-locs: pick highest occupancy, prefer 'A' on tie.
            let best = indices
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let atom_a = &raw_atoms[a];
                    let atom_b = &raw_atoms[b];

                    // Primary: higher occupancy wins
                    atom_a
                        .occupancy
                        .partial_cmp(&atom_b.occupancy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // Secondary: prefer 'A' (lower character) on tie
                        .then_with(|| atom_b.alt_loc.cmp(&atom_a.alt_loc))
                })
                .expect("indices vec is non-empty");
            kept_indices.push(best);
        }
    }

    // Sort by original order so atoms come out in file order.
    kept_indices.sort_unstable();

    kept_indices
        .into_iter()
        .map(|i| raw_atoms[i].clone())
        .collect()
}

// ============================================================================
// Structure Building
// ============================================================================

/// Convert a RawAtom into a PdbAtom.
fn raw_to_pdb_atom(raw: &RawAtom) -> PdbAtom {
    PdbAtom {
        serial: raw.serial,
        name: raw.name.clone(),
        alt_loc: raw.alt_loc,
        residue_name: raw.residue_name.clone(),
        chain_id: raw.chain_id,
        res_seq: raw.res_seq,
        i_code: raw.i_code,
        position: raw.position,
        occupancy: raw.occupancy,
        temp_factor: raw.temp_factor,
        element: raw.element.clone(),
        is_hetatm: raw.is_hetatm,
    }
}

/// Build residue and chain groupings from a list of atoms.
///
/// Residues are identified by unique (chain_id, res_seq, i_code) tuples in the
/// order they first appear. Chains are identified by chain_id in the order
/// they first appear.
fn build_hierarchy(atoms: &[PdbAtom]) -> (Vec<PdbResidue>, Vec<PdbChain>) {
    // -- Build residues --
    // Key: (chain_id, res_seq, i_code)
    type ResKey = (char, i32, char);
    let mut residue_map: HashMap<ResKey, usize> = HashMap::new();
    let mut residues: Vec<PdbResidue> = Vec::new();
    let mut residue_order: Vec<ResKey> = Vec::new();

    for (atom_idx, atom) in atoms.iter().enumerate() {
        let key: ResKey = (atom.chain_id, atom.res_seq, atom.i_code);
        if let Some(&res_idx) = residue_map.get(&key) {
            residues[res_idx].atom_indices.push(atom_idx);
        } else {
            let res_idx = residues.len();
            residue_map.insert(key, res_idx);
            residue_order.push(key);
            residues.push(PdbResidue {
                name: atom.residue_name.clone(),
                chain_id: atom.chain_id,
                res_seq: atom.res_seq,
                i_code: atom.i_code,
                atom_indices: vec![atom_idx],
            });
        }
    }

    // -- Build chains --
    let mut chain_map: HashMap<char, usize> = HashMap::new();
    let mut chains: Vec<PdbChain> = Vec::new();

    for (res_idx, residue) in residues.iter().enumerate() {
        if let Some(&chain_idx) = chain_map.get(&residue.chain_id) {
            chains[chain_idx].residue_indices.push(res_idx);
        } else {
            let chain_idx = chains.len();
            chain_map.insert(residue.chain_id, chain_idx);
            chains.push(PdbChain {
                chain_id: residue.chain_id,
                residue_indices: vec![res_idx],
            });
        }
    }

    (residues, chains)
}

/// Build CONECT bonds, translating serial numbers to 0-based atom indices.
/// Duplicate bonds (A-B and B-A from separate CONECT lines) are deduplicated.
fn build_conect_bonds(
    conect_records: &[(usize, Vec<usize>)],
    serial_to_index: &HashMap<usize, usize>,
) -> Vec<PdbBond> {
    let mut seen: HashMap<(usize, usize), bool> = HashMap::new();
    let mut bonds: Vec<PdbBond> = Vec::new();

    for (central_serial, bonded_serials) in conect_records {
        let Some(&idx1) = serial_to_index.get(central_serial) else {
            continue;
        };
        for bonded_serial in bonded_serials {
            let Some(&idx2) = serial_to_index.get(bonded_serial) else {
                continue;
            };
            if idx1 == idx2 {
                continue;
            }
            // Canonical pair ordering for deduplication
            let pair = if idx1 < idx2 {
                (idx1, idx2)
            } else {
                (idx2, idx1)
            };
            if seen.contains_key(&pair) {
                continue;
            }
            seen.insert(pair, true);
            bonds.push(PdbBond {
                atom1: pair.0,
                atom2: pair.1,
            });
        }
    }

    bonds
}

// ============================================================================
// Public API
// ============================================================================

/// Parse PDB content from a string.
///
/// Handles ATOM, HETATM, CONECT, CRYST1, TER, END, MODEL, and ENDMDL records.
/// Only model 1 is parsed for multi-model files. Alternate conformations are
/// resolved by keeping the highest-occupancy entry (preferring alt-loc 'A' on
/// ties).
///
/// # Arguments
/// * `content` - Full text content of a PDB file
///
/// # Returns
/// * `Ok(PdbStructure)` - Parsed structure with atoms, residues, chains, bonds,
///   and unit cell
/// * `Err(String)` - Descriptive error message on parse failure
pub fn parse_pdb(content: &str) -> Result<PdbStructure, String> {
    let mut raw_atoms: Vec<RawAtom> = Vec::new();
    let mut conect_records: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut unit_cell: Option<UnitCell> = None;

    // MODEL tracking: if we see MODEL records, only parse model 1.
    let mut in_model = false;
    let mut model_number: Option<usize> = None;
    let mut seen_model_record = false;
    let mut past_model_1 = false;

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1; // 1-based for error messages

        // Fast prefix check (PDB record types are in columns 1-6)
        if line.len() < 3 {
            continue;
        }

        // Check for MODEL / ENDMDL to restrict to model 1
        if line.starts_with("MODEL") && !line.starts_with("MODELE") {
            seen_model_record = true;
            // Parse model number from columns 11-14 (0-based 10..14) or just
            // use whitespace parsing since MODEL format varies.
            let model_str = if line.len() > 6 {
                line[6..].trim()
            } else {
                "1"
            };
            let num = model_str.parse::<usize>().unwrap_or(1);
            if num == 1 {
                in_model = true;
                model_number = Some(1);
            } else {
                past_model_1 = true;
            }
            continue;
        }

        if line.starts_with("ENDMDL") {
            if model_number == Some(1) {
                in_model = false;
                past_model_1 = true;
            }
            continue;
        }

        // If we have seen MODEL records, only process lines within model 1.
        if seen_model_record && !in_model {
            // Still process CONECT and CRYST1 outside model blocks (they are
            // typically at the end of the file, after ENDMDL).
            if line.starts_with("CONECT") {
                match parse_conect_line(line) {
                    Ok(record) => conect_records.push(record),
                    Err(e) => {
                        return Err(format!("line {}: {}", line_num, e));
                    }
                }
                continue;
            }
            if line.starts_with("CRYST1") && unit_cell.is_none() {
                match parse_cryst1_line(line) {
                    Ok(uc) => unit_cell = Some(uc),
                    Err(e) => {
                        return Err(format!("line {}: {}", line_num, e));
                    }
                }
                continue;
            }
            // Skip everything else if past model 1
            if past_model_1 {
                continue;
            }
        }

        // ATOM / HETATM
        if line.starts_with("ATOM  ") || line.starts_with("ATOM\t") {
            match parse_atom_line(line, false) {
                Ok(atom) => raw_atoms.push(atom),
                Err(e) => {
                    return Err(format!("line {}: {}", line_num, e));
                }
            }
            continue;
        }

        if line.starts_with("HETATM") {
            match parse_atom_line(line, true) {
                Ok(atom) => raw_atoms.push(atom),
                Err(e) => {
                    return Err(format!("line {}: {}", line_num, e));
                }
            }
            continue;
        }

        // CONECT
        if line.starts_with("CONECT") {
            match parse_conect_line(line) {
                Ok(record) => conect_records.push(record),
                Err(e) => {
                    return Err(format!("line {}: {}", line_num, e));
                }
            }
            continue;
        }

        // CRYST1
        if line.starts_with("CRYST1") && unit_cell.is_none() {
            match parse_cryst1_line(line) {
                Ok(uc) => unit_cell = Some(uc),
                Err(e) => {
                    return Err(format!("line {}: {}", line_num, e));
                }
            }
            continue;
        }

        // END terminates parsing entirely
        if line.starts_with("END") && !line.starts_with("ENDMDL") {
            break;
        }

        // TER, REMARK, TITLE, HEADER, etc. are silently skipped.
    }

    // -- Resolve alternate conformations --
    let resolved = resolve_alt_locs(raw_atoms);

    // -- Convert to PdbAtom --
    let atoms: Vec<PdbAtom> = resolved.iter().map(raw_to_pdb_atom).collect();

    // -- Build serial-to-index map --
    let mut serial_to_index: HashMap<usize, usize> = HashMap::with_capacity(atoms.len());
    for (idx, atom) in atoms.iter().enumerate() {
        // If duplicate serials exist, the last one wins (unusual but safe).
        serial_to_index.insert(atom.serial, idx);
    }

    // -- Build CONECT bonds --
    let conect_bonds = build_conect_bonds(&conect_records, &serial_to_index);

    // -- Build residue and chain hierarchy --
    let (residues, chains) = build_hierarchy(&atoms);

    Ok(PdbStructure {
        atoms,
        residues,
        chains,
        conect_bonds,
        unit_cell,
    })
}

/// Parse a PDB file from a filesystem path.
///
/// Reads the entire file into memory and delegates to [`parse_pdb`].
///
/// # Arguments
/// * `path` - Path to the PDB file
///
/// # Returns
/// * `Ok(PdbStructure)` - Parsed structure
/// * `Err(String)` - Error message (includes I/O errors)
pub fn parse_pdb_file(path: &Path) -> Result<PdbStructure, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("failed to read PDB file '{}': {}", path.display(), e))?;
    parse_pdb(&content)
}

// ============================================================================
// Convenience Methods
// ============================================================================

impl PdbStructure {
    /// Return the number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Return the number of residues.
    pub fn n_residues(&self) -> usize {
        self.residues.len()
    }

    /// Return the number of chains.
    pub fn n_chains(&self) -> usize {
        self.chains.len()
    }

    /// Return a sorted, deduplicated list of chain identifiers present in the
    /// structure.
    pub fn chain_ids(&self) -> Vec<char> {
        self.chains.iter().map(|c| c.chain_id).collect()
    }

    /// Get all atoms belonging to a specific chain.
    pub fn atoms_in_chain(&self, chain_id: char) -> Vec<&PdbAtom> {
        self.atoms
            .iter()
            .filter(|a| a.chain_id == chain_id)
            .collect()
    }

    /// Get the atoms for a specific residue by residue index (0-based).
    pub fn atoms_in_residue(&self, residue_idx: usize) -> Option<Vec<&PdbAtom>> {
        self.residues.get(residue_idx).map(|res| {
            res.atom_indices
                .iter()
                .filter_map(|&i| self.atoms.get(i))
                .collect()
        })
    }

    /// Collect all unique residue names present in the structure.
    pub fn residue_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .residues
            .iter()
            .map(|r| r.name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        names
    }
}
