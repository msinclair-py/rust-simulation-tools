//! AMBER residue template library (.lib / .off) parser.
//!
//! This module reads AMBER library files that define residue templates
//! with atoms, charges, bonds, positions, and inter-residue connectivity.
//! These files are used by the LEaP tool and define how residues are
//! constructed and linked in molecular systems.
//!
//! Supported files include `amino19.lib`, `aminont12.lib`, `aminoct12.lib`,
//! `atomic_ions.lib`, `solvents.lib`, `lipid21.lib`, and custom `.off` files.

use std::collections::HashMap;

/// An atom within a residue template.
#[derive(Debug, Clone)]
pub struct TemplateAtom {
    /// Atom name (e.g. "CA", "N", "O").
    pub name: String,
    /// AMBER atom type (e.g. "XC", "N", "O").
    pub atom_type: String,
    /// Partial charge in elementary charge units.
    pub charge: f64,
    /// Atomic number (1=H, 6=C, 7=N, 8=O, etc.; 0 for virtual sites/EP; -1 for unknown).
    pub element_number: i32,
    /// Which sub-residue this atom belongs to (1-based in the file, stored as-is here).
    pub residue_index: usize,
}

/// A bond between two atoms in a residue template.
#[derive(Debug, Clone)]
pub struct TemplateBond {
    /// First atom index (0-based).
    pub atom1: usize,
    /// Second atom index (0-based).
    pub atom2: usize,
}

/// A sub-residue entry within a unit.
///
/// Most amino acid units have a single residue entry. Solvent boxes (e.g.
/// `TIP3PBOX`, `OPCBOX`) contain hundreds or thousands of sub-residue entries
/// (one per water molecule).
#[derive(Debug, Clone)]
pub struct TemplateResidue {
    /// Residue name (e.g. "ALA", "WAT").
    pub name: String,
    /// Residue type character: 'p' = protein, 'w' = water, '?' = other.
    pub res_type: char,
    /// 0-based index of the first atom in the parent unit's atom list.
    pub start_atom: usize,
    /// Number of atoms in this sub-residue.
    pub atom_count: usize,
}

/// A complete residue template (one unit from an AMBER library file).
#[derive(Debug, Clone)]
pub struct ResidueTemplate {
    /// Unit name (e.g. "ALA", "TIP3PBOX", "Na+").
    pub name: String,
    /// Atoms in this unit.
    pub atoms: Vec<TemplateAtom>,
    /// Intra-unit bonds.
    pub bonds: Vec<TemplateBond>,
    /// Template coordinates in Angstroms, one `[x, y, z]` per atom.
    pub positions: Vec<[f64; 3]>,
    /// 0-based index of the head connection atom (e.g. N for amino acids).
    /// `None` if the unit has no head connection (value 0 in the file).
    pub head_atom: Option<usize>,
    /// 0-based index of the tail connection atom (e.g. C for amino acids).
    /// `None` if the unit has no tail connection (value 0 in the file).
    pub tail_atom: Option<usize>,
    /// Sub-residues within this unit.
    pub residues: Vec<TemplateResidue>,
    /// Periodic box dimensions `[x, y, z]` in Angstroms (present for solvent boxes).
    pub box_dimensions: Option<[f64; 3]>,
    /// Periodic box angle in radians (present for solvent boxes).
    pub box_angle: Option<f64>,
}

/// A collection of residue templates loaded from one or more AMBER library files.
#[derive(Debug, Clone, Default)]
pub struct ResidueLibrary {
    /// Templates keyed by unit name.
    pub templates: HashMap<String, ResidueTemplate>,
}

impl ResidueLibrary {
    /// Create an empty library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a template by name.
    pub fn get(&self, name: &str) -> Option<&ResidueTemplate> {
        self.templates.get(name)
    }

    /// Merge another library into this one, overwriting existing entries on collision.
    pub fn merge(&mut self, other: &ResidueLibrary) {
        for (name, template) in &other.templates {
            self.templates.insert(name.clone(), template.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse an AMBER `.lib` or `.off` file and return a [`ResidueLibrary`].
///
/// The parser extracts the following sections for each unit listed in the
/// `!!index` block:
///
/// - `atoms` -- atom names, types, charges, and element numbers
/// - `connectivity` -- intra-unit bonds
/// - `connect` -- head/tail inter-residue connection atoms
/// - `positions` -- template Cartesian coordinates
/// - `residues` -- sub-residue definitions
/// - `boundbox` -- periodic box information
///
/// All other sections (e.g. `hierarchy`, `velocities`, `atomspertinfo`) are
/// silently skipped.
pub fn parse_lib(content: &str) -> Result<ResidueLibrary, String> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(ResidueLibrary::new());
    }

    // --- Step 1: Parse the !!index block to discover unit names. ---
    let unit_names = parse_index(&lines)?;

    // --- Step 2: Build a section map for fast lookup. ---
    // Maps (unit_name, section_name) -> (start_line, end_line) where start_line
    // is the first data line after the header, and end_line is exclusive.
    let section_map = build_section_map(&lines);

    // --- Step 3: Parse each unit. ---
    let mut library = ResidueLibrary::new();
    for unit_name in &unit_names {
        let template = parse_unit(unit_name, &lines, &section_map)?;
        library.templates.insert(unit_name.clone(), template);
    }

    Ok(library)
}

/// Extract unit names from the `!!index array str` block at the top of the file.
fn parse_index(lines: &[&str]) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    let mut in_index = false;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.starts_with("!!index") {
            in_index = true;
            continue;
        }
        if in_index {
            // The index block ends at the first `!entry.` line or any other
            // non-quoted-string line.
            if trimmed.starts_with('!') || trimmed.is_empty() {
                break;
            }
            if let Some(name) = strip_quotes(trimmed) {
                names.push(name);
            }
        }
    }

    if names.is_empty() {
        return Err("No unit names found in !!index block".to_string());
    }
    Ok(names)
}

/// Scan all lines and record `(start, end)` line ranges for every section header
/// matching `!entry.NAME.unit.SECTION`.
///
/// Returns a map from `(unit_name, section_name)` to `(data_start, data_end)`
/// where `data_start` is the line immediately after the header and `data_end`
/// is the line of the next section header (or end of file).
fn build_section_map(lines: &[&str]) -> HashMap<(String, String), (usize, usize)> {
    // Collect all section header positions first.
    let mut headers: Vec<(usize, String, String)> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("!entry.") {
            // Format: !entry.NAME.unit.SECTION <type info>
            // NAME may contain special chars like '+' (e.g. "H3O+", "K+").
            // We split on ".unit." to reliably separate name from section.
            if let Some(dot_unit_pos) = rest.find(".unit.") {
                let unit_name = &rest[..dot_unit_pos];
                let after_unit = &rest[dot_unit_pos + 6..]; // skip ".unit."
                // Section name is everything up to the first space (the type declaration).
                let section_name = after_unit.split_whitespace().next().unwrap_or(after_unit);
                headers.push((i, unit_name.to_string(), section_name.to_string()));
            }
        }
    }

    let mut map = HashMap::new();
    for (idx, (line_num, unit_name, section_name)) in headers.iter().enumerate() {
        let data_start = line_num + 1;
        let data_end = if idx + 1 < headers.len() {
            headers[idx + 1].0
        } else {
            lines.len()
        };
        map.insert(
            (unit_name.clone(), section_name.clone()),
            (data_start, data_end),
        );
    }

    map
}

/// Retrieve the data lines for a given unit and section from the section map.
fn section_lines<'a>(
    unit_name: &str,
    section: &str,
    lines: &[&'a str],
    section_map: &HashMap<(String, String), (usize, usize)>,
) -> Vec<&'a str> {
    let key = (unit_name.to_string(), section.to_string());
    match section_map.get(&key) {
        Some(&(start, end)) => lines[start..end]
            .iter()
            .copied()
            .filter(|l| !l.trim().is_empty())
            .collect(),
        None => Vec::new(),
    }
}

/// Parse a single unit into a [`ResidueTemplate`].
fn parse_unit(
    unit_name: &str,
    lines: &[&str],
    section_map: &HashMap<(String, String), (usize, usize)>,
) -> Result<ResidueTemplate, String> {
    let atoms = parse_atoms(unit_name, lines, section_map)?;
    let bonds = parse_connectivity(unit_name, lines, section_map)?;
    let (head_atom, tail_atom) = parse_connect(unit_name, lines, section_map)?;
    let positions = parse_positions(unit_name, lines, section_map)?;
    let residues = parse_residues(unit_name, lines, section_map, atoms.len())?;
    let (box_dimensions, box_angle) = parse_boundbox(unit_name, lines, section_map)?;

    Ok(ResidueTemplate {
        name: unit_name.to_string(),
        atoms,
        bonds,
        positions,
        head_atom,
        tail_atom,
        residues,
        box_dimensions,
        box_angle,
    })
}

// ---------------------------------------------------------------------------
// Section parsers
// ---------------------------------------------------------------------------

/// Parse the `atoms` section.
///
/// Line format:
/// ```text
///  "N" "N" 0 1 131072 1 7 -0.415700
/// ```
/// Fields: name(quoted) type(quoted) typex(int) resx(int) flags(int) seq(int) elmnt(int) chg(dbl)
fn parse_atoms(
    unit_name: &str,
    lines: &[&str],
    section_map: &HashMap<(String, String), (usize, usize)>,
) -> Result<Vec<TemplateAtom>, String> {
    let data = section_lines(unit_name, "atoms", lines, section_map);
    let mut atoms = Vec::with_capacity(data.len());

    for line in data {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Tokenize respecting quoted strings.
        let tokens = tokenize_line(trimmed);
        if tokens.len() < 8 {
            return Err(format!(
                "atoms section for '{}': expected 8 fields, got {} in line: {}",
                unit_name,
                tokens.len(),
                trimmed
            ));
        }

        let name = strip_quotes_token(&tokens[0]);
        let atom_type = strip_quotes_token(&tokens[1]);
        // tokens[2] = typex (unused)
        let residue_index: usize = tokens[3]
            .parse()
            .map_err(|e| format!("atoms: bad resx '{}': {}", tokens[3], e))?;
        // tokens[4] = flags (unused)
        // tokens[5] = seq (unused)
        let element_number: i32 = tokens[6]
            .parse()
            .map_err(|e| format!("atoms: bad elmnt '{}': {}", tokens[6], e))?;
        let charge: f64 = tokens[7]
            .parse()
            .map_err(|e| format!("atoms: bad charge '{}': {}", tokens[7], e))?;

        atoms.push(TemplateAtom {
            name,
            atom_type,
            charge,
            element_number,
            residue_index,
        });
    }

    Ok(atoms)
}

/// Parse the `connectivity` section.
///
/// Line format:
/// ```text
///  1 2 1
/// ```
/// Fields: atom1x(1-based) atom2x(1-based) flags(int)
fn parse_connectivity(
    unit_name: &str,
    lines: &[&str],
    section_map: &HashMap<(String, String), (usize, usize)>,
) -> Result<Vec<TemplateBond>, String> {
    let data = section_lines(unit_name, "connectivity", lines, section_map);
    let mut bonds = Vec::with_capacity(data.len());

    for line in data {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() < 2 {
            return Err(format!(
                "connectivity section for '{}': expected at least 2 fields in line: {}",
                unit_name, trimmed
            ));
        }

        let a1: usize = tokens[0]
            .parse()
            .map_err(|e| format!("connectivity: bad atom1 '{}': {}", tokens[0], e))?;
        let a2: usize = tokens[1]
            .parse()
            .map_err(|e| format!("connectivity: bad atom2 '{}': {}", tokens[1], e))?;

        if a1 == 0 || a2 == 0 {
            return Err(format!(
                "connectivity section for '{}': unexpected 0-based index in line: {}",
                unit_name, trimmed
            ));
        }

        bonds.push(TemplateBond {
            atom1: a1 - 1,
            atom2: a2 - 1,
        });
    }

    Ok(bonds)
}

/// Parse the `connect` section.
///
/// Two values: head_atom_index(1-based) and tail_atom_index(1-based).
/// A value of 0 means no connection, mapped to `None`.
fn parse_connect(
    unit_name: &str,
    lines: &[&str],
    section_map: &HashMap<(String, String), (usize, usize)>,
) -> Result<(Option<usize>, Option<usize>), String> {
    let data = section_lines(unit_name, "connect", lines, section_map);
    if data.is_empty() {
        return Ok((None, None));
    }

    // The connect section has exactly two values, each on its own line.
    let mut values = Vec::new();
    for line in &data {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: usize = trimmed
            .parse()
            .map_err(|e| format!("connect section for '{}': bad value '{}': {}", unit_name, trimmed, e))?;
        values.push(v);
    }

    let head = values.first().copied().and_then(|v| {
        if v == 0 {
            None
        } else {
            Some(v - 1)
        }
    });
    let tail = values.get(1).copied().and_then(|v| {
        if v == 0 {
            None
        } else {
            Some(v - 1)
        }
    });

    Ok((head, tail))
}

/// Parse the `positions` section.
///
/// Line format:
/// ```text
///  3.325770 1.547909 -1.607204E-06
/// ```
fn parse_positions(
    unit_name: &str,
    lines: &[&str],
    section_map: &HashMap<(String, String), (usize, usize)>,
) -> Result<Vec<[f64; 3]>, String> {
    let data = section_lines(unit_name, "positions", lines, section_map);
    let mut positions = Vec::with_capacity(data.len());

    for line in data {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() < 3 {
            return Err(format!(
                "positions section for '{}': expected 3 values in line: {}",
                unit_name, trimmed
            ));
        }

        let x: f64 = tokens[0]
            .parse()
            .map_err(|e| format!("positions: bad x '{}': {}", tokens[0], e))?;
        let y: f64 = tokens[1]
            .parse()
            .map_err(|e| format!("positions: bad y '{}': {}", tokens[1], e))?;
        let z: f64 = tokens[2]
            .parse()
            .map_err(|e| format!("positions: bad z '{}': {}", tokens[2], e))?;

        positions.push([x, y, z]);
    }

    Ok(positions)
}

/// Parse the `residues` section.
///
/// Line format:
/// ```text
///  "ALA" 1 11 1 "p" 0
/// ```
/// Fields: name(quoted) seq(int) childseq(int) startatomx(1-based) restype(quoted) imagingx(int)
fn parse_residues(
    unit_name: &str,
    lines: &[&str],
    section_map: &HashMap<(String, String), (usize, usize)>,
    total_atoms: usize,
) -> Result<Vec<TemplateResidue>, String> {
    let data = section_lines(unit_name, "residues", lines, section_map);
    let mut residues = Vec::with_capacity(data.len());

    for line in data {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tokens = tokenize_line(trimmed);
        if tokens.len() < 5 {
            return Err(format!(
                "residues section for '{}': expected at least 5 fields in line: {}",
                unit_name, trimmed
            ));
        }

        let name = strip_quotes_token(&tokens[0]);
        // tokens[1] = seq (unused directly)
        // tokens[2] = childseq (unused directly)
        let start_atom_1based: usize = tokens[3]
            .parse()
            .map_err(|e| format!("residues: bad startatomx '{}': {}", tokens[3], e))?;
        let res_type_str = strip_quotes_token(&tokens[4]);
        let res_type = res_type_str.chars().next().unwrap_or('?');

        let start_atom = if start_atom_1based == 0 {
            0
        } else {
            start_atom_1based - 1
        };

        residues.push(TemplateResidue {
            name,
            res_type,
            start_atom,
            atom_count: 0, // filled in below
        });
    }

    // Compute atom_count for each sub-residue. Each sub-residue spans from its
    // start_atom to the next sub-residue's start_atom (or end of the atom list).
    let n = residues.len();
    for i in 0..n {
        let next_start = if i + 1 < n {
            residues[i + 1].start_atom
        } else {
            total_atoms
        };
        residues[i].atom_count = next_start.saturating_sub(residues[i].start_atom);
    }

    Ok(residues)
}

/// Parse the `boundbox` section.
///
/// Five values:
/// ```text
///  flag          (-1.0 = no box, 1.0 = has box)
///  angle         (in degrees or radians depending on AMBER version)
///  x_size
///  y_size
///  z_size
/// ```
fn parse_boundbox(
    unit_name: &str,
    lines: &[&str],
    section_map: &HashMap<(String, String), (usize, usize)>,
) -> Result<(Option<[f64; 3]>, Option<f64>), String> {
    let data = section_lines(unit_name, "boundbox", lines, section_map);
    if data.is_empty() {
        return Ok((None, None));
    }

    let mut values = Vec::new();
    for line in &data {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: f64 = trimmed
            .parse()
            .map_err(|e| format!("boundbox section for '{}': bad value '{}': {}", unit_name, trimmed, e))?;
        values.push(v);
    }

    if values.len() < 5 {
        return Ok((None, None));
    }

    let flag = values[0];
    if flag < 0.0 {
        // No periodic box.
        return Ok((None, None));
    }

    let angle = values[1];
    let dims = [values[2], values[3], values[4]];

    Ok((Some(dims), Some(angle)))
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Strip surrounding double quotes from a trimmed string.
/// Returns `None` if the string does not start with `"`.
fn strip_quotes(s: &str) -> Option<String> {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        Some(s[1..s.len() - 1].to_string())
    } else {
        None
    }
}

/// Strip quotes from a token, returning the unquoted content.
/// If the token is not quoted, returns it unchanged.
fn strip_quotes_token(s: &str) -> String {
    strip_quotes(s).unwrap_or_else(|| s.to_string())
}

/// Tokenize a line respecting double-quoted strings.
///
/// Quoted tokens are returned with their quotes intact so that callers can
/// distinguish them from unquoted tokens.  Use [`strip_quotes_token`] to
/// remove the quotes afterwards.
fn tokenize_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c == '"' {
            // Consume the entire quoted string including quotes.
            let mut tok = String::new();
            tok.push(chars.next().unwrap()); // opening quote
            loop {
                match chars.next() {
                    Some('"') => {
                        tok.push('"');
                        break;
                    }
                    Some(ch) => tok.push(ch),
                    None => break, // unterminated quote -- best effort
                }
            }
            tokens.push(tok);
        } else {
            // Unquoted token: consume until whitespace.
            let mut tok = String::new();
            while let Some(&ch) = chars.peek() {
                if ch.is_whitespace() {
                    break;
                }
                tok.push(ch);
                chars.next();
            }
            tokens.push(tok);
        }
    }

    tokens
}
