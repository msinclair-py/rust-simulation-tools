//! Parser for AMBER force field parameter `.dat` files.
//!
//! This module reads AMBER parm parameter files such as `parm19.dat` and
//! `gaff2.dat`. These files use a fixed-width, section-based format with
//! blank lines as section delimiters.
//!
//! # Sections (in order)
//!
//! 1. **Title** -- single comment/title line
//! 2. **MASS** -- atom type definitions (until blank line)
//! 3. **Hydrophilic atom types** -- one or two lines, then blank line
//! 4. **BOND** -- bond stretching parameters (until blank line)
//! 5. **ANGLE** -- angle bending parameters (until blank line)
//! 6. **DIHEDRAL** -- proper torsion parameters (until blank line)
//! 7. **IMPROPER** -- improper torsion parameters (until blank line)
//! 8. **H-bond 10-12** -- (skipped) until blank line
//! 9. **Nonbond equivalences** -- groups of atom types sharing vdW params
//! 10. **MOD4/NONBON** -- Lennard-Jones parameters (until `END`)

use super::parameters::{
    AngleParam, AtomType, BondParam, DihedralParam, DihedralTerm, ForceFieldParams, ImproperParam,
    LjParam, NbEquiv,
};

use crate::forcefield::atom_types::element_from_type;

/// Parse the complete contents of an AMBER parm `.dat` file into a
/// [`ForceFieldParams`] structure.
///
/// The input should be the full text content of a file such as `parm19.dat`
/// or `gaff2.dat`. All sections are parsed sequentially; blank lines act as
/// section delimiters.
///
/// # Errors
///
/// Returns `Err(String)` if a required section is missing, a line cannot be
/// parsed, or the file structure does not match the expected format.
pub fn parse_parm_dat(content: &str) -> Result<ForceFieldParams, String> {
    let mut params = ForceFieldParams::new();
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return Err("Empty parm .dat file".to_string());
    }

    // Index tracks our position through the file.
    let mut idx = 0;

    // --- Section 1: Title line ---
    // Skip the title line.
    idx += 1;

    // --- Section 2: MASS (atom types) ---
    idx = parse_mass_section(&lines, idx, &mut params)?;

    // --- Section 3: Hydrophilic atom types ---
    // One or two lines of space-separated atom type names (no `-` separators).
    // The bond section starts immediately after (no blank line between them).
    // We skip lines until we see a line containing `-` (bond format).
    idx = skip_hydrophilic_lines(&lines, idx);

    // --- Section 4: BOND parameters ---
    idx = parse_bond_section(&lines, idx, &mut params)?;

    // --- Section 5: ANGLE parameters ---
    idx = parse_angle_section(&lines, idx, &mut params)?;

    // --- Section 6: DIHEDRAL parameters ---
    idx = parse_dihedral_section(&lines, idx, &mut params)?;

    // --- Section 7: IMPROPER parameters ---
    idx = parse_improper_section(&lines, idx, &mut params)?;

    // --- Section 8: H-bond 10-12 (skip) ---
    // This section may or may not be present. If the next non-blank content
    // looks like an H-bond header or numeric pairs, skip it. Otherwise we
    // are already past it.
    idx = skip_hbond_section(&lines, idx);

    // --- Section 9: Nonbond equivalences ---
    idx = parse_nb_equiv_section(&lines, idx, &mut params);

    // --- Section 10: MOD4/NONBON (LJ parameters) ---
    parse_lj_section(&lines, idx, &mut params)?;

    Ok(params)
}

// ---------------------------------------------------------------------------
// Section parsers
// ---------------------------------------------------------------------------

/// Parse the MASS section (atom type definitions).
///
/// Each line has the format:
///   `atom_type(2 chars)  mass(float)  polarizability(float, optional)  comment`
///
/// Returns the index of the first line after the terminating blank line.
fn parse_mass_section(
    lines: &[&str],
    start: usize,
    params: &mut ForceFieldParams,
) -> Result<usize, String> {
    let mut idx = start;
    while idx < lines.len() {
        let line = lines[idx];
        if is_blank(line) {
            return Ok(idx + 1);
        }

        // Atom type occupies the first 2 characters; may be followed by spaces.
        let atom_type = line.get(..2).unwrap_or("").trim();
        if atom_type.is_empty() {
            idx += 1;
            continue;
        }

        let rest = line.get(2..).unwrap_or("").trim();

        // Split off the comment (anything after ! or after the numeric columns).
        // The numeric portion: mass, then optional polarizability.
        let tokens: Vec<&str> = rest.split_whitespace().collect();
        if tokens.is_empty() {
            idx += 1;
            continue;
        }

        let mass: f64 = tokens[0]
            .parse()
            .map_err(|e| format!("Bad mass on line {}: {}", idx + 1, e))?;

        // Polarizability is optional -- if the second token parses as a float
        // and is not a comment, treat it as polarizability.
        let polarizability = if tokens.len() > 1 {
            tokens[1].parse::<f64>().unwrap_or(0.0)
        } else {
            0.0
        };

        let element = element_from_type(atom_type).unwrap_or("").to_string();

        params.atom_types.insert(
            atom_type.to_string(),
            AtomType {
                name: atom_type.to_string(),
                element,
                mass,
                polarizability,
                hybridization: String::new(),
            },
        );

        idx += 1;
    }

    Err("Unexpected end of file in MASS section".to_string())
}

/// Parse the BOND section.
///
/// Line format: `type1-type2  force_constant  eq_length  comment`
/// Each atom type field is 2 chars wide, separated by `-`.
fn parse_bond_section(
    lines: &[&str],
    start: usize,
    params: &mut ForceFieldParams,
) -> Result<usize, String> {
    let mut idx = start;
    while idx < lines.len() {
        let line = lines[idx];
        if is_blank(line) {
            return Ok(idx + 1);
        }

        let (t1, t2) = parse_two_types(line)
            .ok_or_else(|| format!("Bad bond type spec on line {}: {}", idx + 1, line))?;

        let rest = line.get(5..).unwrap_or("").trim();
        let tokens: Vec<&str> = rest.split_whitespace().collect();
        if tokens.len() < 2 {
            return Err(format!(
                "Insufficient bond parameters on line {}: {}",
                idx + 1,
                line
            ));
        }

        let force_constant: f64 = tokens[0]
            .parse()
            .map_err(|e| format!("Bad bond force constant on line {}: {}", idx + 1, e))?;
        let eq_length: f64 = tokens[1]
            .parse()
            .map_err(|e| format!("Bad bond eq_length on line {}: {}", idx + 1, e))?;

        params.bond_params.push(BondParam {
            type1: t1,
            type2: t2,
            force_constant,
            eq_length,
        });

        idx += 1;
    }

    Err("Unexpected end of file in BOND section".to_string())
}

/// Parse the ANGLE section.
///
/// Line format: `type1-type2-type3  force_constant  eq_angle(degrees)  comment`
fn parse_angle_section(
    lines: &[&str],
    start: usize,
    params: &mut ForceFieldParams,
) -> Result<usize, String> {
    let mut idx = start;
    while idx < lines.len() {
        let line = lines[idx];
        if is_blank(line) {
            return Ok(idx + 1);
        }

        let (t1, t2, t3) = parse_three_types(line)
            .ok_or_else(|| format!("Bad angle type spec on line {}: {}", idx + 1, line))?;

        let rest = line.get(8..).unwrap_or("").trim();
        let tokens: Vec<&str> = rest.split_whitespace().collect();
        if tokens.len() < 2 {
            return Err(format!(
                "Insufficient angle parameters on line {}: {}",
                idx + 1,
                line
            ));
        }

        let force_constant: f64 = tokens[0]
            .parse()
            .map_err(|e| format!("Bad angle force constant on line {}: {}", idx + 1, e))?;
        let eq_angle: f64 = tokens[1]
            .parse()
            .map_err(|e| format!("Bad angle eq_angle on line {}: {}", idx + 1, e))?;

        params.angle_params.push(AngleParam {
            type1: t1,
            type2: t2,
            type3: t3,
            force_constant,
            eq_angle,
        });

        idx += 1;
    }

    Err("Unexpected end of file in ANGLE section".to_string())
}

/// Parse the DIHEDRAL section.
///
/// Line format:
///   `type1-type2-type3-type4  IDIVF  PK(barrier/2)  PHASE(degrees)  PN(periodicity)`
///
/// When the periodicity (PN) is negative, more Fourier terms follow for the
/// same dihedral on subsequent lines. All terms are collected into a single
/// [`DihedralParam`] entry.
fn parse_dihedral_section(
    lines: &[&str],
    start: usize,
    params: &mut ForceFieldParams,
) -> Result<usize, String> {
    let mut idx = start;
    while idx < lines.len() {
        let line = lines[idx];
        if is_blank(line) {
            return Ok(idx + 1);
        }

        let (t1, t2, t3, t4) = parse_four_types(line)
            .ok_or_else(|| format!("Bad dihedral type spec on line {}: {}", idx + 1, line))?;

        let rest = line.get(11..).unwrap_or("").trim();
        let tokens: Vec<&str> = rest.split_whitespace().collect();
        if tokens.len() < 4 {
            return Err(format!(
                "Insufficient dihedral parameters on line {}: {}",
                idx + 1,
                line
            ));
        }

        let divider: f64 = tokens[0]
            .parse()
            .map_err(|e| format!("Bad dihedral IDIVF on line {}: {}", idx + 1, e))?;
        let pk: f64 = tokens[1]
            .parse()
            .map_err(|e| format!("Bad dihedral PK on line {}: {}", idx + 1, e))?;
        let phase: f64 = tokens[2]
            .parse()
            .map_err(|e| format!("Bad dihedral PHASE on line {}: {}", idx + 1, e))?;
        let pn: f64 = tokens[3]
            .trim_end_matches('.')
            .parse()
            .map_err(|e| format!("Bad dihedral PN on line {}: {}", idx + 1, e))?;

        let mut terms = vec![DihedralTerm {
            force_constant: pk,
            periodicity: pn.abs(),
            phase,
        }];

        idx += 1;

        // If PN was negative, gather continuation lines.
        if pn < 0.0 {
            while idx < lines.len() {
                let cline = lines[idx];
                if is_blank(cline) {
                    break;
                }

                // Continuation lines repeat the type specification.
                let c_rest = cline.get(11..).unwrap_or("").trim();
                let c_tokens: Vec<&str> = c_rest.split_whitespace().collect();
                if c_tokens.len() < 4 {
                    break;
                }

                let c_pk: f64 = match c_tokens[1].parse() {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let c_phase: f64 = match c_tokens[2].parse() {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let c_pn: f64 = match c_tokens[3].trim_end_matches('.').parse() {
                    Ok(v) => v,
                    Err(_) => break,
                };

                terms.push(DihedralTerm {
                    force_constant: c_pk,
                    periodicity: c_pn.abs(),
                    phase: c_phase,
                });

                idx += 1;

                // Positive PN means this is the last term for this dihedral.
                if c_pn > 0.0 {
                    break;
                }
            }
        }

        params.dihedral_params.push(DihedralParam {
            type1: t1,
            type2: t2,
            type3: t3,
            type4: t4,
            divider,
            terms,
        });
    }

    Err("Unexpected end of file in DIHEDRAL section".to_string())
}

/// Parse the IMPROPER dihedral section.
///
/// Line format:
///   `type1-type2-type3-type4  PK  PHASE(degrees)  PN(periodicity)`
///
/// In AMBER convention, `type3` is always the central (sp2) atom.
fn parse_improper_section(
    lines: &[&str],
    start: usize,
    params: &mut ForceFieldParams,
) -> Result<usize, String> {
    let mut idx = start;
    while idx < lines.len() {
        let line = lines[idx];
        if is_blank(line) {
            return Ok(idx + 1);
        }

        let (t1, t2, t3, t4) = parse_four_types(line)
            .ok_or_else(|| format!("Bad improper type spec on line {}: {}", idx + 1, line))?;

        let rest = line.get(11..).unwrap_or("").trim();
        let tokens: Vec<&str> = rest.split_whitespace().collect();
        if tokens.len() < 3 {
            return Err(format!(
                "Insufficient improper parameters on line {}: {}",
                idx + 1,
                line
            ));
        }

        let force_constant: f64 = tokens[0]
            .parse()
            .map_err(|e| format!("Bad improper PK on line {}: {}", idx + 1, e))?;
        let phase: f64 = tokens[1]
            .parse()
            .map_err(|e| format!("Bad improper PHASE on line {}: {}", idx + 1, e))?;
        let periodicity: f64 = tokens[2]
            .trim_end_matches('.')
            .parse()
            .map_err(|e| format!("Bad improper PN on line {}: {}", idx + 1, e))?;

        params.improper_params.push(ImproperParam {
            type1: t1,
            type2: t2,
            type3: t3,
            type4: t4,
            force_constant,
            periodicity,
            phase,
        });

        idx += 1;
    }

    Err("Unexpected end of file in IMPROPER section".to_string())
}

/// Skip the H-bond 10-12 section, if present.
///
/// This section is identified by a header line followed by numeric pairs,
/// or it may simply be absent. We detect it by checking whether the next
/// non-blank content resembles LJ-type entries or alphabetic atom-type
/// equivalences. If neither the `MOD4` header nor atom-type equivalence
/// lines appear, we assume we are in the H-bond section and skip to the
/// next blank line.
fn skip_hbond_section(lines: &[&str], start: usize) -> usize {
    let mut idx = start;

    // First, consume any leading blank lines.
    while idx < lines.len() && is_blank(lines[idx]) {
        idx += 1;
    }

    if idx >= lines.len() {
        return idx;
    }

    // Peek at the first non-blank line. If it starts with `MOD4` or looks
    // like an atom-type equivalence line (starts with letter and contains
    // only short whitespace-separated alphabetic tokens), there is no
    // H-bond section.
    let peek = lines[idx].trim();
    if peek.starts_with("MOD4") || looks_like_equiv_line(peek) {
        return idx;
    }

    // Otherwise skip to the next blank line (H-bond section content).
    while idx < lines.len() && !is_blank(lines[idx]) {
        idx += 1;
    }
    // Advance past the blank line.
    if idx < lines.len() {
        idx += 1;
    }
    idx
}

/// Parse the nonbond equivalence section.
///
/// Each line contains a list of atom types that share the same vdW
/// parameters. The section ends at a blank line or when the `MOD4` header
/// is reached.
fn parse_nb_equiv_section(lines: &[&str], start: usize, params: &mut ForceFieldParams) -> usize {
    let mut idx = start;
    while idx < lines.len() {
        let line = lines[idx];
        if is_blank(line) {
            idx += 1;
            // Possibly more blank lines before MOD4.
            continue;
        }
        let trimmed = line.trim();
        if trimmed.starts_with("MOD4") {
            break;
        }
        // This should be a line of space-separated atom type names.
        let types: Vec<String> = trimmed.split_whitespace().map(String::from).collect();
        if types.len() > 1 {
            params.nb_equivalences.push(NbEquiv { types });
        }
        idx += 1;
    }
    idx
}

/// Parse the MOD4/NONBON (Lennard-Jones) section.
///
/// Starts with a `MOD4      RE` header line. Each subsequent data line has
/// the format:
///   `  atom_type  Rmin/2(A)  epsilon(kcal/mol)  comment`
///
/// The section ends at the `END` keyword.
fn parse_lj_section(
    lines: &[&str],
    start: usize,
    params: &mut ForceFieldParams,
) -> Result<usize, String> {
    let mut idx = start;

    // Find the MOD4 header.
    while idx < lines.len() {
        if lines[idx].trim().starts_with("MOD4") {
            idx += 1;
            break;
        }
        idx += 1;
    }

    while idx < lines.len() {
        let line = lines[idx];
        let trimmed = line.trim();

        if trimmed == "END" {
            return Ok(idx + 1);
        }

        if trimmed.is_empty() {
            idx += 1;
            continue;
        }

        // Strip inline comments starting with `!`.
        let data = if let Some(pos) = trimmed.find('!') {
            trimmed[..pos].trim()
        } else {
            trimmed
        };

        let tokens: Vec<&str> = data.split_whitespace().collect();
        if tokens.len() < 3 {
            idx += 1;
            continue;
        }

        let atom_type = tokens[0];
        let radius: f64 = tokens[1]
            .parse()
            .map_err(|e| format!("Bad LJ radius on line {}: {}", idx + 1, e))?;
        let well_depth: f64 = tokens[2]
            .parse()
            .map_err(|e| format!("Bad LJ well_depth on line {}: {}", idx + 1, e))?;

        params.lj_params.insert(
            atom_type.to_string(),
            LjParam {
                atom_type: atom_type.to_string(),
                radius,
                well_depth,
            },
        );

        idx += 1;
    }

    // If we reach the end without seeing END, still return what we have.
    // Some truncated files may not have a trailing END.
    Ok(idx)
}

// ---------------------------------------------------------------------------
// Atom type field parsers
// ---------------------------------------------------------------------------

/// Parse two atom types separated by `-` from the first 5 characters.
///
/// The format is `XX-YY` where each type is 2 characters wide and the
/// separator `-` sits at position 2.
fn parse_two_types(line: &str) -> Option<(String, String)> {
    if line.len() < 5 {
        return None;
    }
    let t1 = line.get(..2)?.trim().to_string();
    let t2 = line.get(3..5)?.trim().to_string();
    if t1.is_empty() || t2.is_empty() {
        return None;
    }
    Some((t1, t2))
}

/// Parse three atom types separated by `-` from the first 8 characters.
///
/// Format: `XX-YY-ZZ`
fn parse_three_types(line: &str) -> Option<(String, String, String)> {
    if line.len() < 8 {
        return None;
    }
    let t1 = line.get(..2)?.trim().to_string();
    let t2 = line.get(3..5)?.trim().to_string();
    let t3 = line.get(6..8)?.trim().to_string();
    if t1.is_empty() || t2.is_empty() || t3.is_empty() {
        return None;
    }
    Some((t1, t2, t3))
}

/// Parse four atom types separated by `-` from the first 11 characters.
///
/// Format: `XX-YY-ZZ-WW`
fn parse_four_types(line: &str) -> Option<(String, String, String, String)> {
    if line.len() < 11 {
        return None;
    }
    let t1 = line.get(..2)?.trim().to_string();
    let t2 = line.get(3..5)?.trim().to_string();
    let t3 = line.get(6..8)?.trim().to_string();
    let t4 = line.get(9..11)?.trim().to_string();
    if t1.is_empty() || t2.is_empty() || t3.is_empty() || t4.is_empty() {
        return None;
    }
    Some((t1, t2, t3, t4))
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Returns `true` if a line is blank (empty or contains only whitespace).
fn is_blank(line: &str) -> bool {
    line.trim().is_empty()
}

/// Skip the hydrophilic atom type lines that appear between the MASS and
/// BOND sections.
///
/// These lines contain space-separated atom type names (e.g.,
/// `"C   H   HO  N   NA  ..."`). The bond section starts on the first line
/// that contains a `-` character within the first 5 characters (the atom
/// type separator in bond format `XX-YY`).
fn skip_hydrophilic_lines(lines: &[&str], start: usize) -> usize {
    let mut idx = start;
    while idx < lines.len() {
        let line = lines[idx];
        // A bond line has the format "XX-YY ..." with `-` at position 2.
        if line.len() >= 5 && line.as_bytes().get(2) == Some(&b'-') {
            return idx;
        }
        idx += 1;
    }
    idx
}

/// Heuristic: does a line look like an NB equivalence line?
///
/// Equivalence lines consist of short, all-alphabetic (or `*`-containing)
/// atom type tokens with no `-` separators and no numeric values.
fn looks_like_equiv_line(line: &str) -> bool {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.is_empty() {
        return false;
    }
    // Must have multiple tokens (atom types grouped together).
    if tokens.len() < 2 {
        return false;
    }
    // Every token should be a short atom type name (letters, digits, `*`, `+`)
    // with no `-` separators and must start with a letter.
    tokens.iter().all(|t| {
        t.len() <= 3
            && !t.contains('-')
            && t.starts_with(|c: char| c.is_ascii_alphabetic())
            && t.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '*' || c == '+')
    })
}
