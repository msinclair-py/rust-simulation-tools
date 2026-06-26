//! Parser for AMBER force field modification (frcmod) files.
//!
//! The frcmod format contains sections that override or add to a base AMBER
//! parameter set. Unlike the full parm `.dat` format, frcmod files use explicit
//! section headers (`MASS`, `BOND`, `ANGL`, `DIHE`, `IMPR`, `NONBON`) and omit
//! sections that are not modified (e.g., hydrophilic atoms, H-bond 10-12, and
//! non-bonded equivalences).
//!
//! Typical frcmod files include `frcmod.ff19SB` (amino acid backbone parameters),
//! `frcmod.opc` (OPC water model), and `frcmod.ionslm_126_opc` (ion parameters).

use super::parameters::{
    AngleParam, AtomType, BondParam, DihedralParam, DihedralTerm, ForceFieldParams, ImproperParam,
    LjParam,
};

/// Sections recognized in an frcmod file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Title,
    Mass,
    Bond,
    Angle,
    Dihedral,
    Improper,
    Nonbon,
    /// Sections we recognize but skip (e.g., CMAP).
    Unknown,
}

/// Attempt to classify a line as a section header.
///
/// Returns `Some(section)` if the trimmed line matches a known header keyword,
/// or `None` if it is not a section header.
fn classify_header(line: &str) -> Option<Section> {
    let trimmed = line.trim();
    match trimmed {
        "MASS" => Some(Section::Mass),
        "BOND" => Some(Section::Bond),
        s if s == "ANGL" || s == "ANGLE" => Some(Section::Angle),
        s if s == "DIHE" || s == "DIHEDRAL" => Some(Section::Dihedral),
        s if s == "IMPR" || s == "IMPROPER" => Some(Section::Improper),
        s if s == "NONBON" || s == "NONB" => Some(Section::Nonbon),
        s if s == "CMAP" || s.starts_with("CMAP") => Some(Section::Unknown),
        _ => None,
    }
}

/// Parse an AMBER frcmod file and return the contained force field parameters.
///
/// The returned [`ForceFieldParams`] is intended to be merged into a base
/// parameter set via [`ForceFieldParams::merge`].
///
/// # Errors
///
/// Returns `Err(String)` when a line within a recognized section cannot be
/// parsed (malformed numeric fields, missing required columns, etc.).
pub fn parse_frcmod(content: &str) -> Result<ForceFieldParams, String> {
    let mut params = ForceFieldParams::new();
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return Ok(params);
    }

    // First line is always the title -- skip it.
    let mut idx = 1;
    let mut current_section = Section::Title;

    // Accumulator for multi-term dihedrals. When a dihedral line has a negative
    // periodicity (PN < 0), additional terms follow for the same type combination.
    let mut pending_dihedral: Option<DihedralParam> = None;

    while idx < lines.len() {
        let line = lines[idx];

        // Check for a section header transition.
        if let Some(section) = classify_header(line) {
            // Flush any pending multi-term dihedral before switching sections.
            if let Some(dih) = pending_dihedral.take() {
                params.dihedral_params.push(dih);
            }
            current_section = section;
            idx += 1;
            continue;
        }

        // A blank line terminates the current section.
        if line.trim().is_empty() {
            if let Some(dih) = pending_dihedral.take() {
                params.dihedral_params.push(dih);
            }
            idx += 1;
            continue;
        }

        // Skip content in unknown sections (e.g., CMAP data).
        if current_section == Section::Unknown || current_section == Section::Title {
            idx += 1;
            continue;
        }

        match current_section {
            Section::Mass => parse_mass_line(line, &mut params)?,
            Section::Bond => parse_bond_line(line, &mut params)?,
            Section::Angle => parse_angle_line(line, &mut params)?,
            Section::Dihedral => {
                parse_dihedral_line(line, &mut pending_dihedral, &mut params)?;
            }
            Section::Improper => parse_improper_line(line, &mut params)?,
            Section::Nonbon => parse_nonbon_line(line, &mut params)?,
            Section::Title | Section::Unknown => unreachable!(),
        }

        idx += 1;
    }

    // Flush trailing multi-term dihedral if the file ends without a blank line.
    if let Some(dih) = pending_dihedral.take() {
        params.dihedral_params.push(dih);
    }

    Ok(params)
}

/// Parse a MASS section line.
///
/// Format: `atom_type  mass  [polarizability]  [comment]`
///
/// Polarizability is optional; if absent it defaults to 0.0. Anything after
/// the numeric fields (or a `!` comment marker) is treated as a comment and
/// used to populate the hybridization field heuristically.
fn parse_mass_line(line: &str, params: &mut ForceFieldParams) -> Result<(), String> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 2 {
        return Err(format!("MASS line too short: '{line}'"));
    }

    let name = tokens[0].to_string();
    let mass: f64 = tokens[1]
        .parse()
        .map_err(|e| format!("Bad mass value on MASS line '{line}': {e}"))?;

    // Try to read polarizability. It may be absent, or the next token may be a
    // comment indicator like '!' or a non-numeric string.
    let polarizability = if tokens.len() > 2 {
        let candidate = tokens[2].trim_start_matches('!');
        candidate.parse::<f64>().unwrap_or(0.0)
    } else {
        0.0
    };

    // Derive a rough hybridization string from the comment portion.
    let comment_start = tokens
        .iter()
        .position(|t| t.starts_with('!'))
        .unwrap_or(tokens.len());
    let hybridization = if comment_start < tokens.len() {
        tokens[comment_start..]
            .join(" ")
            .trim_start_matches('!')
            .trim()
            .to_string()
    } else if tokens.len() > 3 {
        // No '!' marker but there are trailing tokens past polarizability --
        // they are probably a description.
        tokens[3..].join(" ").to_string()
    } else {
        String::new()
    };

    let element = super::atom_types::element_from_type(&name)
        .unwrap_or("")
        .to_string();

    params.atom_types.insert(
        name.clone(),
        AtomType {
            name,
            element,
            mass,
            polarizability,
            hybridization,
        },
    );

    Ok(())
}

/// Parse a BOND section line.
///
/// Format: `type1-type2  force_constant  eq_length`
///
/// Atom types are exactly 2 characters each, separated by a dash. The type
/// field occupies columns 0..5 in the canonical format, but we parse robustly
/// by splitting on `-`.
fn parse_bond_line(line: &str, params: &mut ForceFieldParams) -> Result<(), String> {
    // The type pair is at the start, separated by '-'.
    let dash_pos = line
        .find('-')
        .ok_or_else(|| format!("No dash in BOND line: '{line}'"))?;

    let type1 = line[..dash_pos].trim().to_string();

    // Everything after the dash: the second type followed by numeric fields.
    let rest = &line[dash_pos + 1..];
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.len() < 3 {
        return Err(format!("BOND line too short after types: '{line}'"));
    }

    let type2 = tokens[0].to_string();
    let force_constant: f64 = tokens[1]
        .parse()
        .map_err(|e| format!("Bad force constant on BOND line '{line}': {e}"))?;
    let eq_length: f64 = tokens[2]
        .parse()
        .map_err(|e| format!("Bad eq_length on BOND line '{line}': {e}"))?;

    params.bond_params.push(BondParam {
        type1,
        type2,
        force_constant,
        eq_length,
    });

    Ok(())
}

/// Parse an ANGL (ANGLE) section line.
///
/// Format: `type1-type2-type3  force_constant  eq_angle`
fn parse_angle_line(line: &str, params: &mut ForceFieldParams) -> Result<(), String> {
    let types_and_rest = split_angle_types(line)?;
    let (type1, type2, type3, numeric_part) = types_and_rest;

    let tokens: Vec<&str> = numeric_part.split_whitespace().collect();
    if tokens.len() < 2 {
        return Err(format!("ANGL line missing numeric fields: '{line}'"));
    }

    let force_constant: f64 = tokens[0]
        .parse()
        .map_err(|e| format!("Bad force constant on ANGL line '{line}': {e}"))?;
    let eq_angle: f64 = tokens[1]
        .parse()
        .map_err(|e| format!("Bad eq_angle on ANGL line '{line}': {e}"))?;

    params.angle_params.push(AngleParam {
        type1,
        type2,
        type3,
        force_constant,
        eq_angle,
    });

    Ok(())
}

/// Split a three-type field (e.g., `N -C -2C`) into its component types and
/// the remaining numeric portion of the line.
///
/// The type field consists of three atom type names separated by dashes, where
/// each name may contain spaces around it (AMBER fixed-width formatting).
fn split_angle_types(line: &str) -> Result<(String, String, String, &str), String> {
    let parts: Vec<&str> = line.splitn(4, '-').collect();
    if parts.len() < 3 {
        return Err(format!("Cannot parse 3-type field from: '{line}'"));
    }

    let type1 = parts[0].trim().to_string();

    // The third part contains type3 followed by the numeric fields.
    let type2 = parts[1].trim().to_string();

    // parts[2] may be "type3  70.0  116.60" or parts may extend to parts[3]
    // if the third type itself contains no further dashes. In AMBER frcmod,
    // dashes only separate types, so we rejoin parts[2..] and split on
    // the first whitespace gap after the type name.
    let remainder = if parts.len() > 3 {
        // There was an extra dash -- rejoin (this shouldn't happen for angles
        // but be safe).
        let rejoined = parts[2..].join("-");
        let type3_end = rejoined
            .find(|c: char| c.is_whitespace())
            .ok_or_else(|| format!("Cannot find numeric fields on ANGL line: '{line}'"))?;
        let type3 = rejoined[..type3_end].trim().to_string();
        let numeric = &line[line.len() - (rejoined.len() - type3_end)..];
        return Ok((type1, type2, type3, numeric));
    } else {
        parts[2]
    };

    let type3_end = remainder
        .find(|c: char| c.is_ascii_digit() || c == '.')
        .ok_or_else(|| format!("Cannot find numeric fields on ANGL line: '{line}'"))?;

    // The type3 name is everything before the first digit, trimmed.
    let type3 = remainder[..type3_end].trim().to_string();
    // The numeric portion starts at that digit.
    let numeric_start = line.len() - remainder.len() + type3_end;
    let numeric = &line[numeric_start..];

    Ok((type1, type2, type3, numeric))
}

/// Split a four-type dihedral/improper field (e.g., `X -HW-OW-X `) into
/// component types and return the remaining numeric portion.
fn split_four_types(line: &str) -> Result<(String, String, String, String, &str), String> {
    let parts: Vec<&str> = line.splitn(5, '-').collect();
    if parts.len() < 4 {
        return Err(format!("Cannot parse 4-type field from: '{line}'"));
    }

    let type1 = parts[0].trim().to_string();
    let type2 = parts[1].trim().to_string();
    let type3 = parts[2].trim().to_string();

    // parts[3] contains type4 followed by numeric fields (or parts[4] if there
    // was a 5th dash, which shouldn't happen).
    let remainder = if parts.len() > 4 {
        let rejoined = parts[3..].join("-");
        let _ = rejoined; // fall through -- shouldn't happen for 4-type fields
        parts[3]
    } else {
        parts[3]
    };

    // Find the type4 name and where the numeric fields begin.
    let trimmed = remainder.trim_start();
    let first_space = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
    let type4 = trimmed[..first_space].to_string();

    let numeric_start_in_remainder = remainder.len() - trimmed.len() + first_space;
    let numeric_start = line.len() - remainder.len() + numeric_start_in_remainder;
    let numeric = &line[numeric_start..];

    Ok((type1, type2, type3, type4, numeric))
}

/// Parse a DIHE (DIHEDRAL) section line.
///
/// Format: `type1-type2-type3-type4  IDIVF  PK  PHASE  PN`
///
/// When PN is negative, additional terms follow for the same type combination.
/// The absolute value of PN is the periodicity.
fn parse_dihedral_line(
    line: &str,
    pending: &mut Option<DihedralParam>,
    params: &mut ForceFieldParams,
) -> Result<(), String> {
    let (type1, type2, type3, type4, numeric) = split_four_types(line)?;

    let tokens: Vec<&str> = numeric.split_whitespace().collect();
    if tokens.len() < 4 {
        return Err(format!("DIHE line missing numeric fields: '{line}'"));
    }

    let divider: f64 = tokens[0]
        .parse()
        .map_err(|e| format!("Bad IDIVF on DIHE line '{line}': {e}"))?;
    let pk: f64 = tokens[1]
        .parse()
        .map_err(|e| format!("Bad PK on DIHE line '{line}': {e}"))?;
    let phase: f64 = tokens[2]
        .parse()
        .map_err(|e| format!("Bad PHASE on DIHE line '{line}': {e}"))?;
    let pn_raw: f64 = tokens[3]
        .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .parse()
        .map_err(|e| format!("Bad PN on DIHE line '{line}': {e}"))?;

    let periodicity = pn_raw.abs();
    let more_terms = pn_raw < 0.0;

    let term = DihedralTerm {
        force_constant: pk,
        periodicity,
        phase,
    };

    // Determine if this line continues a pending multi-term dihedral.
    let continues_pending = pending.as_ref().is_some_and(|p| {
        p.type1 == type1 && p.type2 == type2 && p.type3 == type3 && p.type4 == type4
    });

    if continues_pending {
        // Append term to existing pending dihedral.
        let dih = pending.as_mut().unwrap();
        dih.terms.push(term);

        if !more_terms {
            // This is the last term -- flush.
            params.dihedral_params.push(pending.take().unwrap());
        }
    } else {
        // Flush any previously pending dihedral for a different type combo.
        if let Some(prev) = pending.take() {
            params.dihedral_params.push(prev);
        }

        let dih = DihedralParam {
            type1,
            type2,
            type3,
            type4,
            divider,
            terms: vec![term],
        };

        if more_terms {
            *pending = Some(dih);
        } else {
            params.dihedral_params.push(dih);
        }
    }

    Ok(())
}

/// Parse an IMPR (IMPROPER) section line.
///
/// Format: `type1-type2-type3-type4  PK  PHASE  PN`
///
/// Unlike proper dihedrals, impropers have no IDIVF field and are always
/// single-term in frcmod files.
fn parse_improper_line(line: &str, params: &mut ForceFieldParams) -> Result<(), String> {
    let (type1, type2, type3, type4, numeric) = split_four_types(line)?;

    let tokens: Vec<&str> = numeric.split_whitespace().collect();
    if tokens.len() < 3 {
        return Err(format!("IMPR line missing numeric fields: '{line}'"));
    }

    let force_constant: f64 = tokens[0]
        .parse()
        .map_err(|e| format!("Bad PK on IMPR line '{line}': {e}"))?;
    let phase: f64 = tokens[1]
        .parse()
        .map_err(|e| format!("Bad PHASE on IMPR line '{line}': {e}"))?;
    let periodicity: f64 = tokens[2]
        .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
        .parse()
        .map_err(|e| format!("Bad PN on IMPR line '{line}': {e}"))?;

    params.improper_params.push(ImproperParam {
        type1,
        type2,
        type3,
        type4,
        force_constant,
        phase,
        periodicity,
    });

    Ok(())
}

/// Parse a NONBON section line.
///
/// Format: `  atom_type  Rmin/2  epsilon  [comment]`
///
/// The atom type may include charge indicators (e.g., `Na+`, `Cl-`, `Fe3+`).
/// Fields are separated by arbitrary whitespace. Anything after the two
/// numeric columns is treated as an optional comment and discarded.
fn parse_nonbon_line(line: &str, params: &mut ForceFieldParams) -> Result<(), String> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 3 {
        return Err(format!("NONBON line too short: '{line}'"));
    }

    let atom_type = tokens[0].to_string();
    let radius: f64 = tokens[1]
        .parse()
        .map_err(|e| format!("Bad Rmin/2 on NONBON line '{line}': {e}"))?;
    let well_depth: f64 = tokens[2]
        .parse()
        .map_err(|e| format!("Bad epsilon on NONBON line '{line}': {e}"))?;

    params.lj_params.insert(
        atom_type.clone(),
        LjParam {
            atom_type,
            radius,
            well_depth,
        },
    );

    Ok(())
}
