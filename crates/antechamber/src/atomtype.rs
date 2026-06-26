//! GAFF2 and BCC atom type assignment.
//!
//! Parses ATOMTYPE_GFF2.DEF and ATOMTYPE_BCC.DEF, then matches each atom
//! against the rules to assign atom types. First matching rule wins.

use crate::data;
use crate::molecule::{AcMolecule, AromaticType, BondOrder};

/// A parsed atom type rule from ATD lines.
#[derive(Debug, Clone)]
struct AtomTypeRule {
    /// The atom type name to assign (e.g., "ca", "c3").
    type_name: String,
    /// For BCC rules, the numeric type ID.
    type_id: u32,
    /// Required atomic number (0 = any).
    atomic_number: u8,
    /// Required connectivity/degree (0 = any).
    connectivity: u8,
    /// Required number of attached H (255 = any).
    n_hydrogens: u8,
    /// For H atoms: required number of electron-withdrawing groups on parent (255 = any).
    n_ew_groups: u8,
    /// Required aromatic types (empty = any).
    aromatic_types: Vec<AromaticType>,
    /// Required ring size (0 = any).
    ring_size: u8,
    /// Whether the atom must be in a ring.
    must_be_in_ring: bool,
    /// Required bond types (simplified).
    bond_constraints: Vec<String>,
    /// Required neighbor patterns (simplified).
    neighbor_constraints: Vec<String>,
}

/// Parse ATOMTYPE_GFF2.DEF into rules.
fn parse_gaff2_rules() -> Vec<AtomTypeRule> {
    parse_atomtype_def(data::ATOMTYPE_GFF2, false)
}

/// Parse ATOMTYPE_BCC.DEF into rules.
fn parse_bcc_rules() -> Vec<AtomTypeRule> {
    parse_atomtype_def(data::ATOMTYPE_BCC, true)
}

/// Parse an ATOMTYPE DEF file into rules.
///
/// In the GAFF2 DEF format, `&` at the end of a line is a TERMINATOR (end of rule).
/// Each `ATD ... &` line is a complete, independent rule.
fn parse_atomtype_def(content: &str, is_bcc: bool) -> Vec<AtomTypeRule> {
    let mut rules = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Skip comments and empty lines
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("//")
            || trimmed.starts_with("===")
            || trimmed.starts_with("---")
            || trimmed.starts_with("WILDATOM")
            || trimmed.starts_with(" f1")
        {
            continue;
        }

        if !trimmed.starts_with("ATD") {
            continue;
        }

        // Strip trailing & (line terminator) and parse
        let clean = if trimmed.ends_with('&') {
            trimmed[..trimmed.len() - 1].trim()
        } else {
            trimmed
        };

        if let Some(rule) = parse_atd_line(clean, is_bcc) {
            rules.push(rule);
        }
    }

    rules
}

/// Parse a single ATD line into a rule.
/// Format: ATD  type  f3  f4  f5  f6  f7  f8  f9
///   f3 = residue name or *
///   f4 = atomic number or *
///   f5 = connectivity or *
///   f6 = number of attached H or *
///   f7 = for H: number of electron-withdrawing groups on parent; else *
///   f8 = atomic property constraints [AR1, RG6, sb, db, ...] or *
///   f9 = neighbor constraints (XX(YY)) or &
fn parse_atd_line(line: &str, is_bcc: bool) -> Option<AtomTypeRule> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 || parts[0] != "ATD" {
        return None;
    }

    let type_name = parts[1].to_string();
    let type_id = if is_bcc {
        type_name.parse::<u32>().unwrap_or(0)
    } else {
        0
    };

    // f3: residue (skip, always *)
    // f4: element/atomic number
    let atomic_number = if parts.len() > 3 && parts[3] != "*" {
        parts[3].parse::<u8>().unwrap_or(0)
    } else {
        0
    };

    // f5: connectivity
    let connectivity = if parts.len() > 4 && parts[4] != "*" {
        parts[4].parse::<u8>().unwrap_or(0)
    } else {
        0
    };

    // f6: hydrogen count
    let n_hydrogens = if parts.len() > 5 && parts[5] != "*" {
        parts[5].parse::<u8>().unwrap_or(255)
    } else {
        255
    };

    // f7: for H atoms, number of electron-withdrawing groups on parent
    let n_ew_groups = if parts.len() > 6 && parts[6] != "*" {
        parts[6].parse::<u8>().unwrap_or(255)
    } else {
        255
    };

    // Parse remaining fields for bracket and paren tokens
    let mut aromatic_types = Vec::new();
    let mut ring_size: u8 = 0;
    let mut must_be_in_ring = false;
    let mut bond_constraints = Vec::new();
    let mut neighbor_constraints = Vec::new();

    // Start from field 7 (index 7) for bracket/paren tokens
    let start_idx = 7.min(parts.len());
    for &part in &parts[start_idx..] {
        if part == "*" || part == "&" {
            continue;
        }
        if part.starts_with('[') && part.ends_with(']') {
            let inner = &part[1..part.len() - 1];
            for token in inner.split(',') {
                let token = token.trim();
                if token.starts_with("AR") {
                    match token {
                        "AR1" => aromatic_types.push(AromaticType::Ar1),
                        "AR2" => aromatic_types.push(AromaticType::Ar2),
                        "AR3" => aromatic_types.push(AromaticType::Ar3),
                        "AR4" => aromatic_types.push(AromaticType::Ar4),
                        "AR5" => aromatic_types.push(AromaticType::Ar5),
                        "AR1.AR2" | "AR1.AR2.AR3" => {
                            aromatic_types.push(AromaticType::Ar1);
                            aromatic_types.push(AromaticType::Ar2);
                            if token.contains("AR3") {
                                aromatic_types.push(AromaticType::Ar3);
                            }
                        }
                        _ => {}
                    }
                } else if token.contains("RG") {
                    // Handles: "RG", "RG5", "RG6", "1RG6", etc.
                    // Extract the ring size digit(s) after "RG"
                    if let Some(pos) = token.find("RG") {
                        let after = &token[pos + 2..];
                        ring_size = after.parse().unwrap_or(0);
                    }
                    must_be_in_ring = true;
                } else if token == "NR" {
                    // Non-ring atom: explicitly not in a ring
                    // We don't enforce this currently but don't add to bond_constraints
                } else {
                    bond_constraints.push(token.to_string());
                }
            }
        } else if part.starts_with('(') {
            neighbor_constraints.push(part.to_string());
        }
    }

    Some(AtomTypeRule {
        type_name,
        type_id,
        atomic_number,
        connectivity,
        n_hydrogens,
        n_ew_groups,
        aromatic_types,
        ring_size,
        must_be_in_ring,
        bond_constraints,
        neighbor_constraints,
    })
}

/// Check if an atom matches a rule.
fn matches_rule(mol: &AcMolecule, atom_idx: usize, rule: &AtomTypeRule) -> bool {
    let atom = &mol.atoms[atom_idx];

    // Check atomic number
    if rule.atomic_number != 0 && atom.atomic_number != rule.atomic_number {
        return false;
    }

    // Check connectivity
    if rule.connectivity != 0 && atom.degree as u8 != rule.connectivity {
        return false;
    }

    // Check hydrogen count (f6)
    if rule.n_hydrogens != 255 && atom.n_hydrogens as u8 != rule.n_hydrogens {
        return false;
    }

    // Check electron-withdrawing group count (f7, for H atoms)
    if rule.n_ew_groups != 255 {
        if atom.atomic_number == 1 {
            // For H: count EW groups on the parent atom
            let n_ew = count_ew_groups_on_parent(mol, atom_idx);
            if n_ew != rule.n_ew_groups as usize {
                return false;
            }
        }
        // For non-H atoms, field 7 can have other meanings; skip for now
    }

    // Check aromatic type
    if !rule.aromatic_types.is_empty() && !rule.aromatic_types.contains(&atom.aromatic_type) {
        return false;
    }

    // Check ring size
    if rule.must_be_in_ring {
        if atom.ring_sizes.is_empty() {
            return false;
        }
        if rule.ring_size != 0 && !atom.ring_sizes.contains(&(rule.ring_size as usize)) {
            return false;
        }
    }

    // Check bond constraints from atom property field [db], [2sb], [sb,db], etc.
    if !rule.bond_constraints.is_empty() {
        if !check_atom_bond_constraints(mol, atom_idx, &rule.bond_constraints) {
            return false;
        }
    }

    // Neighbor constraints
    if !rule.neighbor_constraints.is_empty() {
        if !check_neighbor_constraints(mol, atom_idx, &rule.neighbor_constraints) {
            return false;
        }
    }

    true
}

/// Check bond type constraints from the atom property field (f8).
///
/// Constraints like `db` (has double bond), `2sb` (exactly 2 single bonds),
/// `sb,db` (parsed as separate constraints, each must be satisfied).
///
/// Bond type classification:
/// - `sb` (lowercase) = Single, Aromatic, Amide, Deloc (inclusive)
/// - `db` (lowercase) = Double, Aromatic (inclusive)
/// - `tb` = Triple
/// - `SB` (uppercase) = strict Single only
/// - `DB` (uppercase) = strict Double only
fn check_atom_bond_constraints(mol: &AcMolecule, atom_idx: usize, constraints: &[String]) -> bool {
    for constraint in constraints {
        // Parse optional leading count digit
        let mut chars = constraint.chars().peekable();
        let count_required: Option<usize> = if chars.peek().map_or(false, |c| c.is_ascii_digit()) {
            let d = chars.next().unwrap();
            Some(d.to_digit(10).unwrap() as usize)
        } else {
            None
        };
        let bond_type: String = chars.collect();

        // Count bonds of this type for the atom
        let actual_count = count_bond_type_for_atom(mol, atom_idx, &bond_type);

        if let Some(required) = count_required {
            if actual_count != required {
                return false;
            }
        } else {
            // No count specified: require at least one
            if actual_count == 0 {
                return false;
            }
        }
    }
    true
}

/// Count how many bonds of a given type an atom has.
fn count_bond_type_for_atom(mol: &AcMolecule, atom_idx: usize, bond_type: &str) -> usize {
    mol.bonds
        .iter()
        .filter(|b| b.atom1 == atom_idx || b.atom2 == atom_idx)
        .filter(|b| bond_matches_type(&b.order, bond_type))
        .count()
}

/// Check if a bond order matches a bond type constraint string.
fn bond_matches_type(order: &BondOrder, bond_type: &str) -> bool {
    match bond_type {
        // Lowercase (inclusive)
        "sb" => matches!(order, BondOrder::Single | BondOrder::Aromatic | BondOrder::Amide | BondOrder::Deloc),
        "db" => matches!(order, BondOrder::Double | BondOrder::Aromatic),
        "tb" => matches!(order, BondOrder::Triple),
        // Uppercase (strict)
        "SB" => matches!(order, BondOrder::Single),
        "DB" => matches!(order, BondOrder::Double),
        "TB" => matches!(order, BondOrder::Triple),
        "AB" => matches!(order, BondOrder::Aromatic),
        "DL" => matches!(order, BondOrder::Deloc),
        _ => false,
    }
}

/// Count electron-withdrawing atoms bonded to the parent of a hydrogen atom.
/// EW atoms: N, O, F, S, Cl, Br, I
fn count_ew_groups_on_parent(mol: &AcMolecule, h_idx: usize) -> usize {
    let h_atom = &mol.atoms[h_idx];
    if h_atom.neighbors.is_empty() {
        return 0;
    }
    let parent_idx = h_atom.neighbors[0];
    let parent = &mol.atoms[parent_idx];

    parent
        .neighbors
        .iter()
        .filter(|&&n| {
            let an = mol.atoms[n].atomic_number;
            // Electron-withdrawing: N(7), O(8), F(9), S(16), Cl(17), Br(35), I(53)
            matches!(an, 7 | 8 | 9 | 16 | 17 | 35 | 53)
        })
        .count()
}

/// Check neighbor constraints from GAFF2 DEF rules.
///
/// Constraint format: `(spec1,spec2,...)` where each spec is like:
/// - `N` = neighbor is nitrogen (any degree)
/// - `C4` = neighbor is carbon with degree 4 (sp3)
/// - `C3` = neighbor is carbon with degree 3 (sp2)
/// - `O1` = neighbor is oxygen with degree 1
/// - `XX` = any element
/// - `XX[AR1]` = any element with aromatic type AR1
/// - `C(N4)` = neighbor C that itself has an N4 neighbor (recursive)
///
/// Multiple comma-separated specs each require one matching neighbor.
fn check_neighbor_constraints(mol: &AcMolecule, atom_idx: usize, constraints: &[String]) -> bool {
    let atom = &mol.atoms[atom_idx];

    for constraint in constraints {
        let inner = constraint.trim_start_matches('(').trim_end_matches(')');
        if inner.is_empty() {
            continue;
        }

        // Split by comma for multiple neighbor requirements
        let specs: Vec<&str> = inner.split(',').collect();

        // Track which neighbors have been "used" to satisfy specs
        let mut used = vec![false; atom.neighbors.len()];

        for spec in &specs {
            let spec = spec.trim();
            if spec.is_empty() {
                continue;
            }

            // Parse the spec: element + optional degree + optional [aromatic] + optional (sub-constraint)
            let parsed = parse_neighbor_spec_full(spec);

            // Find a matching neighbor that hasn't been used
            let mut found = false;
            for (ni, &neighbor_idx) in atom.neighbors.iter().enumerate() {
                if used[ni] {
                    continue;
                }
                if neighbor_matches_spec(mol, atom_idx, neighbor_idx, &parsed) {
                    used[ni] = true;
                    found = true;
                    break;
                }
            }

            if !found {
                return false;
            }
        }
    }

    true
}

/// Parsed neighbor specification.
#[derive(Debug)]
struct NeighborSpec {
    /// Element symbol or wildcard (XX, XA, XB, XC, XD).
    element: String,
    /// Required degree (0 = any).
    degree: u8,
    /// Required aromatic type (None = any).
    aromatic_type: Option<AromaticType>,
    /// Sub-constraint: the neighbor must itself have a neighbor matching this.
    sub_constraint: Option<Box<NeighborSpec>>,
    /// Whether this neighbor must be in a ring.
    must_be_in_ring: bool,
    /// Required ring size (0 = any ring).
    ring_size: u8,
    /// Bond type constraint with predecessor (the bond between from_atom and this neighbor).
    /// e.g., "db'" means bond to predecessor must be double; "sb'" means single.
    pred_bond_type: Option<String>,
    /// Whether the pred bond constraint is negated ('' suffix).
    pred_bond_negated: bool,
    /// General bond inventory constraints on this neighbor (without ' suffix).
    bond_inventory: Vec<String>,
}

/// Parse a full neighbor spec like "C4", "N", "XX[AR1]", "C3[RG]", "N3[db']", "C(N4)".
fn parse_neighbor_spec_full(spec: &str) -> NeighborSpec {
    let mut element = String::new();
    let mut degree_str = String::new();
    let mut aromatic_type = None;
    let mut sub_constraint = None;
    let mut must_be_in_ring = false;
    let mut ring_size: u8 = 0;
    let mut pred_bond_type = None;
    let mut pred_bond_negated = false;
    let mut bond_inventory = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = spec.chars().collect();

    // Parse element name (uppercase letter possibly followed by lowercase)
    while i < chars.len() && chars[i] != '[' && chars[i] != '(' && !chars[i].is_ascii_digit() {
        element.push(chars[i]);
        i += 1;
    }

    // Parse optional degree digit
    while i < chars.len() && chars[i].is_ascii_digit() {
        degree_str.push(chars[i]);
        i += 1;
    }

    let degree = degree_str.parse::<u8>().unwrap_or(0);

    // Parse optional [aromatic/ring/bond] constraint
    if i < chars.len() && chars[i] == '[' {
        let bracket_start = i + 1;
        let mut bracket_end = bracket_start;
        while bracket_end < chars.len() && chars[bracket_end] != ']' {
            bracket_end += 1;
        }
        let bracket_content: String = chars[bracket_start..bracket_end].iter().collect();

        for token in bracket_content.split(',') {
            let token = token.trim();
            if token.starts_with("AR") {
                match token {
                    "AR1" => aromatic_type = Some(AromaticType::Ar1),
                    "AR2" => aromatic_type = Some(AromaticType::Ar2),
                    "AR3" => aromatic_type = Some(AromaticType::Ar3),
                    "AR1.AR2" | "AR1.AR2.AR3" => {
                        aromatic_type = Some(AromaticType::Ar1); // simplified
                    }
                    _ => {}
                }
            } else if token == "RG" {
                must_be_in_ring = true;
            } else if token.starts_with("RG") {
                must_be_in_ring = true;
                ring_size = token[2..].parse().unwrap_or(0);
            } else if token.ends_with("''") {
                // Negated predecessor bond constraint: e.g., "db''"
                let bt = token.trim_end_matches("''");
                pred_bond_type = Some(bt.to_string());
                pred_bond_negated = true;
            } else if token.ends_with('\'') {
                // Predecessor bond constraint: e.g., "db'", "sb'", "SB'"
                let bt = token.trim_end_matches('\'');
                pred_bond_type = Some(bt.to_string());
                pred_bond_negated = false;
            } else {
                // General bond inventory constraint: e.g., "db", "sb"
                bond_inventory.push(token.to_string());
            }
        }

        i = bracket_end + 1;
    }

    // Skip optional <a1> tag annotations (used for cross-ring constraints we don't support)
    if i < chars.len() && chars[i] == '<' {
        while i < chars.len() && chars[i] != '>' {
            i += 1;
        }
        if i < chars.len() {
            i += 1; // skip '>'
        }
    }

    // Parse optional (sub-constraint)
    if i < chars.len() && chars[i] == '(' {
        let paren_start = i + 1;
        // Find matching closing paren, handling nesting
        let mut depth = 1;
        let mut paren_end = paren_start;
        while paren_end < chars.len() && depth > 0 {
            if chars[paren_end] == '(' {
                depth += 1;
            }
            if chars[paren_end] == ')' {
                depth -= 1;
            }
            if depth > 0 {
                paren_end += 1;
            }
        }
        let sub_spec: String = chars[paren_start..paren_end].iter().collect();
        if !sub_spec.is_empty() {
            sub_constraint = Some(Box::new(parse_neighbor_spec_full(&sub_spec)));
        }
    }

    NeighborSpec {
        element,
        degree,
        aromatic_type,
        sub_constraint,
        must_be_in_ring,
        ring_size,
        pred_bond_type,
        pred_bond_negated,
        bond_inventory,
    }
}

/// Check if a neighbor atom matches a parsed spec.
///
/// `from_idx` is the atom that this neighbor is bonded to (the predecessor).
/// This is needed for bond-with-predecessor constraints like `[db']`.
fn neighbor_matches_spec(mol: &AcMolecule, from_idx: usize, neighbor_idx: usize, spec: &NeighborSpec) -> bool {
    let neighbor = &mol.atoms[neighbor_idx];

    // Check element
    if !element_matches(neighbor, &spec.element) {
        return false;
    }

    // Check degree
    if spec.degree > 0 && neighbor.degree as u8 != spec.degree {
        return false;
    }

    // Check aromatic type
    if let Some(ref ar) = spec.aromatic_type {
        if neighbor.aromatic_type != *ar {
            return false;
        }
    }

    // Check ring membership
    if spec.must_be_in_ring {
        if neighbor.ring_sizes.is_empty() {
            return false;
        }
        if spec.ring_size != 0 && !neighbor.ring_sizes.contains(&(spec.ring_size as usize)) {
            return false;
        }
    }

    // Check bond-with-predecessor constraint
    if let Some(ref bt) = spec.pred_bond_type {
        if let Some(bond) = mol.get_bond(from_idx, neighbor_idx) {
            let matches = bond_matches_type(&bond.order, bt);
            if spec.pred_bond_negated {
                if matches {
                    return false; // Must NOT have this bond type
                }
            } else if !matches {
                return false; // Must have this bond type
            }
        } else {
            return false; // No bond found
        }
    }

    // Check general bond inventory constraints on the neighbor
    for bt in &spec.bond_inventory {
        let count = count_bond_type_for_atom(mol, neighbor_idx, bt);
        if count == 0 {
            return false;
        }
    }

    // Check sub-constraint (recursive neighbor matching)
    if let Some(ref sub) = spec.sub_constraint {
        let has_matching_sub = neighbor
            .neighbors
            .iter()
            .any(|&sub_idx| neighbor_matches_spec(mol, neighbor_idx, sub_idx, sub));
        if !has_matching_sub {
            return false;
        }
    }

    true
}

/// Check if an atom matches an element specification.
fn element_matches(atom: &crate::molecule::AcAtom, element: &str) -> bool {
    match element {
        "XX" => true,
        "XA" => atom.atomic_number == 8 || atom.atomic_number == 16, // O, S
        "XB" => atom.atomic_number == 7 || atom.atomic_number == 15, // N, P
        "XC" => {
            // F, Cl, Br, I (halogens)
            matches!(atom.atomic_number, 9 | 17 | 35 | 53)
        }
        "XD" => atom.atomic_number == 16 || atom.atomic_number == 15, // S, P
        "C" => atom.atomic_number == 6,
        "N" => atom.atomic_number == 7,
        "O" => atom.atomic_number == 8,
        "S" => atom.atomic_number == 16,
        "P" => atom.atomic_number == 15,
        "H" => atom.atomic_number == 1,
        "F" => atom.atomic_number == 9,
        "Cl" => atom.atomic_number == 17,
        "Br" => atom.atomic_number == 35,
        "I" => atom.atomic_number == 53,
        _ => false,
    }
}

/// Assign GAFF2 atom types to all atoms in the molecule.
pub fn assign_gaff2_types(mol: &mut AcMolecule) -> Result<(), String> {
    let rules = parse_gaff2_rules();

    for i in 0..mol.atoms.len() {
        let mut assigned = false;

        for rule in &rules {
            if matches_rule(mol, i, rule) {
                mol.atoms[i].gaff2_type = rule.type_name.clone();
                assigned = true;
                break;
            }
        }

        if !assigned {
            // Default type based on element
            let default_type = match mol.atoms[i].atomic_number {
                1 => "ha",
                6 => "c3",
                7 => "n",
                8 => "o",
                9 => "f",
                15 => "p5",
                16 => "s",
                17 => "cl",
                35 => "br",
                53 => "i",
                _ => "du",
            };
            mol.atoms[i].gaff2_type = default_type.to_string();
        }
    }

    Ok(())
}

/// Assign BCC atom type indices for AM1-BCC correction lookup.
pub fn assign_bcc_types(mol: &mut AcMolecule) -> Result<(), String> {
    let rules = parse_bcc_rules();

    for i in 0..mol.atoms.len() {
        let mut assigned = false;

        for rule in &rules {
            if matches_rule(mol, i, rule) {
                mol.atoms[i].bcc_type = rule.type_id;
                assigned = true;
                break;
            }
        }

        if !assigned {
            // Default BCC type based on element
            mol.atoms[i].bcc_type = match mol.atoms[i].atomic_number {
                1 => 11,
                6 => 11,
                7 => 21,
                8 => 31,
                9 => 41,
                16 => 51,
                17 => 61,
                35 => 71,
                53 => 81,
                15 => 91,
                _ => 0,
            };
        }
    }

    Ok(())
}
