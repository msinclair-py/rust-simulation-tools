//! PDBx/mmCIF file parser with biological assembly support.
//!
//! Parses mmCIF files as defined by the Protein Data Bank (PDBx/mmCIF dictionary),
//! extracting atomic coordinates from the `_atom_site` category and optionally
//! constructing biological assemblies from `_pdbx_struct_assembly_gen` and
//! `_pdbx_struct_oper_list` symmetry operations.
//!
//! Only model 1 atoms are retained, and alternate location conflicts are resolved
//! by keeping the conformer with the highest occupancy (falling back to the first
//! encountered on tie).

use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;

// ============================================================================
// Data Structures
// ============================================================================

/// A single atom from mmCIF.
#[derive(Debug, Clone)]
pub struct CifAtom {
    /// Atom serial from `_atom_site.id`.
    pub id: usize,
    /// Atom name (`label_atom_id`).
    pub name: String,
    /// Alternate location indicator (`label_alt_id`, empty when `'.'`).
    pub alt_id: String,
    /// Residue name (`label_comp_id`).
    pub residue_name: String,
    /// Chain ID (`label_asym_id`).
    pub chain_id: String,
    /// Author chain ID (`auth_asym_id`).
    pub auth_chain_id: String,
    /// Residue sequence number (`auth_seq_id`).
    pub res_seq: i32,
    /// Insertion code (`pdbx_PDB_ins_code`, `'?'` means none).
    pub ins_code: String,
    /// Cartesian coordinates in Angstroms `[x, y, z]`.
    pub position: [f64; 3],
    /// Occupancy factor.
    pub occupancy: f64,
    /// Isotropic B-factor (`B_iso_or_equiv`).
    pub b_factor: f64,
    /// Element symbol (`type_symbol`).
    pub element: String,
    /// Record type: `ATOM` or `HETATM` (`group_PDB`).
    pub group: String,
    /// Model number (`pdbx_PDB_model_num`).
    pub model_num: i32,
}

/// A symmetry operation consisting of a 3x3 rotation matrix and a translation
/// vector, as read from `_pdbx_struct_oper_list`.
#[derive(Debug, Clone)]
pub struct SymOp {
    /// Operation ID (from `_pdbx_struct_oper_list.id`).
    pub id: String,
    /// 3x3 rotation matrix stored in row-major order.
    pub rotation: [[f64; 3]; 3],
    /// Translation vector in Angstroms.
    pub translation: [f64; 3],
}

/// Assembly generation instruction parsed from `_pdbx_struct_assembly_gen`.
#[derive(Debug, Clone)]
pub struct AssemblyGen {
    /// Assembly ID.
    pub assembly_id: String,
    /// Raw operation expression (e.g. `"1"`, `"1,2"`, `"(1-60)"`).
    pub oper_expression: String,
    /// Chain IDs (`asym_id_list` split on commas).
    pub chain_ids: Vec<String>,
}

/// Complete mmCIF structure after optional assembly generation and alternate
/// location filtering.
#[derive(Debug, Clone)]
pub struct CifStructure {
    /// All atoms (after assembly generation and alt-loc filtering).
    pub atoms: Vec<CifAtom>,
    /// Whether biological assembly operations were applied.
    pub assembly_applied: bool,
}

// ============================================================================
// mmCIF Tokenizer
// ============================================================================

/// Tokenize a single mmCIF data line into whitespace-separated tokens,
/// respecting single-quoted strings (which may contain spaces).
///
/// Single-quoted tokens have their surrounding quotes stripped.  The special
/// values `'.'` and `'?'` are returned verbatim (without quotes).
fn tokenize_cif_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Skip whitespace.
        if bytes[i] == b' ' || bytes[i] == b'\t' {
            i += 1;
            continue;
        }

        // Single-quoted string.
        if bytes[i] == b'\'' {
            i += 1; // skip opening quote
            let start = i;
            // Scan until closing quote followed by whitespace or end-of-line.
            while i < len {
                if bytes[i] == b'\'' && (i + 1 >= len || bytes[i + 1] == b' ' || bytes[i + 1] == b'\t') {
                    break;
                }
                i += 1;
            }
            tokens.push(line[start..i].to_string());
            if i < len {
                i += 1; // skip closing quote
            }
            continue;
        }

        // Double-quoted string.
        if bytes[i] == b'"' {
            i += 1; // skip opening quote
            let start = i;
            while i < len {
                if bytes[i] == b'"' && (i + 1 >= len || bytes[i + 1] == b' ' || bytes[i + 1] == b'\t') {
                    break;
                }
                i += 1;
            }
            tokens.push(line[start..i].to_string());
            if i < len {
                i += 1; // skip closing quote
            }
            continue;
        }

        // Unquoted token.
        let start = i;
        while i < len && bytes[i] != b' ' && bytes[i] != b'\t' {
            i += 1;
        }
        tokens.push(line[start..i].to_string());
    }

    tokens
}

/// Return the empty string for mmCIF missing-value sentinels (`"."` and `"?"`),
/// otherwise return the value unchanged.
fn cif_value(s: &str) -> String {
    if s == "." || s == "?" {
        String::new()
    } else {
        s.to_string()
    }
}

/// Parse a float from a token, treating `"."` and `"?"` as `0.0`.
fn cif_float(s: &str) -> Result<f64, String> {
    if s == "." || s == "?" {
        return Ok(0.0);
    }
    s.parse::<f64>()
        .map_err(|e| format!("Failed to parse float '{}': {}", s, e))
}

/// Parse an integer from a token, treating `"."` and `"?"` as `0`.
fn cif_int(s: &str) -> Result<i32, String> {
    if s == "." || s == "?" {
        return Ok(0);
    }
    s.parse::<i32>()
        .map_err(|e| format!("Failed to parse integer '{}': {}", s, e))
}

// ============================================================================
// Loop Block Parser
// ============================================================================

/// A parsed `loop_` block with column names and rows of tokenized values.
struct LoopBlock {
    /// Column names with the full `_category.item` form (lowercased).
    columns: Vec<String>,
    /// Each row is a vector of token strings aligned with `columns`.
    rows: Vec<Vec<String>>,
}

impl LoopBlock {
    /// Look up the column index for `col_name` (matched case-insensitively).
    fn col_index(&self, col_name: &str) -> Option<usize> {
        let lower = col_name.to_ascii_lowercase();
        self.columns.iter().position(|c| c == &lower)
    }

    /// Retrieve a value from a row by column name.  Returns `""` if the column
    /// is absent.
    fn get<'a>(&self, row: &'a [String], col_name: &str) -> &'a str {
        match self.col_index(col_name) {
            Some(idx) if idx < row.len() => &row[idx],
            _ => "",
        }
    }
}

/// Parse all `loop_` blocks whose columns start with `prefix` from the mmCIF
/// content lines.  Returns a map from category prefix (lowercased, e.g.
/// `"_atom_site"`) to the parsed `LoopBlock`.
///
/// This is a simplified parser sufficient for the categories we need.  It does
/// not handle multi-line semicolon-delimited values (which are not used in the
/// categories we target).
fn parse_loop_blocks(lines: &[&str]) -> HashMap<String, LoopBlock> {
    let mut blocks: HashMap<String, LoopBlock> = HashMap::new();
    let n = lines.len();
    let mut i = 0;

    while i < n {
        let trimmed = lines[i].trim();

        if trimmed.eq_ignore_ascii_case("loop_") {
            i += 1;
            // Collect column names.
            let mut columns: Vec<String> = Vec::new();
            while i < n {
                let t = lines[i].trim();
                if t.starts_with('_') {
                    columns.push(t.to_ascii_lowercase().to_string());
                    i += 1;
                } else {
                    break;
                }
            }

            if columns.is_empty() {
                continue;
            }

            // Determine the category prefix from the first column.
            let category = match columns[0].find('.') {
                Some(dot) => columns[0][..dot].to_string(),
                None => columns[0].clone(),
            };

            let num_cols = columns.len();

            // Collect data rows.  We accumulate tokens across lines until we
            // hit a line that starts a new category, a new loop_, or the data_
            // header of another block.
            let mut rows: Vec<Vec<String>> = Vec::new();
            let mut current_tokens: Vec<String> = Vec::new();

            while i < n {
                let t = lines[i].trim();
                if t.is_empty() {
                    i += 1;
                    // An empty line inside a loop can mean the loop is done,
                    // but only if we have no partial row pending.
                    if current_tokens.is_empty() {
                        // Peek ahead: if the next non-empty line starts with _
                        // or loop_ or data_ or #, the block is over.
                        let mut j = i;
                        while j < n && lines[j].trim().is_empty() {
                            j += 1;
                        }
                        if j >= n {
                            break;
                        }
                        let peek = lines[j].trim();
                        if peek.starts_with('_')
                            || peek.eq_ignore_ascii_case("loop_")
                            || peek.starts_with("data_")
                            || peek.starts_with('#')
                        {
                            break;
                        }
                    }
                    continue;
                }

                if t.starts_with('_')
                    || t.eq_ignore_ascii_case("loop_")
                    || t.starts_with("data_")
                {
                    break;
                }

                // Handle semicolon-delimited multi-line text fields.
                if let Some(stripped) = t.strip_prefix(';') {
                    // The value starts after the semicolon on this line.
                    let mut text_value = stripped.to_string();
                    i += 1;
                    while i < n {
                        let tl = lines[i];
                        if tl.starts_with(';') {
                            break;
                        }
                        text_value.push(' ');
                        text_value.push_str(tl.trim());
                        i += 1;
                    }
                    if i < n {
                        i += 1; // skip closing semicolon line
                    }
                    current_tokens.push(text_value);
                    if current_tokens.len() == num_cols {
                        rows.push(current_tokens);
                        current_tokens = Vec::new();
                    }
                    continue;
                }

                // Skip comment lines.
                if t.starts_with('#') {
                    i += 1;
                    continue;
                }

                let line_tokens = tokenize_cif_line(t);
                for tok in line_tokens {
                    current_tokens.push(tok);
                    if current_tokens.len() == num_cols {
                        rows.push(current_tokens);
                        current_tokens = Vec::new();
                    }
                }
                i += 1;
            }

            // If there are leftover tokens that don't fill a complete row,
            // discard them (malformed data).
            blocks.insert(
                category,
                LoopBlock { columns, rows },
            );
        } else {
            i += 1;
        }
    }

    blocks
}

// ============================================================================
// Atom Site Parsing
// ============================================================================

/// Parse all model-1 atoms from the `_atom_site` loop block.
fn parse_atom_site(block: &LoopBlock) -> Result<Vec<CifAtom>, String> {
    let mut atoms = Vec::with_capacity(block.rows.len());

    for (row_idx, row) in block.rows.iter().enumerate() {
        // Model number filter: only keep model 1.
        let model_str = block.get(row, "_atom_site.pdbx_pdb_model_num");
        let model_num = if model_str.is_empty() {
            1
        } else {
            model_str.parse::<i32>().unwrap_or(1)
        };
        if model_num != 1 {
            continue;
        }

        let id_str = block.get(row, "_atom_site.id");
        let id: usize = id_str
            .parse()
            .map_err(|e| format!("_atom_site row {}: bad atom id '{}': {}", row_idx + 1, id_str, e))?;

        let name = cif_value(block.get(row, "_atom_site.label_atom_id"));
        let alt_id = cif_value(block.get(row, "_atom_site.label_alt_id"));
        let residue_name = cif_value(block.get(row, "_atom_site.label_comp_id"));
        let chain_id = cif_value(block.get(row, "_atom_site.label_asym_id"));
        let auth_chain_id = cif_value(block.get(row, "_atom_site.auth_asym_id"));

        let res_seq_str = block.get(row, "_atom_site.auth_seq_id");
        let res_seq = cif_int(res_seq_str)
            .map_err(|e| format!("_atom_site row {}: bad auth_seq_id: {}", row_idx + 1, e))?;

        let ins_code = {
            let v = block.get(row, "_atom_site.pdbx_pdb_ins_code");
            if v == "?" || v == "." {
                String::new()
            } else {
                v.to_string()
            }
        };

        let x = cif_float(block.get(row, "_atom_site.cartn_x"))
            .map_err(|e| format!("_atom_site row {}: {}", row_idx + 1, e))?;
        let y = cif_float(block.get(row, "_atom_site.cartn_y"))
            .map_err(|e| format!("_atom_site row {}: {}", row_idx + 1, e))?;
        let z = cif_float(block.get(row, "_atom_site.cartn_z"))
            .map_err(|e| format!("_atom_site row {}: {}", row_idx + 1, e))?;

        let occupancy = cif_float(block.get(row, "_atom_site.occupancy"))
            .map_err(|e| format!("_atom_site row {}: {}", row_idx + 1, e))?;
        let b_factor = cif_float(block.get(row, "_atom_site.b_iso_or_equiv"))
            .map_err(|e| format!("_atom_site row {}: {}", row_idx + 1, e))?;

        let element = cif_value(block.get(row, "_atom_site.type_symbol"));
        let group = cif_value(block.get(row, "_atom_site.group_pdb"));

        atoms.push(CifAtom {
            id,
            name,
            alt_id,
            residue_name,
            chain_id,
            auth_chain_id,
            res_seq,
            ins_code,
            position: [x, y, z],
            occupancy,
            b_factor,
            element,
            group,
            model_num,
        });
    }

    Ok(atoms)
}

// ============================================================================
// Alternate Location Filtering
// ============================================================================

/// Filter atoms with alternate location indicators, keeping the conformer with
/// the highest occupancy for each unique site.  Atoms without an alt_id pass
/// through unchanged.
///
/// A "site" is identified by the tuple (chain_id, res_seq, ins_code, name).
fn filter_alt_locs(atoms: Vec<CifAtom>) -> Vec<CifAtom> {
    // First pass: for each site with alt locs, determine the winning alt_id.
    // Key: (chain_id, res_seq, ins_code, atom_name) -> best (alt_id, occupancy).
    let mut best_alt: HashMap<(String, i32, String, String), (String, f64)> = HashMap::new();

    for atom in &atoms {
        if atom.alt_id.is_empty() {
            continue;
        }
        let key = (
            atom.chain_id.clone(),
            atom.res_seq,
            atom.ins_code.clone(),
            atom.name.clone(),
        );
        let entry = best_alt
            .entry(key)
            .or_insert_with(|| (atom.alt_id.clone(), atom.occupancy));
        if atom.occupancy > entry.1 {
            *entry = (atom.alt_id.clone(), atom.occupancy);
        }
    }

    // Second pass: keep atoms that either have no alt_id or match the winning
    // alt_id for their site.
    atoms
        .into_iter()
        .filter(|atom| {
            if atom.alt_id.is_empty() {
                return true;
            }
            let key = (
                atom.chain_id.clone(),
                atom.res_seq,
                atom.ins_code.clone(),
                atom.name.clone(),
            );
            match best_alt.get(&key) {
                Some((best_id, _)) => atom.alt_id == *best_id,
                None => true,
            }
        })
        .collect()
}

// ============================================================================
// Symmetry Operations Parsing
// ============================================================================

/// Parse symmetry operations from the `_pdbx_struct_oper_list` loop block.
fn parse_oper_list(block: &LoopBlock) -> Result<Vec<SymOp>, String> {
    let mut ops = Vec::with_capacity(block.rows.len());

    for (row_idx, row) in block.rows.iter().enumerate() {
        let id = cif_value(block.get(row, "_pdbx_struct_oper_list.id"));
        if id.is_empty() {
            return Err(format!(
                "_pdbx_struct_oper_list row {}: missing operation id",
                row_idx + 1
            ));
        }

        // Parse the 3x3 rotation matrix.
        let mut rotation = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                let col_name = format!(
                    "_pdbx_struct_oper_list.matrix[{}][{}]",
                    r + 1,
                    c + 1
                );
                let val_str = block.get(row, &col_name);
                rotation[r][c] = cif_float(val_str).map_err(|e| {
                    format!(
                        "_pdbx_struct_oper_list row {}: bad {}: {}",
                        row_idx + 1,
                        col_name,
                        e
                    )
                })?;
            }
        }

        // Parse the translation vector.
        let mut translation = [0.0f64; 3];
        for r in 0..3 {
            let col_name = format!("_pdbx_struct_oper_list.vector[{}]", r + 1);
            let val_str = block.get(row, &col_name);
            translation[r] = cif_float(val_str).map_err(|e| {
                format!(
                    "_pdbx_struct_oper_list row {}: bad {}: {}",
                    row_idx + 1,
                    col_name,
                    e
                )
            })?;
        }

        ops.push(SymOp {
            id,
            rotation,
            translation,
        });
    }

    Ok(ops)
}

// ============================================================================
// Assembly Generation Parsing
// ============================================================================

/// Parse assembly generation entries from the `_pdbx_struct_assembly_gen`
/// loop block, keeping only those matching `assembly_id`.
fn parse_assembly_gen(block: &LoopBlock, assembly_id: &str) -> Vec<AssemblyGen> {
    let mut gens = Vec::new();

    for row in &block.rows {
        let aid = cif_value(block.get(row, "_pdbx_struct_assembly_gen.assembly_id"));
        if aid != assembly_id {
            continue;
        }

        let oper_expression =
            cif_value(block.get(row, "_pdbx_struct_assembly_gen.oper_expression"));
        let asym_id_list =
            cif_value(block.get(row, "_pdbx_struct_assembly_gen.asym_id_list"));

        let chain_ids: Vec<String> = asym_id_list
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        gens.push(AssemblyGen {
            assembly_id: aid,
            oper_expression,
            chain_ids,
        });
    }

    gens
}

// ============================================================================
// Operation Expression Parsing
// ============================================================================

/// Parse an operation expression string into a list of lists of operation IDs.
///
/// Each parenthesized group becomes one list.  When multiple groups appear
/// consecutively (e.g. `"(1-5)(6-10)"`), they represent the Cartesian product
/// of operations to apply sequentially.
///
/// Syntax examples:
/// - `"1"` -> `[[1]]` (single operation)
/// - `"1,2,3"` -> `[[1, 2, 3]]`
/// - `"(1-60)"` -> `[[1, 2, ..., 60]]`
/// - `"(1-5)(6-10)"` -> `[[1,..,5], [6,..,10]]` (Cartesian product)
fn parse_oper_expression(expr: &str) -> Result<Vec<Vec<String>>, String> {
    let expr = expr.trim();
    if expr.is_empty() {
        return Err("Empty operation expression".into());
    }

    // Check if expression contains parenthesized groups.
    if expr.contains('(') {
        let mut groups: Vec<Vec<String>> = Vec::new();
        let mut rest = expr;

        while let Some(open) = rest.find('(') {
            // If there is text before the parenthesis, parse it as a simple list.
            let prefix = rest[..open].trim();
            if !prefix.is_empty() {
                groups.push(parse_simple_oper_list(prefix)?);
            }

            let close = rest[open..]
                .find(')')
                .ok_or_else(|| format!("Unmatched '(' in operation expression '{}'", expr))?;

            let inner = &rest[open + 1..open + close];
            groups.push(parse_simple_oper_list(inner)?);

            rest = &rest[open + close + 1..];
        }

        // Any trailing text after the last ')'.
        let trailing = rest.trim();
        if !trailing.is_empty() {
            groups.push(parse_simple_oper_list(trailing)?);
        }

        if groups.is_empty() {
            return Err(format!(
                "No operation IDs found in expression '{}'",
                expr
            ));
        }

        Ok(groups)
    } else {
        // Simple comma-separated list without parentheses.
        Ok(vec![parse_simple_oper_list(expr)?])
    }
}

/// Parse a simple comma-separated and/or dash-range list of operation IDs.
///
/// Examples: `"1"`, `"1,2,3"`, `"1-5"`, `"1,3-5,7"`.
fn parse_simple_oper_list(expr: &str) -> Result<Vec<String>, String> {
    let mut ids = Vec::new();

    for part in expr.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        if part.contains('-') {
            // Range: "start-end" (inclusive, numeric).
            let dash_pos = part.find('-').unwrap();
            let start_str = part[..dash_pos].trim();
            let end_str = part[dash_pos + 1..].trim();

            let start: i64 = start_str
                .parse()
                .map_err(|e| format!("Bad range start '{}': {}", start_str, e))?;
            let end: i64 = end_str
                .parse()
                .map_err(|e| format!("Bad range end '{}': {}", end_str, e))?;

            if end < start {
                return Err(format!("Invalid range {}-{}: end < start", start, end));
            }

            for v in start..=end {
                ids.push(v.to_string());
            }
        } else {
            ids.push(part.to_string());
        }
    }

    if ids.is_empty() {
        return Err(format!("No operation IDs in '{}'", expr));
    }

    Ok(ids)
}

/// Expand a parsed operation expression (groups) into a flat list of operation
/// ID sequences using the Cartesian product.
///
/// For a single group `[[1, 2, 3]]`, the result is `[[1], [2], [3]]`.
/// For two groups `[[1, 2], [3, 4]]`, the Cartesian product gives
/// `[[1, 3], [1, 4], [2, 3], [2, 4]]`.
fn expand_oper_expression(groups: &[Vec<String>]) -> Vec<Vec<String>> {
    if groups.is_empty() {
        return vec![vec![]];
    }

    if groups.len() == 1 {
        return groups[0].iter().map(|id| vec![id.clone()]).collect();
    }

    // Cartesian product via iterative expansion.
    let mut result: Vec<Vec<String>> = vec![vec![]];

    for group in groups {
        let mut new_result = Vec::with_capacity(result.len() * group.len());
        for prefix in &result {
            for id in group {
                let mut combo = prefix.clone();
                combo.push(id.clone());
                new_result.push(combo);
            }
        }
        result = new_result;
    }

    result
}

// ============================================================================
// Matrix Application
// ============================================================================

/// Apply a rotation matrix and translation vector to a 3D position.
///
/// `pos_new = R * pos_old + T`
fn apply_transform(pos: &[f64; 3], rotation: &[[f64; 3]; 3], translation: &[f64; 3]) -> [f64; 3] {
    [
        rotation[0][0] * pos[0] + rotation[0][1] * pos[1] + rotation[0][2] * pos[2] + translation[0],
        rotation[1][0] * pos[0] + rotation[1][1] * pos[1] + rotation[1][2] * pos[2] + translation[1],
        rotation[2][0] * pos[0] + rotation[2][1] * pos[1] + rotation[2][2] * pos[2] + translation[2],
    ]
}

/// Compose two symmetry operations: `result = B(A(x)) = B.R * A.R * x + B.R * A.T + B.T`.
fn compose_transforms(a: &SymOp, b: &SymOp) -> SymOp {
    let mut rotation = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rotation[i][j] = b.rotation[i][0] * a.rotation[0][j]
                + b.rotation[i][1] * a.rotation[1][j]
                + b.rotation[i][2] * a.rotation[2][j];
        }
    }

    let translation = [
        b.rotation[0][0] * a.translation[0]
            + b.rotation[0][1] * a.translation[1]
            + b.rotation[0][2] * a.translation[2]
            + b.translation[0],
        b.rotation[1][0] * a.translation[0]
            + b.rotation[1][1] * a.translation[1]
            + b.rotation[1][2] * a.translation[2]
            + b.translation[1],
        b.rotation[2][0] * a.translation[0]
            + b.rotation[2][1] * a.translation[1]
            + b.rotation[2][2] * a.translation[2]
            + b.translation[2],
    ];

    SymOp {
        id: format!("{}_{}", a.id, b.id),
        rotation,
        translation,
    }
}

/// Check whether a symmetry operation is the identity (within tolerance).
fn is_identity(op: &SymOp) -> bool {
    let eps = 1e-6;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (op.rotation[i][j] - expected).abs() > eps {
                return false;
            }
        }
        if op.translation[i].abs() > eps {
            return false;
        }
    }
    true
}

// ============================================================================
// Assembly Builder
// ============================================================================

/// Build the biological assembly from the asymmetric unit atoms, assembly
/// generation instructions, and symmetry operations.
///
/// For each `AssemblyGen` entry, the operation expression is expanded into a
/// list of composite operations (via Cartesian product if multiple groups are
/// present).  Each composite operation is applied to the matching chains.
/// Identity operations leave atoms unchanged; non-identity operations produce
/// copies with suffixed chain IDs to distinguish them.
fn build_assembly(
    atoms: &[CifAtom],
    assembly_gens: &[AssemblyGen],
    sym_ops: &[SymOp],
) -> Result<Vec<CifAtom>, String> {
    // Index symmetry operations by ID for fast lookup.
    let op_map: HashMap<&str, &SymOp> = sym_ops.iter().map(|op| (op.id.as_str(), op)).collect();

    let mut result: Vec<CifAtom> = Vec::new();

    for gen in assembly_gens {
        let groups = parse_oper_expression(&gen.oper_expression)?;
        let op_sequences = expand_oper_expression(&groups);

        // Select atoms that belong to the specified chain IDs.
        let selected_atoms: Vec<&CifAtom> = atoms
            .iter()
            .filter(|a| gen.chain_ids.contains(&a.chain_id))
            .collect();

        if selected_atoms.is_empty() {
            continue;
        }

        for op_seq in &op_sequences {
            // Resolve each operation ID and compose the sequence into a single
            // combined transform.
            let mut combined: Option<SymOp> = None;

            for op_id in op_seq {
                let op = op_map.get(op_id.as_str()).ok_or_else(|| {
                    format!(
                        "Assembly references operation '{}' but it was not found in \
                         _pdbx_struct_oper_list",
                        op_id
                    )
                })?;

                combined = Some(match combined {
                    None => (*op).clone(),
                    Some(ref prev) => compose_transforms(prev, op),
                });
            }

            let combined = combined.ok_or_else(|| {
                "Empty operation sequence in assembly generation".to_string()
            })?;

            let identity = is_identity(&combined);

            // Build a chain suffix for non-identity operations so copied chains
            // are distinguishable.  For identity, keep the original chain ID.
            let suffix = if identity {
                String::new()
            } else {
                // Use the joined operation IDs as the suffix.
                format!("_{}", op_seq.join("_"))
            };

            for atom in &selected_atoms {
                let new_position = if identity {
                    atom.position
                } else {
                    apply_transform(&atom.position, &combined.rotation, &combined.translation)
                };

                let new_chain_id = if suffix.is_empty() {
                    atom.chain_id.clone()
                } else {
                    format!("{}{}", atom.chain_id, suffix)
                };

                let new_auth_chain_id = if suffix.is_empty() {
                    atom.auth_chain_id.clone()
                } else {
                    format!("{}{}", atom.auth_chain_id, suffix)
                };

                result.push(CifAtom {
                    id: atom.id,
                    name: atom.name.clone(),
                    alt_id: atom.alt_id.clone(),
                    residue_name: atom.residue_name.clone(),
                    chain_id: new_chain_id,
                    auth_chain_id: new_auth_chain_id,
                    res_seq: atom.res_seq,
                    ins_code: atom.ins_code.clone(),
                    position: new_position,
                    occupancy: atom.occupancy,
                    b_factor: atom.b_factor,
                    element: atom.element.clone(),
                    group: atom.group.clone(),
                    model_num: atom.model_num,
                });
            }
        }
    }

    Ok(result)
}

// ============================================================================
// Renumber Atoms
// ============================================================================

/// Assign sequential 1-based atom IDs to the atom list.
fn renumber_atoms(atoms: &mut [CifAtom]) {
    for (i, atom) in atoms.iter_mut().enumerate() {
        atom.id = i + 1;
    }
}

// ============================================================================
// Top-Level Parsers
// ============================================================================

/// Parse mmCIF from text content, applying biological assembly 1 if available.
///
/// This is the primary entry point.  If the file contains
/// `_pdbx_struct_assembly_gen` data for assembly `"1"`, those operations will
/// be applied to produce the biological unit.  Otherwise the asymmetric unit is
/// returned as-is.
///
/// Atoms with alternate location indicators are filtered so that only the
/// highest-occupancy conformer is retained for each site.
///
/// # Arguments
/// * `content` - The full text content of an mmCIF file.
///
/// # Returns
/// * `Ok(CifStructure)` with the parsed (and optionally assembled) atoms.
/// * `Err(String)` with a descriptive error message on failure.
pub fn parse_mmcif(content: &str) -> Result<CifStructure, String> {
    parse_mmcif_assembly(content, Some("1"))
}

/// Parse mmCIF from text content with a specific assembly ID.
///
/// When `assembly_id` is `Some("1")` (or another valid ID), the parser applies
/// the corresponding symmetry operations from `_pdbx_struct_assembly_gen` and
/// `_pdbx_struct_oper_list` to build the biological assembly.
///
/// When `assembly_id` is `None`, the asymmetric unit is returned without any
/// assembly operations applied.
///
/// # Arguments
/// * `content` - The full text content of an mmCIF file.
/// * `assembly_id` - Assembly to build, or `None` for the asymmetric unit.
///
/// # Returns
/// * `Ok(CifStructure)` with the parsed atoms.
/// * `Err(String)` with a descriptive error message on failure.
pub fn parse_mmcif_assembly(
    content: &str,
    assembly_id: Option<&str>,
) -> Result<CifStructure, String> {
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return Err("mmCIF content is empty".into());
    }

    // Parse all loop blocks.
    let blocks = parse_loop_blocks(&lines);

    // --- Atom site ---
    let atom_block = blocks
        .get("_atom_site")
        .ok_or("No _atom_site loop found in mmCIF content")?;

    let atoms = parse_atom_site(atom_block)?;

    if atoms.is_empty() {
        return Err("No model 1 atoms found in _atom_site".into());
    }

    // Filter alternate locations.
    let atoms = filter_alt_locs(atoms);

    // --- Assembly generation ---
    let want_assembly = assembly_id.is_some();
    let mut assembly_applied = false;

    let final_atoms = if want_assembly {
        let aid = assembly_id.unwrap();

        // Check if assembly data is available.
        let has_assembly_gen = blocks.contains_key("_pdbx_struct_assembly_gen");
        let has_oper_list = blocks.contains_key("_pdbx_struct_oper_list");

        if has_assembly_gen && has_oper_list {
            let assembly_gen_block = blocks.get("_pdbx_struct_assembly_gen").unwrap();
            let oper_list_block = blocks.get("_pdbx_struct_oper_list").unwrap();

            let assembly_gens = parse_assembly_gen(assembly_gen_block, aid);
            let sym_ops = parse_oper_list(oper_list_block)?;

            if assembly_gens.is_empty() {
                // Requested assembly not found; fall back to asymmetric unit.
                atoms
            } else if sym_ops.is_empty() {
                // No symmetry operations; fall back to asymmetric unit.
                atoms
            } else {
                let mut assembled = build_assembly(&atoms, &assembly_gens, &sym_ops)?;
                if assembled.is_empty() {
                    // Assembly produced no atoms (chain mismatch); return asymmetric unit.
                    atoms
                } else {
                    renumber_atoms(&mut assembled);
                    assembly_applied = true;
                    assembled
                }
            }
        } else {
            // No assembly metadata in file; return asymmetric unit.
            atoms
        }
    } else {
        atoms
    };

    Ok(CifStructure {
        atoms: final_atoms,
        assembly_applied,
    })
}

/// Parse mmCIF from a file path, applying biological assembly 1 if available.
///
/// Reads the entire file into memory and delegates to [`parse_mmcif`].
///
/// # Arguments
/// * `path` - Path to the mmCIF file.
///
/// # Returns
/// * `Ok(CifStructure)` with the parsed atoms.
/// * `Err(String)` with a descriptive error message on failure.
pub fn parse_mmcif_file(path: &Path) -> Result<CifStructure, String> {
    let content = read_to_string(path)
        .map_err(|e| format!("Failed to read mmCIF file '{}': {}", path.display(), e))?;
    parse_mmcif(&content)
}
