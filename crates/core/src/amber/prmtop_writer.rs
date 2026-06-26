//! AMBER prmtop (topology) file writer.
//!
//! Writes AMBER7-format topology files compatible with sander, cpptraj,
//! and the parser in [`super::prmtop`]. The input is a [`PrmtopData`] struct
//! containing all raw parameter arrays produced by the system builder's
//! parameterization pipeline.
//!
//! # Format overview
//!
//! The prmtop file consists of a `%VERSION` header followed by a series of
//! `%FLAG` / `%FORMAT` / data sections. Integer sections use `(10I8)`,
//! float sections use `(5E16.8)`, and string sections use `(20a4)` or
//! `(1a80)`.

use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

use super::prmtop::AMBER_CHARGE_FACTOR;

// ============================================================================
// Data Structures
// ============================================================================

/// All data needed to write a prmtop file.
///
/// Index conventions follow what the caller provides:
/// - `atom_type_indices`: 0-based (written as 1-based)
/// - `residue_pointers`: 0-based atom indices (written as 1-based)
/// - `excluded_atoms_list`: 0-based atom indices (written as 1-based, 0 as placeholder)
/// - `charges`: elementary charge units (multiplied by 18.2223 on write)
/// - Topology lists (`bonds_inc_hydrogen`, etc.): already in AMBER coordinate-index
///   form (atom_i*3, atom_j*3, type_1based) and written verbatim.
pub struct PrmtopData {
    pub title: String,
    /// 4-char atom names, one per atom.
    pub atom_names: Vec<String>,
    /// Partial charges in elementary charge units.
    pub charges: Vec<f64>,
    /// Atomic numbers, one per atom.
    pub atomic_numbers: Vec<i32>,
    /// Atomic masses in amu, one per atom.
    pub masses: Vec<f64>,
    /// 0-based atom type index per atom.
    pub atom_type_indices: Vec<usize>,
    /// Number of excluded atoms per atom.
    pub num_excluded_atoms: Vec<usize>,
    /// Flat excluded-atoms list (0-based indices; 0 used as placeholder for no exclusions).
    pub excluded_atoms_list: Vec<usize>,
    /// Residue names, one per residue.
    pub residue_labels: Vec<String>,
    /// 0-based index of first atom in each residue.
    pub residue_pointers: Vec<usize>,
    /// AMBER atom type names (up to 4 chars), one per atom.
    pub amber_atom_types: Vec<String>,

    /// Number of unique atom types.
    pub n_types: usize,
    /// Nonbonded parameter index array, ntypes*ntypes elements (1-based).
    pub nb_parm_index: Vec<i32>,

    // Bond parameters (indexed by bond type)
    pub bond_force_constants: Vec<f64>,
    pub bond_equil_values: Vec<f64>,

    // Angle parameters (indexed by angle type)
    pub angle_force_constants: Vec<f64>,
    /// Angle equilibrium values in radians.
    pub angle_equil_values: Vec<f64>,

    // Dihedral parameters (indexed by dihedral type)
    pub dihedral_force_constants: Vec<f64>,
    pub dihedral_periodicities: Vec<f64>,
    /// Dihedral phases in radians.
    pub dihedral_phases: Vec<f64>,
    /// 1-4 electrostatic scaling factors, one per dihedral type.
    pub scee_scale_factors: Vec<f64>,
    /// 1-4 van der Waals scaling factors, one per dihedral type.
    pub scnb_scale_factors: Vec<f64>,

    // Lennard-Jones parameters
    pub lj_acoef: Vec<f64>,
    pub lj_bcoef: Vec<f64>,

    // Topology lists: already in AMBER format (coordinate indices, 1-based types).
    pub bonds_inc_hydrogen: Vec<i32>,
    pub bonds_without_hydrogen: Vec<i32>,
    pub angles_inc_hydrogen: Vec<i32>,
    pub angles_without_hydrogen: Vec<i32>,
    pub dihedrals_inc_hydrogen: Vec<i32>,
    pub dihedrals_without_hydrogen: Vec<i32>,

    /// Periodic box information. `None` means no periodic box (IFBOX=0).
    pub box_info: Option<BoxInfo>,

    /// Born radii per atom (Angstroms).
    pub radii: Vec<f64>,
    /// GB screening parameters per atom.
    pub screen: Vec<f64>,
}

/// Periodic box and solvent information for the SOLVENT_POINTERS,
/// ATOMS_PER_MOLECULE, and BOX_DIMENSIONS sections.
pub struct BoxInfo {
    /// Box angle beta in degrees (typically 90.0 for orthorhombic).
    pub beta: f64,
    /// Box dimensions [x, y, z] in Angstroms.
    pub dimensions: [f64; 3],
    /// Last solute residue index (1-based, written to SOLVENT_POINTERS[0]).
    pub last_solute_residue: usize,
    /// Number of atoms in each molecule.
    pub atoms_per_molecule: Vec<usize>,
    /// First solvent molecule index (1-based, written to SOLVENT_POINTERS[2]).
    pub first_solvent_molecule: usize,
    /// Number of extra points (virtual sites).
    pub num_extra_points: usize,
}

// ============================================================================
// Public API
// ============================================================================

/// Write a prmtop file to disk.
///
/// # Errors
///
/// Returns `Err(String)` if the data is internally inconsistent (e.g. array
/// length mismatches) or if the file cannot be written.
pub fn write_prmtop(data: &PrmtopData, path: &Path) -> Result<(), String> {
    let content = write_prmtop_string(data)?;
    fs::write(path, content).map_err(|e| format!("Failed to write prmtop file: {}", e))
}

/// Write a prmtop to a `String`.
///
/// This is useful for testing without touching the filesystem.
///
/// # Errors
///
/// Returns `Err(String)` if the data is internally inconsistent.
pub fn write_prmtop_string(data: &PrmtopData) -> Result<String, String> {
    validate(data)?;

    let n_atoms = data.atom_names.len();

    // Pre-allocate a generous buffer.
    let mut out = String::with_capacity(n_atoms * 120);

    // ---- VERSION header ----
    out.push_str("%VERSION  VERSION_STAMP = V0001.000  DATE = 01/01/00  00:00:00\n");

    // ---- TITLE ----
    write_flag_format(&mut out, "TITLE", "(20a4)");
    write_title_line(&mut out, &data.title);

    // ---- POINTERS ----
    let pointers = build_pointers(data);
    write_integer_section(&mut out, "POINTERS", &pointers);

    // ---- ATOM_NAME ----
    let atom_name_refs: Vec<&str> = data.atom_names.iter().map(String::as_str).collect();
    write_string_section(&mut out, "ATOM_NAME", &atom_name_refs, 4);

    // ---- CHARGE (multiply by AMBER_CHARGE_FACTOR) ----
    let amber_charges: Vec<f64> = data.charges.iter().map(|&q| q * AMBER_CHARGE_FACTOR).collect();
    write_float_section(&mut out, "CHARGE", &amber_charges);

    // ---- ATOMIC_NUMBER ----
    write_integer_section(&mut out, "ATOMIC_NUMBER", &data.atomic_numbers);

    // ---- MASS ----
    write_float_section(&mut out, "MASS", &data.masses);

    // ---- ATOM_TYPE_INDEX (0-based -> 1-based) ----
    let type_indices_1based: Vec<i32> = data
        .atom_type_indices
        .iter()
        .map(|&i| (i + 1) as i32)
        .collect();
    write_integer_section(&mut out, "ATOM_TYPE_INDEX", &type_indices_1based);

    // ---- NUMBER_EXCLUDED_ATOMS ----
    let num_excl_i32: Vec<i32> = data
        .num_excluded_atoms
        .iter()
        .map(|&n| n as i32)
        .collect();
    write_integer_section(&mut out, "NUMBER_EXCLUDED_ATOMS", &num_excl_i32);

    // ---- NONBONDED_PARM_INDEX ----
    write_integer_section(&mut out, "NONBONDED_PARM_INDEX", &data.nb_parm_index);

    // ---- RESIDUE_LABEL ----
    let res_label_refs: Vec<&str> = data.residue_labels.iter().map(String::as_str).collect();
    write_string_section(&mut out, "RESIDUE_LABEL", &res_label_refs, 4);

    // ---- RESIDUE_POINTER (0-based -> 1-based) ----
    let res_ptr_1based: Vec<i32> = data
        .residue_pointers
        .iter()
        .map(|&p| (p + 1) as i32)
        .collect();
    write_integer_section(&mut out, "RESIDUE_POINTER", &res_ptr_1based);

    // ---- Bond parameters ----
    write_float_section(&mut out, "BOND_FORCE_CONSTANT", &data.bond_force_constants);
    write_float_section(&mut out, "BOND_EQUIL_VALUE", &data.bond_equil_values);

    // ---- Angle parameters ----
    write_float_section(&mut out, "ANGLE_FORCE_CONSTANT", &data.angle_force_constants);
    write_float_section(&mut out, "ANGLE_EQUIL_VALUE", &data.angle_equil_values);

    // ---- Dihedral parameters ----
    write_float_section(
        &mut out,
        "DIHEDRAL_FORCE_CONSTANT",
        &data.dihedral_force_constants,
    );
    write_float_section(&mut out, "DIHEDRAL_PERIODICITY", &data.dihedral_periodicities);
    write_float_section(&mut out, "DIHEDRAL_PHASE", &data.dihedral_phases);
    write_float_section(&mut out, "SCEE_SCALE_FACTOR", &data.scee_scale_factors);
    write_float_section(&mut out, "SCNB_SCALE_FACTOR", &data.scnb_scale_factors);

    // ---- LJ coefficients ----
    write_float_section(&mut out, "LENNARD_JONES_ACOEF", &data.lj_acoef);
    write_float_section(&mut out, "LENNARD_JONES_BCOEF", &data.lj_bcoef);

    // ---- Topology lists ----
    write_integer_section(&mut out, "BONDS_INC_HYDROGEN", &data.bonds_inc_hydrogen);
    write_integer_section(
        &mut out,
        "BONDS_WITHOUT_HYDROGEN",
        &data.bonds_without_hydrogen,
    );
    write_integer_section(&mut out, "ANGLES_INC_HYDROGEN", &data.angles_inc_hydrogen);
    write_integer_section(
        &mut out,
        "ANGLES_WITHOUT_HYDROGEN",
        &data.angles_without_hydrogen,
    );
    write_integer_section(
        &mut out,
        "DIHEDRALS_INC_HYDROGEN",
        &data.dihedrals_inc_hydrogen,
    );
    write_integer_section(
        &mut out,
        "DIHEDRALS_WITHOUT_HYDROGEN",
        &data.dihedrals_without_hydrogen,
    );

    // ---- EXCLUDED_ATOMS_LIST (0-based -> 1-based, usize::MAX -> 0 placeholder) ----
    // Convention: 0-based atom index i becomes file value i+1. The sentinel
    // usize::MAX means "no exclusion" and maps to 0 in the file (placeholder).
    let excl_1based: Vec<i32> = data
        .excluded_atoms_list
        .iter()
        .map(|&idx| {
            if idx == usize::MAX {
                0_i32
            } else {
                (idx + 1) as i32
            }
        })
        .collect();
    write_integer_section(&mut out, "EXCLUDED_ATOMS_LIST", &excl_1based);

    // ---- H-bond placeholders (empty) ----
    write_float_section(&mut out, "HBOND_ACOEF", &[]);
    write_float_section(&mut out, "HBOND_BCOEF", &[]);
    write_float_section(&mut out, "HBCUT", &[]);

    // ---- AMBER_ATOM_TYPE ----
    let atom_type_refs: Vec<&str> = data.amber_atom_types.iter().map(String::as_str).collect();
    write_string_section(&mut out, "AMBER_ATOM_TYPE", &atom_type_refs, 4);

    // ---- TREE_CHAIN_CLASSIFICATION (all "BLA ") ----
    let bla_entries: Vec<&str> = vec!["BLA"; n_atoms];
    write_string_section(&mut out, "TREE_CHAIN_CLASSIFICATION", &bla_entries, 4);

    // ---- JOIN_ARRAY (all zeros) ----
    let zeros: Vec<i32> = vec![0; n_atoms];
    write_integer_section(&mut out, "JOIN_ARRAY", &zeros);

    // ---- IROTAT (all zeros) ----
    write_integer_section(&mut out, "IROTAT", &zeros);

    // ---- Box sections (if periodic) ----
    if let Some(ref box_info) = data.box_info {
        let solvent_pointers: Vec<i32> = vec![
            box_info.last_solute_residue as i32,
            box_info.atoms_per_molecule.len() as i32,
            box_info.first_solvent_molecule as i32,
        ];
        write_integer_section(&mut out, "SOLVENT_POINTERS", &solvent_pointers);

        let atoms_per_mol: Vec<i32> = box_info
            .atoms_per_molecule
            .iter()
            .map(|&n| n as i32)
            .collect();
        write_integer_section(&mut out, "ATOMS_PER_MOLECULE", &atoms_per_mol);

        let box_dims: Vec<f64> = vec![
            box_info.beta,
            box_info.dimensions[0],
            box_info.dimensions[1],
            box_info.dimensions[2],
        ];
        write_float_section(&mut out, "BOX_DIMENSIONS", &box_dims);
    }

    // ---- RADIUS_SET ----
    write_flag_format(&mut out, "RADIUS_SET", "(1a80)");
    let radius_set = "modified Bondi radii (mbondi3)";
    let _ = writeln!(out, "{:<80}", radius_set);

    // ---- RADII ----
    write_float_section(&mut out, "RADII", &data.radii);

    // ---- SCREEN ----
    write_float_section(&mut out, "SCREEN", &data.screen);

    // Compute the maximum residue size for POINTERS[28] (NMXRS).
    // This was already done in build_pointers, but we verify the output is
    // consistent by construction.

    Ok(out)
}

// ============================================================================
// Validation
// ============================================================================

/// Basic consistency checks before writing.
fn validate(data: &PrmtopData) -> Result<(), String> {
    let n_atoms = data.atom_names.len();

    if data.charges.len() != n_atoms {
        return Err(format!(
            "charges length ({}) != atom_names length ({})",
            data.charges.len(),
            n_atoms
        ));
    }
    if data.atomic_numbers.len() != n_atoms {
        return Err(format!(
            "atomic_numbers length ({}) != atom count ({})",
            data.atomic_numbers.len(),
            n_atoms
        ));
    }
    if data.masses.len() != n_atoms {
        return Err(format!(
            "masses length ({}) != atom count ({})",
            data.masses.len(),
            n_atoms
        ));
    }
    if data.atom_type_indices.len() != n_atoms {
        return Err(format!(
            "atom_type_indices length ({}) != atom count ({})",
            data.atom_type_indices.len(),
            n_atoms
        ));
    }
    if data.num_excluded_atoms.len() != n_atoms {
        return Err(format!(
            "num_excluded_atoms length ({}) != atom count ({})",
            data.num_excluded_atoms.len(),
            n_atoms
        ));
    }
    if data.amber_atom_types.len() != n_atoms {
        return Err(format!(
            "amber_atom_types length ({}) != atom count ({})",
            data.amber_atom_types.len(),
            n_atoms
        ));
    }
    if data.radii.len() != n_atoms {
        return Err(format!(
            "radii length ({}) != atom count ({})",
            data.radii.len(),
            n_atoms
        ));
    }
    if data.screen.len() != n_atoms {
        return Err(format!(
            "screen length ({}) != atom count ({})",
            data.screen.len(),
            n_atoms
        ));
    }

    let n_res = data.residue_labels.len();
    if data.residue_pointers.len() != n_res {
        return Err(format!(
            "residue_pointers length ({}) != residue_labels length ({})",
            data.residue_pointers.len(),
            n_res
        ));
    }

    let expected_nb = data.n_types * data.n_types;
    if data.nb_parm_index.len() != expected_nb {
        return Err(format!(
            "nb_parm_index length ({}) != n_types^2 ({})",
            data.nb_parm_index.len(),
            expected_nb
        ));
    }

    if data.bond_force_constants.len() != data.bond_equil_values.len() {
        return Err(format!(
            "bond_force_constants length ({}) != bond_equil_values length ({})",
            data.bond_force_constants.len(),
            data.bond_equil_values.len()
        ));
    }
    if data.angle_force_constants.len() != data.angle_equil_values.len() {
        return Err(format!(
            "angle_force_constants length ({}) != angle_equil_values length ({})",
            data.angle_force_constants.len(),
            data.angle_equil_values.len()
        ));
    }
    if data.dihedral_force_constants.len() != data.dihedral_periodicities.len() {
        return Err("dihedral_force_constants and dihedral_periodicities length mismatch".into());
    }
    if data.dihedral_force_constants.len() != data.dihedral_phases.len() {
        return Err("dihedral_force_constants and dihedral_phases length mismatch".into());
    }
    if data.dihedral_force_constants.len() != data.scee_scale_factors.len() {
        return Err("dihedral_force_constants and scee_scale_factors length mismatch".into());
    }
    if data.dihedral_force_constants.len() != data.scnb_scale_factors.len() {
        return Err("dihedral_force_constants and scnb_scale_factors length mismatch".into());
    }

    if !data.bonds_inc_hydrogen.len().is_multiple_of(3) {
        return Err(format!(
            "bonds_inc_hydrogen length ({}) not divisible by 3",
            data.bonds_inc_hydrogen.len()
        ));
    }
    if !data.bonds_without_hydrogen.len().is_multiple_of(3) {
        return Err(format!(
            "bonds_without_hydrogen length ({}) not divisible by 3",
            data.bonds_without_hydrogen.len()
        ));
    }
    if !data.angles_inc_hydrogen.len().is_multiple_of(4) {
        return Err(format!(
            "angles_inc_hydrogen length ({}) not divisible by 4",
            data.angles_inc_hydrogen.len()
        ));
    }
    if !data.angles_without_hydrogen.len().is_multiple_of(4) {
        return Err(format!(
            "angles_without_hydrogen length ({}) not divisible by 4",
            data.angles_without_hydrogen.len()
        ));
    }
    if !data.dihedrals_inc_hydrogen.len().is_multiple_of(5) {
        return Err(format!(
            "dihedrals_inc_hydrogen length ({}) not divisible by 5",
            data.dihedrals_inc_hydrogen.len()
        ));
    }
    if !data.dihedrals_without_hydrogen.len().is_multiple_of(5) {
        return Err(format!(
            "dihedrals_without_hydrogen length ({}) not divisible by 5",
            data.dihedrals_without_hydrogen.len()
        ));
    }

    Ok(())
}

// ============================================================================
// POINTERS Construction
// ============================================================================

/// Build the 31-element POINTERS array from `PrmtopData`.
fn build_pointers(data: &PrmtopData) -> Vec<i32> {
    let n_atoms = data.atom_names.len() as i32;
    let n_types = data.n_types as i32;
    let nbonh = (data.bonds_inc_hydrogen.len() / 3) as i32;
    let mbona = (data.bonds_without_hydrogen.len() / 3) as i32;
    let ntheth = (data.angles_inc_hydrogen.len() / 4) as i32;
    let mtheta = (data.angles_without_hydrogen.len() / 4) as i32;
    let nphih = (data.dihedrals_inc_hydrogen.len() / 5) as i32;
    let mphia = (data.dihedrals_without_hydrogen.len() / 5) as i32;
    let nnb: i32 = data.excluded_atoms_list.len() as i32;
    let nres = data.residue_labels.len() as i32;
    let numbnd = data.bond_force_constants.len() as i32;
    let numang = data.angle_force_constants.len() as i32;
    let nptra = data.dihedral_force_constants.len() as i32;

    let ifbox: i32 = if data.box_info.is_some() { 1 } else { 0 };

    // NMXRS: maximum number of atoms in any residue.
    let nmxrs = compute_max_residue_size(data);

    // NUMEXTRA: number of extra points (virtual sites).
    let numextra: i32 = data
        .box_info
        .as_ref()
        .map_or(0, |b| b.num_extra_points as i32);

    vec![
        n_atoms,  // [0]  NATOM
        n_types,  // [1]  NTYPES
        nbonh,    // [2]  NBONH
        mbona,    // [3]  MBONA
        ntheth,   // [4]  NTHETH
        mtheta,   // [5]  MTHETA
        nphih,    // [6]  NPHIH
        mphia,    // [7]  MPHIA
        0,        // [8]  NHPARM
        0,        // [9]  NPARM
        nnb,      // [10] NNB (total excluded atoms)
        nres,     // [11] NRES
        mbona,    // [12] NBONA = MBONA
        mtheta,   // [13] NTHETA = MTHETA
        mphia,    // [14] NPHIA = MPHIA
        numbnd,   // [15] NUMBND
        numang,   // [16] NUMANG
        nptra,    // [17] NPTRA
        n_types,  // [18] NATYP = NTYPES
        0,        // [19] NPHB
        0,        // [20] IFPERT
        0,        // [21] NBPER
        0,        // [22] NGPER
        0,        // [23] NDPER
        0,        // [24] MBPER
        0,        // [25] MGPER
        0,        // [26] MDPER
        ifbox,    // [27] IFBOX
        nmxrs,    // [28] NMXRS
        0,        // [29] IFCAP
        numextra, // [30] NUMEXTRA
    ]
}

/// Compute the maximum number of atoms across all residues.
fn compute_max_residue_size(data: &PrmtopData) -> i32 {
    let n_atoms = data.atom_names.len();
    let n_res = data.residue_labels.len();
    if n_res == 0 {
        return 0;
    }

    let mut max_size: usize = 0;
    for i in 0..n_res {
        let start = data.residue_pointers[i];
        let end = if i + 1 < n_res {
            data.residue_pointers[i + 1]
        } else {
            n_atoms
        };
        let size = end.saturating_sub(start);
        if size > max_size {
            max_size = size;
        }
    }
    max_size as i32
}

// ============================================================================
// Section Writers
// ============================================================================

/// Write a `%FLAG` and `%FORMAT` header pair.
fn write_flag_format(out: &mut String, flag: &str, format: &str) {
    let _ = writeln!(out, "%FLAG {}", flag);
    let _ = writeln!(out, "%FORMAT({})", format);
}

/// Write the TITLE data line. The title is written as a single line, padded
/// or truncated to 80 characters, using `(20a4)` format.
fn write_title_line(out: &mut String, title: &str) {
    // Truncate to 80 chars if needed.
    let t = if title.len() > 80 {
        &title[..80]
    } else {
        title
    };
    let _ = writeln!(out, "{:<80}", t);
}

/// Write an integer section in `(10I8)` format.
///
/// Each value is right-justified in an 8-character field, 10 values per line.
/// Empty sections produce a single blank line (AMBER convention).
fn write_integer_section(out: &mut String, flag: &str, values: &[i32]) {
    write_flag_format(out, flag, "10I8");

    if values.is_empty() {
        out.push('\n');
        return;
    }

    for (i, &val) in values.iter().enumerate() {
        let _ = write!(out, "{:>8}", val);
        if (i + 1) % 10 == 0 {
            out.push('\n');
        }
    }
    // Final newline if the last line was not complete.
    if !values.len().is_multiple_of(10) {
        out.push('\n');
    }
}

/// Write a floating-point section in `(5E16.8)` format.
///
/// Each value is formatted as a 16-character scientific notation field
/// (`%16.8E` in Fortran terms), 5 values per line. Empty sections produce
/// a single blank line.
fn write_float_section(out: &mut String, flag: &str, values: &[f64]) {
    write_flag_format(out, flag, "5E16.8");

    if values.is_empty() {
        out.push('\n');
        return;
    }

    for (i, &val) in values.iter().enumerate() {
        write_amber_float(out, val);
        if (i + 1) % 5 == 0 {
            out.push('\n');
        }
    }
    if !values.len().is_multiple_of(5) {
        out.push('\n');
    }
}

/// Write a string section in `(20a4)` format (or similar fixed-width string format).
///
/// Each string is left-justified in a field of `width` characters, 20 per line
/// (for width=4, giving 80 chars per line). Empty sections produce a single
/// blank line.
fn write_string_section(out: &mut String, flag: &str, values: &[&str], width: usize) {
    let items_per_line = 80 / width; // 20 for width=4
    let format_str = format!("{}a{}", items_per_line, width);
    write_flag_format(out, flag, &format_str);

    if values.is_empty() {
        out.push('\n');
        return;
    }

    for (i, &val) in values.iter().enumerate() {
        // Left-justify within the field width. Truncate if longer.
        let truncated = if val.len() > width { &val[..width] } else { val };
        let _ = write!(out, "{:<width$}", truncated, width = width);
        if (i + 1) % items_per_line == 0 {
            out.push('\n');
        }
    }
    if !values.len().is_multiple_of(items_per_line) {
        out.push('\n');
    }
}

// ============================================================================
// Float Formatting
// ============================================================================

/// Format a floating-point value in AMBER's `(E16.8)` style.
///
/// AMBER/Fortran scientific notation: `  1.23456789E+02`
/// - Total field width: 16 characters
/// - 8 digits after the decimal point
/// - Two-digit exponent with explicit sign
/// - No leading `+` on positive mantissa (space instead)
///
/// Rust's `{:16.8E}` produces a compatible output on most platforms, but
/// we normalize the exponent to always be at least two digits with an
/// explicit `+` or `-` sign to match Fortran conventions exactly.
fn write_amber_float(out: &mut String, val: f64) {
    // Use Rust's built-in scientific notation as a starting point.
    let raw = format!("{:16.8E}", val);

    // Rust formats exponents like "E2" or "E-2" or "E10". AMBER wants
    // "E+02" or "E-02" or "E+10". We need to normalize.
    if let Some(e_pos) = raw.find('E') {
        let (mantissa, exp_part) = raw.split_at(e_pos);
        let exp_str = &exp_part[1..]; // skip 'E'

        let (sign, digits) = if let Some(stripped) = exp_str.strip_prefix('-') {
            ("-", stripped)
        } else if let Some(stripped) = exp_str.strip_prefix('+') {
            ("+", stripped)
        } else {
            ("+", exp_str)
        };

        // Parse the exponent digits and re-format with at least 2 digits.
        let exp_val: i32 = digits.trim().parse().unwrap_or(0);
        let formatted = format!("{}E{}{:02}", mantissa, sign, exp_val);

        // Right-justify in 16 characters.
        let _ = write!(out, "{:>16}", formatted);
    } else {
        // Fallback: should not happen for finite floats.
        let _ = write!(out, "{:>16}", raw);
    }
}
