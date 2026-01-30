//! AMBER prmtop (topology) file parser.
//!
//! Parses AMBER7-format topology files to extract:
//! - Atom charges (converted to elementary charge units)
//! - Lennard-Jones parameters (sigma in nm, epsilon in kJ/mol)
//! - Residue information for mapping atoms to residues
//! - Atom names and types

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// Unit Conversion Constants
// ============================================================================

/// AMBER charge to elementary charge: q_e = q_amber / 18.2223
pub const AMBER_CHARGE_FACTOR: f64 = 18.2223;

/// kcal/mol to kJ/mol
pub const KCAL_TO_KJ: f64 = 4.184;

/// Angstrom to nm
pub const ANGSTROM_TO_NM: f64 = 0.1;

// ============================================================================
// Data Structures
// ============================================================================

/// Parsed AMBER topology containing all relevant force field data.
#[derive(Debug, Clone)]
pub struct AmberTopology {
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of residues
    pub n_residues: usize,
    /// Number of atom types
    pub n_types: usize,
    /// Atom names (4-char strings)
    pub atom_names: Vec<String>,
    /// Atom type indices (0-based)
    pub atom_type_indices: Vec<usize>,
    /// Partial charges in elementary charge units
    pub charges: Vec<f64>,
    /// Raw AMBER charges (before dividing by 18.2223)
    pub charges_amber: Vec<f64>,
    /// Residue labels
    pub residue_labels: Vec<String>,
    /// Residue pointers: first atom index (0-based) for each residue
    pub residue_pointers: Vec<usize>,
    /// LJ sigma parameters in nm (per atom type)
    pub lj_sigma: Arc<Vec<f64>>,
    /// LJ epsilon parameters in kJ/mol (per atom type)
    pub lj_epsilon: Arc<Vec<f64>>,
    /// Per-atom sigma values (looked up from type)
    pub atom_sigmas: Vec<f64>,
    /// Per-atom epsilon values (looked up from type)
    pub atom_epsilons: Vec<f64>,
    /// Bond pairs as (atom_i, atom_j) tuples (0-indexed)
    pub bonds: Vec<(usize, usize)>,
    /// Bond type index parallel to `bonds`
    pub bond_types: Vec<usize>,

    // Per-atom properties
    /// Atomic masses
    pub masses: Vec<f64>,
    /// Born radii (Angstrom)
    pub radii: Vec<f64>,
    /// GB screening parameters
    pub screen: Vec<f64>,

    // Bond parameters (indexed by bond type)
    pub bond_force_constants: Arc<Vec<f64>>,
    pub bond_equil_values: Arc<Vec<f64>>,

    // Angle parameters (indexed by angle type)
    pub angle_force_constants: Arc<Vec<f64>>,
    pub angle_equil_values: Arc<Vec<f64>>,

    // Dihedral parameters (indexed by dihedral type)
    pub dihedral_force_constants: Arc<Vec<f64>>,
    pub dihedral_periodicities: Arc<Vec<f64>>,
    pub dihedral_phases: Arc<Vec<f64>>,

    // Topology lists
    /// Angles: (atom_i, atom_j, atom_k, type_index) all 0-based
    pub angles: Vec<(usize, usize, usize, usize)>,
    /// Dihedrals: (i, j, k, l, type_index, is_improper)
    pub dihedrals: Vec<(usize, usize, usize, usize, usize, bool)>,

    // Exclusion lists
    pub num_excluded_atoms: Vec<usize>,
    pub excluded_atoms_list: Vec<usize>,

    // 1-4 scaling factors
    pub scee_scale_factor: f64,
    pub scnb_scale_factor: f64,

    // Raw LJ coefficients
    pub lj_acoef: Arc<Vec<f64>>,
    pub lj_bcoef: Arc<Vec<f64>>,
    pub nb_parm_index: Arc<Vec<i64>>,
}

/// Atom selection grouped by residue for fingerprint calculations.
#[derive(Debug, Clone)]
pub struct AtomSelection {
    /// Flat list of atom indices (global, 0-based)
    pub atom_indices: Vec<usize>,
    /// Start offset per residue in atom_indices
    pub residue_offsets: Vec<usize>,
    /// Residue names/labels
    pub residue_labels: Vec<String>,
}

impl AmberTopology {
    /// Get the residue index for each atom.
    pub fn atom_residue_indices(&self) -> Vec<usize> {
        let mut result = vec![0usize; self.n_atoms];
        for res_idx in 0..self.n_residues {
            let start = self.residue_pointers[res_idx];
            let end = if res_idx + 1 < self.n_residues {
                self.residue_pointers[res_idx + 1]
            } else {
                self.n_atoms
            };
            result[start..end].fill(res_idx);
        }
        result
    }

    /// Build resmap (indices and offsets) for fingerprint calculations.
    /// Returns (indices, offsets) where offsets[i] is the start index for residue i.
    pub fn build_resmap(&self) -> (Vec<i64>, Vec<i64>) {
        let mut indices: Vec<i64> = Vec::with_capacity(self.n_atoms);
        let mut offsets: Vec<i64> = Vec::with_capacity(self.n_residues + 1);

        for res_idx in 0..self.n_residues {
            offsets.push(indices.len() as i64);
            let start = self.residue_pointers[res_idx];
            let end = if res_idx + 1 < self.n_residues {
                self.residue_pointers[res_idx + 1]
            } else {
                self.n_atoms
            };
            for atom_idx in start..end {
                indices.push(atom_idx as i64);
            }
        }
        offsets.push(indices.len() as i64);

        (indices, offsets)
    }

    /// Build an atom selection for a specific set of residues.
    ///
    /// # Arguments
    /// * `residue_indices` - 0-based residue indices to include in the selection
    ///
    /// # Returns
    /// * `Ok(AtomSelection)` - Selection with atom indices grouped by residue
    /// * `Err(String)` - If any residue index is out of range
    pub fn build_selection(&self, residue_indices: &[usize]) -> Result<AtomSelection, String> {
        // Validate indices
        for &res_idx in residue_indices {
            if res_idx >= self.n_residues {
                return Err(format!(
                    "Residue index {} out of range (0-{})",
                    res_idx,
                    self.n_residues - 1
                ));
            }
        }

        let mut atom_indices = Vec::new();
        let mut residue_offsets = Vec::with_capacity(residue_indices.len() + 1);
        let mut residue_labels = Vec::with_capacity(residue_indices.len());

        for &res_idx in residue_indices {
            residue_offsets.push(atom_indices.len());

            let start = self.residue_pointers[res_idx];
            let end = if res_idx + 1 < self.n_residues {
                self.residue_pointers[res_idx + 1]
            } else {
                self.n_atoms
            };

            for atom_idx in start..end {
                atom_indices.push(atom_idx);
            }

            residue_labels.push(self.residue_labels[res_idx].clone());
        }
        residue_offsets.push(atom_indices.len());

        Ok(AtomSelection {
            atom_indices,
            residue_offsets,
            residue_labels,
        })
    }

    /// Get all atom indices for a set of residues (flat list, no grouping).
    ///
    /// # Arguments
    /// * `residue_indices` - 0-based residue indices
    ///
    /// # Returns
    /// * `Ok(Vec<usize>)` - Flat list of atom indices
    /// * `Err(String)` - If any residue index is out of range
    pub fn get_atom_indices_for_residues(
        &self,
        residue_indices: &[usize],
    ) -> Result<Vec<usize>, String> {
        let mut atom_indices = Vec::new();

        for &res_idx in residue_indices {
            if res_idx >= self.n_residues {
                return Err(format!(
                    "Residue index {} out of range (0-{})",
                    res_idx,
                    self.n_residues - 1
                ));
            }

            let start = self.residue_pointers[res_idx];
            let end = if res_idx + 1 < self.n_residues {
                self.residue_pointers[res_idx + 1]
            } else {
                self.n_atoms
            };

            for atom_idx in start..end {
                atom_indices.push(atom_idx);
            }
        }

        Ok(atom_indices)
    }

    /// Get all bonds for a specific residue.
    ///
    /// Returns bonds where both atoms belong to the specified residue.
    ///
    /// # Arguments
    /// * `residue_idx` - 0-based residue index
    ///
    /// # Returns
    /// * `Ok(Vec<(usize, usize)>)` - List of (atom_i, atom_j) tuples
    /// * `Err(String)` - If residue index is out of range
    pub fn get_bonds_for_residue(&self, residue_idx: usize) -> Result<Vec<(usize, usize)>, String> {
        if residue_idx >= self.n_residues {
            return Err(format!(
                "Residue index {} out of range (0-{})",
                residue_idx,
                self.n_residues - 1
            ));
        }

        let start = self.residue_pointers[residue_idx];
        let end = if residue_idx + 1 < self.n_residues {
            self.residue_pointers[residue_idx + 1]
        } else {
            self.n_atoms
        };

        // Filter bonds where both atoms are in the residue
        let bonds: Vec<(usize, usize)> = self
            .bonds
            .iter()
            .filter(|(a, b)| *a >= start && *a < end && *b >= start && *b < end)
            .copied()
            .collect();

        Ok(bonds)
    }

    /// Get all bonds involving atoms in a set of residues.
    ///
    /// Returns bonds where at least one atom belongs to the specified residues.
    /// Useful for getting internal bonds within a ligand (multiple residues).
    ///
    /// # Arguments
    /// * `residue_indices` - 0-based residue indices
    ///
    /// # Returns
    /// * `Ok(Vec<(usize, usize)>)` - List of (atom_i, atom_j) tuples
    /// * `Err(String)` - If any residue index is out of range
    pub fn get_bonds_for_residues(
        &self,
        residue_indices: &[usize],
    ) -> Result<Vec<(usize, usize)>, String> {
        // Build set of atom indices for the specified residues
        let atom_set: std::collections::HashSet<usize> = self
            .get_atom_indices_for_residues(residue_indices)?
            .into_iter()
            .collect();

        // Filter bonds where both atoms are in the set
        let bonds: Vec<(usize, usize)> = self
            .bonds
            .iter()
            .filter(|(a, b)| atom_set.contains(a) && atom_set.contains(b))
            .copied()
            .collect();

        Ok(bonds)
    }
}

// ============================================================================
// Parser Implementation
// ============================================================================

/// Internal parser state for reading FLAG/FORMAT sections.
struct PrmtopParser {
    sections: HashMap<String, Vec<String>>,
}

impl PrmtopParser {
    fn new() -> Self {
        Self {
            sections: HashMap::new(),
        }
    }

    /// Parse the prmtop file and extract all sections.
    fn parse_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), String> {
        let file =
            File::open(path.as_ref()).map_err(|e| format!("Failed to open prmtop file: {}", e))?;
        let reader = BufReader::new(file);

        let mut current_flag: Option<String> = None;
        let mut current_data: Vec<String> = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;

            if let Some(flag_content) = line.strip_prefix("%FLAG") {
                // Save previous section if any
                if let Some(flag) = current_flag.take() {
                    self.sections.insert(flag, current_data);
                    current_data = Vec::new();
                }
                // Extract flag name
                let flag_name = flag_content.trim().to_string();
                current_flag = Some(flag_name);
            } else if line.starts_with("%FORMAT") {
                // Skip format line, we'll parse dynamically
                continue;
            } else if line.starts_with("%VERSION") || line.starts_with("%COMMENT") {
                // Skip header lines
                continue;
            } else if current_flag.is_some() {
                // Accumulate data lines
                current_data.push(line);
            }
        }

        // Save last section
        if let Some(flag) = current_flag {
            self.sections.insert(flag, current_data);
        }

        Ok(())
    }

    /// Parse integer values from a section.
    fn parse_integers(&self, flag: &str) -> Result<Vec<i64>, String> {
        let lines = self
            .sections
            .get(flag)
            .ok_or_else(|| format!("Missing section: {}", flag))?;

        let mut values = Vec::new();
        for line in lines {
            for word in line.split_whitespace() {
                let val: i64 = word
                    .parse()
                    .map_err(|e| format!("Failed to parse integer in {}: {}", flag, e))?;
                values.push(val);
            }
        }
        Ok(values)
    }

    /// Parse floating point values from a section.
    fn parse_floats(&self, flag: &str) -> Result<Vec<f64>, String> {
        let lines = self
            .sections
            .get(flag)
            .ok_or_else(|| format!("Missing section: {}", flag))?;

        let mut values = Vec::new();
        for line in lines {
            for word in line.split_whitespace() {
                let val: f64 = word
                    .parse()
                    .map_err(|e| format!("Failed to parse float in {}: {}", flag, e))?;
                values.push(val);
            }
        }
        Ok(values)
    }

    /// Parse fixed-width string values (like atom names).
    fn parse_strings(&self, flag: &str, width: usize) -> Result<Vec<String>, String> {
        let lines = self
            .sections
            .get(flag)
            .ok_or_else(|| format!("Missing section: {}", flag))?;

        let mut values = Vec::new();
        for line in lines {
            let mut pos = 0;
            while pos + width <= line.len() {
                let s = line[pos..pos + width].trim().to_string();
                values.push(s);
                pos += width;
            }
            // Handle any remaining partial field
            if pos < line.len() {
                let s = line[pos..].trim().to_string();
                if !s.is_empty() {
                    values.push(s);
                }
            }
        }
        Ok(values)
    }
}

/// Parse an AMBER prmtop file and return the topology data.
///
/// # Arguments
/// * `path` - Path to the prmtop file
///
/// # Returns
/// * `Ok(AmberTopology)` - Parsed topology with converted units
/// * `Err(String)` - Error message if parsing fails
pub fn parse_prmtop<P: AsRef<Path>>(path: P) -> Result<AmberTopology, String> {
    let mut parser = PrmtopParser::new();
    parser.parse_file(&path)?;

    // Get counts from POINTERS section
    let pointers = parser.parse_integers("POINTERS")?;
    if pointers.len() < 12 {
        return Err("POINTERS section too short".to_string());
    }

    if pointers[0] < 0 || pointers[1] < 0 || pointers[11] < 0 {
        return Err(format!(
            "POINTERS contains negative values: n_atoms={}, n_types={}, n_residues={}",
            pointers[0], pointers[1], pointers[11]
        ));
    }
    let n_atoms = pointers[0] as usize;
    let n_types = pointers[1] as usize;
    let n_residues = pointers[11] as usize;

    // Parse atom names (4 characters each)
    let atom_names = parser.parse_strings("ATOM_NAME", 4)?;
    if atom_names.len() < n_atoms {
        return Err(format!(
            "ATOM_NAME has {} entries, expected {}",
            atom_names.len(),
            n_atoms
        ));
    }

    // Parse atom type indices (1-based in file, convert to 0-based)
    let type_indices_raw = parser.parse_integers("ATOM_TYPE_INDEX")?;
    let mut atom_type_indices = Vec::with_capacity(n_atoms);
    for (i, &x) in type_indices_raw.iter().take(n_atoms).enumerate() {
        if x < 1 {
            return Err(format!(
                "Invalid ATOM_TYPE_INDEX at position {}: {} (must be >= 1)",
                i, x
            ));
        }
        atom_type_indices.push((x - 1) as usize);
    }

    // Parse charges and convert from AMBER units to elementary charge
    let charges_raw = parser.parse_floats("CHARGE")?;
    let charges: Vec<f64> = charges_raw
        .iter()
        .take(n_atoms)
        .map(|&q| q / AMBER_CHARGE_FACTOR)
        .collect();

    // Parse residue labels
    let residue_labels = parser.parse_strings("RESIDUE_LABEL", 4)?;

    // Parse residue pointers (1-based in file, convert to 0-based)
    let res_ptr_raw = parser.parse_integers("RESIDUE_POINTER")?;
    let mut residue_pointers = Vec::with_capacity(n_residues);
    for (i, &x) in res_ptr_raw.iter().take(n_residues).enumerate() {
        if x < 1 {
            return Err(format!(
                "Invalid RESIDUE_POINTER at position {}: {} (must be >= 1)",
                i, x
            ));
        }
        residue_pointers.push((x - 1) as usize);
    }

    // Parse Lennard-Jones parameters
    // LENNARD_JONES_ACOEF and LENNARD_JONES_BCOEF contain the combined parameters
    // A_ij = 4 * epsilon * sigma^12
    // B_ij = 4 * epsilon * sigma^6
    // For self-interactions (i=j): sigma_i = (A_ii/B_ii)^(1/6), epsilon_i = B_ii^2/(4*A_ii)

    // Store raw AMBER charges
    let charges_amber: Vec<f64> = charges_raw.iter().take(n_atoms).copied().collect();

    let acoef = parser.parse_floats("LENNARD_JONES_ACOEF")?;
    let bcoef = parser.parse_floats("LENNARD_JONES_BCOEF")?;
    let nb_parm_index = parser.parse_integers("NONBONDED_PARM_INDEX")?;

    // Extract per-type sigma and epsilon from self-interaction terms
    let mut lj_sigma = vec![0.0f64; n_types];
    let mut lj_epsilon = vec![0.0f64; n_types];

    for i in 0..n_types {
        let nb_idx = n_types * i + i;
        if nb_idx >= nb_parm_index.len() {
            return Err(format!(
                "NONBONDED_PARM_INDEX out of bounds: index {} >= length {}",
                nb_idx,
                nb_parm_index.len()
            ));
        }
        let raw_idx = nb_parm_index[nb_idx];
        if raw_idx < 0 {
            continue;
        }
        let idx = raw_idx as usize;
        if idx == 0 {
            continue;
        }
        let idx = idx - 1;

        let a_ii = acoef[idx];
        let b_ii = bcoef[idx];

        if b_ii.abs() > 1e-20 && a_ii.abs() > 1e-20 {
            let sigma6 = a_ii / b_ii;
            let sigma = sigma6.powf(1.0 / 6.0);
            let epsilon = b_ii * b_ii / (4.0 * a_ii);

            lj_sigma[i] = sigma * ANGSTROM_TO_NM;
            lj_epsilon[i] = epsilon * KCAL_TO_KJ;
        }
    }

    let atom_sigmas: Vec<f64> = atom_type_indices.iter().map(|&i| lj_sigma[i]).collect();
    let atom_epsilons: Vec<f64> = atom_type_indices.iter().map(|&i| lj_epsilon[i]).collect();

    // Parse bond information (triplets: atom_i*3, atom_j*3, bond_type_index)
    let mut bonds = Vec::new();
    let mut bond_types = Vec::new();

    if let Ok(bonds_h_raw) = parser.parse_integers("BONDS_INC_HYDROGEN") {
        for chunk in bonds_h_raw.chunks(3) {
            if chunk.len() == 3 {
                if chunk[0] < 0 || chunk[1] < 0 || chunk[2] < 1 {
                    continue;
                }
                bonds.push(((chunk[0] / 3) as usize, (chunk[1] / 3) as usize));
                bond_types.push((chunk[2] - 1) as usize);
            }
        }
    }

    if let Ok(bonds_heavy_raw) = parser.parse_integers("BONDS_WITHOUT_HYDROGEN") {
        for chunk in bonds_heavy_raw.chunks(3) {
            if chunk.len() == 3 {
                if chunk[0] < 0 || chunk[1] < 0 || chunk[2] < 1 {
                    continue;
                }
                bonds.push(((chunk[0] / 3) as usize, (chunk[1] / 3) as usize));
                bond_types.push((chunk[2] - 1) as usize);
            }
        }
    }

    // Parse per-atom properties
    let masses = parser
        .parse_floats("MASS")
        .unwrap_or_else(|_| vec![0.0; n_atoms]);
    let radii = parser
        .parse_floats("RADII")
        .unwrap_or_else(|_| vec![0.0; n_atoms]);
    let screen = parser
        .parse_floats("SCREEN")
        .unwrap_or_else(|_| vec![0.0; n_atoms]);

    // Parse bond parameters
    let bond_force_constants = parser
        .parse_floats("BOND_FORCE_CONSTANT")
        .unwrap_or_default();
    let bond_equil_values = parser.parse_floats("BOND_EQUIL_VALUE").unwrap_or_default();

    // Parse angle parameters
    let angle_force_constants = parser
        .parse_floats("ANGLE_FORCE_CONSTANT")
        .unwrap_or_default();
    let angle_equil_values = parser.parse_floats("ANGLE_EQUIL_VALUE").unwrap_or_default();

    // Parse dihedral parameters
    let dihedral_force_constants = parser
        .parse_floats("DIHEDRAL_FORCE_CONSTANT")
        .unwrap_or_default();
    let dihedral_periodicities = parser
        .parse_floats("DIHEDRAL_PERIODICITY")
        .unwrap_or_default();
    let dihedral_phases = parser.parse_floats("DIHEDRAL_PHASE").unwrap_or_default();

    // Parse angles (quads: i*3, j*3, k*3, type_index)
    let mut angles = Vec::new();
    for section in &["ANGLES_INC_HYDROGEN", "ANGLES_WITHOUT_HYDROGEN"] {
        if let Ok(raw) = parser.parse_integers(section) {
            for chunk in raw.chunks(4) {
                if chunk.len() == 4 {
                    if chunk[0] < 0 || chunk[1] < 0 || chunk[2] < 0 || chunk[3] < 1 {
                        continue;
                    }
                    angles.push((
                        (chunk[0] / 3) as usize,
                        (chunk[1] / 3) as usize,
                        (chunk[2] / 3) as usize,
                        (chunk[3] - 1) as usize,
                    ));
                }
            }
        }
    }

    // Parse dihedrals (quints: i*3, j*3, k*3, l*3, type_index)
    // Negative k → improper torsion (skip 1-4), negative l → multi-term (skip 1-4)
    let mut dihedrals = Vec::new();
    for section in &["DIHEDRALS_INC_HYDROGEN", "DIHEDRALS_WITHOUT_HYDROGEN"] {
        if let Ok(raw) = parser.parse_integers(section) {
            for chunk in raw.chunks(5) {
                if chunk.len() == 5 {
                    if chunk[4] < 1 {
                        continue;
                    }
                    let i = (chunk[0].unsigned_abs() / 3) as usize;
                    let j = (chunk[1].unsigned_abs() / 3) as usize;
                    let k = (chunk[2].unsigned_abs() / 3) as usize;
                    let l = (chunk[3].unsigned_abs() / 3) as usize;
                    let type_idx = (chunk[4] - 1) as usize;
                    // If k or l is negative, this is improper or multi-term → skip 1-4
                    let ignore_14 = chunk[2] < 0 || chunk[3] < 0;
                    dihedrals.push((i, j, k, l, type_idx, ignore_14));
                }
            }
        }
    }

    // Parse exclusion lists
    let num_excluded_atoms: Vec<usize> = parser
        .parse_integers("NUMBER_EXCLUDED_ATOMS")
        .unwrap_or_default()
        .into_iter()
        .map(|x| x as usize)
        .collect();

    // AMBER uses 1-based indices; a value of 0 is a placeholder meaning
    // "no excluded atoms" (used when NUMBER_EXCLUDED_ATOMS is 1 but
    // the atom has no real exclusions).  We convert to 0-based and mark
    // placeholders with usize::MAX so build_exclusion_set can skip them.
    let excluded_atoms_list: Vec<usize> = parser
        .parse_integers("EXCLUDED_ATOMS_LIST")
        .unwrap_or_default()
        .into_iter()
        .map(|x| if x > 0 { (x - 1) as usize } else { usize::MAX })
        .collect();

    // Parse optional 1-4 scaling factors
    let scee_scale_factor = parser
        .parse_floats("SCEE_SCALE_FACTOR")
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(1.2);
    let scnb_scale_factor = parser
        .parse_floats("SCNB_SCALE_FACTOR")
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(2.0);

    Ok(AmberTopology {
        n_atoms,
        n_residues,
        n_types,
        atom_names: atom_names.into_iter().take(n_atoms).collect(),
        atom_type_indices,
        charges,
        charges_amber,
        residue_labels: residue_labels.into_iter().take(n_residues).collect(),
        residue_pointers,
        lj_sigma: Arc::new(lj_sigma),
        lj_epsilon: Arc::new(lj_epsilon),
        atom_sigmas,
        atom_epsilons,
        bonds,
        bond_types,
        masses,
        radii,
        screen,
        bond_force_constants: Arc::new(bond_force_constants),
        bond_equil_values: Arc::new(bond_equil_values),
        angle_force_constants: Arc::new(angle_force_constants),
        angle_equil_values: Arc::new(angle_equil_values),
        dihedral_force_constants: Arc::new(dihedral_force_constants),
        dihedral_periodicities: Arc::new(dihedral_periodicities),
        dihedral_phases: Arc::new(dihedral_phases),
        angles,
        dihedrals,
        num_excluded_atoms,
        excluded_atoms_list,
        scee_scale_factor,
        scnb_scale_factor,
        lj_acoef: Arc::new(acoef),
        lj_bcoef: Arc::new(bcoef),
        nb_parm_index: Arc::new(nb_parm_index),
    })
}
