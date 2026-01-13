//! AMBER prmtop (topology) file parser.
//!
//! Parses AMBER7-format topology files to extract:
//! - Atom charges (converted to elementary charge units)
//! - Lennard-Jones parameters (sigma in nm, epsilon in kJ/mol)
//! - Residue information for mapping atoms to residues
//! - Atom names and types

use numpy::PyArray1;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// ============================================================================
// Unit Conversion Constants
// ============================================================================

/// AMBER charge to elementary charge: q_e = q_amber / 18.2223
const AMBER_CHARGE_FACTOR: f64 = 18.2223;

/// kcal/mol to kJ/mol
const KCAL_TO_KJ: f64 = 4.184;

/// Angstrom to nm
const ANGSTROM_TO_NM: f64 = 0.1;

// ============================================================================
// Data Structures
// ============================================================================

/// Parsed AMBER topology containing all relevant force field data.
#[derive(Debug, Clone)]
#[allow(dead_code)]
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
    /// Residue labels
    pub residue_labels: Vec<String>,
    /// Residue pointers: first atom index (0-based) for each residue
    pub residue_pointers: Vec<usize>,
    /// LJ sigma parameters in nm (per atom type)
    pub lj_sigma: Vec<f64>,
    /// LJ epsilon parameters in kJ/mol (per atom type)
    pub lj_epsilon: Vec<f64>,
    /// Per-atom sigma values (looked up from type)
    pub atom_sigmas: Vec<f64>,
    /// Per-atom epsilon values (looked up from type)
    pub atom_epsilons: Vec<f64>,
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
            for atom_idx in start..end {
                result[atom_idx] = res_idx;
            }
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
        let file = File::open(path.as_ref())
            .map_err(|e| format!("Failed to open prmtop file: {}", e))?;
        let reader = BufReader::new(file);

        let mut current_flag: Option<String> = None;
        let mut current_data: Vec<String> = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;

            if line.starts_with("%FLAG") {
                // Save previous section if any
                if let Some(flag) = current_flag.take() {
                    self.sections.insert(flag, current_data);
                    current_data = Vec::new();
                }
                // Extract flag name
                let flag_name = line[5..].trim().to_string();
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
    let atom_type_indices: Vec<usize> = type_indices_raw
        .iter()
        .take(n_atoms)
        .map(|&x| (x - 1) as usize)
        .collect();

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
    let residue_pointers: Vec<usize> = res_ptr_raw
        .iter()
        .take(n_residues)
        .map(|&x| (x - 1) as usize)
        .collect();

    // Parse Lennard-Jones parameters
    // LENNARD_JONES_ACOEF and LENNARD_JONES_BCOEF contain the combined parameters
    // A_ij = 4 * epsilon * sigma^12
    // B_ij = 4 * epsilon * sigma^6
    // For self-interactions (i=j): sigma_i = (A_ii/B_ii)^(1/6), epsilon_i = B_ii^2/(4*A_ii)

    let acoef = parser.parse_floats("LENNARD_JONES_ACOEF")?;
    let bcoef = parser.parse_floats("LENNARD_JONES_BCOEF")?;
    let nb_parm_index = parser.parse_integers("NONBONDED_PARM_INDEX")?;

    // Extract per-type sigma and epsilon from self-interaction terms
    let mut lj_sigma = vec![0.0f64; n_types];
    let mut lj_epsilon = vec![0.0f64; n_types];

    for i in 0..n_types {
        // Index into the packed triangular matrix for self-interaction
        // nb_parm_index is 1-based and stores n_types * (i-1) + j
        let idx = nb_parm_index[n_types * i + i] as usize;
        if idx == 0 {
            continue; // No LJ interaction for this type
        }
        let idx = idx - 1; // Convert to 0-based

        let a_ii = acoef[idx];
        let b_ii = bcoef[idx];

        if b_ii.abs() > 1e-20 && a_ii.abs() > 1e-20 {
            // sigma^6 = A/B, sigma = (A/B)^(1/6)
            // epsilon = B^2 / (4*A)
            let sigma6 = a_ii / b_ii;
            let sigma = sigma6.powf(1.0 / 6.0); // in Angstrom
            let epsilon = b_ii * b_ii / (4.0 * a_ii); // in kcal/mol

            lj_sigma[i] = sigma * ANGSTROM_TO_NM;
            lj_epsilon[i] = epsilon * KCAL_TO_KJ;
        }
    }

    // Build per-atom sigma and epsilon arrays
    let atom_sigmas: Vec<f64> = atom_type_indices.iter().map(|&i| lj_sigma[i]).collect();
    let atom_epsilons: Vec<f64> = atom_type_indices.iter().map(|&i| lj_epsilon[i]).collect();

    Ok(AmberTopology {
        n_atoms,
        n_residues,
        n_types,
        atom_names: atom_names.into_iter().take(n_atoms).collect(),
        atom_type_indices,
        charges,
        residue_labels: residue_labels.into_iter().take(n_residues).collect(),
        residue_pointers,
        lj_sigma,
        lj_epsilon,
        atom_sigmas,
        atom_epsilons,
    })
}

// ============================================================================
// Python Interface
// ============================================================================

/// Python-accessible wrapper for AmberTopology.
#[pyclass(name = "AmberTopology")]
pub struct PyAmberTopology {
    inner: AmberTopology,
}

#[pymethods]
impl PyAmberTopology {
    /// Number of atoms in the topology.
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms
    }

    /// Number of residues in the topology.
    #[getter]
    fn n_residues(&self) -> usize {
        self.inner.n_residues
    }

    /// Atom names as a list of strings.
    #[getter]
    fn atom_names(&self) -> Vec<String> {
        self.inner.atom_names.clone()
    }

    /// Residue labels as a list of strings.
    #[getter]
    fn residue_labels(&self) -> Vec<String> {
        self.inner.residue_labels.clone()
    }

    /// Partial charges in elementary charge units.
    fn charges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.charges.clone())
    }

    /// Per-atom LJ sigma values in nm.
    fn sigmas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.atom_sigmas.clone())
    }

    /// Per-atom LJ epsilon values in kJ/mol.
    fn epsilons<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.atom_epsilons.clone())
    }

    /// Get residue index for each atom.
    fn atom_residue_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let indices: Vec<i64> = self
            .inner
            .atom_residue_indices()
            .into_iter()
            .map(|x| x as i64)
            .collect();
        PyArray1::from_vec_bound(py, indices)
    }

    /// Build resmap for fingerprint calculations.
    /// Returns (indices, offsets) arrays.
    fn build_resmap<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
        let (indices, offsets) = self.inner.build_resmap();
        (
            PyArray1::from_vec_bound(py, indices),
            PyArray1::from_vec_bound(py, offsets),
        )
    }

    /// First atom index (0-based) for each residue.
    fn residue_pointers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let ptrs: Vec<i64> = self
            .inner
            .residue_pointers
            .iter()
            .map(|&x| x as i64)
            .collect();
        PyArray1::from_vec_bound(py, ptrs)
    }
}

/// Parse an AMBER prmtop file.
///
/// Parameters
/// ----------
/// path : str
///     Path to the prmtop file.
///
/// Returns
/// -------
/// AmberTopology
///     Parsed topology with charges, LJ parameters, and residue mapping.
///     Units are converted: charges to e, sigma to nm, epsilon to kJ/mol.
#[pyfunction]
#[pyo3(name = "read_prmtop")]
pub fn read_prmtop_py(path: &str) -> PyResult<PyAmberTopology> {
    let topo = parse_prmtop(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
    Ok(PyAmberTopology { inner: topo })
}
