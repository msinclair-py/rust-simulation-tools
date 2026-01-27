//! High-level fingerprint session for simplified trajectory processing.
//!
//! Provides a `FingerprintSession` class that integrates topology and DCD reading
//! under the hood, reducing Python boilerplate from 25+ lines to ~5 lines.
//! Supports fingerprinting either target or binder selections.

use crate::amber::prmtop::{parse_prmtop, AmberTopology, AtomSelection};
use crate::fingerprint::{compute_fingerprints_from_residues, PartnerData, ResidueData};
use crate::trajectory::dcd::DcdReader;
use numpy::ndarray::Array1;
use numpy::PyArray1;
use pyo3::prelude::*;
use std::collections::HashSet;

// ============================================================================
// Fingerprint Mode
// ============================================================================

/// Which selection to compute per-residue fingerprints for.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FingerprintMode {
    /// Fingerprint target residues (default): per-residue energies for target
    #[default]
    Target,
    /// Fingerprint binder residues: per-residue energies for binder
    Binder,
}

/// Python-accessible fingerprint mode enum.
#[pyclass(name = "FingerprintMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyFingerprintMode {
    Target = 0,
    Binder = 1,
}

impl From<PyFingerprintMode> for FingerprintMode {
    fn from(mode: PyFingerprintMode) -> Self {
        match mode {
            PyFingerprintMode::Target => FingerprintMode::Target,
            PyFingerprintMode::Binder => FingerprintMode::Binder,
        }
    }
}

// ============================================================================
// Fingerprint Session
// ============================================================================

/// High-level fingerprint calculator with integrated topology and trajectory support.
///
/// This class simplifies fingerprint calculations by:
/// 1. Automatically extracting force field parameters from topology
/// 2. Managing DCD trajectory reading internally
/// 3. Supporting both target and binder fingerprinting modes
///
/// Example
/// -------
/// ```python
/// from rust_simulation_tools import FingerprintSession, FingerprintMode
///
/// session = FingerprintSession("system.prmtop", "trajectory.dcd")
/// session.set_target_residues(range(10))
/// session.set_binder_residues(range(10, 100))
///
/// # Fingerprint target (default)
/// for lj_fp, es_fp in session:
///     process(lj_fp, es_fp)
///
/// # Fingerprint binder instead
/// session.set_fingerprint_mode(FingerprintMode.Binder)
/// session.seek(0)
/// for lj_fp, es_fp in session:
///     process_binder(lj_fp, es_fp)
/// ```
#[pyclass(name = "FingerprintSession")]
pub struct PyFingerprintSession {
    /// Parsed topology
    topology: AmberTopology,
    /// DCD trajectory reader
    dcd_reader: Option<DcdReader>,

    /// Pre-extracted charges for all atoms
    charges: Vec<f64>,
    /// Pre-extracted sigmas for all atoms
    sigmas: Vec<f64>,
    /// Pre-extracted epsilons for all atoms
    epsilons: Vec<f64>,

    /// Target residue selection (what we call "target")
    target_selection: Option<AtomSelection>,
    /// Binder residue selection (what we call "binder")
    binder_selection: Option<AtomSelection>,

    /// Current fingerprinting mode
    mode: FingerprintMode,
}

#[pymethods]
impl PyFingerprintSession {
    /// Create a new fingerprint session.
    ///
    /// Parameters
    /// ----------
    /// topology_path : str
    ///     Path to AMBER prmtop file.
    /// dcd_path : str, optional
    ///     Path to DCD trajectory file. If not provided, positions must be
    ///     supplied manually via compute_frame().
    #[new]
    #[pyo3(signature = (topology_path, dcd_path=None))]
    fn new(topology_path: &str, dcd_path: Option<&str>) -> PyResult<Self> {
        let topology = parse_prmtop(topology_path).map_err(pyo3::exceptions::PyIOError::new_err)?;

        // Pre-extract force field parameters
        let charges = topology.charges.clone();
        let sigmas = topology.atom_sigmas.clone();
        let epsilons = topology.atom_epsilons.clone();

        // Open DCD if provided
        let dcd_reader = if let Some(path) = dcd_path {
            let reader = DcdReader::open(path).map_err(pyo3::exceptions::PyIOError::new_err)?;

            // Validate atom count matches
            if reader.n_atoms() != topology.n_atoms {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "DCD has {} atoms but topology has {} atoms",
                    reader.n_atoms(),
                    topology.n_atoms
                )));
            }

            Some(reader)
        } else {
            None
        };

        Ok(Self {
            topology,
            dcd_reader,
            charges,
            sigmas,
            epsilons,
            target_selection: None,
            binder_selection: None,
            mode: FingerprintMode::Target,
        })
    }

    /// Number of atoms in the system.
    #[getter]
    fn n_atoms(&self) -> usize {
        self.topology.n_atoms
    }

    /// Number of residues in the system.
    #[getter]
    fn n_residues(&self) -> usize {
        self.topology.n_residues
    }

    /// Number of frames in the trajectory (0 if no DCD loaded).
    #[getter]
    fn n_frames(&self) -> usize {
        self.dcd_reader.as_ref().map_or(0, |r| r.n_frames())
    }

    /// Current frame index in the trajectory.
    #[getter]
    fn current_frame(&self) -> usize {
        self.dcd_reader.as_ref().map_or(0, |r| r.current_frame())
    }

    /// Residue labels for the current fingerprinting mode.
    #[getter]
    fn residue_labels(&self) -> PyResult<Vec<String>> {
        let selection = match self.mode {
            FingerprintMode::Target => self.target_selection.as_ref(),
            FingerprintMode::Binder => self.binder_selection.as_ref(),
        };

        selection.map(|s| s.residue_labels.clone()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Selection not set. Call set_target_residues() or set_binder_residues() first.",
            )
        })
    }

    /// Number of residues being fingerprinted in current mode.
    #[getter]
    fn n_fingerprint_residues(&self) -> PyResult<usize> {
        let selection = match self.mode {
            FingerprintMode::Target => self.target_selection.as_ref(),
            FingerprintMode::Binder => self.binder_selection.as_ref(),
        };

        selection.map(|s| s.residue_labels.len()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Selection not set. Call set_target_residues() or set_binder_residues() first.",
            )
        })
    }

    /// Set the target residue selection.
    ///
    /// Parameters
    /// ----------
    /// residue_indices : list or range
    ///     0-based residue indices for the target selection.
    fn set_target_residues(&mut self, residue_indices: Vec<usize>) -> PyResult<()> {
        let selection = self
            .topology
            .build_selection(&residue_indices)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Check for overlap with binder if already set
        if let Some(ref binder) = self.binder_selection {
            let target_atoms: HashSet<usize> = selection.atom_indices.iter().copied().collect();
            let binder_atoms: HashSet<usize> = binder.atom_indices.iter().copied().collect();

            if !target_atoms.is_disjoint(&binder_atoms) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Target and binder selections must not overlap",
                ));
            }
        }

        self.target_selection = Some(selection);
        Ok(())
    }

    /// Set the binder residue selection.
    ///
    /// Parameters
    /// ----------
    /// residue_indices : list or range
    ///     0-based residue indices for the binder selection.
    fn set_binder_residues(&mut self, residue_indices: Vec<usize>) -> PyResult<()> {
        let selection = self
            .topology
            .build_selection(&residue_indices)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Check for overlap with target if already set
        if let Some(ref target) = self.target_selection {
            let target_atoms: HashSet<usize> = target.atom_indices.iter().copied().collect();
            let binder_atoms: HashSet<usize> = selection.atom_indices.iter().copied().collect();

            if !target_atoms.is_disjoint(&binder_atoms) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Target and binder selections must not overlap",
                ));
            }
        }

        self.binder_selection = Some(selection);
        Ok(())
    }

    /// Set the fingerprinting mode.
    ///
    /// Parameters
    /// ----------
    /// mode : FingerprintMode
    ///     FingerprintMode.Target to fingerprint target residues (default),
    ///     FingerprintMode.Binder to fingerprint binder residues.
    fn set_fingerprint_mode(&mut self, mode: PyFingerprintMode) {
        self.mode = mode.into();
    }

    /// Get the current fingerprinting mode.
    #[getter]
    fn fingerprint_mode(&self) -> PyFingerprintMode {
        match self.mode {
            FingerprintMode::Target => PyFingerprintMode::Target,
            FingerprintMode::Binder => PyFingerprintMode::Binder,
        }
    }

    /// Seek to a specific frame in the trajectory.
    ///
    /// Parameters
    /// ----------
    /// frame : int
    ///     Frame index to seek to (0-based).
    fn seek(&mut self, frame: usize) -> PyResult<()> {
        let reader = self
            .dcd_reader
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No DCD trajectory loaded"))?;

        reader
            .seek_frame(frame)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Compute fingerprints for the next frame.
    ///
    /// Returns
    /// -------
    /// tuple or None
    ///     (lj_fingerprint, es_fingerprint) arrays in kJ/mol, or None if at end.
    fn compute_next_frame<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>> {
        // Validate selections
        let target = self.target_selection.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Target selection not set. Call set_target_residues() first.",
            )
        })?;

        let binder = self.binder_selection.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Binder selection not set. Call set_binder_residues() first.",
            )
        })?;

        // Read next frame
        let reader = self
            .dcd_reader
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No DCD trajectory loaded"))?;

        let frame_result = reader
            .read_frame()
            .map_err(pyo3::exceptions::PyIOError::new_err)?;

        let (positions, _box_info) = match frame_result {
            Some(data) => data,
            None => return Ok(None),
        };

        // Compute fingerprints
        let (lj_fp, es_fp) = self.compute_fingerprints_for_positions(&positions, target, binder);

        Ok(Some((
            PyArray1::from_owned_array_bound(py, lj_fp),
            PyArray1::from_owned_array_bound(py, es_fp),
        )))
    }

    /// Make the session iterable over frames.
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    /// Get next frame's fingerprints (iterator protocol).
    fn __next__<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>> {
        slf.compute_next_frame(py)
    }
}

impl PyFingerprintSession {
    /// Internal method to compute fingerprints for given positions.
    fn compute_fingerprints_for_positions(
        &self,
        positions: &[[f64; 3]],
        target: &AtomSelection,
        binder: &AtomSelection,
    ) -> (Array1<f64>, Array1<f64>) {
        // Determine primary (what we're fingerprinting) and partner (what we interact with)
        let (primary, partner_indices) = match self.mode {
            FingerprintMode::Target => (target, &binder.atom_indices),
            FingerprintMode::Binder => (binder, &target.atom_indices),
        };

        // Build partner data (cell list for efficient neighbor lookup)
        let partner_positions: Vec<[f64; 3]> =
            partner_indices.iter().map(|&idx| positions[idx]).collect();
        let partner_charges: Vec<f64> = partner_indices
            .iter()
            .map(|&idx| self.charges[idx])
            .collect();
        let partner_sigmas: Vec<f64> = partner_indices
            .iter()
            .map(|&idx| self.sigmas[idx])
            .collect();
        let partner_epsilons: Vec<f64> = partner_indices
            .iter()
            .map(|&idx| self.epsilons[idx])
            .collect();

        let partner_data = PartnerData::from_vecs(
            partner_positions,
            partner_charges,
            partner_sigmas,
            partner_epsilons,
        );

        // Build residue data for each primary residue
        let n_residues = primary.residue_labels.len();
        let residue_data: Vec<ResidueData> = (0..n_residues)
            .map(|res_idx| {
                let start = primary.residue_offsets[res_idx];
                let end = primary.residue_offsets[res_idx + 1];
                let atom_indices = &primary.atom_indices[start..end];

                ResidueData::from_global_indices(
                    positions,
                    &self.charges,
                    &self.sigmas,
                    &self.epsilons,
                    atom_indices,
                )
            })
            .collect();

        // Compute fingerprints
        compute_fingerprints_from_residues(&residue_data, &partner_data)
    }
}
