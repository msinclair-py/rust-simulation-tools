#![allow(clippy::type_complexity)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::too_many_arguments)]

use ndarray::{Array2, Array3};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayDescrMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use rst_core::amber::inpcrd::parse_inpcrd;
use rst_core::amber::prmtop::{parse_prmtop, AmberTopology};
use rst_core::fingerprint::{compute_fingerprints_from_residues, PartnerData, ResidueData};
use rst_core::kabsch::kabsch_align;
use rst_core::sasa::{get_vdw_radius, SASAEngine};
use rst_core::trajectory::dcd::{read_dcd_header, DcdReader};
use rst_core::wrapping::unwrap_system;

use rst_mmpbsa::binding::{self, BindingConfig, TrajectoryFormat};
use rst_mmpbsa::decomposition;
use rst_mmpbsa::entropy;
use rst_mmpbsa::gb_energy::{self, GbModel, GbParams};
use rst_mmpbsa::mdcrd::MdcrdReader;
use rst_mmpbsa::mm_energy;
use rst_mmpbsa::sa_energy::{self, SaParams};
use rst_mmpbsa::subsystem;

// ============================================================================
// Helper: convert ndarray3 to Vec<Vec<[f64;3]>>
// ============================================================================

fn array3_to_trajectory(arr: &ndarray::ArrayView3<f64>) -> Vec<Vec<[f64; 3]>> {
    let n_frames = arr.shape()[0];
    let n_atoms = arr.shape()[1];
    let mut traj = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let mut frame = Vec::with_capacity(n_atoms);
        for j in 0..n_atoms {
            frame.push([arr[[i, j, 0]], arr[[i, j, 1]], arr[[i, j, 2]]]);
        }
        traj.push(frame);
    }
    traj
}

fn trajectory_to_array3(traj: &[Vec<[f64; 3]>]) -> Array3<f64> {
    let n_frames = traj.len();
    let n_atoms = if n_frames > 0 { traj[0].len() } else { 0 };
    let mut result = Array3::<f64>::zeros((n_frames, n_atoms, 3));
    for (i, frame) in traj.iter().enumerate() {
        for (j, atom) in frame.iter().enumerate() {
            result[[i, j, 0]] = atom[0];
            result[[i, j, 1]] = atom[1];
            result[[i, j, 2]] = atom[2];
        }
    }
    result
}

fn array2_to_coords(arr: &ndarray::ArrayView2<f64>) -> Vec<[f64; 3]> {
    let n = arr.shape()[0];
    (0..n)
        .map(|i| [arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]])
        .collect()
}

/// Convert coordinates from nm (rst_core convention) to Angstroms (AMBER/mmpbsa convention).
fn coords_nm_to_angstrom(coords: &[[f64; 3]]) -> Vec<[f64; 3]> {
    coords
        .iter()
        .map(|c| [c[0] * 10.0, c[1] * 10.0, c[2] * 10.0])
        .collect()
}

// ============================================================================
// KABSCH ALIGNMENT
// ============================================================================

#[pyfunction]
#[pyo3(name = "kabsch_align", signature = (trajectory, reference, align_indices))]
fn kabsch_align_py<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray3<'py, f64>,
    reference: PyReadonlyArray2<'py, f64>,
    align_indices: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let traj_arr = trajectory.as_array();
    let ref_arr = reference.as_array();

    let traj_vec = array3_to_trajectory(&traj_arr);
    let ref_vec = array2_to_coords(&ref_arr);

    let align_idx: Vec<usize> = align_indices
        .as_array()
        .iter()
        .map(|&i| i as usize)
        .collect();

    let aligned = kabsch_align(&traj_vec, &ref_vec, &align_idx);

    Ok(trajectory_to_array3(&aligned).to_pyarray_bound(py))
}

// ============================================================================
// UNWRAPPING
// ============================================================================

#[pyfunction]
#[pyo3(name = "unwrap_system", signature = (trajectory, box_dimensions, fragment_indices=None))]
fn unwrap_system_py<'py>(
    py: Python<'py>,
    trajectory: &Bound<'py, numpy::PyUntypedArray>,
    box_dimensions: &Bound<'py, numpy::PyUntypedArray>,
    #[allow(unused_variables)] fragment_indices: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<PyObject> {
    // Check if input is f32
    let is_f32 = trajectory
        .dtype()
        .is_equiv_to(&numpy::dtype_bound::<f32>(py));

    // Convert to f64 for processing
    let traj_f64: PyReadonlyArray3<'py, f64> = if is_f32 {
        let arr = trajectory.call_method1("astype", (numpy::dtype_bound::<f64>(py),))?;
        arr.extract()?
    } else {
        trajectory.extract()?
    };
    let box_f64: PyReadonlyArray2<'py, f64> = if is_f32 {
        let arr = box_dimensions.call_method1("astype", (numpy::dtype_bound::<f64>(py),))?;
        arr.extract()?
    } else {
        box_dimensions.extract()?
    };

    let traj_arr = traj_f64.as_array();
    let box_arr = box_f64.as_array();
    let n_frames = traj_arr.shape()[0];

    let traj_vec = array3_to_trajectory(&traj_arr);

    let box_vec: Vec<[f64; 3]> = (0..n_frames)
        .map(|i| [box_arr[[i, 0]], box_arr[[i, 1]], box_arr[[i, 2]]])
        .collect();

    let unwrapped = unwrap_system(&traj_vec, &box_vec).map_err(|e| PyValueError::new_err(e))?;

    let result_f64 = trajectory_to_array3(&unwrapped).to_pyarray_bound(py);

    if is_f32 {
        let result_f32 = result_f64.call_method1("astype", (numpy::dtype_bound::<f32>(py),))?;
        Ok(result_f32.into())
    } else {
        Ok(result_f64.into())
    }
}

// ============================================================================
// SASA
// ============================================================================

#[pyfunction]
#[pyo3(signature = (coordinates, radii, residue_indices, probe_radius=1.4, n_sphere_points=960))]
fn calculate_sasa<'py>(
    py: Python<'py>,
    coordinates: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    residue_indices: PyReadonlyArray1<'py, i64>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let coords_vec = array2_to_coords(&coordinates.as_array());
    let radii_slice = radii.as_array();
    let radii_vec: Vec<f64> = radii_slice.iter().copied().collect();
    let res_idx: Vec<usize> = residue_indices
        .as_array()
        .iter()
        .map(|&x| x as usize)
        .collect();

    let engine = SASAEngine::new(
        &coords_vec,
        &radii_vec,
        &res_idx,
        probe_radius,
        n_sphere_points,
    );

    let per_atom = engine.calculate_per_atom_sasa();
    let per_residue_map = engine.calculate_per_residue_sasa();

    // Convert HashMap to sorted Vec<f64>
    let per_residue = hashmap_to_sorted_vec(&per_residue_map);

    let total: f64 = per_atom.iter().sum();

    let result = PyDict::new_bound(py);
    result.set_item("per_atom", PyArray1::from_vec_bound(py, per_atom))?;
    result.set_item("per_residue", PyArray1::from_vec_bound(py, per_residue))?;
    result.set_item("total", total)?;

    Ok(result)
}

fn hashmap_to_sorted_vec(map: &std::collections::HashMap<usize, f64>) -> Vec<f64> {
    if map.is_empty() {
        return Vec::new();
    }
    let max_key = *map.keys().max().unwrap();
    let mut vec = vec![0.0; max_key + 1];
    for (&k, &v) in map {
        vec[k] = v;
    }
    vec
}

#[pyfunction]
#[pyo3(signature = (trajectory, radii, residue_indices, probe_radius=1.4, n_sphere_points=960))]
fn calculate_sasa_trajectory<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray3<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    residue_indices: PyReadonlyArray1<'py, i64>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let traj = trajectory.as_array();
    let n_frames = traj.shape()[0];
    let n_atoms = traj.shape()[1];

    let radii_vec: Vec<f64> = radii.as_array().iter().copied().collect();
    let res_idx: Vec<usize> = residue_indices
        .as_array()
        .iter()
        .map(|&x| x as usize)
        .collect();
    let mut all_per_atom: Vec<f64> = Vec::with_capacity(n_frames * n_atoms);
    let mut totals: Vec<f64> = Vec::with_capacity(n_frames);
    let mut per_residue_list: Vec<Bound<'py, PyDict>> = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let coords: Vec<[f64; 3]> = (0..n_atoms)
            .map(|i| {
                [
                    traj[[frame_idx, i, 0]],
                    traj[[frame_idx, i, 1]],
                    traj[[frame_idx, i, 2]],
                ]
            })
            .collect();

        let engine = SASAEngine::new(&coords, &radii_vec, &res_idx, probe_radius, n_sphere_points);
        let per_atom = engine.calculate_per_atom_sasa();
        let per_residue_map = engine.calculate_per_residue_sasa();

        let total: f64 = per_atom.iter().sum();
        totals.push(total);
        all_per_atom.extend_from_slice(&per_atom);

        let frame_dict = PyDict::new_bound(py);
        for (&k, &v) in &per_residue_map {
            frame_dict.set_item(k, v)?;
        }
        per_residue_list.push(frame_dict);
    }

    let result = PyDict::new_bound(py);
    result.set_item("per_atom", PyArray1::from_vec_bound(py, all_per_atom))?;
    result.set_item("per_residue", PyList::new_bound(py, &per_residue_list))?;
    result.set_item("total", PyArray1::from_vec_bound(py, totals))?;

    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (coordinates, radii, residue_indices, probe_radius=1.4, n_sphere_points=960))]
fn calculate_residue_sasa<'py>(
    py: Python<'py>,
    coordinates: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    residue_indices: PyReadonlyArray1<'py, i64>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let coords_vec = array2_to_coords(&coordinates.as_array());
    let radii_vec: Vec<f64> = radii.as_array().iter().copied().collect();
    let res_idx: Vec<usize> = residue_indices
        .as_array()
        .iter()
        .map(|&x| x as usize)
        .collect();

    let engine = SASAEngine::new(
        &coords_vec,
        &radii_vec,
        &res_idx,
        probe_radius,
        n_sphere_points,
    );
    let per_residue_map = engine.calculate_per_residue_sasa();
    let per_residue = hashmap_to_sorted_vec(&per_residue_map);

    Ok(PyArray1::from_vec_bound(py, per_residue))
}

#[pyfunction]
#[pyo3(signature = (coordinates, radii, probe_radius=1.4, n_sphere_points=960))]
fn calculate_total_sasa(
    coordinates: PyReadonlyArray2<f64>,
    radii: PyReadonlyArray1<f64>,
    probe_radius: f64,
    n_sphere_points: usize,
) -> PyResult<f64> {
    let coords_vec = array2_to_coords(&coordinates.as_array());
    let radii_vec: Vec<f64> = radii.as_array().iter().copied().collect();
    let res_idx: Vec<usize> = (0..coords_vec.len()).collect();

    let engine = SASAEngine::new(
        &coords_vec,
        &radii_vec,
        &res_idx,
        probe_radius,
        n_sphere_points,
    );
    Ok(engine.calculate_total_sasa())
}

#[pyfunction]
#[pyo3(name = "get_vdw_radius")]
fn get_vdw_radius_py(element: &str) -> f64 {
    get_vdw_radius(element)
}

#[pyfunction]
fn get_radii_array<'py>(py: Python<'py>, elements: Vec<String>) -> Bound<'py, PyArray1<f64>> {
    let radii: Vec<f64> = elements.iter().map(|e| get_vdw_radius(e)).collect();
    PyArray1::from_vec_bound(py, radii)
}

// ============================================================================
// FINGERPRINTS
// ============================================================================

#[pyfunction]
#[pyo3(name = "compute_fingerprints")]
fn compute_fingerprints_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    charges: PyReadonlyArray1<'py, f64>,
    sigmas: PyReadonlyArray1<'py, f64>,
    epsilons: PyReadonlyArray1<'py, f64>,
    resmap_indices: PyReadonlyArray1<'py, i64>,
    resmap_offsets: PyReadonlyArray1<'py, i64>,
    binder_indices: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let positions_vec = array2_to_coords(&positions.as_array());
    let charges_arr = charges.as_array();
    let sigmas_arr = sigmas.as_array();
    let epsilons_arr = epsilons.as_array();
    let indices_arr = resmap_indices.as_array();
    let offsets_arr = resmap_offsets.as_array();
    let binder_arr = binder_indices.as_array();

    let charges_slice = charges_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Charges array must be contiguous"))?;
    let sigmas_slice = sigmas_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Sigmas array must be contiguous"))?;
    let epsilons_slice = epsilons_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Epsilons array must be contiguous"))?;
    let indices_slice = indices_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Resmap indices array must be contiguous"))?;
    let offsets_slice = offsets_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Resmap offsets array must be contiguous"))?;
    let binder_slice = binder_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Binder indices array must be contiguous"))?;

    // Build ResidueData for each residue
    let n_residues = offsets_slice.len() - 1;
    let mut residues = Vec::with_capacity(n_residues);

    for r in 0..n_residues {
        let start = offsets_slice[r] as usize;
        let end = offsets_slice[r + 1] as usize;
        let atom_indices: Vec<usize> = (start..end).map(|k| indices_slice[k] as usize).collect();

        let residue = ResidueData::from_global_indices(
            &positions_vec,
            charges_slice,
            sigmas_slice,
            epsilons_slice,
            &atom_indices,
        );
        residues.push(residue);
    }

    // Build PartnerData
    let binder_indices_usize: Vec<usize> = binder_slice.iter().map(|&i| i as usize).collect();
    let partner = PartnerData::new(
        &positions_vec,
        charges_slice,
        sigmas_slice,
        epsilons_slice,
        &binder_indices_usize,
    );

    let (elec, vdw) = compute_fingerprints_from_residues(&residues, &partner);

    Ok((
        PyArray1::from_vec_bound(py, vdw),
        PyArray1::from_vec_bound(py, elec),
    ))
}

// ============================================================================
// AMBER PRMTOP
// ============================================================================

#[pyclass(name = "AmberTopology")]
struct PyAmberTopology {
    inner: AmberTopology,
}

#[pymethods]
impl PyAmberTopology {
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms
    }

    #[getter]
    fn n_residues(&self) -> usize {
        self.inner.n_residues
    }

    #[getter]
    fn atom_names<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::new_bound(py, &self.inner.atom_names)
    }

    #[getter]
    fn residue_labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::new_bound(py, &self.inner.residue_labels)
    }

    fn charges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.charges.clone())
    }

    fn sigmas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.atom_sigmas.clone())
    }

    fn epsilons<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.atom_epsilons.clone())
    }

    fn atom_residue_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let indices = self.inner.atom_residue_indices();
        let i64_indices: Vec<i64> = indices.iter().map(|&x| x as i64).collect();
        PyArray1::from_vec_bound(py, i64_indices)
    }

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

    fn residue_pointers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let ptrs: Vec<i64> = self
            .inner
            .residue_pointers
            .iter()
            .map(|&x| x as i64)
            .collect();
        PyArray1::from_vec_bound(py, ptrs)
    }

    fn build_selection<'py>(
        &self,
        py: Python<'py>,
        residue_indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let selection = self
            .inner
            .build_selection(&residue_indices)
            .map_err(|e| PyValueError::new_err(e))?;

        let dict = PyDict::new_bound(py);
        let atom_indices: Vec<i64> = selection.atom_indices.iter().map(|&x| x as i64).collect();
        let offsets: Vec<i64> = selection
            .residue_offsets
            .iter()
            .map(|&x| x as i64)
            .collect();
        dict.set_item("atom_indices", PyArray1::from_vec_bound(py, atom_indices))?;
        dict.set_item("residue_offsets", PyArray1::from_vec_bound(py, offsets))?;
        dict.set_item(
            "residue_labels",
            PyList::new_bound(py, &selection.residue_labels),
        )?;

        Ok(dict)
    }

    fn get_atom_indices<'py>(
        &self,
        py: Python<'py>,
        residue_indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let indices = self
            .inner
            .get_atom_indices_for_residues(&residue_indices)
            .map_err(|e| PyValueError::new_err(e))?;
        let i64_indices: Vec<i64> = indices.iter().map(|&x| x as i64).collect();
        Ok(PyArray1::from_vec_bound(py, i64_indices))
    }

    #[pyo3(signature = (expression, coordinates=None))]
    fn select_atoms(
        &self,
        expression: &str,
        coordinates: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Vec<usize>> {
        match coordinates {
            Some(coords) => {
                let coords_vec = array2_to_coords(&coords.as_array());
                rst_core::selection::select_with_coordinates(&self.inner, expression, &coords_vec)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }
            None => rst_core::selection::select(&self.inner, expression)
                .map_err(|e| PyValueError::new_err(e.to_string())),
        }
    }

    fn bonds(&self) -> Vec<(usize, usize)> {
        self.inner.bonds.clone()
    }

    fn get_bonds_for_residue(&self, residue_idx: usize) -> PyResult<Vec<(usize, usize)>> {
        self.inner
            .get_bonds_for_residue(residue_idx)
            .map_err(|e| PyValueError::new_err(e))
    }

    fn get_bonds_for_residues(&self, residue_indices: Vec<usize>) -> PyResult<Vec<(usize, usize)>> {
        self.inner
            .get_bonds_for_residues(&residue_indices)
            .map_err(|e| PyValueError::new_err(e))
    }
}

#[pyfunction]
#[pyo3(name = "read_prmtop")]
fn read_prmtop_py(path: &str) -> PyResult<PyAmberTopology> {
    let inner = parse_prmtop(path).map_err(|e| PyIOError::new_err(e))?;
    Ok(PyAmberTopology { inner })
}

// ============================================================================
// AMBER INPCRD
// ============================================================================

#[pyfunction]
#[pyo3(name = "read_inpcrd")]
fn read_inpcrd_py<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Option<Vec<f64>>)> {
    let coords = parse_inpcrd(path).map_err(|e| PyIOError::new_err(e))?;

    let n_atoms = coords.positions.len();
    let mut coord_array = Array2::<f64>::zeros((n_atoms, 3));
    for (i, &[x, y, z]) in coords.positions.iter().enumerate() {
        coord_array[[i, 0]] = x;
        coord_array[[i, 1]] = y;
        coord_array[[i, 2]] = z;
    }

    let box_dims = coords.box_dimensions.map(|b| vec![b[0], b[1], b[2]]);

    Ok((coord_array.to_pyarray_bound(py), box_dims))
}

// ============================================================================
// DCD TRAJECTORY
// ============================================================================

#[pyclass(name = "DcdReader")]
struct PyDcdReader {
    reader: DcdReader,
}

#[pymethods]
impl PyDcdReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let reader = DcdReader::open(path).map_err(|e| PyIOError::new_err(e))?;
        Ok(PyDcdReader { reader })
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.reader.n_frames()
    }

    #[getter]
    fn n_atoms(&self) -> usize {
        self.reader.n_atoms()
    }

    #[getter]
    fn has_unit_cell(&self) -> bool {
        self.reader.header().has_unit_cell
    }

    #[getter]
    fn current_frame(&self) -> usize {
        self.reader.current_frame()
    }

    fn seek(&mut self, frame: usize) -> PyResult<()> {
        self.reader
            .seek_frame(frame)
            .map_err(|e| PyIOError::new_err(e))
    }

    fn read_frame<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Option<Vec<f64>>)>> {
        match self.reader.read_frame() {
            Ok(Some((coords, box_info))) => {
                let n_atoms = coords.len();
                let mut coord_array = Array2::<f64>::zeros((n_atoms, 3));
                for (i, &[x, y, z]) in coords.iter().enumerate() {
                    coord_array[[i, 0]] = x;
                    coord_array[[i, 1]] = y;
                    coord_array[[i, 2]] = z;
                }

                let box_list = box_info.map(|b| vec![b[0], b[1], b[2], b[3], b[4], b[5]]);

                Ok(Some((coord_array.to_pyarray_bound(py), box_list)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyIOError::new_err(e)),
        }
    }

    fn read_frame_at<'py>(
        &mut self,
        py: Python<'py>,
        frame_index: usize,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Option<Vec<f64>>)>> {
        self.seek(frame_index)?;
        self.read_frame(py)
    }

    fn read_all<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray3<f64>>, Option<Bound<'py, PyArray2<f64>>>)> {
        let (all_positions, all_boxes) = self
            .reader
            .read_all_frames()
            .map_err(|e| PyIOError::new_err(e))?;

        if all_positions.is_empty() {
            return Err(PyValueError::new_err("No frames to read"));
        }

        let n_frames = all_positions.len();
        let n_atoms = all_positions[0].len();

        let mut traj_array = Array3::<f64>::zeros((n_frames, n_atoms, 3));
        let mut has_box = false;
        let mut box_array = Array2::<f64>::zeros((n_frames, 6));

        for (frame_idx, coords) in all_positions.iter().enumerate() {
            for (atom_idx, &[x, y, z]) in coords.iter().enumerate() {
                traj_array[[frame_idx, atom_idx, 0]] = x;
                traj_array[[frame_idx, atom_idx, 1]] = y;
                traj_array[[frame_idx, atom_idx, 2]] = z;
            }

            if let Some(box_dims) = &all_boxes[frame_idx] {
                has_box = true;
                for (i, &val) in box_dims.iter().enumerate() {
                    box_array[[frame_idx, i]] = val;
                }
            }
        }

        let box_result = if has_box {
            Some(box_array.to_pyarray_bound(py))
        } else {
            None
        };

        Ok((traj_array.to_pyarray_bound(py), box_result))
    }
}

#[pyfunction]
#[pyo3(name = "read_dcd_header")]
fn read_dcd_header_py<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let header = read_dcd_header(path).map_err(|e| PyIOError::new_err(e))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("n_frames", header.n_frames)?;
    dict.set_item("n_atoms", header.n_atoms)?;
    dict.set_item("start_timestep", header.start_timestep)?;
    dict.set_item("timestep_interval", header.timestep_interval)?;
    dict.set_item("timestep", header.timestep)?;
    dict.set_item("has_unit_cell", header.has_unit_cell)?;
    dict.set_item("is_charmm", header.is_charmm)?;
    dict.set_item("is_64bit", header.is_64bit)?;
    dict.set_item("is_big_endian", header.is_big_endian)?;
    dict.set_item("titles", PyList::new_bound(py, &header.titles))?;
    dict.set_item("first_frame_offset", header.first_frame_offset)?;
    dict.set_item("frame_size", header.frame_size)?;

    Ok(dict)
}

// ============================================================================
// FINGERPRINT SESSION
// ============================================================================

#[pyclass(name = "FingerprintMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyFingerprintMode {
    Target = 0,
    Binder = 1,
}

#[pyclass(name = "FingerprintSession")]
struct PyFingerprintSession {
    topology: AmberTopology,
    dcd_reader: Option<DcdReader>,
    target_residues: Vec<usize>,
    binder_residues: Vec<usize>,
    fingerprint_mode: PyFingerprintMode,
    return_residue_names: bool,
    // Cached selections
    target_selection: Option<rst_core::amber::prmtop::AtomSelection>,
    binder_atom_indices: Vec<usize>,
}

#[pymethods]
impl PyFingerprintSession {
    #[new]
    #[pyo3(signature = (topology_path, dcd_path=None))]
    fn new(topology_path: &str, dcd_path: Option<&str>) -> PyResult<Self> {
        let topology = parse_prmtop(topology_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read topology: {}", e)))?;

        let dcd_reader = if let Some(path) = dcd_path {
            Some(
                DcdReader::open(path)
                    .map_err(|e| PyIOError::new_err(format!("Failed to open DCD: {}", e)))?,
            )
        } else {
            None
        };

        Ok(PyFingerprintSession {
            topology,
            dcd_reader,
            target_residues: Vec::new(),
            binder_residues: Vec::new(),
            fingerprint_mode: PyFingerprintMode::Target,
            return_residue_names: false,
            target_selection: None,
            binder_atom_indices: Vec::new(),
        })
    }

    fn set_target_residues(&mut self, residue_indices: Vec<usize>) -> PyResult<()> {
        let selection = self
            .topology
            .build_selection(&residue_indices)
            .map_err(|e| PyValueError::new_err(e))?;
        self.target_residues = residue_indices;
        self.target_selection = Some(selection);
        Ok(())
    }

    fn set_binder_residues(&mut self, residue_indices: Vec<usize>) -> PyResult<()> {
        // Check for overlap with target residues
        if !self.target_residues.is_empty() {
            let target_set: std::collections::HashSet<usize> =
                self.target_residues.iter().copied().collect();
            for &idx in &residue_indices {
                if target_set.contains(&idx) {
                    return Err(PyValueError::new_err(format!(
                        "Binder and target selections overlap at residue {}",
                        idx
                    )));
                }
            }
        }

        let indices = self
            .topology
            .get_atom_indices_for_residues(&residue_indices)
            .map_err(|e| PyValueError::new_err(e))?;
        self.binder_residues = residue_indices;
        self.binder_atom_indices = indices;
        Ok(())
    }

    fn set_fingerprint_mode(&mut self, mode: PyFingerprintMode) -> PyResult<()> {
        self.fingerprint_mode = mode;
        Ok(())
    }

    #[getter]
    fn return_residue_names(&self) -> bool {
        self.return_residue_names
    }

    #[setter]
    fn set_return_residue_names(&mut self, value: bool) {
        self.return_residue_names = value;
    }

    fn compute_next_frame<'py>(&mut self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        let reader = self
            .dcd_reader
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("No DCD file loaded"))?;

        let frame_data = reader.read_frame().map_err(|e| PyIOError::new_err(e))?;

        let (positions, _box_info) = match frame_data {
            Some(data) => data,
            None => return Ok(None),
        };

        if self.target_selection.is_none() {
            return Err(PyValueError::new_err("Target selection not set"));
        }
        if self.binder_atom_indices.is_empty() && self.binder_residues.is_empty() {
            return Err(PyValueError::new_err("Binder selection not set"));
        }

        // Determine which residues to fingerprint and which are partners based on mode
        let (fingerprint_residues, partner_indices) = match self.fingerprint_mode {
            PyFingerprintMode::Target => {
                let selection = self.target_selection.as_ref().unwrap();
                let n_residues = selection.residue_offsets.len() - 1;
                let mut residues = Vec::with_capacity(n_residues);
                for r in 0..n_residues {
                    let start = selection.residue_offsets[r];
                    let end = selection.residue_offsets[r + 1];
                    let atom_indices = &selection.atom_indices[start..end];
                    let residue = ResidueData::from_global_indices(
                        &positions,
                        &self.topology.charges,
                        &self.topology.atom_sigmas,
                        &self.topology.atom_epsilons,
                        atom_indices,
                    );
                    residues.push(residue);
                }
                (residues, self.binder_atom_indices.clone())
            }
            PyFingerprintMode::Binder => {
                // Build binder selection for fingerprinting
                let binder_selection = self
                    .topology
                    .build_selection(&self.binder_residues)
                    .map_err(|e| PyValueError::new_err(e))?;
                let n_residues = binder_selection.residue_offsets.len() - 1;
                let mut residues = Vec::with_capacity(n_residues);
                for r in 0..n_residues {
                    let start = binder_selection.residue_offsets[r];
                    let end = binder_selection.residue_offsets[r + 1];
                    let atom_indices = &binder_selection.atom_indices[start..end];
                    let residue = ResidueData::from_global_indices(
                        &positions,
                        &self.topology.charges,
                        &self.topology.atom_sigmas,
                        &self.topology.atom_epsilons,
                        atom_indices,
                    );
                    residues.push(residue);
                }
                // Partner is the target atoms
                let target_selection = self.target_selection.as_ref().unwrap();
                (residues, target_selection.atom_indices.clone())
            }
        };

        let partner = PartnerData::new(
            &positions,
            &self.topology.charges,
            &self.topology.atom_sigmas,
            &self.topology.atom_epsilons,
            &partner_indices,
        );

        let (elec, vdw) = compute_fingerprints_from_residues(&fingerprint_residues, &partner);

        let vdw_arr = PyArray1::from_vec_bound(py, vdw);
        let elec_arr = PyArray1::from_vec_bound(py, elec);

        if self.return_residue_names {
            let indices = match self.fingerprint_mode {
                PyFingerprintMode::Target => &self.target_residues,
                PyFingerprintMode::Binder => &self.binder_residues,
            };
            let labels: Vec<&str> = indices
                .iter()
                .filter_map(|&i| self.topology.residue_labels.get(i).map(|s| s.as_str()))
                .collect();
            let names = PyList::new_bound(py, &labels);
            Ok(Some((vdw_arr, elec_arr, names).into_py(py)))
        } else {
            Ok(Some((vdw_arr, elec_arr).into_py(py)))
        }
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        self.compute_next_frame(py)
    }

    // Properties

    #[getter]
    fn n_atoms(&self) -> usize {
        self.topology.n_atoms
    }

    #[getter]
    fn n_residues(&self) -> usize {
        self.topology.n_residues
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.dcd_reader.as_ref().map(|r| r.n_frames()).unwrap_or(0)
    }

    #[getter]
    fn current_frame(&self) -> usize {
        self.dcd_reader
            .as_ref()
            .map(|r| r.current_frame())
            .unwrap_or(0)
    }

    fn seek(&mut self, frame: usize) -> PyResult<()> {
        let reader = self
            .dcd_reader
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("No DCD trajectory loaded"))?;
        reader.seek_frame(frame).map_err(|e| PyIOError::new_err(e))
    }

    #[getter]
    fn residue_labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let indices = match self.fingerprint_mode {
            PyFingerprintMode::Target => &self.target_residues,
            PyFingerprintMode::Binder => &self.binder_residues,
        };
        let labels: Vec<&str> = indices
            .iter()
            .filter_map(|&i| self.topology.residue_labels.get(i).map(|s| s.as_str()))
            .collect();
        PyList::new_bound(py, &labels)
    }

    #[getter]
    fn n_fingerprint_residues(&self) -> usize {
        match self.fingerprint_mode {
            PyFingerprintMode::Target => self.target_residues.len(),
            PyFingerprintMode::Binder => self.binder_residues.len(),
        }
    }

    #[getter]
    fn fingerprint_mode(&self) -> PyFingerprintMode {
        self.fingerprint_mode
    }
}

// ============================================================================
// MM-PBSA: Configuration types
// ============================================================================

#[pyclass(name = "GbModel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyGbModel {
    Hct = 0,
    ObcI = 1,
    ObcII = 2,
}

impl PyGbModel {
    fn to_rust(&self) -> GbModel {
        match self {
            PyGbModel::Hct => GbModel::Hct,
            PyGbModel::ObcI => GbModel::ObcI,
            PyGbModel::ObcII => GbModel::ObcII,
        }
    }
}

#[pyclass(name = "GbParams")]
#[derive(Clone)]
struct PyGbParams {
    inner: GbParams,
}

#[pymethods]
impl PyGbParams {
    #[new]
    #[pyo3(signature = (model=None, solute_dielectric=None, solvent_dielectric=None, salt_concentration=None, temperature=None, offset=None, rgbmax=None, cutoff=None))]
    fn new(
        model: Option<PyGbModel>,
        solute_dielectric: Option<f64>,
        solvent_dielectric: Option<f64>,
        salt_concentration: Option<f64>,
        temperature: Option<f64>,
        offset: Option<f64>,
        rgbmax: Option<f64>,
        cutoff: Option<f64>,
    ) -> Self {
        let mut params = GbParams::default();
        if let Some(m) = model {
            params.model = m.to_rust();
        }
        if let Some(v) = solute_dielectric {
            params.solute_dielectric = v;
        }
        if let Some(v) = solvent_dielectric {
            params.solvent_dielectric = v;
        }
        if let Some(v) = salt_concentration {
            params.salt_concentration = v;
        }
        if let Some(v) = temperature {
            params.temperature = v;
        }
        if let Some(v) = offset {
            params.offset = v;
        }
        if let Some(v) = rgbmax {
            params.rgbmax = v;
        }
        if let Some(v) = cutoff {
            params.cutoff = v;
        }
        PyGbParams { inner: params }
    }

    #[getter]
    fn solute_dielectric(&self) -> f64 {
        self.inner.solute_dielectric
    }
    #[getter]
    fn solvent_dielectric(&self) -> f64 {
        self.inner.solvent_dielectric
    }
    #[getter]
    fn salt_concentration(&self) -> f64 {
        self.inner.salt_concentration
    }
    #[getter]
    fn temperature(&self) -> f64 {
        self.inner.temperature
    }
    #[getter]
    fn cutoff(&self) -> f64 {
        self.inner.cutoff
    }
}

#[pyclass(name = "SaParams")]
#[derive(Clone)]
struct PySaParams {
    inner: SaParams,
}

#[pymethods]
impl PySaParams {
    #[new]
    #[pyo3(signature = (surface_tension=None, offset=None, probe_radius=None, n_sphere_points=None))]
    fn new(
        surface_tension: Option<f64>,
        offset: Option<f64>,
        probe_radius: Option<f64>,
        n_sphere_points: Option<usize>,
    ) -> Self {
        let mut params = SaParams::default();
        if let Some(v) = surface_tension {
            params.surface_tension = v;
        }
        if let Some(v) = offset {
            params.offset = v;
        }
        if let Some(v) = probe_radius {
            params.probe_radius = v;
        }
        if let Some(v) = n_sphere_points {
            params.n_sphere_points = v;
        }
        PySaParams { inner: params }
    }

    #[getter]
    fn surface_tension(&self) -> f64 {
        self.inner.surface_tension
    }
    #[getter]
    fn offset(&self) -> f64 {
        self.inner.offset
    }
    #[getter]
    fn probe_radius(&self) -> f64 {
        self.inner.probe_radius
    }
    #[getter]
    fn n_sphere_points(&self) -> usize {
        self.inner.n_sphere_points
    }
}

// ============================================================================
// MM-PBSA: Result types
// ============================================================================

#[pyclass(name = "MmEnergy")]
struct PyMmEnergy {
    inner: mm_energy::MmEnergy,
}

#[pymethods]
impl PyMmEnergy {
    #[getter]
    fn bond(&self) -> f64 {
        self.inner.bond
    }
    #[getter]
    fn angle(&self) -> f64 {
        self.inner.angle
    }
    #[getter]
    fn dihedral(&self) -> f64 {
        self.inner.dihedral
    }
    #[getter]
    fn vdw(&self) -> f64 {
        self.inner.vdw
    }
    #[getter]
    fn elec(&self) -> f64 {
        self.inner.elec
    }
    #[getter]
    fn vdw_14(&self) -> f64 {
        self.inner.vdw_14
    }
    #[getter]
    fn elec_14(&self) -> f64 {
        self.inner.elec_14
    }

    fn total(&self) -> f64 {
        self.inner.total()
    }
}

#[pyclass(name = "GbEnergy")]
struct PyGbEnergy {
    inner: gb_energy::GbEnergy,
}

#[pymethods]
impl PyGbEnergy {
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total
    }

    fn born_radii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.born_radii.clone())
    }
}

#[pyclass(name = "SaEnergy")]
struct PySaEnergy {
    inner: sa_energy::SaEnergy,
}

#[pymethods]
impl PySaEnergy {
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total
    }
    #[getter]
    fn total_sasa(&self) -> f64 {
        self.inner.total_sasa
    }

    fn per_atom_sasa<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.per_atom_sasa.clone())
    }
}

#[pyclass(name = "FrameEnergy")]
struct PyFrameEnergy {
    inner: binding::FrameEnergy,
}

#[pymethods]
impl PyFrameEnergy {
    #[getter]
    fn delta_mm(&self) -> f64 {
        self.inner.delta_mm
    }
    #[getter]
    fn delta_gb(&self) -> f64 {
        self.inner.delta_gb
    }
    #[getter]
    fn delta_sa(&self) -> f64 {
        self.inner.delta_sa
    }
    #[getter]
    fn delta_total(&self) -> f64 {
        self.inner.delta_total
    }

    #[getter]
    fn complex_mm(&self) -> f64 {
        self.inner.complex.mm
    }
    #[getter]
    fn complex_gb(&self) -> f64 {
        self.inner.complex.gb
    }
    #[getter]
    fn complex_sa(&self) -> f64 {
        self.inner.complex.sa
    }
    #[getter]
    fn complex_total(&self) -> f64 {
        self.inner.complex.total()
    }

    #[getter]
    fn receptor_mm(&self) -> f64 {
        self.inner.receptor.mm
    }
    #[getter]
    fn receptor_gb(&self) -> f64 {
        self.inner.receptor.gb
    }
    #[getter]
    fn receptor_sa(&self) -> f64 {
        self.inner.receptor.sa
    }
    #[getter]
    fn receptor_total(&self) -> f64 {
        self.inner.receptor.total()
    }

    #[getter]
    fn ligand_mm(&self) -> f64 {
        self.inner.ligand.mm
    }
    #[getter]
    fn ligand_gb(&self) -> f64 {
        self.inner.ligand.gb
    }
    #[getter]
    fn ligand_sa(&self) -> f64 {
        self.inner.ligand.sa
    }
    #[getter]
    fn ligand_total(&self) -> f64 {
        self.inner.ligand.total()
    }
}

#[pyclass(name = "BindingResult")]
struct PyBindingResult {
    inner: binding::BindingResult,
}

#[pymethods]
impl PyBindingResult {
    #[getter]
    fn mean_delta_mm(&self) -> f64 {
        self.inner.mean_delta_mm
    }
    #[getter]
    fn mean_delta_gb(&self) -> f64 {
        self.inner.mean_delta_gb
    }
    #[getter]
    fn mean_delta_sa(&self) -> f64 {
        self.inner.mean_delta_sa
    }
    #[getter]
    fn mean_delta_total(&self) -> f64 {
        self.inner.mean_delta_total
    }
    #[getter]
    fn std_delta_total(&self) -> f64 {
        self.inner.std_delta_total
    }
    #[getter]
    fn std_delta_mm(&self) -> f64 {
        self.inner.std_delta_mm
    }
    #[getter]
    fn std_delta_gb(&self) -> f64 {
        self.inner.std_delta_gb
    }
    #[getter]
    fn std_delta_sa(&self) -> f64 {
        self.inner.std_delta_sa
    }
    #[getter]
    fn sem_delta_total(&self) -> f64 {
        self.inner.sem_delta_total
    }

    #[getter]
    fn frames(&self) -> Vec<PyFrameEnergy> {
        self.inner
            .frames
            .iter()
            .map(|f| PyFrameEnergy { inner: f.clone() })
            .collect()
    }

    fn last_frame_coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.last_frame_coords.len();
        let mut arr = Array2::<f64>::zeros((n, 3));
        for (i, &[x, y, z]) in self.inner.last_frame_coords.iter().enumerate() {
            arr[[i, 0]] = x;
            arr[[i, 1]] = y;
            arr[[i, 2]] = z;
        }
        arr.to_pyarray_bound(py)
    }
}

#[pyclass(name = "ResidueContribution")]
struct PyResidueContribution {
    inner: decomposition::ResidueContribution,
}

#[pymethods]
impl PyResidueContribution {
    #[getter]
    fn residue_index(&self) -> usize {
        self.inner.residue_index
    }
    #[getter]
    fn residue_label(&self) -> &str {
        &self.inner.residue_label
    }
    #[getter]
    fn vdw(&self) -> f64 {
        self.inner.vdw
    }
    #[getter]
    fn elec(&self) -> f64 {
        self.inner.elec
    }
    #[getter]
    fn gb(&self) -> f64 {
        self.inner.gb
    }
    #[getter]
    fn sa(&self) -> f64 {
        self.inner.sa
    }

    fn total(&self) -> f64 {
        self.inner.total()
    }
}

#[pyclass(name = "DecompositionResult")]
struct PyDecompositionResult {
    inner: decomposition::DecompositionResult,
}

#[pymethods]
impl PyDecompositionResult {
    #[getter]
    fn receptor_residues(&self) -> Vec<PyResidueContribution> {
        self.inner
            .receptor_residues
            .iter()
            .map(|r| PyResidueContribution { inner: r.clone() })
            .collect()
    }
    #[getter]
    fn ligand_residues(&self) -> Vec<PyResidueContribution> {
        self.inner
            .ligand_residues
            .iter()
            .map(|r| PyResidueContribution { inner: r.clone() })
            .collect()
    }
}

#[pyclass(name = "EntropyEstimate")]
struct PyEntropyEstimate {
    inner: entropy::EntropyEstimate,
}

#[pymethods]
impl PyEntropyEstimate {
    #[getter]
    fn minus_tds(&self) -> f64 {
        self.inner.minus_tds
    }
    #[getter]
    fn method(&self) -> String {
        match self.inner.method {
            entropy::EntropyMethod::InteractionEntropy => "interaction_entropy".to_string(),
            entropy::EntropyMethod::QuasiHarmonic => "quasi_harmonic".to_string(),
        }
    }
}

// ============================================================================
// MM-PBSA: MdcrdReader
// ============================================================================

#[pyclass(name = "MdcrdReader")]
struct PyMdcrdReader {
    reader: MdcrdReader,
}

#[pymethods]
impl PyMdcrdReader {
    #[new]
    fn new(path: &str, n_atoms: usize, has_box: bool) -> PyResult<Self> {
        let reader =
            MdcrdReader::open(path, n_atoms, has_box).map_err(|e| PyIOError::new_err(e))?;
        Ok(PyMdcrdReader { reader })
    }

    fn read_frame<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match self.reader.read_frame() {
            Ok(Some(coords)) => {
                let n = coords.len();
                let mut arr = Array2::<f64>::zeros((n, 3));
                for (i, &[x, y, z]) in coords.iter().enumerate() {
                    arr[[i, 0]] = x;
                    arr[[i, 1]] = y;
                    arr[[i, 2]] = z;
                }
                Ok(Some(arr.to_pyarray_bound(py)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyIOError::new_err(e)),
        }
    }

    #[getter]
    fn current_frame(&self) -> usize {
        self.reader.current_frame()
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        self.read_frame(py)
    }
}

// ============================================================================
// MM-PBSA: Functions
// ============================================================================

#[pyfunction]
#[pyo3(name = "compute_mm_energy")]
fn compute_mm_energy_py(
    topology: &PyAmberTopology,
    coords: PyReadonlyArray2<f64>,
) -> PyResult<PyMmEnergy> {
    let coords_nm = array2_to_coords(&coords.as_array());
    let coords_vec = coords_nm_to_angstrom(&coords_nm);
    let inner = mm_energy::compute_mm_energy(&topology.inner, &coords_vec);
    Ok(PyMmEnergy { inner })
}

#[pyfunction]
#[pyo3(name = "compute_gb_energy", signature = (topology, coords, gb_params=None))]
fn compute_gb_energy_py(
    topology: &PyAmberTopology,
    coords: PyReadonlyArray2<f64>,
    gb_params: Option<&PyGbParams>,
) -> PyResult<PyGbEnergy> {
    let coords_nm = array2_to_coords(&coords.as_array());
    let coords_vec = coords_nm_to_angstrom(&coords_nm);
    let params = gb_params.map(|p| p.inner.clone()).unwrap_or_default();
    let inner = gb_energy::compute_gb_energy(&topology.inner, &coords_vec, &params);
    Ok(PyGbEnergy { inner })
}

#[pyfunction]
#[pyo3(name = "compute_sa_energy", signature = (topology, coords, sa_params=None))]
fn compute_sa_energy_py(
    topology: &PyAmberTopology,
    coords: PyReadonlyArray2<f64>,
    sa_params: Option<&PySaParams>,
) -> PyResult<PySaEnergy> {
    let coords_nm = array2_to_coords(&coords.as_array());
    let coords_vec = coords_nm_to_angstrom(&coords_nm);
    let params = sa_params.map(|p| p.inner.clone()).unwrap_or_default();
    let inner = sa_energy::compute_sa_energy(&topology.inner, &coords_vec, &params);
    Ok(PySaEnergy { inner })
}

#[pyfunction]
#[pyo3(name = "compute_binding_energy", signature = (topology, trajectory_path, receptor_residues, ligand_residues, gb_params=None, sa_params=None, trajectory_format=None, has_box=false, stride=1, start_frame=0, end_frame=usize::MAX))]
fn compute_binding_energy_py(
    topology: &Bound<'_, PyAny>,
    trajectory_path: &str,
    receptor_residues: Vec<usize>,
    ligand_residues: Vec<usize>,
    gb_params: Option<&PyGbParams>,
    sa_params: Option<&PySaParams>,
    trajectory_format: Option<&str>,
    has_box: bool,
    stride: usize,
    start_frame: usize,
    end_frame: usize,
) -> PyResult<PyBindingResult> {
    let topo = if let Ok(t) = topology.extract::<PyRef<PyAmberTopology>>() {
        t.inner.clone()
    } else if let Ok(path) = topology.extract::<&str>() {
        parse_prmtop(path).map_err(|e| PyIOError::new_err(e))?
    } else {
        return Err(PyValueError::new_err(
            "topology must be an AmberTopology object or a path string",
        ));
    };
    let fmt = match trajectory_format.unwrap_or("mdcrd") {
        "dcd" => TrajectoryFormat::Dcd,
        _ => TrajectoryFormat::Mdcrd { has_box },
    };
    let config = BindingConfig {
        receptor_residues,
        ligand_residues,
        gb_params: gb_params.map(|p| p.inner.clone()).unwrap_or_default(),
        sa_params: sa_params.map(|p| p.inner.clone()).unwrap_or_default(),
        trajectory_format: fmt,
        stride,
        start_frame,
        end_frame,
    };
    let inner =
        binding::compute_binding_energy(&topo, std::path::Path::new(trajectory_path), &config)
            .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyBindingResult { inner })
}

#[pyfunction]
#[pyo3(name = "compute_binding_energy_single_frame", signature = (topology, coords, receptor_residues, ligand_residues, gb_params=None, sa_params=None))]
fn compute_binding_energy_single_frame_py(
    topology: &PyAmberTopology,
    coords: PyReadonlyArray2<f64>,
    receptor_residues: Vec<usize>,
    ligand_residues: Vec<usize>,
    gb_params: Option<&PyGbParams>,
    sa_params: Option<&PySaParams>,
) -> PyResult<PyFrameEnergy> {
    let coords_nm = array2_to_coords(&coords.as_array());
    let coords_vec = coords_nm_to_angstrom(&coords_nm);
    let config = BindingConfig {
        receptor_residues,
        ligand_residues,
        gb_params: gb_params.map(|p| p.inner.clone()).unwrap_or_default(),
        sa_params: sa_params.map(|p| p.inner.clone()).unwrap_or_default(),
        trajectory_format: TrajectoryFormat::Mdcrd { has_box: false },
        stride: 1,
        start_frame: 0,
        end_frame: 0,
    };
    let inner = binding::compute_binding_energy_single_frame(&topology.inner, &coords_vec, &config)
        .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyFrameEnergy { inner })
}

#[pyfunction]
#[pyo3(name = "decompose_binding_energy", signature = (topology, coords, receptor_residues, ligand_residues, gb_params=None, sa_params=None))]
fn decompose_binding_energy_py(
    topology: &PyAmberTopology,
    coords: PyReadonlyArray2<f64>,
    receptor_residues: Vec<usize>,
    ligand_residues: Vec<usize>,
    gb_params: Option<&PyGbParams>,
    sa_params: Option<&PySaParams>,
) -> PyResult<PyDecompositionResult> {
    let coords_nm = array2_to_coords(&coords.as_array());
    let coords_vec = coords_nm_to_angstrom(&coords_nm);
    let gb = gb_params.map(|p| p.inner.clone()).unwrap_or_default();
    let sa = sa_params.map(|p| p.inner.clone()).unwrap_or_default();
    let inner = decomposition::decompose_binding_energy(
        &topology.inner,
        &coords_vec,
        &receptor_residues,
        &ligand_residues,
        &gb,
        &sa,
    )
    .map_err(|e| PyValueError::new_err(e))?;
    Ok(PyDecompositionResult { inner })
}

#[pyfunction]
#[pyo3(name = "interaction_entropy", signature = (frames, temperature=298.15))]
fn interaction_entropy_py(
    frames: Vec<PyRef<PyFrameEnergy>>,
    temperature: f64,
) -> Option<PyEntropyEstimate> {
    let frame_refs: Vec<binding::FrameEnergy> = frames.iter().map(|f| f.inner.clone()).collect();
    entropy::interaction_entropy(&frame_refs, temperature).map(|inner| PyEntropyEstimate { inner })
}

#[pyfunction]
#[pyo3(name = "quasi_harmonic_entropy", signature = (trajectory_3d, masses, temperature=298.15))]
fn quasi_harmonic_entropy_py(
    trajectory_3d: PyReadonlyArray3<f64>,
    masses: PyReadonlyArray1<f64>,
    temperature: f64,
) -> Option<PyEntropyEstimate> {
    let traj = array3_to_trajectory(&trajectory_3d.as_array());
    let masses_vec: Vec<f64> = masses.as_array().iter().copied().collect();
    entropy::quasi_harmonic_entropy(&traj, &masses_vec, temperature)
        .map(|inner| PyEntropyEstimate { inner })
}

#[pyfunction]
#[pyo3(name = "extract_subtopology")]
fn extract_subtopology_py(topology: &PyAmberTopology, atom_indices: Vec<usize>) -> PyAmberTopology {
    let inner = subsystem::extract_subtopology(&topology.inner, &atom_indices);
    PyAmberTopology { inner }
}

// ============================================================================
// MODULE DEFINITION
// ============================================================================

#[pymodule]
fn rust_simulation_tools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Kabsch
    m.add_function(wrap_pyfunction!(kabsch_align_py, m)?)?;

    // Wrapping
    m.add_function(wrap_pyfunction!(unwrap_system_py, m)?)?;

    // SASA
    m.add_function(wrap_pyfunction!(calculate_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_sasa_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_residue_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_total_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(get_vdw_radius_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_radii_array, m)?)?;

    // Fingerprints
    m.add_function(wrap_pyfunction!(compute_fingerprints_py, m)?)?;

    // AMBER
    m.add_class::<PyAmberTopology>()?;
    m.add_function(wrap_pyfunction!(read_prmtop_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_inpcrd_py, m)?)?;

    // DCD
    m.add_class::<PyDcdReader>()?;
    m.add_function(wrap_pyfunction!(read_dcd_header_py, m)?)?;

    // Fingerprint Session
    m.add_class::<PyFingerprintSession>()?;
    m.add_class::<PyFingerprintMode>()?;

    // MM-PBSA
    m.add_class::<PyGbModel>()?;
    m.add_class::<PyGbParams>()?;
    m.add_class::<PySaParams>()?;
    m.add_class::<PyMmEnergy>()?;
    m.add_class::<PyGbEnergy>()?;
    m.add_class::<PySaEnergy>()?;
    m.add_class::<PyFrameEnergy>()?;
    m.add_class::<PyBindingResult>()?;
    m.add_class::<PyResidueContribution>()?;
    m.add_class::<PyDecompositionResult>()?;
    m.add_class::<PyEntropyEstimate>()?;
    m.add_class::<PyMdcrdReader>()?;
    m.add_function(wrap_pyfunction!(compute_mm_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gb_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sa_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_binding_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_binding_energy_single_frame_py, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_binding_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(interaction_entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(quasi_harmonic_entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_subtopology_py, m)?)?;

    Ok(())
}
