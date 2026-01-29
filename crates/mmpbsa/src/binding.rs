//! MM-GBSA binding free energy workflow.
//!
//! Implements the 1-trajectory approach:
//!   ΔG_bind = G(complex) - G(receptor) - G(ligand)
//!
//! For each frame, computes MM energy, GB polar solvation, and SA non-polar
//! solvation for the complex, receptor, and ligand subsystems.

use crate::gb_energy::{compute_gb_energy, GbParams};
use crate::mdcrd::MdcrdReader;
use crate::mm_energy::{build_14_pairs, build_exclusion_set, compute_mm_energy_with_nb};
use crate::sa_energy::{compute_sa_energy, SaParams};
use crate::subsystem::{extract_coords, extract_subtopology};
use rayon::prelude::*;
use rst_core::amber::prmtop::AmberTopology;
use rst_core::trajectory::dcd::DcdReader;
use std::collections::HashSet;
use std::path::Path;

/// Energy components for a single subsystem in one frame.
#[derive(Debug, Clone, Default)]
pub struct SubsystemEnergy {
    pub mm: f64,
    pub gb: f64,
    pub sa: f64,
}

impl SubsystemEnergy {
    pub fn total(&self) -> f64 {
        self.mm + self.gb + self.sa
    }
}

/// Per-frame binding energy decomposition.
#[derive(Debug, Clone)]
pub struct FrameEnergy {
    pub complex: SubsystemEnergy,
    pub receptor: SubsystemEnergy,
    pub ligand: SubsystemEnergy,
    pub delta_mm: f64,
    pub delta_gb: f64,
    pub delta_sa: f64,
    pub delta_total: f64,
}

/// Summary statistics for binding energy over multiple frames.
#[derive(Debug, Clone)]
pub struct BindingResult {
    pub frames: Vec<FrameEnergy>,
    pub mean_delta_mm: f64,
    pub mean_delta_gb: f64,
    pub mean_delta_sa: f64,
    pub mean_delta_total: f64,
    pub std_delta_total: f64,
    pub std_delta_mm: f64,
    pub std_delta_gb: f64,
    pub std_delta_sa: f64,
    /// Standard error of the mean for delta_total.
    pub sem_delta_total: f64,
    /// Coordinates from the last trajectory frame (Angstroms), retained so
    /// callers can run follow-up analyses (e.g. decomposition) without
    /// re-reading the trajectory.
    pub last_frame_coords: Vec<[f64; 3]>,
}

/// Trajectory file format.
#[derive(Debug, Clone)]
pub enum TrajectoryFormat {
    /// AMBER mdcrd (ASCII) format.
    Mdcrd {
        /// Whether the mdcrd file contains box dimensions.
        has_box: bool,
    },
    /// DCD (binary) format. Coordinates are converted from nm to Angstroms.
    Dcd,
}

/// Configuration for the binding free energy calculation.
#[derive(Debug, Clone)]
pub struct BindingConfig {
    /// 0-based residue indices for the receptor.
    pub receptor_residues: Vec<usize>,
    /// 0-based residue indices for the ligand.
    pub ligand_residues: Vec<usize>,
    /// GB calculation parameters.
    pub gb_params: GbParams,
    /// SA calculation parameters.
    pub sa_params: SaParams,
    /// Trajectory file format (defaults to Mdcrd without box).
    pub trajectory_format: TrajectoryFormat,
    /// Process every Nth frame. Default: 1 (every frame).
    pub stride: usize,
    /// First frame to process (0-based). Default: 0.
    pub start_frame: usize,
    /// Last frame to process (exclusive). Default: usize::MAX (all frames).
    pub end_frame: usize,
}

/// Compute binding energy for a single coordinate frame.
///
/// # Arguments
/// * `complex_top` - Complex topology
/// * `receptor_top` - Pre-built receptor sub-topology
/// * `ligand_top` - Pre-built ligand sub-topology
/// * `receptor_atoms` - Atom indices of receptor in complex
/// * `ligand_atoms` - Atom indices of ligand in complex
/// * `coords` - Full complex coordinates (Angstroms)
/// * `gb_params` - GB parameters
/// * `sa_params` - SA parameters
/// Pre-built non-bonded pair sets for complex, receptor, and ligand topologies.
struct PrebuiltNbSets {
    complex_excluded: HashSet<(usize, usize)>,
    complex_14: HashSet<(usize, usize)>,
    receptor_excluded: HashSet<(usize, usize)>,
    receptor_14: HashSet<(usize, usize)>,
    ligand_excluded: HashSet<(usize, usize)>,
    ligand_14: HashSet<(usize, usize)>,
}

/// Build the effective complex subsystem from receptor + ligand residues.
///
/// If the union of receptor and ligand residues covers all atoms in the
/// topology, this is a no-op (returns the original topology and identity
/// mapping). Otherwise, it extracts a complex sub-topology and remaps
/// receptor/ligand selections to be relative to that sub-topology.
///
/// Returns `(effective_complex_top, complex_atom_indices, receptor_sel, ligand_sel)`.
/// `complex_atom_indices` is `None` when no extraction is needed (all atoms
/// are part of the complex), or `Some(indices)` for slicing frame coordinates.
fn build_complex_subsystem(
    topology: &AmberTopology,
    receptor_residues: &[usize],
    ligand_residues: &[usize],
) -> Result<
    (
        AmberTopology,
        Option<Vec<usize>>,
        rst_core::amber::prmtop::AtomSelection,
        rst_core::amber::prmtop::AtomSelection,
    ),
    String,
> {
    use std::collections::BTreeSet;

    // Combine receptor + ligand residues into a sorted, deduplicated list.
    let complex_residues: Vec<usize> = {
        let mut set = BTreeSet::new();
        set.extend(receptor_residues.iter().copied());
        set.extend(ligand_residues.iter().copied());
        set.into_iter().collect()
    };

    // Build complex atom selection on the original topology.
    let complex_sel = topology.build_selection(&complex_residues)?;

    // Fast path: complex covers all atoms — no extraction needed.
    if complex_sel.atom_indices.len() == topology.n_atoms {
        let receptor_sel = topology.build_selection(receptor_residues)?;
        let ligand_sel = topology.build_selection(ligand_residues)?;
        return Ok((topology.clone(), None, receptor_sel, ligand_sel));
    }

    // Solvated path: extract a complex sub-topology.
    log::info!(
        "Solvated topology detected ({} total atoms). Extracting complex sub-topology ({} atoms from {} residues).",
        topology.n_atoms,
        complex_sel.atom_indices.len(),
        complex_residues.len(),
    );
    let complex_top = extract_subtopology(topology, &complex_sel.atom_indices);

    // Re-map receptor/ligand residue indices from original to complex-local.
    // The complex sub-topology has residues numbered 0..N corresponding to
    // the sorted complex_residues list.
    let orig_to_complex_res: std::collections::HashMap<usize, usize> = complex_residues
        .iter()
        .enumerate()
        .map(|(new, &orig)| (orig, new))
        .collect();

    let remapped_receptor: Vec<usize> = receptor_residues
        .iter()
        .map(|r| orig_to_complex_res[r])
        .collect();
    let remapped_ligand: Vec<usize> = ligand_residues
        .iter()
        .map(|r| orig_to_complex_res[r])
        .collect();

    let receptor_sel = complex_top.build_selection(&remapped_receptor)?;
    let ligand_sel = complex_top.build_selection(&remapped_ligand)?;

    Ok((
        complex_top,
        Some(complex_sel.atom_indices),
        receptor_sel,
        ligand_sel,
    ))
}

/// Slice frame coordinates to only the complex atoms.
/// If `complex_atom_indices` is `None`, returns the coordinates as-is.
fn slice_frame_to_complex(
    coords: &[[f64; 3]],
    complex_atom_indices: &Option<Vec<usize>>,
) -> Vec<[f64; 3]> {
    match complex_atom_indices {
        Some(indices) => extract_coords(coords, indices),
        None => coords.to_vec(),
    }
}

fn compute_frame_energy(
    complex_top: &AmberTopology,
    receptor_top: &AmberTopology,
    ligand_top: &AmberTopology,
    receptor_atoms: &[usize],
    ligand_atoms: &[usize],
    coords: &[[f64; 3]],
    gb_params: &GbParams,
    sa_params: &SaParams,
    nb_sets: &PrebuiltNbSets,
) -> FrameEnergy {
    // Complex energies
    let c_mm = compute_mm_energy_with_nb(
        complex_top,
        coords,
        &nb_sets.complex_excluded,
        &nb_sets.complex_14,
    );
    let c_gb = compute_gb_energy(complex_top, coords, gb_params);
    let c_sa = compute_sa_energy(complex_top, coords, sa_params);

    // Receptor energies (extract coordinates)
    let r_coords = extract_coords(coords, receptor_atoms);
    let r_mm = compute_mm_energy_with_nb(
        receptor_top,
        &r_coords,
        &nb_sets.receptor_excluded,
        &nb_sets.receptor_14,
    );
    let r_gb = compute_gb_energy(receptor_top, &r_coords, gb_params);
    let r_sa = compute_sa_energy(receptor_top, &r_coords, sa_params);

    // Ligand energies
    let l_coords = extract_coords(coords, ligand_atoms);
    let l_mm = compute_mm_energy_with_nb(
        ligand_top,
        &l_coords,
        &nb_sets.ligand_excluded,
        &nb_sets.ligand_14,
    );
    let l_gb = compute_gb_energy(ligand_top, &l_coords, gb_params);
    let l_sa = compute_sa_energy(ligand_top, &l_coords, sa_params);

    let complex_e = SubsystemEnergy {
        mm: c_mm.total(),
        gb: c_gb.total,
        sa: c_sa.total,
    };
    let receptor_e = SubsystemEnergy {
        mm: r_mm.total(),
        gb: r_gb.total,
        sa: r_sa.total,
    };
    let ligand_e = SubsystemEnergy {
        mm: l_mm.total(),
        gb: l_gb.total,
        sa: l_sa.total,
    };

    let delta_mm = complex_e.mm - receptor_e.mm - ligand_e.mm;
    let delta_gb = complex_e.gb - receptor_e.gb - ligand_e.gb;
    let delta_sa = complex_e.sa - receptor_e.sa - ligand_e.sa;
    let delta_total = delta_mm + delta_gb + delta_sa;

    FrameEnergy {
        complex: complex_e,
        receptor: receptor_e,
        ligand: ligand_e,
        delta_mm,
        delta_gb,
        delta_sa,
        delta_total,
    }
}

/// Process a single frame and log the result.
fn process_frame(
    complex_top: &AmberTopology,
    receptor_top: &AmberTopology,
    ligand_top: &AmberTopology,
    receptor_sel: &rst_core::amber::prmtop::AtomSelection,
    ligand_sel: &rst_core::amber::prmtop::AtomSelection,
    coords: &[[f64; 3]],
    gb_params: &GbParams,
    sa_params: &SaParams,
    frame_index: usize,
    nb_sets: &PrebuiltNbSets,
) -> FrameEnergy {
    let energy = compute_frame_energy(
        complex_top,
        receptor_top,
        ligand_top,
        &receptor_sel.atom_indices,
        &ligand_sel.atom_indices,
        coords,
        gb_params,
        sa_params,
        nb_sets,
    );
    log::debug!(
        "Frame {}: ΔG = {:.2} (ΔMM={:.2}, ΔGB={:.2}, ΔSA={:.2})",
        frame_index + 1,
        energy.delta_total,
        energy.delta_mm,
        energy.delta_gb,
        energy.delta_sa,
    );
    energy
}

/// Run the full MM-GBSA binding free energy calculation over a trajectory.
///
/// # Arguments
/// * `complex_top` - Parsed complex topology
/// * `trajectory_path` - Path to mdcrd trajectory file (coordinates in Angstroms)
/// * `config` - Binding calculation configuration
///
/// # Returns
/// `BindingResult` with per-frame energies and summary statistics.
pub fn compute_binding_energy(
    complex_top: &AmberTopology,
    trajectory_path: &Path,
    config: &BindingConfig,
) -> Result<BindingResult, String> {
    // Derive the complex as the union of receptor + ligand residues.
    let (effective_complex_top, complex_atom_indices, receptor_sel, ligand_sel) =
        build_complex_subsystem(
            complex_top,
            &config.receptor_residues,
            &config.ligand_residues,
        )?;

    // Build sub-topologies relative to the (possibly extracted) complex topology.
    let receptor_top = extract_subtopology(&effective_complex_top, &receptor_sel.atom_indices);
    let ligand_top = extract_subtopology(&effective_complex_top, &ligand_sel.atom_indices);

    log::info!(
        "Complex: {} atoms, Receptor: {} atoms ({} residues), Ligand: {} atoms ({} residues)",
        effective_complex_top.n_atoms,
        receptor_top.n_atoms,
        config.receptor_residues.len(),
        ligand_top.n_atoms,
        config.ligand_residues.len(),
    );

    // Pre-build topology-invariant exclusion and 1-4 pair sets once,
    // rather than rebuilding them for every frame and subsystem.
    let nb_sets = PrebuiltNbSets {
        complex_excluded: build_exclusion_set(&effective_complex_top),
        complex_14: build_14_pairs(&effective_complex_top),
        receptor_excluded: build_exclusion_set(&receptor_top),
        receptor_14: build_14_pairs(&receptor_top),
        ligand_excluded: build_exclusion_set(&ligand_top),
        ligand_14: build_14_pairs(&ligand_top),
    };

    let stride = config.stride.max(1);
    let start_frame = config.start_frame;
    let end_frame = config.end_frame;

    // Collect qualifying frame coordinates into memory for parallel processing.
    // When the topology is solvated, slice each frame to complex atoms only.
    let mut frame_coords: Vec<Vec<[f64; 3]>> = Vec::new();

    match &config.trajectory_format {
        TrajectoryFormat::Mdcrd { has_box } => {
            let mut reader = MdcrdReader::open(trajectory_path, complex_top.n_atoms, *has_box)?;
            let mut frame_idx: usize = 0;
            while let Some(coords) = reader.read_frame()? {
                if frame_idx >= end_frame {
                    break;
                }
                if frame_idx >= start_frame && (frame_idx - start_frame) % stride == 0 {
                    let coords = slice_frame_to_complex(&coords, &complex_atom_indices);
                    frame_coords.push(coords);
                }
                frame_idx += 1;
            }
        }
        TrajectoryFormat::Dcd => {
            let mut reader = DcdReader::open(trajectory_path)?;
            let mut frame_idx: usize = 0;
            while let Some((coords_nm, _box_info)) = reader.read_frame()? {
                if frame_idx >= end_frame {
                    break;
                }
                if frame_idx >= start_frame && (frame_idx - start_frame) % stride == 0 {
                    // DCD reader returns coordinates in nm; convert to Angstroms.
                    let coords: Vec<[f64; 3]> = coords_nm
                        .iter()
                        .map(|c| [c[0] * 10.0, c[1] * 10.0, c[2] * 10.0])
                        .collect();
                    let coords = slice_frame_to_complex(&coords, &complex_atom_indices);
                    frame_coords.push(coords);
                }
                frame_idx += 1;
            }
        }
    }

    if frame_coords.is_empty() {
        return Err("No frames found in trajectory".to_string());
    }

    if frame_coords.len() > 1000 {
        log::warn!(
            "Loading {} frames into memory for parallel processing. \
             Consider using stride/start_frame/end_frame to reduce memory usage.",
            frame_coords.len(),
        );
    }

    log::info!(
        "Processing {} frames in parallel (start={}, end={}, stride={})",
        frame_coords.len(),
        start_frame,
        end_frame,
        stride,
    );

    let last_frame_coords = frame_coords.last().cloned().unwrap_or_default();

    let frames: Vec<FrameEnergy> = frame_coords
        .par_iter()
        .enumerate()
        .map(|(idx, coords)| {
            process_frame(
                &effective_complex_top,
                &receptor_top,
                &ligand_top,
                &receptor_sel,
                &ligand_sel,
                coords,
                &config.gb_params,
                &config.sa_params,
                idx,
                &nb_sets,
            )
        })
        .collect();

    if frames.is_empty() {
        return Err("No frames found in trajectory".to_string());
    }

    // Compute statistics
    let n = frames.len() as f64;
    let mean_delta_mm: f64 = frames.iter().map(|f| f.delta_mm).sum::<f64>() / n;
    let mean_delta_gb: f64 = frames.iter().map(|f| f.delta_gb).sum::<f64>() / n;
    let mean_delta_sa: f64 = frames.iter().map(|f| f.delta_sa).sum::<f64>() / n;
    let mean_delta_total: f64 = frames.iter().map(|f| f.delta_total).sum::<f64>() / n;

    let sample_std = |values: &[f64], mean: f64| -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        let n_minus_1 = (values.len() - 1) as f64;
        let variance: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_minus_1;
        variance.sqrt()
    };

    let delta_totals: Vec<f64> = frames.iter().map(|f| f.delta_total).collect();
    let delta_mms: Vec<f64> = frames.iter().map(|f| f.delta_mm).collect();
    let delta_gbs: Vec<f64> = frames.iter().map(|f| f.delta_gb).collect();
    let delta_sas: Vec<f64> = frames.iter().map(|f| f.delta_sa).collect();

    let std_delta_total = sample_std(&delta_totals, mean_delta_total);
    let std_delta_mm = sample_std(&delta_mms, mean_delta_mm);
    let std_delta_gb = sample_std(&delta_gbs, mean_delta_gb);
    let std_delta_sa = sample_std(&delta_sas, mean_delta_sa);
    let sem_delta_total = std_delta_total / n.sqrt();

    Ok(BindingResult {
        frames,
        mean_delta_mm,
        mean_delta_gb,
        mean_delta_sa,
        mean_delta_total,
        std_delta_total,
        std_delta_mm,
        std_delta_gb,
        std_delta_sa,
        sem_delta_total,
        last_frame_coords,
    })
}

/// Compute binding energy for a single frame from pre-loaded coordinates.
///
/// Useful when coordinates are already in memory (e.g., from DCD or inpcrd).
pub fn compute_binding_energy_single_frame(
    complex_top: &AmberTopology,
    coords: &[[f64; 3]],
    config: &BindingConfig,
) -> Result<FrameEnergy, String> {
    let (effective_complex_top, complex_atom_indices, receptor_sel, ligand_sel) =
        build_complex_subsystem(
            complex_top,
            &config.receptor_residues,
            &config.ligand_residues,
        )?;

    let complex_coords = slice_frame_to_complex(coords, &complex_atom_indices);

    let receptor_top = extract_subtopology(&effective_complex_top, &receptor_sel.atom_indices);
    let ligand_top = extract_subtopology(&effective_complex_top, &ligand_sel.atom_indices);

    let nb_sets = PrebuiltNbSets {
        complex_excluded: build_exclusion_set(&effective_complex_top),
        complex_14: build_14_pairs(&effective_complex_top),
        receptor_excluded: build_exclusion_set(&receptor_top),
        receptor_14: build_14_pairs(&receptor_top),
        ligand_excluded: build_exclusion_set(&ligand_top),
        ligand_14: build_14_pairs(&ligand_top),
    };

    Ok(compute_frame_energy(
        &effective_complex_top,
        &receptor_top,
        &ligand_top,
        &receptor_sel.atom_indices,
        &ligand_sel.atom_indices,
        &complex_coords,
        &config.gb_params,
        &config.sa_params,
        &nb_sets,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gb_energy::GbModel;

    #[test]
    fn test_binding_energy_single_frame() {
        let prmtop_path = "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/ras-raf.prmtop";
        let mdcrd_path =
            "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/_MMPBSA_complex.mdcrd.0";
        if !Path::new(prmtop_path).exists() || !Path::new(mdcrd_path).exists() {
            return;
        }

        let top =
            rst_core::amber::prmtop::parse_prmtop(prmtop_path).expect("Failed to parse prmtop");

        // Read first frame
        let mut reader =
            MdcrdReader::open(mdcrd_path, top.n_atoms, false).expect("Failed to open mdcrd");
        let coords = reader
            .read_frame()
            .expect("Failed to read frame")
            .expect("No frames");

        // Ras = residues 0..165, Raf = residues 166..end (approximate split)
        // For testing, just use a small split
        let receptor_residues: Vec<usize> = (0..166).collect();
        let ligand_residues: Vec<usize> = (166..top.n_residues).collect();

        let config = BindingConfig {
            receptor_residues,
            ligand_residues,
            gb_params: GbParams {
                model: GbModel::ObcI,
                salt_concentration: 0.15,
                ..GbParams::default()
            },
            sa_params: SaParams::default(),
            trajectory_format: TrajectoryFormat::Mdcrd { has_box: false },
            stride: 1,
            start_frame: 0,
            end_frame: usize::MAX,
        };

        let result = compute_binding_energy_single_frame(&top, &coords, &config)
            .expect("Failed to compute binding energy");

        // Sanity checks: delta values should be finite and in a reasonable range
        assert!(result.delta_total.is_finite(), "Total energy is not finite");
        assert!(result.delta_mm.is_finite(), "MM energy is not finite");
        assert!(result.delta_gb.is_finite(), "GB energy is not finite");
        assert!(result.delta_sa.is_finite(), "SA energy is not finite");

        // SA should be negative (buried surface upon binding)
        assert!(
            result.delta_sa <= 0.0,
            "ΔSA should be ≤ 0 (buried surface), got {}",
            result.delta_sa
        );

        println!(
            "ΔG = {:.2} kcal/mol (ΔMM={:.2}, ΔGB={:.2}, ΔSA={:.2})",
            result.delta_total, result.delta_mm, result.delta_gb, result.delta_sa
        );
    }
}
