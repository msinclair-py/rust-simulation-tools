//! Rectangular box solvation with OPC water.
//!
//! Implements the solvation algorithm based on AMBER's `zToolSolvateAndShell`:
//!
//! 1. Center the solute at the origin.
//! 2. Compute the solute bounding box.
//! 3. Determine target box dimensions: `bbox_size + 2 * buffer`.
//! 4. Tile pre-equilibrated water box replicas to fill the volume.
//! 5. Remove water molecules that clash with the solute (cell-list accelerated).
//! 6. Remove water molecules whose oxygen falls outside the target box.
//! 7. Add surviving waters as WAT residues and set periodic box dimensions.

use std::collections::HashMap;

use rst_core::forcefield::residue_lib::ResidueTemplate;

use crate::system::{Atom, System};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for solvation.
pub struct SolvateConfig {
    /// Buffer distance in Angstroms on each side of the solute.
    pub buffer: f64,
    /// Minimum distance between solute and solvent atoms (Angstroms).
    pub closeness: f64,
}

impl Default for SolvateConfig {
    fn default() -> Self {
        Self {
            buffer: 12.0,
            closeness: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// OPC water constants
// ---------------------------------------------------------------------------

/// OPC water atom names.
const WATER_NAMES: [&str; 4] = ["O", "H1", "H2", "EPW"];

/// OPC water AMBER atom types.
const WATER_TYPES: [&str; 4] = ["OW", "HW", "HW", "EP"];

/// OPC water element symbols.
const WATER_ELEMENTS: [&str; 4] = ["O", "H", "H", ""];

/// OPC water partial charges (elementary charge units).
const WATER_CHARGES: [f64; 4] = [0.0, 0.679_142, 0.679_142, -1.358_284];

/// OPC water atomic masses (amu).
const WATER_MASSES: [f64; 4] = [16.0, 1.008, 1.008, 0.0];

/// OPC water atomic numbers.
const WATER_ATOMIC_NUMBERS: [i32; 4] = [8, 1, 1, 0];

/// Internal bonds within a single water molecule (0-based relative indices).
/// O-H1, O-H2, O-EPW.
const WATER_BONDS: [(usize, usize); 3] = [(0, 1), (0, 2), (0, 3)];

// ---------------------------------------------------------------------------
// Cell list for spatial hashing
// ---------------------------------------------------------------------------

/// A spatial hash grid for fast proximity queries.
///
/// Solute atom indices are inserted into grid cells of side length `cell_size`.
/// Proximity checks against query positions visit only the 27 neighboring cells
/// (the target cell plus 26 Moore neighbors), ensuring worst-case O(1) lookups
/// for uniform distributions.
struct CellList {
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f64,
}

impl CellList {
    /// Create an empty cell list with the given cell side length.
    fn new(cell_size: f64) -> Self {
        Self {
            cells: HashMap::new(),
            cell_size,
        }
    }

    /// Map a position to its cell key.
    fn cell_key(&self, pos: &[f64; 3]) -> (i32, i32, i32) {
        (
            (pos[0] / self.cell_size).floor() as i32,
            (pos[1] / self.cell_size).floor() as i32,
            (pos[2] / self.cell_size).floor() as i32,
        )
    }

    /// Insert an atom index at the given position.
    fn insert(&mut self, idx: usize, pos: &[f64; 3]) {
        let key = self.cell_key(pos);
        self.cells.entry(key).or_default().push(idx);
    }

    /// Check whether any point in `query_positions` is within `cutoff` of any
    /// indexed atom whose position is looked up in `all_positions`.
    ///
    /// Returns `true` at the first collision found (short-circuits).
    fn any_within_cutoff(
        &self,
        query_positions: &[[f64; 3]],
        all_positions: &[[f64; 3]],
        cutoff: f64,
    ) -> bool {
        let cutoff_sq = cutoff * cutoff;

        for qpos in query_positions {
            let (cx, cy, cz) = self.cell_key(qpos);

            // Check the 3x3x3 neighborhood (27 cells).
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor_key = (cx + dx, cy + dy, cz + dz);
                        if let Some(indices) = self.cells.get(&neighbor_key) {
                            for &idx in indices {
                                let sp = &all_positions[idx];
                                let dx = qpos[0] - sp[0];
                                let dy = qpos[1] - sp[1];
                                let dz = qpos[2] - sp[2];
                                let dist_sq = dx * dx + dy * dy + dz * dz;
                                if dist_sq < cutoff_sq {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// Water molecule extraction from the OPCBOX template
// ---------------------------------------------------------------------------

/// Positions for one water molecule (O, H1, H2, EPW).
struct WaterMol {
    positions: [[f64; 3]; 4],
}

/// Extract individual water molecules from the OPCBOX residue template.
///
/// Each sub-residue in the template corresponds to a single water molecule.
/// We read the 4 atom positions for each sub-residue and return them as a list
/// of `WaterMol`.
fn extract_water_molecules(template: &ResidueTemplate) -> Result<Vec<WaterMol>, String> {
    let mut waters = Vec::with_capacity(template.residues.len());

    for sub_res in &template.residues {
        if sub_res.atom_count != 4 {
            return Err(format!(
                "Expected 4 atoms per water molecule in OPCBOX template, found {} \
                 in sub-residue '{}'",
                sub_res.atom_count, sub_res.name
            ));
        }

        let start = sub_res.start_atom;
        let mut positions = [[0.0_f64; 3]; 4];
        for (i, pos) in positions.iter_mut().enumerate() {
            *pos = template.positions[start + i];
        }
        waters.push(WaterMol { positions });
    }

    Ok(waters)
}

// ---------------------------------------------------------------------------
// Public solvation entry point
// ---------------------------------------------------------------------------

/// Solvate the system in a rectangular box of OPC water.
///
/// Uses the pre-equilibrated water box template (OPCBOX from solvents.lib)
/// to fill a rectangular box around the solute with a specified buffer distance.
///
/// # Algorithm
///
/// 1. Center solute at origin.
/// 2. Calculate solute bounding box.
/// 3. Compute target box: `box_dim = bbox_size + 2 * buffer`.
/// 4. Determine how many template replicas are needed in each dimension.
/// 5. Tile water molecules, offsetting by replica vectors.
/// 6. Remove any water molecule that has an atom within `closeness` of any
///    solute atom (cell-list accelerated).
/// 7. Remove any water molecule whose oxygen is outside the target box.
/// 8. Add surviving water molecules to the system and set box dimensions.
///
/// # Arguments
///
/// * `system` - The solute system to solvate (modified in place).
/// * `water_template` - The OPCBOX residue template from the force field library.
/// * `config` - Solvation parameters (buffer, closeness).
///
/// # Errors
///
/// Returns `Err` if the template lacks box dimensions, contains malformed
/// sub-residues, or the system has no atoms.
pub fn solvate_box(
    system: &mut System,
    water_template: &ResidueTemplate,
    config: &SolvateConfig,
) -> Result<(), String> {
    // -----------------------------------------------------------------------
    // Validate inputs
    // -----------------------------------------------------------------------
    if system.atoms.is_empty() {
        return Err("Cannot solvate an empty system".to_string());
    }

    let template_box = water_template
        .box_dimensions
        .ok_or("Water template has no box_dimensions")?;

    if template_box[0] <= 0.0 || template_box[1] <= 0.0 || template_box[2] <= 0.0 {
        return Err(format!(
            "Invalid template box dimensions: [{}, {}, {}]",
            template_box[0], template_box[1], template_box[2]
        ));
    }

    // -----------------------------------------------------------------------
    // Step 1: Center solute at origin
    // -----------------------------------------------------------------------
    system.center_at_origin();

    // -----------------------------------------------------------------------
    // Step 2: Compute solute bounding box
    // -----------------------------------------------------------------------
    let (bbox_min, bbox_max) = system.bounding_box();
    let bbox_size = [
        bbox_max[0] - bbox_min[0],
        bbox_max[1] - bbox_min[1],
        bbox_max[2] - bbox_min[2],
    ];

    // -----------------------------------------------------------------------
    // Step 3: Target box dimensions
    // -----------------------------------------------------------------------
    let box_x = bbox_size[0] + 2.0 * config.buffer;
    let box_y = bbox_size[1] + 2.0 * config.buffer;
    let box_z = bbox_size[2] + 2.0 * config.buffer;

    let half_box = [box_x / 2.0, box_y / 2.0, box_z / 2.0];

    // -----------------------------------------------------------------------
    // Step 4: Number of template replicas in each dimension
    // -----------------------------------------------------------------------
    let n_x = (box_x / template_box[0]).ceil() as i32;
    let n_y = (box_y / template_box[1]).ceil() as i32;
    let n_z = (box_z / template_box[2]).ceil() as i32;

    // Total replicated box size (before trimming).
    let total_x = n_x as f64 * template_box[0];
    let total_y = n_y as f64 * template_box[1];
    let total_z = n_z as f64 * template_box[2];

    let half_total = [total_x / 2.0, total_y / 2.0, total_z / 2.0];

    // -----------------------------------------------------------------------
    // Step 5: Extract template water molecules
    // -----------------------------------------------------------------------
    let template_waters = extract_water_molecules(water_template)?;

    // -----------------------------------------------------------------------
    // Step 6: Build cell list for the solute
    // -----------------------------------------------------------------------
    let cell_size = if config.closeness > 0.0 {
        config.closeness
    } else {
        1.0
    };

    // Collect solute positions into a contiguous array for cache efficiency.
    let solute_positions: Vec<[f64; 3]> = system.atoms.iter().map(|a| a.position).collect();

    let mut cell_list = CellList::new(cell_size);
    for (i, pos) in solute_positions.iter().enumerate() {
        cell_list.insert(i, pos);
    }

    // -----------------------------------------------------------------------
    // Step 7: Tile water molecules, reject clashing / out-of-box ones
    // -----------------------------------------------------------------------

    // Collect surviving water molecule positions. Each entry holds
    // the 4 atom positions [O, H1, H2, EPW] after offset.
    let mut surviving_waters: Vec<[[f64; 3]; 4]> = Vec::new();

    // Pre-estimate capacity: total waters = n_x * n_y * n_z * template_waters.len()
    // A large fraction will be removed, but reserving avoids many reallocations.
    let estimated_total =
        (n_x as usize) * (n_y as usize) * (n_z as usize) * template_waters.len();
    surviving_waters.reserve(estimated_total / 4); // rough guess

    for ix in 0..n_x {
        for iy in 0..n_y {
            for iz in 0..n_z {
                // Offset for this replica, centered at origin.
                let offset = [
                    ix as f64 * template_box[0] - half_total[0],
                    iy as f64 * template_box[1] - half_total[1],
                    iz as f64 * template_box[2] - half_total[2],
                ];

                for water in &template_waters {
                    // Compute offset positions for all 4 atoms.
                    let mut positions = [[0.0_f64; 3]; 4];
                    for (i, template_pos) in water.positions.iter().enumerate() {
                        positions[i] = [
                            template_pos[0] + offset[0],
                            template_pos[1] + offset[1],
                            template_pos[2] + offset[2],
                        ];
                    }

                    // --- Boundary check: oxygen center must be inside the target box ---
                    let o_pos = &positions[0];
                    if o_pos[0] < -half_box[0]
                        || o_pos[0] > half_box[0]
                        || o_pos[1] < -half_box[1]
                        || o_pos[1] > half_box[1]
                        || o_pos[2] < -half_box[2]
                        || o_pos[2] > half_box[2]
                    {
                        continue;
                    }

                    // --- Collision check: any water atom within closeness of solute ---
                    if cell_list.any_within_cutoff(
                        &positions,
                        &solute_positions,
                        config.closeness,
                    ) {
                        continue;
                    }

                    surviving_waters.push(positions);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 8: Sort surviving waters for cache-friendly insertion
    // -----------------------------------------------------------------------
    // Sort by oxygen Z, then Y, then X (space-filling order).
    surviving_waters.sort_unstable_by(|a, b| {
        let cmp_z = a[0][2].partial_cmp(&b[0][2]).unwrap_or(std::cmp::Ordering::Equal);
        if cmp_z != std::cmp::Ordering::Equal {
            return cmp_z;
        }
        let cmp_y = a[0][1].partial_cmp(&b[0][1]).unwrap_or(std::cmp::Ordering::Equal);
        if cmp_y != std::cmp::Ordering::Equal {
            return cmp_y;
        }
        a[0][0]
            .partial_cmp(&b[0][0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // -----------------------------------------------------------------------
    // Step 9: Add surviving water molecules to the system
    // -----------------------------------------------------------------------
    let n_waters = surviving_waters.len();

    // Pre-allocate capacity in the system vectors.
    system.atoms.reserve(n_waters * 4);
    system.bonds.reserve(n_waters * 3);
    system.residues.reserve(n_waters);

    for (wat_idx, positions) in surviving_waters.iter().enumerate() {
        let mut atoms = Vec::with_capacity(4);
        for i in 0..4 {
            atoms.push(Atom {
                name: WATER_NAMES[i].to_owned(),
                atom_type: WATER_TYPES[i].to_owned(),
                element: WATER_ELEMENTS[i].to_owned(),
                charge: WATER_CHARGES[i],
                mass: WATER_MASSES[i],
                atomic_number: WATER_ATOMIC_NUMBERS[i],
                position: positions[i],
                residue_idx: 0, // will be overwritten by add_residue
                born_radius: 0.0,
                screen: 0.0,
            });
        }

        // Sequential sequence numbers starting after existing residues.
        let seq_num = (system.n_residues() as i32) + 1;
        let bonds: Vec<(usize, usize)> = WATER_BONDS.to_vec();
        system.add_residue("WAT", 'W', seq_num, atoms, bonds);

        // Log progress periodically for large solvation runs.
        if (wat_idx + 1) % 5000 == 0 {
            log::debug!("Added {}/{} water molecules", wat_idx + 1, n_waters);
        }
    }

    // -----------------------------------------------------------------------
    // Step 10: Set periodic box dimensions
    // -----------------------------------------------------------------------
    system.box_dimensions = Some([box_x, box_y, box_z]);
    system.box_angles = Some([90.0, 90.0, 90.0]);

    log::info!(
        "Solvation complete: added {} water molecules ({} atoms). \
         Box dimensions: [{:.3}, {:.3}, {:.3}] A",
        n_waters,
        n_waters * 4,
        box_x,
        box_y,
        box_z,
    );

    Ok(())
}
