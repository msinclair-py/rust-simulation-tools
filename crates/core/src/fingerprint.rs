//! Fingerprint calculation module for molecular dynamics energy decomposition.
//!
//! This module provides efficient computation of per-residue interaction energies
//! using cell lists for neighbor searching and reaction field electrostatics.

use rayon::prelude::*;
use std::cell::RefCell;

use crate::util::distance_squared;

// Physical constants
const COULOMB_CONSTANT: f64 = 138.935458; // kJ/(mol·nm·e²)
const ELEMENTARY_CHARGE: f64 = 1.0; // in e
const EPSILON_SOLVENT: f64 = 80.0;
const CUTOFF_DISTANCE: f64 = 1.0; // nm
const CUTOFF_SQUARED: f64 = CUTOFF_DISTANCE * CUTOFF_DISTANCE;

/// Reaction field parameters for electrostatic calculations.
#[derive(Debug, Clone, Copy)]
pub struct ReactionFieldParams {
    pub krf: f64,
    pub crf: f64,
}

impl Default for ReactionFieldParams {
    fn default() -> Self {
        Self::new(CUTOFF_DISTANCE, EPSILON_SOLVENT)
    }
}

impl ReactionFieldParams {
    /// Create new reaction field parameters.
    ///
    /// # Arguments
    /// * `cutoff` - Cutoff distance in nm
    /// * `epsilon_solvent` - Solvent dielectric constant
    pub fn new(cutoff: f64, epsilon_solvent: f64) -> Self {
        let cutoff_cubed = cutoff * cutoff * cutoff;
        let krf = (1.0 / cutoff_cubed) * ((epsilon_solvent - 1.0) / (2.0 * epsilon_solvent + 1.0));
        let crf = (1.0 / cutoff) * ((3.0 * epsilon_solvent) / (2.0 * epsilon_solvent + 1.0));

        ReactionFieldParams { krf, crf }
    }

    /// Compute reaction field electrostatic energy between two charges.
    ///
    /// # Arguments
    /// * `qi` - Charge of particle i in elementary charges
    /// * `qj` - Charge of particle j in elementary charges
    /// * `r_squared` - Squared distance between particles in nm²
    ///
    /// # Returns
    /// Electrostatic energy in kJ/mol
    #[inline]
    pub fn compute_rf_energy(&self, qi: f64, qj: f64, r_squared: f64) -> f64 {
        if r_squared > CUTOFF_SQUARED {
            return 0.0;
        }

        let r = r_squared.sqrt();
        let charge_product = qi * qj * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE;

        COULOMB_CONSTANT * charge_product * ((1.0 / r) + self.krf * r_squared - self.crf)
    }
}

thread_local! {
    /// Thread-local reaction field parameters.
    static RF_PARAMS: RefCell<ReactionFieldParams> = RefCell::new(ReactionFieldParams::default());

    /// Thread-local buffer for neighbor searches to avoid allocations.
    static NEIGHBOR_BUFFER: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(256));
}

/// Cell list data structure for efficient neighbor searching.
#[derive(Debug)]
pub struct CellList {
    cell_size: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    cells: Vec<Vec<usize>>,
    min_coords: [f64; 3],
}

impl CellList {
    /// Create a new cell list from positions.
    ///
    /// # Arguments
    /// * `positions` - Particle positions as [x, y, z] coordinates in nm
    /// * `cell_size` - Size of each cell in nm (typically cutoff distance)
    pub fn new(positions: &[[f64; 3]], cell_size: f64) -> Self {
        if positions.is_empty() {
            return Self {
                cell_size,
                nx: 1,
                ny: 1,
                nz: 1,
                cells: vec![Vec::new()],
                min_coords: [0.0; 3],
            };
        }

        // Find bounding box
        let mut min_coords = positions[0];
        let mut max_coords = positions[0];

        for pos in positions.iter() {
            for dim in 0..3 {
                min_coords[dim] = min_coords[dim].min(pos[dim]);
                max_coords[dim] = max_coords[dim].max(pos[dim]);
            }
        }

        // Calculate grid dimensions
        let nx = ((max_coords[0] - min_coords[0]) / cell_size).ceil() as usize + 1;
        let ny = ((max_coords[1] - min_coords[1]) / cell_size).ceil() as usize + 1;
        let nz = ((max_coords[2] - min_coords[2]) / cell_size).ceil() as usize + 1;

        let total_cells = nx * ny * nz;
        let mut cells = vec![Vec::new(); total_cells];

        // Assign particles to cells
        for (idx, pos) in positions.iter().enumerate() {
            let cell_idx = Self::position_to_cell_index(pos, &min_coords, cell_size, nx, ny, nz);
            cells[cell_idx].push(idx);
        }

        Self {
            cell_size,
            nx,
            ny,
            nz,
            cells,
            min_coords,
        }
    }

    /// Convert position to cell index.
    #[inline]
    fn position_to_cell_index(
        pos: &[f64; 3],
        min_coords: &[f64; 3],
        cell_size: f64,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> usize {
        let ix = ((pos[0] - min_coords[0]) / cell_size) as usize;
        let iy = ((pos[1] - min_coords[1]) / cell_size) as usize;
        let iz = ((pos[2] - min_coords[2]) / cell_size) as usize;

        let ix = ix.min(nx - 1);
        let iy = iy.min(ny - 1);
        let iz = iz.min(nz - 1);

        ix + iy * nx + iz * nx * ny
    }

    /// Find neighbors within cutoff distance of a position.
    ///
    /// # Arguments
    /// * `pos` - Query position
    /// * `positions` - All particle positions
    /// * `cutoff_squared` - Squared cutoff distance
    /// * `neighbors` - Output buffer for neighbor indices
    pub fn find_neighbors(
        &self,
        pos: &[f64; 3],
        positions: &[[f64; 3]],
        cutoff_squared: f64,
        neighbors: &mut Vec<usize>,
    ) {
        neighbors.clear();

        // Get cell coordinates for query position (clamp to grid bounds for
        // queries outside the bounding box that may still be within cutoff)
        let ix = ((pos[0] - self.min_coords[0]) / self.cell_size) as isize;
        let iy = ((pos[1] - self.min_coords[1]) / self.cell_size) as isize;
        let iz = ((pos[2] - self.min_coords[2]) / self.cell_size) as isize;

        let ix = ix.clamp(0, self.nx as isize - 1);
        let iy = iy.clamp(0, self.ny as isize - 1);
        let iz = iz.clamp(0, self.nz as isize - 1);

        // Search neighboring cells
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let cx = ix + dx;
                    let cy = iy + dy;
                    let cz = iz + dz;

                    // Check bounds
                    if cx < 0 || cy < 0 || cz < 0 {
                        continue;
                    }
                    if cx >= self.nx as isize || cy >= self.ny as isize || cz >= self.nz as isize {
                        continue;
                    }

                    let cell_idx =
                        cx as usize + cy as usize * self.nx + cz as usize * self.nx * self.ny;

                    // Check all particles in this cell
                    for &particle_idx in &self.cells[cell_idx] {
                        let r_sq = distance_squared(pos, &positions[particle_idx]);
                        if r_sq <= cutoff_squared {
                            neighbors.push(particle_idx);
                        }
                    }
                }
            }
        }
    }
}

/// Data for partner molecules/residues.
#[derive(Debug, Clone)]
pub struct PartnerData {
    pub positions: Vec<[f64; 3]>,
    pub charges: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub epsilons: Vec<f64>,
}

impl PartnerData {
    /// Create new partner data from slices.
    ///
    /// # Arguments
    /// * `all_positions` - All particle positions
    /// * `all_charges` - All particle charges
    /// * `all_sigmas` - All particle LJ sigma parameters
    /// * `all_epsilons` - All particle LJ epsilon parameters
    /// * `partner_indices` - Indices of particles belonging to partner
    pub fn new(
        all_positions: &[[f64; 3]],
        all_charges: &[f64],
        all_sigmas: &[f64],
        all_epsilons: &[f64],
        partner_indices: &[usize],
    ) -> Self {
        let positions: Vec<[f64; 3]> = partner_indices
            .iter()
            .map(|&idx| all_positions[idx])
            .collect();

        let charges: Vec<f64> = partner_indices
            .iter()
            .map(|&idx| all_charges[idx])
            .collect();

        let sigmas: Vec<f64> = partner_indices.iter().map(|&idx| all_sigmas[idx]).collect();

        let epsilons: Vec<f64> = partner_indices
            .iter()
            .map(|&idx| all_epsilons[idx])
            .collect();

        Self {
            positions,
            charges,
            sigmas,
            epsilons,
        }
    }

    /// Create partner data from pre-extracted vectors (for FingerprintSession).
    pub fn from_vecs(
        positions: Vec<[f64; 3]>,
        charges: Vec<f64>,
        sigmas: Vec<f64>,
        epsilons: Vec<f64>,
    ) -> Self {
        Self {
            positions,
            charges,
            sigmas,
            epsilons,
        }
    }
}

/// Data for a single residue.
#[derive(Debug, Clone)]
pub struct ResidueData {
    pub positions: Vec<[f64; 3]>,
    pub charges: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub epsilons: Vec<f64>,
    pub residue_index: usize,
}

impl ResidueData {
    /// Create new residue data from slices.
    ///
    /// # Arguments
    /// * `positions` - All particle positions
    /// * `charges` - All particle charges
    /// * `sigmas` - All particle LJ sigma parameters
    /// * `epsilons` - All particle LJ epsilon parameters
    /// * `resmap_indices` - Residue assignment for each particle
    /// * `start` - Start index in arrays
    /// * `end` - End index in arrays (exclusive)
    pub fn new(
        positions: &[[f64; 3]],
        charges: &[f64],
        sigmas: &[f64],
        epsilons: &[f64],
        resmap_indices: &[usize],
        start: usize,
        end: usize,
    ) -> Self {
        let residue_index = resmap_indices[start];

        let mut res_positions = Vec::new();
        let mut res_charges = Vec::new();
        let mut res_sigmas = Vec::new();
        let mut res_epsilons = Vec::new();

        for i in start..end {
            if resmap_indices[i] == residue_index {
                res_positions.push(positions[i]);
                res_charges.push(charges[i]);
                res_sigmas.push(sigmas[i]);
                res_epsilons.push(epsilons[i]);
            }
        }

        Self {
            positions: res_positions,
            charges: res_charges,
            sigmas: res_sigmas,
            epsilons: res_epsilons,
            residue_index,
        }
    }

    /// Create residue data from global atom indices (for FingerprintSession).
    pub fn from_global_indices(
        positions: &[[f64; 3]],
        charges: &[f64],
        sigmas: &[f64],
        epsilons: &[f64],
        atom_indices: &[usize],
    ) -> Self {
        let mut res_positions = Vec::with_capacity(atom_indices.len());
        let mut res_charges = Vec::with_capacity(atom_indices.len());
        let mut res_sigmas = Vec::with_capacity(atom_indices.len());
        let mut res_epsilons = Vec::with_capacity(atom_indices.len());

        for &idx in atom_indices {
            res_positions.push(positions[idx]);
            res_charges.push(charges[idx]);
            res_sigmas.push(sigmas[idx]);
            res_epsilons.push(epsilons[idx]);
        }

        Self {
            positions: res_positions,
            charges: res_charges,
            sigmas: res_sigmas,
            epsilons: res_epsilons,
            residue_index: 0,
        }
    }
}

/// Compute interaction energy between a residue and partner.
///
/// # Arguments
/// * `residue` - Residue data
/// * `partner` - Partner data
/// * `partner_cell_list` - Cell list for partner particles
///
/// # Returns
/// Tuple of (electrostatic_energy, van_der_waals_energy) in kJ/mol
pub fn compute_residue_energy(
    residue: &ResidueData,
    partner: &PartnerData,
    partner_cell_list: &CellList,
) -> (f64, f64) {
    let mut total_elec = 0.0;
    let mut total_vdw = 0.0;

    RF_PARAMS.with(|params| {
        let rf_params = params.borrow();

        NEIGHBOR_BUFFER.with(|buffer| {
            let mut neighbors = buffer.borrow_mut();

            for (i, pos_i) in residue.positions.iter().enumerate() {
                let qi = residue.charges[i];
                let sigma_i = residue.sigmas[i];
                let epsilon_i = residue.epsilons[i];

                // Find neighbors using cell list
                partner_cell_list.find_neighbors(
                    pos_i,
                    &partner.positions,
                    CUTOFF_SQUARED,
                    &mut neighbors,
                );

                for &j in neighbors.iter() {
                    let qj = partner.charges[j];
                    let sigma_j = partner.sigmas[j];
                    let epsilon_j = partner.epsilons[j];

                    let r_sq = distance_squared(pos_i, &partner.positions[j]);

                    if r_sq > CUTOFF_SQUARED {
                        continue;
                    }

                    // Electrostatic energy (reaction field)
                    let elec_energy = rf_params.compute_rf_energy(qi, qj, r_sq);
                    total_elec += elec_energy;

                    // Van der Waals energy (Lennard-Jones)
                    if epsilon_i > 0.0 && epsilon_j > 0.0 {
                        let sigma = (sigma_i + sigma_j) * 0.5;
                        let epsilon = (epsilon_i * epsilon_j).sqrt();

                        let sigma_sq = sigma * sigma;
                        let ratio_sq = sigma_sq / r_sq;
                        let ratio_6 = ratio_sq * ratio_sq * ratio_sq;
                        let ratio_12 = ratio_6 * ratio_6;

                        let vdw_energy = 4.0 * epsilon * (ratio_12 - ratio_6);
                        total_vdw += vdw_energy;
                    }
                }
            }
        });
    });

    (total_elec, total_vdw)
}

/// Compute fingerprints (per-residue energies) from residue and partner data.
///
/// # Arguments
/// * `residues` - Vector of residue data
/// * `partner` - Partner data
///
/// # Returns
/// Tuple of (electrostatic_energies, vdw_energies) vectors in kJ/mol
pub fn compute_fingerprints_from_residues(
    residues: &[ResidueData],
    partner: &PartnerData,
) -> (Vec<f64>, Vec<f64>) {
    // Build cell list for partner
    let partner_cell_list = CellList::new(&partner.positions, CUTOFF_DISTANCE);

    // Compute energies in parallel
    let results: Vec<(f64, f64)> = residues
        .par_iter()
        .map(|residue| compute_residue_energy(residue, partner, &partner_cell_list))
        .collect();

    // Separate into electrostatic and vdw vectors
    let mut elec_energies = Vec::with_capacity(results.len());
    let mut vdw_energies = Vec::with_capacity(results.len());

    for (elec, vdw) in results {
        elec_energies.push(elec);
        vdw_energies.push(vdw);
    }

    (elec_energies, vdw_energies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reaction_field_params() {
        let rf = ReactionFieldParams::new(1.0, 80.0);
        assert!(rf.krf > 0.0);
        assert!(rf.crf > 0.0);
    }

    #[test]
    fn test_rf_energy_beyond_cutoff() {
        let rf = ReactionFieldParams::default();
        let energy = rf.compute_rf_energy(1.0, 1.0, CUTOFF_SQUARED + 0.1);
        assert_eq!(energy, 0.0);
    }

    #[test]
    fn test_cell_list_creation() {
        let positions = vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]];

        let cell_list = CellList::new(&positions, 1.0);
        assert!(cell_list.nx > 0);
        assert!(cell_list.ny > 0);
        assert!(cell_list.nz > 0);
    }

    #[test]
    fn test_cell_list_find_neighbors() {
        let positions = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]];

        let cell_list = CellList::new(&positions, 1.0);
        let mut neighbors = Vec::new();

        cell_list.find_neighbors(&positions[0], &positions, 1.0, &mut neighbors);

        // Should find itself and the particle at 0.5
        assert!(neighbors.len() >= 1);
    }

    #[test]
    fn test_partner_data_creation() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let charges = vec![1.0, -1.0, 0.5];
        let sigmas = vec![0.3, 0.3, 0.3];
        let epsilons = vec![0.5, 0.5, 0.5];
        let indices = vec![0, 2];

        let partner = PartnerData::new(&positions, &charges, &sigmas, &epsilons, &indices);

        assert_eq!(partner.positions.len(), 2);
        assert_eq!(partner.charges[0], 1.0);
        assert_eq!(partner.charges[1], 0.5);
    }

    #[test]
    fn test_residue_data_creation() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let charges = vec![1.0, -1.0, 0.5];
        let sigmas = vec![0.3, 0.3, 0.3];
        let epsilons = vec![0.5, 0.5, 0.5];
        let resmap = vec![0, 0, 1];

        let residue = ResidueData::new(&positions, &charges, &sigmas, &epsilons, &resmap, 0, 3);

        assert_eq!(residue.residue_index, 0);
        assert_eq!(residue.positions.len(), 2);
    }

    #[test]
    fn test_compute_residue_energy() {
        let residue = ResidueData {
            positions: vec![[0.0, 0.0, 0.0]],
            charges: vec![1.0],
            sigmas: vec![0.3],
            epsilons: vec![0.5],
            residue_index: 0,
        };

        let partner = PartnerData {
            positions: vec![[0.5, 0.0, 0.0]],
            charges: vec![-1.0],
            sigmas: vec![0.3],
            epsilons: vec![0.5],
        };

        let cell_list = CellList::new(&partner.positions, CUTOFF_DISTANCE);
        let (elec, vdw) = compute_residue_energy(&residue, &partner, &cell_list);

        // Should have negative electrostatic energy (opposite charges)
        assert!(elec < 0.0);
        // Should have some vdw interaction
        assert!(vdw.abs() > 0.0);
    }

    #[test]
    fn test_compute_fingerprints() {
        let residues = vec![
            ResidueData {
                positions: vec![[0.0, 0.0, 0.0]],
                charges: vec![1.0],
                sigmas: vec![0.3],
                epsilons: vec![0.5],
                residue_index: 0,
            },
            ResidueData {
                positions: vec![[5.0, 5.0, 5.0]], // Far away, no interaction
                charges: vec![1.0],
                sigmas: vec![0.3],
                epsilons: vec![0.5],
                residue_index: 1,
            },
        ];

        let partner = PartnerData {
            positions: vec![[0.5, 0.0, 0.0]],
            charges: vec![-1.0],
            sigmas: vec![0.3],
            epsilons: vec![0.5],
        };

        let (elec, vdw) = compute_fingerprints_from_residues(&residues, &partner);

        assert_eq!(elec.len(), 2);
        assert_eq!(vdw.len(), 2);

        // First residue should have interactions
        assert!(elec[0].abs() > 0.0);
        assert!(vdw[0].abs() > 0.0);

        // Second residue should have no interactions (too far)
        assert_eq!(elec[1], 0.0);
        assert_eq!(vdw[1], 0.0);
    }

    #[test]
    fn test_empty_cell_list() {
        let positions: Vec<[f64; 3]> = vec![];
        let cell_list = CellList::new(&positions, 1.0);

        assert_eq!(cell_list.nx, 1);
        assert_eq!(cell_list.ny, 1);
        assert_eq!(cell_list.nz, 1);
    }
}
