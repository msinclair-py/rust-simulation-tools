//! Cell-list based neighbor list for non-bonded interactions.
//!
//! Provides an efficient O(N) neighbor list construction algorithm using
//! spatial hashing (cell lists) with periodic boundary conditions.  The list
//! stores unique pairs (i, j) with i < j whose minimum-image distance is less
//! than `cutoff + skin`.  A skin distance allows reusing the list across
//! several force evaluations -- the list only needs rebuilding when any atom
//! has displaced by more than `skin / 2` from the position it had when the
//! list was last built.

/// A neighbor list storing pairs of atoms within cutoff distance.
///
/// Pairs are built using a cell-list (spatial hashing) algorithm with
/// minimum-image periodic boundary conditions.  The optional *skin*
/// distance widens the search radius so the list can be reused across
/// multiple time-steps or minimization iterations without rebuilding.
///
/// # Examples
///
/// ```
/// use rst_minimize::neighbor_list::NeighborList;
///
/// let mut nlist = NeighborList::new(10.0, 2.0);
/// // Two atoms 5 A apart in a 30x30x30 box.
/// let positions = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
/// let box_dims = [30.0, 30.0, 30.0];
/// nlist.build(&positions, &box_dims);
/// assert_eq!(nlist.pairs.len(), 1);
/// assert_eq!(nlist.pairs[0], (0, 1));
/// ```
pub struct NeighborList {
    /// Pairs of atom indices (i, j) where i < j.
    pub pairs: Vec<(usize, usize)>,
    /// Cutoff distance used to build this list.
    pub cutoff: f64,
    /// Skin distance for list rebuilding.
    pub skin: f64,
    /// Positions at the time the list was last built (for displacement checking).
    last_positions: Vec<[f64; 3]>,
}

impl NeighborList {
    /// Create a new empty neighbor list with the given cutoff and skin distances.
    ///
    /// The effective search radius is `cutoff + skin`.  A larger skin reduces
    /// the frequency of rebuilds at the cost of evaluating more pairs per
    /// force call.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Non-bonded interaction cutoff in Angstroms.
    /// * `skin` - Buffer distance in Angstroms.
    pub fn new(cutoff: f64, skin: f64) -> Self {
        Self {
            pairs: Vec::new(),
            cutoff,
            skin,
            last_positions: Vec::new(),
        }
    }

    /// Build (or rebuild) the neighbor list from scratch.
    ///
    /// Uses a cell-list algorithm for O(N) construction with periodic
    /// boundary conditions.  After building, `last_positions` is updated
    /// so that [`needs_rebuild`](Self::needs_rebuild) can check displacement.
    ///
    /// # Arguments
    ///
    /// * `positions` - Cartesian coordinates of all atoms (Angstroms).
    /// * `box_dims`  - Orthorhombic box dimensions `[Lx, Ly, Lz]` (Angstroms).
    pub fn build(&mut self, positions: &[[f64; 3]], box_dims: &[f64; 3]) {
        let n_atoms = positions.len();
        let r_list = self.cutoff + self.skin;
        let r_list_sq = r_list * r_list;

        self.pairs.clear();

        if n_atoms == 0 {
            self.last_positions.clear();
            return;
        }

        // ------------------------------------------------------------------
        // 1. Determine cell grid dimensions.
        //    Each cell side must be >= r_list so that only the 26 immediate
        //    neighbors (plus self) need to be checked.
        // ------------------------------------------------------------------
        let n_cells_x = (box_dims[0] / r_list).floor().max(1.0) as usize;
        let n_cells_y = (box_dims[1] / r_list).floor().max(1.0) as usize;
        let n_cells_z = (box_dims[2] / r_list).floor().max(1.0) as usize;

        let n_cells_total = n_cells_x * n_cells_y * n_cells_z;

        // ------------------------------------------------------------------
        // 2. Assign atoms to cells.
        //    Wrap positions into [0, L) before binning.
        // ------------------------------------------------------------------
        let inv_lx = 1.0 / box_dims[0];
        let inv_ly = 1.0 / box_dims[1];
        let inv_lz = 1.0 / box_dims[2];

        // cell_of[atom] = flat cell index
        let mut cell_of = vec![0usize; n_atoms];
        // heads[cell] = start of linked list (atom index), usize::MAX = end
        let mut heads: Vec<usize> = vec![usize::MAX; n_cells_total];
        // next[atom] = next atom in same cell, usize::MAX = end
        let mut next: Vec<usize> = vec![usize::MAX; n_atoms];

        for i in 0..n_atoms {
            // Fractional coordinate in [0, 1)
            let fx = positions[i][0] * inv_lx - (positions[i][0] * inv_lx).floor();
            let fy = positions[i][1] * inv_ly - (positions[i][1] * inv_ly).floor();
            let fz = positions[i][2] * inv_lz - (positions[i][2] * inv_lz).floor();

            let cx = ((fx * n_cells_x as f64) as usize).min(n_cells_x - 1);
            let cy = ((fy * n_cells_y as f64) as usize).min(n_cells_y - 1);
            let cz = ((fz * n_cells_z as f64) as usize).min(n_cells_z - 1);

            let cell_idx = cx * n_cells_y * n_cells_z + cy * n_cells_z + cz;
            cell_of[i] = cell_idx;
            next[i] = heads[cell_idx];
            heads[cell_idx] = i;
        }

        // ------------------------------------------------------------------
        // 3. Build neighbor half-shell offsets.
        //    To avoid double-counting, we enumerate the 13 distinct neighbor
        //    directions (half of 26) plus the self-cell.  Within the self-cell
        //    we only pair (i, j) with i < j.
        // ------------------------------------------------------------------

        // Precompute the 13 "forward" neighbor offsets (dx, dy, dz) where
        // the flattened offset is strictly positive, plus (0,0,0) for self.
        // We iterate all 27 combinations and keep those with offset > 0 in
        // the linearised cell index, plus (0,0,0).
        let mut half_shell: Vec<(i32, i32, i32)> = Vec::with_capacity(14);
        half_shell.push((0, 0, 0)); // self-cell
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    // Use lexicographic ordering to pick exactly one of
                    // each (dx, dy, dz) / (-dx, -dy, -dz) pair.
                    if (dx, dy, dz) > (0, 0, 0) {
                        half_shell.push((dx, dy, dz));
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // 4. Iterate over cells and neighbors to find pairs.
        // ------------------------------------------------------------------
        let nc_x = n_cells_x as i32;
        let nc_y = n_cells_y as i32;
        let nc_z = n_cells_z as i32;

        for cx in 0..n_cells_x {
            for cy in 0..n_cells_y {
                for cz in 0..n_cells_z {
                    let cell_a = cx * n_cells_y * n_cells_z + cy * n_cells_z + cz;

                    for &(dx, dy, dz) in &half_shell {
                        // Neighbor cell with PBC wrapping.
                        let nx = ((cx as i32 + dx).rem_euclid(nc_x)) as usize;
                        let ny = ((cy as i32 + dy).rem_euclid(nc_y)) as usize;
                        let nz = ((cz as i32 + dz).rem_euclid(nc_z)) as usize;
                        let cell_b = nx * n_cells_y * n_cells_z + ny * n_cells_z + nz;

                        // When the grid is very small, a neighbor offset may
                        // wrap back to the same cell.  Treat that as a self-
                        // cell interaction only for the (0,0,0) offset to
                        // avoid double-counting.
                        let is_self_offset = dx == 0 && dy == 0 && dz == 0;
                        if cell_b == cell_a && !is_self_offset {
                            continue;
                        }
                        let is_self = is_self_offset;

                        // Walk linked lists for cell_a and cell_b.
                        let mut atom_i = heads[cell_a];
                        while atom_i != usize::MAX {
                            let start_j = if is_self {
                                next[atom_i] // only pair with atoms later in list
                            } else {
                                heads[cell_b]
                            };

                            let mut atom_j = start_j;
                            while atom_j != usize::MAX {
                                // Minimum image distance squared.
                                let mut r2 = 0.0f64;
                                for dim in 0..3 {
                                    let mut d = positions[atom_i][dim]
                                        - positions[atom_j][dim];
                                    d -= (d / box_dims[dim]).round() * box_dims[dim];
                                    r2 += d * d;
                                }

                                if r2 < r_list_sq {
                                    let (lo, hi) = if atom_i < atom_j {
                                        (atom_i, atom_j)
                                    } else {
                                        (atom_j, atom_i)
                                    };
                                    self.pairs.push((lo, hi));
                                }

                                atom_j = next[atom_j];
                            }

                            atom_i = next[atom_i];
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // 5. Save positions for displacement checking.
        // ------------------------------------------------------------------
        self.last_positions.clear();
        self.last_positions.extend_from_slice(positions);

        // Sort pairs for deterministic iteration order (optional but helpful
        // for debugging and reproducibility).
        self.pairs.sort_unstable();

        // Remove any duplicates that may arise from cells mapping to the
        // same index when the number of cells is very small (e.g. 1 or 2).
        self.pairs.dedup();
    }

    /// Check if the list needs rebuilding.
    ///
    /// Returns `true` if any atom has moved more than `skin / 2` from the
    /// position it had when the list was last built, or if the list has
    /// never been built.
    ///
    /// # Arguments
    ///
    /// * `positions` - Current Cartesian coordinates of all atoms.
    pub fn needs_rebuild(&self, positions: &[[f64; 3]]) -> bool {
        if self.last_positions.len() != positions.len() {
            return true;
        }
        let half_skin_sq = (self.skin * 0.5) * (self.skin * 0.5);
        for (cur, old) in positions.iter().zip(self.last_positions.iter()) {
            let dx = cur[0] - old[0];
            let dy = cur[1] - old[1];
            let dz = cur[2] - old[2];
            if dx * dx + dy * dy + dz * dz > half_skin_sq {
                return true;
            }
        }
        false
    }

    /// Rebuild the list only if the displacement criterion is exceeded.
    ///
    /// This is the recommended entry point during a minimization or MD loop:
    /// it avoids the cost of a full rebuild when atom displacements are small.
    ///
    /// # Arguments
    ///
    /// * `positions` - Current Cartesian coordinates of all atoms.
    /// * `box_dims`  - Orthorhombic box dimensions `[Lx, Ly, Lz]`.
    pub fn update_if_needed(
        &mut self,
        positions: &[[f64; 3]],
        box_dims: &[f64; 3],
    ) {
        if self.needs_rebuild(positions) {
            self.build(positions, box_dims);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_atoms_within_cutoff() {
        let mut nlist = NeighborList::new(10.0, 2.0);
        let positions = [[1.0, 1.0, 1.0], [5.0, 1.0, 1.0]];
        let box_dims = [30.0, 30.0, 30.0];
        nlist.build(&positions, &box_dims);
        assert_eq!(nlist.pairs.len(), 1);
        assert_eq!(nlist.pairs[0], (0, 1));
    }

    #[test]
    fn two_atoms_beyond_cutoff() {
        let mut nlist = NeighborList::new(5.0, 1.0);
        let positions = [[1.0, 1.0, 1.0], [20.0, 1.0, 1.0]];
        let box_dims = [30.0, 30.0, 30.0];
        nlist.build(&positions, &box_dims);
        assert!(nlist.pairs.is_empty());
    }

    #[test]
    fn pbc_wrap_around() {
        // Atoms on opposite sides of the box: direct distance = 28,
        // but minimum-image distance = 2.  With cutoff=5, skin=1 they
        // should be neighbors.
        let mut nlist = NeighborList::new(5.0, 1.0);
        let positions = [[1.0, 15.0, 15.0], [29.0, 15.0, 15.0]];
        let box_dims = [30.0, 30.0, 30.0];
        nlist.build(&positions, &box_dims);
        assert_eq!(nlist.pairs.len(), 1, "PBC wrap-around pair missing");
        assert_eq!(nlist.pairs[0], (0, 1));
    }

    #[test]
    fn pair_ordering_i_lt_j() {
        let mut nlist = NeighborList::new(100.0, 0.0);
        let positions = [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
        ];
        let box_dims = [200.0, 200.0, 200.0];
        nlist.build(&positions, &box_dims);
        for &(i, j) in &nlist.pairs {
            assert!(i < j, "pair ({}, {}) violates i < j", i, j);
        }
    }

    #[test]
    fn no_duplicate_pairs() {
        let mut nlist = NeighborList::new(100.0, 0.0);
        let positions = [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
        ];
        let box_dims = [200.0, 200.0, 200.0];
        nlist.build(&positions, &box_dims);

        let mut sorted = nlist.pairs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            nlist.pairs.len(),
            sorted.len(),
            "duplicate pairs detected"
        );
    }

    #[test]
    fn needs_rebuild_after_large_displacement() {
        let mut nlist = NeighborList::new(10.0, 2.0);
        let positions_a = [[1.0, 1.0, 1.0], [5.0, 1.0, 1.0]];
        let box_dims = [30.0, 30.0, 30.0];
        nlist.build(&positions_a, &box_dims);
        assert!(!nlist.needs_rebuild(&positions_a));

        // Move atom 0 by 1.5 A > skin/2 = 1.0.
        let positions_b = [[2.5, 1.0, 1.0], [5.0, 1.0, 1.0]];
        assert!(nlist.needs_rebuild(&positions_b));
    }

    #[test]
    fn needs_rebuild_false_for_small_displacement() {
        let mut nlist = NeighborList::new(10.0, 4.0);
        let positions_a = [[1.0, 1.0, 1.0], [5.0, 1.0, 1.0]];
        let box_dims = [30.0, 30.0, 30.0];
        nlist.build(&positions_a, &box_dims);

        // Move atom 0 by 0.5 A < skin/2 = 2.0.
        let positions_b = [[1.5, 1.0, 1.0], [5.0, 1.0, 1.0]];
        assert!(!nlist.needs_rebuild(&positions_b));
    }

    #[test]
    fn update_if_needed_skips_rebuild() {
        let mut nlist = NeighborList::new(10.0, 4.0);
        let positions = [[1.0, 1.0, 1.0], [5.0, 1.0, 1.0]];
        let box_dims = [30.0, 30.0, 30.0];
        nlist.build(&positions, &box_dims);
        let count_before = nlist.pairs.len();

        // Tiny displacement -- should not trigger rebuild.
        let positions2 = [[1.1, 1.0, 1.0], [5.0, 1.0, 1.0]];
        nlist.update_if_needed(&positions2, &box_dims);
        assert_eq!(nlist.pairs.len(), count_before);
    }

    #[test]
    fn empty_system() {
        let mut nlist = NeighborList::new(10.0, 2.0);
        let positions: &[[f64; 3]] = &[];
        let box_dims = [30.0, 30.0, 30.0];
        nlist.build(positions, &box_dims);
        assert!(nlist.pairs.is_empty());
    }

    #[test]
    fn single_atom() {
        let mut nlist = NeighborList::new(10.0, 2.0);
        let positions = [[5.0, 5.0, 5.0]];
        let box_dims = [30.0, 30.0, 30.0];
        nlist.build(&positions, &box_dims);
        assert!(nlist.pairs.is_empty());
    }

    #[test]
    fn all_pairs_found_small_system() {
        // 4 atoms in a tiny box where every pair is within cutoff.
        let mut nlist = NeighborList::new(100.0, 0.0);
        let positions = [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ];
        let box_dims = [200.0, 200.0, 200.0];
        nlist.build(&positions, &box_dims);
        // n*(n-1)/2 = 6 pairs
        assert_eq!(nlist.pairs.len(), 6);
    }

    #[test]
    fn small_box_single_cell() {
        // Box so small that only 1 cell in each dimension.
        let mut nlist = NeighborList::new(10.0, 2.0);
        let positions = [
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
        ];
        let box_dims = [5.0, 5.0, 5.0];
        nlist.build(&positions, &box_dims);
        assert_eq!(nlist.pairs.len(), 1);
        assert_eq!(nlist.pairs[0], (0, 1));
    }
}
