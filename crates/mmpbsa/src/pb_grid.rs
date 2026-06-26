//! Grid infrastructure for finite-difference Poisson-Boltzmann solver.
//!
//! Provides a 3D grid, charge mapping via trilinear interpolation,
//! dielectric map assignment using molecular surface tests, and
//! ionic strength (kappa²) map assignment.

use rayon::prelude::*;
use rst_core::amber::prmtop::AmberTopology;
use std::collections::VecDeque;

/// A regular 3D grid for finite-difference PB calculations.
#[derive(Debug, Clone)]
pub struct PbGrid {
    /// Number of grid points in each dimension (nx, ny, nz).
    pub dims: [usize; 3],
    /// Grid spacing in Angstroms (hx, hy, hz).
    pub spacing: [f64; 3],
    /// Origin (minimum corner) in Angstroms.
    pub origin: [f64; 3],
    /// Flat storage: data[ix + nx*(iy + ny*iz)].
    pub data: Vec<f64>,
}

impl PbGrid {
    /// Create a new grid initialized to zero.
    pub fn new(dims: [usize; 3], spacing: [f64; 3], origin: [f64; 3]) -> Self {
        let n = dims[0] * dims[1] * dims[2];
        Self {
            dims,
            spacing,
            origin,
            data: vec![0.0; n],
        }
    }

    /// Create a lightweight grid descriptor without allocating data storage.
    ///
    /// Useful when the grid is used only for its geometry (dims, spacing, origin)
    /// and indexing methods, not for storing per-point data (e.g. multigrid level
    /// descriptors, coarse grids used only as stencil geometry).
    pub fn descriptor(dims: [usize; 3], spacing: [f64; 3], origin: [f64; 3]) -> Self {
        Self {
            dims,
            spacing,
            origin,
            data: Vec::new(),
        }
    }

    /// Create a grid filled with a constant value.
    pub fn filled(dims: [usize; 3], spacing: [f64; 3], origin: [f64; 3], value: f64) -> Self {
        let n = dims[0] * dims[1] * dims[2];
        Self {
            dims,
            spacing,
            origin,
            data: vec![value; n],
        }
    }

    /// Total number of grid points.
    #[inline]
    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Flat index from 3D indices.
    #[inline]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + self.dims[0] * (iy + self.dims[1] * iz)
    }

    /// Cartesian point for grid indices.
    #[inline]
    pub fn point(&self, ix: usize, iy: usize, iz: usize) -> [f64; 3] {
        [
            self.origin[0] + ix as f64 * self.spacing[0],
            self.origin[1] + iy as f64 * self.spacing[1],
            self.origin[2] + iz as f64 * self.spacing[2],
        ]
    }

    /// Fractional grid coordinates for a Cartesian point.
    #[inline]
    pub fn frac_coords(&self, point: &[f64; 3]) -> [f64; 3] {
        [
            (point[0] - self.origin[0]) / self.spacing[0],
            (point[1] - self.origin[1]) / self.spacing[1],
            (point[2] - self.origin[2]) / self.spacing[2],
        ]
    }

    /// Trilinear interpolation of grid data at a Cartesian point.
    pub fn interpolate(&self, point: &[f64; 3]) -> f64 {
        let fc = self.frac_coords(point);
        let ix = fc[0].floor() as isize;
        let iy = fc[1].floor() as isize;
        let iz = fc[2].floor() as isize;

        // Clamp to valid range
        let nx = self.dims[0] as isize;
        let ny = self.dims[1] as isize;
        let nz = self.dims[2] as isize;

        if ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1 || iz < 0 || iz >= nz - 1 {
            return 0.0;
        }

        let ix = ix as usize;
        let iy = iy as usize;
        let iz = iz as usize;

        let dx = fc[0] - ix as f64;
        let dy = fc[1] - iy as f64;
        let dz = fc[2] - iz as f64;

        let c000 = self.data[self.index(ix, iy, iz)];
        let c100 = self.data[self.index(ix + 1, iy, iz)];
        let c010 = self.data[self.index(ix, iy + 1, iz)];
        let c110 = self.data[self.index(ix + 1, iy + 1, iz)];
        let c001 = self.data[self.index(ix, iy, iz + 1)];
        let c101 = self.data[self.index(ix + 1, iy, iz + 1)];
        let c011 = self.data[self.index(ix, iy + 1, iz + 1)];
        let c111 = self.data[self.index(ix + 1, iy + 1, iz + 1)];

        let c00 = c000 * (1.0 - dx) + c100 * dx;
        let c10 = c010 * (1.0 - dx) + c110 * dx;
        let c01 = c001 * (1.0 - dx) + c101 * dx;
        let c11 = c011 * (1.0 - dx) + c111 * dx;

        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;

        c0 * (1.0 - dz) + c1 * dz
    }

    /// Trilinear interpolation using an external data array instead of `self.data`.
    ///
    /// This avoids cloning the data into a temporary `PbGrid` when the caller
    /// already has the values in a separate buffer (e.g. the potential array
    /// from the PB solver).
    pub fn interpolate_with_data(&self, point: &[f64; 3], data: &[f64]) -> f64 {
        let fc = self.frac_coords(point);
        let ix = fc[0].floor() as isize;
        let iy = fc[1].floor() as isize;
        let iz = fc[2].floor() as isize;

        let nx = self.dims[0] as isize;
        let ny = self.dims[1] as isize;
        let nz = self.dims[2] as isize;

        if ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1 || iz < 0 || iz >= nz - 1 {
            return 0.0;
        }

        let ix = ix as usize;
        let iy = iy as usize;
        let iz = iz as usize;

        let dx = fc[0] - ix as f64;
        let dy = fc[1] - iy as f64;
        let dz = fc[2] - iz as f64;

        let c000 = data[self.index(ix, iy, iz)];
        let c100 = data[self.index(ix + 1, iy, iz)];
        let c010 = data[self.index(ix, iy + 1, iz)];
        let c110 = data[self.index(ix + 1, iy + 1, iz)];
        let c001 = data[self.index(ix, iy, iz + 1)];
        let c101 = data[self.index(ix + 1, iy, iz + 1)];
        let c011 = data[self.index(ix, iy + 1, iz + 1)];
        let c111 = data[self.index(ix + 1, iy + 1, iz + 1)];

        let c00 = c000 * (1.0 - dx) + c100 * dx;
        let c10 = c010 * (1.0 - dx) + c110 * dx;
        let c01 = c001 * (1.0 - dx) + c101 * dx;
        let c11 = c011 * (1.0 - dx) + c111 * dx;

        let c0 = c00 * (1.0 - dy) + c10 * dy;
        let c1 = c01 * (1.0 - dy) + c11 * dy;

        c0 * (1.0 - dz) + c1 * dz
    }
}

/// Auto-size a grid from molecular coordinates.
///
/// Ensures odd dimensions so that there is a well-defined center point.
pub fn auto_grid(coords: &[[f64; 3]], spacing: f64, buffer: f64) -> PbGrid {
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for c in coords {
        for d in 0..3 {
            if c[d] < min[d] {
                min[d] = c[d];
            }
            if c[d] > max[d] {
                max[d] = c[d];
            }
        }
    }

    let mut dims = [0usize; 3];
    let mut origin = [0.0f64; 3];
    for d in 0..3 {
        let extent = max[d] - min[d] + 2.0 * buffer;
        let n = (extent / spacing).ceil() as usize + 1;
        // Ensure odd
        dims[d] = if n.is_multiple_of(2) { n + 1 } else { n };
        let grid_extent = (dims[d] - 1) as f64 * spacing;
        let center = 0.5 * (min[d] + max[d]);
        origin[d] = center - 0.5 * grid_extent;
    }

    PbGrid::new(dims, [spacing; 3], origin)
}

/// Map atomic charges onto a grid using trilinear interpolation.
///
/// Returns the charge density map (charges spread to grid points).
pub fn map_charges(grid: &PbGrid, coords: &[[f64; 3]], charges: &[f64]) -> Vec<f64> {
    let n = grid.len();
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];

    // Use per-thread accumulation for parallelism
    let n_atoms = coords.len();
    if n_atoms == 0 {
        return vec![0.0; n];
    }

    // Parallel: split atoms into chunks, accumulate per-thread, then sum
    let chunk_size = (n_atoms / rayon::current_num_threads()).max(64);
    let thread_maps: Vec<Vec<f64>> = coords
        .par_chunks(chunk_size)
        .zip(charges.par_chunks(chunk_size))
        .map(|(coord_chunk, charge_chunk)| {
            let mut local = vec![0.0f64; n];
            for (c, &q) in coord_chunk.iter().zip(charge_chunk.iter()) {
                let fc = grid.frac_coords(c);
                let ix = fc[0].floor() as isize;
                let iy = fc[1].floor() as isize;
                let iz = fc[2].floor() as isize;

                if ix < 0
                    || ix >= nx as isize - 1
                    || iy < 0
                    || iy >= ny as isize - 1
                    || iz < 0
                    || iz >= nz as isize - 1
                {
                    continue;
                }

                let ix = ix as usize;
                let iy = iy as usize;
                let iz = iz as usize;

                let dx = fc[0] - ix as f64;
                let dy = fc[1] - iy as f64;
                let dz = fc[2] - iz as f64;

                // Spread charge to 8 corners
                let hx = grid.spacing[0];
                let hy = grid.spacing[1];
                let hz = grid.spacing[2];
                let vol = hx * hy * hz;

                // Charge density: q / volume, distributed by trilinear weights
                let q_density = q / vol;
                for (dix, wx) in [(0, 1.0 - dx), (1, dx)] {
                    for (diy, wy) in [(0, 1.0 - dy), (1, dy)] {
                        for (diz, wz) in [(0, 1.0 - dz), (1, dz)] {
                            let idx = grid.index(ix + dix, iy + diy, iz + diz);
                            local[idx] += q_density * wx * wy * wz;
                        }
                    }
                }
            }
            local
        })
        .collect();

    // Sum thread-local maps
    let mut result = vec![0.0f64; n];
    for local in &thread_maps {
        for i in 0..n {
            result[i] += local[i];
        }
    }
    result
}

/// Face-centered dielectric maps for the three grid dimensions.
#[derive(Debug, Clone)]
pub struct DielectricMaps {
    /// ε at x-faces: eps_x[i,j,k] = ε at midpoint between (i,j,k) and (i+1,j,k).
    pub eps_x: Vec<f64>,
    /// ε at y-faces.
    pub eps_y: Vec<f64>,
    /// ε at z-faces.
    pub eps_z: Vec<f64>,
    /// Grid dimensions (same as PbGrid).
    pub dims: [usize; 3],
}

impl DielectricMaps {
    /// Flat index from 3D indices.
    #[inline]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + self.dims[0] * (iy + self.dims[1] * iz)
    }
}

/// Cell-based spatial hash for fast proximity queries against atom spheres.
///
/// Bins atoms into a uniform grid of cells, enabling O(1) amortized
/// `point_inside_molecule` queries instead of the O(N_atoms) linear scan.
struct AtomSpatialHash {
    #[allow(dead_code)]
    cell_size: f64,
    inv_cell_size: f64,
    origin: [f64; 3],
    grid_dims: [usize; 3],
    /// For each cell, the indices of atoms whose inflated sphere may overlap it.
    cells: Vec<Vec<usize>>,
}

impl AtomSpatialHash {
    /// Build a spatial hash from atom positions and the maximum effective radius.
    ///
    /// `max_effective_radius` should be `max(radii) + max(probe_radii_used)`.
    fn new(coords: &[[f64; 3]], max_effective_radius: f64) -> Self {
        if coords.is_empty() {
            return Self {
                cell_size: 1.0,
                inv_cell_size: 1.0,
                origin: [0.0; 3],
                grid_dims: [0; 3],
                cells: Vec::new(),
            };
        }

        let cell_size = max_effective_radius.max(0.1);
        let inv_cell_size = 1.0 / cell_size;

        // Compute bounding box of atoms expanded by max_effective_radius
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        for c in coords {
            for d in 0..3 {
                if c[d] < min[d] { min[d] = c[d]; }
                if c[d] > max[d] { max[d] = c[d]; }
            }
        }

        let origin = [
            min[0] - max_effective_radius,
            min[1] - max_effective_radius,
            min[2] - max_effective_radius,
        ];
        let grid_dims = [
            ((max[0] - origin[0]) * inv_cell_size).ceil() as usize + 1,
            ((max[1] - origin[1]) * inv_cell_size).ceil() as usize + 1,
            ((max[2] - origin[2]) * inv_cell_size).ceil() as usize + 1,
        ];
        let n_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
        let mut cells = vec![Vec::new(); n_cells];

        for (i, c) in coords.iter().enumerate() {
            let cx = ((c[0] - origin[0]) * inv_cell_size) as usize;
            let cy = ((c[1] - origin[1]) * inv_cell_size) as usize;
            let cz = ((c[2] - origin[2]) * inv_cell_size) as usize;
            let cell_idx = cx + grid_dims[0] * (cy + grid_dims[1] * cz);
            cells[cell_idx].push(i);
        }

        Self {
            cell_size,
            inv_cell_size,
            origin,
            grid_dims,
            cells,
        }
    }

    /// Test whether a point is inside any atom's inflated sphere.
    ///
    /// Only checks atoms in the local 3x3x3 cell neighborhood.
    #[inline]
    fn point_inside(
        &self,
        point: &[f64; 3],
        coords: &[[f64; 3]],
        radii: &[f64],
        probe_radius: f64,
    ) -> bool {
        if self.cells.is_empty() {
            return false;
        }

        let fx = (point[0] - self.origin[0]) * self.inv_cell_size;
        let fy = (point[1] - self.origin[1]) * self.inv_cell_size;
        let fz = (point[2] - self.origin[2]) * self.inv_cell_size;

        let cx = fx as isize;
        let cy = fy as isize;
        let cz = fz as isize;

        let gnx = self.grid_dims[0] as isize;
        let gny = self.grid_dims[1] as isize;
        let gnz = self.grid_dims[2] as isize;

        for dz in -1..=1 {
            let jz = cz + dz;
            if jz < 0 || jz >= gnz { continue; }
            for dy in -1..=1 {
                let jy = cy + dy;
                if jy < 0 || jy >= gny { continue; }
                for dx in -1..=1 {
                    let jx = cx + dx;
                    if jx < 0 || jx >= gnx { continue; }

                    let cell_idx = jx as usize
                        + self.grid_dims[0] * (jy as usize + self.grid_dims[1] * jz as usize);
                    for &atom_i in &self.cells[cell_idx] {
                        let c = &coords[atom_i];
                        let ddx = point[0] - c[0];
                        let ddy = point[1] - c[1];
                        let ddz = point[2] - c[2];
                        let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                        let threshold = radii[atom_i] + probe_radius;
                        if dist_sq < threshold * threshold {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}


/// 1D squared Euclidean distance transform (Felzenszwalb & Huttenlocher).
///
/// Transforms `f` in-place: on input, `f[i] = 0` for set members and `f64::MAX/2`
/// for non-members. On output, `f[i]` is the squared distance (in grid-index
/// units) to the nearest set member along this 1D slice.
fn edt_1d(f: &mut [f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        // Single element: distance is 0 if set member, stays unchanged otherwise
        return;
    }

    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;

    for q in 1..n {
        loop {
            let vk = v[k];
            let s = (f[q] + (q * q) as f64 - f[vk] - (vk * vk) as f64)
                / (2 * (q - vk)) as f64;
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
            if k == 0 {
                // Replace the only parabola
                v[0] = q;
                z[0] = f64::NEG_INFINITY;
                z[1] = f64::INFINITY;
                break;
            }
            k -= 1;
        }
    }

    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let diff = q as f64 - v[k] as f64;
        d[q] = diff * diff + f[v[k]];
    }

    f.copy_from_slice(&d[..n]);
}

/// 3D squared Euclidean distance transform via separable 1D passes.
///
/// On input, `dist[idx] = 0.0` for "seed" points (connected exterior) and
/// `f64::MAX/2` for others. On output, `dist[idx]` is the squared Euclidean
/// distance in grid-index units to the nearest seed point.
///
/// The distance in Angstroms² is `dist[idx] * h²` (for uniform spacing `h`).
fn edt_3d(dist: &mut [f64], dims: [usize; 3]) {
    let nx = dims[0];
    let ny = dims[1];
    let nz = dims[2];
    let max_dim = nx.max(ny).max(nz);

    // Scratch buffers for edt_1d (reused across all 1D slices)
    let mut d_buf = vec![0.0f64; max_dim];
    let mut v_buf = vec![0usize; max_dim];
    let mut z_buf = vec![0.0f64; max_dim + 1];

    // Pass 1: along x (contiguous in memory)
    for iz in 0..nz {
        for iy in 0..ny {
            let start = iy * nx + iz * nx * ny;
            let slice = &mut dist[start..start + nx];
            edt_1d(slice, &mut d_buf[..nx], &mut v_buf[..nx], &mut z_buf[..nx + 1]);
        }
    }

    // Pass 2: along y (strided, need to gather/scatter)
    let mut col = vec![0.0f64; ny];
    for iz in 0..nz {
        for ix in 0..nx {
            // Gather y-column
            for iy in 0..ny {
                col[iy] = dist[ix + nx * (iy + ny * iz)];
            }
            edt_1d(&mut col, &mut d_buf[..ny], &mut v_buf[..ny], &mut z_buf[..ny + 1]);
            // Scatter back
            for iy in 0..ny {
                dist[ix + nx * (iy + ny * iz)] = col[iy];
            }
        }
    }

    // Pass 3: along z (strided)
    let mut col_z = vec![0.0f64; nz];
    for iy in 0..ny {
        for ix in 0..nx {
            let stride = nx * ny;
            let base = ix + nx * iy;
            // Gather z-column
            for iz in 0..nz {
                col_z[iz] = dist[base + stride * iz];
            }
            edt_1d(&mut col_z, &mut d_buf[..nz], &mut v_buf[..nz], &mut z_buf[..nz + 1]);
            // Scatter back
            for iz in 0..nz {
                dist[base + stride * iz] = col_z[iz];
            }
        }
    }
}

/// Compute Solvent-Excluded Surface (SES) inside/outside classification.
///
/// The SES (molecular surface) is the boundary between regions accessible
/// to a probe sphere of the given radius and regions that are not. This is
/// tighter than the SAS (Solvent-Accessible Surface) because crevices and
/// grooves between atoms are classified as interior even though they are
/// outside the union of inflated spheres.
///
/// Algorithm:
/// 1. Identify valid probe-center positions: grid points outside the SAS
///    (where a probe sphere wouldn't overlap any atom) connected to the
///    bulk (grid boundary) via a path of such positions.
/// 2. A grid point is outside the SES if any valid probe center within
///    `probe_radius` can reach it; otherwise it is inside.
fn compute_inside_ses(
    grid: &PbGrid,
    coords: &[[f64; 3]],
    radii: &[f64],
    probe_radius: f64,
) -> Vec<bool> {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];
    let n = nx * ny * nz;

    // Build spatial hash with max effective radius covering both VDW and SAS queries
    let max_r = radii.iter().cloned().fold(0.0f64, f64::max);
    let max_effective_r = max_r + probe_radius.max(0.0);
    let spatial_hash = AtomSpatialHash::new(coords, max_effective_r);

    if probe_radius <= 0.0 {
        // No probe: SES = VDW surface
        return (0..n)
            .into_par_iter()
            .map(|idx| {
                let iz = idx / (nx * ny);
                let iy = (idx % (nx * ny)) / nx;
                let ix = idx % nx;
                let pt = grid.point(ix, iy, iz);
                spatial_hash.point_inside(&pt, coords, radii, 0.0)
            })
            .collect();
    }

    // Step 1: Compute inside_sas (SAS = union of spheres with r + probe).
    // Outside SAS = valid probe-center positions (probe fits without overlap).
    let inside_sas: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let iz = idx / (nx * ny);
            let iy = (idx % (nx * ny)) / nx;
            let ix = idx % nx;
            let pt = grid.point(ix, iy, iz);
            spatial_hash.point_inside(&pt, coords, radii, probe_radius)
        })
        .collect();

    // Step 2: Flood-fill from boundary through outside_sas to find the
    // connected exterior (probe centers reachable from the bulk).
    let mut connected_exterior = vec![false; n];
    let mut queue = VecDeque::new();

    // Seed: boundary points that are outside SAS
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let is_boundary =
                    ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1 || iz == 0 || iz == nz - 1;
                if is_boundary {
                    let idx = grid.index(ix, iy, iz);
                    if !inside_sas[idx] {
                        connected_exterior[idx] = true;
                        queue.push_back(idx);
                    }
                }
            }
        }
    }

    // BFS through outside_sas
    while let Some(idx) = queue.pop_front() {
        let iz = idx / (nx * ny);
        let iy = (idx % (nx * ny)) / nx;
        let ix = idx % nx;

        let neighbors: [Option<usize>; 6] = [
            if ix > 0 {
                Some(grid.index(ix - 1, iy, iz))
            } else {
                None
            },
            if ix + 1 < nx {
                Some(grid.index(ix + 1, iy, iz))
            } else {
                None
            },
            if iy > 0 {
                Some(grid.index(ix, iy - 1, iz))
            } else {
                None
            },
            if iy + 1 < ny {
                Some(grid.index(ix, iy + 1, iz))
            } else {
                None
            },
            if iz > 0 {
                Some(grid.index(ix, iy, iz - 1))
            } else {
                None
            },
            if iz + 1 < nz {
                Some(grid.index(ix, iy, iz + 1))
            } else {
                None
            },
        ];

        for ni in neighbors.into_iter().flatten() {
            if !connected_exterior[ni] && !inside_sas[ni] {
                connected_exterior[ni] = true;
                queue.push_back(ni);
            }
        }
    }

    // Step 3: Compute squared distance from each grid point to the nearest
    // connected exterior point using an O(N) Euclidean distance transform,
    // replacing the O(N*R^3) brute-force dilation.
    let h = grid.spacing[0]; // uniform spacing
    // Threshold in grid-index units: point is outside SES if distance <= probe_radius/h
    let threshold_sq = (probe_radius / h) * (probe_radius / h);
    let big = (n as f64) * 2.0; // large sentinel (greater than any real squared distance)

    let mut dist = vec![0.0f64; n];
    for i in 0..n {
        dist[i] = if connected_exterior[i] { 0.0 } else { big };
    }

    edt_3d(&mut dist, [nx, ny, nz]);

    // A point is inside SES if no connected exterior point is within probe_radius
    (0..n)
        .into_par_iter()
        .map(|idx| dist[idx] > threshold_sq)
        .collect()
}

/// Compute the fraction of an axis-aligned edge segment that lies inside the
/// molecular surface (union of atom spheres inflated by probe radius).
///
/// The edge runs from `lo` to `hi` along `axis` (0=x, 1=y, 2=z), at fixed
/// perpendicular coordinates `perp1` and `perp2`.
///
/// Returns a value in [0, 1].
fn fraction_inside_along_axis(
    lo: f64,
    hi: f64,
    perp1: f64,
    perp2: f64,
    axis: usize,
    coords: &[[f64; 3]],
    radii: &[f64],
    probe_radius: f64,
) -> f64 {
    let edge_len = hi - lo;
    if edge_len <= 0.0 {
        return 0.0;
    }

    // Perpendicular axis indices
    let (a1, a2) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };

    // Collect intervals along the edge that lie inside each atom's sphere.
    // Use a stack buffer for the common case (few overlapping atoms per edge)
    // to avoid heap allocation. Falls back to Vec if the buffer overflows.
    const STACK_CAP: usize = 16;
    let mut stack_buf: [(f64, f64); STACK_CAP] = [(0.0, 0.0); STACK_CAP];
    let mut stack_len = 0usize;
    let mut heap_buf: Vec<(f64, f64)> = Vec::new();

    for (c, &r) in coords.iter().zip(radii.iter()) {
        let big_r = r + probe_radius;
        let d1 = perp1 - c[a1];
        let d2 = perp2 - c[a2];
        let rho_sq = big_r * big_r - d1 * d1 - d2 * d2;
        if rho_sq <= 0.0 {
            continue;
        }
        let rho = rho_sq.sqrt();
        let ilo = (c[axis] - rho).max(lo);
        let ihi = (c[axis] + rho).min(hi);
        if ilo < ihi {
            if stack_len < STACK_CAP {
                stack_buf[stack_len] = (ilo, ihi);
                stack_len += 1;
            } else {
                // Spill to heap on first overflow
                if heap_buf.is_empty() {
                    heap_buf.extend_from_slice(&stack_buf[..stack_len]);
                }
                heap_buf.push((ilo, ihi));
            }
        }
    }

    let intervals: &mut [(f64, f64)] = if heap_buf.is_empty() {
        &mut stack_buf[..stack_len]
    } else {
        &mut heap_buf[..]
    };

    if intervals.is_empty() {
        return 0.0;
    }

    // Sort by interval start
    intervals.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Merge overlapping intervals and sum total inside length
    let mut total = 0.0;
    let mut cur_lo = intervals[0].0;
    let mut cur_hi = intervals[0].1;
    for &(ilo, ihi) in &intervals[1..] {
        if ilo <= cur_hi {
            cur_hi = cur_hi.max(ihi);
        } else {
            total += cur_hi - cur_lo;
            cur_lo = ilo;
            cur_hi = ihi;
        }
    }
    total += cur_hi - cur_lo;

    (total / edge_len).clamp(0.0, 1.0)
}

/// Weighted harmonic mean of dielectrics.
///
/// `1/ε = f/ε_in + (1-f)/ε_out` where `f` is the fraction inside.
/// This is the standard approach used by AMBER and other PB solvers
/// for face-centered dielectrics at boundary edges.
#[inline]
fn weighted_harmonic_eps(f: f64, eps_in: f64, eps_out: f64) -> f64 {
    if f <= 0.0 {
        return eps_out;
    }
    if f >= 1.0 {
        return eps_in;
    }
    1.0 / (f / eps_in + (1.0 - f) / eps_out)
}

/// Assign face-centered dielectric values based on the Solvent-Excluded Surface.
///
/// Uses the SES (molecular surface) to classify grid points as inside or
/// outside, then assigns edge dielectrics:
/// - Both endpoints inside → ε_in
/// - Both endpoints outside → ε_out
/// - Mixed → weighted harmonic mean using the fraction of the edge inside
///   the molecular surface (computed analytically from sphere geometry).
pub fn assign_dielectrics(
    grid: &PbGrid,
    coords: &[[f64; 3]],
    radii: &[f64],
    probe_radius: f64,
    eps_in: f64,
    eps_out: f64,
) -> DielectricMaps {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];
    let n = nx * ny * nz;

    // Short-circuit: when eps_in == eps_out (e.g. reference solve), the
    // dielectric is uniform everywhere. Skip the expensive SES computation.
    if (eps_in - eps_out).abs() < f64::EPSILON {
        return DielectricMaps {
            eps_x: vec![eps_in; n],
            eps_y: vec![eps_in; n],
            eps_z: vec![eps_in; n],
            dims: grid.dims,
        };
    }

    // Use the Solvent-Excluded Surface for inside/outside classification.
    // This handles re-entrant regions correctly, unlike the simpler SAS
    // (union of inflated spheres) which over-estimates the cavity size.
    let inside = compute_inside_ses(grid, coords, radii, probe_radius);

    let mut eps_x = vec![eps_out; n];
    let mut eps_y = vec![eps_out; n];
    let mut eps_z = vec![eps_out; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = grid.index(ix, iy, iz);
                let a = inside[idx];

                // X-edge: (ix,iy,iz) → (ix+1,iy,iz)
                if ix + 1 < nx {
                    let b = inside[grid.index(ix + 1, iy, iz)];
                    if a == b {
                        eps_x[idx] = if a { eps_in } else { eps_out };
                    } else {
                        // Use VDW sphere geometry (probe=0) for the fraction
                        // since the SES contact surface is at the VDW boundary.
                        let pt = grid.point(ix, iy, iz);
                        let f = fraction_inside_along_axis(
                            pt[0],
                            pt[0] + grid.spacing[0],
                            pt[1],
                            pt[2],
                            0,
                            coords,
                            radii,
                            0.0, // VDW radii, no probe inflation
                        );
                        eps_x[idx] = weighted_harmonic_eps(f, eps_in, eps_out);
                    }
                }

                // Y-edge: (ix,iy,iz) → (ix,iy+1,iz)
                if iy + 1 < ny {
                    let b = inside[grid.index(ix, iy + 1, iz)];
                    if a == b {
                        eps_y[idx] = if a { eps_in } else { eps_out };
                    } else {
                        let pt = grid.point(ix, iy, iz);
                        let f = fraction_inside_along_axis(
                            pt[1],
                            pt[1] + grid.spacing[1],
                            pt[0],
                            pt[2],
                            1,
                            coords,
                            radii,
                            0.0,
                        );
                        eps_y[idx] = weighted_harmonic_eps(f, eps_in, eps_out);
                    }
                }

                // Z-edge: (ix,iy,iz) → (ix,iy,iz+1)
                if iz + 1 < nz {
                    let b = inside[grid.index(ix, iy, iz + 1)];
                    if a == b {
                        eps_z[idx] = if a { eps_in } else { eps_out };
                    } else {
                        let pt = grid.point(ix, iy, iz);
                        let f = fraction_inside_along_axis(
                            pt[2],
                            pt[2] + grid.spacing[2],
                            pt[0],
                            pt[1],
                            2,
                            coords,
                            radii,
                            0.0,
                        );
                        eps_z[idx] = weighted_harmonic_eps(f, eps_in, eps_out);
                    }
                }
            }
        }
    }

    DielectricMaps {
        eps_x,
        eps_y,
        eps_z,
        dims: grid.dims,
    }
}

/// Assign κ̄² map: zero inside ion-exclusion surface, ε_s·κ² outside.
///
/// The linearized PBE is: ∇·[ε∇φ] - κ̄²φ = -4π·ec·ρ
/// where κ̄² = ε_s · κ² (the Debye-Hückel κ² multiplied by the solvent
/// dielectric). The factor of ε_s arises because the mobile ion charge
/// density is -2c₀·φ/(kBT) (no ε_s), while κ² = 8π·ec·c₀·N_A/(ε_s·kBT)
/// already has 1/ε_s from the Debye-Hückel derivation.
///
/// The ion-exclusion surface is the molecular surface inflated by the ion radius.
pub fn assign_kappa(
    grid: &PbGrid,
    coords: &[[f64; 3]],
    radii: &[f64],
    ion_radius: f64,
    kappa_bulk: f64,
    solvent_dielectric: f64,
) -> Vec<f64> {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];
    let n = nx * ny * nz;
    // κ̄² = ε_s · κ² is the coefficient in the linearized PBE
    let kappa_bar_sq = solvent_dielectric * kappa_bulk * kappa_bulk;

    // Spatial hash for O(1) amortized lookups instead of O(N_atoms)
    let max_r = radii.iter().cloned().fold(0.0f64, f64::max) + ion_radius;
    let spatial_hash = AtomSpatialHash::new(coords, max_r);

    (0..n)
        .into_par_iter()
        .map(|idx| {
            let iz = idx / (nx * ny);
            let iy = (idx % (nx * ny)) / nx;
            let ix = idx % nx;
            let pt = grid.point(ix, iy, iz);
            // Ion exclusion: atom radius + ion_radius
            if spatial_hash.point_inside(&pt, coords, radii, ion_radius) {
                0.0
            } else {
                kappa_bar_sq
            }
        })
        .collect()
}

/// Extract charges (in elementary charge units) from an AMBER topology.
///
/// AMBER stores charges as q * 18.2223; we divide to get elementary charges.
pub fn topology_charges(topology: &AmberTopology) -> Vec<f64> {
    topology.charges.clone()
}

/// Extract atomic radii from an AMBER topology.
pub fn topology_radii(topology: &AmberTopology) -> Vec<f64> {
    topology.radii.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_grid_odd_dims() {
        let coords = vec![[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]];
        let grid = auto_grid(&coords, 0.5, 5.0);
        assert!(grid.dims[0] % 2 == 1);
        assert!(grid.dims[1] % 2 == 1);
        assert!(grid.dims[2] % 2 == 1);
    }

    #[test]
    fn test_charge_conservation() {
        let coords = vec![[5.0, 5.0, 5.0]];
        let grid = auto_grid(&coords, 1.0, 5.0);
        let charges = vec![1.0];
        let charge_map = map_charges(&grid, &coords, &charges);
        let total: f64 =
            charge_map.iter().sum::<f64>() * grid.spacing[0] * grid.spacing[1] * grid.spacing[2];
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Charge not conserved: total = {}",
            total
        );
    }

    #[test]
    fn test_interpolation_at_grid_point() {
        let grid = PbGrid {
            dims: [3, 3, 3],
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            data: vec![0.0; 27],
        };
        // Set center point
        let mut g = grid;
        let idx = g.index(1, 1, 1);
        g.data[idx] = 1.0;
        let val = g.interpolate(&[1.0, 1.0, 1.0]);
        assert!((val - 1.0).abs() < 1e-10);
    }
}
