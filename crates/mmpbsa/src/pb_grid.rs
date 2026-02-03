//! Grid infrastructure for finite-difference Poisson-Boltzmann solver.
//!
//! Provides a 3D grid, charge mapping via trilinear interpolation,
//! dielectric map assignment using molecular surface tests, and
//! ionic strength (kappa²) map assignment.

use rayon::prelude::*;
use rst_core::amber::prmtop::AmberTopology;

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
        dims[d] = if n % 2 == 0 { n + 1 } else { n };
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

/// Test whether a point is inside the molecular surface (within any atom's radius).
fn point_inside_molecule(
    point: &[f64; 3],
    coords: &[[f64; 3]],
    radii: &[f64],
    probe_radius: f64,
) -> bool {
    for (c, &r) in coords.iter().zip(radii.iter()) {
        let dx = point[0] - c[0];
        let dy = point[1] - c[1];
        let dz = point[2] - c[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let threshold = r + probe_radius;
        if dist_sq < threshold * threshold {
            return true;
        }
    }
    false
}

/// Assign face-centered dielectric values based on molecular surface.
///
/// Uses a simple approach: for each grid edge, test both endpoints.
/// If both are inside the molecule → eps_in, both outside → eps_out,
/// mixed → harmonic mean.
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

    // First, compute per-node inside/outside flag
    let inside: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let iz = idx / (nx * ny);
            let iy = (idx % (nx * ny)) / nx;
            let ix = idx % nx;
            let pt = grid.point(ix, iy, iz);
            point_inside_molecule(&pt, coords, radii, probe_radius)
        })
        .collect();

    let eps_for = |a: bool, b: bool| -> f64 {
        match (a, b) {
            (true, true) => eps_in,
            (false, false) => eps_out,
            _ => 2.0 * eps_in * eps_out / (eps_in + eps_out), // harmonic mean
        }
    };

    let mut eps_x = vec![eps_out; n];
    let mut eps_y = vec![eps_out; n];
    let mut eps_z = vec![eps_out; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = grid.index(ix, iy, iz);
                let a = inside[idx];
                if ix + 1 < nx {
                    eps_x[idx] = eps_for(a, inside[grid.index(ix + 1, iy, iz)]);
                }
                if iy + 1 < ny {
                    eps_y[idx] = eps_for(a, inside[grid.index(ix, iy + 1, iz)]);
                }
                if iz + 1 < nz {
                    eps_z[idx] = eps_for(a, inside[grid.index(ix, iy, iz + 1)]);
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

/// Assign κ² map: zero inside ion-exclusion surface, κ²_bulk outside.
///
/// The ion-exclusion surface is the molecular surface inflated by the ion radius.
pub fn assign_kappa(
    grid: &PbGrid,
    coords: &[[f64; 3]],
    radii: &[f64],
    ion_radius: f64,
    kappa_bulk: f64,
) -> Vec<f64> {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];
    let n = nx * ny * nz;
    let kappa_sq = kappa_bulk * kappa_bulk;

    (0..n)
        .into_par_iter()
        .map(|idx| {
            let iz = idx / (nx * ny);
            let iy = (idx % (nx * ny)) / nx;
            let ix = idx % nx;
            let pt = grid.point(ix, iy, iz);
            // Ion exclusion: atom radius + ion_radius
            if point_inside_molecule(&pt, coords, radii, ion_radius) {
                0.0
            } else {
                kappa_sq
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
