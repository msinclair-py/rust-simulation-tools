//! Successive Over-Relaxation (SOR) solver for the linearized Poisson-Boltzmann equation,
//! with multigrid V-cycle acceleration.
//!
//! Solves: nabla.[eps(r) nabla phi(r)] - kappa^2(r) phi(r) = -4 pi rho(r)
//! using red-black SOR on a regular 3D grid, optionally accelerated by
//! a multigrid V-cycle that restricts residuals to coarser grids and
//! prolongates corrections back.

use crate::pb_grid::{DielectricMaps, PbGrid};
use rayon::prelude::*;

/// Boundary condition for the PB solve.
#[derive(Debug, Clone)]
pub enum BoundaryCondition {
    /// phi = 0 on all boundaries.
    Zero,
    /// Debye-Huckel single-sphere approximation on boundaries.
    DebyeHuckel,
    /// Interpolated from a coarse grid solution (for focusing).
    /// Stores only boundary point (grid_index, value) pairs instead of a
    /// full-grid Vec, saving ~30 MB for a typical 161³ grid.
    Interpolated(Vec<(usize, f64)>),
}

/// Iterate over all boundary points of a 3D grid, calling `f(ix, iy, iz)` for each.
///
/// Directly iterates the 6 faces (two z-slabs, two y-slabs, two x-slabs)
/// instead of a full 3D loop with a `continue` check, reducing overhead
/// for large grids where the boundary is O(N^2) but the interior is O(N^3).
#[inline]
fn for_each_boundary_point(dims: &[usize; 3], mut f: impl FnMut(usize, usize, usize)) {
    let [nx, ny, nz] = *dims;

    // z = 0 and z = nz-1 faces (full xy slabs)
    for &iz in &[0, nz - 1] {
        for iy in 0..ny {
            for ix in 0..nx {
                f(ix, iy, iz);
            }
        }
    }

    // y = 0 and y = ny-1 faces (excluding z corners already covered)
    for iz in 1..nz - 1 {
        for &iy in &[0, ny - 1] {
            for ix in 0..nx {
                f(ix, iy, iz);
            }
        }
    }

    // x = 0 and x = nx-1 edges (excluding z and y corners already covered)
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for &ix in &[0, nx - 1] {
                f(ix, iy, iz);
            }
        }
    }
}

/// Set Debye-Huckel boundary conditions on the grid.
///
/// For a collection of charges, the DH potential at a boundary point is:
///   phi(r) = sum_i q_i * exp(-kappa|r-r_i|) / (eps_out * |r-r_i|)
///
/// where the sum is over all atoms, with conversion factor 332.0522 for kcal*A/(mol*e^2).
fn set_dh_boundary(
    potential: &mut [f64],
    grid: &PbGrid,
    coords: &[[f64; 3]],
    charges: &[f64],
    kappa: f64,
    eps_out: f64,
) {
    let conversion = 332.0522; // kcal*A/(mol*e^2)

    for_each_boundary_point(&grid.dims, |ix, iy, iz| {
        let pt = grid.point(ix, iy, iz);
        let mut phi = 0.0;
        for (c, &q) in coords.iter().zip(charges.iter()) {
            let dx = pt[0] - c[0];
            let dy = pt[1] - c[1];
            let dz = pt[2] - c[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r > 1e-10 {
                phi += conversion * q * (-kappa * r).exp() / (eps_out * r);
            }
        }
        let idx = grid.index(ix, iy, iz);
        potential[idx] = phi;
    });
}

/// Compute interpolated boundary conditions from a coarse grid solution.
///
/// For each boundary point of `fine_grid`, looks up the corresponding
/// position in the coarse solution via trilinear interpolation.
/// Returns a sparse list of (grid_index, value) pairs for boundary points only.
pub fn interpolated_boundary(
    fine_grid: &PbGrid,
    coarse_grid: &PbGrid,
    coarse_potential: &[f64],
) -> Vec<(usize, f64)> {
    let mut boundary = Vec::new();

    for_each_boundary_point(&fine_grid.dims, |ix, iy, iz| {
        let pt = fine_grid.point(ix, iy, iz);
        let idx = fine_grid.index(ix, iy, iz);
        let val = coarse_grid.interpolate_with_data(&pt, coarse_potential);
        boundary.push((idx, val));
    });
    boundary
}

/// Set boundary conditions from precomputed sparse interpolated values.
fn set_interpolated_boundary(potential: &mut [f64], boundary: &[(usize, f64)]) {
    for &(idx, val) in boundary {
        potential[idx] = val;
    }
}

/// Estimate the spectral radius of the Jacobi iteration for a 3D Laplacian.
fn spectral_radius(dims: &[usize; 3]) -> f64 {
    let nx = dims[0] as f64;
    let ny = dims[1] as f64;
    let nz = dims[2] as f64;
    ((std::f64::consts::PI / nx).cos()
        + (std::f64::consts::PI / ny).cos()
        + (std::f64::consts::PI / nz).cos())
        / 3.0
}

/// Estimate the optimal SOR relaxation parameter from grid dimensions.
fn estimate_omega(dims: &[usize; 3]) -> f64 {
    let rho = spectral_radius(dims);
    // Optimal SOR omega
    2.0 / (1.0 + (1.0 - rho * rho).sqrt())
}

/// Result of the PB solve.
#[derive(Debug, Clone)]
pub struct PbSolveResult {
    /// Electrostatic potential at each grid point (kcal/mol/e).
    pub potential: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final RMS residual.
    pub final_residual: f64,
    /// Whether the solve converged.
    pub converged: bool,
}

// 4pi * 332.0522 converts from e/A^3 charge density to kcal/mol/e potential units
const FOUR_PI_EC: f64 = 4.0 * std::f64::consts::PI * 332.0522;

/// Compute the neighbor sum and diagonal coefficient at a grid point.
///
/// This is the core finite-difference stencil for the linearized PBE,
/// shared between SOR smoothing and residual computation.
#[inline(always)]
fn stencil_at(
    idx: usize,
    idx_xm: usize,
    idx_xp: usize,
    idx_ym: usize,
    idx_yp: usize,
    idx_zm: usize,
    idx_zp: usize,
    potential: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    inv_hx2: f64,
    inv_hy2: f64,
    inv_hz2: f64,
) -> (f64, f64) {
    let ex_p = dielectrics.eps_x[idx];
    let ex_m = dielectrics.eps_x[idx_xm];
    let ey_p = dielectrics.eps_y[idx];
    let ey_m = dielectrics.eps_y[idx_ym];
    let ez_p = dielectrics.eps_z[idx];
    let ez_m = dielectrics.eps_z[idx_zm];

    let neighbor_sum = (ex_p * potential[idx_xp] + ex_m * potential[idx_xm]) * inv_hx2
        + (ey_p * potential[idx_yp] + ey_m * potential[idx_ym]) * inv_hy2
        + (ez_p * potential[idx_zp] + ez_m * potential[idx_zm]) * inv_hz2;

    let diag = (ex_p + ex_m) * inv_hx2
        + (ey_p + ey_m) * inv_hy2
        + (ez_p + ez_m) * inv_hz2
        + kappa_sq[idx];

    (neighbor_sum, diag)
}

/// Tile size for cache-friendly traversal. An 8x8x8 tile touching 7 arrays
/// uses 8*8*8*8*7 ≈ 28 KB, fitting comfortably in L1 cache (32-48 KB).
/// The previous 16^3 tile (229 KB working set) exceeded L1.
const TILE: usize = 8;

/// Wrapper for safe parallel access to the potential array.
/// Safety: red-black ordering guarantees that within a single color sweep,
/// no two threads write to adjacent indices.
struct UnsafePotential(*mut f64);
unsafe impl Send for UnsafePotential {}
unsafe impl Sync for UnsafePotential {}

/// Run a fixed number of SOR smoothing sweeps on the given potential.
/// Returns the RMS residual after the last sweep.
///
/// `rhs_scale` is multiplied by `charge_map` to form the RHS of the linear system.
/// For the top-level PBE solve this is `FOUR_PI_EC`; for coarse-grid error
/// equations where `charge_map` already contains the restricted residual, use `1.0`.
fn sor_smooth(
    grid: &PbGrid,
    potential: &mut [f64],
    charge_map: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    omega: f64,
    n_sweeps: usize,
    rhs_scale: f64,
) -> f64 {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];
    let n = nx * ny * nz;

    // Precompute strides for direct index arithmetic
    let stride_y = nx;
    let stride_z = nx * ny;

    // Precompute reciprocals to replace divisions with multiplications
    let inv_hx2 = 1.0 / (grid.spacing[0] * grid.spacing[0]);
    let inv_hy2 = 1.0 / (grid.spacing[1] * grid.spacing[1]);
    let inv_hz2 = 1.0 / (grid.spacing[2] * grid.spacing[2]);

    // Collect tile starts once, reused across all sweeps and colors
    let z_tiles: Vec<usize> = (1..nz - 1).step_by(TILE).collect();

    let mut last_rms = 0.0;

    for _sweep in 0..n_sweeps {
        let mut rms_residual = 0.0;
        let mut count = 0u64;

        let ptr = UnsafePotential(potential.as_mut_ptr());

        // Red-black SOR: two half-sweeps
        for color in 0..2usize {
            let (color_rms, color_count): (f64, u64) = z_tiles
                .par_iter()
                .map({
                    let ptr = &ptr;
                    move |&tz| {
                        // Safety: each thread only reads neighbors (opposite color,
                        // not being written) and writes to its own color points.
                        let potential =
                            unsafe { std::slice::from_raw_parts_mut(ptr.0, n) };

                        let mut local_rms = 0.0;
                        let mut local_count = 0u64;
                        let iz_end = (tz + TILE).min(nz - 1);
                        for ty in (1..ny - 1).step_by(TILE) {
                            let iy_end = (ty + TILE).min(ny - 1);
                            for tx in (1..nx - 1).step_by(TILE) {
                                let ix_end = (tx + TILE).min(nx - 1);
                                for iz in tz..iz_end {
                                    for iy in ty..iy_end {
                                        // Step by 2 with correct parity offset,
                                        // eliminating the color-check branch entirely
                                        let ix_start = tx + ((tx + iy + iz + color) % 2);
                                        for ix in (ix_start..ix_end).step_by(2) {
                                            let idx = ix + stride_y * iy + stride_z * iz;

                                            let (neighbor_sum, diag) = stencil_at(
                                                idx,
                                                idx - 1,
                                                idx + 1,
                                                idx - stride_y,
                                                idx + stride_y,
                                                idx - stride_z,
                                                idx + stride_z,
                                                potential,
                                                dielectrics,
                                                kappa_sq,
                                                inv_hx2,
                                                inv_hy2,
                                                inv_hz2,
                                            );

                                            let rhs = rhs_scale * charge_map[idx];
                                            let phi_gs = (neighbor_sum + rhs) / diag;
                                            let phi_new =
                                                potential[idx] + omega * (phi_gs - potential[idx]);

                                            let residual = phi_gs - potential[idx];
                                            local_rms += residual * residual;
                                            local_count += 1;

                                            potential[idx] = phi_new;
                                        }
                                    }
                                }
                            }
                        }
                        (local_rms, local_count)
                    }
                })
                .reduce(
                    || (0.0, 0),
                    |(a_rms, a_cnt), (b_rms, b_cnt)| (a_rms + b_rms, a_cnt + b_cnt),
                );

            rms_residual += color_rms;
            count += color_count;
        }

        last_rms = if count > 0 {
            (rms_residual / count as f64).sqrt()
        } else {
            0.0
        };
    }

    last_rms
}

/// Solve the linearized Poisson-Boltzmann equation using SOR.
///
/// The discretized equation at interior point (i,j,k) is:
///
///   eps_x[i,j,k]*(phi[i+1]-phi[i]) - eps_x[i-1,j,k]*(phi[i]-phi[i-1])  (x-direction)
/// + eps_y[i,j,k]*(phi[j+1]-phi[j]) - eps_y[i,j-1,k]*(phi[j]-phi[j-1])  (y-direction)
/// + eps_z[i,j,k]*(phi[k+1]-phi[k]) - eps_z[i,j,k-1]*(phi[k]-phi[k-1])  (z-direction)
/// - kappa^2[i,j,k]*h^2*phi[i,j,k] = -4*pi*rho[i,j,k]*h^2
///
/// For non-uniform spacing, divide by h^2 in each direction appropriately.
pub fn solve_lpbe(
    grid: &PbGrid,
    charge_map: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    boundary: BoundaryCondition,
    coords: &[[f64; 3]],
    charges: &[f64],
    kappa_bulk: f64,
    eps_out: f64,
    tolerance: f64,
    max_iterations: usize,
) -> PbSolveResult {
    solve_lpbe_initial(
        grid,
        charge_map,
        dielectrics,
        kappa_sq,
        boundary,
        coords,
        charges,
        kappa_bulk,
        eps_out,
        tolerance,
        max_iterations,
        None,
    )
}

/// Solve the linearized PBE using SOR, optionally starting from an initial potential.
///
/// When `initial_potential` is `Some(...)`, uses it as the starting point
/// (boundary conditions are still applied on top). This avoids discarding
/// work from a partially-converged multigrid solve.
fn solve_lpbe_initial(
    grid: &PbGrid,
    charge_map: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    boundary: BoundaryCondition,
    coords: &[[f64; 3]],
    charges: &[f64],
    kappa_bulk: f64,
    eps_out: f64,
    tolerance: f64,
    max_iterations: usize,
    initial_potential: Option<Vec<f64>>,
) -> PbSolveResult {
    let n = grid.len();
    let mut potential = initial_potential.unwrap_or_else(|| vec![0.0f64; n]);

    // Set boundary conditions (re-applied even with initial potential)
    match &boundary {
        BoundaryCondition::Zero => {}
        BoundaryCondition::DebyeHuckel => {
            set_dh_boundary(&mut potential, grid, coords, charges, kappa_bulk, eps_out);
        }
        BoundaryCondition::Interpolated(bvals) => {
            set_interpolated_boundary(&mut potential, bvals);
        }
    }

    // Use Chebyshev-accelerated SOR: omega ramps from 1.0 toward the
    // optimal value, providing faster convergence in the early iterations.
    let rho_j = spectral_radius(&grid.dims);
    let rho_sq = rho_j * rho_j;

    let mut iterations = 0;
    let mut final_residual = f64::MAX;
    let mut cheb_omega = 1.0;

    for iter in 0..max_iterations {
        final_residual = sor_smooth(
            grid,
            &mut potential,
            charge_map,
            dielectrics,
            kappa_sq,
            cheb_omega,
            1,
            FOUR_PI_EC,
        );
        iterations = iter + 1;

        // Advance Chebyshev omega using the recurrence relation:
        // omega_0 = 1, omega_1 = 1/(1-rho^2/2), omega_n = 1/(1-rho^2*omega_{n-1}/4)
        if iter == 0 {
            cheb_omega = 1.0 / (1.0 - rho_sq / 2.0);
        } else {
            cheb_omega = 1.0 / (1.0 - rho_sq * cheb_omega / 4.0);
        }

        if final_residual < tolerance {
            return PbSolveResult {
                potential,
                iterations,
                final_residual,
                converged: true,
            };
        }
    }

    PbSolveResult {
        potential,
        iterations,
        final_residual,
        converged: false,
    }
}

/// Compute residual into a pre-allocated buffer (avoids allocation).
///
/// `out` must be at least `grid.len()` elements. Boundary values are set to 0.
/// Uses rayon parallelism over z-slabs for large grids.
fn compute_residual_into(
    grid: &PbGrid,
    potential: &[f64],
    charge_map: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    rhs_scale: f64,
    out: &mut [f64],
) {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];

    let stride_y = nx;
    let stride_z = nx * ny;
    let inv_hx2 = 1.0 / (grid.spacing[0] * grid.spacing[0]);
    let inv_hy2 = 1.0 / (grid.spacing[1] * grid.spacing[1]);
    let inv_hz2 = 1.0 / (grid.spacing[2] * grid.spacing[2]);

    // Zero the entire buffer (boundaries stay 0)
    out[..nx * ny * nz].fill(0.0);

    // Parallel over z-slabs: each z-slab writes to disjoint output indices
    let out_ptr = UnsafePotential(out.as_mut_ptr());
    let n = nx * ny * nz;

    (1..nz - 1).into_par_iter().for_each(|iz| {
        // Safety: each z iteration writes only to indices in [stride_z*iz .. stride_z*(iz+1)),
        // and these ranges are disjoint across different iz values.
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr.0, n) };
        let _ = &out_ptr; // ensure out_ptr is captured, not out

        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let idx = ix + stride_y * iy + stride_z * iz;

                let (neighbor_sum, diag) = stencil_at(
                    idx,
                    idx - 1,
                    idx + 1,
                    idx - stride_y,
                    idx + stride_y,
                    idx - stride_z,
                    idx + stride_z,
                    potential,
                    dielectrics,
                    kappa_sq,
                    inv_hx2,
                    inv_hy2,
                    inv_hz2,
                );

                out_slice[idx] = rhs_scale * charge_map[idx] + neighbor_sum - diag * potential[idx];
            }
        }
    });
}

/// Restrict a fine-grid array to a coarse grid using full-weighting.
/// Coarse grid has dims roughly half the fine grid.
#[cfg(test)]
fn restrict(fine: &[f64], fine_dims: &[usize; 3]) -> (Vec<f64>, [usize; 3]) {
    let coarse_dims = [
        fine_dims[0].div_ceil(2),
        fine_dims[1].div_ceil(2),
        fine_dims[2].div_ceil(2),
    ];
    let cnx = coarse_dims[0];
    let cny = coarse_dims[1];
    let cnz = coarse_dims[2];
    let fnx = fine_dims[0];
    let fny = fine_dims[1];
    let fnz = fine_dims[2];
    let fi = |x: usize, y: usize, z: usize| x + fnx * (y + fny * z);
    let ci_fn = |x: usize, y: usize, z: usize| x + cnx * (y + cny * z);

    let mut coarse = vec![0.0; cnx * cny * cnz];
    for cz in 0..cnz {
        for cy in 0..cny {
            for cx in 0..cnx {
                let fx = 2 * cx;
                let fy = 2 * cy;
                let fz = 2 * cz;

                // Full-weighting: sum over 3x3x3 neighborhood of (fx, fy, fz)
                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let ix = fx as i32 + dx;
                            let iy = fy as i32 + dy;
                            let iz = fz as i32 + dz;
                            if ix < 0
                                || ix >= fnx as i32
                                || iy < 0
                                || iy >= fny as i32
                                || iz < 0
                                || iz >= fnz as i32
                            {
                                continue;
                            }
                            // Weight: 2^(number of zero offsets)
                            // center(0,0,0)=8, face=4, edge=2, corner=1
                            let w = (if dx == 0 { 2.0 } else { 1.0 })
                                * (if dy == 0 { 2.0 } else { 1.0 })
                                * (if dz == 0 { 2.0 } else { 1.0 });
                            sum += w * fine[fi(ix as usize, iy as usize, iz as usize)];
                            weight_sum += w;
                        }
                    }
                }
                coarse[ci_fn(cx, cy, cz)] = sum / weight_sum;
            }
        }
    }
    (coarse, coarse_dims)
}

/// Restrict a fine-grid array into a pre-allocated coarse buffer.
///
/// Same algorithm as `restrict` but writes into `coarse_out` instead of allocating.
/// The `scale` parameter is multiplied into every output value, allowing the caller
/// to fuse post-restriction scaling (e.g. the (h_fine/h_coarse)² = 0.25 factor)
/// into the restriction pass, eliminating an extra loop over the coarse data.
fn restrict_into(fine: &[f64], fine_dims: &[usize; 3], coarse_out: &mut [f64], coarse_dims: &[usize; 3], scale: f64) {
    let cnx = coarse_dims[0];
    let cny = coarse_dims[1];
    let cnz = coarse_dims[2];
    let fnx = fine_dims[0];
    let fny = fine_dims[1];
    let fnz = fine_dims[2];
    let fi = |x: usize, y: usize, z: usize| x + fnx * (y + fny * z);
    let ci_fn = |x: usize, y: usize, z: usize| x + cnx * (y + cny * z);

    for cz in 0..cnz {
        for cy in 0..cny {
            for cx in 0..cnx {
                let fx = 2 * cx;
                let fy = 2 * cy;
                let fz = 2 * cz;

                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let ix = fx as i32 + dx;
                            let iy = fy as i32 + dy;
                            let iz = fz as i32 + dz;
                            if ix < 0
                                || ix >= fnx as i32
                                || iy < 0
                                || iy >= fny as i32
                                || iz < 0
                                || iz >= fnz as i32
                            {
                                continue;
                            }
                            let w = (if dx == 0 { 2.0 } else { 1.0 })
                                * (if dy == 0 { 2.0 } else { 1.0 })
                                * (if dz == 0 { 2.0 } else { 1.0 });
                            sum += w * fine[fi(ix as usize, iy as usize, iz as usize)];
                            weight_sum += w;
                        }
                    }
                }
                coarse_out[ci_fn(cx, cy, cz)] = scale * sum / weight_sum;
            }
        }
    }
}

/// Prolongate a coarse-grid correction to the fine grid using trilinear interpolation.
/// Adds the interpolated correction to the fine-grid array.
fn prolongate_add(
    coarse: &[f64],
    coarse_dims: &[usize; 3],
    fine: &mut [f64],
    fine_dims: &[usize; 3],
) {
    let fnx = fine_dims[0];
    let fny = fine_dims[1];
    let fnz = fine_dims[2];
    let cnx = coarse_dims[0];
    let cny = coarse_dims[1];
    let cnz = coarse_dims[2];
    let ci = |x: usize, y: usize, z: usize| x + cnx * (y + cny * z);

    for fz in 0..fnz {
        for fy in 0..fny {
            for fx in 0..fnx {
                // Coarse-grid fractional coordinates
                let cx_f = fx as f64 / 2.0;
                let cy_f = fy as f64 / 2.0;
                let cz_f = fz as f64 / 2.0;

                let cx0 = (cx_f.floor() as usize).min(cnx.saturating_sub(2));
                let cy0 = (cy_f.floor() as usize).min(cny.saturating_sub(2));
                let cz0 = (cz_f.floor() as usize).min(cnz.saturating_sub(2));

                let dx = cx_f - cx0 as f64;
                let dy = cy_f - cy0 as f64;
                let dz = cz_f - cz0 as f64;

                // Trilinear interpolation
                let c000 = coarse[ci(cx0, cy0, cz0)];
                let c100 = coarse[ci(cx0 + 1, cy0, cz0)];
                let c010 = coarse[ci(cx0, cy0 + 1, cz0)];
                let c110 = coarse[ci(cx0 + 1, cy0 + 1, cz0)];
                let c001 = coarse[ci(cx0, cy0, cz0 + 1)];
                let c101 = coarse[ci(cx0 + 1, cy0, cz0 + 1)];
                let c011 = coarse[ci(cx0, cy0 + 1, cz0 + 1)];
                let c111 = coarse[ci(cx0 + 1, cy0 + 1, cz0 + 1)];

                let c00 = c000 * (1.0 - dx) + c100 * dx;
                let c10 = c010 * (1.0 - dx) + c110 * dx;
                let c01 = c001 * (1.0 - dx) + c101 * dx;
                let c11 = c011 * (1.0 - dx) + c111 * dx;
                let c0 = c00 * (1.0 - dy) + c10 * dy;
                let c1 = c01 * (1.0 - dy) + c11 * dy;
                let val = c0 * (1.0 - dz) + c1 * dz;

                let fi = fx + fnx * (fy + fny * fz);
                fine[fi] += val;
            }
        }
    }
}

/// Coarsen dielectric maps to a coarser grid using injection.
fn coarsen_dielectrics(
    fine_diel: &DielectricMaps,
    fine_dims: &[usize; 3],
    coarse_dims: &[usize; 3],
) -> DielectricMaps {
    let cnx = coarse_dims[0];
    let cny = coarse_dims[1];
    let cnz = coarse_dims[2];
    let fnx = fine_dims[0];
    let fny = fine_dims[1];
    let cn = cnx * cny * cnz;
    let ci = |x: usize, y: usize, z: usize| x + cnx * (y + cny * z);
    let fi = |x: usize, y: usize, z: usize| x + fnx * (y + fny * z);

    let mut eps_x = vec![0.0; cn];
    let mut eps_y = vec![0.0; cn];
    let mut eps_z = vec![0.0; cn];

    for cz in 0..cnz {
        for cy in 0..cny {
            for cx in 0..cnx {
                let fx = (2 * cx).min(fine_dims[0] - 1);
                let fy = (2 * cy).min(fine_dims[1] - 1);
                let fz = (2 * cz).min(fine_dims[2] - 1);
                let c_idx = ci(cx, cy, cz);
                let f_idx = fi(fx, fy, fz);
                eps_x[c_idx] = fine_diel.eps_x[f_idx];
                eps_y[c_idx] = fine_diel.eps_y[f_idx];
                eps_z[c_idx] = fine_diel.eps_z[f_idx];
            }
        }
    }

    DielectricMaps {
        eps_x,
        eps_y,
        eps_z,
        dims: *coarse_dims,
    }
}

/// Coarsen kappa^2 map to a coarser grid using injection.
fn coarsen_kappa(fine_kappa: &[f64], fine_dims: &[usize; 3], coarse_dims: &[usize; 3]) -> Vec<f64> {
    let cnx = coarse_dims[0];
    let cny = coarse_dims[1];
    let cnz = coarse_dims[2];
    let fnx = fine_dims[0];
    let fny = fine_dims[1];
    let cn = cnx * cny * cnz;
    let ci = |x: usize, y: usize, z: usize| x + cnx * (y + cny * z);
    let fi = |x: usize, y: usize, z: usize| x + fnx * (y + fny * z);

    let mut coarse = vec![0.0; cn];
    for cz in 0..cnz {
        for cy in 0..cny {
            for cx in 0..cnx {
                let fx = (2 * cx).min(fine_dims[0] - 1);
                let fy = (2 * cy).min(fine_dims[1] - 1);
                let fz = (2 * cz).min(fine_dims[2] - 1);
                coarse[ci(cx, cy, cz)] = fine_kappa[fi(fx, fy, fz)];
            }
        }
    }
    coarse
}

/// One level of the multigrid hierarchy with pre-computed operators and
/// pre-allocated work buffers.
struct MultigridLevel {
    /// Coarse grid descriptor for this level.
    coarse_grid: PbGrid,
    /// Pre-coarsened dielectric maps (invariant across V-cycles).
    coarse_dielectrics: DielectricMaps,
    /// Pre-coarsened κ² map (invariant across V-cycles).
    coarse_kappa: Vec<f64>,
    /// Pre-computed SOR omega for the coarse grid.
    coarse_omega: f64,
    /// Fine-grid dimensions (for restrict/prolongate).
    fine_dims: [usize; 3],
    /// Pre-allocated buffer for fine-level residual.
    fine_residual_buf: Vec<f64>,
    /// Pre-allocated buffer for coarse-level RHS (restricted residual).
    coarse_rhs_buf: Vec<f64>,
    /// Pre-allocated buffer for coarse-level correction.
    coarse_correction_buf: Vec<f64>,
}

/// Build the multigrid level hierarchy from the finest grid operators.
///
/// Pre-computes coarsened dielectrics and kappa (which are invariant across
/// V-cycles) and pre-allocates all work buffers at each level, eliminating
/// per-cycle allocations.
fn build_multigrid_levels(
    grid: &PbGrid,
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    min_grid_size: usize,
) -> Vec<MultigridLevel> {
    let mut levels = Vec::new();
    let mut current_dims = grid.dims;
    let mut current_spacing = grid.spacing;
    let mut current_diel = dielectrics;
    let mut current_kappa = kappa_sq;

    // Owned storage for intermediate levels
    let mut owned_diels: Vec<DielectricMaps> = Vec::new();
    let mut owned_kappas: Vec<Vec<f64>> = Vec::new();

    loop {
        let min_dim = current_dims.iter().copied().min().unwrap_or(0);
        if min_dim <= min_grid_size {
            break;
        }

        let coarse_dims = [
            current_dims[0].div_ceil(2),
            current_dims[1].div_ceil(2),
            current_dims[2].div_ceil(2),
        ];
        let coarse_spacing = [
            current_spacing[0] * 2.0,
            current_spacing[1] * 2.0,
            current_spacing[2] * 2.0,
        ];
        let coarse_n = coarse_dims[0] * coarse_dims[1] * coarse_dims[2];
        let fine_n = current_dims[0] * current_dims[1] * current_dims[2];

        let coarse_diel = coarsen_dielectrics(current_diel, &current_dims, &coarse_dims);
        let coarse_kappa = coarsen_kappa(current_kappa, &current_dims, &coarse_dims);
        let coarse_omega = estimate_omega(&coarse_dims);

        levels.push(MultigridLevel {
            coarse_grid: PbGrid::descriptor(coarse_dims, coarse_spacing, grid.origin),
            coarse_dielectrics: coarse_diel,
            coarse_kappa: coarse_kappa,
            coarse_omega,
            fine_dims: current_dims,
            fine_residual_buf: vec![0.0; fine_n],
            coarse_rhs_buf: vec![0.0; coarse_n],
            coarse_correction_buf: vec![0.0; coarse_n],
        });

        // For next iteration, reference the coarsened operators we just built
        owned_diels.push(levels.last().unwrap().coarse_dielectrics.clone());
        owned_kappas.push(levels.last().unwrap().coarse_kappa.clone());
        current_diel = owned_diels.last().unwrap();
        current_kappa = owned_kappas.last().unwrap();
        current_dims = coarse_dims;
        current_spacing = coarse_spacing;
    }

    levels
}

/// Perform one V-cycle of multigrid using pre-allocated workspace.
///
/// Uses `split_first_mut` on the levels slice to satisfy the borrow checker
/// while recursing through the level hierarchy.
fn v_cycle_workspace(
    grid: &PbGrid,
    potential: &mut [f64],
    charge_map: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    omega: f64,
    levels: &mut [MultigridLevel],
    pre_smooth: usize,
    post_smooth: usize,
    rhs_scale: f64,
) {
    // Base case: no more coarser levels, solve directly
    if levels.is_empty() {
        sor_smooth(
            grid,
            potential,
            charge_map,
            dielectrics,
            kappa_sq,
            omega,
            50,
            rhs_scale,
        );
        return;
    }

    let (level, rest) = levels.split_first_mut().unwrap();

    // 1. Pre-smooth
    sor_smooth(
        grid,
        potential,
        charge_map,
        dielectrics,
        kappa_sq,
        omega,
        pre_smooth,
        rhs_scale,
    );

    // 2. Compute residual into pre-allocated buffer
    compute_residual_into(
        grid,
        potential,
        charge_map,
        dielectrics,
        kappa_sq,
        rhs_scale,
        &mut level.fine_residual_buf,
    );

    // 3. Restrict residual to coarse grid with fused (h_fine/h_coarse)² = 0.25 scaling
    restrict_into(
        &level.fine_residual_buf,
        &level.fine_dims,
        &mut level.coarse_rhs_buf,
        &level.coarse_grid.dims,
        0.25,
    );

    // 4. Zero the correction buffer
    level.coarse_correction_buf.fill(0.0);

    // 5. Recurse into next coarser level
    v_cycle_workspace(
        &level.coarse_grid,
        &mut level.coarse_correction_buf,
        &level.coarse_rhs_buf,
        &level.coarse_dielectrics,
        &level.coarse_kappa,
        level.coarse_omega,
        rest,
        pre_smooth,
        post_smooth,
        1.0, // coarse levels use rhs_scale=1.0
    );

    // 6. Prolongate coarse correction and add to fine potential
    prolongate_add(
        &level.coarse_correction_buf,
        &level.coarse_grid.dims,
        potential,
        &grid.dims,
    );

    // 7. Post-smooth
    sor_smooth(
        grid,
        potential,
        charge_map,
        dielectrics,
        kappa_sq,
        omega,
        post_smooth,
        rhs_scale,
    );
}

/// Compute the RMS residual without allocating a full residual vector.
///
/// This fuses `compute_residual` and `rms_interior_residual` into a single
/// pass, avoiding the ~33 MB allocation per call on a 161^3 grid.
/// Uses rayon parallelism over z-slabs.
fn compute_rms_residual(
    grid: &PbGrid,
    potential: &[f64],
    charge_map: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    rhs_scale: f64,
) -> f64 {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];

    let stride_y = nx;
    let stride_z = nx * ny;
    let inv_hx2 = 1.0 / (grid.spacing[0] * grid.spacing[0]);
    let inv_hy2 = 1.0 / (grid.spacing[1] * grid.spacing[1]);
    let inv_hz2 = 1.0 / (grid.spacing[2] * grid.spacing[2]);

    // Parallel reduction over z-slabs
    let (rms, count) = (1..nz - 1)
        .into_par_iter()
        .map(|iz| {
            let mut local_rms = 0.0;
            let mut local_count = 0u64;

            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let idx = ix + stride_y * iy + stride_z * iz;

                    let (neighbor_sum, diag) = stencil_at(
                        idx,
                        idx - 1,
                        idx + 1,
                        idx - stride_y,
                        idx + stride_y,
                        idx - stride_z,
                        idx + stride_z,
                        potential,
                        dielectrics,
                        kappa_sq,
                        inv_hx2,
                        inv_hy2,
                        inv_hz2,
                    );

                    let r = rhs_scale * charge_map[idx] + neighbor_sum - diag * potential[idx];
                    local_rms += r * r;
                    local_count += 1;
                }
            }

            (local_rms, local_count)
        })
        .reduce(
            || (0.0, 0u64),
            |(a_rms, a_cnt), (b_rms, b_cnt)| (a_rms + b_rms, a_cnt + b_cnt),
        );

    if count > 0 {
        (rms / count as f64).sqrt()
    } else {
        0.0
    }
}

/// Solve the linearized PBE using multigrid-accelerated SOR.
///
/// Uses V-cycles with the existing parallel red-black SOR as the smoother.
/// Falls back to plain SOR for grids smaller than the minimum coarsening threshold.
pub fn solve_lpbe_multigrid(
    grid: &PbGrid,
    charge_map: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    boundary: BoundaryCondition,
    coords: &[[f64; 3]],
    charges: &[f64],
    kappa_bulk: f64,
    eps_out: f64,
    tolerance: f64,
    max_iterations: usize,
) -> PbSolveResult {
    let n = grid.len();
    let mut potential = vec![0.0f64; n];

    // Set boundary conditions
    match &boundary {
        BoundaryCondition::Zero => {}
        BoundaryCondition::DebyeHuckel => {
            set_dh_boundary(&mut potential, grid, coords, charges, kappa_bulk, eps_out);
        }
        BoundaryCondition::Interpolated(bvals) => {
            set_interpolated_boundary(&mut potential, bvals);
        }
    }

    let min_dim = grid.dims.iter().copied().min().unwrap_or(0);

    // For small grids, fall back to plain SOR
    if min_dim <= 8 {
        return solve_lpbe(
            grid,
            charge_map,
            dielectrics,
            kappa_sq,
            boundary,
            coords,
            charges,
            kappa_bulk,
            eps_out,
            tolerance,
            max_iterations,
        );
    }

    let pre_smooth = 3;
    let post_smooth = 3;
    let min_grid_size = 5;
    // Each V-cycle does pre_smooth + post_smooth sweeps at finest level (plus coarser),
    // roughly equivalent to ~6-10 SOR iterations in cost. Cap at 200 cycles to
    // avoid runaway computation on large grids in debug builds.
    let max_vcycles = (max_iterations / 6).clamp(1, 200);

    // Build multigrid workspace once: pre-computes coarsened operators
    // (invariant across V-cycles) and pre-allocates all work buffers.
    let mut mg_levels = build_multigrid_levels(grid, dielectrics, kappa_sq, min_grid_size);
    let omega = estimate_omega(&grid.dims);

    let mut best_rms = f64::MAX;
    let mut growing_count = 0u32;
    let mut prev_rms = f64::MAX;

    for cycle in 0..max_vcycles {
        v_cycle_workspace(
            grid,
            &mut potential,
            charge_map,
            dielectrics,
            kappa_sq,
            omega,
            &mut mg_levels,
            pre_smooth,
            post_smooth,
            FOUR_PI_EC,
        );

        // Check convergence without allocating a full residual vector
        let rms_val = compute_rms_residual(
            grid,
            &potential,
            charge_map,
            dielectrics,
            kappa_sq,
            FOUR_PI_EC,
        );

        // Track best residual seen
        if rms_val < best_rms {
            best_rms = rms_val;
            growing_count = 0;
        } else {
            growing_count += 1;
        }

        // Detect divergence: NaN/Inf, large growth from best, or sustained growth
        let diverged = !rms_val.is_finite()
            || rms_val > best_rms * 1e4
            || (growing_count >= 5 && rms_val > prev_rms);

        if diverged {
            log::warn!(
                "Multigrid diverged at cycle {} (rms={:.2e}, best={:.2e}), falling back to plain SOR with partial solution",
                cycle + 1,
                rms_val,
                best_rms,
            );
            // Continue from the partially-converged potential instead of
            // starting from scratch, saving the work done so far.
            return solve_lpbe_initial(
                grid,
                charge_map,
                dielectrics,
                kappa_sq,
                boundary,
                coords,
                charges,
                kappa_bulk,
                eps_out,
                tolerance,
                max_iterations,
                Some(potential),
            );
        }
        prev_rms = rms_val;

        if rms_val < tolerance {
            return PbSolveResult {
                potential,
                iterations: cycle + 1,
                final_residual: rms_val,
                converged: true,
            };
        }
    }

    // Did not converge -- compute final residual
    let rms_val = compute_rms_residual(
        grid,
        &potential,
        charge_map,
        dielectrics,
        kappa_sq,
        FOUR_PI_EC,
    );

    PbSolveResult {
        potential,
        iterations: max_vcycles,
        final_residual: rms_val,
        converged: false,
    }
}

/// Compute the electrostatic energy from the potential and atomic charges.
///
/// E = 0.5 * sum_i q_i * phi(r_i)
///
/// where phi is in kcal/mol/e (the 332.0522 conversion is embedded in the PBE).
pub fn compute_elec_energy(
    grid: &PbGrid,
    potential: &[f64],
    coords: &[[f64; 3]],
    charges: &[f64],
) -> f64 {
    let mut energy = 0.0;
    for (c, &q) in coords.iter().zip(charges.iter()) {
        let phi = grid.interpolate_with_data(c, potential);
        energy += q * phi;
    }
    0.5 * energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb_grid::{assign_dielectrics, auto_grid, map_charges};

    #[test]
    fn test_sor_convergence() {
        // Simple test: single charge in vacuum, should converge
        let coords = vec![[0.0, 0.0, 0.0]];
        let charges = vec![1.0];
        let radii = vec![1.5];

        let grid = auto_grid(&coords, 1.0, 10.0);
        let charge_map = map_charges(&grid, &coords, &charges);
        let dielectrics = assign_dielectrics(&grid, &coords, &radii, 1.4, 1.0, 1.0);
        let kappa_sq = vec![0.0; grid.len()];

        let result = solve_lpbe(
            &grid,
            &charge_map,
            &dielectrics,
            &kappa_sq,
            BoundaryCondition::Zero,
            &coords,
            &charges,
            0.0,
            1.0,
            1e-6,
            5000,
        );

        assert!(
            result.converged,
            "SOR did not converge: residual = {}",
            result.final_residual
        );
    }

    #[test]
    fn test_multigrid_convergence() {
        // Same test as SOR but using multigrid
        let coords = vec![[0.0, 0.0, 0.0]];
        let charges = vec![1.0];
        let radii = vec![1.5];

        let grid = auto_grid(&coords, 1.0, 10.0);
        let charge_map = map_charges(&grid, &coords, &charges);
        let dielectrics = assign_dielectrics(&grid, &coords, &radii, 1.4, 1.0, 1.0);
        let kappa_sq = vec![0.0; grid.len()];

        let result = solve_lpbe_multigrid(
            &grid,
            &charge_map,
            &dielectrics,
            &kappa_sq,
            BoundaryCondition::Zero,
            &coords,
            &charges,
            0.0,
            1.0,
            1e-6,
            5000,
        );

        assert!(
            result.converged,
            "Multigrid did not converge: residual = {}",
            result.final_residual
        );
    }

    #[test]
    fn test_restrict_prolongate_identity() {
        // Restricting a constant field and prolongating back should give the same constant
        let dims = [9, 9, 9];
        let fine = vec![3.0; 9 * 9 * 9];
        let (coarse, cdims) = restrict(&fine, &dims);
        // All coarse values should be 3.0
        for &v in &coarse {
            assert!(
                (v - 3.0).abs() < 1e-12,
                "Restrict failed on constant: got {}",
                v
            );
        }
        // Prolongate to a zero fine grid, should add 3.0 everywhere
        let mut fine_out = vec![0.0; 9 * 9 * 9];
        prolongate_add(&coarse, &cdims, &mut fine_out, &dims);
        for &v in &fine_out {
            assert!(
                (v - 3.0).abs() < 0.5,
                "Prolongate failed on constant: got {}",
                v
            );
        }
    }
}
