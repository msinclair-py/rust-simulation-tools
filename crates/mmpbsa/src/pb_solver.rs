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
    /// Contains the precomputed boundary values for each boundary grid point.
    Interpolated(Vec<f64>),
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
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];
    let conversion = 332.0522; // kcal*A/(mol*e^2)

    // Iterate over all boundary points
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let is_boundary =
                    ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1 || iz == 0 || iz == nz - 1;
                if !is_boundary {
                    continue;
                }

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
            }
        }
    }
}

/// Compute interpolated boundary conditions from a coarse grid solution.
///
/// For each boundary point of `fine_grid`, looks up the corresponding
/// position in the coarse solution via trilinear interpolation.
pub fn interpolated_boundary(
    fine_grid: &PbGrid,
    coarse_grid: &PbGrid,
    coarse_potential: &[f64],
) -> Vec<f64> {
    let nx = fine_grid.dims[0];
    let ny = fine_grid.dims[1];
    let nz = fine_grid.dims[2];
    let n = nx * ny * nz;
    let mut boundary = vec![0.0f64; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let is_boundary =
                    ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1 || iz == 0 || iz == nz - 1;
                if !is_boundary {
                    continue;
                }
                let pt = fine_grid.point(ix, iy, iz);
                let idx = fine_grid.index(ix, iy, iz);
                boundary[idx] = coarse_grid.interpolate_with_data(&pt, coarse_potential);
            }
        }
    }
    boundary
}

/// Set boundary conditions from precomputed interpolated values.
fn set_interpolated_boundary(potential: &mut [f64], grid: &PbGrid, boundary: &[f64]) {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let is_boundary =
                    ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1 || iz == 0 || iz == nz - 1;
                if !is_boundary {
                    continue;
                }
                let idx = grid.index(ix, iy, iz);
                potential[idx] = boundary[idx];
            }
        }
    }
}

/// Estimate the optimal SOR relaxation parameter from grid dimensions.
fn estimate_omega(dims: &[usize; 3]) -> f64 {
    // Spectral radius of Jacobi iteration for 3D Laplacian
    let nx = dims[0] as f64;
    let ny = dims[1] as f64;
    let nz = dims[2] as f64;
    let rho = ((std::f64::consts::PI / nx).cos()
        + (std::f64::consts::PI / ny).cos()
        + (std::f64::consts::PI / nz).cos())
        / 3.0;
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

/// Tile size for cache-friendly traversal. A 16x16x16 tile uses
/// 16*16*16*8 = 32KB, fitting in L1 cache.
const TILE: usize = 16;

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

    let hx = grid.spacing[0];
    let hy = grid.spacing[1];
    let hz = grid.spacing[2];
    let hx2 = hx * hx;
    let hy2 = hy * hy;
    let hz2 = hz * hz;

    let mut last_rms = 0.0;

    for _sweep in 0..n_sweeps {
        let mut rms_residual = 0.0;
        let mut count = 0u64;

        let ptr = UnsafePotential(potential.as_mut_ptr());

        // Red-black SOR: two half-sweeps
        for color in 0..2 {
            let z_tiles: Vec<usize> = (1..nz - 1).step_by(TILE).collect();
            let (color_rms, color_count): (f64, u64) = z_tiles
                .into_par_iter()
                .map({
                    let ptr = &ptr;
                    move |tz| {
                        let mut local_rms = 0.0;
                        let mut local_count = 0u64;
                        let iz_end = (tz + TILE).min(nz - 1);
                        for ty in (1..ny - 1).step_by(TILE) {
                            let iy_end = (ty + TILE).min(ny - 1);
                            for tx in (1..nx - 1).step_by(TILE) {
                                let ix_end = (tx + TILE).min(nx - 1);
                                for iz in tz..iz_end {
                                    for iy in ty..iy_end {
                                        for ix in tx..ix_end {
                                            if (ix + iy + iz) % 2 != color {
                                                continue;
                                            }

                                            let idx = grid.index(ix, iy, iz);

                                            let ex_p = dielectrics.eps_x[grid.index(ix, iy, iz)];
                                            let ex_m =
                                                dielectrics.eps_x[grid.index(ix - 1, iy, iz)];
                                            let ey_p = dielectrics.eps_y[grid.index(ix, iy, iz)];
                                            let ey_m =
                                                dielectrics.eps_y[grid.index(ix, iy - 1, iz)];
                                            let ez_p = dielectrics.eps_z[grid.index(ix, iy, iz)];
                                            let ez_m =
                                                dielectrics.eps_z[grid.index(ix, iy, iz - 1)];

                                            // Safety: each thread only reads neighbors (opposite color,
                                            // not being written) and writes to its own color points.
                                            let potential =
                                                unsafe { std::slice::from_raw_parts_mut(ptr.0, n) };

                                            let neighbor_sum =
                                                ex_p * potential[grid.index(ix + 1, iy, iz)] / hx2
                                                    + ex_m * potential[grid.index(ix - 1, iy, iz)]
                                                        / hx2
                                                    + ey_p * potential[grid.index(ix, iy + 1, iz)]
                                                        / hy2
                                                    + ey_m * potential[grid.index(ix, iy - 1, iz)]
                                                        / hy2
                                                    + ez_p * potential[grid.index(ix, iy, iz + 1)]
                                                        / hz2
                                                    + ez_m * potential[grid.index(ix, iy, iz - 1)]
                                                        / hz2;

                                            let diag = (ex_p + ex_m) / hx2
                                                + (ey_p + ey_m) / hy2
                                                + (ez_p + ez_m) / hz2
                                                + kappa_sq[idx];

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
    let n = grid.len();
    let mut potential = vec![0.0f64; n];

    // Set boundary conditions
    match &boundary {
        BoundaryCondition::Zero => {}
        BoundaryCondition::DebyeHuckel => {
            set_dh_boundary(&mut potential, grid, coords, charges, kappa_bulk, eps_out);
        }
        BoundaryCondition::Interpolated(bvals) => {
            set_interpolated_boundary(&mut potential, grid, bvals);
        }
    }

    let omega = estimate_omega(&grid.dims);

    let mut iterations = 0;
    let mut final_residual = f64::MAX;

    for iter in 0..max_iterations {
        final_residual = sor_smooth(
            grid,
            &mut potential,
            charge_map,
            dielectrics,
            kappa_sq,
            omega,
            1,
            FOUR_PI_EC,
        );
        iterations = iter + 1;

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

/// Compute the residual r = b - A*x at all interior points.
/// Returns the residual vector (same size as potential). Boundary values are 0.
///
/// `rhs_scale` multiplies `charge_map` to form the RHS, same as in `sor_smooth`.
fn compute_residual(
    grid: &PbGrid,
    potential: &[f64],
    charge_map: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    rhs_scale: f64,
) -> Vec<f64> {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];
    let n = nx * ny * nz;

    let hx2 = grid.spacing[0] * grid.spacing[0];
    let hy2 = grid.spacing[1] * grid.spacing[1];
    let hz2 = grid.spacing[2] * grid.spacing[2];

    let mut residual = vec![0.0; n];

    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let idx = grid.index(ix, iy, iz);

                let ex_p = dielectrics.eps_x[grid.index(ix, iy, iz)];
                let ex_m = dielectrics.eps_x[grid.index(ix - 1, iy, iz)];
                let ey_p = dielectrics.eps_y[grid.index(ix, iy, iz)];
                let ey_m = dielectrics.eps_y[grid.index(ix, iy - 1, iz)];
                let ez_p = dielectrics.eps_z[grid.index(ix, iy, iz)];
                let ez_m = dielectrics.eps_z[grid.index(ix, iy, iz - 1)];

                let neighbor_sum = ex_p * potential[grid.index(ix + 1, iy, iz)] / hx2
                    + ex_m * potential[grid.index(ix - 1, iy, iz)] / hx2
                    + ey_p * potential[grid.index(ix, iy + 1, iz)] / hy2
                    + ey_m * potential[grid.index(ix, iy - 1, iz)] / hy2
                    + ez_p * potential[grid.index(ix, iy, iz + 1)] / hz2
                    + ez_m * potential[grid.index(ix, iy, iz - 1)] / hz2;

                let diag =
                    (ex_p + ex_m) / hx2 + (ey_p + ey_m) / hy2 + (ez_p + ez_m) / hz2 + kappa_sq[idx];

                // residual = rhs - A*x = (rhs_scale * charge) - (diag * phi - neighbor_sum)
                residual[idx] = rhs_scale * charge_map[idx] + neighbor_sum - diag * potential[idx];
            }
        }
    }

    residual
}

/// Restrict a fine-grid array to a coarse grid using full-weighting.
/// Coarse grid has dims roughly half the fine grid.
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

/// Perform one V-cycle of multigrid.
///
/// Recursively coarsens, solves, and prolongates.
/// `rhs_scale` is the factor applied to `charge_map` to form the RHS.
/// At the finest level this is `FOUR_PI_EC`; at coarse levels it is `1.0`
/// because the restricted residual already has the correct units.
fn v_cycle(
    grid: &PbGrid,
    potential: &mut [f64],
    charge_map: &[f64],
    dielectrics: &DielectricMaps,
    kappa_sq: &[f64],
    pre_smooth: usize,
    post_smooth: usize,
    min_grid_size: usize,
    rhs_scale: f64,
) {
    let min_dim = grid.dims.iter().copied().min().unwrap_or(0);

    // Base case: grid is small enough, solve directly with many SOR iterations
    if min_dim <= min_grid_size {
        let omega = estimate_omega(&grid.dims);
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

    let omega = estimate_omega(&grid.dims);

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

    // 2. Compute residual
    let residual = compute_residual(
        grid,
        potential,
        charge_map,
        dielectrics,
        kappa_sq,
        rhs_scale,
    );

    // 3. Restrict residual to coarse grid.
    // The fine-grid operator uses 1/h^2 scaling while the coarse-grid operator uses
    // 1/(2h)^2 = 1/(4h^2). To compensate, we scale the restricted residual by the
    // ratio (h_fine/h_coarse)^2 = 1/4 so the coarse-grid equation A_c * e = R(r)
    // produces corrections of the right magnitude.
    let (mut coarse_rhs, coarse_dims) = restrict(&residual, &grid.dims);
    let h_ratio_sq = 0.25; // (h / 2h)^2 = 1/4
    for v in &mut coarse_rhs {
        *v *= h_ratio_sq;
    }

    // 4. Build coarse grid and operators
    let coarse_spacing = [
        grid.spacing[0] * 2.0,
        grid.spacing[1] * 2.0,
        grid.spacing[2] * 2.0,
    ];
    let coarse_grid = PbGrid::new(coarse_dims, coarse_spacing, grid.origin);
    let coarse_diel = coarsen_dielectrics(dielectrics, &grid.dims, &coarse_dims);
    let coarse_kappa = coarsen_kappa(kappa_sq, &grid.dims, &coarse_dims);

    // 5. Initialize coarse correction to zero and solve A_c * e_c = r_c
    // The restricted residual IS the RHS for the coarse error equation,
    // so rhs_scale = 1.0 on coarse levels.
    let mut coarse_correction = vec![0.0; coarse_grid.len()];

    // Recurse
    v_cycle(
        &coarse_grid,
        &mut coarse_correction,
        &coarse_rhs,
        &coarse_diel,
        &coarse_kappa,
        pre_smooth,
        post_smooth,
        min_grid_size,
        1.0,
    );

    // 6. Prolongate coarse correction and add to fine potential
    prolongate_add(&coarse_correction, &coarse_dims, potential, &grid.dims);

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

/// Compute the RMS residual over interior grid points.
fn rms_interior_residual(grid: &PbGrid, residual: &[f64]) -> f64 {
    let nx = grid.dims[0];
    let ny = grid.dims[1];
    let nz = grid.dims[2];
    let mut rms = 0.0;
    let mut count = 0u64;
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let idx = grid.index(ix, iy, iz);
                rms += residual[idx] * residual[idx];
                count += 1;
            }
        }
    }
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
            set_interpolated_boundary(&mut potential, grid, bvals);
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

    let mut prev_rms = f64::MAX;

    for cycle in 0..max_vcycles {
        v_cycle(
            grid,
            &mut potential,
            charge_map,
            dielectrics,
            kappa_sq,
            pre_smooth,
            post_smooth,
            min_grid_size,
            FOUR_PI_EC,
        );

        // Check convergence by computing residual norm
        let residual = compute_residual(
            grid,
            &potential,
            charge_map,
            dielectrics,
            kappa_sq,
            FOUR_PI_EC,
        );
        let rms_val = rms_interior_residual(grid, &residual);

        // Detect divergence: if residual grows significantly, fall back to plain SOR
        if rms_val > prev_rms * 1e6 || !rms_val.is_finite() {
            log::warn!(
                "Multigrid diverged at cycle {} (rms={:.2e}), falling back to plain SOR",
                cycle + 1,
                rms_val
            );
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
    let residual = compute_residual(
        grid,
        &potential,
        charge_map,
        dielectrics,
        kappa_sq,
        FOUR_PI_EC,
    );
    let rms_val = rms_interior_residual(grid, &residual);

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
