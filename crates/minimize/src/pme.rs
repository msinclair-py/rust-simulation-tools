//! Particle Mesh Ewald (PME) reciprocal-space electrostatics.
//!
//! Implements the smooth PME method for computing long-range electrostatic
//! energies and forces in periodic systems. The algorithm decomposes Coulomb
//! interactions into a short-range real-space sum (handled elsewhere) and a
//! long-range reciprocal-space sum computed here via 3D FFT on a charge grid.
//!
//! # Algorithm
//!
//! 1. Spread atomic charges onto a regular grid using 4th-order Cardinal B-splines.
//! 2. Forward 3D FFT of the charge grid.
//! 3. Multiply by the influence function (reciprocal-space Green's function).
//! 4. Inverse 3D FFT to obtain the potential on the grid.
//! 5. Interpolate forces back to atoms using B-spline derivatives.
//! 6. Apply the self-energy correction.
//!
//! # Units
//!
//! All inputs and outputs use AMBER internal units:
//! - Positions in Angstroms
//! - Charges in AMBER internal units (`q_amber = q_real * 18.2223`)
//! - Energies in kcal/mol
//! - Forces in kcal/(mol*A)
//!
//! With AMBER internal charges, `q_i * q_j / r` directly yields kcal/mol
//! because `18.2223^2 = 332.0522` (the Coulomb constant in kcal*A/(mol*e^2)).

use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

/// Default Ewald tolerance for determining alpha.
const EWALD_TOLERANCE: f64 = 1.0e-5;

/// Default grid spacing in Angstroms.
const DEFAULT_GRID_SPACING: f64 = 1.0;

/// B-spline order (4 = cubic B-spline, standard for PME).
const PME_ORDER: usize = 4;

// ============================================================================
// Utility: smooth FFT grid sizes
// ============================================================================

/// Find the smallest integer >= `n` that is a product of small primes (2, 3, 5, 7).
///
/// FFT algorithms are most efficient for transform lengths that factor into
/// small primes. This function finds the next such "smooth" number.
fn next_smooth_size(n: usize) -> usize {
    if n <= 1 {
        return 2;
    }
    let mut candidate = n;
    loop {
        if is_smooth(candidate) {
            return candidate;
        }
        candidate += 1;
    }
}

/// Check whether `n` factors completely into primes 2, 3, 5, and 7.
#[inline]
fn is_smooth(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    for &p in &[2, 3, 5, 7] {
        while n.is_multiple_of(p) {
            n /= p;
        }
    }
    n == 1
}

// ============================================================================
// 4th-order Cardinal B-spline
// ============================================================================

/// Evaluate the 4th-order Cardinal B-spline M_4(u) for u in [0, 4].
///
/// The piecewise polynomial is:
/// - `[0, 1)`: `u^3 / 6`
/// - `[1, 2)`: `(-3u^3 + 12u^2 - 12u + 4) / 6`
/// - `[2, 3)`: `(3u^3 - 24u^2 + 60u - 44) / 6`
/// - `[3, 4)`: `(4 - u)^3 / 6`
///
/// Returns 0 outside `[0, 4)`.
#[inline]
fn bspline4(u: f64) -> f64 {
    if !(0.0..4.0).contains(&u) {
        return 0.0;
    }
    if u < 1.0 {
        u * u * u / 6.0
    } else if u < 2.0 {
        let u2 = u * u;
        let u3 = u2 * u;
        (-3.0 * u3 + 12.0 * u2 - 12.0 * u + 4.0) / 6.0
    } else if u < 3.0 {
        let u2 = u * u;
        let u3 = u2 * u;
        (3.0 * u3 - 24.0 * u2 + 60.0 * u - 44.0) / 6.0
    } else {
        let t = 4.0 - u;
        t * t * t / 6.0
    }
}

/// Evaluate the derivative dM_4/du of the 4th-order Cardinal B-spline.
///
/// Computed analytically from the piecewise polynomial:
/// - `[0, 1)`: `u^2 / 2`
/// - `[1, 2)`: `(-9u^2 + 24u - 12) / 6`
/// - `[2, 3)`: `(9u^2 - 48u + 60) / 6`
/// - `[3, 4)`: `-(4 - u)^2 / 2`
///
/// Returns 0 outside `[0, 4)`.
#[inline]
fn bspline4_deriv(u: f64) -> f64 {
    if !(0.0..4.0).contains(&u) {
        return 0.0;
    }
    if u < 1.0 {
        u * u * 0.5
    } else if u < 2.0 {
        (-9.0 * u * u + 24.0 * u - 12.0) / 6.0
    } else if u < 3.0 {
        (9.0 * u * u - 48.0 * u + 60.0) / 6.0
    } else {
        let t = 4.0 - u;
        -t * t * 0.5
    }
}

/// Compute B-spline values for a single fractional coordinate.
///
/// Given the fractional grid coordinate `w` (e.g., `x * nx / box_x`),
/// fills `spline_values` with the 4 B-spline weights and returns the
/// integer grid index of the first affected grid point.
///
/// # Arguments
///
/// * `w` - Fractional grid coordinate (may be outside `[0, grid_dim)`)
/// * `grid_dim` - Number of grid points along this axis
/// * `spline_values` - Output array of length `PME_ORDER` for B-spline weights
/// * `spline_derivs` - Output array of length `PME_ORDER` for B-spline derivatives
///
/// # Returns
///
/// The integer grid index of the first affected grid point.
#[inline]
fn compute_bspline_1d(
    w: f64,
    grid_dim: usize,
    spline_values: &mut [f64; PME_ORDER],
    spline_derivs: &mut [f64; PME_ORDER],
) -> usize {
    // Wrap w into [0, grid_dim)
    let gd = grid_dim as f64;
    let w_wrapped = w - (w / gd).floor() * gd;

    let iw = w_wrapped.floor() as usize;
    let frac = w_wrapped - iw as f64;

    // B-spline arguments: the spline support covers grid points
    // iw - (order-1), ..., iw
    // which maps to u values: order - 1 + frac, ..., frac
    // For order=4: u values are (3+frac), (2+frac), (1+frac), frac
    // But the standard PME convention is:
    //   grid points: iw - (order-1) + k for k = 0..order-1
    //   u = order - 1 - k + frac
    for k in 0..PME_ORDER {
        let u = (PME_ORDER - 1 - k) as f64 + frac;
        spline_values[k] = bspline4(u);
        spline_derivs[k] = bspline4_deriv(u);
    }

    // The first grid point in the stencil (may need periodic wrapping)
    // iw - (order - 1) with periodic wrapping
    (iw + grid_dim - (PME_ORDER - 1)) % grid_dim
}

// ============================================================================
// 3D FFT composed from 1D transforms
// ============================================================================

/// Plans for forward and inverse 1D FFTs along each axis.
struct FftPlans {
    forward_x: Arc<dyn Fft<f64>>,
    forward_y: Arc<dyn Fft<f64>>,
    forward_z: Arc<dyn Fft<f64>>,
    inverse_x: Arc<dyn Fft<f64>>,
    inverse_y: Arc<dyn Fft<f64>>,
    inverse_z: Arc<dyn Fft<f64>>,
}

impl FftPlans {
    fn new(dims: [usize; 3]) -> Self {
        let mut planner = FftPlanner::new();
        FftPlans {
            forward_x: planner.plan_fft_forward(dims[0]),
            forward_y: planner.plan_fft_forward(dims[1]),
            forward_z: planner.plan_fft_forward(dims[2]),
            inverse_x: planner.plan_fft_inverse(dims[0]),
            inverse_y: planner.plan_fft_inverse(dims[1]),
            inverse_z: planner.plan_fft_inverse(dims[2]),
        }
    }
}

/// Perform a forward 3D FFT in-place on a complex grid of dimensions `[nx, ny, nz]`.
///
/// The 3D transform is decomposed into batches of 1D FFTs:
/// 1. Transform along z (contiguous in memory) for each (x, y)
/// 2. Transform along y for each (x, z)
/// 3. Transform along x for each (y, z)
fn fft3d_forward(
    data: &mut [Complex64],
    dims: [usize; 3],
    plans: &FftPlans,
    scratch: &mut Vec<Complex64>,
) {
    let [nx, ny, nz] = dims;
    debug_assert_eq!(data.len(), nx * ny * nz);

    // FFT along z: for each (ix, iy), transform data[ix*ny*nz + iy*nz .. +nz]
    let scratch_len = plans.forward_z.get_inplace_scratch_len();
    scratch.resize(scratch_len, Complex64::new(0.0, 0.0));
    for ix in 0..nx {
        for iy in 0..ny {
            let offset = ix * ny * nz + iy * nz;
            plans
                .forward_z
                .process_with_scratch(&mut data[offset..offset + nz], scratch);
        }
    }

    // FFT along y: for each (ix, iz), gather y-stride, transform, scatter back
    let scratch_len = plans.forward_y.get_inplace_scratch_len().max(ny);
    scratch.resize(scratch_len, Complex64::new(0.0, 0.0));
    let mut row_buf = vec![Complex64::new(0.0, 0.0); ny];
    for ix in 0..nx {
        for iz in 0..nz {
            // Gather
            for iy in 0..ny {
                row_buf[iy] = data[ix * ny * nz + iy * nz + iz];
            }
            plans
                .forward_y
                .process_with_scratch(&mut row_buf, scratch);
            // Scatter
            for iy in 0..ny {
                data[ix * ny * nz + iy * nz + iz] = row_buf[iy];
            }
        }
    }

    // FFT along x: for each (iy, iz), gather x-stride, transform, scatter back
    let scratch_len = plans.forward_x.get_inplace_scratch_len().max(nx);
    scratch.resize(scratch_len, Complex64::new(0.0, 0.0));
    let mut row_buf = vec![Complex64::new(0.0, 0.0); nx];
    for iy in 0..ny {
        for iz in 0..nz {
            for ix in 0..nx {
                row_buf[ix] = data[ix * ny * nz + iy * nz + iz];
            }
            plans
                .forward_x
                .process_with_scratch(&mut row_buf, scratch);
            for ix in 0..nx {
                data[ix * ny * nz + iy * nz + iz] = row_buf[ix];
            }
        }
    }
}

/// Perform an inverse 3D FFT in-place on a complex grid of dimensions `[nx, ny, nz]`.
///
/// Same decomposition as forward but using inverse plans. The result is NOT
/// normalized; the caller must divide by `nx * ny * nz` if needed.
fn fft3d_inverse(
    data: &mut [Complex64],
    dims: [usize; 3],
    plans: &FftPlans,
    scratch: &mut Vec<Complex64>,
) {
    let [nx, ny, nz] = dims;
    debug_assert_eq!(data.len(), nx * ny * nz);

    // Inverse FFT along x
    let scratch_len = plans.inverse_x.get_inplace_scratch_len().max(nx);
    scratch.resize(scratch_len, Complex64::new(0.0, 0.0));
    let mut row_buf = vec![Complex64::new(0.0, 0.0); nx];
    for iy in 0..ny {
        for iz in 0..nz {
            for ix in 0..nx {
                row_buf[ix] = data[ix * ny * nz + iy * nz + iz];
            }
            plans
                .inverse_x
                .process_with_scratch(&mut row_buf, scratch);
            for ix in 0..nx {
                data[ix * ny * nz + iy * nz + iz] = row_buf[ix];
            }
        }
    }

    // Inverse FFT along y
    let scratch_len = plans.inverse_y.get_inplace_scratch_len().max(ny);
    scratch.resize(scratch_len, Complex64::new(0.0, 0.0));
    let mut row_buf = vec![Complex64::new(0.0, 0.0); ny];
    for ix in 0..nx {
        for iz in 0..nz {
            for iy in 0..ny {
                row_buf[iy] = data[ix * ny * nz + iy * nz + iz];
            }
            plans
                .inverse_y
                .process_with_scratch(&mut row_buf, scratch);
            for iy in 0..ny {
                data[ix * ny * nz + iy * nz + iz] = row_buf[iy];
            }
        }
    }

    // Inverse FFT along z
    let scratch_len = plans.inverse_z.get_inplace_scratch_len();
    scratch.resize(scratch_len, Complex64::new(0.0, 0.0));
    for ix in 0..nx {
        for iy in 0..ny {
            let offset = ix * ny * nz + iy * nz;
            plans
                .inverse_z
                .process_with_scratch(&mut data[offset..offset + nz], scratch);
        }
    }
}

// ============================================================================
// Influence function (reciprocal-space Green's function)
// ============================================================================

/// Compute the B-spline structure factor squared for one axis.
///
/// For a grid of dimension `n` and B-spline order `order`, the structure
/// factor at reciprocal-space index `m` is:
///
/// ```text
/// b(m) = exp(2*pi*i*(order-1)*m/n) / sum_{k=0}^{order-1} M_order(k+1) * exp(2*pi*i*m*k/n)
/// ```
///
/// Returns `|b(m)|^2` for `m = 0..n`.
fn bspline_moduli_squared(n: usize, order: usize) -> Vec<f64> {
    let mut result = vec![0.0; n];
    let n_f = n as f64;

    for m in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for k in 0..order {
            let theta = 2.0 * PI * (m as f64) * (k as f64) / n_f;
            let mk = bspline4((k + 1) as f64);
            re += mk * theta.cos();
            im += mk * theta.sin();
        }
        let denom_sq = re * re + im * im;
        if denom_sq > 1.0e-30 {
            result[m] = 1.0 / denom_sq;
        } else {
            result[m] = 0.0;
        }
    }
    result
}

/// Compute the combined influence function (BC grid) on the reciprocal-space grid.
///
/// For each reciprocal-space point `(mx, my, mz)`:
/// ```text
/// BC(m) = |b_x(mx)|^2 * |b_y(my)|^2 * |b_z(mz)|^2 * exp(-k^2/(4*alpha^2)) / k^2
/// ```
///
/// where `k = 2*pi*(mx/box_x, my/box_y, mz/box_z)` and the `(0,0,0)` point
/// is set to zero (no net-charge correction needed for neutral systems).
fn compute_influence_function(
    dims: [usize; 3],
    box_dims: [f64; 3],
    alpha: f64,
) -> Vec<f64> {
    let [nx, ny, nz] = dims;
    let total = nx * ny * nz;

    let bmod_x = bspline_moduli_squared(nx, PME_ORDER);
    let bmod_y = bspline_moduli_squared(ny, PME_ORDER);
    let bmod_z = bspline_moduli_squared(nz, PME_ORDER);

    let inv_4alpha2 = 1.0 / (4.0 * alpha * alpha);
    let two_pi_over_box = [
        2.0 * PI / box_dims[0],
        2.0 * PI / box_dims[1],
        2.0 * PI / box_dims[2],
    ];

    let mut bc = vec![0.0; total];

    for ix in 0..nx {
        // Wrap frequency index: mx in [0, nx/2] maps to positive freq,
        // (nx/2, nx) maps to negative freq.
        let mx = if ix <= nx / 2 {
            ix as f64
        } else {
            ix as f64 - nx as f64
        };
        let kx = mx * two_pi_over_box[0];

        for iy in 0..ny {
            let my = if iy <= ny / 2 {
                iy as f64
            } else {
                iy as f64 - ny as f64
            };
            let ky = my * two_pi_over_box[1];

            for iz in 0..nz {
                let mz = if iz <= nz / 2 {
                    iz as f64
                } else {
                    iz as f64 - nz as f64
                };
                let kz = mz * two_pi_over_box[2];

                let k2 = kx * kx + ky * ky + kz * kz;

                let idx = ix * ny * nz + iy * nz + iz;

                if k2 < 1.0e-30 {
                    // Zero-frequency term: set to zero (assumes charge neutrality).
                    bc[idx] = 0.0;
                } else {
                    let gaussian = (-k2 * inv_4alpha2).exp();
                    let bmod = bmod_x[ix] * bmod_y[iy] * bmod_z[iz];
                    bc[idx] = bmod * gaussian / k2;
                }
            }
        }
    }

    bc
}

// ============================================================================
// PME Calculator
// ============================================================================

/// Particle Mesh Ewald calculator for reciprocal-space electrostatics.
///
/// Precomputes FFT plans, influence functions, and working buffers. Created
/// once for a given box geometry and reused across force evaluations.
///
/// # Units
///
/// - Charges must be in AMBER internal units (`q * 18.2223`).
/// - Positions in Angstroms.
/// - Returned energies in kcal/mol.
/// - Forces accumulated in kcal/(mol*A).
pub struct PmeCalculator {
    /// Ewald splitting parameter (1/Angstrom).
    alpha: f64,
    /// FFT grid dimensions `[nx, ny, nz]`.
    grid_dims: [usize; 3],
    /// B-spline order (always 4 for this implementation).
    order: usize,
    /// Real-valued charge grid, length `nx * ny * nz`.
    charge_grid: Vec<f64>,
    /// Complex grid for FFT, length `nx * ny * nz`.
    fft_grid: Vec<Complex64>,
    /// Influence function (BC values) on the reciprocal grid.
    bc_grid: Vec<f64>,
    /// FFT plans for forward and inverse transforms.
    fft_plans: FftPlans,
    /// Reusable scratch buffer for FFT operations.
    fft_scratch: Vec<Complex64>,
    /// Box dimensions used to compute the current influence function.
    current_box: [f64; 3],
}

impl PmeCalculator {
    /// Create a new PME calculator.
    ///
    /// # Arguments
    ///
    /// * `box_dims` - Periodic box dimensions `[x, y, z]` in Angstroms.
    /// * `cutoff` - Real-space cutoff in Angstroms.
    /// * `n_atoms` - Number of atoms (used only for capacity hints).
    /// * `charges` - Atomic charges in AMBER internal units. Only used for
    ///   validating array length; charges are passed again at each force call.
    ///
    /// # Panics
    ///
    /// Panics if `charges.len() != n_atoms` or if `cutoff <= 0`.
    pub fn new(
        box_dims: [f64; 3],
        cutoff: f64,
        n_atoms: usize,
        charges: &[f64],
    ) -> Self {
        assert_eq!(
            charges.len(),
            n_atoms,
            "charges length {} != n_atoms {}",
            charges.len(),
            n_atoms,
        );
        assert!(cutoff > 0.0, "cutoff must be positive, got {cutoff}");

        // Ewald splitting parameter
        let alpha = (-EWALD_TOLERANCE.ln()).sqrt() / cutoff;

        // Grid dimensions: choose smooth sizes >= box_dim / grid_spacing
        let grid_dims = [
            next_smooth_size((box_dims[0] / DEFAULT_GRID_SPACING).ceil() as usize),
            next_smooth_size((box_dims[1] / DEFAULT_GRID_SPACING).ceil() as usize),
            next_smooth_size((box_dims[2] / DEFAULT_GRID_SPACING).ceil() as usize),
        ];

        let total = grid_dims[0] * grid_dims[1] * grid_dims[2];

        let fft_plans = FftPlans::new(grid_dims);
        let bc_grid = compute_influence_function(grid_dims, box_dims, alpha);

        log::debug!(
            "PME initialized: alpha={:.6}, grid={:?}, total_points={}",
            alpha,
            grid_dims,
            total,
        );

        PmeCalculator {
            alpha,
            grid_dims,
            order: PME_ORDER,
            charge_grid: vec![0.0; total],
            fft_grid: vec![Complex64::new(0.0, 0.0); total],
            bc_grid,
            fft_plans,
            fft_scratch: Vec::new(),
            current_box: box_dims,
        }
    }

    /// Get the Ewald splitting parameter alpha (1/Angstrom).
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the FFT grid dimensions.
    #[inline]
    pub fn grid_dims(&self) -> [usize; 3] {
        self.grid_dims
    }

    /// Get the B-spline interpolation order (always 4 for this implementation).
    #[inline]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Recompute the influence function if the box dimensions have changed.
    ///
    /// This should be called before `compute_reciprocal_forces` if the box
    /// has changed since the last call (e.g., during NPT simulations).
    pub fn update_box(&mut self, box_dims: &[f64; 3]) {
        if (self.current_box[0] - box_dims[0]).abs() > 1.0e-10
            || (self.current_box[1] - box_dims[1]).abs() > 1.0e-10
            || (self.current_box[2] - box_dims[2]).abs() > 1.0e-10
        {
            self.bc_grid =
                compute_influence_function(self.grid_dims, *box_dims, self.alpha);
            self.current_box = *box_dims;
        }
    }

    /// Compute reciprocal-space energy and forces.
    ///
    /// Forces are **added** to the `forces` array (not overwritten). This allows
    /// accumulation with other force contributions.
    ///
    /// Returns the reciprocal-space energy including the self-energy correction
    /// term, in kcal/mol.
    ///
    /// # Arguments
    ///
    /// * `positions` - Atomic positions in Angstroms.
    /// * `charges` - Atomic charges in AMBER internal units.
    /// * `box_dims` - Current periodic box dimensions `[x, y, z]` in Angstroms.
    /// * `forces` - Mutable force array; reciprocal-space forces are added here.
    ///
    /// # Panics
    ///
    /// Panics if `positions.len() != charges.len()` or
    /// `positions.len() != forces.len()`.
    pub fn compute_reciprocal_forces(
        &mut self,
        positions: &[[f64; 3]],
        charges: &[f64],
        box_dims: &[f64; 3],
        forces: &mut [[f64; 3]],
    ) -> f64 {
        let n_atoms = positions.len();
        assert_eq!(charges.len(), n_atoms);
        assert_eq!(forces.len(), n_atoms);

        // Update influence function if box changed
        self.update_box(box_dims);

        let [nx, ny, nz] = self.grid_dims;
        let total = nx * ny * nz;

        // ----------------------------------------------------------------
        // Step 1: Zero the charge grid and spread charges
        // ----------------------------------------------------------------
        self.charge_grid.iter_mut().for_each(|v| *v = 0.0);
        self.spread_charges(positions, charges, box_dims);

        // ----------------------------------------------------------------
        // Step 2: Copy real charge grid to complex buffer and forward FFT
        // ----------------------------------------------------------------
        for i in 0..total {
            self.fft_grid[i] = Complex64::new(self.charge_grid[i], 0.0);
        }
        fft3d_forward(
            &mut self.fft_grid,
            self.grid_dims,
            &self.fft_plans,
            &mut self.fft_scratch,
        );

        // ----------------------------------------------------------------
        // Step 3: Compute reciprocal energy and apply influence function
        // ----------------------------------------------------------------
        // Following Essmann et al. (1995), the reciprocal energy is:
        //   E_recip = (1/2) * sum_{k!=0} |Q(k)|^2 * theta_rec(k) * |F(k)|^2
        //
        // where theta_rec(k) = (1/(pi*V)) * exp(-pi^2*M^2/alpha^2) / M^2
        //                     = (4*pi/V) * exp(-k^2/(4*alpha^2)) / k^2
        // and k = 2*pi*M.
        //
        // Our bc_grid already contains |F(k)|^2 * exp(-k^2/(4*alpha^2)) / k^2,
        // so:
        //   E_recip = (2*pi/V) * sum_{k!=0} |Q(k)|^2 * bc_grid(k)
        //
        // With AMBER internal charges, the result is directly in kcal/mol.
        let volume = box_dims[0] * box_dims[1] * box_dims[2];
        let energy_prefactor = 2.0 * PI / volume;

        let mut e_recip = 0.0;
        for i in 0..total {
            let q_re = self.fft_grid[i].re;
            let q_im = self.fft_grid[i].im;
            let q_sq = q_re * q_re + q_im * q_im;
            e_recip += q_sq * self.bc_grid[i];

            // Multiply Q(k) by G(k) for the inverse transform.
            // The result after iFFT will be the convolution = potential on grid.
            self.fft_grid[i] = Complex64::new(
                q_re * self.bc_grid[i],
                q_im * self.bc_grid[i],
            );
        }
        e_recip *= energy_prefactor;

        // ----------------------------------------------------------------
        // Step 4: Inverse FFT to get the potential on the real-space grid
        // ----------------------------------------------------------------
        fft3d_inverse(
            &mut self.fft_grid,
            self.grid_dims,
            &self.fft_plans,
            &mut self.fft_scratch,
        );

        // After the inverse FFT (unnormalized in rustfft), fft_grid[r] contains
        // N * theta(r), where theta(r) = (1/N) sum_k bc*Q_hat*exp(ikr).
        //
        // The force derivation via Parseval's theorem shows:
        //   F_i = -q_i * 2 * energy_prefactor * N * sum_r theta(r) * dW_i/dr_i
        //       = -q_i * 2 * energy_prefactor * sum_r fft_grid[r] * dW_i/dr_i
        //
        // So we store phi(r) = 2 * energy_prefactor * fft_grid[r].re on the grid
        // and then F_i = -q_i * sum_r phi(r) * dW_i/dr_i.
        let force_prefactor = 2.0 * energy_prefactor;
        for i in 0..total {
            self.charge_grid[i] = self.fft_grid[i].re * force_prefactor;
        }

        // ----------------------------------------------------------------
        // Step 5: Interpolate forces back to atoms
        // ----------------------------------------------------------------
        self.interpolate_forces(positions, charges, box_dims, forces);

        // ----------------------------------------------------------------
        // Step 6: Self-energy correction
        // ----------------------------------------------------------------
        // E_self = -alpha / sqrt(pi) * sum(q_i^2)
        // With AMBER internal charges, q_i^2 = q_real^2 * 332.0522, so the
        // result is directly in kcal/mol.
        let sum_q2: f64 = charges.iter().map(|&q| q * q).sum();
        let e_self = -self.alpha / PI.sqrt() * sum_q2;

        e_recip + e_self
    }

    /// Spread atomic charges onto the FFT grid using 4th-order B-splines.
    ///
    /// Each atom's charge is distributed over a 4x4x4 neighborhood of grid
    /// points using the product of 1D B-spline weights.
    #[inline]
    fn spread_charges(
        &mut self,
        positions: &[[f64; 3]],
        charges: &[f64],
        box_dims: &[f64; 3],
    ) {
        let [nx, ny, nz] = self.grid_dims;
        let nx_f = nx as f64;
        let ny_f = ny as f64;
        let nz_f = nz as f64;

        for (atom_idx, pos) in positions.iter().enumerate() {
            let q = charges[atom_idx];
            if q.abs() < 1.0e-20 {
                continue;
            }

            // Fractional grid coordinates
            let wx = pos[0] * nx_f / box_dims[0];
            let wy = pos[1] * ny_f / box_dims[1];
            let wz = pos[2] * nz_f / box_dims[2];

            let mut sx = [0.0; PME_ORDER];
            let mut sy = [0.0; PME_ORDER];
            let mut sz = [0.0; PME_ORDER];
            let mut dx = [0.0; PME_ORDER];
            let mut dy = [0.0; PME_ORDER];
            let mut dz = [0.0; PME_ORDER];

            let ix0 = compute_bspline_1d(wx, nx, &mut sx, &mut dx);
            let iy0 = compute_bspline_1d(wy, ny, &mut sy, &mut dy);
            let iz0 = compute_bspline_1d(wz, nz, &mut sz, &mut dz);

            for kx in 0..PME_ORDER {
                let gx = (ix0 + kx) % nx;
                let qsx = q * sx[kx];
                for ky in 0..PME_ORDER {
                    let gy = (iy0 + ky) % ny;
                    let qsxy = qsx * sy[ky];
                    for kz in 0..PME_ORDER {
                        let gz = (iz0 + kz) % nz;
                        let weight = qsxy * sz[kz];
                        self.charge_grid[gx * ny * nz + gy * nz + gz] += weight;
                    }
                }
            }
        }
    }

    /// Interpolate forces from the potential grid back to atoms.
    ///
    /// The force on each atom is computed as the negative gradient of the
    /// potential at the atom's position, obtained by differentiating the
    /// B-spline interpolation. Forces are added to the existing values in
    /// the `forces` array.
    #[inline]
    fn interpolate_forces(
        &self,
        positions: &[[f64; 3]],
        charges: &[f64],
        box_dims: &[f64; 3],
        forces: &mut [[f64; 3]],
    ) {
        let [nx, ny, nz] = self.grid_dims;
        let nx_f = nx as f64;
        let ny_f = ny as f64;
        let nz_f = nz as f64;

        // Scale factors for converting grid-space derivatives to real-space:
        // d/dx = d/dw * dw/dx = d/dw * (n / box_dim)
        let scale_x = nx_f / box_dims[0];
        let scale_y = ny_f / box_dims[1];
        let scale_z = nz_f / box_dims[2];

        for (atom_idx, pos) in positions.iter().enumerate() {
            let q = charges[atom_idx];
            if q.abs() < 1.0e-20 {
                continue;
            }

            let wx = pos[0] * scale_x;
            let wy = pos[1] * scale_y;
            let wz = pos[2] * scale_z;

            let mut sx = [0.0; PME_ORDER];
            let mut sy = [0.0; PME_ORDER];
            let mut sz = [0.0; PME_ORDER];
            let mut dx = [0.0; PME_ORDER];
            let mut dy = [0.0; PME_ORDER];
            let mut dz = [0.0; PME_ORDER];

            let ix0 = compute_bspline_1d(wx, nx, &mut sx, &mut dx);
            let iy0 = compute_bspline_1d(wy, ny, &mut sy, &mut dy);
            let iz0 = compute_bspline_1d(wz, nz, &mut sz, &mut dz);

            let mut fx = 0.0;
            let mut fy = 0.0;
            let mut fz = 0.0;

            for kx in 0..PME_ORDER {
                let gx = (ix0 + kx) % nx;
                for ky in 0..PME_ORDER {
                    let gy = (iy0 + ky) % ny;
                    for kz in 0..PME_ORDER {
                        let gz = (iz0 + kz) % nz;
                        let theta = self.charge_grid[gx * ny * nz + gy * nz + gz];

                        // Force = -q * grad(theta)
                        // grad_x(theta) = sum theta(grid) * dM/du_x * M(u_y) * M(u_z) * (nx/box_x)
                        fx += theta * dx[kx] * sy[ky] * sz[kz];
                        fy += theta * sx[kx] * dy[ky] * sz[kz];
                        fz += theta * sx[kx] * sy[ky] * dz[kz];
                    }
                }
            }

            // Apply charge and scale factors.
            // The negative sign gives the force from F = -q * grad(theta).
            forces[atom_idx][0] -= q * fx * scale_x;
            forces[atom_idx][1] -= q * fy * scale_y;
            forces[atom_idx][2] -= q * fz * scale_z;
        }
    }

    /// Compute the self-energy correction term.
    ///
    /// This is the correction for the artificial self-interaction introduced
    /// by the Ewald sum:
    ///
    /// ```text
    /// E_self = -alpha / sqrt(pi) * sum_i(q_i^2)
    /// ```
    ///
    /// With AMBER internal charges, the result is directly in kcal/mol.
    ///
    /// # Arguments
    ///
    /// * `charges` - Atomic charges in AMBER internal units.
    pub fn self_energy_correction(&self, charges: &[f64]) -> f64 {
        let sum_q2: f64 = charges.iter().map(|&q| q * q).sum();
        -self.alpha / PI.sqrt() * sum_q2
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // ------------------------------------------------------------------
    // Utility function tests
    // ------------------------------------------------------------------

    #[test]
    fn test_next_smooth_size() {
        assert_eq!(next_smooth_size(1), 2);
        assert_eq!(next_smooth_size(2), 2);
        assert_eq!(next_smooth_size(3), 3);
        assert_eq!(next_smooth_size(4), 4);
        assert_eq!(next_smooth_size(11), 12); // 12 = 2^2 * 3
        assert_eq!(next_smooth_size(13), 14); // 14 = 2 * 7
        assert_eq!(next_smooth_size(17), 18); // 18 = 2 * 3^2
        assert_eq!(next_smooth_size(23), 24); // 24 = 2^3 * 3
        assert_eq!(next_smooth_size(29), 30); // 30 = 2 * 3 * 5
        assert_eq!(next_smooth_size(31), 32); // 32 = 2^5
    }

    #[test]
    fn test_is_smooth() {
        assert!(is_smooth(1));
        assert!(is_smooth(2));
        assert!(is_smooth(6));   // 2 * 3
        assert!(is_smooth(210)); // 2 * 3 * 5 * 7
        assert!(!is_smooth(11));
        assert!(!is_smooth(13));
        assert!(!is_smooth(0));
    }

    // ------------------------------------------------------------------
    // B-spline tests
    // ------------------------------------------------------------------

    #[test]
    fn test_bspline4_partition_of_unity() {
        // The Cardinal B-spline of order 4 satisfies sum_{k in Z} M_4(u - k) = 1
        // for any u. M_4 has support [0, 4), so for a given u we need to sum
        // over all integers k where (u - k) falls in [0, 4).
        for i in 0..100 {
            let u = 2.0 + (i as f64) * 0.01;
            // Sum over enough integer translates to cover the support.
            // For u in [2, 3], the contributing k values are -1, 0, 1, 2.
            let sum: f64 = (-4..8).map(|k| bspline4(u - k as f64)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Partition of unity failed at u={}: sum={}",
                u,
                sum
            );
        }
    }

    #[test]
    fn test_bspline4_nonnegative() {
        for i in 0..400 {
            let u = i as f64 * 0.01;
            assert!(
                bspline4(u) >= -1e-15,
                "B-spline negative at u={}: {}",
                u,
                bspline4(u)
            );
        }
    }

    #[test]
    fn test_bspline4_boundary_values() {
        assert!((bspline4(0.0) - 0.0).abs() < TOL);
        assert!((bspline4(4.0) - 0.0).abs() < TOL); // M_4(4) = 0 (open interval)
        // M_4(2) = maximum, should be 4/6 = 2/3
        assert!(
            (bspline4(2.0) - 2.0 / 3.0).abs() < TOL,
            "M_4(2) = {} expected {}",
            bspline4(2.0),
            2.0 / 3.0,
        );
    }

    #[test]
    fn test_bspline4_symmetry() {
        // M_4(u) is symmetric about u=2: M_4(2+t) = M_4(2-t)
        for i in 0..20 {
            let t = i as f64 * 0.1;
            let a = bspline4(2.0 + t);
            let b = bspline4(2.0 - t);
            assert!(
                (a - b).abs() < 1e-12,
                "Symmetry failed: M4({})={} vs M4({})={}",
                2.0 + t,
                a,
                2.0 - t,
                b,
            );
        }
    }

    #[test]
    fn test_bspline4_continuity() {
        // Check continuity at knot points u = 1, 2, 3 by comparing left and right limits.
        for &knot in &[1.0, 2.0, 3.0] {
            let eps = 1e-8;
            let left = bspline4(knot - eps);
            let right = bspline4(knot);
            assert!(
                (left - right).abs() < 1e-6,
                "Discontinuity at u={}: left={} right={}",
                knot,
                left,
                right,
            );
        }
    }

    #[test]
    fn test_bspline4_deriv_numerical() {
        // Verify the analytical derivative against numerical differentiation.
        let eps = 1e-6;
        for i in 1..39 {
            let u = i as f64 * 0.1;
            let numerical = (bspline4(u + eps) - bspline4(u - eps)) / (2.0 * eps);
            let analytical = bspline4_deriv(u);
            assert!(
                (numerical - analytical).abs() < 1e-4,
                "Derivative mismatch at u={}: numerical={} analytical={}",
                u,
                numerical,
                analytical,
            );
        }
    }

    #[test]
    fn test_bspline4_deriv_continuity() {
        // Derivative should be continuous at knots 1, 2, 3.
        for &knot in &[1.0, 2.0, 3.0] {
            let eps = 1e-8;
            let left = bspline4_deriv(knot - eps);
            let right = bspline4_deriv(knot);
            assert!(
                (left - right).abs() < 1e-5,
                "Derivative discontinuity at u={}: left={} right={}",
                knot,
                left,
                right,
            );
        }
    }

    // ------------------------------------------------------------------
    // B-spline moduli tests
    // ------------------------------------------------------------------

    #[test]
    fn test_bspline_moduli_squared_positive() {
        let n = 32;
        let bmod = bspline_moduli_squared(n, PME_ORDER);
        assert_eq!(bmod.len(), n);
        // All entries should be non-negative (they are |b|^(-2))
        for (i, &v) in bmod.iter().enumerate() {
            assert!(v >= 0.0, "Negative modulus at index {}: {}", i, v);
        }
    }

    // ------------------------------------------------------------------
    // compute_bspline_1d tests
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_bspline_1d_weights_sum_to_one() {
        // For any fractional coordinate, the B-spline weights should sum to 1.
        let grid_dim = 32;
        for i in 0..100 {
            let w = i as f64 * 0.37; // Arbitrary fractional coords
            let mut sv = [0.0; PME_ORDER];
            let mut sd = [0.0; PME_ORDER];
            let _ = compute_bspline_1d(w, grid_dim, &mut sv, &mut sd);
            let sum: f64 = sv.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Weights don't sum to 1 at w={}: sum={}",
                w,
                sum,
            );
        }
    }

    // ------------------------------------------------------------------
    // 3D FFT tests
    // ------------------------------------------------------------------

    #[test]
    fn test_fft3d_roundtrip() {
        // Forward then inverse FFT should recover the original data (up to
        // normalization by N).
        let dims = [4, 6, 8];
        let total = dims[0] * dims[1] * dims[2];
        let plans = FftPlans::new(dims);
        let mut scratch = Vec::new();

        // Create a known signal
        let mut data: Vec<Complex64> = (0..total)
            .map(|i| Complex64::new((i as f64) * 0.1, (i as f64) * -0.05))
            .collect();
        let original = data.clone();

        fft3d_forward(&mut data, dims, &plans, &mut scratch);
        fft3d_inverse(&mut data, dims, &plans, &mut scratch);

        // After forward+inverse, result = original * N
        let n = total as f64;
        for i in 0..total {
            let expected_re = original[i].re * n;
            let expected_im = original[i].im * n;
            assert!(
                (data[i].re - expected_re).abs() < 1e-8,
                "FFT roundtrip real mismatch at {}: {} vs {}",
                i,
                data[i].re,
                expected_re,
            );
            assert!(
                (data[i].im - expected_im).abs() < 1e-8,
                "FFT roundtrip imag mismatch at {}: {} vs {}",
                i,
                data[i].im,
                expected_im,
            );
        }
    }

    #[test]
    fn test_fft3d_parseval() {
        // Parseval's theorem: sum |x|^2 = (1/N) sum |X|^2
        let dims = [4, 4, 4];
        let total = dims[0] * dims[1] * dims[2];
        let plans = FftPlans::new(dims);
        let mut scratch = Vec::new();

        let mut data: Vec<Complex64> = (0..total)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let energy_time: f64 = data.iter().map(|c| c.norm_sqr()).sum();

        fft3d_forward(&mut data, dims, &plans, &mut scratch);

        let energy_freq: f64 = data.iter().map(|c| c.norm_sqr()).sum();

        let n = total as f64;
        assert!(
            (energy_time - energy_freq / n).abs() < 1e-8,
            "Parseval failed: time={} freq/N={}",
            energy_time,
            energy_freq / n,
        );
    }

    // ------------------------------------------------------------------
    // Influence function tests
    // ------------------------------------------------------------------

    #[test]
    fn test_influence_function_zero_at_origin() {
        let dims = [8, 8, 8];
        let box_dims = [10.0, 10.0, 10.0];
        let alpha = 0.3;
        let bc = compute_influence_function(dims, box_dims, alpha);
        assert!(bc[0].abs() < 1e-30, "BC at origin should be zero");
    }

    #[test]
    fn test_influence_function_symmetry() {
        // For a cubic box and cubic grid, G should have cubic symmetry.
        let n = 8;
        let dims = [n, n, n];
        let box_dims = [10.0, 10.0, 10.0];
        let alpha = 0.3;
        let bc = compute_influence_function(dims, box_dims, alpha);

        // G(1,0,0) should equal G(0,1,0) and G(0,0,1)
        let g100 = bc[1 * n * n + 0 * n + 0];
        let g010 = bc[0 * n * n + 1 * n + 0];
        let g001 = bc[0 * n * n + 0 * n + 1];
        assert!(
            (g100 - g010).abs() < 1e-12,
            "Cubic symmetry violated: G(1,0,0)={} vs G(0,1,0)={}",
            g100,
            g010,
        );
        assert!(
            (g100 - g001).abs() < 1e-12,
            "Cubic symmetry violated: G(1,0,0)={} vs G(0,0,1)={}",
            g100,
            g001,
        );
    }

    #[test]
    fn test_influence_function_decay() {
        // Higher-frequency components should have smaller influence
        // (Gaussian decay dominates at large k).
        let n = 16;
        let dims = [n, n, n];
        let box_dims = [20.0, 20.0, 20.0];
        let alpha = 0.3;
        let bc = compute_influence_function(dims, box_dims, alpha);

        let g1 = bc[1 * n * n]; // k = (1,0,0)
        let g2 = bc[2 * n * n]; // k = (2,0,0)
        let g3 = bc[3 * n * n]; // k = (3,0,0)

        assert!(
            g1 > g2,
            "Influence should decay: G(1,0,0)={} > G(2,0,0)={}",
            g1,
            g2,
        );
        assert!(
            g2 > g3,
            "Influence should decay: G(2,0,0)={} > G(3,0,0)={}",
            g2,
            g3,
        );
    }

    // ------------------------------------------------------------------
    // PmeCalculator construction tests
    // ------------------------------------------------------------------

    #[test]
    fn test_pme_calculator_new() {
        let box_dims = [30.0, 30.0, 30.0];
        let cutoff = 10.0;
        let n_atoms = 3;
        let charges = vec![1.0, -1.0, 0.5];

        let pme = PmeCalculator::new(box_dims, cutoff, n_atoms, &charges);

        // Alpha should be sqrt(-ln(1e-5)) / 10
        let expected_alpha = (-EWALD_TOLERANCE.ln()).sqrt() / cutoff;
        assert!(
            (pme.alpha() - expected_alpha).abs() < 1e-12,
            "alpha mismatch: {} vs {}",
            pme.alpha(),
            expected_alpha,
        );

        // Grid dims should be smooth numbers >= 30
        for &d in &pme.grid_dims() {
            assert!(d >= 30, "Grid dim {} < 30", d);
            assert!(is_smooth(d), "Grid dim {} is not smooth", d);
        }
    }

    #[test]
    #[should_panic(expected = "charges length")]
    fn test_pme_calculator_new_mismatched_charges() {
        let charges = vec![1.0, -1.0]; // 2 charges but 3 atoms
        PmeCalculator::new([30.0, 30.0, 30.0], 10.0, 3, &charges);
    }

    #[test]
    #[should_panic(expected = "cutoff must be positive")]
    fn test_pme_calculator_new_zero_cutoff() {
        let charges = vec![1.0];
        PmeCalculator::new([30.0, 30.0, 30.0], 0.0, 1, &charges);
    }

    // ------------------------------------------------------------------
    // Self-energy correction tests
    // ------------------------------------------------------------------

    #[test]
    fn test_self_energy_correction() {
        let box_dims = [30.0, 30.0, 30.0];
        let cutoff = 10.0;
        // Use AMBER internal charges: q_amber = q_real * 18.2223
        // For a single unit charge: q_amber = 18.2223
        let q_amber = 18.2223;
        let charges = vec![q_amber];
        let pme = PmeCalculator::new(box_dims, cutoff, 1, &charges);

        let e_self = pme.self_energy_correction(&charges);

        // E_self = -alpha/sqrt(pi) * q_amber^2
        let expected = -pme.alpha() / PI.sqrt() * q_amber * q_amber;
        assert!(
            (e_self - expected).abs() < 1e-10,
            "Self energy: {} vs expected {}",
            e_self,
            expected,
        );

        // Should be negative (removing self-interaction)
        assert!(e_self < 0.0, "Self energy should be negative");
    }

    #[test]
    fn test_self_energy_zero_charges() {
        let charges = vec![0.0, 0.0, 0.0];
        let pme = PmeCalculator::new([30.0; 3], 10.0, 3, &charges);
        let e_self = pme.self_energy_correction(&charges);
        assert!(e_self.abs() < 1e-20, "Zero charges => zero self energy");
    }

    // ------------------------------------------------------------------
    // Charge spreading tests
    // ------------------------------------------------------------------

    #[test]
    fn test_charge_spreading_conservation() {
        // Total charge on the grid should equal the sum of atomic charges.
        let box_dims = [10.0, 10.0, 10.0];
        let charges = vec![1.5, -0.7, 0.3];
        let positions = [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 1.0, 3.0]];
        let mut pme = PmeCalculator::new(box_dims, 5.0, 3, &charges);

        pme.charge_grid.iter_mut().for_each(|v| *v = 0.0);
        pme.spread_charges(&positions, &charges, &box_dims);

        let grid_total: f64 = pme.charge_grid.iter().sum();
        let charge_total: f64 = charges.iter().sum();

        assert!(
            (grid_total - charge_total).abs() < 1e-10,
            "Charge not conserved: grid={} atoms={}",
            grid_total,
            charge_total,
        );
    }

    #[test]
    fn test_charge_spreading_nonnegative_for_positive_charge() {
        // A single positive charge should produce only non-negative grid values.
        let box_dims = [10.0, 10.0, 10.0];
        let charges = vec![1.0];
        let positions = [[5.0, 5.0, 5.0]];
        let mut pme = PmeCalculator::new(box_dims, 5.0, 1, &charges);

        pme.charge_grid.iter_mut().for_each(|v| *v = 0.0);
        pme.spread_charges(&positions, &charges, &box_dims);

        for (i, &v) in pme.charge_grid.iter().enumerate() {
            assert!(
                v >= -1e-15,
                "Negative grid value at index {}: {}",
                i,
                v,
            );
        }
    }

    #[test]
    fn test_charge_spreading_periodic_wrapping() {
        // An atom near the box edge should spread charge that wraps around.
        let box_dims = [10.0, 10.0, 10.0];
        let charges = vec![1.0];
        // Place atom very close to box boundary
        let positions = [[0.1, 0.1, 0.1]];
        let mut pme = PmeCalculator::new(box_dims, 5.0, 1, &charges);

        pme.charge_grid.iter_mut().for_each(|v| *v = 0.0);
        pme.spread_charges(&positions, &charges, &box_dims);

        // Verify charge conservation still holds
        let grid_total: f64 = pme.charge_grid.iter().sum();
        assert!(
            (grid_total - 1.0).abs() < 1e-10,
            "Charge not conserved at boundary: {}",
            grid_total,
        );

        // Verify some charge landed on high-index grid points (wrapping)
        let [nx, ny, nz] = pme.grid_dims;
        let last_x_slice_sum: f64 = (0..ny)
            .flat_map(|iy| (0..nz).map(move |iz| (iy, iz)))
            .map(|(iy, iz)| pme.charge_grid[(nx - 1) * ny * nz + iy * nz + iz])
            .sum();
        assert!(
            last_x_slice_sum > 0.0,
            "No charge wrapped to last x slice",
        );
    }

    // ------------------------------------------------------------------
    // Full reciprocal energy/force tests
    // ------------------------------------------------------------------

    #[test]
    fn test_neutral_system_energy_finite() {
        // A simple neutral dipole should produce a finite energy.
        let box_dims = [20.0, 20.0, 20.0];
        let cutoff = 8.0;
        // Two opposite charges separated by 3 Angstroms
        let q = 18.2223; // 1 elementary charge in AMBER units
        let charges = vec![q, -q];
        let positions = [[10.0, 10.0, 10.0], [10.0, 10.0, 13.0]];
        let mut forces = [[0.0; 3]; 2];

        let mut pme = PmeCalculator::new(box_dims, cutoff, 2, &charges);
        let energy = pme.compute_reciprocal_forces(&positions, &charges, &box_dims, &mut forces);

        // Energy should be finite and not NaN
        assert!(energy.is_finite(), "Energy is not finite: {}", energy);

        // For a dipole in PME, the reciprocal energy should be negative
        // (attractive interaction dominates in reciprocal space after self-correction).
    }

    #[test]
    fn test_zero_charges_zero_energy() {
        let box_dims = [20.0, 20.0, 20.0];
        let charges = vec![0.0, 0.0, 0.0];
        let positions = [[5.0, 5.0, 5.0], [10.0, 10.0, 10.0], [15.0, 15.0, 15.0]];
        let mut forces = [[0.0; 3]; 3];

        let mut pme = PmeCalculator::new(box_dims, 8.0, 3, &charges);
        let energy = pme.compute_reciprocal_forces(&positions, &charges, &box_dims, &mut forces);

        assert!(
            energy.abs() < 1e-15,
            "Zero charges should give zero energy: {}",
            energy,
        );

        for (i, f) in forces.iter().enumerate() {
            for (d, &fval) in f.iter().enumerate() {
                assert!(
                    fval.abs() < 1e-15,
                    "Zero charges should give zero force: atom {} dim {} = {}",
                    i,
                    d,
                    fval,
                );
            }
        }
    }

    #[test]
    fn test_forces_are_accumulated() {
        // Forces should be ADDED to existing values, not overwritten.
        let box_dims = [20.0, 20.0, 20.0];
        let q = 18.2223;
        let charges = vec![q, -q];
        let positions = [[10.0, 10.0, 10.0], [10.0, 10.0, 13.0]];

        let mut forces1 = [[0.0; 3]; 2];
        let mut pme = PmeCalculator::new(box_dims, 8.0, 2, &charges);
        let _ = pme.compute_reciprocal_forces(&positions, &charges, &box_dims, &mut forces1);

        // Now start with non-zero forces
        let mut forces2 = [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]];
        let _ = pme.compute_reciprocal_forces(&positions, &charges, &box_dims, &mut forces2);

        for i in 0..2 {
            for d in 0..3 {
                let base = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0][i * 3 + d];
                let expected = base + forces1[i][d];
                assert!(
                    (forces2[i][d] - expected).abs() < 1e-10,
                    "Forces not accumulated: atom {} dim {}: {} vs expected {}",
                    i,
                    d,
                    forces2[i][d],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_forces_newton_third_law_dipole() {
        // For a two-atom system, forces should sum to zero (Newton's third law)
        // within numerical precision.
        let box_dims = [20.0, 20.0, 20.0];
        let q = 18.2223;
        let charges = vec![q, -q];
        let positions = [[10.0, 10.0, 10.0], [10.0, 10.0, 13.0]];
        let mut forces = [[0.0; 3]; 2];

        let mut pme = PmeCalculator::new(box_dims, 8.0, 2, &charges);
        let _ = pme.compute_reciprocal_forces(&positions, &charges, &box_dims, &mut forces);

        for d in 0..3 {
            let total = forces[0][d] + forces[1][d];
            assert!(
                total.abs() < 1e-6,
                "Newton's 3rd law violated in dim {}: F1={} F2={} sum={}",
                d,
                forces[0][d],
                forces[1][d],
                total,
            );
        }
    }

    #[test]
    fn test_forces_along_separation_axis() {
        // For two charges separated along z, the force should be primarily along z.
        let box_dims = [20.0, 20.0, 20.0];
        let q = 18.2223;
        let charges = vec![q, -q];
        let positions = [[10.0, 10.0, 9.0], [10.0, 10.0, 11.0]];
        let mut forces = [[0.0; 3]; 2];

        let mut pme = PmeCalculator::new(box_dims, 8.0, 2, &charges);
        let _ = pme.compute_reciprocal_forces(&positions, &charges, &box_dims, &mut forces);

        // Z-force should dominate
        let fz_mag = forces[0][2].abs();
        let fx_mag = forces[0][0].abs();
        let fy_mag = forces[0][1].abs();

        assert!(
            fz_mag > fx_mag * 100.0 || fx_mag < 1e-10,
            "x-force should be negligible: fx={} fz={}",
            fx_mag,
            fz_mag,
        );
        assert!(
            fz_mag > fy_mag * 100.0 || fy_mag < 1e-10,
            "y-force should be negligible: fy={} fz={}",
            fy_mag,
            fz_mag,
        );
    }

    #[test]
    fn test_energy_translational_invariance() {
        // Energy should not depend on where in the box the system is placed
        // (periodic boundary conditions).
        let box_dims = [20.0, 20.0, 20.0];
        let q = 18.2223;
        let charges = vec![q, -q];

        let mut forces = [[0.0; 3]; 2];
        let mut pme = PmeCalculator::new(box_dims, 8.0, 2, &charges);

        // Position 1: centered
        let positions1 = [[10.0, 10.0, 10.0], [10.0, 10.0, 13.0]];
        let e1 = pme.compute_reciprocal_forces(&positions1, &charges, &box_dims, &mut forces);

        // Position 2: shifted
        forces = [[0.0; 3]; 2];
        let positions2 = [[3.0, 7.0, 2.0], [3.0, 7.0, 5.0]];
        let e2 = pme.compute_reciprocal_forces(&positions2, &charges, &box_dims, &mut forces);

        // Energies should be nearly identical
        assert!(
            (e1 - e2).abs() < 0.01 * e1.abs().max(1.0),
            "Translational invariance violated: e1={} e2={}",
            e1,
            e2,
        );
    }

    #[test]
    fn test_update_box() {
        // Changing box dimensions should update the influence function.
        let charges = vec![1.0, -1.0];
        let mut pme = PmeCalculator::new([20.0, 20.0, 20.0], 8.0, 2, &charges);

        let old_bc_sum: f64 = pme.bc_grid.iter().sum();

        pme.update_box(&[25.0, 25.0, 25.0]);

        let new_bc_sum: f64 = pme.bc_grid.iter().sum();

        // The sums should differ since the box changed.
        assert!(
            (old_bc_sum - new_bc_sum).abs() > 1e-10,
            "BC grid did not change when box changed",
        );
    }

    #[test]
    fn test_update_box_no_change() {
        // If box hasn't changed, BC grid should remain identical.
        let charges = vec![1.0, -1.0];
        let mut pme = PmeCalculator::new([20.0, 20.0, 20.0], 8.0, 2, &charges);

        let old_bc: Vec<f64> = pme.bc_grid.clone();

        pme.update_box(&[20.0, 20.0, 20.0]);

        assert_eq!(pme.bc_grid, old_bc, "BC grid changed when box was identical");
    }

    // ------------------------------------------------------------------
    // Numerical gradient test (forces vs finite-difference energy)
    // ------------------------------------------------------------------

    #[test]
    fn test_forces_vs_finite_difference() {
        // Verify that the analytical forces match numerical finite differences
        // of the energy. This is the most rigorous correctness test.
        let box_dims = [20.0, 20.0, 20.0];
        let q = 18.2223;
        let charges = vec![q, -q, 0.5 * q];
        let mut pme = PmeCalculator::new(box_dims, 8.0, 3, &charges);

        let positions = [[8.0, 10.0, 10.0], [12.0, 10.0, 10.0], [10.0, 13.0, 10.0]];

        // Compute analytical forces
        let mut forces = [[0.0; 3]; 3];
        let _ = pme.compute_reciprocal_forces(&positions, &charges, &box_dims, &mut forces);

        // Numerical gradient via central differences
        let h = 1e-4;
        for atom in 0..3 {
            for dim in 0..3 {
                let mut pos_plus = positions;
                let mut pos_minus = positions;
                pos_plus[atom][dim] += h;
                pos_minus[atom][dim] -= h;

                let mut f_dummy = [[0.0; 3]; 3];
                let e_plus = pme.compute_reciprocal_forces(
                    &pos_plus,
                    &charges,
                    &box_dims,
                    &mut f_dummy,
                );
                f_dummy = [[0.0; 3]; 3];
                let e_minus = pme.compute_reciprocal_forces(
                    &pos_minus,
                    &charges,
                    &box_dims,
                    &mut f_dummy,
                );

                let numerical_force = -(e_plus - e_minus) / (2.0 * h);
                let analytical_force = forces[atom][dim];

                // Allow some tolerance due to finite grid interpolation
                let tol = 0.05 * analytical_force.abs().max(0.1);
                assert!(
                    (numerical_force - analytical_force).abs() < tol,
                    "Force mismatch atom {} dim {}: analytical={:.6} numerical={:.6} diff={:.6}",
                    atom,
                    dim,
                    analytical_force,
                    numerical_force,
                    (numerical_force - analytical_force).abs(),
                );
            }
        }
    }
}
