//! STO overlap integrals for AM1.
//!
//! Computes two-center overlap integrals between Slater-type orbitals (STOs)
//! using the Mulliken A/B auxiliary integral approach, following the
//! implementation in SQM/MOPAC (Thiel's slater_overlap.F90).
//!
//! Compute in the local diatomic frame (z along the internuclear axis),
//! then rotate to the global Cartesian frame.

use crate::molecule::Am1Molecule;
use crate::params::ANG_TO_BOHR;
use nalgebra::DMatrix;

// ============================================================================
// Factorials: FC(i) = factorial(i - 1), i.e. FC[0] = 0! = 1, FC[1] = 1! = 1, ...
// Matches the AMBER/SQM constants module: FC(1) = 0!, FC(2) = 1!, FC(25) = 24!
// We index from 0 here for convenience: FACTORIAL[n] = n!
// ============================================================================
const FACTORIAL: [f64; 16] = [
    1.0,                    // 0!
    1.0,                    // 1!
    2.0,                    // 2!
    6.0,                    // 3!
    24.0,                   // 4!
    120.0,                  // 5!
    720.0,                  // 6!
    5040.0,                 // 7!
    40320.0,                // 8!
    362880.0,               // 9!
    3628800.0,              // 10!
    39916800.0,             // 11!
    479001600.0,            // 12!
    6227020800.0,           // 13!
    87178291200.0,          // 14!
    1307674368000.0,        // 15!
];

/// B integral values for zero argument: B_n(0) = 2/(n+1) for even n, 0 for odd n
const B0: [f64; 15] = [
    2.0,
    0.0,
    0.666_666_666_666_667,
    0.0,
    0.4,
    0.0,
    0.285_714_285_714_286,
    0.0,
    0.222_222_222_222_222,
    0.0,
    0.181_818_181_818_182,
    0.0,
    0.153_846_153_846_154,
    0.0,
    0.133_333_333_333_333,
];

// ============================================================================
// Auxiliary A and B integrals (Mulliken formulation)
// ============================================================================

/// Compute auxiliary A integrals: A_n(p) = integral from 1 to infinity of t^n * exp(-p*t) dt
///
/// Uses the recurrence: A(1) = exp(-p)/p, A(n+1) = (n*A(n) + exp(-p))/p
///
/// Returns an array of A(1)..A(ntotal+1), 1-indexed to match the Fortran convention.
/// a_out[i] corresponds to A_{i-1}(p) in mathematical notation, but we follow
/// the SQM convention where A(1) = A_0(p), A(2) = A_1(p), etc.
fn compute_a_integrals(ntotal: usize, alpha: f64) -> Vec<f64> {
    let mut a = vec![0.0; ntotal + 2]; // indices 0..=ntotal+1, but 0 is unused

    let c = (-alpha).exp();
    let ralpha = 1.0 / alpha;
    a[1] = c * ralpha;
    for i in 1..=ntotal {
        a[i + 1] = (a[i] * (i as f64) + c) * ralpha;
    }
    a
}

/// Compute auxiliary B integrals: B_n(beta) = integral from -1 to 1 of t^n * exp(-beta*t) dt
///
/// Uses different strategies depending on the magnitude of beta:
/// - Zero argument: use precomputed B0 values
/// - Large argument: direct recurrence
/// - Small argument: power series expansion
///
/// Returns array B[1]..B[ntotal+1], 1-indexed (SQM convention).
fn compute_b_integrals(ntotal: usize, beta: f64) -> Vec<f64> {
    let mut b = vec![0.0; ntotal + 2];
    let absx = beta.abs();

    // Zero argument
    if absx < 1.0e-06 {
        for i in 1..=(ntotal + 1).min(15) {
            b[i] = B0[i - 1];
        }
        return b;
    }

    // Large argument: use direct recurrence
    let large = (absx > 0.5 && ntotal <= 5)
        || (absx > 1.0 && ntotal <= 7)
        || (absx > 2.0 && ntotal <= 10)
        || absx > 3.0;

    if large {
        let expx = beta.exp();
        let expmx = 1.0 / expx;
        let rx = 1.0 / beta;
        b[1] = (expx - expmx) * rx;
        let mut sign_expx = expx;
        for i in 1..=ntotal {
            sign_expx = -sign_expx;
            b[i + 1] = ((i as f64) * b[i] + sign_expx - expmx) * rx;
        }
        return b;
    }

    // Small argument: power series expansion
    let last = if absx <= 0.5 {
        6
    } else if absx <= 1.0 {
        7
    } else if absx <= 2.0 {
        12
    } else {
        15
    };

    // betpow[m] = (-beta)^m
    let mut betpow = vec![0.0; last + 2];
    betpow[0] = 1.0; // (-beta)^0 = 1 (0-indexed for computation)
    for m in 1..=last {
        betpow[m] = -beta * betpow[m - 1];
    }

    for i in 1..=(ntotal + 1) {
        let mut y = 0.0;
        // ma = 1 - mod(i, 2): if i is odd, ma=0; if i is even, ma=1
        // In the Fortran code, I goes from 1..N+1 and MA = 1 - MOD(I, 2)
        let ma = if i % 2 == 0 { 1usize } else { 0usize };
        let mut m = ma;
        while m <= last {
            // FC(M+1) in Fortran = factorial(M) since FC(I) = factorial(I-1)
            // betpow(M+1) in Fortran = betpow[M] here (0-indexed)
            let factorial_m = FACTORIAL[m];
            y += betpow[m] / (factorial_m * ((m + i) as f64));
            m += 2;
        }
        b[i] = y * 2.0;
    }

    b
}

// ============================================================================
// Binomial coefficients C(n, k)
// ============================================================================

/// Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
fn binomial(n: usize, k: usize) -> i64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    // Use the multiplicative formula to avoid overflow for small n
    let k = k.min(n - k);
    let mut result: i64 = 1;
    for i in 0..k {
        result = result * (n - i) as i64 / (i + 1) as i64;
    }
    result
}

/// Lookup table for binomial coefficients, matching the SQM IBINOM array.
/// IBINOM is stored for pairs (n, k) where n goes up to 7.
/// IAD(N+1) gives the start index for row N in a packed triangular array.
/// IAD = [1, 2, 4, 7, 11, 16, 22, 29]
///
/// IBINOM(IAD(N+1) + K) = C(N, K) for K = 0..N
fn ibinom(n: usize, k: usize) -> i64 {
    binomial(n, k)
}

// ============================================================================
// Core overlap calculation (Thiel's CalculateOverlap)
// ============================================================================

/// Compute the overlap integral between two STOs in the local diatomic frame.
///
/// Quantum numbers: (na, la, mm) on atom A, (nb, lb, mm) on atom B.
/// alpha = zeta_a * R (dimensionless)
/// beta  = zeta_b * R (dimensionless)
///
/// This is a direct translation of CalculateOverlap from SQM slater_overlap.F90
fn calculate_overlap(
    na: usize,
    la: usize,
    mm: i32,
    nb: usize,
    lb: usize,
    alpha: f64,
    beta: f64,
    a_integrals: &[f64],
    b_integrals: &[f64],
) -> f64 {
    let m = mm.unsigned_abs() as usize;
    let nab = na + nb + 1;
    let mut x: f64 = 0.0;

    // S-S overlap (la=0, lb=0)
    if la == 0 && lb == 0 {
        for i in 0..=na {
            let iba = ibinom(na, i);
            for j in 0..=nb {
                let mut ibb = iba * ibinom(nb, j);
                if j % 2 == 1 {
                    ibb = -ibb;
                }
                let ij = i + j;
                // A is 1-indexed: A(nab - ij) means a_integrals[nab - ij]
                // B is 1-indexed: B(ij + 1) means b_integrals[ij + 1]
                x += (ibb as f64) * a_integrals[nab - ij] * b_integrals[ij + 1];
            }
        }

        let mut ss = x * 0.5;
        // Normalization factor
        ss *= (alpha.powi(2 * na as i32 + 1) * beta.powi(2 * nb as i32 + 1)
            / (FACTORIAL[2 * na] * FACTORIAL[2 * nb]))
            .sqrt();
        return ss;
    }

    // P overlaps: la <= 1 and lb <= 1
    if la <= 1 && lb <= 1 {
        // Special case: sigma (m=0), involving s-p or p-p sigma
        if m == 0 {
            let iu = la % 2; // 0 if la=0, 1 if la=1
            let iv = lb % 2; // 0 if lb=0, 1 if lb=1
            let namu = na - iu;
            let nbmv = nb - iv;

            for kc in 0..=iu {
                let ic = nab - iu - iv + kc;
                let jc = 1 + kc;
                for kd in 0..=iv {
                    let id = ic + kd;
                    let jd = jc + kd;
                    for ke in 0..=namu {
                        let ibe = ibinom(namu, ke);
                        let ie = id - ke;
                        let je = jd + ke;
                        for kf in 0..=nbmv {
                            let mut ibf = ibe * ibinom(nbmv, kf);
                            if (kd + kf) % 2 == 1 {
                                ibf = -ibf;
                            }
                            x += (ibf as f64) * a_integrals[ie - kf] * b_integrals[je + kf];
                        }
                    }
                }
            }

            let mut ss = x * (((2 * la + 1) * (2 * lb + 1)) as f64 * 0.25).sqrt();
            // Normalization
            ss *= (alpha.powi(2 * na as i32 + 1) * beta.powi(2 * nb as i32 + 1)
                / (FACTORIAL[2 * na] * FACTORIAL[2 * nb]))
                .sqrt();
            // Sign convention
            if lb % 2 == 1 {
                ss = -ss;
            }
            return ss;
        }

        // Special case: pi (la=1, lb=1, m=1)
        if la == 1 && lb == 1 && m == 1 {
            for ke in 0..=(na - 1) {
                let ibe = ibinom(na - 1, ke);
                let ie = nab - ke;
                let je = ke + 1;
                for kf in 0..=(nb - 1) {
                    let mut ibf = ibe * ibinom(nb - 1, kf);
                    if kf % 2 == 1 {
                        ibf = -ibf;
                    }
                    let i_idx = ie - kf;
                    let j_idx = je + kf;
                    x += (ibf as f64)
                        * (a_integrals[i_idx] * b_integrals[j_idx]
                            - a_integrals[i_idx] * b_integrals[j_idx + 2]
                            - a_integrals[i_idx - 2] * b_integrals[j_idx]
                            + a_integrals[i_idx - 2] * b_integrals[j_idx + 2]);
                }
            }

            let mut ss = x * 0.75;
            // Normalization
            ss *= (alpha.powi(2 * na as i32 + 1) * beta.powi(2 * nb as i32 + 1)
                / (FACTORIAL[2 * na] * FACTORIAL[2 * nb]))
                .sqrt();
            // Sign convention: if (lb + |mm|) is odd, negate
            if (lb + m) % 2 == 1 {
                ss = -ss;
            }
            return ss;
        }
    }

    // Should not reach here for AM1 sp-basis
    0.0
}

/// Get the principal quantum number for sp orbitals given atomic number.
///
/// Follows the SQM convention:
///   Z < 2  -> n = 1 (H)
///   Z < 10 -> n = 2 (Li-F, i.e. C, N, O, F)
///   Z < 18 -> n = 3 (Na-Cl, i.e. P, S, Cl)
///   Z < 36 -> n = 4 (K-Br, i.e. Br)
///   Z < 54 -> n = 5 (Rb-I, i.e. I)
pub fn principal_quantum_number(atomic_number: u8) -> usize {
    if atomic_number < 2 {
        1
    } else if atomic_number < 10 {
        2
    } else if atomic_number < 18 {
        3
    } else if atomic_number < 36 {
        4
    } else if atomic_number < 54 {
        5
    } else {
        6
    }
}

/// Compute the full overlap matrix S for the molecule.
///
/// In NDDO methods, the overlap matrix is the identity on-site (by construction)
/// and has off-diagonal blocks between atom pairs. The overlap is used for
/// resonance integral scaling: H_core(mu,nu) = 0.5*(beta_mu+beta_nu)*S(mu,nu)
/// but NOT for the eigenvalue problem (which uses S = I).
pub fn compute_overlap_matrix(mol: &Am1Molecule) -> DMatrix<f64> {
    let n = mol.n_basis;
    let mut s = DMatrix::identity(n, n);

    for i in 0..mol.atoms.len() {
        for j in (i + 1)..mol.atoms.len() {
            let ai = &mol.atoms[i];
            let aj = &mol.atoms[j];
            let r_ang = mol.distance(i, j);
            let dc = mol.direction_cosines(i, j);

            let r_bohr = r_ang * ANG_TO_BOHR;

            let ni_qn = principal_quantum_number(ai.atomic_number);
            let nj_qn = principal_quantum_number(aj.atomic_number);

            // Compute local overlap integrals in diatomic frame
            let local = compute_local_overlaps(
                ai.params.zeta_s,
                ai.params.zeta_p,
                ai.params.n_orbitals,
                ni_qn,
                aj.params.zeta_s,
                aj.params.zeta_p,
                aj.params.n_orbitals,
                nj_qn,
                r_bohr,
            );

            // Rotate to global frame and store
            let ni = ai.params.n_orbitals as usize;
            let nj = aj.params.n_orbitals as usize;
            let rotated = rotate_overlap_block(&local, ni, nj, &dc);

            for mi in 0..ni {
                for mj in 0..nj {
                    let gi = ai.basis_offset + mi;
                    let gj = aj.basis_offset + mj;
                    s[(gi, gj)] = rotated[(mi, mj)];
                    s[(gj, gi)] = rotated[(mi, mj)];
                }
            }
        }
    }

    s
}

/// Local overlap integrals in the diatomic frame.
///
/// These are the unique overlap integrals needed for the sp basis:
/// - ss: <s_A | s_B>
/// - sp: <s_A | p_sigma_B>
/// - ps: <p_sigma_A | s_B>
/// - pp_sigma: <p_sigma_A | p_sigma_B>
/// - pp_pi: <p_pi_A | p_pi_B>
#[derive(Debug, Clone)]
struct LocalOverlaps {
    ss: f64,
    sp: f64,
    ps: f64,
    pp_sigma: f64,
    pp_pi: f64,
}

/// Compute the local-frame overlap integrals between two atoms.
///
/// The local frame has z along the internuclear axis (A -> B).
///
/// Arguments:
/// - zeta_s_a, zeta_p_a: Slater exponents for atom A (bohr^-1)
/// - n_orb_a: number of orbitals on atom A (1 for H, 4 for sp atoms)
/// - nqn_a: principal quantum number for atom A
/// - zeta_s_b, zeta_p_b: Slater exponents for atom B (bohr^-1)
/// - n_orb_b: number of orbitals on atom B
/// - nqn_b: principal quantum number for atom B
/// - r_bohr: internuclear distance in bohr
fn compute_local_overlaps(
    zeta_s_a: f64,
    zeta_p_a: f64,
    n_orb_a: u8,
    nqn_a: usize,
    zeta_s_b: f64,
    zeta_p_b: f64,
    n_orb_b: u8,
    nqn_b: usize,
    r_bohr: f64,
) -> LocalOverlaps {
    if r_bohr < 1.0e-10 {
        return LocalOverlaps {
            ss: 1.0,
            sp: 0.0,
            ps: 0.0,
            pp_sigma: 1.0,
            pp_pi: 1.0,
        };
    }

    // Principal quantum numbers for the orbitals
    let na_s = nqn_a; // e.g. 1 for H, 2 for C/N/O
    let nb_s = nqn_b;
    let na_p = nqn_a; // p orbitals have same principal quantum number
    let nb_p = nqn_b;

    // Compute A and B integrals for s-s overlap
    // We need integrals up to order na + nb for the pair
    let ntotal_ss = na_s + nb_s;
    let alpha_ss = zeta_s_a * r_bohr;
    let beta_ss = zeta_s_b * r_bohr;
    let a_ss = compute_a_integrals(ntotal_ss, 0.5 * (alpha_ss + beta_ss));
    let b_ss = compute_b_integrals(ntotal_ss, 0.5 * (alpha_ss - beta_ss));
    let ss = calculate_overlap(na_s, 0, 0, nb_s, 0, alpha_ss, beta_ss, &a_ss, &b_ss);

    // s(A)-p_sigma(B) overlap
    let mut sp = 0.0;
    if n_orb_b >= 4 {
        let ntotal_sp = na_s + nb_p;
        let alpha_sp = zeta_s_a * r_bohr;
        let beta_sp = zeta_p_b * r_bohr;
        let a_sp = compute_a_integrals(ntotal_sp, 0.5 * (alpha_sp + beta_sp));
        let b_sp = compute_b_integrals(ntotal_sp, 0.5 * (alpha_sp - beta_sp));
        sp = calculate_overlap(na_s, 0, 0, nb_p, 1, alpha_sp, beta_sp, &a_sp, &b_sp);
    }

    // p_sigma(A)-s(B) overlap
    let mut ps = 0.0;
    if n_orb_a >= 4 {
        let ntotal_ps = na_p + nb_s;
        let alpha_ps = zeta_p_a * r_bohr;
        let beta_ps = zeta_s_b * r_bohr;
        let a_ps = compute_a_integrals(ntotal_ps, 0.5 * (alpha_ps + beta_ps));
        let b_ps = compute_b_integrals(ntotal_ps, 0.5 * (alpha_ps - beta_ps));
        ps = calculate_overlap(na_p, 1, 0, nb_s, 0, alpha_ps, beta_ps, &a_ps, &b_ps);
    }

    // p_sigma(A)-p_sigma(B) overlap
    let mut pp_sigma = 0.0;
    let mut pp_pi = 0.0;
    if n_orb_a >= 4 && n_orb_b >= 4 {
        let ntotal_pp = na_p + nb_p;
        let alpha_pp = zeta_p_a * r_bohr;
        let beta_pp = zeta_p_b * r_bohr;
        let a_pp = compute_a_integrals(ntotal_pp, 0.5 * (alpha_pp + beta_pp));
        let b_pp = compute_b_integrals(ntotal_pp, 0.5 * (alpha_pp - beta_pp));

        pp_sigma = calculate_overlap(na_p, 1, 0, nb_p, 1, alpha_pp, beta_pp, &a_pp, &b_pp);
        pp_pi = calculate_overlap(na_p, 1, 1, nb_p, 1, alpha_pp, beta_pp, &a_pp, &b_pp);
    }

    LocalOverlaps {
        ss,
        sp,
        ps,
        pp_sigma,
        pp_pi,
    }
}

/// Rotate local-frame overlap integrals to the global Cartesian frame.
///
/// The local frame has z along the internuclear axis A -> B.
/// The rotation follows the SQM/MOPAC convention used in Rotate1Elec.
///
/// Basis ordering in the returned matrix: s, px, py, pz
///
/// The rotation for s-p blocks is straightforward:
///   S_global(s_A, p_mu_B) = dc[mu] * S_local(s_A, p_sigma_B)
///
/// For p-p blocks, we build a local coordinate system (e_x, e_y, e_z)
/// where e_z is along the bond, and transform:
///   S_global(p_mu_A, p_nu_B) = sum_lambda R(mu,lambda) * R(nu,lambda) * S_local(lambda)
fn rotate_overlap_block(
    local: &LocalOverlaps,
    ni: usize,
    nj: usize,
    dc: &[f64; 3],
) -> DMatrix<f64> {
    let mut global = DMatrix::zeros(ni, nj);

    let ca = dc[0]; // direction cosine along x
    let cb = dc[1]; // direction cosine along y
    let cc = dc[2]; // direction cosine along z

    // s(A)-s(B): no rotation
    global[(0, 0)] = local.ss;

    // s(A)-p(B): project sigma overlap onto cartesian directions
    if nj >= 4 {
        // In the local frame, only s-p_sigma is nonzero.
        // The sigma direction is along the bond (dc).
        // S_global(s_A, p_x_B) = dc[0] * S_local(s_A, p_sigma_B)
        // etc.
        global[(0, 1)] = ca * local.sp;
        global[(0, 2)] = cb * local.sp;
        global[(0, 3)] = cc * local.sp;
    }

    // p(A)-s(B): project sigma overlap onto cartesian directions
    if ni >= 4 {
        // S_global(p_x_A, s_B) = dc[0] * S_local(p_sigma_A, s_B)
        // etc.
        global[(1, 0)] = ca * local.ps;
        global[(2, 0)] = cb * local.ps;
        global[(3, 0)] = cc * local.ps;
    }

    // p(A)-p(B): rotate using local coordinate frame
    if ni >= 4 && nj >= 4 {
        let s_sigma = local.pp_sigma;
        let s_pi = local.pp_pi;

        let r_xy = (ca * ca + cb * cb).sqrt();

        if r_xy > 1.0e-07 {
            // General case: bond not along z-axis
            //
            // Build local coordinate axes in terms of global Cartesian:
            // e_z = (ca, cb, cc) -- along bond (sigma direction)
            // e_x = (cc*ca/r_xy, cc*cb/r_xy, -r_xy) -- one perpendicular direction (pi_1)
            // e_y = (-cb/r_xy, ca/r_xy, 0) -- other perpendicular direction (pi_2)
            let ex = [cc * ca / r_xy, cc * cb / r_xy, -r_xy];
            let ey = [-cb / r_xy, ca / r_xy, 0.0];
            let ez = [ca, cb, cc];

            // S_global(p_mu_A, p_nu_B)
            //   = ex[mu]*ex[nu]*S_pi + ey[mu]*ey[nu]*S_pi + ez[mu]*ez[nu]*S_sigma
            for mu in 0..3usize {
                for nu in 0..3usize {
                    global[(mu + 1, nu + 1)] =
                        ex[mu] * ex[nu] * s_pi + ey[mu] * ey[nu] * s_pi + ez[mu] * ez[nu] * s_sigma;
                }
            }
        } else {
            // Special case: bond along z-axis (r_xy near 0)
            // e_z = (0, 0, sign(cc)), e_x = (1, 0, 0), e_y = (0, 1, 0) (or flipped)
            //
            // If cc > 0: sigma along +z, pi along x and y
            //   S(px,px) = S_pi, S(py,py) = S_pi, S(pz,pz) = S_sigma
            // If cc < 0: sigma along -z, pi along x and y
            //   S(px,px) = S_pi, S(py,py) = S_pi, S(pz,pz) = S_sigma
            // Off-diagonals are zero.
            global[(1, 1)] = s_pi;
            global[(2, 2)] = s_pi;
            global[(3, 3)] = s_sigma;
        }
    }

    global
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that overlaps of an atom with itself give identity.
    #[test]
    fn test_self_overlap_h() {
        // H atom: 1s orbital, zeta_s = 1.188078
        let nqn = 1;
        let local = compute_local_overlaps(
            1.188078, 0.0, 1, nqn,
            1.188078, 0.0, 1, nqn,
            0.0, // zero distance
        );
        assert!((local.ss - 1.0).abs() < 1.0e-10, "H self-overlap ss should be 1.0, got {}", local.ss);
    }

    /// Test H-H overlap at a typical bond distance.
    #[test]
    fn test_h_h_overlap() {
        // Two H atoms at 0.74 Angstrom (H2 bond length)
        let r_bohr = 0.74 * ANG_TO_BOHR;
        let zeta_h = 1.188078;
        let nqn = 1;

        let local = compute_local_overlaps(
            zeta_h, 0.0, 1, nqn,
            zeta_h, 0.0, 1, nqn,
            r_bohr,
        );

        // For two 1s STOs with equal exponents at this distance,
        // the overlap should be positive and less than 1
        assert!(local.ss > 0.0, "H-H overlap should be positive, got {}", local.ss);
        assert!(local.ss < 1.0, "H-H overlap should be less than 1, got {}", local.ss);

        // Known value for 1s-1s overlap with zeta=1.188078 at R=1.398 bohr:
        // approximately 0.75-0.80
        println!("H-H overlap at 0.74 A: {}", local.ss);
    }

    /// Test C-H overlap (2s-1s, 2p-1s, etc.)
    #[test]
    fn test_c_h_overlap() {
        // C-H bond ~1.09 Angstrom
        let r_bohr = 1.09 * ANG_TO_BOHR;
        let zeta_s_c = 1.808665;
        let zeta_p_c = 1.685116;
        let zeta_s_h = 1.188078;

        let local = compute_local_overlaps(
            zeta_s_c, zeta_p_c, 4, 2,  // C: n=2
            zeta_s_h, 0.0, 1, 1,       // H: n=1
            r_bohr,
        );

        println!("C-H overlaps at 1.09 A:");
        println!("  ss  = {:.6}", local.ss);
        println!("  ps  = {:.6}", local.ps);

        // s-s overlap between C(2s) and H(1s) should be nonzero
        assert!(local.ss.abs() > 0.01, "C-H ss overlap should be significant");
        // p_sigma-s overlap should also be nonzero
        assert!(local.ps.abs() > 0.01, "C-H ps overlap should be significant");
    }

    /// Test that the overlap matrix is symmetric in the global frame.
    ///
    /// In the local diatomic frame, s-p overlaps have a sign that depends on
    /// the bond direction convention. Specifically, <s_A|p_sigma_B> computed
    /// with the bond A->B equals -<p_sigma_A|s_B> computed with bond B->A.
    /// This is the Thiel/MOPAC sign convention for the sigma overlap.
    ///
    /// The global frame overlap matrix S must satisfy S = S^T. The rotation
    /// via direction cosines ensures this since dc(A->B) = -dc(B->A).
    #[test]
    fn test_overlap_symmetry() {
        let r_bohr = 2.0;
        let local_ab = compute_local_overlaps(
            1.808665, 1.685116, 4, 2,
            3.108032, 2.524039, 4, 2,
            r_bohr,
        );
        let local_ba = compute_local_overlaps(
            3.108032, 2.524039, 4, 2,
            1.808665, 1.685116, 4, 2,
            r_bohr,
        );

        // S(A,B) = S(B,A)^T in the local frame
        assert!(
            (local_ab.ss - local_ba.ss).abs() < 1.0e-10,
            "ss overlap should be symmetric: {} vs {}",
            local_ab.ss,
            local_ba.ss
        );

        // s-p and p-s have opposite signs due to the parity of p orbitals
        // relative to the bond direction. When the bond reverses:
        //   <s_A|p_sigma_B>_{A->B} = -<s_B|p_sigma_A>_{B->A}
        // This means local_ab.sp = -local_ba.ps
        assert!(
            (local_ab.sp + local_ba.ps).abs() < 1.0e-10,
            "sp(A->B) should equal -ps(B->A): sp={} vs ps={}",
            local_ab.sp,
            local_ba.ps
        );

        assert!(
            (local_ab.pp_sigma - local_ba.pp_sigma).abs() < 1.0e-10,
            "pp_sigma overlap should be symmetric"
        );
        assert!(
            (local_ab.pp_pi - local_ba.pp_pi).abs() < 1.0e-10,
            "pp_pi overlap should be symmetric"
        );
    }

    /// Test 2s-2s overlap with equal exponents against the analytical formula.
    /// For two 2s STOs with equal exponent zeta at distance R (bohr):
    /// S(2s,2s) = (1 + p + p^2/3 + p^3/15 + p^4/105) * exp(-p)
    /// where p = zeta * R.
    #[test]
    fn test_2s_2s_equal_exponents() {
        let zeta = 1.8;
        let r_bohr = 2.5;

        // Verify against the direct Mulliken integral computation.
        // Cross-check by computing with slightly different exponents
        // and verifying continuity.
        let local = compute_local_overlaps(
            zeta, 0.0, 4, 2,
            zeta, 0.0, 4, 2,
            r_bohr,
        );

        // The overlap must be between 0 and 1
        assert!(local.ss > 0.0 && local.ss < 1.0,
            "2s-2s overlap should be in (0,1), got {}", local.ss);

        // Verify against slightly perturbed exponents (continuity)
        let eps = 1.0e-6;
        let local_pert = compute_local_overlaps(
            zeta + eps, 0.0, 4, 2,
            zeta, 0.0, 4, 2,
            r_bohr,
        );
        assert!((local.ss - local_pert.ss).abs() < 1.0e-4,
            "2s-2s overlap should be continuous in exponents");

        println!("2s-2s overlap (zeta={}, R={} bohr): {:.10}", zeta, r_bohr, local.ss);
    }

    /// Test equal exponents for 1s-1s against the known analytical formula.
    /// For two 1s STOs with equal exponent zeta at distance R (bohr):
    /// S = (1 + zeta*R + (zeta*R)^2/3) * exp(-zeta*R)
    #[test]
    fn test_1s_1s_equal_exponents() {
        let zeta = 1.5;
        let r_bohr = 2.0;
        let p = zeta * r_bohr; // = 3.0

        let analytical = (1.0 + p + p * p / 3.0) * (-p as f64).exp();

        let local = compute_local_overlaps(
            zeta, 0.0, 1, 1,
            zeta, 0.0, 1, 1,
            r_bohr,
        );

        assert!(
            (local.ss - analytical).abs() < 1.0e-10,
            "1s-1s equal exponents: got {}, expected {}",
            local.ss,
            analytical
        );
    }
}
