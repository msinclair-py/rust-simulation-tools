//! Two-center two-electron integrals for AM1 (NDDO approximation).
//!
//! Implements the MOPAC/SQM point-charge multipole model for NDDO integrals.
//!
//! The approach closely follows the SQM implementation in qm2_repp.F90 and
//! qm2_rotate_qmqm.F90, which originate from the MOPAC/MNDO lineage.
//!
//! Each pair of AOs on an atom defines a charge distribution with multipole
//! character. The charge distributions are expanded as point charges separated
//! by distances DD (dipole) and QQ (quadrupole), computed from Slater orbital
//! exponents.
//!
//! The two-center interaction between these distributions is damped using
//! additive terms (rho values, called po/AM/AD/AQ in MOPAC). The basic
//! interaction kernel is:
//!
//!   1 / sqrt(R^2 + (rho_A + rho_B)^2)
//!
//! where R is the internuclear distance in Bohr, and rho values are in Bohr.
//!
//! All 22 unique local-frame integrals are computed for an sp-sp atom pair.
//! These are rotated to the molecular (global) frame using direction cosines.
//!
//! Reference: Dewar & Thiel, JACS 99, 4899 (1977); Thiel, TCA 81, 391 (1992).

use crate::molecule::Am1Molecule;
use crate::params::{Am1Element, BOHR_TO_ANG};

// --------------------------------------------------------------------------
// MOPAC-compatible constant: AU_TO_EV = 27.21 eV per Hartree.
// SQM and MOPAC use this truncated value for historical compatibility.
// The AM1 parameters were fit with this value, so it must be used here.
// --------------------------------------------------------------------------
const AU_TO_EV: f64 = 27.21;

/// Charge separation and additive-term parameters for an atom.
///
/// These follow the MOPAC convention:
/// - DD: dipole charge separation (Bohr), computed from Slater exponents
/// - QQ: quadrupole charge separation (Bohr), computed from Slater exponents
/// - AM: monopole additive term = 0.5 * AU_TO_EV / GSS (Bohr)
/// - AD: dipole additive term, found by iterative solver (Bohr)
/// - AQ: quadrupole additive term, found by iterative solver (Bohr)
#[derive(Debug, Clone)]
pub struct ChargeParams {
    /// Dipole charge separation (Bohr). SQM name: DD(2) or multip_2c_elec_params(1).
    pub dd: f64,
    /// Quadrupole charge separation (Bohr). SQM name: QQ or multip_2c_elec_params(2).
    pub qq: f64,
    /// Monopole additive term (Bohr). SQM name: AM or multip_2c_elec_params(3).
    pub am: f64,
    /// Dipole additive term (Bohr). SQM name: AD or multip_2c_elec_params(4).
    pub ad: f64,
    /// Quadrupole additive term (Bohr). SQM name: AQ or multip_2c_elec_params(5).
    pub aq: f64,
}

/// Compute charge separation and additive-term parameters for an atom.
///
/// This matches the SQM code in `qm2_load_params_and_allocate.F90` for the
/// AM1 parameter block. The key formulas are:
///
/// DD = AIJL(zs,zp,ns,ns,1) / sqrt(3)   (dipole separation in Bohr)
/// QQ = sqrt((4*ns^2 + 6*ns + 2)/20) / zp   (quadrupole separation in Bohr)
/// AM = 0.5 * AU_TO_EV / GSS   (monopole additive term)
/// AD = 0.5 / dd2_temp   (dipole additive term, iterative)
/// AQ = 0.5 / dd2_temp   (quadrupole additive term, iterative)
pub fn compute_charge_params(p: &Am1Element) -> ChargeParams {
    let ns = principal_quantum_number(p.atomic_number) as f64;
    let zs = p.zeta_s;
    let zp = p.zeta_p;

    // DD: dipole charge separation
    // DD = AIJL(zs, zp, ns, np, L=1) / sqrt(3)
    // where np = max(2, ns) but for sp atoms np = ns (same shell).
    // AIJL = FC(ns+np+L+1) / sqrt(FC(2*ns+1)*FC(2*np+1))
    //        * (2*zs/(zs+zp))^ns * sqrt(2*zs/(zs+zp))
    //        * (2*zp/(zs+zp))^np * sqrt(2*zp/(zs+zp))
    //        / (zs+zp)^L
    //
    // Simplified for ns=np and L=1:
    // DD = (( (4*zs*zp)^(ns+0.5) ) * (2*ns + 1))
    //      / (((zs+zp)^(2*ns+2)) * sqrt(3))
    let dd = if zs > 0.0 && zp > 0.0 {
        let base = 4.0 * zs * zp;
        let exp1 = ns + 0.5;
        let base2 = zs + zp;
        let exp2 = 2.0 * ns + 2.0;
        (base.powf(exp1) * (2.0 * ns + 1.0)) / (base2.powf(exp2) * 3.0_f64.sqrt())
    } else {
        0.0
    };

    // QQ: quadrupole charge separation
    // QQ = sqrt((4*ns^2 + 6*ns + 2) / 20) / zp
    let qq = if zp > 0.0 {
        ((4.0 * ns * ns + 6.0 * ns + 2.0) / 20.0).sqrt() / zp
    } else {
        0.0
    };

    // AM: monopole additive term
    let am = if p.gss.abs() > 1.0e-10 {
        0.5 * AU_TO_EV / p.gss
    } else {
        0.0
    };

    // For hydrogen (s-only atom), AD = AQ = AM
    if p.n_orbitals == 1 {
        return ChargeParams {
            dd,
            qq,
            am,
            ad: am,
            aq: am,
        };
    }

    // AD: dipole additive term (iterative solver)
    // Solve: 0.5*x - 0.5/sqrt(4*DD^2 + 1/x^2) = HSP/AU_TO_EV
    // where x is the additive term parameter, and AD = 0.5/x.
    let ad = compute_ad(p.hsp, dd);

    // AQ: quadrupole additive term (iterative solver)
    // The reference one-center integral is HPP = 0.5*(GPP - GP2)
    // Solve: 0.25*x - 0.5/sqrt(4*QQ^2 + 1/x^2) + 0.25/sqrt(8*QQ^2 + 1/x^2) = HPP/AU_TO_EV
    // where x is the additive term parameter, and AQ = 0.5/x.
    let aq = compute_aq(p.gpp, p.gp2, qq);

    ChargeParams { dd, qq, am, ad, aq }
}

/// Solve for AD (dipole additive term) using Newton-like iteration.
///
/// This matches the SQM code exactly: secant method starting from an
/// initial guess derived from (HSP/(AU_TO_EV * DD^2))^(1/3).
fn compute_ad(hsp: f64, dd: f64) -> f64 {
    if hsp.abs() < 1.0e-10 || dd.abs() < 1.0e-10 {
        return 0.5 * AU_TO_EV / hsp.max(0.1);
    }

    let target = hsp / AU_TO_EV; // target value in atomic units
    let dd2 = dd * dd;

    // Initial guess: (HSP / (AU_TO_EV * DD^2))^(1/3)
    let mut x1 = (hsp / (AU_TO_EV * dd2)).powf(1.0 / 3.0);
    let mut x2 = x1 + 0.04;

    // Secant method (5 iterations, matching SQM)
    for _ in 0..5 {
        let diff = x2 - x1;
        let f1 = 0.5 * x1 - 0.5 / (4.0 * dd2 + 1.0 / (x1 * x1)).sqrt();
        let f2 = 0.5 * x2 - 0.5 / (4.0 * dd2 + 1.0 / (x2 * x2)).sqrt();

        if (f2 - f1).abs() < 1.0e-25 {
            break;
        }

        let x3 = x1 + diff * (target - f1) / (f2 - f1);
        x1 = x2;
        x2 = x3;
    }

    // AD = 0.5 / x2
    0.5 / x2
}

/// Solve for AQ (quadrupole additive term) using Newton-like iteration.
///
/// This matches the SQM code exactly: the reference integral is
/// HPP = 0.5*(GPP - GP2), clamped to a minimum of 0.1 eV.
fn compute_aq(gpp: f64, gp2: f64, qq: f64) -> f64 {
    let hpp = 0.5 * (gpp - gp2);
    // SQM clamps hpp to minimum 0.1 eV (required for Cl and others)
    let hpp = hpp.max(0.1);

    if qq.abs() < 1.0e-10 {
        return 0.5 * AU_TO_EV / hpp;
    }

    let target = hpp / AU_TO_EV;
    let qq2 = qq * qq;

    // Initial guess
    let mut x1 = (16.0 * hpp / (AU_TO_EV * 48.0 * qq2 * qq2)).powf(1.0 / 5.0);
    let mut x2 = x1 + 0.04;

    // Secant method (5 iterations)
    for _ in 0..5 {
        let diff = x2 - x1;
        let f1 = 0.25 * x1 - 0.5 / (4.0 * qq2 + 1.0 / (x1 * x1)).sqrt()
            + 0.25 / (8.0 * qq2 + 1.0 / (x1 * x1)).sqrt();
        let f2 = 0.25 * x2 - 0.5 / (4.0 * qq2 + 1.0 / (x2 * x2)).sqrt()
            + 0.25 / (8.0 * qq2 + 1.0 / (x2 * x2)).sqrt();

        if (f2 - f1).abs() < 1.0e-25 {
            break;
        }

        let x3 = x1 + diff * (target - f1) / (f2 - f1);
        x1 = x2;
        x2 = x3;
    }

    0.5 / x2
}

/// Get the principal quantum number for the valence s shell.
///
/// This is the row in the periodic table: H,He -> 1; Li..Ne -> 2;
/// Na..Ar -> 3; K..Kr -> 4; Rb..Xe -> 5; Cs..Rn -> 6.
fn principal_quantum_number(z: u8) -> u8 {
    match z {
        1..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        55..=86 => 6,
        _ => 7,
    }
}

// ==========================================================================
// Two-electron integral block (22 local-frame integrals)
// ==========================================================================

/// Two-center two-electron integral block between atoms A and B.
///
/// Stores the 22 unique local-frame integrals following the MOPAC/SQM
/// convention (indexed 1..22 in Fortran, 0..21 here):
///
/// ```text
///  0: (ss|ss)       1: (so|ss)       2: (oo|ss)       3: (pp|ss)
///  4: (ss|os)       5: (so|so)       6: (sp|sp)       7: (oo|so)
///  8: (pp|so)       9: (po|sp)      10: (ss|oo)      11: (ss|pp)
/// 12: (so|oo)      13: (so|pp)      14: (sp|op)      15: (oo|oo)
/// 16: (pp|oo)      17: (oo|pp)      18: (pp|pp)      19: (po|po)
/// 20: (pp|p*p*)    21: (p*p|p*p)
/// ```
///
/// where s = s-orbital, o = p-sigma (along bond), p = p-pi (in plane),
/// p* = p-pi-star (perpendicular to plane). Left of | = atom A, right = atom B.
/// All values in eV.
#[derive(Debug, Clone)]
pub struct TwoElectronBlock {
    /// The 22 unique local-frame integrals (eV). Indexed 0..21.
    pub ri: [f64; 22],
    /// Number of orbitals on atom A.
    pub norb_a: usize,
    /// Number of orbitals on atom B.
    pub norb_b: usize,
    // Keep old fields for backward compatibility with fock.rs
    pub n_pairs_a: usize,
    pub n_pairs_b: usize,
    pub integrals: Vec<Vec<f64>>,
}

/// Compute all two-center two-electron integrals for a molecule.
///
/// Returns a matrix indexed by atom pairs: `tei[i][j]` gives the integral
/// block between atoms i and j. Only the upper triangle (i < j) is computed;
/// the lower triangle is set to the same block.
///
/// The block is always computed with atom i as "A" and atom j as "B" (i < j).
/// Direction cosines from i to j define the local frame.
pub fn compute_two_electron_integrals(mol: &Am1Molecule) -> Vec<Vec<Option<TwoElectronBlock>>> {
    let na = mol.atoms.len();
    let mut tei = vec![vec![None; na]; na];

    // Precompute charge parameters for each atom
    let charge_params: Vec<ChargeParams> = mol
        .atoms
        .iter()
        .map(|a| compute_charge_params(a.params))
        .collect();

    for i in 0..na {
        for j in (i + 1)..na {
            let r_ang = mol.distance(i, j);
            let r_bohr = r_ang / BOHR_TO_ANG;

            // Block with A=i, B=j (for tei[i][j])
            let block_ij = compute_tei_block(
                mol.atoms[i].params,
                mol.atoms[j].params,
                &charge_params[i],
                &charge_params[j],
                r_bohr,
            );

            // Block with A=j, B=i (for tei[j][i])
            // This ensures that when calling code accesses tei[j][i],
            // the block's left side (A) corresponds to atom j and
            // right side (B) corresponds to atom i.
            let block_ji = compute_tei_block(
                mol.atoms[j].params,
                mol.atoms[i].params,
                &charge_params[j],
                &charge_params[i],
                r_bohr,
            );

            tei[i][j] = Some(block_ij);
            tei[j][i] = Some(block_ji);
        }
    }

    tei
}

/// Compute the 22 local-frame two-electron integrals for atom pair (A, B).
///
/// This closely follows `qm2_repp` in SQM. R is the interatomic distance
/// in Bohr. Atom A is on the left, atom B on the right. The local z-axis
/// points from A to B.
///
/// Returns integrals in eV.
fn compute_tei_block(
    pa: &Am1Element,
    pb: &Am1Element,
    cpa: &ChargeParams,
    cpb: &ChargeParams,
    r_bohr: f64,
) -> TwoElectronBlock {
    let norb_a = pa.n_orbitals as usize;
    let norb_b = pb.n_orbitals as usize;
    let rr2 = r_bohr * r_bohr;

    let mut ri = [0.0_f64; 22];

    // (ss|ss) = 1/sqrt(R^2 + (AM_A + AM_B)^2), converted to eV
    let bdd1_sum = cpa.am + cpb.am;
    let sqrtaee = 1.0 / (rr2 + bdd1_sum * bdd1_sum).sqrt();
    ri[0] = AU_TO_EV * sqrtaee;

    let a_is_sp = norb_a == 4;
    let b_is_sp = norb_b == 4;

    if a_is_sp && !b_is_sp {
        // SP atom A, S atom B (e.g., C-H with C on left)
        compute_ri_sp_s(&mut ri, cpa, cpb, r_bohr, rr2);
    } else if !a_is_sp && b_is_sp {
        // S atom A, SP atom B (e.g., H-C with H on left)
        compute_ri_s_sp(&mut ri, cpa, cpb, r_bohr, rr2);
    } else if a_is_sp && b_is_sp {
        // SP atom A, SP atom B
        compute_ri_sp_sp(&mut ri, cpa, cpb, r_bohr, rr2);
    }
    // else: both s-only, only ri[0] is needed

    TwoElectronBlock {
        ri,
        norb_a,
        norb_b,
        n_pairs_a: if norb_a == 1 { 1 } else { 5 },
        n_pairs_b: if norb_b == 1 { 1 } else { 5 },
        integrals: vec![vec![]],
    }
}

/// Helper: 1/sqrt(x) for the point-charge model.
#[inline]
fn inv_sqrt(x: f64) -> f64 {
    1.0 / x.sqrt()
}

/// Compute local integrals for SP-atom(A) vs S-atom(B).
///
/// Follows the SP-HYDROGEN branch of qm2_repp. Computes RI(2..4).
fn compute_ri_sp_s(
    ri: &mut [f64; 22],
    cpa: &ChargeParams,
    cpb: &ChargeParams,
    r: f64,
    rr2: f64,
) {
    let da = cpa.dd;
    let qa = cpa.qq * 2.0; // SQM convention: QA = multip_2c_elec_params(2,i) * 2

    let ade = cpa.ad + cpb.am;
    let ade2 = ade * ade;
    let aqe = cpa.aq + cpb.am;
    let aqe2 = aqe * aqe;

    let arg1 = (r + da).powi(2) + ade2;
    let arg2 = (r - da).powi(2) + ade2;
    let arg3 = (r - qa).powi(2) + aqe2;
    let arg4 = (r + qa).powi(2) + aqe2;
    let arg5 = rr2 + aqe2;
    let arg6 = arg5 + qa * qa;

    let s1 = inv_sqrt(arg1);
    let s2 = inv_sqrt(arg2);
    let s3 = inv_sqrt(arg3);
    let s4 = inv_sqrt(arg4);
    let s5 = inv_sqrt(arg5);
    let s6 = inv_sqrt(arg6);

    let half_ev = 0.5 * AU_TO_EV;
    let quarter_ev = 0.25 * AU_TO_EV;

    // RI(2) = (s*sigma_A | s*s_B) -- dipole on A, monopole on B
    // In SQM: RI(2) = HALF_AU*SQR(1) - HALF_AU*SQR(2)
    // Note: SQM defines DZE = -HALF_AU*SQR(1) + HALF_AU*SQR(2), then RI(2) = -DZE
    ri[1] = half_ev * s1 - half_ev * s2;

    // RI(3) = (sigma*sigma_A | s*s_B) = (oo|ss) -- linear quadrupole on A
    // = RI(1) + FOURTH_AU*(SQR(3)+SQR(4)) - HALF_AU*SQR(5)
    ri[2] = ri[0] + quarter_ev * s3 + quarter_ev * s4 - half_ev * s5;

    // RI(4) = (pi*pi_A | s*s_B) = (pp|ss) -- square quadrupole on A
    // = RI(1) + HALF_AU*SQR(6) - HALF_AU*SQR(5)
    ri[3] = ri[0] + half_ev * s6 - half_ev * s5;
}

/// Compute local integrals for S-atom(A) vs SP-atom(B).
///
/// Follows the HYDROGEN-SP branch of qm2_repp. Computes RI(5), RI(11), RI(12).
fn compute_ri_s_sp(
    ri: &mut [f64; 22],
    cpa: &ChargeParams,
    cpb: &ChargeParams,
    r: f64,
    rr2: f64,
) {
    let db = cpb.dd;
    let qb = cpb.qq * 2.0;

    let aed = cpa.am + cpb.ad;
    let aed2 = aed * aed;
    let aeq = cpa.am + cpb.aq;
    let aeq2 = aeq * aeq;

    let arg1 = (r - db).powi(2) + aed2;
    let arg2 = (r + db).powi(2) + aed2;
    let arg3 = (r - qb).powi(2) + aeq2;
    let arg4 = (r + qb).powi(2) + aeq2;
    let arg5 = rr2 + aeq2;
    let arg6 = arg5 + qb * qb;

    let s1 = inv_sqrt(arg1);
    let s2 = inv_sqrt(arg2);
    let s3 = inv_sqrt(arg3);
    let s4 = inv_sqrt(arg4);
    let s5 = inv_sqrt(arg5);
    let s6 = inv_sqrt(arg6);

    let half_ev = 0.5 * AU_TO_EV;
    let quarter_ev = 0.25 * AU_TO_EV;

    // RI(5) = (s*s_A | s*sigma_B) -- monopole on A, dipole on B
    // In SQM: RI(5) = HALF_AU*SQR(1) - HALF_AU*SQR(2)
    ri[4] = half_ev * s1 - half_ev * s2;

    // RI(11) = (s*s_A | sigma*sigma_B) = (ss|oo)
    ri[10] = ri[0] + quarter_ev * s3 + quarter_ev * s4 - half_ev * s5;

    // RI(12) = (s*s_A | pi*pi_B) = (ss|pp)
    ri[11] = ri[0] + half_ev * s6 - half_ev * s5;
}

/// Compute all 22 local-frame integrals for SP-atom(A) vs SP-atom(B).
///
/// This is the most complex case, following the SP-SP branch of qm2_repp
/// in SQM exactly.
#[allow(clippy::too_many_lines)]
fn compute_ri_sp_sp(
    ri: &mut [f64; 22],
    cpa: &ChargeParams,
    cpb: &ChargeParams,
    r: f64,
    rr2: f64,
) {
    let da = cpa.dd;
    let db = cpb.dd;
    // SQM convention: QA and QB are doubled for the first 52 ARG computations
    let mut qa = cpa.qq * 2.0;
    let mut qb = cpb.qq * 2.0;

    // Additive-term combinations (squared)
    let ade = cpa.ad + cpb.am; // dipole_A + monopole_B
    let ade2 = ade * ade;
    let aqe = cpa.aq + cpb.am; // quadrupole_A + monopole_B
    let aqe2 = aqe * aqe;
    let aed = cpa.am + cpb.ad; // monopole_A + dipole_B
    let aed2 = aed * aed;
    let aeq = cpa.am + cpb.aq; // monopole_A + quadrupole_B
    let aeq2 = aeq * aeq;
    let axx = cpa.ad + cpb.ad; // dipole_A + dipole_B
    let axx2 = axx * axx;
    let adq = cpa.ad + cpb.aq; // dipole_A + quadrupole_B
    let adq2 = adq * adq;
    let aqd = cpa.aq + cpb.ad; // quadrupole_A + dipole_B
    let aqd2 = aqd * aqd;
    let aqq = cpa.aq + cpb.aq; // quadrupole_A + quadrupole_B
    let aqq2 = aqq * aqq;

    // Compute all 71 ARG values (matching SQM indexing: ARG(1)..ARG(71))
    // We use 0-based indexing here: arg[0]..arg[70].
    let mut arg = [0.0_f64; 71];

    // ARG(1..6): dipole/quadrupole on A vs monopole on B
    arg[0] = (r + da).powi(2) + ade2;
    arg[1] = (r - da).powi(2) + ade2;
    arg[2] = (r - qa).powi(2) + aqe2;
    arg[3] = (r + qa).powi(2) + aqe2;
    arg[4] = rr2 + aqe2;
    arg[5] = arg[4] + qa * qa;

    // ARG(7..12): monopole on A vs dipole/quadrupole on B
    arg[6] = (r - db).powi(2) + aed2;
    arg[7] = (r + db).powi(2) + aed2;
    arg[8] = (r - qb).powi(2) + aeq2;
    arg[9] = (r + qb).powi(2) + aeq2;
    arg[10] = rr2 + aeq2;
    arg[11] = arg[10] + qb * qb;

    // ARG(13..18): dipole-dipole
    arg[12] = rr2 + axx2 + (da - db).powi(2);
    arg[13] = rr2 + axx2 + (da + db).powi(2);
    arg[14] = (r + da - db).powi(2) + axx2;
    arg[15] = (r - da + db).powi(2) + axx2;
    arg[16] = (r - da - db).powi(2) + axx2;
    arg[17] = (r + da + db).powi(2) + axx2;

    // ARG(19..22): dipole_A - quadrupole_B (square type)
    arg[18] = (r + da).powi(2) + adq2;
    arg[19] = arg[18] + qb * qb;
    arg[20] = (r - da).powi(2) + adq2;
    arg[21] = arg[20] + qb * qb;

    // ARG(23..26): quadrupole_A - dipole_B (square type)
    arg[22] = (r - db).powi(2) + aqd2;
    arg[23] = arg[22] + qa * qa;
    arg[24] = (r + db).powi(2) + aqd2;
    arg[25] = arg[24] + qa * qa;

    // ARG(27..30): dipole_A - quadrupole_B (linear type)
    arg[26] = (r + da - qb).powi(2) + adq2;
    arg[27] = (r - da - qb).powi(2) + adq2;
    arg[28] = (r + da + qb).powi(2) + adq2;
    arg[29] = (r - da + qb).powi(2) + adq2;

    // ARG(31..34): quadrupole_A - dipole_B (linear type)
    arg[30] = (r + qa - db).powi(2) + aqd2;
    arg[31] = (r + qa + db).powi(2) + aqd2;
    arg[32] = (r - qa - db).powi(2) + aqd2;
    arg[33] = (r - qa + db).powi(2) + aqd2;

    // ARG(35..52): quadrupole-quadrupole
    arg[34] = rr2 + aqq2;
    arg[35] = arg[34] + (qa - qb).powi(2);
    arg[36] = arg[34] + (qa + qb).powi(2);
    arg[37] = arg[34] + qa * qa;
    arg[38] = arg[34] + qb * qb;
    arg[39] = arg[37] + qb * qb;
    arg[40] = (r - qb).powi(2) + aqq2;
    arg[41] = arg[40] + qa * qa;
    arg[42] = (r + qb).powi(2) + aqq2;
    arg[43] = arg[42] + qa * qa;
    arg[44] = (r + qa).powi(2) + aqq2;
    arg[45] = arg[44] + qb * qb;
    arg[46] = (r - qa).powi(2) + aqq2;
    arg[47] = arg[46] + qb * qb;
    arg[48] = (r + qa - qb).powi(2) + aqq2;
    arg[49] = (r + qa + qb).powi(2) + aqq2;
    arg[50] = (r - qa - qb).powi(2) + aqq2;
    arg[51] = (r - qa + qb).powi(2) + aqq2;

    // Now redefine QA and QB to the non-doubled values for ARG(53..71)
    qa = cpa.qq;
    qb = cpb.qq;

    // ARG(53..56): dipole_A crossed with quadrupole_B (3D geometry)
    {
        let da_m_qb_sq = (da - qb).powi(2);
        let da_p_qb_sq = (da + qb).powi(2);
        let r_m_qb_sq = (r - qb).powi(2);
        let r_p_qb_sq = (r + qb).powi(2);
        arg[52] = da_m_qb_sq + r_m_qb_sq + adq2;
        arg[53] = da_m_qb_sq + r_p_qb_sq + adq2;
        arg[54] = da_p_qb_sq + r_m_qb_sq + adq2;
        arg[55] = da_p_qb_sq + r_p_qb_sq + adq2;
    }

    // ARG(57..60): quadrupole_A crossed with dipole_B
    {
        let qa_m_db_sq = (qa - db).powi(2);
        let qa_p_db_sq = (qa + db).powi(2);
        let r_p_qa_sq = (r + qa).powi(2);
        let r_m_qa_sq = (r - qa).powi(2);
        arg[56] = r_p_qa_sq + qa_m_db_sq + aqd2;
        arg[57] = r_m_qa_sq + qa_m_db_sq + aqd2;
        arg[58] = r_p_qa_sq + qa_p_db_sq + aqd2;
        arg[59] = r_m_qa_sq + qa_p_db_sq + aqd2;
    }

    // ARG(61..71): quadrupole-quadrupole (3D geometry)
    {
        let qa_m_qb_sq = (qa - qb).powi(2);
        let qa_p_qb_sq = (qa + qb).powi(2);
        arg[60] = arg[34] + 2.0 * qa_m_qb_sq;
        arg[61] = arg[34] + 2.0 * qa_p_qb_sq;
        arg[62] = arg[34] + 2.0 * (qa * qa + qb * qb);

        let r_pa_mq = r + qa - qb; // Note: qa here is the non-doubled value
        let r_pa_pq = r + qa + qb;
        let r_ma_mq = r - qa - qb;
        let r_ma_pq = r - qa + qb;

        arg[63] = r_pa_mq.powi(2) + qa_m_qb_sq + aqq2;
        arg[64] = r_pa_mq.powi(2) + qa_p_qb_sq + aqq2;
        arg[65] = r_pa_pq.powi(2) + qa_m_qb_sq + aqq2;
        arg[66] = r_pa_pq.powi(2) + qa_p_qb_sq + aqq2;
        arg[67] = r_ma_mq.powi(2) + qa_m_qb_sq + aqq2;
        arg[68] = r_ma_mq.powi(2) + qa_p_qb_sq + aqq2;
        arg[69] = r_ma_pq.powi(2) + qa_m_qb_sq + aqq2;
        arg[70] = r_ma_pq.powi(2) + qa_p_qb_sq + aqq2;
    }

    // Compute all inverse square roots
    let mut sqr = [0.0_f64; 71];
    for i in 0..71 {
        sqr[i] = inv_sqrt(arg[i]);
    }

    // Named intermediates (exactly matching SQM variable names)
    let half_ev = 0.5 * AU_TO_EV;
    let quarter_ev = 0.25 * AU_TO_EV;
    let eighth_ev = 0.125 * AU_TO_EV;
    let sixteenth_ev = 0.0625 * AU_TO_EV;

    // DZE = -HALF*s1 + HALF*s2  (dipole on A interacting with monopole on B)
    let dze = -half_ev * sqr[0] + half_ev * sqr[1];
    // QZZE = quadrupole_zz on A vs monopole on B
    let qzze = quarter_ev * sqr[2] + quarter_ev * sqr[3] - half_ev * sqr[4];
    // QXXE = quadrupole_xx on A vs monopole on B
    let qxxe = half_ev * sqr[5] - half_ev * sqr[4];

    // EDZ = monopole on A vs dipole on B
    let edz = -half_ev * sqr[6] + half_ev * sqr[7];
    // EQZZ, EQXX = monopole on A vs quadrupole on B
    let eqzz = quarter_ev * sqr[8] + quarter_ev * sqr[9] - half_ev * sqr[10];
    let eqxx = half_ev * sqr[11] - half_ev * sqr[10];

    // DXDX = dipole_x on A vs dipole_x on B (perpendicular dipoles)
    let dxdx = half_ev * sqr[12] - half_ev * sqr[13];
    // DZDZ = dipole_z on A vs dipole_z on B (parallel dipoles)
    let dzdz = quarter_ev * sqr[14] + quarter_ev * sqr[15]
        - quarter_ev * sqr[16] - quarter_ev * sqr[17];

    // DZQXX = dipole_z on A vs quadrupole_xx on B
    let dzqxx = quarter_ev * sqr[18] - quarter_ev * sqr[19]
        - quarter_ev * sqr[20] + quarter_ev * sqr[21];
    // QXXDZ = quadrupole_xx on A vs dipole_z on B
    let qxxdz = quarter_ev * sqr[22] - quarter_ev * sqr[23]
        - quarter_ev * sqr[24] + quarter_ev * sqr[25];

    // DZQZZ = dipole_z on A vs quadrupole_zz on B
    let dzqzz = -eighth_ev * sqr[26] + eighth_ev * sqr[27]
        - eighth_ev * sqr[28] + eighth_ev * sqr[29]
        - quarter_ev * sqr[20] + quarter_ev * sqr[18];
    // QZZDZ = quadrupole_zz on A vs dipole_z on B
    let qzzdz = -eighth_ev * sqr[30] + eighth_ev * sqr[31]
        - eighth_ev * sqr[32] + eighth_ev * sqr[33]
        + quarter_ev * sqr[22] - quarter_ev * sqr[24];

    // Quadrupole-quadrupole terms
    let qxxqxx = eighth_ev * sqr[35] + eighth_ev * sqr[36]
        - quarter_ev * sqr[37] - quarter_ev * sqr[38]
        + quarter_ev * sqr[34];
    let qxxqyy = quarter_ev * sqr[39] - quarter_ev * sqr[37]
        - quarter_ev * sqr[38] + quarter_ev * sqr[34];
    let qxxqzz = eighth_ev * sqr[41] + eighth_ev * sqr[43]
        - eighth_ev * sqr[40] - eighth_ev * sqr[42]
        - quarter_ev * sqr[37] + quarter_ev * sqr[34];
    let qzzqxx = eighth_ev * sqr[45] + eighth_ev * sqr[47]
        - eighth_ev * sqr[44] - eighth_ev * sqr[46]
        - quarter_ev * sqr[38] + quarter_ev * sqr[34];
    let qzzqzz = sixteenth_ev * sqr[48] + sixteenth_ev * sqr[49]
        + sixteenth_ev * sqr[50] + sixteenth_ev * sqr[51]
        - eighth_ev * sqr[46] - eighth_ev * sqr[44]
        - eighth_ev * sqr[40] - eighth_ev * sqr[42]
        + quarter_ev * sqr[34];

    // Cross terms with 3D geometry (using non-doubled QA, QB)
    let dxqxz = -quarter_ev * sqr[52] + quarter_ev * sqr[53]
        + quarter_ev * sqr[54] - quarter_ev * sqr[55];
    let qxzdx = -quarter_ev * sqr[56] + quarter_ev * sqr[57]
        + quarter_ev * sqr[58] - quarter_ev * sqr[59];
    let qxzqxz = eighth_ev * sqr[63] - eighth_ev * sqr[65]
        - eighth_ev * sqr[67] + eighth_ev * sqr[69]
        - eighth_ev * sqr[64] + eighth_ev * sqr[66]
        + eighth_ev * sqr[68] - eighth_ev * sqr[70];

    // Now assign to the 22 RI values (SQM convention)
    ri[1] = -dze;                               //  2: (so|ss)
    ri[2] = ri[0] + qzze;                       //  3: (oo|ss)
    ri[3] = ri[0] + qxxe;                       //  4: (pp|ss)
    ri[4] = -edz;                               //  5: (ss|os)
    ri[5] = dzdz;                               //  6: (so|so)
    ri[6] = dxdx;                               //  7: (sp|sp)
    ri[7] = -edz - qzzdz;                       //  8: (oo|so)
    ri[8] = -edz - qxxdz;                       //  9: (pp|so)
    ri[9] = -qxzdx;                             // 10: (po|sp)
    ri[10] = ri[0] + eqzz;                      // 11: (ss|oo)
    ri[11] = ri[0] + eqxx;                      // 12: (ss|pp)
    ri[12] = -dze - dzqzz;                      // 13: (so|oo)
    ri[13] = -dze - dzqxx;                      // 14: (so|pp)
    ri[14] = -dxqxz;                            // 15: (sp|op)
    ri[15] = ri[0] + eqzz + qzze + qzzqzz;     // 16: (oo|oo)
    ri[16] = ri[0] + eqzz + qxxe + qxxqzz;     // 17: (pp|oo)
    ri[17] = ri[0] + eqxx + qzze + qzzqxx;      // 18: (oo|pp)
    ri[18] = ri[0] + eqxx + qxxe + qxxqxx;      // 19: (pp|pp)
    ri[19] = qxzqxz;                            // 20: (po|po)
    ri[20] = ri[0] + eqxx + qxxe + qxxqyy;      // 21: (pp|p*p*)
    ri[21] = 0.5 * (qxxqxx - qxxqyy);           // 22: (p*p|p*p)
}

// ==========================================================================
// Public API: gamma_ss (monopole-monopole integral)
// ==========================================================================

/// Get the (ss|ss) gamma integral between two atoms (in eV).
///
/// This is the monopole-monopole Coulomb integral with Klopman-Ohno damping.
/// Used for core-electron attraction and related terms.
pub fn gamma_ss(pa: &Am1Element, pb: &Am1Element, r_ang: f64) -> f64 {
    let r_bohr = r_ang / BOHR_TO_ANG;
    let cpa = compute_charge_params(pa);
    let cpb = compute_charge_params(pb);
    let rho_sum = cpa.am + cpb.am;
    AU_TO_EV / (r_bohr * r_bohr + rho_sum * rho_sum).sqrt()
}

// ==========================================================================
// Global frame rotation: get_tei_global
// ==========================================================================

/// Convert local-frame two-electron integrals to a specific global-frame
/// AO integral (mu_A nu_A | lambda_B sigma_B).
///
/// This follows `qm2_rotate_qmqm` in SQM. The direction cosines `dc`
/// point from atom A to atom B: dc = (B - A) / |B - A|.
///
/// Orbital ordering: 0 = s, 1 = px, 2 = py, 3 = pz.
///
/// mu, nu are AO indices on atom A (0..norb_a-1).
/// lam, sig are AO indices on atom B (0..norb_b-1).
///
/// Returns the integral in eV.
pub fn get_tei_global(
    block: &TwoElectronBlock,
    _pa: &Am1Element,
    _pb: &Am1Element,
    dc: &[f64; 3],
    mu: usize,
    nu: usize,
    lam: usize,
    sig: usize,
) -> f64 {
    let ri = &block.ri;

    // Build the local frame axes from direction cosines.
    // X vector = direction from A to B (the "sigma" or "bond" direction).
    // We follow SQM convention where X = (XI - XJ)/|XI-XJ| but note that
    // in our calling code dc = direction from A to B = (B-A)/|B-A|.
    // SQM computes X = (XI-XJ)/|XI-XJ| which is from J to I, i.e. from B to A.
    //
    // To match SQM, we negate dc so our local X points from B to A.
    let x = [-dc[0], -dc[1], -dc[2]];

    // Build local Y and Z axes (perpendicular to X), matching SQM exactly.
    let (y, z) = build_local_axes(&x);

    // Now use the rotation formulas from qm2_rotate_qmqm to compute the
    // specific integral W(mu,nu,lam,sig) from the 22 local RI values.
    //
    // The W array in SQM is indexed by the combined pair (mu_A, nu_A) on the left
    // and (lam_B, sig_B) on the right, laid out as a 10x10 matrix for sp-sp.
    // Here we compute a single element on the fly.

    rotate_single_integral(ri, &x, &y, &z, block.norb_a, block.norb_b, mu, nu, lam, sig)
}

/// Build local Y and Z axes from the X axis (bond direction from B to A).
///
/// This matches the SQM convention in qm2_rotate_qmqm exactly.
fn build_local_axes(x: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    if x[2].abs() > 0.99999999 {
        // Bond nearly along global z-axis
        let xsign = x[2].signum();
        let y = [0.0, 1.0, 0.0];
        let z = [1.0, 0.0, 0.0];
        // Adjust x to be exactly along z
        let _ = [0.0, 0.0, xsign];
        (y, z)
    } else {
        let rxy = (1.0 - x[2] * x[2]).sqrt();
        let a = 1.0 / rxy;
        let y = [
            -a * x[1] * x[0].signum(),
            (a * x[0]).abs(),
            0.0,
        ];
        let z = [
            -a * x[0] * x[2],
            -a * x[1] * x[2],
            rxy,
        ];
        (y, z)
    }
}

/// Compute a single rotated integral W(mu_A, nu_A | lam_B, sig_B).
///
/// This is a direct implementation of the rotation formulas from
/// qm2_rotate_qmqm in SQM, computing only the requested element.
///
/// The approach: compute the intermediate products (XX11, XX21, etc.)
/// as needed, then apply the rotation formula for the specific W element.
fn rotate_single_integral(
    ri: &[f64; 22],
    x: &[f64; 3],
    y: &[f64; 3],
    z: &[f64; 3],
    norb_a: usize,
    norb_b: usize,
    mu: usize,
    nu: usize,
    lam: usize,
    sig: usize,
) -> f64 {
    // Encode the pair (mu, nu) on A as a single "pair index" in the SQM ordering.
    // For atom A (left side), the 10 pairs are:
    //   0: (s,s)   1: (px,s)   2: (px,px)   3: (py,s)   4: (py,px)
    //   5: (py,py)  6: (pz,s)   7: (pz,px)   8: (pz,py)  9: (pz,pz)
    //
    // For atom B (right side), same indexing.
    //
    // The pair index maps to position in the W(10*10) array as:
    //   W(pair_a * n_pairs_b + pair_b + 1)  [Fortran 1-based]

    // Convert AO pair (mu, nu) to pair index. We ensure mu >= nu for canonical form.
    let (mu_a, nu_a) = if mu >= nu { (mu, nu) } else { (nu, mu) };
    let (mu_b, nu_b) = if lam >= sig { (lam, sig) } else { (sig, lam) };

    let pair_a = ao_pair_index(mu_a, nu_a, norb_a);
    let pair_b = ao_pair_index(mu_b, nu_b, norb_b);

    if pair_a.is_none() || pair_b.is_none() {
        return 0.0;
    }

    let pa = pair_a.unwrap();
    let pb = pair_b.unwrap();

    // Compute the rotation formula for W(pa, pb).
    // This requires the direction cosine products.
    compute_w_element(ri, x, y, z, pa, pb)
}

/// Convert an AO pair (mu, nu) with mu >= nu to a pair index.
///
/// Returns None for invalid pairs (e.g., p orbital on s-only atom).
fn ao_pair_index(mu: usize, nu: usize, norb: usize) -> Option<usize> {
    if norb == 1 {
        // s-only atom: only (s,s) = pair 0
        if mu == 0 && nu == 0 {
            return Some(0);
        }
        return None;
    }

    // sp atom: orbitals 0=s, 1=px, 2=py, 3=pz
    // Pair indices (mu >= nu):
    //   (0,0)=0  (1,0)=1  (1,1)=2  (2,0)=3  (2,1)=4
    //   (2,2)=5  (3,0)=6  (3,1)=7  (3,2)=8  (3,3)=9
    if mu > 3 || nu > 3 {
        return None;
    }
    Some(mu * (mu + 1) / 2 + nu)
}

/// Compute a single W element from the 22 RI values and the local frame axes.
///
/// pa = pair index on atom A (0..9), pb = pair index on atom B (0..9).
///
/// This is a direct translation of qm2_rotate_qmqm, computing only the
/// specific element requested rather than the full 100-element W array.
fn compute_w_element(
    ri: &[f64; 22],
    x: &[f64; 3],
    y: &[f64; 3],
    z: &[f64; 3],
    pa: usize,
    pb: usize,
) -> f64 {
    // Precompute direction cosine products that appear in the rotation formulas.
    // These are named exactly as in SQM.
    //
    // XX_ij = X(i)*X(j) etc., with i,j in {1,2,3} (Fortran convention).
    // We use 0-based: X(1)->x[0], X(2)->x[1], X(3)->x[2].

    // For pair index mapping:
    // pair 0: (s,s)     -> row/col 0
    // pair 1: (px,s)    -> involves X(1)=x[0]
    // pair 2: (px,px)   -> involves X(1)^2, Y(1)^2+Z(1)^2
    // pair 3: (py,s)    -> involves X(2)=x[1]
    // pair 4: (py,px)   -> involves X(2)*X(1), etc.
    // pair 5: (py,py)   -> involves X(2)^2, Y(2)^2+Z(2)^2
    // pair 6: (pz,s)    -> involves X(3)=x[2]
    // pair 7: (pz,px)   -> involves X(3)*X(1), Z(3)*Z(1)
    // pair 8: (pz,py)   -> involves X(3)*X(2), Z(3)*Z(2)
    // pair 9: (pz,pz)   -> involves X(3)^2, Z(3)^2

    // When both atoms are s-only, only W(0,0) = RI(1) is needed.
    if pa == 0 && pb == 0 {
        return ri[0]; // (ss|ss)
    }

    // Compute the needed intermediate products.
    // For efficiency, we compute them all since many formulas share them.

    let xx11 = x[0] * x[0];
    let xx21 = x[1] * x[0];
    let xx22 = x[1] * x[1];
    let xx31 = x[2] * x[0];
    let xx32 = x[2] * x[1];
    let xx33 = x[2] * x[2];

    let yy11 = y[0] * y[0];
    let yy21 = y[1] * y[0];
    let yy22 = y[1] * y[1];

    let zz11 = z[0] * z[0];
    let zz21 = z[1] * z[0];
    let zz22 = z[1] * z[1];
    let zz31 = z[2] * z[0];
    let zz32 = z[2] * z[1];
    let zz33 = z[2] * z[2];

    let yyzz11 = yy11 + zz11;
    let yyzz21 = yy21 + zz21;
    let yyzz22 = yy22 + zz22;

    let xy11 = 2.0 * x[0] * y[0];
    let xy21 = x[0] * y[1] + x[1] * y[0];
    let xy22 = 2.0 * x[1] * y[1];
    let xy31 = x[2] * y[0];
    let xy32 = x[2] * y[1];

    let xz11 = 2.0 * x[0] * z[0];
    let xz21 = x[0] * z[1] + x[1] * z[0];
    let xz22 = 2.0 * x[1] * z[1];
    let xz31 = x[0] * z[2] + x[2] * z[0];
    let xz32 = x[1] * z[2] + x[2] * z[1];
    let xz33 = 2.0 * x[2] * z[2];

    let yz11 = 2.0 * y[0] * z[0];
    let yz21 = y[0] * z[1] + y[1] * z[0];
    let yz22 = 2.0 * y[1] * z[1];
    let yz31 = y[0] * z[2];
    let yz32 = y[1] * z[2];

    // Now look up the rotation formula based on (pa, pb).
    // We encode this as a big match. Each case directly implements
    // the corresponding line from qm2_rotate_qmqm.

    // Helper: given a "left pair" described by its XX, YYZZ, ZZ components
    // and a "right pair", compute the W element.
    //
    // The left pair is on atom A, the right on atom B.

    // For readability, use short names for RI values (0-indexed):
    let r1 = ri[0];   // (ss|ss)
    let r2 = ri[1];   // (so|ss)
    let r3 = ri[2];   // (oo|ss)
    let r4 = ri[3];   // (pp|ss)
    let r5 = ri[4];   // (ss|os)
    let r6 = ri[5];   // (so|so)
    let r7 = ri[6];   // (sp|sp)
    let r8 = ri[7];   // (oo|so)
    let r9 = ri[8];   // (pp|so)
    let r10 = ri[9];  // (po|sp)
    let r11 = ri[10]; // (ss|oo)
    let r12 = ri[11]; // (ss|pp)
    let r13 = ri[12]; // (so|oo)
    let r14 = ri[13]; // (so|pp)
    let r15 = ri[14]; // (sp|op)
    let r16 = ri[15]; // (oo|oo)
    let r17 = ri[16]; // (pp|oo)
    let r18 = ri[17]; // (oo|pp)
    let r19 = ri[18]; // (pp|pp)
    let r20 = ri[19]; // (po|po)
    let r21 = ri[20]; // (pp|p*p*)
    let r22 = ri[21]; // (p*p|p*p)

    match (pa, pb) {
        // ----- A = (s,s) -----
        (0, 0) => r1, // already handled above, but for completeness
        // (ss | px_B s_B)
        (0, 1) => r5 * x[0],
        // (ss | px_B px_B)
        (0, 2) => r11 * xx11 + r12 * yyzz11,
        // (ss | py_B s_B)
        (0, 3) => r5 * x[1],
        // (ss | py_B px_B)
        (0, 4) => r11 * xx21 + r12 * yyzz21,
        // (ss | py_B py_B)
        (0, 5) => r11 * xx22 + r12 * yyzz22,
        // (ss | pz_B s_B)
        (0, 6) => r5 * x[2],
        // (ss | pz_B px_B)
        (0, 7) => r11 * xx31 + r12 * zz31,
        // (ss | pz_B py_B)
        (0, 8) => r11 * xx32 + r12 * zz32,
        // (ss | pz_B pz_B)
        (0, 9) => r11 * xx33 + r12 * zz33,

        // ----- A = (px, s) -----
        // (px_A s_A | ss)
        (1, 0) => r2 * x[0],
        // (px_A s_A | px_B s_B)
        (1, 1) => r6 * xx11 + r7 * yyzz11,
        // (px_A s_A | px_B px_B)
        (1, 2) => x[0] * (r13 * xx11 + r14 * yyzz11) + r15 * (y[0] * xy11 + z[0] * xz11),
        // (px_A s_A | py_B s_B)
        (1, 3) => r6 * xx21 + r7 * yyzz21,
        // (px_A s_A | py_B px_B)
        (1, 4) => x[0] * (r13 * xx21 + r14 * yyzz21) + r15 * (y[0] * xy21 + z[0] * xz21),
        // (px_A s_A | py_B py_B)
        (1, 5) => x[0] * (r13 * xx22 + r14 * yyzz22) + r15 * (y[0] * xy22 + z[0] * xz22),
        // (px_A s_A | pz_B s_B)
        (1, 6) => r6 * xx31 + r7 * zz31,
        // (px_A s_A | pz_B px_B)
        (1, 7) => x[0] * (r13 * xx31 + r14 * zz31) + r15 * (y[0] * xy31 + z[0] * xz31),
        // (px_A s_A | pz_B py_B)
        (1, 8) => x[0] * (r13 * xx32 + r14 * zz32) + r15 * (y[0] * xy32 + z[0] * xz32),
        // (px_A s_A | pz_B pz_B)
        (1, 9) => x[0] * (r13 * xx33 + r14 * zz33) + r15 * (z[0] * xz33),

        // ----- A = (px, px) -----
        // (px_A px_A | ss)
        (2, 0) => r3 * xx11 + r4 * yyzz11,
        // (px_A px_A | px_B s_B)
        (2, 1) => x[0] * (r8 * xx11 + r9 * yyzz11) + r10 * (y[0] * xy11 + z[0] * xz11),
        // (px_A px_A | px_B px_B)
        (2, 2) => {
            (r16 * xx11 + r17 * yyzz11) * xx11 + r18 * xx11 * yyzz11
                + r19 * (yy11 * yy11 + zz11 * zz11)
                + r20 * (xy11 * xy11 + xz11 * xz11) + r21 * (yy11 * zz11 + zz11 * yy11)
                + r22 * yz11 * yz11
        }
        // (px_A px_A | py_B s_B)
        (2, 3) => x[1] * (r8 * xx11 + r9 * yyzz11) + r10 * (y[1] * xy11 + z[1] * xz11),
        // (px_A px_A | py_B px_B)
        (2, 4) => {
            (r16 * xx11 + r17 * yyzz11) * xx21 + r18 * xx11 * yyzz21
                + r19 * (yy11 * yy21 + zz11 * zz21)
                + r20 * (xy11 * xy21 + xz11 * xz21) + r21 * (yy11 * zz21 + zz11 * yy21)
                + r22 * yz11 * yz21
        }
        // (px_A px_A | py_B py_B)
        (2, 5) => {
            (r16 * xx11 + r17 * yyzz11) * xx22 + r18 * xx11 * yyzz22
                + r19 * (yy11 * yy22 + zz11 * zz22)
                + r20 * (xy11 * xy22 + xz11 * xz22) + r21 * (yy11 * zz22 + zz11 * yy22)
                + r22 * yz11 * yz22
        }
        // (px_A px_A | pz_B s_B)
        (2, 6) => x[2] * (r8 * xx11 + r9 * yyzz11) + r10 * (z[2] * xz11),
        // (px_A px_A | pz_B px_B)
        (2, 7) => {
            (r16 * xx11 + r17 * yyzz11) * xx31
                + (r18 * xx11 + r19 * zz11 + r21 * yy11) * zz31
                + r20 * (xy11 * xy31 + xz11 * xz31) + r22 * yz11 * yz31
        }
        // (px_A px_A | pz_B py_B)
        (2, 8) => {
            (r16 * xx11 + r17 * yyzz11) * xx32
                + (r18 * xx11 + r19 * zz11 + r21 * yy11) * zz32
                + r20 * (xy11 * xy32 + xz11 * xz32) + r22 * yz11 * yz32
        }
        // (px_A px_A | pz_B pz_B)
        (2, 9) => {
            (r16 * xx11 + r17 * yyzz11) * xx33
                + (r18 * xx11 + r19 * zz11 + r21 * yy11) * zz33
                + r20 * xz11 * xz33
        }

        // ----- A = (py, s) -----
        (3, 0) => r2 * x[1],
        (3, 1) => r6 * xx21 + r7 * yyzz21,
        (3, 2) => x[1] * (r13 * xx11 + r14 * yyzz11) + r15 * (y[1] * xy11 + z[1] * xz11),
        (3, 3) => r6 * xx22 + r7 * yyzz22,
        (3, 4) => x[1] * (r13 * xx21 + r14 * yyzz21) + r15 * (y[1] * xy21 + z[1] * xz21),
        (3, 5) => x[1] * (r13 * xx22 + r14 * yyzz22) + r15 * (y[1] * xy22 + z[1] * xz22),
        (3, 6) => r6 * xx32 + r7 * zz32,
        (3, 7) => x[1] * (r13 * xx31 + r14 * zz31) + r15 * (y[1] * xy31 + z[1] * xz31),
        (3, 8) => x[1] * (r13 * xx32 + r14 * zz32) + r15 * (y[1] * xy32 + z[1] * xz32),
        (3, 9) => x[1] * (r13 * xx33 + r14 * zz33) + r15 * (z[1] * xz33),

        // ----- A = (py, px) -----
        (4, 0) => r3 * xx21 + r4 * yyzz21,
        (4, 1) => x[0] * (r8 * xx21 + r9 * yyzz21) + r10 * (y[0] * xy21 + z[0] * xz21),
        (4, 2) => {
            (r16 * xx21 + r17 * yyzz21) * xx11 + r18 * xx21 * yyzz11
                + r19 * (yy21 * yy11 + zz21 * zz11)
                + r20 * (xy21 * xy11 + xz21 * xz11) + r21 * (yy21 * zz11 + zz21 * yy11)
                + r22 * yz21 * yz11
        }
        (4, 3) => x[1] * (r8 * xx21 + r9 * yyzz21) + r10 * (y[1] * xy21 + z[1] * xz21),
        (4, 4) => {
            (r16 * xx21 + r17 * yyzz21) * xx21 + r18 * xx21 * yyzz21
                + r19 * (yy21 * yy21 + zz21 * zz21)
                + r20 * (xy21 * xy21 + xz21 * xz21) + r21 * (yy21 * zz21 + zz21 * yy21)
                + r22 * yz21 * yz21
        }
        (4, 5) => {
            (r16 * xx21 + r17 * yyzz21) * xx22 + r18 * xx21 * yyzz22
                + r19 * (yy21 * yy22 + zz21 * zz22)
                + r20 * (xy21 * xy22 + xz21 * xz22) + r21 * (yy21 * zz22 + zz21 * yy22)
                + r22 * yz21 * yz22
        }
        (4, 6) => x[2] * (r8 * xx21 + r9 * yyzz21) + r10 * (z[2] * xz21),
        (4, 7) => {
            (r16 * xx21 + r17 * yyzz21) * xx31
                + (r18 * xx21 + r19 * zz21 + r21 * yy21) * zz31
                + r20 * (xy21 * xy31 + xz21 * xz31) + r22 * yz21 * yz31
        }
        (4, 8) => {
            (r16 * xx21 + r17 * yyzz21) * xx32
                + (r18 * xx21 + r19 * zz21 + r21 * yy21) * zz32
                + r20 * (xy21 * xy32 + xz21 * xz32) + r22 * yz21 * yz32
        }
        (4, 9) => {
            (r16 * xx21 + r17 * yyzz21) * xx33
                + (r18 * xx21 + r19 * zz21 + r21 * yy21) * zz33
                + r20 * xz21 * xz33
        }

        // ----- A = (py, py) -----
        (5, 0) => r3 * xx22 + r4 * yyzz22,
        (5, 1) => x[0] * (r8 * xx22 + r9 * yyzz22) + r10 * (y[0] * xy22 + z[0] * xz22),
        (5, 2) => {
            (r16 * xx22 + r17 * yyzz22) * xx11 + r18 * xx22 * yyzz11
                + r19 * (yy22 * yy11 + zz22 * zz11)
                + r20 * (xy22 * xy11 + xz22 * xz11) + r21 * (yy22 * zz11 + zz22 * yy11)
                + r22 * yz22 * yz11
        }
        (5, 3) => x[1] * (r8 * xx22 + r9 * yyzz22) + r10 * (y[1] * xy22 + z[1] * xz22),
        (5, 4) => {
            (r16 * xx22 + r17 * yyzz22) * xx21 + r18 * xx22 * yyzz21
                + r19 * (yy22 * yy21 + zz22 * zz21)
                + r20 * (xy22 * xy21 + xz22 * xz21) + r21 * (yy22 * zz21 + zz22 * yy21)
                + r22 * yz22 * yz21
        }
        (5, 5) => {
            (r16 * xx22 + r17 * yyzz22) * xx22 + r18 * xx22 * yyzz22
                + r19 * (yy22 * yy22 + zz22 * zz22)
                + r20 * (xy22 * xy22 + xz22 * xz22) + r21 * (yy22 * zz22 + zz22 * yy22)
                + r22 * yz22 * yz22
        }
        (5, 6) => x[2] * (r8 * xx22 + r9 * yyzz22) + r10 * (z[2] * xz22),
        (5, 7) => {
            (r16 * xx22 + r17 * yyzz22) * xx31
                + (r18 * xx22 + r19 * zz22 + r21 * yy22) * zz31
                + r20 * (xy22 * xy31 + xz22 * xz31) + r22 * yz22 * yz31
        }
        (5, 8) => {
            (r16 * xx22 + r17 * yyzz22) * xx32
                + (r18 * xx22 + r19 * zz22 + r21 * yy22) * zz32
                + r20 * (xy22 * xy32 + xz22 * xz32) + r22 * yz22 * yz32
        }
        (5, 9) => {
            (r16 * xx22 + r17 * yyzz22) * xx33
                + (r18 * xx22 + r19 * zz22 + r21 * yy22) * zz33
                + r20 * xz22 * xz33
        }

        // ----- A = (pz, s) -----
        (6, 0) => r2 * x[2],
        (6, 1) => r6 * xx31 + r7 * zz31,
        (6, 2) => x[2] * (r13 * xx11 + r14 * yyzz11) + r15 * (z[2] * xz11),
        (6, 3) => r6 * xx32 + r7 * zz32,
        (6, 4) => x[2] * (r13 * xx21 + r14 * yyzz21) + r15 * (z[2] * xz21),
        (6, 5) => x[2] * (r13 * xx22 + r14 * yyzz22) + r15 * (z[2] * xz22),
        (6, 6) => r6 * xx33 + r7 * zz33,
        (6, 7) => x[2] * (r13 * xx31 + r14 * zz31) + r15 * (z[2] * xz31),
        (6, 8) => x[2] * (r13 * xx32 + r14 * zz32) + r15 * (z[2] * xz32),
        (6, 9) => x[2] * (r13 * xx33 + r14 * zz33) + r15 * (z[2] * xz33),

        // ----- A = (pz, px) -----
        (7, 0) => r3 * xx31 + r4 * zz31,
        (7, 1) => x[0] * (r8 * xx31 + r9 * zz31) + r10 * (y[0] * xy31 + z[0] * xz31),
        (7, 2) => {
            (r16 * xx31 + r17 * zz31) * xx11 + r18 * xx31 * yyzz11
                + r19 * zz31 * zz11 + r20 * (xy31 * xy11 + xz31 * xz11)
                + r21 * zz31 * yy11 + r22 * yz31 * yz11
        }
        (7, 3) => x[1] * (r8 * xx31 + r9 * zz31) + r10 * (y[1] * xy31 + z[1] * xz31),
        (7, 4) => {
            (r16 * xx31 + r17 * zz31) * xx21 + r18 * xx31 * yyzz21
                + r19 * zz31 * zz21 + r20 * (xy31 * xy21 + xz31 * xz21)
                + r21 * zz31 * yy21 + r22 * yz31 * yz21
        }
        (7, 5) => {
            (r16 * xx31 + r17 * zz31) * xx22 + r18 * xx31 * yyzz22
                + r19 * zz31 * zz22 + r20 * (xy31 * xy22 + xz31 * xz22)
                + r21 * zz31 * yy22 + r22 * yz31 * yz22
        }
        (7, 6) => x[2] * (r8 * xx31 + r9 * zz31) + r10 * (z[2] * xz31),
        (7, 7) => {
            (r16 * xx31 + r17 * zz31) * xx31
                + (r18 * xx31 + r19 * zz31) * zz31
                + r20 * (xy31 * xy31 + xz31 * xz31)
                + r22 * yz31 * yz31
        }
        (7, 8) => {
            (r16 * xx31 + r17 * zz31) * xx32
                + (r18 * xx31 + r19 * zz31) * zz32
                + r20 * (xy31 * xy32 + xz31 * xz32)
                + r22 * yz31 * yz32
        }
        (7, 9) => {
            (r16 * xx31 + r17 * zz31) * xx33
                + (r18 * xx31 + r19 * zz31) * zz33
                + r20 * xz31 * xz33
        }

        // ----- A = (pz, py) -----
        (8, 0) => r3 * xx32 + r4 * zz32,
        (8, 1) => x[0] * (r8 * xx32 + r9 * zz32) + r10 * (y[0] * xy32 + z[0] * xz32),
        (8, 2) => {
            (r16 * xx32 + r17 * zz32) * xx11 + r18 * xx32 * yyzz11
                + r19 * zz32 * zz11 + r20 * (xy32 * xy11 + xz32 * xz11)
                + r21 * zz32 * yy11 + r22 * yz32 * yz11
        }
        (8, 3) => x[1] * (r8 * xx32 + r9 * zz32) + r10 * (y[1] * xy32 + z[1] * xz32),
        (8, 4) => {
            (r16 * xx32 + r17 * zz32) * xx21 + r18 * xx32 * yyzz21
                + r19 * zz32 * zz21 + r20 * (xy32 * xy21 + xz32 * xz21)
                + r21 * zz32 * yy21 + r22 * yz32 * yz21
        }
        (8, 5) => {
            (r16 * xx32 + r17 * zz32) * xx22 + r18 * xx32 * yyzz22
                + r19 * zz32 * zz22 + r20 * (xy32 * xy22 + xz32 * xz22)
                + r21 * zz32 * yy22 + r22 * yz32 * yz22
        }
        (8, 6) => x[2] * (r8 * xx32 + r9 * zz32) + r10 * (z[2] * xz32),
        (8, 7) => {
            (r16 * xx32 + r17 * zz32) * xx31 + (r18 * xx32 + r19 * zz32) * zz31
                + r20 * (xy32 * xy31 + xz32 * xz31) + r22 * yz32 * yz31
        }
        (8, 8) => {
            (r16 * xx32 + r17 * zz32) * xx32 + (r18 * xx32 + r19 * zz32) * zz32
                + r20 * (xy32 * xy32 + xz32 * xz32) + r22 * yz32 * yz32
        }
        (8, 9) => {
            (r16 * xx32 + r17 * zz32) * xx33 + (r18 * xx32 + r19 * zz32) * zz33
                + r20 * xz32 * xz33
        }

        // ----- A = (pz, pz) -----
        (9, 0) => r3 * xx33 + r4 * zz33,
        (9, 1) => x[0] * (r8 * xx33 + r9 * zz33) + r10 * (z[0] * xz33),
        (9, 2) => {
            (r16 * xx33 + r17 * zz33) * xx11 + r18 * xx33 * yyzz11
                + r19 * zz33 * zz11 + r20 * xz33 * xz11
                + r21 * zz33 * yy11
        }
        (9, 3) => x[1] * (r8 * xx33 + r9 * zz33) + r10 * (z[1] * xz33),
        (9, 4) => {
            (r16 * xx33 + r17 * zz33) * xx21 + r18 * xx33 * yyzz21
                + r19 * zz33 * zz21 + r20 * xz33 * xz21
                + r21 * zz33 * yy21
        }
        (9, 5) => {
            (r16 * xx33 + r17 * zz33) * xx22 + r18 * xx33 * yyzz22
                + r19 * zz33 * zz22 + r20 * xz33 * xz22 + r21 * zz33 * yy22
        }
        (9, 6) => x[2] * (r8 * xx33 + r9 * zz33) + r10 * (z[2] * xz33),
        (9, 7) => {
            (r16 * xx33 + r17 * zz33) * xx31 + (r18 * xx33 + r19 * zz33) * zz31
                + r20 * xz33 * xz31
        }
        (9, 8) => {
            (r16 * xx33 + r17 * zz33) * xx32 + (r18 * xx33 + r19 * zz33) * zz32
                + r20 * xz33 * xz32
        }
        (9, 9) => {
            (r16 * xx33 + r17 * zz33) * xx33 + (r18 * xx33 + r19 * zz33) * zz33
                + r20 * xz33 * xz33
        }

        _ => 0.0,
    }
}
