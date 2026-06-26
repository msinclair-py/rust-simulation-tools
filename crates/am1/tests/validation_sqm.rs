//! Validation tests comparing our AM1 implementation against SQM (AMBER) reference output.
//!
//! These tests use the OPTIMIZED geometries from SQM, so we can compare single-point
//! AM1 charges and energies directly against the SQM final values.
//!
//! SQM reference: AMBER SQM VERSION 19, AM1 Hamiltonian.
//! The charges are Mulliken charges at the geometry-optimized structure.

use rst_am1::compute_am1;

/// Helper to print charges side by side for debugging.
fn print_comparison(elements: &[&str], our_charges: &[f64], ref_charges: &[f64]) {
    println!("  {:>3}  {:>6}  {:>10}  {:>10}  {:>10}", "Idx", "Elem", "Ours", "SQM Ref", "Diff");
    for (i, (elem, (ours, refs))) in elements
        .iter()
        .zip(our_charges.iter().zip(ref_charges.iter()))
        .enumerate()
    {
        println!(
            "  {:>3}  {:>6}  {:>10.6}  {:>10.6}  {:>10.6}",
            i + 1,
            elem,
            ours,
            refs,
            ours - refs
        );
    }
    let our_sum: f64 = our_charges.iter().sum();
    let ref_sum: f64 = ref_charges.iter().sum();
    println!("  Total charge: ours={:.6}, ref={:.6}", our_sum, ref_sum);
}

// ============================================================================
// Water: simplest multi-atom test (H, O only)
// ============================================================================

#[test]
fn test_water_charges_reasonable() {
    // Water molecule: O-H bond ~0.96 A, H-O-H angle ~104.5 deg
    let atomic_numbers = vec![8, 1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.587],
        [0.0, -0.757, 0.587],
    ];

    let result = compute_am1(&atomic_numbers, &coords, 0, None).unwrap();

    println!("\n=== Water ===");
    println!("Converged: {}, iterations: {}", result.converged, result.n_iterations);
    println!("Total energy: {:.6} eV", result.total_energy);
    println!("Electronic energy: {:.6} eV", result.electronic_energy);
    println!("Nuclear repulsion: {:.6} eV", result.nuclear_repulsion);
    println!("Heat of formation: {:.4} kcal/mol", result.heat_of_formation);
    println!("Charges: {:?}", result.charges);
    println!("Orbital energies: {:?}", result.orbital_energies);

    assert!(result.converged, "SCF should converge for water");

    // Water should have negative charge on O, positive on H
    assert!(result.charges[0] < -0.1, "Oxygen should be negative, got {}", result.charges[0]);
    assert!(result.charges[1] > 0.05, "Hydrogen should be positive, got {}", result.charges[1]);
    assert!(result.charges[2] > 0.05, "Hydrogen should be positive, got {}", result.charges[2]);

    // The two hydrogens should have equal charges (by symmetry)
    let h_diff = (result.charges[1] - result.charges[2]).abs();
    assert!(h_diff < 0.01, "H charges should be symmetric, diff = {}", h_diff);

    // Charges should sum to zero
    let total: f64 = result.charges.iter().sum();
    assert!(total.abs() < 1e-6, "Total charge should be zero, got {}", total);
}

// ============================================================================
// Benzene: SQM optimized geometry
// SQM reference charges: C = -0.130, H = +0.130
// SQM reference: Heat of formation = 21.93 kcal/mol
//                Total SCF energy = -850.34 eV
// ============================================================================

#[test]
fn test_benzene_sqm_comparison() {
    // SQM final (optimized) geometry for benzene
    let atomic_numbers = vec![6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1];
    let coords = vec![
        [1.2081, 0.6975, 0.0000],
        [1.2081, -0.6975, 0.0000],
        [0.0000, -1.3950, -0.0000],
        [-1.2081, -0.6975, -0.0000],
        [-1.2081, 0.6975, 0.0000],
        [0.0000, 1.3950, -0.0000],
        [2.1605, 1.2474, -0.0000],
        [2.1605, -1.2474, 0.0000],
        [0.0000, -2.4947, -0.0000],
        [-2.1605, -1.2474, -0.0000],
        [-2.1605, 1.2474, 0.0000],
        [-0.0000, 2.4947, -0.0000],
    ];

    let elements = vec!["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"];
    let sqm_charges = vec![
        -0.130, -0.130, -0.130, -0.130, -0.130, -0.130,
        0.130, 0.130, 0.130, 0.130, 0.130, 0.130,
    ];

    let result = compute_am1(&atomic_numbers, &coords, 0, None).unwrap();

    println!("\n=== Benzene (SQM optimized geometry) ===");
    println!("Converged: {}, iterations: {}", result.converged, result.n_iterations);
    println!("Total energy: {:.6} eV", result.total_energy);
    println!("Heat of formation: {:.4} kcal/mol", result.heat_of_formation);
    println!("SQM reference: Total SCF energy = -850.340 eV, HoF = 21.933 kcal/mol");
    print_comparison(&elements, &result.charges, &sqm_charges);

    assert!(result.converged, "SCF should converge for benzene");

    // All carbon charges should be equal (by symmetry) and negative
    let c_charges: Vec<f64> = result.charges[..6].to_vec();
    let c_mean = c_charges.iter().sum::<f64>() / 6.0;
    for (i, &c) in c_charges.iter().enumerate() {
        assert!(
            (c - c_mean).abs() < 0.005,
            "Carbon {} charge ({}) deviates from mean ({})",
            i, c, c_mean
        );
    }
    assert!(c_mean < 0.0, "Carbon charges should be negative");

    // All hydrogen charges should be equal (by symmetry) and positive
    let h_charges: Vec<f64> = result.charges[6..].to_vec();
    let h_mean = h_charges.iter().sum::<f64>() / 6.0;
    for (i, &h) in h_charges.iter().enumerate() {
        assert!(
            (h - h_mean).abs() < 0.005,
            "Hydrogen {} charge ({}) deviates from mean ({})",
            i, h, h_mean
        );
    }
    assert!(h_mean > 0.0, "Hydrogen charges should be positive");

    // Quantitative comparison with SQM (tolerance: 0.03 per atom)
    for (i, (ours, refs)) in result.charges.iter().zip(sqm_charges.iter()).enumerate() {
        let diff = (ours - refs).abs();
        assert!(
            diff < 0.03,
            "Atom {} charge differs from SQM by {:.4} (ours={:.4}, ref={:.4})",
            i, diff, ours, refs
        );
    }
}

// ============================================================================
// Methanol: SQM optimized geometry
// SQM reference charges:
//   C: -0.073, O: -0.326, H: 0.098, H: 0.053, H: 0.053, H(OH): 0.195
// SQM reference: Heat of formation = -57.067 kcal/mol
//                Total SCF energy = -504.006 eV
// ============================================================================

#[test]
fn test_methanol_sqm_comparison() {
    // SQM final (optimized) geometry for methanol
    let atomic_numbers = vec![6, 8, 1, 1, 1, 1];
    let coords = vec![
        [0.1104, -0.6471, 0.0544],
        [0.0855, 0.7521, 0.2299],
        [-0.6718, -1.0427, 0.7500],
        [1.1115, -1.0665, 0.3252],
        [-0.1405, -0.9224, -1.0005],
        [0.7519, 1.1246, -0.3590],
    ];

    let elements = vec!["C", "O", "H", "H", "H", "H"];
    let sqm_charges = vec![-0.073, -0.326, 0.098, 0.053, 0.053, 0.195];

    let result = compute_am1(&atomic_numbers, &coords, 0, None).unwrap();

    println!("\n=== Methanol (SQM optimized geometry) ===");
    println!("Converged: {}, iterations: {}", result.converged, result.n_iterations);
    println!("Total energy: {:.6} eV", result.total_energy);
    println!("Heat of formation: {:.4} kcal/mol", result.heat_of_formation);
    println!("SQM reference: Total SCF energy = -504.006 eV, HoF = -57.067 kcal/mol");
    print_comparison(&elements, &result.charges, &sqm_charges);

    assert!(result.converged, "SCF should converge for methanol");

    // Qualitative checks: O should be negative, C slightly negative, OH hydrogen most positive
    assert!(result.charges[1] < -0.1, "Oxygen should be significantly negative");
    assert!(result.charges[5] > 0.1, "OH hydrogen should be most positive");

    // Quantitative comparison (tolerance: 0.03 per atom)
    for (i, (ours, refs)) in result.charges.iter().zip(sqm_charges.iter()).enumerate() {
        let diff = (ours - refs).abs();
        assert!(
            diff < 0.03,
            "Atom {} ({}) charge differs from SQM by {:.4} (ours={:.4}, ref={:.4})",
            i, elements[i], diff, ours, refs
        );
    }
}

// ============================================================================
// Methylamine: SQM optimized geometry
// SQM reference charges:
//   C: -0.129, N: -0.351, H: 0.031, H: 0.082, H: 0.082, H(NH): 0.143, H(NH): 0.143
// SQM reference: Heat of formation = -7.420 kcal/mol
//                Total SCF energy = -404.134 eV
// ============================================================================

#[test]
fn test_methylamine_sqm_comparison() {
    // SQM final (optimized) geometry for methylamine
    let atomic_numbers = vec![6, 7, 1, 1, 1, 1, 1];
    let coords = vec![
        [0.0093, 0.6964, -0.0000],
        [-0.0046, -0.7359, 0.0001],
        [1.0400, 1.1498, -0.0002],
        [-0.5317, 1.0675, 0.9105],
        [-0.5318, 1.0674, -0.9105],
        [0.4446, -1.1040, 0.8145],
        [0.4442, -1.1042, -0.8144],
    ];

    let elements = vec!["C", "N", "H", "H", "H", "H", "H"];
    let sqm_charges = vec![-0.129, -0.351, 0.031, 0.082, 0.082, 0.143, 0.143];

    let result = compute_am1(&atomic_numbers, &coords, 0, None).unwrap();

    println!("\n=== Methylamine (SQM optimized geometry) ===");
    println!("Converged: {}, iterations: {}", result.converged, result.n_iterations);
    println!("Total energy: {:.6} eV", result.total_energy);
    println!("Heat of formation: {:.4} kcal/mol", result.heat_of_formation);
    println!("SQM reference: Total SCF energy = -404.134 eV, HoF = -7.420 kcal/mol");
    print_comparison(&elements, &result.charges, &sqm_charges);

    assert!(result.converged, "SCF should converge for methylamine");

    // Qualitative checks: N should be most negative, C slightly negative
    assert!(result.charges[1] < -0.1, "Nitrogen should be significantly negative");
    assert!(result.charges[0] < 0.0, "Carbon should be negative");

    // NH hydrogens should be more positive than CH hydrogens
    let h_nh_avg = (result.charges[5] + result.charges[6]) / 2.0;
    let h_ch_avg = (result.charges[3] + result.charges[4]) / 2.0;
    assert!(
        h_nh_avg > h_ch_avg,
        "NH hydrogens ({:.4}) should be more positive than CH hydrogens ({:.4})",
        h_nh_avg, h_ch_avg
    );

    // Quantitative comparison (tolerance: 0.03 per atom)
    for (i, (ours, refs)) in result.charges.iter().zip(sqm_charges.iter()).enumerate() {
        let diff = (ours - refs).abs();
        assert!(
            diff < 0.03,
            "Atom {} ({}) charge differs from SQM by {:.4} (ours={:.4}, ref={:.4})",
            i, elements[i], diff, ours, refs
        );
    }
}

// ============================================================================
// Acetic acid: SQM optimized geometry
// SQM reference charges:
//   C(methyl): -0.217, C(carboxyl): 0.306, O(carbonyl): -0.361,
//   O(hydroxyl): -0.321, H: 0.117, H: 0.117, H: 0.117, H(OH): 0.243
// SQM reference: Heat of formation = -103.022 kcal/mol
//                Total SCF energy = -952.907 eV
// ============================================================================

#[test]
fn test_acetic_acid_sqm_comparison() {
    // SQM final (optimized) geometry for acetic acid
    let atomic_numbers = vec![6, 6, 8, 8, 1, 1, 1, 1];
    let coords = vec![
        [-0.0268, 1.0895, 0.0056],
        [0.0420, -0.3951, 0.0055],
        [1.0314, -1.1327, 0.0046],
        [-1.1766, -1.0088, 0.0065],
        [1.0075, 1.5106, 0.0057],
        [-0.5764, 1.4396, 0.9135],
        [-0.5761, 1.4396, -0.9026],
        [-1.0569, -1.9726, 0.0063],
    ];

    let elements = vec!["C", "C", "O", "O", "H", "H", "H", "H"];
    let sqm_charges = vec![-0.217, 0.306, -0.361, -0.321, 0.117, 0.117, 0.117, 0.243];

    let result = compute_am1(&atomic_numbers, &coords, 0, None).unwrap();

    println!("\n=== Acetic Acid (SQM optimized geometry) ===");
    println!("Converged: {}, iterations: {}", result.converged, result.n_iterations);
    println!("Total energy: {:.6} eV", result.total_energy);
    println!("Heat of formation: {:.4} kcal/mol", result.heat_of_formation);
    println!("SQM reference: Total SCF energy = -952.907 eV, HoF = -103.022 kcal/mol");
    print_comparison(&elements, &result.charges, &sqm_charges);

    assert!(result.converged, "SCF should converge for acetic acid");

    // Qualitative checks
    assert!(result.charges[1] > 0.1, "Carboxyl carbon should be positive");
    assert!(result.charges[2] < -0.1, "Carbonyl oxygen should be negative");
    assert!(result.charges[3] < -0.1, "Hydroxyl oxygen should be negative");
    assert!(result.charges[7] > 0.1, "OH hydrogen should be most positive");

    // Quantitative comparison (tolerance: 0.03 per atom)
    for (i, (ours, refs)) in result.charges.iter().zip(sqm_charges.iter()).enumerate() {
        let diff = (ours - refs).abs();
        assert!(
            diff < 0.03,
            "Atom {} ({}) charge differs from SQM by {:.4} (ours={:.4}, ref={:.4})",
            i, elements[i], diff, ours, refs
        );
    }
}

// ============================================================================
// Energy comparison tests
// ============================================================================

#[test]
fn test_benzene_energy_sqm_comparison() {
    // Same benzene setup as above
    let atomic_numbers = vec![6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1];
    let coords = vec![
        [1.2081, 0.6975, 0.0000],
        [1.2081, -0.6975, 0.0000],
        [0.0000, -1.3950, -0.0000],
        [-1.2081, -0.6975, -0.0000],
        [-1.2081, 0.6975, 0.0000],
        [0.0000, 1.3950, -0.0000],
        [2.1605, 1.2474, -0.0000],
        [2.1605, -1.2474, 0.0000],
        [0.0000, -2.4947, -0.0000],
        [-2.1605, -1.2474, -0.0000],
        [-2.1605, 1.2474, 0.0000],
        [-0.0000, 2.4947, -0.0000],
    ];

    let result = compute_am1(&atomic_numbers, &coords, 0, None).unwrap();
    assert!(result.converged);

    // SQM reference energies:
    // Total SCF energy = -850.340 eV
    // Electronic energy = -3253.409 eV
    // Core-core repulsion = 2403.069 eV
    // Heat of formation = 21.933 kcal/mol

    println!("\n=== Benzene Energy Comparison ===");
    println!("  Total energy:     {:.3} eV (SQM: -850.340 eV)", result.total_energy);
    println!("  Electronic:       {:.3} eV (SQM: -3253.409 eV)", result.electronic_energy);
    println!("  Nuclear repulsion: {:.3} eV (SQM: 2403.069 eV)", result.nuclear_repulsion);
    println!("  Heat of formation: {:.3} kcal/mol (SQM: 21.933 kcal/mol)", result.heat_of_formation);

    // Energy should be within 1 eV (accounting for slight geometry differences
    // and the truncated AU_TO_EV)
    let energy_diff = (result.total_energy - (-850.340)).abs();
    assert!(
        energy_diff < 1.0,
        "Total energy differs from SQM by {:.3} eV",
        energy_diff
    );
}

// ============================================================================
// H2 molecule: simplest possible test case
// ============================================================================

#[test]
fn test_h2_charges_zero() {
    // H2 at ~0.74 A bond length
    let atomic_numbers = vec![1, 1];
    let coords = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74],
    ];

    let result = compute_am1(&atomic_numbers, &coords, 0, None).unwrap();

    println!("\n=== H2 ===");
    println!("Converged: {}, iterations: {}", result.converged, result.n_iterations);
    println!("Charges: {:?}", result.charges);
    println!("Total energy: {:.6} eV", result.total_energy);

    assert!(result.converged, "SCF should converge for H2");

    // Both charges should be zero by symmetry
    assert!(
        result.charges[0].abs() < 1e-6,
        "H1 charge should be zero, got {}",
        result.charges[0]
    );
    assert!(
        result.charges[1].abs() < 1e-6,
        "H2 charge should be zero, got {}",
        result.charges[1]
    );
}
