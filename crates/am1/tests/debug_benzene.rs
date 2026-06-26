//! Debug test for benzene symmetry issues.
//!
//! All C atoms in benzene should be equivalent by D6h symmetry.
//! This test checks intermediate quantities to find where symmetry breaks.

use rst_am1::compute_am1;

#[test]
fn debug_benzene_perfect_symmetry() {
    // Generate EXACTLY D6h symmetric benzene coordinates
    // C-C bond length = 1.395 A, C-H bond length = 1.0997 A
    let rcc = 1.395_f64;
    let rch = 1.0997_f64;
    let rc = rcc; // distance from center to C atom for regular hexagon = bond length
    let rh = rc + rch; // distance from center to H atom

    let atomic_numbers = vec![6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1];
    let mut coords: Vec<[f64; 3]> = Vec::new();

    // Place C atoms at 30, 90, 150, 210, 270, 330 degrees (matching SQM-like ordering)
    // Actually, let's use angles that match the approximate SQM geometry:
    // C1 at ~30 deg, C2 at ~330 deg, C3 at ~270 deg, etc.
    let angles_deg = [30.0, -30.0, -90.0, -150.0, 150.0, 90.0_f64];
    for &ang in &angles_deg {
        let rad = ang * std::f64::consts::PI / 180.0;
        coords.push([rc * rad.cos(), rc * rad.sin(), 0.0]);
    }
    for &ang in &angles_deg {
        let rad = ang * std::f64::consts::PI / 180.0;
        coords.push([rh * rad.cos(), rh * rad.sin(), 0.0]);
    }

    // Verify symmetry of coordinates
    println!("\nC-C bond distances (perfect hexagon):");
    for i in 0..6 {
        let j = (i + 1) % 6;
        let dx = coords[i][0] - coords[j][0];
        let dy = coords[i][1] - coords[j][1];
        let dz = coords[i][2] - coords[j][2];
        let d = (dx*dx + dy*dy + dz*dz).sqrt();
        println!("  C{}-C{}: {:.10} A", i+1, j+1, d);
    }

    println!("\nC-H bond distances:");
    for i in 0..6 {
        let h = i + 6;
        let dx = coords[i][0] - coords[h][0];
        let dy = coords[i][1] - coords[h][1];
        let dz = coords[i][2] - coords[h][2];
        let d = (dx*dx + dy*dy + dz*dz).sqrt();
        println!("  C{}-H{}: {:.10} A", i+1, h+1, d);
    }

    let result = compute_am1(&atomic_numbers, &coords, 0, None).unwrap();
    assert!(result.converged);

    println!("\nCharges (perfect D6h benzene):");
    for (i, q) in result.charges.iter().enumerate() {
        let elem = if i < 6 { "C" } else { "H" };
        println!("  {}{}:  {:.10}", elem, i+1, q);
    }

    // Check symmetry of charges
    let c_charges: Vec<f64> = result.charges[..6].to_vec();
    let c_mean = c_charges.iter().sum::<f64>() / 6.0;
    let c_max_dev = c_charges.iter().map(|c| (c - c_mean).abs()).fold(0.0_f64, f64::max);
    println!("\nC charge mean: {:.10}, max deviation: {:.10}", c_mean, c_max_dev);

    let h_charges: Vec<f64> = result.charges[6..].to_vec();
    let h_mean = h_charges.iter().sum::<f64>() / 6.0;
    let h_max_dev = h_charges.iter().map(|h| (h - h_mean).abs()).fold(0.0_f64, f64::max);
    println!("H charge mean: {:.10}, max deviation: {:.10}", h_mean, h_max_dev);

    println!("\nDensity matrix diagonal (C atoms):");
    for i in 0..6 {
        let off = i * 4;
        println!("  C{}: s={:.10}, px={:.10}, py={:.10}, pz={:.10}",
            i+1,
            result.density[(off, off)],
            result.density[(off+1, off+1)],
            result.density[(off+2, off+2)],
            result.density[(off+3, off+3)],
        );
    }

    // The pz orbital population should be the SAME for all C atoms
    // because all atoms are in the xy-plane and pz is the pi orbital
    println!("\npz populations:");
    let mut pz_pops = Vec::new();
    for i in 0..6 {
        let off = i * 4;
        let pz = result.density[(off+3, off+3)];
        pz_pops.push(pz);
        println!("  C{}: pz = {:.10}", i+1, pz);
    }
    let pz_mean = pz_pops.iter().sum::<f64>() / 6.0;
    let pz_max_dev = pz_pops.iter().map(|p| (p - pz_mean).abs()).fold(0.0_f64, f64::max);
    println!("pz mean: {:.10}, max deviation: {:.10}", pz_mean, pz_max_dev);

    // For a perfect benzene, all C charges should be identical
    // Tolerance should be very tight since the geometry is exact
    assert!(
        c_max_dev < 0.001,
        "Carbon charges should be symmetric. Max deviation from mean: {:.6}",
        c_max_dev
    );
}

#[test]
fn debug_benzene_core_hamiltonian() {
    // Near-perfect hexagonal benzene (SQM optimized)
    let atomic_numbers = vec![6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1];
    let coords: Vec<[f64; 3]> = vec![
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

    // Check that all C-C distances are equal
    let n = 6;
    let mut cc_dists: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = coords[i][0] - coords[j][0];
        let dy = coords[i][1] - coords[j][1];
        let dz = coords[i][2] - coords[j][2];
        let d = (dx*dx + dy*dy + dz*dz).sqrt() as f64;
        cc_dists.push((i, j, d));
    }
    println!("\nC-C bond distances:");
    for (i, j, d) in &cc_dists {
        println!("  C{}-C{}: {:.6} A", i+1, j+1, d);
    }

    // Check C-H distances
    for i in 0..6 {
        let h = i + 6;
        let dx = coords[i][0] - coords[h][0];
        let dy = coords[i][1] - coords[h][1];
        let dz = coords[i][2] - coords[h][2];
        let d: f64 = (dx*dx + dy*dy + dz*dz).sqrt();
        println!("  C{}-H{}: {:.6} A", i+1, h+1, d);
    }

    // Run the calculation
    let result = compute_am1(&atomic_numbers, &coords, 0, None).unwrap();
    assert!(result.converged);

    println!("\nCharges:");
    for (i, q) in result.charges.iter().enumerate() {
        let elem = if i < 6 { "C" } else { "H" };
        println!("  {}{}:  {:.6}", elem, i+1, q);
    }

    // Check symmetry of the density matrix
    // All C atoms should have the same diagonal density elements
    // (basis size is 4 per C, 1 per H = 30)
    println!("\nDensity matrix diagonal (C atoms):");
    for i in 0..6 {
        let off = i * 4;
        println!("  C{}: s={:.6}, px={:.6}, py={:.6}, pz={:.6}",
            i+1,
            result.density[(off, off)],
            result.density[(off+1, off+1)],
            result.density[(off+2, off+2)],
            result.density[(off+3, off+3)],
        );
    }

    println!("\nDensity matrix diagonal (H atoms):");
    for i in 0..6 {
        let off = 24 + i; // H atoms start at basis 24
        println!("  H{}: s={:.6}", i+7, result.density[(off, off)]);
    }

    // Check total population per C atom (should be same for all)
    println!("\nTotal electron population per C atom:");
    for i in 0..6 {
        let off = i * 4;
        let pop: f64 = (0..4).map(|k| result.density[(off+k, off+k)]).sum();
        println!("  C{}: {:.6} (charge = {:.6})", i+1, pop, 4.0 - pop);
    }
}
