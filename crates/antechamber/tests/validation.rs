//! Validation tests: compare our antechamber output against AmberTools.

use rst_antechamber::molecule::AcMolecule;
use rst_antechamber::{AntechamberConfig, ChargeMethod};

fn run_pipeline(sdf_content: &str, name: &str) -> rst_antechamber::AntechamberResult {
    let mols = rst_core::sdf::parse_sdf(sdf_content).expect("Failed to parse SDF");
    let ac_mol = AcMolecule::from_sdf(&mols[0]);

    let config = AntechamberConfig {
        net_charge: 0,
        charge_method: ChargeMethod::Am1Bcc,
    };

    rst_antechamber::run_antechamber(ac_mol, &config)
        .unwrap_or_else(|e| panic!("Antechamber failed for {}: {}", name, e))
}

const METHANOL_SDF: &str = "\
methanol
     RDKit          3D

  6  5  0  0  0  0  0  0  0  0999 V2000
    0.0464   -0.6416    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0464    0.7684    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4374   -1.0249    0.8949 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0835   -0.9902    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4374   -1.0249   -0.8949 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.9461    1.1118    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
  2  6  1  0
M  END
$$$$
";

const ACETIC_ACID_SDF: &str = "\
acetic_acid
     RDKit          3D

  8  7  0  0  0  0  0  0  0  0999 V2000
   -0.0127    1.0858    0.0080 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0021   -0.4042    0.0020 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.9273   -0.9610   -0.0027 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1335   -1.0632    0.0060 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.0117    1.4454    0.0098 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5407    1.4454    0.8929 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5407    1.4454   -0.8769 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0448   -2.0232    0.0057 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  2  4  1  0
  1  5  1  0
  1  6  1  0
  1  7  1  0
  4  8  1  0
M  END
$$$$
";

const BENZENE_SDF: &str = "\
benzene
     RDKit          3D

 12 12  0  0  0  0  0  0  0  0999 V2000
    1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1560    1.2450    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1560   -1.2450    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -2.4900    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1560   -1.2450    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1560    1.2450    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    2.4900    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  4  0
  2  3  4  0
  3  4  4  0
  4  5  4  0
  5  6  4  0
  6  1  4  0
  1  7  1  0
  2  8  1  0
  3  9  1  0
  4 10  1  0
  5 11  1  0
  6 12  1  0
M  END
$$$$
";

const METHYLAMINE_SDF: &str = "\
methylamine
     RDKit          3D

  7  6  0  0  0  0  0  0  0  0999 V2000
   -0.0187    0.7454    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.7746    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.0171    1.1071    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5341    1.1071    0.8929 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5341    1.1071   -0.8929 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4697   -1.1271    0.8365 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4697   -1.1271   -0.8365 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
  2  6  1  0
  2  7  1  0
M  END
$$$$
";

#[test]
fn test_methanol_pipeline() {
    let result = run_pipeline(METHANOL_SDF, "methanol");
    let mol = &result.molecule;

    println!("\n=== METHANOL (RST) ===");
    let total_charge: f64 = mol.atoms.iter().map(|a| a.charge).sum();
    for (i, a) in mol.atoms.iter().enumerate() {
        println!(
            "  {:>2} {:<2} type={:<6} bcc={:<3} charge={:>10.6}",
            i + 1,
            a.element,
            a.gaff2_type,
            a.bcc_type,
            a.charge
        );
    }
    println!("  Total charge: {:.6}", total_charge);

    assert_eq!(mol.atoms.len(), 6);
    assert!(total_charge.abs() < 0.01, "Total charge should be ~0, got {}", total_charge);

    // GAFF2 types
    assert_eq!(mol.atoms[0].gaff2_type, "c3");  // C sp3
    assert_eq!(mol.atoms[1].gaff2_type, "oh");  // O hydroxyl

    // BCC types
    assert_eq!(mol.atoms[0].bcc_type, 11);  // C sp3
    assert_eq!(mol.atoms[1].bcc_type, 31);  // O generic
    for i in 2..6 { assert_eq!(mol.atoms[i].bcc_type, 91); }  // H

    // O should be negative, C slightly positive after BCC
    assert!(mol.atoms[1].charge < -0.3, "O charge should be < -0.3, got {}", mol.atoms[1].charge);
}

#[test]
fn test_acetic_acid_pipeline() {
    let result = run_pipeline(ACETIC_ACID_SDF, "acetic_acid");
    let mol = &result.molecule;

    println!("\n=== ACETIC ACID (RST) ===");
    let total_charge: f64 = mol.atoms.iter().map(|a| a.charge).sum();
    for (i, a) in mol.atoms.iter().enumerate() {
        println!(
            "  {:>2} {:<2} type={:<6} bcc={:<3} charge={:>10.6}",
            i + 1,
            a.element,
            a.gaff2_type,
            a.bcc_type,
            a.charge
        );
    }
    println!("  Total charge: {:.6}", total_charge);

    assert_eq!(mol.atoms.len(), 8);
    assert!(total_charge.abs() < 0.01, "Total charge should be ~0, got {}", total_charge);

    // GAFF2 types
    assert_eq!(mol.atoms[0].gaff2_type, "c3");  // methyl C
    assert_eq!(mol.atoms[1].gaff2_type, "c");   // carbonyl C
    assert_eq!(mol.atoms[2].gaff2_type, "o");   // =O
    assert_eq!(mol.atoms[3].gaff2_type, "oh");  // -OH

    // BCC types
    assert_eq!(mol.atoms[0].bcc_type, 11);  // C sp3
    assert_eq!(mol.atoms[1].bcc_type, 14);  // C sp2 with O1 neighbor
    assert_eq!(mol.atoms[2].bcc_type, 32);  // O=, neighbor C has O2
    assert_eq!(mol.atoms[3].bcc_type, 31);  // O-H

    // Carbonyl C should be most positive
    assert!(mol.atoms[1].charge > 0.5, "Carbonyl C should be > 0.5, got {}", mol.atoms[1].charge);
}

#[test]
fn test_benzene_pipeline() {
    let result = run_pipeline(BENZENE_SDF, "benzene");
    let mol = &result.molecule;

    println!("\n=== BENZENE (RST) ===");
    let total_charge: f64 = mol.atoms.iter().map(|a| a.charge).sum();
    for (i, a) in mol.atoms.iter().enumerate() {
        println!(
            "  {:>2} {:<2} type={:<6} bcc={:<3} charge={:>10.6}",
            i + 1,
            a.element,
            a.gaff2_type,
            a.bcc_type,
            a.charge
        );
    }
    println!("  Total charge: {:.6}", total_charge);

    assert_eq!(mol.atoms.len(), 12);
    assert!(total_charge.abs() < 0.01, "Total charge should be ~0, got {}", total_charge);

    // All carbons should be aromatic (ca) with BCC type 16
    for i in 0..6 {
        assert_eq!(mol.atoms[i].gaff2_type, "ca", "Benzene C{} type", i + 1);
        assert_eq!(mol.atoms[i].bcc_type, 16, "Benzene C{} BCC type", i + 1);
    }
    // All H should be ha with BCC type 91
    for i in 6..12 {
        assert_eq!(mol.atoms[i].gaff2_type, "ha", "Benzene H{} type", i - 5);
        assert_eq!(mol.atoms[i].bcc_type, 91, "Benzene H{} BCC type", i - 5);
    }

    // Benzene charges should be symmetric: C ~ -0.13, H ~ +0.13
    assert!((mol.atoms[0].charge - (-0.1298)).abs() < 0.01, "Benzene C charge");
    assert!((mol.atoms[6].charge - 0.1298).abs() < 0.01, "Benzene H charge");
}

#[test]
fn test_methylamine_pipeline() {
    let result = run_pipeline(METHYLAMINE_SDF, "methylamine");
    let mol = &result.molecule;

    println!("\n=== METHYLAMINE (RST) ===");
    let total_charge: f64 = mol.atoms.iter().map(|a| a.charge).sum();
    for (i, a) in mol.atoms.iter().enumerate() {
        println!(
            "  {:>2} {:<2} type={:<6} bcc={:<3} charge={:>10.6}",
            i + 1,
            a.element,
            a.gaff2_type,
            a.bcc_type,
            a.charge
        );
    }
    println!("  Total charge: {:.6}", total_charge);

    assert_eq!(mol.atoms.len(), 7);
    assert!(total_charge.abs() < 0.01, "Total charge should be ~0, got {}", total_charge);

    // GAFF2 types
    assert_eq!(mol.atoms[0].gaff2_type, "c3");  // C sp3
    assert_eq!(mol.atoms[1].gaff2_type, "n8");  // N amine (sp3, no pi)

    // BCC types: C=11, N=21 (not 23!), H=91
    assert_eq!(mol.atoms[0].bcc_type, 11);
    assert_eq!(mol.atoms[1].bcc_type, 21);  // N with 3 bonds, no double bond
    for i in 2..7 { assert_eq!(mol.atoms[i].bcc_type, 91); }

    // N should be strongly negative (BCC correction ~ -0.57)
    assert!(mol.atoms[1].charge < -0.8, "N charge should be < -0.8, got {}", mol.atoms[1].charge);
}
