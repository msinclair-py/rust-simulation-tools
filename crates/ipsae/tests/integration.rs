use rst_ipsae::{ChainType, ScoringParams, Structure};
use std::io::Write;

fn make_test_pdb() -> String {
    // 2 chains, 4 residues each, positioned along x-axis
    // Chain A: residues at x=0,3,6,7  Chain B: residues at x=8,9,12,15
    // A[3] (x=7) to B[0] (x=8) = 1.0 Angstrom (strong contact)
    "\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00
ATOM      2  CA  ALA A   2       3.000   0.000   0.000  1.00  0.00
ATOM      3  CA  ALA A   3       6.000   0.000   0.000  1.00  0.00
ATOM      4  CA  ALA A   4       7.000   0.000   0.000  1.00  0.00
ATOM      5  CA  GLY B   1       8.000   0.000   0.000  1.00  0.00
ATOM      6  CA  GLY B   2       9.000   0.000   0.000  1.00  0.00
ATOM      7  CA  GLY B   3      12.000   0.000   0.000  1.00  0.00
ATOM      8  CA  GLY B   4      15.000   0.000   0.000  1.00  0.00
END
"
    .to_string()
}

fn make_test_data() -> (Vec<f64>, Vec<f64>) {
    let n = 8;
    let plddt = vec![90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0];

    // PAE: mostly 20 (high error), with low values near the interface
    let mut pae = vec![20.0; n * n];
    // Set intra-chain to low PAE
    for i in 0..4 {
        for j in 0..4 {
            pae[i * n + j] = 2.0;
        }
    }
    for i in 4..8 {
        for j in 4..8 {
            pae[i * n + j] = 2.0;
        }
    }
    // Set interface PAE (near contact region)
    pae[3 * n + 4] = 4.0; // A[3] -> B[0]
    pae[4 * n + 3] = 4.5; // B[0] -> A[3]
    pae[3 * n + 5] = 8.0; // A[3] -> B[1]
    pae[5 * n + 3] = 8.5;
    pae[2 * n + 4] = 9.0; // A[2] -> B[0]
    pae[4 * n + 2] = 9.5;

    (plddt, pae)
}

#[test]
fn test_full_pipeline_from_file() {
    let pdb_content = make_test_pdb();
    let dir = std::env::temp_dir().join("ipsae_test");
    std::fs::create_dir_all(&dir).unwrap();
    let pdb_path = dir.join("test.pdb");
    {
        let mut f = std::fs::File::create(&pdb_path).unwrap();
        f.write_all(pdb_content.as_bytes()).unwrap();
    }

    let (plddt, pae) = make_test_data();
    let params = ScoringParams::default();

    let result = rst_ipsae::compute_ipsae(&pdb_path, &plddt, &pae, &params).unwrap();

    // 2 chains -> 2 directed pairs (A->B, B->A)
    assert_eq!(result.directed_pairs.len(), 2);
    // 1 undirected max pair
    assert_eq!(result.max_pairs.len(), 1);

    // Verify chain labels
    let chains: Vec<(&str, &str)> = result
        .directed_pairs
        .iter()
        .map(|p| (p.chain1.as_str(), p.chain2.as_str()))
        .collect();
    assert!(chains.contains(&("A", "B")));
    assert!(chains.contains(&("B", "A")));

    // All scores should be in valid ranges
    for pair in &result.directed_pairs {
        assert!(pair.pdockq >= 0.0 && pair.pdockq <= 1.0);
        assert!(pair.pdockq2 >= 0.0 && pair.pdockq2 <= 2.0);
        assert!(pair.lis >= 0.0 && pair.lis <= 1.0);
        assert!(pair.iptm >= 0.0 && pair.iptm <= 1.0);
        assert!(pair.ipsae >= 0.0 && pair.ipsae <= 1.0);
    }

    // Contacts exist (A[3]->B[0] = 1 Å, A[2]->B[0] = 2 Å, A[3]->B[1] = 2 Å)
    let ab = result
        .directed_pairs
        .iter()
        .find(|p| p.chain1 == "A" && p.chain2 == "B")
        .unwrap();
    assert!(ab.pdockq > 0.0, "pDockQ should be > 0 with contacts");
    assert!(ab.pdockq2 > 0.0, "pDockQ2 should be > 0");
    assert!(ab.lis > 0.0, "LIS should be > 0");
    assert!(ab.iptm > 0.0, "ipTM should be > 0");
    assert!(ab.ipsae > 0.0, "ipSAE should be > 0");

    // Clean up
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_from_structure_directly() {
    let structure = Structure {
        coords: vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ],
        chains: vec![
            "A".into(),
            "A".into(),
            "A".into(),
            "A".into(),
            "B".into(),
            "B".into(),
            "B".into(),
            "B".into(),
        ],
        residue_names: vec![
            "ALA".into(),
            "ALA".into(),
            "ALA".into(),
            "ALA".into(),
            "GLY".into(),
            "GLY".into(),
            "GLY".into(),
            "GLY".into(),
        ],
        chain_types: [
            ("A".into(), ChainType::Protein),
            ("B".into(), ChainType::Protein),
        ]
        .into_iter()
        .collect(),
    };

    let (plddt, pae) = make_test_data();
    let params = ScoringParams::default();

    let result =
        rst_ipsae::scoring::compute_ipsae_scores(&structure, &plddt, &pae, &params).unwrap();

    assert_eq!(result.directed_pairs.len(), 2);
    assert_eq!(result.max_pairs.len(), 1);
}

#[test]
fn test_three_chains() {
    let structure = Structure {
        coords: vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [103.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
        ],
        chains: vec![
            "A".into(),
            "A".into(),
            "B".into(),
            "B".into(),
            "C".into(),
            "C".into(),
        ],
        residue_names: vec![
            "ALA".into(),
            "ALA".into(),
            "GLY".into(),
            "GLY".into(),
            "VAL".into(),
            "VAL".into(),
        ],
        chain_types: [
            ("A".into(), ChainType::Protein),
            ("B".into(), ChainType::Protein),
            ("C".into(), ChainType::Protein),
        ]
        .into_iter()
        .collect(),
    };

    let n = 6;
    let plddt = vec![90.0; n];
    let pae = vec![5.0; n * n];
    let params = ScoringParams::default();

    let result =
        rst_ipsae::scoring::compute_ipsae_scores(&structure, &plddt, &pae, &params).unwrap();

    // 3 chains -> 6 directed pairs (3P2), 3 max pairs
    assert_eq!(result.directed_pairs.len(), 6);
    assert_eq!(result.max_pairs.len(), 3);
}
