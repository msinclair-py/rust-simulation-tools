#![allow(clippy::too_many_arguments)]

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::{ChainPairScore, ChainType, IpsaeResult, ScoringParams, Structure};

pub fn compute_ipsae_scores(
    structure: &Structure,
    plddt: &[f64],
    pae: &[f64],
    params: &ScoringParams,
) -> Result<IpsaeResult, String> {
    let n = structure.coords.len();

    if plddt.len() != n {
        return Err(format!(
            "pLDDT length ({}) does not match number of residues ({})",
            plddt.len(),
            n
        ));
    }
    if pae.len() != n * n {
        return Err(format!(
            "PAE length ({}) does not match N*N ({})",
            pae.len(),
            n * n
        ));
    }

    // Compute distance matrix
    let dist = compute_distance_matrix(&structure.coords);

    // Get unique chains preserving order
    let unique_chains: Vec<String> = {
        let mut seen = HashSet::new();
        structure
            .chains
            .iter()
            .filter(|c| seen.insert((*c).clone()))
            .cloned()
            .collect()
    };

    if unique_chains.len() < 2 {
        return Ok(IpsaeResult {
            directed_pairs: Vec::new(),
            max_pairs: Vec::new(),
        });
    }

    // Generate directed pairs (permutations of 2)
    let mut chain_pairs = Vec::new();
    for i in 0..unique_chains.len() {
        for j in 0..unique_chains.len() {
            if i != j {
                chain_pairs.push((unique_chains[i].clone(), unique_chains[j].clone()));
            }
        }
    }

    // Pre-compute chain masks (residue indices belonging to each chain)
    let chain_masks: HashMap<String, Vec<usize>> = unique_chains
        .iter()
        .map(|c| {
            let mask: Vec<usize> = structure
                .chains
                .iter()
                .enumerate()
                .filter(|(_, ch)| *ch == c)
                .map(|(i, _)| i)
                .collect();
            (c.clone(), mask)
        })
        .collect();

    // Compute scores in parallel
    let directed_pairs: Vec<ChainPairScore> = chain_pairs
        .par_iter()
        .map(|(c1, c2)| {
            let mask1 = &chain_masks[c1];
            let mask2 = &chain_masks[c2];

            let (pdockq, pdockq2) =
                compute_pdockq(mask1, mask2, &dist, plddt, pae, n, params.pdockq_cutoff);
            let lis = compute_lis(mask1, mask2, pae, n, params.pae_cutoff);
            let (iptm, ipsae) = compute_iptm_ipsae(
                mask1,
                mask2,
                pae,
                n,
                &structure.chain_types,
                c1,
                c2,
                params.pae_cutoff,
            );

            ChainPairScore {
                chain1: c1.clone(),
                chain2: c2.clone(),
                pdockq,
                pdockq2,
                lis,
                iptm,
                ipsae,
            }
        })
        .collect();

    let max_pairs = extract_max_pairs(&directed_pairs);

    Ok(IpsaeResult {
        directed_pairs,
        max_pairs,
    })
}

fn compute_distance_matrix(coords: &[[f64; 3]]) -> Vec<f64> {
    let n = coords.len();
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

fn compute_pdockq(
    mask1: &[usize],
    mask2: &[usize],
    dist: &[f64],
    plddt: &[f64],
    pae: &[f64],
    n: usize,
    cutoff: f64,
) -> (f64, f64) {
    let mut n_contacts = 0u64;
    let mut c1_has_contact = vec![false; mask1.len()];
    let mut c2_has_contact = vec![false; mask2.len()];

    for (mi, &i) in mask1.iter().enumerate() {
        for (mj, &j) in mask2.iter().enumerate() {
            if dist[i * n + j] <= cutoff {
                n_contacts += 1;
                c1_has_contact[mi] = true;
                c2_has_contact[mj] = true;
            }
        }
    }

    if n_contacts == 0 {
        return (0.0, 0.0);
    }

    // Mean pLDDT of contacting residues
    let mut plddt_sum = 0.0;
    let mut plddt_count = 0usize;
    for (mi, &i) in mask1.iter().enumerate() {
        if c1_has_contact[mi] {
            plddt_sum += plddt[i];
            plddt_count += 1;
        }
    }
    for (mj, &j) in mask2.iter().enumerate() {
        if c2_has_contact[mj] {
            plddt_sum += plddt[j];
            plddt_count += 1;
        }
    }
    let mean_plddt = plddt_sum / plddt_count as f64;

    // pDockQ
    let x = mean_plddt * (n_contacts as f64).log10();
    let pdockq = pdockq_sigmoid(x);

    // pDockQ2: pTM of PAE at contact positions with d0=10
    let mut ptm_sum = 0.0;
    let mut ptm_count = 0usize;
    for &i in mask1 {
        for &j in mask2 {
            if dist[i * n + j] <= cutoff {
                ptm_sum += compute_ptm_value(pae[i * n + j], 10.0);
                ptm_count += 1;
            }
        }
    }
    let mean_ptm = ptm_sum / ptm_count as f64;
    let x2 = mean_plddt * mean_ptm;
    let pdockq2 = pdockq2_sigmoid(x2);

    (pdockq, pdockq2)
}

fn compute_lis(mask1: &[usize], mask2: &[usize], pae: &[f64], n: usize, cutoff: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;

    for &i in mask1 {
        for &j in mask2 {
            let val = pae[i * n + j];
            if val < cutoff {
                sum += (cutoff - val) / cutoff;
                count += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn compute_iptm_ipsae(
    mask1: &[usize],
    mask2: &[usize],
    pae: &[f64],
    n: usize,
    chain_types: &HashMap<String, ChainType>,
    chain1: &str,
    chain2: &str,
    pae_cutoff: f64,
) -> (f64, f64) {
    let pair_type = if chain_types.get(chain1) == Some(&ChainType::NucleicAcid)
        || chain_types.get(chain2) == Some(&ChainType::NucleicAcid)
    {
        ChainType::NucleicAcid
    } else {
        ChainType::Protein
    };

    let l = mask1.len() + mask2.len();
    let d0 = compute_d0(l, pair_type);

    // ipTM: max over rows (mask1) of mean over cols (mask2) of ptm
    let mut max_iptm = 0.0_f64;
    if !mask2.is_empty() {
        for &i in mask1 {
            let mut row_sum = 0.0;
            for &j in mask2 {
                row_sum += compute_ptm_value(pae[i * n + j], d0);
            }
            let row_mean = row_sum / mask2.len() as f64;
            if row_mean > max_iptm {
                max_iptm = row_mean;
            }
        }
    }

    // ipSAE: max over cols (mask2) of mean over valid rows (mask1 where pae < cutoff)
    let mut max_ipsae = 0.0_f64;
    for &j in mask2 {
        let mut col_sum = 0.0;
        let mut col_count = 0usize;
        for &i in mask1 {
            let pae_val = pae[i * n + j];
            if pae_val < pae_cutoff {
                col_sum += compute_ptm_value(pae_val, d0);
                col_count += 1;
            }
        }
        if col_count > 0 {
            let col_mean = col_sum / col_count as f64;
            if col_mean > max_ipsae {
                max_ipsae = col_mean;
            }
        }
    }

    (max_iptm, max_ipsae)
}

fn extract_max_pairs(directed: &[ChainPairScore]) -> Vec<ChainPairScore> {
    let mut best: HashMap<(String, String), &ChainPairScore> = HashMap::new();

    for score in directed {
        let key = if score.chain1 < score.chain2 {
            (score.chain1.clone(), score.chain2.clone())
        } else {
            (score.chain2.clone(), score.chain1.clone())
        };

        let entry = best.entry(key).or_insert(score);
        if score.ipsae > entry.ipsae {
            *entry = score;
        }
    }

    best.into_values().cloned().collect()
}

#[inline]
pub fn compute_ptm_value(pae: f64, d0: f64) -> f64 {
    1.0 / (1.0 + (pae / d0).powi(2))
}

pub fn compute_d0(l: usize, chain_type: ChainType) -> f64 {
    let l = l.max(27) as f64;
    let min_val = match chain_type {
        ChainType::Protein => 1.0,
        ChainType::NucleicAcid => 2.0,
    };
    (1.24 * (l - 15.0).cbrt() - 1.8).max(min_val)
}

fn pdockq_sigmoid(x: f64) -> f64 {
    0.724 / (1.0 + (-0.052 * (x - 152.611)).exp()) + 0.018
}

fn pdockq2_sigmoid(x: f64) -> f64 {
    1.31 / (1.0 + (-0.075 * (x - 84.733)).exp()) + 0.005
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_d0_protein() {
        // L=27: d0 = max(1.0, 1.24 * (12)^(1/3) - 1.8)
        let d0 = compute_d0(27, ChainType::Protein);
        let expected = 1.24 * 12.0_f64.cbrt() - 1.8;
        assert!((d0 - expected).abs() < 1e-10);
        assert!(d0 > 1.0);
    }

    #[test]
    fn test_compute_d0_protein_small_l() {
        // L < 27 should be clamped to 27
        let d0_small = compute_d0(10, ChainType::Protein);
        let d0_27 = compute_d0(27, ChainType::Protein);
        assert!((d0_small - d0_27).abs() < 1e-10);
    }

    #[test]
    fn test_compute_d0_nucleic_acid() {
        let d0 = compute_d0(27, ChainType::NucleicAcid);
        // min_val=2.0 for nucleic acid, formula gives ~1.04 which is < 2.0
        assert!((d0 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_d0_large_l() {
        let d0 = compute_d0(100, ChainType::Protein);
        let expected = 1.24 * 85.0_f64.cbrt() - 1.8;
        assert!((d0 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ptm_value() {
        // pTM(0, d0) = 1.0
        assert!((compute_ptm_value(0.0, 5.0) - 1.0).abs() < 1e-10);

        // pTM(d0, d0) = 0.5
        assert!((compute_ptm_value(5.0, 5.0) - 0.5).abs() < 1e-10);

        // pTM(10, 5) = 1/(1+(10/5)^2) = 1/(1+4) = 0.2
        assert!((compute_ptm_value(10.0, 5.0) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_pdockq_sigmoid() {
        // At x=152.611, sigmoid midpoint: 0.724/2 + 0.018 = 0.38
        let val = pdockq_sigmoid(152.611);
        assert!((val - 0.38).abs() < 1e-6);
    }

    #[test]
    fn test_pdockq2_sigmoid() {
        // At x=84.733, sigmoid midpoint: 1.31/2 + 0.005 = 0.66
        let val = pdockq2_sigmoid(84.733);
        assert!((val - 0.66).abs() < 1e-6);
    }

    #[test]
    fn test_distance_matrix() {
        let coords = vec![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
        let dist = compute_distance_matrix(&coords);
        assert!((dist[0 * 3 + 0]).abs() < 1e-10); // self-distance
        assert!((dist[0 * 3 + 1] - 5.0).abs() < 1e-10); // 3-4-5 triangle
        assert!((dist[0 * 3 + 2] - 5.0).abs() < 1e-10);
        assert!((dist[1 * 3 + 0] - 5.0).abs() < 1e-10); // symmetric
    }

    #[test]
    fn test_lis_known_values() {
        // 2 residues in chain1 (indices 0,1), 2 in chain2 (indices 2,3)
        // PAE matrix (4x4):
        let n = 4;
        let mut pae = vec![0.0; n * n];
        // PAE[0,2] = 6.0, PAE[0,3] = 15.0 (above cutoff)
        // PAE[1,2] = 3.0, PAE[1,3] = 9.0
        pae[0 * n + 2] = 6.0;
        pae[0 * n + 3] = 15.0;
        pae[1 * n + 2] = 3.0;
        pae[1 * n + 3] = 9.0;

        let mask1 = vec![0, 1];
        let mask2 = vec![2, 3];
        let cutoff = 12.0;

        let lis = compute_lis(&mask1, &mask2, &pae, n, cutoff);
        // Valid: (6, 3, 9). Scores: (6/12, 9/12, 3/12) = (0.5, 0.75, 0.25)
        // Mean = 1.5 / 3 = 0.5
        assert!((lis - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_zero_contacts() {
        // All distances > cutoff
        let mask1 = vec![0];
        let mask2 = vec![1];
        let n = 2;
        let dist = vec![0.0, 100.0, 100.0, 0.0];
        let plddt = vec![90.0, 85.0];
        let pae = vec![0.0; 4];

        let (pdockq, pdockq2) = compute_pdockq(&mask1, &mask2, &dist, &plddt, &pae, n, 8.0);
        assert!((pdockq).abs() < 1e-10);
        assert!((pdockq2).abs() < 1e-10);
    }

    #[test]
    fn test_single_chain_no_pairs() {
        let structure = Structure {
            coords: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            chains: vec!["A".into(), "A".into()],
            residue_names: vec!["ALA".into(), "GLY".into()],
            chain_types: [("A".into(), ChainType::Protein)].into(),
        };
        let plddt = vec![90.0, 85.0];
        let pae = vec![0.0; 4];
        let params = ScoringParams::default();

        let result = compute_ipsae_scores(&structure, &plddt, &pae, &params).unwrap();
        assert!(result.directed_pairs.is_empty());
        assert!(result.max_pairs.is_empty());
    }

    #[test]
    fn test_full_pipeline_two_chains() {
        // Chain A: residues 0,1 at (0,0,0) and (3,0,0)
        // Chain B: residues 2,3 at (6,0,0) and (9,0,0)
        let structure = Structure {
            coords: vec![
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ],
            chains: vec!["A".into(), "A".into(), "B".into(), "B".into()],
            residue_names: vec!["ALA".into(), "ALA".into(), "GLY".into(), "GLY".into()],
            chain_types: [
                ("A".into(), ChainType::Protein),
                ("B".into(), ChainType::Protein),
            ]
            .into(),
        };

        let plddt = vec![90.0, 85.0, 80.0, 75.0];
        // PAE: low for nearby, high for far
        let n = 4;
        let mut pae = vec![10.0; n * n];
        // Set diagonal to 0
        for i in 0..n {
            pae[i * n + i] = 0.0;
        }
        // Set interface PAE values
        pae[1 * n + 2] = 3.0; // A[1] -> B[2]: close, low PAE
        pae[2 * n + 1] = 3.5;

        let params = ScoringParams::default();
        let result = compute_ipsae_scores(&structure, &plddt, &pae, &params).unwrap();

        // Should have 2 directed pairs: A->B, B->A
        assert_eq!(result.directed_pairs.len(), 2);
        // Should have 1 max pair
        assert_eq!(result.max_pairs.len(), 1);

        // All scores should be finite
        for pair in &result.directed_pairs {
            assert!(pair.pdockq.is_finite());
            assert!(pair.pdockq2.is_finite());
            assert!(pair.lis.is_finite());
            assert!(pair.iptm.is_finite());
            assert!(pair.ipsae.is_finite());
        }

        // Distance from A[1] to B[2] = 3.0, which is < 8.0, so there are contacts
        // pDockQ should be > 0
        let ab = result
            .directed_pairs
            .iter()
            .find(|p| p.chain1 == "A" && p.chain2 == "B")
            .unwrap();
        assert!(ab.pdockq > 0.0);
        assert!(ab.pdockq2 > 0.0);

        // LIS should be > 0 (we have PAE values < 12 in the interface)
        assert!(ab.lis > 0.0);
    }
}
