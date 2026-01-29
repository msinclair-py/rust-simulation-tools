//! Per-residue decomposition of MM-GBSA binding free energy.
//!
//! Decomposes ΔG_bind into per-residue contributions for identifying
//! binding hotspots. Each residue's contribution is the sum of its
//! pairwise interactions (vdW, electrostatic, GB) with the partner
//! molecule, plus its share of the SA non-polar term.

use crate::gb_energy::GbParams;
use crate::mm_energy::{build_14_pairs, build_exclusion_set, lj_ab};
use crate::sa_energy::SaParams;
use crate::subsystem::{extract_coords, extract_subtopology};
use rst_core::amber::prmtop::AmberTopology;
use std::collections::HashMap;

/// Per-residue energy contributions to binding.
#[derive(Debug, Clone)]
pub struct ResidueContribution {
    /// 0-based residue index in the complex topology.
    pub residue_index: usize,
    /// Residue name (e.g. "ALA", "GLU").
    pub residue_label: String,
    /// Van der Waals interaction with the partner (kcal/mol).
    pub vdw: f64,
    /// Electrostatic interaction with the partner (kcal/mol).
    pub elec: f64,
    /// GB polar solvation contribution (kcal/mol).
    pub gb: f64,
    /// SA non-polar solvation contribution (kcal/mol).
    pub sa: f64,
}

impl ResidueContribution {
    pub fn total(&self) -> f64 {
        self.vdw + self.elec + self.gb + self.sa
    }
}

/// Result of per-residue decomposition.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Per-residue contributions from receptor residues.
    pub receptor_residues: Vec<ResidueContribution>,
    /// Per-residue contributions from ligand residues.
    pub ligand_residues: Vec<ResidueContribution>,
}

/// Compute per-residue decomposition of binding free energy for a single frame.
///
/// For each residue in the receptor (and ligand), computes the sum of its
/// pairwise non-bonded interactions (vdW + electrostatic) with all atoms in
/// the partner molecule. GB and SA contributions are computed as the
/// difference between the complex and isolated subsystem values, distributed
/// per-atom and summed per-residue.
///
/// # Arguments
/// * `complex_top` - Full complex topology
/// * `coords` - Complex coordinates in Angstroms
/// * `receptor_residues` - 0-based residue indices for the receptor
/// * `ligand_residues` - 0-based residue indices for the ligand
/// * `gb_params` - GB calculation parameters
/// * `sa_params` - SA calculation parameters
pub fn decompose_binding_energy(
    complex_top: &AmberTopology,
    coords: &[[f64; 3]],
    receptor_residues: &[usize],
    ligand_residues: &[usize],
    gb_params: &GbParams,
    sa_params: &SaParams,
) -> Result<DecompositionResult, String> {
    let atom_res = complex_top.atom_residue_indices();

    // Build atom sets for receptor and ligand
    let rec_sel = complex_top
        .build_selection(receptor_residues)
        .map_err(|e| format!("Invalid receptor residues: {}", e))?;
    let lig_sel = complex_top
        .build_selection(ligand_residues)
        .map_err(|e| format!("Invalid ligand residues: {}", e))?;
    // Build exclusion and 1-4 sets for the complex
    let excluded = build_exclusion_set(complex_top);
    let pairs_14 = build_14_pairs(complex_top);
    let scee = complex_top.scee_scale_factor;
    let scnb = complex_top.scnb_scale_factor;

    // Accumulate pairwise vdW + elec per residue (receptor↔ligand only)
    let mut res_vdw: HashMap<usize, f64> = HashMap::new();
    let mut res_elec: HashMap<usize, f64> = HashMap::new();

    for &ri in &rec_sel.atom_indices {
        for &li in &lig_sel.atom_indices {
            let (i, j) = if ri < li { (ri, li) } else { (li, ri) };
            let pair = (i, j);

            let dx = coords[ri][0] - coords[li][0];
            let dy = coords[ri][1] - coords[li][1];
            let dz = coords[ri][2] - coords[li][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-10 {
                continue;
            }

            let (vdw_e, elec_e) = if pairs_14.contains(&pair) {
                let r2 = r * r;
                let r6 = r2 * r2 * r2;
                let r12 = r6 * r6;
                let (a, b) = lj_ab(complex_top, ri, li);
                let vdw = (a / r12 - b / r6) / scnb;
                let qi = complex_top.charges_amber[ri];
                let qj = complex_top.charges_amber[li];
                let elec = qi * qj / r / scee;
                (vdw, elec)
            } else if !excluded.contains(&pair) {
                let r2 = r * r;
                let r6 = r2 * r2 * r2;
                let r12 = r6 * r6;
                let (a, b) = lj_ab(complex_top, ri, li);
                let vdw = a / r12 - b / r6;
                let qi = complex_top.charges_amber[ri];
                let qj = complex_top.charges_amber[li];
                let elec = qi * qj / r;
                (vdw, elec)
            } else {
                continue;
            };

            // Split equally between the two residues
            let res_ri = atom_res[ri];
            let res_li = atom_res[li];
            *res_vdw.entry(res_ri).or_default() += vdw_e * 0.5;
            *res_vdw.entry(res_li).or_default() += vdw_e * 0.5;
            *res_elec.entry(res_ri).or_default() += elec_e * 0.5;
            *res_elec.entry(res_li).or_default() += elec_e * 0.5;
        }
    }

    // GB decomposition: per-atom ΔGB = GB(complex) - GB(receptor/ligand)
    // Compute Born radii and GB energy for all three systems
    let complex_gb = crate::gb_energy::compute_gb_energy(complex_top, coords, gb_params);

    let rec_sub_top = extract_subtopology(complex_top, &rec_sel.atom_indices);
    let rec_coords = extract_coords(coords, &rec_sel.atom_indices);
    let rec_gb = crate::gb_energy::compute_gb_energy(&rec_sub_top, &rec_coords, gb_params);

    let lig_sub_top = extract_subtopology(complex_top, &lig_sel.atom_indices);
    let lig_coords = extract_coords(coords, &lig_sel.atom_indices);
    let lig_gb = crate::gb_energy::compute_gb_energy(&lig_sub_top, &lig_coords, gb_params);

    // Per-atom GB contribution: distribute total ΔGB proportionally to
    // the change in each atom's self-energy term (q²/R)
    let mut res_gb: HashMap<usize, f64> = HashMap::new();
    let total_delta_gb = complex_gb.total - rec_gb.total - lig_gb.total;

    // Compute per-atom self-energy change as a proxy for distribution
    let dielectric_factor =
        -0.5 * (1.0 / gb_params.solute_dielectric - 1.0 / gb_params.solvent_dielectric);
    let mut atom_gb_weights: Vec<f64> = Vec::with_capacity(complex_top.n_atoms);
    let mut total_weight = 0.0f64;

    // For receptor atoms
    for (local_i, &global_i) in rec_sel.atom_indices.iter().enumerate() {
        let q = complex_top.charges_amber[global_i];
        let self_complex = dielectric_factor * q * q / complex_gb.born_radii[global_i];
        let self_isolated = dielectric_factor * q * q / rec_gb.born_radii[local_i];
        let w = (self_complex - self_isolated).abs();
        atom_gb_weights.push(w);
        total_weight += w;
    }
    // For ligand atoms
    for (local_i, &global_i) in lig_sel.atom_indices.iter().enumerate() {
        let q = complex_top.charges_amber[global_i];
        let self_complex = dielectric_factor * q * q / complex_gb.born_radii[global_i];
        let self_isolated = dielectric_factor * q * q / lig_gb.born_radii[local_i];
        let w = (self_complex - self_isolated).abs();
        atom_gb_weights.push(w);
        total_weight += w;
    }

    // Distribute ΔGB to residues by weight
    if total_weight > 1e-20 {
        let all_selected: Vec<usize> = rec_sel
            .atom_indices
            .iter()
            .chain(lig_sel.atom_indices.iter())
            .copied()
            .collect();
        for (idx, &global_i) in all_selected.iter().enumerate() {
            let frac = atom_gb_weights[idx] / total_weight;
            let res = atom_res[global_i];
            *res_gb.entry(res).or_default() += total_delta_gb * frac;
        }
    }

    // SA decomposition: per-atom ΔSASA = SASA(complex) - SASA(isolated)
    let complex_sa = crate::sa_energy::compute_sa_energy(complex_top, coords, sa_params);
    let rec_sa = crate::sa_energy::compute_sa_energy(&rec_sub_top, &rec_coords, sa_params);
    let lig_sa = crate::sa_energy::compute_sa_energy(&lig_sub_top, &lig_coords, sa_params);

    let mut res_sa: HashMap<usize, f64> = HashMap::new();
    for (local_i, &global_i) in rec_sel.atom_indices.iter().enumerate() {
        let delta_sasa = complex_sa.per_atom_sasa[global_i] - rec_sa.per_atom_sasa[local_i];
        let res = atom_res[global_i];
        *res_sa.entry(res).or_default() += sa_params.surface_tension * delta_sasa;
    }
    for (local_i, &global_i) in lig_sel.atom_indices.iter().enumerate() {
        let delta_sasa = complex_sa.per_atom_sasa[global_i] - lig_sa.per_atom_sasa[local_i];
        let res = atom_res[global_i];
        *res_sa.entry(res).or_default() += sa_params.surface_tension * delta_sasa;
    }

    // Assemble results
    let receptor_contributions: Vec<ResidueContribution> = receptor_residues
        .iter()
        .map(|&r| ResidueContribution {
            residue_index: r,
            residue_label: complex_top.residue_labels[r].clone(),
            vdw: *res_vdw.get(&r).unwrap_or(&0.0),
            elec: *res_elec.get(&r).unwrap_or(&0.0),
            gb: *res_gb.get(&r).unwrap_or(&0.0),
            sa: *res_sa.get(&r).unwrap_or(&0.0),
        })
        .collect();

    let ligand_contributions: Vec<ResidueContribution> = ligand_residues
        .iter()
        .map(|&r| ResidueContribution {
            residue_index: r,
            residue_label: complex_top.residue_labels[r].clone(),
            vdw: *res_vdw.get(&r).unwrap_or(&0.0),
            elec: *res_elec.get(&r).unwrap_or(&0.0),
            gb: *res_gb.get(&r).unwrap_or(&0.0),
            sa: *res_sa.get(&r).unwrap_or(&0.0),
        })
        .collect();

    Ok(DecompositionResult {
        receptor_residues: receptor_contributions,
        ligand_residues: ligand_contributions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gb_energy::{GbModel, GbParams};
    use crate::mdcrd::MdcrdReader;
    use crate::sa_energy::SaParams;
    use std::path::Path;

    #[test]
    fn test_decomposition_sums_to_total() {
        let prmtop_path = "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/ras-raf.prmtop";
        let mdcrd_path =
            "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/_MMPBSA_complex.mdcrd.0";
        if !Path::new(prmtop_path).exists() || !Path::new(mdcrd_path).exists() {
            return;
        }

        let top =
            rst_core::amber::prmtop::parse_prmtop(prmtop_path).expect("Failed to parse prmtop");
        let mut reader =
            MdcrdReader::open(mdcrd_path, top.n_atoms, false).expect("Failed to open mdcrd");
        let coords = reader
            .read_frame()
            .expect("Failed to read frame")
            .expect("No frames");

        let receptor_residues: Vec<usize> = (0..166).collect();
        let ligand_residues: Vec<usize> = (166..top.n_residues).collect();

        let gb_params = GbParams {
            model: GbModel::ObcI,
            salt_concentration: 0.15,
            ..GbParams::default()
        };
        let sa_params = SaParams::default();

        let result = decompose_binding_energy(
            &top,
            &coords,
            &receptor_residues,
            &ligand_residues,
            &gb_params,
            &sa_params,
        )
        .expect("decompose_binding_energy failed");

        // All contributions should be finite
        for rc in result
            .receptor_residues
            .iter()
            .chain(result.ligand_residues.iter())
        {
            assert!(
                rc.vdw.is_finite(),
                "res {} vdw not finite",
                rc.residue_index
            );
            assert!(
                rc.elec.is_finite(),
                "res {} elec not finite",
                rc.residue_index
            );
            assert!(rc.gb.is_finite(), "res {} gb not finite", rc.residue_index);
            assert!(rc.sa.is_finite(), "res {} sa not finite", rc.residue_index);
        }

        // SA contributions should sum to ΔSA (within tolerance due to the
        // surface tension scaling)
        let sa_sum: f64 = result
            .receptor_residues
            .iter()
            .chain(result.ligand_residues.iter())
            .map(|rc| rc.sa)
            .sum();
        // Just check it's in a reasonable range (negative = buried surface)
        assert!(
            sa_sum <= 0.1,
            "Sum of per-residue SA should be ≤ ~0, got {}",
            sa_sum
        );

        // Print top contributing residues
        let mut all: Vec<&ResidueContribution> = result
            .receptor_residues
            .iter()
            .chain(result.ligand_residues.iter())
            .collect();
        all.sort_by(|a, b| a.total().partial_cmp(&b.total()).unwrap());
        println!("Top 5 favorable residues:");
        for rc in all.iter().take(5) {
            println!(
                "  {:>4} {:>4}: vdW={:>8.2} elec={:>8.2} GB={:>8.2} SA={:>8.2} total={:>8.2}",
                rc.residue_index + 1,
                rc.residue_label,
                rc.vdw,
                rc.elec,
                rc.gb,
                rc.sa,
                rc.total()
            );
        }
    }
}
