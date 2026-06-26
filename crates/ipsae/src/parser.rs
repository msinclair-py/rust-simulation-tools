use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::{ChainType, Structure};

const NUCLEIC_ACIDS: &[&str] = &["DA", "DC", "DT", "DG", "A", "C", "U", "G"];

pub fn parse_structure(path: &Path) -> Result<Structure, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    match ext {
        "pdb" => parse_pdb(&content),
        "cif" => parse_cif(&content),
        _ => Err(format!("Unsupported file extension: {}", ext)),
    }
}

fn parse_pdb(content: &str) -> Result<Structure, String> {
    let mut coords = Vec::new();
    let mut chains = Vec::new();
    let mut residue_names = Vec::new();

    for line in content.lines() {
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            continue;
        }
        if line.len() < 54 {
            continue;
        }

        let atom_name = line[12..16].trim();
        if atom_name != "CA" && !atom_name.contains("C1") {
            continue;
        }

        let res_name = line[17..20].trim().to_string();
        let chain_id = line[21..22].to_string();
        let x: f64 = line[30..38]
            .trim()
            .parse()
            .map_err(|e| format!("Bad x coord: {}", e))?;
        let y: f64 = line[38..46]
            .trim()
            .parse()
            .map_err(|e| format!("Bad y coord: {}", e))?;
        let z: f64 = line[46..54]
            .trim()
            .parse()
            .map_err(|e| format!("Bad z coord: {}", e))?;

        coords.push([x, y, z]);
        chains.push(chain_id);
        residue_names.push(res_name);
    }

    let chain_types = classify_chains(&chains, &residue_names);

    Ok(Structure {
        coords,
        chains,
        residue_names,
        chain_types,
    })
}

fn parse_cif(content: &str) -> Result<Structure, String> {
    let mut fields: HashMap<String, usize> = HashMap::new();
    let mut field_num = 0;
    let mut coords = Vec::new();
    let mut chains = Vec::new();
    let mut residue_names = Vec::new();

    for line in content.lines() {
        if line.starts_with("_atom_site.") {
            if let Some(field_name) = line.trim().split('.').nth(1) {
                fields.insert(field_name.to_string(), field_num);
                field_num += 1;
            }
            continue;
        }

        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        let atom_name_idx = *fields
            .get("label_atom_id")
            .ok_or("Missing label_atom_id field")?;
        let res_name_idx = *fields
            .get("label_comp_id")
            .ok_or("Missing label_comp_id field")?;
        let chain_idx = *fields
            .get("label_asym_id")
            .ok_or("Missing label_asym_id field")?;
        let resid_idx = *fields
            .get("label_seq_id")
            .ok_or("Missing label_seq_id field")?;
        let x_idx = *fields.get("Cartn_x").ok_or("Missing Cartn_x field")?;
        let y_idx = *fields.get("Cartn_y").ok_or("Missing Cartn_y field")?;
        let z_idx = *fields.get("Cartn_z").ok_or("Missing Cartn_z field")?;

        let max_idx = *[
            atom_name_idx,
            res_name_idx,
            chain_idx,
            resid_idx,
            x_idx,
            y_idx,
            z_idx,
        ]
        .iter()
        .max()
        .unwrap();

        if parts.len() <= max_idx {
            continue;
        }

        let resid = parts[resid_idx];
        if resid == "." {
            continue;
        }

        let atom_name = parts[atom_name_idx];
        if atom_name != "CA" && !atom_name.contains("C1") {
            continue;
        }

        let x: f64 = parts[x_idx]
            .parse()
            .map_err(|e| format!("Bad x coord: {}", e))?;
        let y: f64 = parts[y_idx]
            .parse()
            .map_err(|e| format!("Bad y coord: {}", e))?;
        let z: f64 = parts[z_idx]
            .parse()
            .map_err(|e| format!("Bad z coord: {}", e))?;

        coords.push([x, y, z]);
        chains.push(parts[chain_idx].to_string());
        residue_names.push(parts[res_name_idx].to_string());
    }

    let chain_types = classify_chains(&chains, &residue_names);

    Ok(Structure {
        coords,
        chains,
        residue_names,
        chain_types,
    })
}

fn classify_chains(
    chains: &[String],
    residue_names: &[String],
) -> HashMap<String, ChainType> {
    let na_set: HashSet<&str> = NUCLEIC_ACIDS.iter().copied().collect();
    let mut chain_types = HashMap::new();

    let unique_chains: Vec<&String> = {
        let mut seen = HashSet::new();
        chains.iter().filter(|c| seen.insert(*c)).collect()
    };

    for chain in unique_chains {
        let has_na = chains
            .iter()
            .zip(residue_names.iter())
            .filter(|(c, _)| *c == chain)
            .any(|(_, res)| na_set.contains(res.as_str()));

        chain_types.insert(
            chain.clone(),
            if has_na {
                ChainType::NucleicAcid
            } else {
                ChainType::Protein
            },
        );
    }

    chain_types
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pdb_line() {
        let pdb = "\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00
ATOM      2  CB  ALA A   1       4.000   5.000   6.000  1.00  0.00
ATOM      3  CA  GLY B   2       7.000   8.000   9.000  1.00  0.00
";
        let s = parse_pdb(pdb).unwrap();
        assert_eq!(s.coords.len(), 2); // CB should be skipped
        assert_eq!(s.chains, vec!["A", "B"]);
        assert_eq!(s.residue_names, vec!["ALA", "GLY"]);
        assert!((s.coords[0][0] - 1.0).abs() < 1e-6);
        assert!((s.coords[1][2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_pdb_nucleic_acid() {
        let pdb = "\
ATOM      1  C1' DA  A   1       1.000   2.000   3.000  1.00  0.00
ATOM      2  CA  ALA B   1       4.000   5.000   6.000  1.00  0.00
";
        let s = parse_pdb(pdb).unwrap();
        assert_eq!(s.coords.len(), 2);
        assert_eq!(s.chain_types["A"], ChainType::NucleicAcid);
        assert_eq!(s.chain_types["B"], ChainType::Protein);
    }

    #[test]
    fn test_parse_cif() {
        let cif = "\
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 CA ALA A 1 1.000 2.000 3.000
ATOM 2 CB ALA A 1 4.000 5.000 6.000
ATOM 3 CA GLY B 2 7.000 8.000 9.000
";
        let s = parse_cif(cif).unwrap();
        assert_eq!(s.coords.len(), 2);
        assert_eq!(s.chains, vec!["A", "B"]);
    }

    #[test]
    fn test_cif_skip_missing_resid() {
        let cif = "\
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 CA ALA A . 1.000 2.000 3.000
ATOM 2 CA GLY B 2 7.000 8.000 9.000
";
        let s = parse_cif(cif).unwrap();
        assert_eq!(s.coords.len(), 1);
        assert_eq!(s.chains, vec!["B"]);
    }

    #[test]
    fn test_classify_chains() {
        let chains = vec!["A".into(), "A".into(), "B".into()];
        let res_names = vec!["DA".into(), "DC".into(), "ALA".into()];
        let types = classify_chains(&chains, &res_names);
        assert_eq!(types["A"], ChainType::NucleicAcid);
        assert_eq!(types["B"], ChainType::Protein);
    }
}
