pub mod parser;
pub mod scoring;

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainType {
    Protein,
    NucleicAcid,
}

pub struct Structure {
    pub coords: Vec<[f64; 3]>,
    pub chains: Vec<String>,
    pub residue_names: Vec<String>,
    pub chain_types: HashMap<String, ChainType>,
}

pub struct ScoringParams {
    pub pdockq_cutoff: f64,
    pub pae_cutoff: f64,
}

impl Default for ScoringParams {
    fn default() -> Self {
        Self {
            pdockq_cutoff: 8.0,
            pae_cutoff: 12.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChainPairScore {
    pub chain1: String,
    pub chain2: String,
    pub pdockq: f64,
    pub pdockq2: f64,
    pub lis: f64,
    pub iptm: f64,
    pub ipsae: f64,
}

pub struct IpsaeResult {
    pub directed_pairs: Vec<ChainPairScore>,
    pub max_pairs: Vec<ChainPairScore>,
}

pub fn compute_ipsae(
    structure_path: &Path,
    plddt: &[f64],
    pae: &[f64],
    params: &ScoringParams,
) -> Result<IpsaeResult, String> {
    let structure = parser::parse_structure(structure_path)?;
    scoring::compute_ipsae_scores(&structure, plddt, pae, params)
}
