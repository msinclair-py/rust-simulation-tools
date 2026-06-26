//! Unified force field loading.
//!
//! Combines embedded data with parsers to provide ready-to-use force field
//! parameter sets and residue libraries.

use std::path::Path;

use super::data;
use super::frcmod::parse_frcmod;
use super::parm_dat::parse_parm_dat;
use super::parameters::ForceFieldParams;
use super::residue_lib::{parse_lib, ResidueLibrary};

/// PDB residue name mapping for terminal residues.
/// Maps (position, PDB_name) -> AMBER_name where position: 0=N-terminal, 1=C-terminal
pub type PdbResMap = Vec<(u8, String, String)>;

/// A fully loaded force field with parameters and residue templates.
#[derive(Debug, Clone)]
pub struct LoadedForceField {
    pub params: ForceFieldParams,
    pub residue_lib: ResidueLibrary,
    pub pdb_res_map: PdbResMap,
}

impl LoadedForceField {
    pub fn new() -> Self {
        Self {
            params: ForceFieldParams::new(),
            residue_lib: ResidueLibrary::new(),
            pdb_res_map: Vec::new(),
        }
    }

    /// Load the ff19SB protein force field (parm19.dat + frcmod.ff19SB + amino acid libraries).
    pub fn load_protein_ff19sb(&mut self) -> Result<(), String> {
        // Parse base parameters
        let base = parse_parm_dat(data::PARM19_DAT)?;
        self.params.merge(&base);

        // Parse ff19SB modifications
        let frcmod = parse_frcmod(data::FRCMOD_FF19SB)?;
        self.params.merge(&frcmod);

        // Parse amino acid libraries
        let amino = parse_lib(data::AMINO19_LIB)?;
        self.residue_lib.merge(&amino);

        let aminoct = parse_lib(data::AMINOCT12_LIB)?;
        self.residue_lib.merge(&aminoct);

        let aminont = parse_lib(data::AMINONT12_LIB)?;
        self.residue_lib.merge(&aminont);

        // Set up PDB residue name map for terminal variants
        self.pdb_res_map.extend(vec![
            (0, "ALA".into(), "NALA".into()),
            (1, "ALA".into(), "CALA".into()),
            (0, "ARG".into(), "NARG".into()),
            (1, "ARG".into(), "CARG".into()),
            (0, "ASN".into(), "NASN".into()),
            (1, "ASN".into(), "CASN".into()),
            (0, "ASP".into(), "NASP".into()),
            (1, "ASP".into(), "CASP".into()),
            (0, "CYS".into(), "NCYS".into()),
            (1, "CYS".into(), "CCYS".into()),
            (0, "CYX".into(), "NCYX".into()),
            (1, "CYX".into(), "CCYX".into()),
            (0, "GLN".into(), "NGLN".into()),
            (1, "GLN".into(), "CGLN".into()),
            (0, "GLU".into(), "NGLU".into()),
            (1, "GLU".into(), "CGLU".into()),
            (0, "GLY".into(), "NGLY".into()),
            (1, "GLY".into(), "CGLY".into()),
            (0, "HID".into(), "NHID".into()),
            (1, "HID".into(), "CHID".into()),
            (0, "HIE".into(), "NHIE".into()),
            (1, "HIE".into(), "CHIE".into()),
            (0, "HIP".into(), "NHIP".into()),
            (1, "HIP".into(), "CHIP".into()),
            (0, "HYP".into(), "NHYP".into()),
            (1, "HYP".into(), "CHYP".into()),
            (0, "ILE".into(), "NILE".into()),
            (1, "ILE".into(), "CILE".into()),
            (0, "LEU".into(), "NLEU".into()),
            (1, "LEU".into(), "CLEU".into()),
            (0, "LYS".into(), "NLYS".into()),
            (1, "LYS".into(), "CLYS".into()),
            (0, "MET".into(), "NMET".into()),
            (1, "MET".into(), "CMET".into()),
            (0, "PHE".into(), "NPHE".into()),
            (1, "PHE".into(), "CPHE".into()),
            (0, "PRO".into(), "NPRO".into()),
            (1, "PRO".into(), "CPRO".into()),
            (0, "SER".into(), "NSER".into()),
            (1, "SER".into(), "CSER".into()),
            (0, "THR".into(), "NTHR".into()),
            (1, "THR".into(), "CTHR".into()),
            (0, "TRP".into(), "NTRP".into()),
            (1, "TRP".into(), "CTRP".into()),
            (0, "TYR".into(), "NTYR".into()),
            (1, "TYR".into(), "CTYR".into()),
            (0, "VAL".into(), "NVAL".into()),
            (1, "VAL".into(), "CVAL".into()),
            // HIS -> HIE default mapping
            (0, "HIS".into(), "NHIE".into()),
            (1, "HIS".into(), "CHIE".into()),
        ]);

        Ok(())
    }

    /// Load the Lipid21 force field (lipid21.dat + lipid21.lib).
    pub fn load_lipid21(&mut self) -> Result<(), String> {
        let lipid_params = parse_parm_dat(data::LIPID21_DAT)?;
        self.params.merge(&lipid_params);

        let lipid_lib = parse_lib(data::LIPID21_LIB)?;
        self.residue_lib.merge(&lipid_lib);

        Ok(())
    }

    /// Load the GAFF2 general force field (gaff2.dat).
    /// Note: GAFF2 has no residue library - molecules are loaded via mol2 files.
    pub fn load_gaff2(&mut self) -> Result<(), String> {
        let gaff2 = parse_parm_dat(data::GAFF2_DAT)?;
        self.params.merge(&gaff2);
        Ok(())
    }

    /// Load the OPC water model and ion parameters.
    pub fn load_water_opc(&mut self) -> Result<(), String> {
        // OPC water parameters
        let opc = parse_frcmod(data::FRCMOD_OPC)?;
        self.params.merge(&opc);

        // Li/Merz ion parameters for OPC
        let ions = parse_frcmod(data::FRCMOD_IONS_OPC)?;
        self.params.merge(&ions);

        // Solvent box templates
        let solvents = parse_lib(data::SOLVENTS_LIB)?;
        self.residue_lib.merge(&solvents);

        // Ion residue templates
        let atomic_ions = parse_lib(data::ATOMIC_IONS_LIB)?;
        self.residue_lib.merge(&atomic_ions);

        Ok(())
    }

    /// Load a custom .lib or .off file from disk.
    pub fn load_custom_lib(&mut self, path: &Path) -> Result<(), String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let lib = parse_lib(&content)?;
        self.residue_lib.merge(&lib);
        Ok(())
    }

    /// Load a custom frcmod file from disk.
    pub fn load_custom_frcmod(&mut self, path: &Path) -> Result<(), String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let params = parse_frcmod(&content)?;
        self.params.merge(&params);
        Ok(())
    }

    /// Load a custom parm .dat file from disk.
    pub fn load_custom_parm_dat(&mut self, path: &Path) -> Result<(), String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let params = parse_parm_dat(&content)?;
        self.params.merge(&params);
        Ok(())
    }

    /// Look up the AMBER residue name for a PDB residue at a terminal position.
    /// position: 0 = N-terminal (first residue in chain), 1 = C-terminal (last residue)
    pub fn map_terminal_residue(&self, pdb_name: &str, position: u8) -> Option<&str> {
        self.pdb_res_map
            .iter()
            .find(|(pos, name, _)| *pos == position && name == pdb_name)
            .map(|(_, _, amber_name)| amber_name.as_str())
    }

    /// Map a standard PDB residue name to AMBER name.
    /// Handles HIS -> HIE default, and other common mappings.
    pub fn map_residue_name<'a>(&self, pdb_name: &'a str) -> &'a str {
        match pdb_name {
            "HIS" => "HIE",
            other => other,
        }
    }
}

impl Default for LoadedForceField {
    fn default() -> Self {
        Self::new()
    }
}
