//! High-level system builder API (tleap replacement).
//!
//! [`SystemBuilder`] provides a programmatic, builder-pattern interface for
//! constructing solvated, parameterized molecular systems from PDB, mol2,
//! and mmCIF input files. The output is a set of AMBER prmtop/inpcrd/PDB
//! files ready for simulation or minimization.
//!
//! # Example (Rust)
//!
//! ```no_run
//! use rst_tleap::SystemBuilder;
//! use rst_tleap::ions::IonCount;
//! use std::path::Path;
//!
//! let mut builder = SystemBuilder::new();
//! builder.load_protein_ff19sb().unwrap();
//! builder.load_water_opc().unwrap();
//!
//! let mut system = builder.load_pdb(Path::new("protein.pdb")).unwrap();
//! builder.solvate_box(&mut system, 12.0, 1.0).unwrap();
//! builder.add_ions(&mut system, "Na+", IonCount::Neutralize).unwrap();
//! builder.write_amber(&system, Path::new("system")).unwrap();
//! ```

use std::path::Path;

use rst_core::amber::inpcrd_writer::{write_inpcrd, InpcrdData};
use rst_core::amber::prmtop_writer::{write_prmtop, PrmtopData};
use rst_core::forcefield::atom_types::element_from_type;
use rst_core::forcefield::loader::LoadedForceField;
use rst_core::forcefield::residue_lib::ResidueTemplate;
use rst_core::mmcif::{parse_mmcif_file, CifStructure};
use rst_core::mol2::{parse_mol2_file, Mol2Molecule};
use rst_core::pdb::{parse_pdb_file, PdbStructure};
use rst_core::pdb_writer::{write_pdb, PdbCryst1, PdbWriteAtom, PdbWriteData};

use crate::ions::{add_ions, IonConfig, IonCount};
use crate::parameterize::{generate_inpcrd_data, generate_prmtop_data};
use crate::solvate::{solvate_box, SolvateConfig};
use crate::system::{Atom, System};

// ===========================================================================
// SystemBuilder
// ===========================================================================

/// High-level builder for constructing molecular systems.
///
/// Wraps a [`LoadedForceField`] and exposes a clean API for the full tleap
/// workflow: load force fields, load structures, combine, solvate, add ions,
/// parameterize, and write output files.
pub struct SystemBuilder {
    /// Combined force field data (parameters + residue templates + PDB name map).
    ff: LoadedForceField,
}

impl SystemBuilder {
    /// Create a new builder with no force fields loaded.
    pub fn new() -> Self {
        Self {
            ff: LoadedForceField::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Force field loading
    // -----------------------------------------------------------------------

    /// Load the ff19SB protein force field.
    pub fn load_protein_ff19sb(&mut self) -> Result<&mut Self, String> {
        self.ff.load_protein_ff19sb()?;
        Ok(self)
    }

    /// Load the Lipid21 force field.
    pub fn load_lipid21(&mut self) -> Result<&mut Self, String> {
        self.ff.load_lipid21()?;
        Ok(self)
    }

    /// Load the GAFF2 general force field.
    pub fn load_gaff2(&mut self) -> Result<&mut Self, String> {
        self.ff.load_gaff2()?;
        Ok(self)
    }

    /// Load the OPC water model and associated ion parameters.
    pub fn load_water_opc(&mut self) -> Result<&mut Self, String> {
        self.ff.load_water_opc()?;
        Ok(self)
    }

    /// Load a custom `.lib` or `.off` residue library file from disk.
    pub fn load_custom_lib(&mut self, path: &Path) -> Result<&mut Self, String> {
        self.ff.load_custom_lib(path)?;
        Ok(self)
    }

    /// Load a custom `frcmod` parameter modification file from disk.
    pub fn load_custom_frcmod(&mut self, path: &Path) -> Result<&mut Self, String> {
        self.ff.load_custom_frcmod(path)?;
        Ok(self)
    }

    // -----------------------------------------------------------------------
    // Structure loading
    // -----------------------------------------------------------------------

    /// Load a PDB file and convert it to a [`System`].
    ///
    /// Residues are matched against the loaded force field templates to assign
    /// atom types, charges, masses, and intra-residue bonds. Terminal residues
    /// use N/C-terminal variants from the PDB residue name map.
    pub fn load_pdb(&self, path: &Path) -> Result<System, String> {
        let pdb = parse_pdb_file(path)?;
        self.pdb_to_system(&pdb)
    }

    /// Load a mol2 file (pre-parameterized by antechamber) and convert to a [`System`].
    ///
    /// Atom types and charges are taken directly from the mol2 file. Bonds are
    /// taken from the `@<TRIPOS>BOND` section.
    pub fn load_mol2(&self, path: &Path) -> Result<System, String> {
        let mol2 = parse_mol2_file(path)?;
        mol2_to_system(&mol2)
    }

    /// Load a ligand file (SDF or mol2) and parameterize it automatically.
    ///
    /// Runs the antechamber pipeline internally to assign GAFF2 atom types
    /// and AM1-BCC charges, then converts to a System ready for simulation.
    ///
    /// # Arguments
    /// * `path` - Path to the input file (.sdf, .mol, or .mol2).
    /// * `net_charge` - Net molecular charge.
    pub fn load_ligand(&self, path: &Path, net_charge: i32) -> Result<System, String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let ac_mol = match ext.as_str() {
            "sdf" | "mol" => {
                let mols = rst_core::sdf::parse_sdf_file(path)?;
                if mols.is_empty() {
                    return Err("No molecules found in SDF file".into());
                }
                rst_antechamber::molecule::AcMolecule::from_sdf(&mols[0])
            }
            "mol2" => {
                let mol2_input = parse_mol2_file(path)?;
                rst_antechamber::molecule::AcMolecule::from_mol2(&mol2_input)
            }
            _ => {
                return Err(format!(
                    "Unsupported ligand format '{}'. Use .sdf, .mol, or .mol2",
                    ext
                ));
            }
        };

        let config = rst_antechamber::AntechamberConfig {
            net_charge,
            charge_method: rst_antechamber::ChargeMethod::Am1Bcc,
        };

        let result = rst_antechamber::run_antechamber(ac_mol, &config)?;
        let mol2_out = rst_antechamber::to_mol2(&result);
        mol2_to_system(&mol2_out)
    }

    /// Load an mmCIF file and convert to a [`System`].
    ///
    /// If the file contains biological assembly information, the first
    /// assembly is constructed automatically (symmetry operations are applied).
    pub fn load_mmcif(&self, path: &Path) -> Result<System, String> {
        let cif = parse_mmcif_file(path)?;
        self.cif_to_system(&cif)
    }

    // -----------------------------------------------------------------------
    // System operations
    // -----------------------------------------------------------------------

    /// Combine multiple systems into one.
    ///
    /// All atoms, bonds, and residues are concatenated. Bond indices and
    /// residue atom ranges are offset appropriately.
    pub fn combine(&self, systems: Vec<System>) -> System {
        let mut combined = System::new();

        for sys in systems {
            let atom_offset = combined.atoms.len();
            let residue_offset = combined.residues.len();

            // Copy atoms, updating residue_idx.
            for mut atom in sys.atoms {
                atom.residue_idx += residue_offset;
                combined.atoms.push(atom);
            }

            // Copy residues, updating atom_range.
            for mut res in sys.residues {
                res.atom_range = (res.atom_range.start + atom_offset)
                    ..(res.atom_range.end + atom_offset);
                combined.residues.push(res);
            }

            // Copy bonds, updating atom indices.
            for mut bond in sys.bonds {
                bond.atom1 += atom_offset;
                bond.atom2 += atom_offset;
                combined.bonds.push(bond);
            }
        }

        combined
    }

    /// Solvate the system in a rectangular box of OPC water.
    ///
    /// # Arguments
    /// * `system` - System to solvate (modified in place).
    /// * `buffer` - Buffer distance in Angstroms on each side of the solute.
    /// * `closeness` - Minimum distance between solute and solvent atoms (Angstroms).
    pub fn solvate_box(
        &self,
        system: &mut System,
        buffer: f64,
        closeness: f64,
    ) -> Result<(), String> {
        let water_template = self
            .ff
            .residue_lib
            .get("OPCBOX")
            .or_else(|| self.ff.residue_lib.get("TP4EWBOX"))
            .ok_or("No water box template found. Did you call load_water_opc()?")?;

        let config = SolvateConfig { buffer, closeness };
        solvate_box(system, water_template, &config)
    }

    /// Add ions to a solvated system by replacing water molecules.
    ///
    /// # Arguments
    /// * `system` - The solvated system.
    /// * `ion_name` - Ion residue name (e.g. `"Na+"`, `"Cl-"`, `"K+"`).
    /// * `count` - Number of ions or [`IonCount::Neutralize`].
    pub fn add_ions(
        &self,
        system: &mut System,
        ion_name: &str,
        count: IonCount,
    ) -> Result<usize, String> {
        let config = IonConfig::default();
        add_ions(system, &self.ff.residue_lib, ion_name, count, &config, None)
    }

    /// Add a 1:1 salt (e.g. NaCl, KCl) at a target molar concentration.
    ///
    /// This is a convenience wrapper around [`add_ions`] that:
    /// 1. Neutralizes the system charge with the appropriate counterion.
    /// 2. Adds equal numbers of cation and anion to reach the target
    ///    concentration.
    ///
    /// # Arguments
    /// * `system` - The solvated system.
    /// * `cation` - Cation residue name (e.g. `"Na+"`, `"K+"`).
    /// * `anion` - Anion residue name (e.g. `"Cl-"`).
    /// * `concentration` - Target salt concentration in mol/L (e.g. `0.150`
    ///   for 150 mM).
    ///
    /// # Returns
    /// A tuple `(n_cation, n_anion)` with the total number of each ion placed
    /// (neutralization + excess).
    pub fn add_salt(
        &self,
        system: &mut System,
        cation: &str,
        anion: &str,
        concentration: f64,
    ) -> Result<(usize, usize), String> {
        let config = IonConfig::default();

        // Step 1: Neutralize.
        let n_neutralize_cat =
            add_ions(system, &self.ff.residue_lib, cation, IonCount::Neutralize, &config, None)?;
        let n_neutralize_an =
            add_ions(system, &self.ff.residue_lib, anion, IonCount::Neutralize, &config, None)?;

        // Step 2: Add salt at the target concentration.
        let conc_count = IonCount::Concentration(concentration);
        let n_excess_cat =
            add_ions(system, &self.ff.residue_lib, cation, conc_count, &config, None)?;
        let n_excess_an =
            add_ions(system, &self.ff.residue_lib, anion, conc_count, &config, None)?;

        Ok((
            n_neutralize_cat + n_excess_cat,
            n_neutralize_an + n_excess_an,
        ))
    }

    /// Add ions with custom configuration.
    pub fn add_ions_with_config(
        &self,
        system: &mut System,
        ion_name: &str,
        count: IonCount,
        config: &IonConfig,
        rng_seed: Option<u64>,
    ) -> Result<usize, String> {
        add_ions(
            system,
            &self.ff.residue_lib,
            ion_name,
            count,
            config,
            rng_seed,
        )
    }

    // -----------------------------------------------------------------------
    // Parameterization and output
    // -----------------------------------------------------------------------

    /// Parameterize the system and write prmtop + inpcrd files.
    ///
    /// The `base_path` is used as a prefix: `base_path.prmtop` and
    /// `base_path.inpcrd` are written.
    pub fn write_amber(&self, system: &System, base_path: &Path) -> Result<(), String> {
        let title = base_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("system");

        let prmtop_data = self.parameterize(system, title)?;
        let inpcrd_data = generate_inpcrd_data(system, title);

        let prmtop_path = base_path.with_extension("prmtop");
        let inpcrd_path = base_path.with_extension("inpcrd");

        write_prmtop(&prmtop_data, &prmtop_path)?;
        write_inpcrd(&inpcrd_data, &inpcrd_path)?;

        log::info!(
            "Wrote {} ({} atoms, {} residues)",
            prmtop_path.display(),
            system.n_atoms(),
            system.n_residues(),
        );

        Ok(())
    }

    /// Write only the prmtop file.
    pub fn write_prmtop(&self, system: &System, path: &Path) -> Result<(), String> {
        let title = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("system");
        let prmtop_data = self.parameterize(system, title)?;
        write_prmtop(&prmtop_data, path)
    }

    /// Write only the inpcrd file.
    pub fn write_inpcrd(&self, system: &System, path: &Path) -> Result<(), String> {
        let title = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("system");
        let inpcrd_data = generate_inpcrd_data(system, title);
        write_inpcrd(&inpcrd_data, path)
    }

    /// Write a PDB file from the system.
    pub fn write_pdb(&self, system: &System, path: &Path) -> Result<(), String> {
        let pdb_data = system_to_pdb_write_data(system);
        write_pdb(&pdb_data, path)
    }

    /// Run the parameterization pipeline and return the raw prmtop data.
    ///
    /// This is useful when you want to inspect the topology before writing.
    pub fn parameterize(&self, system: &System, title: &str) -> Result<PrmtopData, String> {
        generate_prmtop_data(system, &self.ff.params, title)
    }

    /// Generate inpcrd data without writing to disk.
    pub fn generate_inpcrd(&self, system: &System, title: &str) -> InpcrdData {
        generate_inpcrd_data(system, title)
    }

    /// Get a reference to the loaded force field.
    pub fn force_field(&self) -> &LoadedForceField {
        &self.ff
    }

    // -----------------------------------------------------------------------
    // Internal conversion: PDB -> System
    // -----------------------------------------------------------------------

    fn pdb_to_system(&self, pdb: &PdbStructure) -> Result<System, String> {
        let mut system = System::new();

        for (chain_idx, chain) in pdb.chains.iter().enumerate() {
            let n_residues_in_chain = chain.residue_indices.len();

            for (pos_in_chain, &res_idx) in chain.residue_indices.iter().enumerate() {
                let pdb_res = &pdb.residues[res_idx];

                // Determine the AMBER residue name, accounting for terminal variants.
                let amber_name = self.resolve_residue_name(
                    &pdb_res.name,
                    pos_in_chain,
                    n_residues_in_chain,
                    chain_idx,
                );

                // Look up the template.
                let template = self.ff.residue_lib.get(&amber_name);

                // Build atoms for this residue.
                let (atoms, internal_bonds) =
                    self.build_residue_atoms(pdb, pdb_res, template, &amber_name)?;

                let res_sys_idx = system.add_residue(
                    &amber_name,
                    pdb_res.chain_id,
                    pdb_res.res_seq,
                    atoms,
                    internal_bonds,
                );

                // Add inter-residue bond to previous residue (peptide bond N-C).
                if pos_in_chain > 0 {
                    self.add_inter_residue_bond(
                        &mut system,
                        res_sys_idx - 1,
                        res_sys_idx,
                        template,
                    );
                }
            }
        }

        Ok(system)
    }

    /// Determine the AMBER residue name, using terminal variants if applicable.
    fn resolve_residue_name(
        &self,
        pdb_name: &str,
        pos_in_chain: usize,
        n_residues_in_chain: usize,
        _chain_idx: usize,
    ) -> String {
        // Map common PDB names first (e.g. HIS -> HIE).
        let mapped = self.ff.map_residue_name(pdb_name);

        // Check if this is a terminal residue.
        if n_residues_in_chain > 1 {
            if pos_in_chain == 0 {
                // N-terminal.
                if let Some(nt_name) = self.ff.map_terminal_residue(mapped, 0) {
                    if self.ff.residue_lib.get(nt_name).is_some() {
                        return nt_name.to_string();
                    }
                }
            } else if pos_in_chain == n_residues_in_chain - 1 {
                // C-terminal.
                if let Some(ct_name) = self.ff.map_terminal_residue(mapped, 1) {
                    if self.ff.residue_lib.get(ct_name).is_some() {
                        return ct_name.to_string();
                    }
                }
            }
        }

        // Check if the mapped name exists in the library.
        if self.ff.residue_lib.get(mapped).is_some() {
            return mapped.to_string();
        }

        // Fall back to the original PDB name.
        pdb_name.to_string()
    }

    /// Build atoms and internal bonds for a single residue from PDB data.
    ///
    /// If a template is available, atom types, charges, and masses are taken
    /// from it. Otherwise, PDB data is used with defaults.
    #[allow(clippy::type_complexity)]
    fn build_residue_atoms(
        &self,
        pdb: &PdbStructure,
        pdb_res: &rst_core::pdb::PdbResidue,
        template: Option<&ResidueTemplate>,
        amber_name: &str,
    ) -> Result<(Vec<Atom>, Vec<(usize, usize)>), String> {
        let mut atoms = Vec::with_capacity(pdb_res.atom_indices.len());
        let mut internal_bonds = Vec::new();

        if let Some(tmpl) = template {
            // Map PDB atoms to template atoms by name.
            let mut pdb_to_local: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();

            for &pdb_atom_idx in &pdb_res.atom_indices {
                let pdb_atom = &pdb.atoms[pdb_atom_idx];
                let atom_name = pdb_atom.name.trim();

                // Find matching template atom.
                if let Some(tmpl_atom) = tmpl.atoms.iter().find(|ta| ta.name == atom_name) {
                    let element =
                        element_from_type(&tmpl_atom.atom_type).unwrap_or("");
                    let mass = mass_for_element(element);
                    let atomic_number = if tmpl_atom.element_number > 0 {
                        tmpl_atom.element_number
                    } else {
                        atomic_number_for_element(element)
                    };

                    let local_idx = atoms.len();
                    pdb_to_local.insert(atom_name.to_string(), local_idx);

                    atoms.push(Atom {
                        name: atom_name.to_string(),
                        atom_type: tmpl_atom.atom_type.clone(),
                        element: element.to_string(),
                        charge: tmpl_atom.charge,
                        mass,
                        atomic_number,
                        position: pdb_atom.position,
                        residue_idx: 0,
                        born_radius: 0.0,
                        screen: 0.0,
                    });
                } else {
                    // Atom not in template - use PDB data with defaults.
                    let local_idx = atoms.len();
                    pdb_to_local.insert(atom_name.to_string(), local_idx);

                    atoms.push(Atom {
                        name: atom_name.to_string(),
                        atom_type: guess_atom_type(atom_name, &pdb_atom.element),
                        element: pdb_atom.element.clone(),
                        charge: 0.0,
                        mass: mass_for_element(&pdb_atom.element),
                        atomic_number: atomic_number_for_element(&pdb_atom.element),
                        position: pdb_atom.position,
                        residue_idx: 0,
                        born_radius: 0.0,
                        screen: 0.0,
                    });

                    log::debug!(
                        "Atom '{}' in residue '{}' not found in template - using defaults",
                        atom_name,
                        amber_name,
                    );
                }
            }

            // Build internal bonds from template.
            for tb in &tmpl.bonds {
                let name1 = &tmpl.atoms[tb.atom1].name;
                let name2 = &tmpl.atoms[tb.atom2].name;
                if let (Some(&idx1), Some(&idx2)) =
                    (pdb_to_local.get(name1), pdb_to_local.get(name2))
                {
                    internal_bonds.push((idx1, idx2));
                }
            }
        } else {
            // No template found - build atoms from PDB data only.
            log::warn!(
                "No template found for residue '{}' - using PDB atom data",
                amber_name,
            );

            for &pdb_atom_idx in &pdb_res.atom_indices {
                let pdb_atom = &pdb.atoms[pdb_atom_idx];
                let atom_name = pdb_atom.name.trim();

                atoms.push(Atom {
                    name: atom_name.to_string(),
                    atom_type: guess_atom_type(atom_name, &pdb_atom.element),
                    element: pdb_atom.element.clone(),
                    charge: 0.0,
                    mass: mass_for_element(&pdb_atom.element),
                    atomic_number: atomic_number_for_element(&pdb_atom.element),
                    position: pdb_atom.position,
                    residue_idx: 0,
                    born_radius: 0.0,
                    screen: 0.0,
                });
            }
        }

        Ok((atoms, internal_bonds))
    }

    /// Add a backbone bond between the tail atom of the previous residue and
    /// the head atom of the current residue.
    fn add_inter_residue_bond(
        &self,
        system: &mut System,
        prev_res_idx: usize,
        curr_res_idx: usize,
        curr_template: Option<&ResidueTemplate>,
    ) {
        let prev_res = &system.residues[prev_res_idx];
        let curr_res = &system.residues[curr_res_idx];

        // Look up templates for both residues.
        let prev_template = self.ff.residue_lib.get(&prev_res.name);

        // Find tail atom of previous residue.
        let tail_atom = prev_template
            .and_then(|t| t.tail_atom)
            .and_then(|tail_idx| {
                let tail_name = &prev_template.unwrap().atoms[tail_idx].name;
                find_atom_by_name(system, prev_res, tail_name)
            });

        // Find head atom of current residue.
        let head_atom = curr_template
            .and_then(|t| t.head_atom)
            .and_then(|head_idx| {
                let head_name = &curr_template.unwrap().atoms[head_idx].name;
                find_atom_by_name(system, curr_res, head_name)
            });

        if let (Some(tail), Some(head)) = (tail_atom, head_atom) {
            system.add_bond(tail, head);
        }
    }

    // -----------------------------------------------------------------------
    // Internal conversion: CIF -> System
    // -----------------------------------------------------------------------

    fn cif_to_system(&self, cif: &CifStructure) -> Result<System, String> {
        let mut system = System::new();

        // Group CIF atoms by (chain_id, res_seq, ins_code) to form residues.
        let mut residue_groups: Vec<ResidueGroup> = Vec::new();
        let mut current_key: Option<(String, i32, String)> = None;

        for cif_atom in &cif.atoms {
            let key = (
                cif_atom.auth_chain_id.clone(),
                cif_atom.res_seq,
                cif_atom.ins_code.clone(),
            );

            if current_key.as_ref() != Some(&key) {
                residue_groups.push(ResidueGroup {
                    name: cif_atom.residue_name.clone(),
                    chain_id: cif_atom
                        .auth_chain_id
                        .chars()
                        .next()
                        .unwrap_or(' '),
                    res_seq: cif_atom.res_seq,
                    atom_indices: Vec::new(),
                });
                current_key = Some(key);
            }

            if let Some(group) = residue_groups.last_mut() {
                group.atom_indices.push(cif_atom);
            }
        }

        // Group residues by chain for terminal detection.
        let mut chain_groups: Vec<Vec<usize>> = Vec::new();
        let mut current_chain: Option<char> = None;

        for (idx, group) in residue_groups.iter().enumerate() {
            if current_chain != Some(group.chain_id) {
                chain_groups.push(Vec::new());
                current_chain = Some(group.chain_id);
            }
            if let Some(chain) = chain_groups.last_mut() {
                chain.push(idx);
            }
        }

        // Process each chain.
        for (chain_idx, chain_res_indices) in chain_groups.iter().enumerate() {
            let n_in_chain = chain_res_indices.len();

            for (pos_in_chain, &group_idx) in chain_res_indices.iter().enumerate() {
                let group = &residue_groups[group_idx];

                let amber_name = self.resolve_residue_name(
                    &group.name,
                    pos_in_chain,
                    n_in_chain,
                    chain_idx,
                );

                let template = self.ff.residue_lib.get(&amber_name);

                let (atoms, internal_bonds) =
                    build_residue_atoms_from_cif(group, template, &amber_name, &self.ff)?;

                let res_sys_idx = system.add_residue(
                    &amber_name,
                    group.chain_id,
                    group.res_seq,
                    atoms,
                    internal_bonds,
                );

                // Inter-residue bond.
                if pos_in_chain > 0 {
                    self.add_inter_residue_bond(
                        &mut system,
                        res_sys_idx - 1,
                        res_sys_idx,
                        template,
                    );
                }
            }
        }

        Ok(system)
    }
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Helpers: mol2 -> System
// ===========================================================================

/// Convert a mol2 molecule (pre-parameterized by antechamber) to a System.
fn mol2_to_system(mol2: &Mol2Molecule) -> Result<System, String> {
    let mut system = System::new();

    // Group atoms by substructure (residue).
    if mol2.substructures.is_empty() {
        // Single residue: all atoms belong to one residue.
        let mut atoms = Vec::with_capacity(mol2.atoms.len());
        for ma in &mol2.atoms {
            atoms.push(mol2_atom_to_system_atom(ma));
        }

        let bonds: Vec<(usize, usize)> = mol2
            .bonds
            .iter()
            .map(|b| (b.atom1, b.atom2))
            .collect();

        let res_name = if mol2.atoms.is_empty() {
            "UNK"
        } else {
            &mol2.atoms[0].residue_name
        };

        system.add_residue(res_name, ' ', 1, atoms, bonds);
    } else {
        // Multiple substructures.
        for (sub_idx, sub) in mol2.substructures.iter().enumerate() {
            // Determine atom range for this substructure.
            let start = sub.root_atom;
            let end = if sub_idx + 1 < mol2.substructures.len() {
                mol2.substructures[sub_idx + 1].root_atom
            } else {
                mol2.atoms.len()
            };

            let mut atoms = Vec::with_capacity(end - start);
            for ma in &mol2.atoms[start..end] {
                atoms.push(mol2_atom_to_system_atom(ma));
            }

            // Collect bonds within this substructure (offset to local indices).
            let mut local_bonds = Vec::new();
            for b in &mol2.bonds {
                if b.atom1 >= start && b.atom1 < end && b.atom2 >= start && b.atom2 < end {
                    local_bonds.push((b.atom1 - start, b.atom2 - start));
                }
            }

            system.add_residue(
                &sub.name,
                ' ',
                sub.id as i32,
                atoms,
                local_bonds,
            );
        }

        // Add inter-substructure bonds as system bonds.
        for b in &mol2.bonds {
            let in_same_sub = mol2.substructures.iter().enumerate().any(|(sub_idx, sub)| {
                let start = sub.root_atom;
                let end = if sub_idx + 1 < mol2.substructures.len() {
                    mol2.substructures[sub_idx + 1].root_atom
                } else {
                    mol2.atoms.len()
                };
                b.atom1 >= start && b.atom1 < end && b.atom2 >= start && b.atom2 < end
            });

            if !in_same_sub {
                system.add_bond(b.atom1, b.atom2);
            }
        }
    }

    Ok(system)
}

/// Convert a single mol2 atom to a system atom.
fn mol2_atom_to_system_atom(ma: &rst_core::mol2::Mol2Atom) -> Atom {
    let element = gaff_type_to_element(&ma.atom_type);
    Atom {
        name: ma.name.clone(),
        atom_type: ma.atom_type.clone(),
        element: element.to_string(),
        charge: ma.charge,
        mass: mass_for_element(element),
        atomic_number: atomic_number_for_element(element),
        position: ma.position,
        residue_idx: 0,
        born_radius: 0.0,
        screen: 0.0,
    }
}

// ===========================================================================
// Helpers: CIF residue building
// ===========================================================================

/// Temporary grouping of CIF atoms by residue.
struct ResidueGroup<'a> {
    name: String,
    chain_id: char,
    res_seq: i32,
    atom_indices: Vec<&'a rst_core::mmcif::CifAtom>,
}

/// Build system atoms and internal bonds from a CIF residue group.
#[allow(clippy::type_complexity)]
fn build_residue_atoms_from_cif(
    group: &ResidueGroup,
    template: Option<&ResidueTemplate>,
    amber_name: &str,
    ff: &LoadedForceField,
) -> Result<(Vec<Atom>, Vec<(usize, usize)>), String> {
    let mut atoms = Vec::with_capacity(group.atom_indices.len());
    let mut internal_bonds = Vec::new();

    if let Some(tmpl) = template {
        let mut name_to_local: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for cif_atom in &group.atom_indices {
            let atom_name = cif_atom.name.trim();

            if let Some(tmpl_atom) = tmpl.atoms.iter().find(|ta| ta.name == atom_name) {
                let element = element_from_type(&tmpl_atom.atom_type).unwrap_or("");
                let mass = mass_for_element(element);
                let atomic_number = if tmpl_atom.element_number > 0 {
                    tmpl_atom.element_number
                } else {
                    atomic_number_for_element(element)
                };

                let local_idx = atoms.len();
                name_to_local.insert(atom_name.to_string(), local_idx);

                atoms.push(Atom {
                    name: atom_name.to_string(),
                    atom_type: tmpl_atom.atom_type.clone(),
                    element: element.to_string(),
                    charge: tmpl_atom.charge,
                    mass,
                    atomic_number,
                    position: cif_atom.position,
                    residue_idx: 0,
                    born_radius: 0.0,
                    screen: 0.0,
                });
            } else {
                let local_idx = atoms.len();
                name_to_local.insert(atom_name.to_string(), local_idx);

                atoms.push(Atom {
                    name: atom_name.to_string(),
                    atom_type: guess_atom_type(atom_name, &cif_atom.element),
                    element: cif_atom.element.clone(),
                    charge: 0.0,
                    mass: mass_for_element(&cif_atom.element),
                    atomic_number: atomic_number_for_element(&cif_atom.element),
                    position: cif_atom.position,
                    residue_idx: 0,
                    born_radius: 0.0,
                    screen: 0.0,
                });
            }
        }

        // Internal bonds from template.
        for tb in &tmpl.bonds {
            let name1 = &tmpl.atoms[tb.atom1].name;
            let name2 = &tmpl.atoms[tb.atom2].name;
            if let (Some(&idx1), Some(&idx2)) =
                (name_to_local.get(name1), name_to_local.get(name2))
            {
                internal_bonds.push((idx1, idx2));
            }
        }
    } else {
        log::warn!(
            "No template found for residue '{}' in mmCIF - using CIF atom data",
            amber_name,
        );

        for cif_atom in &group.atom_indices {
            let atom_name = cif_atom.name.trim();
            atoms.push(Atom {
                name: atom_name.to_string(),
                atom_type: guess_atom_type(atom_name, &cif_atom.element),
                element: cif_atom.element.clone(),
                charge: 0.0,
                mass: mass_for_element(&cif_atom.element),
                atomic_number: atomic_number_for_element(&cif_atom.element),
                position: cif_atom.position,
                residue_idx: 0,
                born_radius: 0.0,
                screen: 0.0,
            });
        }
    }

    // Suppress unused variable warning.
    let _ = ff;

    Ok((atoms, internal_bonds))
}

// ===========================================================================
// System -> PDB writer data
// ===========================================================================

/// Convert a System to PDB writer data.
fn system_to_pdb_write_data(system: &System) -> PdbWriteData {
    let mut atoms = Vec::with_capacity(system.n_atoms());

    for (idx, atom) in system.atoms.iter().enumerate() {
        let res = &system.residues[atom.residue_idx];
        let is_hetatm = !is_standard_residue(&res.name);

        atoms.push(PdbWriteAtom {
            serial: idx + 1,
            name: atom.name.clone(),
            residue_name: res.name.clone(),
            chain_id: res.chain_id,
            res_seq: res.seq_num,
            i_code: ' ',
            position: atom.position,
            occupancy: 1.0,
            temp_factor: 0.0,
            element: atom.element.clone(),
            is_hetatm,
        });
    }

    let cryst1 = system.box_dimensions.map(|dims| {
        let angles = system.box_angles.unwrap_or([90.0, 90.0, 90.0]);
        PdbCryst1 {
            a: dims[0],
            b: dims[1],
            c: dims[2],
            alpha: angles[0],
            beta: angles[1],
            gamma: angles[2],
        }
    });

    PdbWriteData {
        atoms,
        bonds: Vec::new(),
        cryst1,
    }
}

// ===========================================================================
// Utility functions
// ===========================================================================

/// Find a global atom index by name within a residue.
fn find_atom_by_name(
    system: &System,
    res: &crate::system::Residue,
    name: &str,
) -> Option<usize> {
    system.atoms[res.atom_range.clone()]
        .iter()
        .enumerate()
        .find(|(_, a)| a.name == name)
        .map(|(local_idx, _)| res.atom_range.start + local_idx)
}

/// Check whether a residue name is a standard amino acid or nucleotide.
fn is_standard_residue(name: &str) -> bool {
    matches!(
        name,
        "ALA" | "ARG" | "ASN" | "ASP" | "CYS" | "CYX" | "GLN" | "GLU" | "GLY" | "HID"
            | "HIE" | "HIP" | "HIS" | "HYP" | "ILE" | "LEU" | "LYS" | "MET" | "PHE"
            | "PRO" | "SER" | "THR" | "TRP" | "TYR" | "VAL" | "WAT" | "HOH"
            | "NALA" | "NARG" | "NASN" | "NASP" | "NCYS" | "NCYX" | "NGLN" | "NGLU"
            | "NGLY" | "NHID" | "NHIE" | "NHIP" | "NHYP" | "NILE" | "NLEU" | "NLYS"
            | "NMET" | "NPHE" | "NPRO" | "NSER" | "NTHR" | "NTRP" | "NTYR" | "NVAL"
            | "CALA" | "CARG" | "CASN" | "CASP" | "CCYS" | "CCYX" | "CGLN" | "CGLU"
            | "CGLY" | "CHID" | "CHIE" | "CHIP" | "CHYP" | "CILE" | "CLEU" | "CLYS"
            | "CMET" | "CPHE" | "CPRO" | "CSER" | "CTHR" | "CTRP" | "CTYR" | "CVAL"
            | "DA" | "DC" | "DG" | "DT" | "RA" | "RC" | "RG" | "RU"
    )
}

/// Guess a basic AMBER atom type from an atom name and element.
fn guess_atom_type(atom_name: &str, element: &str) -> String {
    match element {
        "C" => {
            if atom_name == "CA" {
                "CA".to_string()
            } else {
                "CT".to_string()
            }
        }
        "N" => "N".to_string(),
        "O" => "O".to_string(),
        "H" => "H".to_string(),
        "S" => "S".to_string(),
        "P" => "P".to_string(),
        _ => element.to_string(),
    }
}

/// Derive element from a GAFF2 atom type.
///
/// GAFF2 types are lowercase and start with the element letter(s).
fn gaff_type_to_element(gaff_type: &str) -> &str {
    // Try the forcefield module first.
    if let Some(elem) = element_from_type(gaff_type) {
        return elem;
    }

    // Fallback: parse the first character(s) of the type name.
    let bytes = gaff_type.as_bytes();
    if bytes.is_empty() {
        return "X";
    }
    match bytes[0] {
        b'c' => "C",
        b'n' => "N",
        b'o' => "O",
        b'h' => "H",
        b's' => "S",
        b'p' => "P",
        b'f' => "F",
        b'i' => "I",
        b'b' if bytes.len() > 1 && bytes[1] == b'r' => "Br",
        _ => "X",
    }
}

/// Look up atomic mass for a given element symbol.
fn mass_for_element(element: &str) -> f64 {
    match element {
        "H" => 1.008,
        "He" => 4.003,
        "Li" => 6.941,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "F" => 18.998,
        "Na" => 22.990,
        "Mg" => 24.305,
        "P" => 30.974,
        "S" => 32.065,
        "Cl" => 35.453,
        "K" => 39.098,
        "Ca" => 40.078,
        "Fe" => 55.845,
        "Zn" => 65.380,
        "Br" => 79.904,
        "Rb" => 85.468,
        "I" => 126.904,
        "Cs" => 132.905,
        "" => 0.0,
        _ => 0.0,
    }
}

/// Look up atomic number for a given element symbol.
fn atomic_number_for_element(element: &str) -> i32 {
    match element {
        "H" => 1,
        "He" => 2,
        "Li" => 3,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "Na" => 11,
        "Mg" => 12,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "K" => 19,
        "Ca" => 20,
        "Fe" => 26,
        "Zn" => 30,
        "Br" => 35,
        "Rb" => 37,
        "I" => 53,
        "Cs" => 55,
        "" => 0,
        _ => 0,
    }
}
