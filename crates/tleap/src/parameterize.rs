//! Parameterization pipeline: assigns AMBER force field parameters to a [`System`]
//! and generates the full topology data needed to write a prmtop file.
//!
//! The main entry point is [`generate_prmtop_data`], which takes a fully built
//! [`System`] (atoms, bonds, residues, coordinates already populated from template
//! matching) and a [`ForceFieldParams`] database, then enumerates all bonded
//! interactions, looks up parameters, builds LJ tables and exclusion lists, and
//! packages everything into a [`PrmtopData`] ready for the prmtop writer.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::system::System;
use rst_core::amber::inpcrd_writer::InpcrdData;
use rst_core::amber::prmtop_writer::{BoxInfo, PrmtopData};
use rst_core::forcefield::parameters::ForceFieldParams;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Degrees-to-radians conversion factor.
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Default 1-4 electrostatic scaling factor for ff19SB.
const DEFAULT_SCEE: f64 = 1.2;

/// Default 1-4 van der Waals scaling factor for ff19SB.
const DEFAULT_SCNB: f64 = 2.0;

/// Residue names recognized as solvent (water models).
const SOLVENT_RESIDUES: &[&str] = &["WAT", "HOH", "TIP", "TP3", "TP4", "TP5", "OPC", "SPC"];

/// Residue names recognized as monatomic ions.
const ION_RESIDUES: &[&str] = &[
    "Na+", "Cl-", "K+", "Mg+", "Ca+", "Zn+", "Li+", "Rb+", "Cs+", "F-", "Br-", "I-",
    "NA", "CL", "MG", "ZN", "CA", "LI", "RB", "CS",
];

// ---------------------------------------------------------------------------
// Helper types for parameter deduplication
// ---------------------------------------------------------------------------

/// A unique bond parameter type, identified by force constant and equilibrium value.
#[derive(Debug, Clone)]
struct BondType {
    force_constant: f64,
    eq_value: f64,
}

/// A unique angle parameter type.
#[derive(Debug, Clone)]
struct AngleType {
    force_constant: f64,
    eq_value: f64, // in radians
}

/// A unique dihedral parameter type (single Fourier term).
#[derive(Debug, Clone)]
struct DihedralType {
    force_constant: f64,
    periodicity: f64,
    phase: f64, // in radians
    scee: f64,
    scnb: f64,
}

/// A resolved dihedral entry: atom quad plus the list of parameter-type indices
/// for each Fourier term.
struct ResolvedDihedral {
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    /// Indices into the dihedral-type array, one per Fourier term.
    type_indices: Vec<usize>,
}

/// A resolved improper entry.
struct ResolvedImproper {
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    type_idx: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate complete AMBER prmtop data from a parameterized system.
///
/// The system must already have all atom types, charges, and masses assigned
/// (from template matching during structure loading).
///
/// This function:
/// 1. Enumerates all angles from the bond graph
/// 2. Enumerates all proper dihedrals from the bond graph
/// 3. Enumerates all improper dihedrals
/// 4. Looks up force field parameters for each bonded term
/// 5. Assigns LJ parameters and builds the nonbonded parameter index
/// 6. Builds exclusion lists (1-2 and 1-3 pairs excluded, 1-4 pairs scaled)
/// 7. Assigns Born radii and screening parameters
/// 8. Packages everything into `PrmtopData`
///
/// # Errors
///
/// Returns `Err(String)` if any required force field parameter is missing for
/// a bonded interaction present in the system.
pub fn generate_prmtop_data(
    system: &System,
    ff_params: &ForceFieldParams,
    title: &str,
) -> Result<PrmtopData, String> {
    let n_atoms = system.n_atoms();

    // Step 1: Build adjacency list from bonds.
    let adj = build_adjacency_list(n_atoms, &system.bonds);

    // Step 2: Enumerate angles.
    let angles = enumerate_angles(n_atoms, &adj);

    // Step 3: Enumerate proper dihedrals.
    let dihedrals = enumerate_dihedrals(n_atoms, &adj);

    // Step 4: Enumerate improper dihedrals.
    let impropers = enumerate_impropers(system, &adj, ff_params);

    // Step 5: Look up bonded parameters and deduplicate types.
    let (bond_types, resolved_bonds) = resolve_bond_params(system, ff_params)?;
    let (angle_types, resolved_angles) = resolve_angle_params(system, &angles, ff_params)?;
    let (dihedral_types, resolved_dihedrals) =
        resolve_dihedral_params(system, &dihedrals, ff_params)?;
    let (imp_dihedral_types, resolved_impropers) =
        resolve_improper_params(system, &impropers, ff_params)?;

    // Merge improper dihedral types into the main dihedral type arrays with an
    // offset so that improper type indices continue after proper dihedral types.
    let dihedral_type_offset = dihedral_types.len();
    let mut all_dihedral_types = dihedral_types;
    all_dihedral_types.extend(imp_dihedral_types);

    // Step 6: Build LJ parameter arrays.
    let (atom_type_indices, n_types, nb_parm_index, lj_acoef, lj_bcoef) =
        build_lj_tables(system, ff_params)?;

    // Step 7: Build exclusion lists.
    let (num_excluded_atoms, excluded_atoms_list) = build_exclusion_lists(n_atoms, &adj);

    // Step 8: Classify bonded terms by hydrogen content and format for prmtop.
    let (bonds_inc_h, bonds_without_h) =
        format_bond_topology(system, &resolved_bonds);
    let (angles_inc_h, angles_without_h) =
        format_angle_topology(system, &resolved_angles);
    let (dihedrals_inc_h, dihedrals_without_h) = format_dihedral_topology(
        system,
        &resolved_dihedrals,
        &resolved_impropers,
        dihedral_type_offset,
    );

    // Step 9: Assign Born radii and screening parameters.
    let (radii, screen) = assign_born_params(system, &adj);

    // Step 10: Build per-atom arrays.
    let atom_names: Vec<String> = system.atoms.iter().map(|a| a.name.clone()).collect();
    let charges: Vec<f64> = system.atoms.iter().map(|a| a.charge).collect();
    let atomic_numbers: Vec<i32> = system.atoms.iter().map(|a| a.atomic_number).collect();
    let masses: Vec<f64> = system.atoms.iter().map(|a| a.mass).collect();
    let amber_atom_types: Vec<String> = system.atoms.iter().map(|a| a.atom_type.clone()).collect();

    // Step 11: Build residue arrays.
    let residue_labels: Vec<String> = system.residues.iter().map(|r| r.name.clone()).collect();
    let residue_pointers: Vec<usize> = system.residues.iter().map(|r| r.atom_range.start).collect();

    // Step 12: Build parameter arrays from deduplicated types.
    let bond_force_constants: Vec<f64> = bond_types.iter().map(|b| b.force_constant).collect();
    let bond_equil_values: Vec<f64> = bond_types.iter().map(|b| b.eq_value).collect();
    let angle_force_constants: Vec<f64> = angle_types.iter().map(|a| a.force_constant).collect();
    let angle_equil_values: Vec<f64> = angle_types.iter().map(|a| a.eq_value).collect();
    let dihedral_force_constants: Vec<f64> =
        all_dihedral_types.iter().map(|d| d.force_constant).collect();
    let dihedral_periodicities: Vec<f64> =
        all_dihedral_types.iter().map(|d| d.periodicity).collect();
    let dihedral_phases: Vec<f64> = all_dihedral_types.iter().map(|d| d.phase).collect();
    let scee_scale_factors: Vec<f64> = all_dihedral_types.iter().map(|d| d.scee).collect();
    let scnb_scale_factors: Vec<f64> = all_dihedral_types.iter().map(|d| d.scnb).collect();

    // Step 13: Package box info.
    let box_info = build_box_info(system);

    Ok(PrmtopData {
        title: title.to_owned(),
        atom_names,
        charges,
        atomic_numbers,
        masses,
        atom_type_indices,
        num_excluded_atoms,
        excluded_atoms_list,
        residue_labels,
        residue_pointers,
        amber_atom_types,
        n_types,
        nb_parm_index,
        bond_force_constants,
        bond_equil_values,
        angle_force_constants,
        angle_equil_values,
        dihedral_force_constants,
        dihedral_periodicities,
        dihedral_phases,
        scee_scale_factors,
        scnb_scale_factors,
        lj_acoef,
        lj_bcoef,
        bonds_inc_hydrogen: bonds_inc_h,
        bonds_without_hydrogen: bonds_without_h,
        angles_inc_hydrogen: angles_inc_h,
        angles_without_hydrogen: angles_without_h,
        dihedrals_inc_hydrogen: dihedrals_inc_h,
        dihedrals_without_hydrogen: dihedrals_without_h,
        box_info,
        radii,
        screen,
    })
}

/// Generate inpcrd data from a system.
///
/// Extracts atomic positions and box information into the [`InpcrdData`]
/// structure ready for writing to an inpcrd/rst7 file.
pub fn generate_inpcrd_data(system: &System, title: &str) -> InpcrdData {
    let positions: Vec<[f64; 3]> = system.atoms.iter().map(|a| a.position).collect();

    InpcrdData {
        title: title.to_owned(),
        positions,
        box_dimensions: system.box_dimensions,
        box_angles: system.box_angles,
    }
}

// ===========================================================================
// Step 1: Adjacency list
// ===========================================================================

/// Build a symmetric adjacency list from the system's bond list.
fn build_adjacency_list(
    n_atoms: usize,
    bonds: &[crate::system::Bond],
) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
    for bond in bonds {
        adj[bond.atom1].push(bond.atom2);
        adj[bond.atom2].push(bond.atom1);
    }
    // Sort each neighbor list for deterministic enumeration.
    for neighbors in &mut adj {
        neighbors.sort_unstable();
    }
    adj
}

// ===========================================================================
// Step 2: Enumerate angles
// ===========================================================================

/// Enumerate all unique angle triplets (i, j, k) where i-j and j-k are bonded,
/// with i < k to avoid duplicates.
fn enumerate_angles(_n_atoms: usize, adj: &[Vec<usize>]) -> Vec<(usize, usize, usize)> {
    let mut angles = Vec::new();
    for (j, neighbors) in adj.iter().enumerate() {
        for (idx_a, &i) in neighbors.iter().enumerate() {
            for &k in &neighbors[idx_a + 1..] {
                // Since neighbors are sorted and we iterate idx_a < idx_k, i < k is guaranteed.
                angles.push((i, j, k));
            }
        }
    }
    angles
}

// ===========================================================================
// Step 3: Enumerate proper dihedrals
// ===========================================================================

/// Enumerate all proper dihedral quartets (i, j, k, l) from the bond graph.
///
/// Each central bond j-k is traversed once (j < k). For each such bond, all
/// neighbors i of j (i != k) and l of k (l != j) form a dihedral.
fn enumerate_dihedrals(n_atoms: usize, adj: &[Vec<usize>]) -> Vec<(usize, usize, usize, usize)> {
    let mut dihedrals = Vec::new();
    for j in 0..n_atoms {
        for &k in &adj[j] {
            if k <= j {
                continue; // each central bond once
            }
            for &i in &adj[j] {
                if i == k {
                    continue;
                }
                for &l in &adj[k] {
                    if l == j {
                        continue;
                    }
                    dihedrals.push((i, j, k, l));
                }
            }
        }
    }
    dihedrals
}

// ===========================================================================
// Step 4: Enumerate improper dihedrals
// ===========================================================================

/// Enumerate candidate improper dihedrals.
///
/// For each atom that has exactly 3 bonds and is a plausible sp2 center
/// (C, N, or similar), check whether the force field defines an improper
/// parameter for the combination. Returns (i, j, k, l) tuples in AMBER
/// convention where the third atom (index 2, 0-based) is the central sp2 atom.
fn enumerate_impropers(
    system: &System,
    adj: &[Vec<usize>],
    ff_params: &ForceFieldParams,
) -> Vec<(usize, usize, usize, usize)> {
    let mut impropers = Vec::new();

    for (center, neighbors) in adj.iter().enumerate() {
        if neighbors.len() != 3 {
            continue;
        }

        let element = &system.atoms[center].element;
        // Only consider sp2-like centers.
        if !matches!(element.as_str(), "C" | "N" | "P") {
            continue;
        }

        let a = neighbors[0];
        let b = neighbors[1];
        let c = neighbors[2];

        let ct = &system.atoms[center].atom_type;

        // Try all permutations of the three bonded atoms at positions 1, 2, 4
        // in the AMBER improper convention (position 3 = center).
        // We check each permutation against the force field and take the
        // first match, which respects AMBER's specificity ordering.
        let perms: [(usize, usize, usize); 6] = [
            (a, b, c),
            (a, c, b),
            (b, a, c),
            (b, c, a),
            (c, a, b),
            (c, b, a),
        ];

        let mut found = false;
        for (p1, p2, p4) in &perms {
            let t1 = &system.atoms[*p1].atom_type;
            let t2 = &system.atoms[*p2].atom_type;
            let t4 = &system.atoms[*p4].atom_type;

            if ff_params.find_improper(t1, t2, ct, t4).is_some() {
                impropers.push((*p1, *p2, center, *p4));
                found = true;
                break;
            }
        }

        // If no specific match found, try with wildcards by using a canonical
        // ordering: sort the three bonded atoms by atom type name, then by index.
        if !found {
            let mut sorted_neighbors = [a, b, c];
            sorted_neighbors.sort_by(|&x, &y| {
                system.atoms[x]
                    .atom_type
                    .cmp(&system.atoms[y].atom_type)
                    .then(x.cmp(&y))
            });
            let s1 = sorted_neighbors[0];
            let s2 = sorted_neighbors[1];
            let s4 = sorted_neighbors[2];

            let t1 = &system.atoms[s1].atom_type;
            let t2 = &system.atoms[s2].atom_type;
            let t4 = &system.atoms[s4].atom_type;

            if ff_params.find_improper(t1, t2, ct, t4).is_some() {
                impropers.push((s1, s2, center, s4));
            }
        }
    }

    impropers
}

// ===========================================================================
// Step 5a: Resolve bond parameters
// ===========================================================================

/// Look up force field parameters for every bond and deduplicate bond types.
///
/// Returns (unique bond types, list of (atom_i, atom_j, type_index) tuples).
#[allow(clippy::type_complexity)]
fn resolve_bond_params(
    system: &System,
    ff_params: &ForceFieldParams,
) -> Result<(Vec<BondType>, Vec<(usize, usize, usize)>), String> {
    let mut bond_types: Vec<BondType> = Vec::new();
    let mut bond_type_map: HashMap<String, usize> = HashMap::new();
    let mut resolved: Vec<(usize, usize, usize)> = Vec::new();

    for bond in &system.bonds {
        let t1 = &system.atoms[bond.atom1].atom_type;
        let t2 = &system.atoms[bond.atom2].atom_type;

        let param = ff_params.find_bond(t1, t2).ok_or_else(|| {
            format!(
                "Missing bond parameter for {}-{} (atoms {} {}, {} {})",
                t1,
                t2,
                bond.atom1,
                system.atoms[bond.atom1].name,
                bond.atom2,
                system.atoms[bond.atom2].name,
            )
        })?;

        let key = canonical_bond_key(t1, t2);
        let type_idx = *bond_type_map.entry(key).or_insert_with(|| {
            let idx = bond_types.len();
            bond_types.push(BondType {
                force_constant: param.force_constant,
                eq_value: param.eq_length,
            });
            idx
        });

        resolved.push((bond.atom1, bond.atom2, type_idx));
    }

    Ok((bond_types, resolved))
}

// ===========================================================================
// Step 5b: Resolve angle parameters
// ===========================================================================

/// Look up force field parameters for every angle and deduplicate angle types.
#[allow(clippy::type_complexity)]
fn resolve_angle_params(
    system: &System,
    angles: &[(usize, usize, usize)],
    ff_params: &ForceFieldParams,
) -> Result<(Vec<AngleType>, Vec<(usize, usize, usize, usize)>), String> {
    let mut angle_types: Vec<AngleType> = Vec::new();
    let mut angle_type_map: HashMap<String, usize> = HashMap::new();
    let mut resolved: Vec<(usize, usize, usize, usize)> = Vec::new();

    for &(i, j, k) in angles {
        let t1 = &system.atoms[i].atom_type;
        let t2 = &system.atoms[j].atom_type;
        let t3 = &system.atoms[k].atom_type;

        let param = ff_params.find_angle(t1, t2, t3).ok_or_else(|| {
            format!(
                "Missing angle parameter for {}-{}-{} (atoms {}, {}, {})",
                t1, t2, t3, i, j, k,
            )
        })?;

        let key = canonical_angle_key(t1, t2, t3);
        let type_idx = *angle_type_map.entry(key).or_insert_with(|| {
            let idx = angle_types.len();
            angle_types.push(AngleType {
                force_constant: param.force_constant,
                eq_value: param.eq_angle * DEG_TO_RAD,
            });
            idx
        });

        resolved.push((i, j, k, type_idx));
    }

    Ok((angle_types, resolved))
}

// ===========================================================================
// Step 5c: Resolve dihedral parameters
// ===========================================================================

/// Look up force field parameters for every proper dihedral.
///
/// A single dihedral quartet may produce multiple Fourier terms, each getting
/// its own dihedral-type entry. The returned `ResolvedDihedral` records which
/// type indices each quartet maps to.
fn resolve_dihedral_params(
    system: &System,
    dihedrals: &[(usize, usize, usize, usize)],
    ff_params: &ForceFieldParams,
) -> Result<(Vec<DihedralType>, Vec<ResolvedDihedral>), String> {
    let mut dihedral_types: Vec<DihedralType> = Vec::new();
    let mut dihedral_type_map: HashMap<String, usize> = HashMap::new();
    let mut resolved: Vec<ResolvedDihedral> = Vec::new();

    let global_scee = if ff_params.scee > 0.0 {
        ff_params.scee
    } else {
        DEFAULT_SCEE
    };
    let global_scnb = if ff_params.scnb > 0.0 {
        ff_params.scnb
    } else {
        DEFAULT_SCNB
    };

    for &(i, j, k, l) in dihedrals {
        let t1 = &system.atoms[i].atom_type;
        let t2 = &system.atoms[j].atom_type;
        let t3 = &system.atoms[k].atom_type;
        let t4 = &system.atoms[l].atom_type;

        let matches = ff_params.find_dihedrals(t1, t2, t3, t4);
        if matches.is_empty() {
            return Err(format!(
                "Missing dihedral parameter for {}-{}-{}-{} (atoms {}, {}, {}, {})",
                t1, t2, t3, t4, i, j, k, l,
            ));
        }

        // Use the best (most specific) match. In AMBER convention, specific
        // matches (no wildcards) override wildcard matches. If there are
        // multiple entries for the same type combo, they represent multi-term
        // Fourier expansions whose terms should all be included.
        let best = select_best_dihedral(&matches, t1, t2, t3, t4);

        let mut type_indices = Vec::new();
        for dp in best {
            let divider = if dp.divider > 0.0 { dp.divider } else { 1.0 };
            for term in &dp.terms {
                let effective_fc = term.force_constant / divider;
                let key = format!(
                    "{:.6}_{:.1}_{:.6}_{:.4}_{:.4}",
                    effective_fc, term.periodicity, term.phase, global_scee, global_scnb,
                );
                let type_idx = *dihedral_type_map.entry(key).or_insert_with(|| {
                    let idx = dihedral_types.len();
                    dihedral_types.push(DihedralType {
                        force_constant: effective_fc,
                        periodicity: term.periodicity,
                        phase: term.phase * DEG_TO_RAD,
                        scee: global_scee,
                        scnb: global_scnb,
                    });
                    idx
                });
                type_indices.push(type_idx);
            }
        }

        resolved.push(ResolvedDihedral {
            i,
            j,
            k,
            l,
            type_indices,
        });
    }

    Ok((dihedral_types, resolved))
}

/// Select the best (most specific) dihedral parameter match.
///
/// AMBER precedence: specific types (no wildcards) win over wildcard entries.
/// If both specific and wildcard matches exist, only the specific ones are used.
fn select_best_dihedral<'a>(
    matches: &[&'a rst_core::forcefield::parameters::DihedralParam],
    _t1: &str,
    _t2: &str,
    _t3: &str,
    _t4: &str,
) -> Vec<&'a rst_core::forcefield::parameters::DihedralParam> {
    // Partition into specific (no X wildcards) and wildcard matches.
    let mut specific: Vec<&rst_core::forcefield::parameters::DihedralParam> = Vec::new();
    let mut wildcard: Vec<&rst_core::forcefield::parameters::DihedralParam> = Vec::new();

    for dp in matches {
        let is_specific =
            dp.type1 != "X" && dp.type2 != "X" && dp.type3 != "X" && dp.type4 != "X";
        if is_specific {
            specific.push(dp);
        } else {
            wildcard.push(dp);
        }
    }

    if !specific.is_empty() {
        specific
    } else {
        wildcard
    }
}

// ===========================================================================
// Step 5d: Resolve improper parameters
// ===========================================================================

/// Look up force field parameters for every improper dihedral.
fn resolve_improper_params(
    system: &System,
    impropers: &[(usize, usize, usize, usize)],
    ff_params: &ForceFieldParams,
) -> Result<(Vec<DihedralType>, Vec<ResolvedImproper>), String> {
    let mut imp_types: Vec<DihedralType> = Vec::new();
    let mut imp_type_map: HashMap<String, usize> = HashMap::new();
    let mut resolved: Vec<ResolvedImproper> = Vec::new();

    let global_scee = if ff_params.scee > 0.0 {
        ff_params.scee
    } else {
        DEFAULT_SCEE
    };
    let global_scnb = if ff_params.scnb > 0.0 {
        ff_params.scnb
    } else {
        DEFAULT_SCNB
    };

    for &(i, j, k, l) in impropers {
        let t1 = &system.atoms[i].atom_type;
        let t2 = &system.atoms[j].atom_type;
        let t3 = &system.atoms[k].atom_type; // central atom
        let t4 = &system.atoms[l].atom_type;

        let param = match ff_params.find_improper(t1, t2, t3, t4) {
            Some(p) => p,
            None => {
                // Improper not found is not fatal; the enumeration step already
                // filters, but a race condition could still produce misses. Skip.
                continue;
            }
        };

        let key = format!(
            "{:.6}_{:.1}_{:.6}_{:.4}_{:.4}",
            param.force_constant, param.periodicity, param.phase, global_scee, global_scnb,
        );
        let type_idx = *imp_type_map.entry(key).or_insert_with(|| {
            let idx = imp_types.len();
            imp_types.push(DihedralType {
                force_constant: param.force_constant,
                periodicity: param.periodicity,
                phase: param.phase * DEG_TO_RAD,
                scee: global_scee,
                scnb: global_scnb,
            });
            idx
        });

        resolved.push(ResolvedImproper {
            i,
            j,
            k,
            l,
            type_idx,
        });
    }

    Ok((imp_types, resolved))
}

// ===========================================================================
// Step 6: Build LJ parameter tables
// ===========================================================================

/// Assign atom type indices and compute the NONBONDED_PARM_INDEX and LJ A/B
/// coefficient arrays.
///
/// Returns `(atom_type_indices, n_types, nb_parm_index, lj_acoef, lj_bcoef)`.
#[allow(clippy::type_complexity)]
fn build_lj_tables(
    system: &System,
    ff_params: &ForceFieldParams,
) -> Result<(Vec<usize>, usize, Vec<i32>, Vec<f64>, Vec<f64>), String> {
    // Collect unique atom types in the order they first appear.
    let mut type_name_to_idx: HashMap<String, usize> = HashMap::new();
    let mut type_names: Vec<String> = Vec::new();

    for atom in &system.atoms {
        let tname = atom.atom_type.clone();
        if !type_name_to_idx.contains_key(&tname) {
            let idx = type_names.len();
            type_name_to_idx.insert(tname.clone(), idx);
            type_names.push(tname);
        }
    }

    let n_types = type_names.len();

    // Build per-atom type index array.
    let atom_type_indices: Vec<usize> = system
        .atoms
        .iter()
        .map(|a| type_name_to_idx[&a.atom_type])
        .collect();

    // Fetch LJ parameters for each type.
    let mut radii_vec: Vec<f64> = Vec::with_capacity(n_types);
    let mut eps_vec: Vec<f64> = Vec::with_capacity(n_types);

    for tname in &type_names {
        match ff_params.find_lj(tname) {
            Some(lj) => {
                radii_vec.push(lj.radius);
                eps_vec.push(lj.well_depth);
            }
            None => {
                // Virtual sites (EP) have zero LJ params.
                if tname == "EP" || tname == "LP" {
                    radii_vec.push(0.0);
                    eps_vec.push(0.0);
                } else {
                    return Err(format!("Missing LJ parameters for atom type '{}'", tname));
                }
            }
        }
    }

    // Build the ntypes*ntypes NONBONDED_PARM_INDEX and the A/B coefficient arrays.
    // The index array maps (type_i, type_j) -> 1-based index into the A/B arrays.
    // The A/B arrays are stored as a triangular matrix with ntypes*(ntypes+1)/2 entries.
    let n_lj_pairs = n_types * (n_types + 1) / 2;
    let mut lj_acoef: Vec<f64> = vec![0.0; n_lj_pairs];
    let mut lj_bcoef: Vec<f64> = vec![0.0; n_lj_pairs];
    let mut nb_parm_index: Vec<i32> = vec![0; n_types * n_types];

    let mut pair_idx: usize = 0;
    for i in 0..n_types {
        for j in 0..=i {
            // Lorentz-Berthelot combining rules.
            // AMBER stores Rmin/2, so Rmin = R_i + R_j (the sum of the two Rmin/2 values).
            let sigma = radii_vec[i] + radii_vec[j];
            let epsilon = (eps_vec[i] * eps_vec[j]).sqrt();

            // A = eps * Rmin^12, B = 2 * eps * Rmin^6
            let sigma6 = sigma.powi(6);
            let sigma12 = sigma6 * sigma6;
            let a = epsilon * sigma12;
            let b = 2.0 * epsilon * sigma6;

            lj_acoef[pair_idx] = a;
            lj_bcoef[pair_idx] = b;

            // 1-based index into the coefficient arrays.
            let idx_1based = (pair_idx + 1) as i32;
            nb_parm_index[n_types * i + j] = idx_1based;
            nb_parm_index[n_types * j + i] = idx_1based;

            pair_idx += 1;
        }
    }

    Ok((atom_type_indices, n_types, nb_parm_index, lj_acoef, lj_bcoef))
}

// ===========================================================================
// Step 7: Exclusion lists
// ===========================================================================

/// Build the NUMBER_EXCLUDED_ATOMS and EXCLUDED_ATOMS_LIST arrays.
///
/// Each atom's exclusion list contains all 1-2 (directly bonded) and 1-3
/// (separated by one bond) neighbors. The list is sorted and deduplicated,
/// and only atoms with index > i are included (AMBER convention: excluded
/// atoms are stored as the higher-numbered atom in each pair).
///
/// If an atom has no excluded atoms, a single `usize::MAX` placeholder is
/// stored (which the prmtop writer converts to 0).
fn build_exclusion_lists(
    n_atoms: usize,
    adj: &[Vec<usize>],
) -> (Vec<usize>, Vec<usize>) {
    let mut num_excluded: Vec<usize> = Vec::with_capacity(n_atoms);
    let mut excluded_list: Vec<usize> = Vec::new();

    for i in 0..n_atoms {
        let mut excluded: BTreeSet<usize> = BTreeSet::new();

        // 1-2 neighbors (directly bonded).
        for &j in &adj[i] {
            if j > i {
                excluded.insert(j);
            }
        }

        // 1-3 neighbors (two bonds away).
        for &j in &adj[i] {
            for &k in &adj[j] {
                if k != i && k > i {
                    excluded.insert(k);
                }
            }
        }

        if excluded.is_empty() {
            num_excluded.push(1);
            excluded_list.push(usize::MAX); // placeholder
        } else {
            num_excluded.push(excluded.len());
            excluded_list.extend(excluded);
        }
    }

    (num_excluded, excluded_list)
}

// ===========================================================================
// Step 8: Format bonded topology lists
// ===========================================================================

/// Returns true if any atom in the given indices has atomic_number == 1 (hydrogen).
fn involves_hydrogen(system: &System, atom_indices: &[usize]) -> bool {
    atom_indices
        .iter()
        .any(|&idx| system.atoms[idx].atomic_number == 1)
}

/// Format bond topology entries, split by hydrogen content.
///
/// Each bond becomes a triplet: (atom_i * 3, atom_j * 3, type_index_1based).
fn format_bond_topology(
    system: &System,
    resolved_bonds: &[(usize, usize, usize)],
) -> (Vec<i32>, Vec<i32>) {
    let mut inc_h: Vec<i32> = Vec::new();
    let mut without_h: Vec<i32> = Vec::new();

    for &(ai, aj, type_idx) in resolved_bonds {
        let entry = [
            (ai * 3) as i32,
            (aj * 3) as i32,
            (type_idx + 1) as i32, // 1-based
        ];

        if involves_hydrogen(system, &[ai, aj]) {
            inc_h.extend_from_slice(&entry);
        } else {
            without_h.extend_from_slice(&entry);
        }
    }

    (inc_h, without_h)
}

/// Format angle topology entries, split by hydrogen content.
///
/// Each angle becomes a quad: (atom_i*3, atom_j*3, atom_k*3, type_index_1based).
fn format_angle_topology(
    system: &System,
    resolved_angles: &[(usize, usize, usize, usize)],
) -> (Vec<i32>, Vec<i32>) {
    let mut inc_h: Vec<i32> = Vec::new();
    let mut without_h: Vec<i32> = Vec::new();

    for &(ai, aj, ak, type_idx) in resolved_angles {
        let entry = [
            (ai * 3) as i32,
            (aj * 3) as i32,
            (ak * 3) as i32,
            (type_idx + 1) as i32,
        ];

        if involves_hydrogen(system, &[ai, aj, ak]) {
            inc_h.extend_from_slice(&entry);
        } else {
            without_h.extend_from_slice(&entry);
        }
    }

    (inc_h, without_h)
}

/// Format proper and improper dihedral topology entries, split by hydrogen content.
///
/// Each dihedral becomes a quintuplet:
///   (atom_i*3, atom_j*3, +/-atom_k*3, +/-atom_l*3, type_index_1based)
///
/// Sign conventions for multi-term dihedrals:
/// - For a dihedral with N Fourier terms (N > 1), the first N-1 entries have
///   negative `atom_k` to signal "ignore end-group interactions" (1-4 nonbonded
///   are only computed once for the last term).
/// - `atom_l` is negated for multi-term proper dihedrals to signal "this is part
///   of a multi-term dihedral, do not compute 1-4 NB again".
///
/// For improper dihedrals, `atom_l` is always negative (AMBER convention).
fn format_dihedral_topology(
    system: &System,
    resolved_dihedrals: &[ResolvedDihedral],
    resolved_impropers: &[ResolvedImproper],
    imp_type_offset: usize,
) -> (Vec<i32>, Vec<i32>) {
    let mut inc_h: Vec<i32> = Vec::new();
    let mut without_h: Vec<i32> = Vec::new();

    // Proper dihedrals.
    for rd in resolved_dihedrals {
        let n_terms = rd.type_indices.len();
        let has_h = involves_hydrogen(system, &[rd.i, rd.j, rd.k, rd.l]);

        for (term_pos, &type_idx) in rd.type_indices.iter().enumerate() {
            let ai = (rd.i * 3) as i32;
            let aj = (rd.j * 3) as i32;
            let mut ak = (rd.k * 3) as i32;
            let mut al = (rd.l * 3) as i32;

            if n_terms > 1 {
                // For multi-term dihedrals: negate l for all terms to signal
                // multi-term, and negate k for all but the last term to suppress
                // redundant 1-4 NB calculations.
                al = -al;
                if term_pos < n_terms - 1 {
                    ak = -ak;
                }
            }

            let type_1based = (type_idx + 1) as i32;
            let entry = [ai, aj, ak, al, type_1based];

            if has_h {
                inc_h.extend_from_slice(&entry);
            } else {
                without_h.extend_from_slice(&entry);
            }
        }
    }

    // Improper dihedrals: atom_l is always negated.
    for ri in resolved_impropers {
        let has_h = involves_hydrogen(system, &[ri.i, ri.j, ri.k, ri.l]);

        let ai = (ri.i * 3) as i32;
        let aj = (ri.j * 3) as i32;
        let ak = (ri.k * 3) as i32;
        let al = -((ri.l * 3) as i32); // negative l for impropers
        let type_1based = (ri.type_idx + imp_type_offset + 1) as i32;

        let entry = [ai, aj, ak, al, type_1based];

        if has_h {
            inc_h.extend_from_slice(&entry);
        } else {
            without_h.extend_from_slice(&entry);
        }
    }

    (inc_h, without_h)
}

// ===========================================================================
// Step 9: Born radii and screening parameters
// ===========================================================================

/// Assign Born radii and screening parameters using mbondi3 radii.
///
/// The mbondi3 set is the default for AMBER simulations with the OPC water model
/// and the GBNeck2 generalized Born model.
fn assign_born_params(system: &System, adj: &[Vec<usize>]) -> (Vec<f64>, Vec<f64>) {
    let n_atoms = system.n_atoms();
    let mut radii = Vec::with_capacity(n_atoms);
    let mut screen = Vec::with_capacity(n_atoms);

    for (idx, atom) in system.atoms.iter().enumerate() {
        let r = assign_born_radius(&atom.element, &atom.name, idx, system, adj);
        let s = assign_screen(&atom.element);
        radii.push(r);
        screen.push(s);
    }

    (radii, screen)
}

/// Assign a single Born radius using the mbondi3 scheme.
///
/// Most elements get a fixed radius. Hydrogen radii depend on what they are
/// bonded to:
/// - H bonded to N: 0.80 Angstroms
/// - H bonded to O: 1.00 Angstroms
/// - All other H: 1.20 Angstroms
fn assign_born_radius(
    element: &str,
    _atom_name: &str,
    atom_idx: usize,
    system: &System,
    adj: &[Vec<usize>],
) -> f64 {
    match element {
        "H" => {
            // Check what element this hydrogen is bonded to.
            for &neighbor in &adj[atom_idx] {
                let neighbor_element = system.atoms[neighbor].element.as_str();
                match neighbor_element {
                    "N" => return 0.80,
                    "O" => return 1.00,
                    _ => {}
                }
            }
            1.20
        }
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.50,
        "S" => 1.80,
        "P" => 1.85,
        "F" => 1.47,
        "Cl" => 1.77,
        "Br" => 1.85,
        "I" => 1.98,
        _ => 1.50,
    }
}

/// Assign a Generalized Born screening parameter for a given element.
fn assign_screen(element: &str) -> f64 {
    match element {
        "C" => 0.72,
        "H" => 0.85,
        "N" => 0.79,
        "O" => 0.85,
        "S" => 0.96,
        _ => 0.80,
    }
}

// ===========================================================================
// Step 10: Box info
// ===========================================================================

/// Build the `BoxInfo` structure if the system has periodic box dimensions.
fn build_box_info(system: &System) -> Option<BoxInfo> {
    let dims = system.box_dimensions?;

    let molecules = system.find_molecules();
    let atoms_per_molecule: Vec<usize> = molecules.iter().map(|m| m.atom_indices.len()).collect();

    // Determine last solute residue (1-based) and first solvent molecule (1-based).
    // A residue is considered solvent if its name matches a known water or ion name.
    let last_solute_residue = find_last_solute_residue(system);
    let first_solvent_molecule = find_first_solvent_molecule(system, &molecules);

    // Count virtual sites (extra points with atomic_number == 0).
    let num_extra_points = system
        .atoms
        .iter()
        .filter(|a| a.atomic_number == 0)
        .count();

    let beta = system
        .box_angles
        .map_or(90.0, |angles| angles[1]);

    Some(BoxInfo {
        beta,
        dimensions: dims,
        last_solute_residue,
        atoms_per_molecule,
        first_solvent_molecule,
        num_extra_points,
    })
}

/// Find the 1-based index of the last non-solvent, non-ion residue.
fn find_last_solute_residue(system: &System) -> usize {
    let mut last_solute: usize = 0; // default: no solute residues

    for (idx, res) in system.residues.iter().enumerate() {
        if !is_solvent_or_ion(&res.name) {
            last_solute = idx + 1; // 1-based
        }
    }

    // If everything is solute (no solvent), return the total number of residues.
    if last_solute == 0 {
        system.residues.len()
    } else {
        last_solute
    }
}

/// Find the 1-based index of the first solvent molecule.
fn find_first_solvent_molecule(
    system: &System,
    molecules: &[crate::system::Molecule],
) -> usize {
    // Build a set of residue indices that belong to solvent.
    let solvent_residues: HashSet<usize> = system
        .residues
        .iter()
        .enumerate()
        .filter(|(_, r)| is_solvent_or_ion(&r.name))
        .map(|(idx, _)| idx)
        .collect();

    if solvent_residues.is_empty() {
        // No solvent; return total molecules + 1 (meaning "no solvent" in AMBER).
        return molecules.len() + 1;
    }

    // Find the first molecule that contains an atom belonging to a solvent residue.
    for (mol_idx, mol) in molecules.iter().enumerate() {
        if let Some(&first_atom) = mol.atom_indices.first() {
            let res_idx = system.atoms[first_atom].residue_idx;
            if solvent_residues.contains(&res_idx) {
                return mol_idx + 1; // 1-based
            }
        }
    }

    molecules.len() + 1
}

/// Check whether a residue name corresponds to solvent or an ion.
fn is_solvent_or_ion(name: &str) -> bool {
    SOLVENT_RESIDUES.contains(&name)
        || ION_RESIDUES.contains(&name)
}

// ===========================================================================
// Canonical key helpers
// ===========================================================================

/// Produce a canonical key for a bond type pair (alphabetically sorted).
fn canonical_bond_key(t1: &str, t2: &str) -> String {
    if t1 <= t2 {
        format!("{}-{}", t1, t2)
    } else {
        format!("{}-{}", t2, t1)
    }
}

/// Produce a canonical key for an angle type triple (central type fixed,
/// end types alphabetically sorted).
fn canonical_angle_key(t1: &str, t2: &str, t3: &str) -> String {
    if t1 <= t3 {
        format!("{}-{}-{}", t1, t2, t3)
    } else {
        format!("{}-{}-{}", t3, t2, t1)
    }
}
