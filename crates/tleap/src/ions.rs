//! Ion placement by random water replacement.
//!
//! Implements the `addIonsRand` algorithm from AMBER's tleap: ions are placed
//! into a solvated system by randomly selecting water molecules and replacing
//! them with monatomic ions. A minimum inter-ion distance constraint is
//! enforced to prevent clustering.

use crate::system::{Atom, System};
use rst_core::forcefield::atom_types::element_from_type;
use rst_core::forcefield::residue_lib::ResidueLibrary;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// How many ions to add.
#[derive(Debug, Clone, Copy)]
pub enum IonCount {
    /// Neutralize the system charge.
    Neutralize,
    /// Add exactly this many ions.
    Count(usize),
    /// Add ions to achieve the given molar concentration of a 1:1 salt.
    ///
    /// The number of ion pairs is computed from the water count using:
    ///   `n = round(C * n_water / (55.5 + 2*C))`
    /// where 55.5 mol/L is the molarity of pure water and the factor of 2
    /// accounts for the two water molecules displaced per ion pair.
    ///
    /// **Important**: this variant is meant to be called for both the cation
    /// and anion of a 1:1 salt (e.g. Na+ then Cl-) with the same
    /// concentration value. If the system needs neutralization, call
    /// `add_ions` with [`IonCount::Neutralize`] first; those neutralization
    /// ions are *not* automatically subtracted here so that users retain
    /// full control over the workflow.
    Concentration(f64),
}

/// Configuration for ion placement.
pub struct IonConfig {
    /// Minimum distance between any two ions (Angstroms).
    pub min_ion_distance: f64,
}

impl Default for IonConfig {
    fn default() -> Self {
        Self {
            min_ion_distance: 4.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Simple RNG (LCG)
// ---------------------------------------------------------------------------

/// A minimal linear congruential generator so that we do not need a `rand`
/// crate dependency for this module. Parameters match `java.util.Random`.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    #[allow(clippy::cast_possible_truncation)]
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two 3-D points.
fn dist_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Compute the total charge of a residue template by summing atom charges.
fn template_charge(lib: &ResidueLibrary, ion_name: &str) -> Result<f64, String> {
    let template = lib
        .get(ion_name)
        .ok_or_else(|| format!("Ion template '{ion_name}' not found in residue library"))?;
    Ok(template.atoms.iter().map(|a| a.charge).sum())
}

/// Look up the mass for a given element symbol.
///
/// Returns a reasonable default (0.0) only for unrecognized elements.
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
        _ => 0.0,
    }
}

/// Atomic number for a given element symbol.
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
        _ => 0,
    }
}

/// A candidate water molecule: its residue index and oxygen position.
struct WaterCandidate {
    residue_idx: usize,
    oxygen_pos: [f64; 3],
}

/// Count the number of water residues (name == `"WAT"`) in the system.
fn count_water_residues(system: &System) -> usize {
    system
        .residues
        .iter()
        .filter(|r| r.name == "WAT")
        .count()
}

/// Collect all water molecules from the system.
///
/// A water molecule is identified by residue name `"WAT"`. The oxygen atom is
/// taken as the first atom in the residue whose name starts with `'O'`, whose
/// element is `"O"`, or whose atomic number is 8. Falls back to the first atom
/// of the residue if no oxygen is found.
fn collect_waters(system: &System) -> Vec<WaterCandidate> {
    let mut waters = Vec::new();

    for (res_idx, res) in system.residues.iter().enumerate() {
        if res.name != "WAT" {
            continue;
        }

        // Find the oxygen atom within this residue.
        let oxygen_pos = find_oxygen_position(system, res);

        waters.push(WaterCandidate {
            residue_idx: res_idx,
            oxygen_pos,
        });
    }

    waters
}

/// Locate the oxygen position in a water residue.
///
/// Searches atoms in the residue's atom range for the first atom whose name
/// starts with `'O'`, whose element is `"O"`, or whose atomic number is 8.
/// Falls back to the position of the first atom in the residue.
fn find_oxygen_position(system: &System, res: &crate::system::Residue) -> [f64; 3] {
    system.atoms[res.atom_range.clone()]
        .iter()
        .find(|a| a.name.starts_with('O') || a.element == "O" || a.atomic_number == 8)
        .map_or_else(
            || system.atoms[res.atom_range.start].position,
            |a| a.position,
        )
}

/// Molarity of pure liquid water (mol/L) at ~25 °C / 1 atm.
const WATER_MOLARITY: f64 = 55.5;

/// Determine how many ions to place given the requested [`IonCount`], the
/// system's current total charge, the charge per ion, and the number of
/// water molecules currently in the system (needed only for the
/// `Concentration` variant).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn resolve_count(
    count: IonCount,
    system_charge: f64,
    ion_charge: f64,
    n_water: usize,
) -> Result<usize, String> {
    match count {
        IonCount::Count(n) => Ok(n),
        IonCount::Neutralize => {
            if ion_charge.abs() < 1e-10 {
                return Err(
                    "Cannot neutralize with a zero-charge ion".to_string(),
                );
            }

            // We need `n` ions such that system_charge + n * ion_charge ~ 0.
            // n = -system_charge / ion_charge, rounded to nearest integer.
            let n_f64 = -system_charge / ion_charge;

            if n_f64 < 0.0 {
                // The ion would make the charge worse, not better. For example
                // the system is positively charged and the user asked to
                // neutralize with Na+ (also positive).
                return Ok(0);
            }

            Ok(n_f64.round() as usize)
        }
        IonCount::Concentration(conc) => {
            if conc <= 0.0 {
                return Err(format!(
                    "Concentration must be positive, got {conc:.4} M"
                ));
            }
            if n_water == 0 {
                return Err(
                    "Cannot compute ion count from concentration: \
                     no water molecules in the system"
                        .to_string(),
                );
            }

            // n = round(C * n_water / (55.5 + 2*C))
            // The denominator accounts for water displacement: each ion
            // pair replaces ~2 water molecules.
            let n_f64 =
                conc * (n_water as f64) / (WATER_MOLARITY + 2.0 * conc);
            let n = n_f64.round().max(0.0) as usize;

            // Sanity check: we need at least 2*n waters (one per ion).
            if 2 * n > n_water {
                return Err(format!(
                    "Requested concentration {conc:.4} M requires {n} ion pairs \
                     but there are only {n_water} water molecules. \
                     Reduce concentration or increase box size."
                ));
            }

            log::info!(
                "Concentration {conc:.4} M with {n_water} waters -> {n} ions"
            );

            Ok(n)
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Maximum number of random attempts to place a single ion before giving up.
const MAX_PLACEMENT_ATTEMPTS: usize = 10_000;

/// Add ions to a solvated system by replacing water molecules.
///
/// This is the Rust equivalent of AMBER tleap's `addIonsRand` command. Ions
/// are placed at the oxygen positions of randomly selected water molecules,
/// subject to a minimum inter-ion distance constraint.
///
/// # Arguments
///
/// * `system` - The solvated system (must already contain water molecules).
/// * `residue_lib` - Residue library containing ion templates (e.g. from
///   `atomic_ions.lib`).
/// * `ion_name` - Ion residue name as it appears in the library (e.g.
///   `"Na+"`, `"Cl-"`, `"K+"`, `"Mg2+"`).
/// * `count` - How many ions to add, or [`IonCount::Neutralize`] to
///   bring the system charge to approximately zero.
/// * `config` - Placement configuration (minimum distance, etc.).
/// * `rng_seed` - Optional seed for the random number generator. Pass
///   `None` for a default deterministic seed, or `Some(seed)`
///   for reproducible results.
///
/// # Returns
///
/// The number of ions actually placed. This equals the requested count unless
/// it is impossible to satisfy the distance constraint, in which case fewer
/// ions are placed and an `Err` is returned.
///
/// # Errors
///
/// Returns `Err` in the following situations:
/// - No water molecules found in the system.
/// - Ion template not found in the residue library.
/// - Not enough water molecules to place the requested number of ions.
/// - Unable to satisfy the minimum distance constraint for all ions after
///   exhausting random attempts.
#[allow(clippy::too_many_lines)]
pub fn add_ions(
    system: &mut System,
    residue_lib: &ResidueLibrary,
    ion_name: &str,
    count: IonCount,
    config: &IonConfig,
    rng_seed: Option<u64>,
) -> Result<usize, String> {
    // ------------------------------------------------------------------
    // 1. Validate the ion template.
    // ------------------------------------------------------------------
    let template = residue_lib
        .get(ion_name)
        .ok_or_else(|| format!("Ion template '{ion_name}' not found in residue library"))?;

    if template.atoms.is_empty() {
        return Err(format!("Ion template '{ion_name}' has no atoms"));
    }

    // Extract properties from the first (and typically only) atom in the ion
    // template.
    let ion_atom_template = &template.atoms[0];
    let ion_atom_type = ion_atom_template.atom_type.clone();
    let ion_charge_per_atom = ion_atom_template.charge;
    let ion_element_number = ion_atom_template.element_number;
    let ion_atom_name = ion_atom_template.name.clone();

    // Derive element symbol and mass.
    let ion_element = element_from_type(&ion_atom_type)
        .unwrap_or_else(|| {
            // Fallback: try to derive from the ion name itself (e.g. "Na+").
            element_from_type(ion_name).unwrap_or("X")
        })
        .to_string();
    let ion_mass = mass_for_element(&ion_element);
    let ion_atomic_number = if ion_element_number > 0 {
        ion_element_number
    } else {
        atomic_number_for_element(&ion_element)
    };

    // Total charge of the ion template (sum of all atoms, but for monatomic
    // ions this is just the single atom charge).
    let ion_template_charge = template_charge(residue_lib, ion_name)?;

    // ------------------------------------------------------------------
    // 2. Determine how many ions to place.
    // ------------------------------------------------------------------
    let system_charge = system.total_charge();
    let n_water = count_water_residues(system);
    let n_ions = resolve_count(count, system_charge, ion_template_charge, n_water)?;

    if n_ions == 0 {
        log::info!(
            "add_ions: system charge is {system_charge:.4}, no {ion_name} ions needed"
        );
        return Ok(0);
    }

    // ------------------------------------------------------------------
    // 3. Collect water molecules.
    // ------------------------------------------------------------------
    let mut waters = collect_waters(system);

    if waters.is_empty() {
        return Err("No water molecules (WAT) found in the system".to_string());
    }

    if waters.len() < n_ions {
        return Err(format!(
            "Not enough water molecules to place {n_ions} ions \
             (only {} WAT residues found)",
            waters.len()
        ));
    }

    log::info!(
        "add_ions: placing {n_ions} {ion_name} ions \
         (system charge {system_charge:.4}, ion charge {ion_template_charge:.4}, \
         {} waters available)",
        waters.len()
    );

    // ------------------------------------------------------------------
    // 4. Randomly select water molecules for replacement.
    // ------------------------------------------------------------------
    let seed = rng_seed.unwrap_or(42);
    let mut rng = SimpleRng::new(seed);
    let min_dist_sq = config.min_ion_distance * config.min_ion_distance;

    // Positions of ions placed so far (for distance checking).
    let mut placed_positions: Vec<[f64; 3]> = Vec::with_capacity(n_ions);
    // (water_residue_idx, ion_position) pairs for the replacements.
    let mut replacements: Vec<(usize, [f64; 3])> = Vec::with_capacity(n_ions);

    for ion_idx in 0..n_ions {
        let placed = try_place_single_ion(
            &mut waters,
            &mut rng,
            min_dist_sq,
            &mut placed_positions,
            &mut replacements,
        );

        if !placed {
            let placed_so_far = replacements.len();
            return Err(format!(
                "Could not place ion {} of {n_ions} after {MAX_PLACEMENT_ATTEMPTS} attempts \
                 (placed {placed_so_far} ions successfully, {} waters remaining). \
                 Consider reducing min_ion_distance (currently {:.1} A) \
                 or adding more solvent.",
                ion_idx + 1,
                waters.len(),
                config.min_ion_distance
            ));
        }
    }

    // ------------------------------------------------------------------
    // 5. Replace water molecules with ions (batch operation).
    // ------------------------------------------------------------------
    apply_replacements(
        system,
        &replacements,
        ion_name,
        &ion_atom_name,
        &ion_atom_type,
        &ion_element,
        ion_charge_per_atom,
        ion_mass,
        ion_atomic_number,
    );

    let n_placed = replacements.len();
    log::info!(
        "add_ions: placed {n_placed} {ion_name} ions, \
         new system charge {:.4}",
        system.total_charge()
    );

    Ok(n_placed)
}

/// Attempt to place a single ion by randomly selecting a water molecule that
/// satisfies the minimum distance constraint to all previously placed ions.
///
/// Returns `true` if a valid placement was found and recorded.
fn try_place_single_ion(
    waters: &mut Vec<WaterCandidate>,
    rng: &mut SimpleRng,
    min_dist_sq: f64,
    placed_positions: &mut Vec<[f64; 3]>,
    replacements: &mut Vec<(usize, [f64; 3])>,
) -> bool {
    for _attempt in 0..MAX_PLACEMENT_ATTEMPTS {
        if waters.is_empty() {
            return false;
        }

        let water_idx = rng.next_usize(waters.len());
        let candidate = &waters[water_idx];

        // Check minimum distance to all previously placed ions.
        let too_close = placed_positions
            .iter()
            .any(|prev| dist_sq(prev, &candidate.oxygen_pos) < min_dist_sq);

        if too_close {
            continue;
        }

        // Accept this water for replacement.
        let pos = candidate.oxygen_pos;
        let res_idx = candidate.residue_idx;

        placed_positions.push(pos);
        replacements.push((res_idx, pos));

        // Remove from available waters (swap-remove for O(1)).
        waters.swap_remove(water_idx);
        return true;
    }

    false
}

/// Apply all ion replacements to the system in batch.
///
/// This removes the selected water molecules' atoms, then appends the ion
/// residues at the recorded positions.
#[allow(clippy::too_many_arguments)]
fn apply_replacements(
    system: &mut System,
    replacements: &[(usize, [f64; 3])],
    ion_name: &str,
    ion_atom_name: &str,
    ion_atom_type: &str,
    ion_element: &str,
    ion_charge: f64,
    ion_mass: f64,
    ion_atomic_number: i32,
) {
    // Sort replacements by residue index so we can collect atoms in order.
    let mut sorted: Vec<(usize, [f64; 3])> = replacements.to_vec();
    sorted.sort_by_key(|&(res_idx, _)| res_idx);

    // Collect atom indices to remove (must be ascending).
    let mut atoms_to_remove: Vec<usize> = Vec::new();
    let mut ion_positions: Vec<[f64; 3]> = Vec::with_capacity(sorted.len());

    for &(res_idx, pos) in &sorted {
        let res = &system.residues[res_idx];
        atoms_to_remove.extend(res.atom_range.clone());
        ion_positions.push(pos);
    }

    // atoms_to_remove is already sorted because we sorted replacements by
    // residue index and residues have non-overlapping ascending atom ranges.
    debug_assert!(atoms_to_remove.windows(2).all(|w| w[0] < w[1]));

    // Remove all water atoms in one pass.
    system.remove_atoms(&atoms_to_remove);

    // Now add each ion as a new residue.
    for pos in &ion_positions {
        let ion_atom = Atom {
            name: ion_atom_name.to_owned(),
            atom_type: ion_atom_type.to_owned(),
            element: ion_element.to_owned(),
            charge: ion_charge,
            mass: ion_mass,
            atomic_number: ion_atomic_number,
            position: *pos,
            residue_idx: 0, // overwritten by add_residue
            born_radius: 0.0,
            screen: 0.0,
        };

        system.add_residue(
            ion_name,
            ' ',  // chain_id: ions typically have no chain
            0,    // seq_num: will be renumbered later if needed
            vec![ion_atom],
            vec![], // monatomic ion has no internal bonds
        );
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_count_exact() {
        assert_eq!(resolve_count(IonCount::Count(5), 0.0, 1.0, 100).unwrap(), 5);
    }

    #[test]
    fn resolve_count_neutralize_positive_charge() {
        // System has +3 charge, adding Cl- (charge -1) should give 3.
        assert_eq!(resolve_count(IonCount::Neutralize, 3.0, -1.0, 100).unwrap(), 3);
    }

    #[test]
    fn resolve_count_neutralize_negative_charge() {
        // System has -4 charge, adding Na+ (charge +1) should give 4.
        assert_eq!(resolve_count(IonCount::Neutralize, -4.0, 1.0, 100).unwrap(), 4);
    }

    #[test]
    fn resolve_count_neutralize_wrong_sign() {
        // System is +3, trying to neutralize with Na+ (+1) should give 0.
        assert_eq!(resolve_count(IonCount::Neutralize, 3.0, 1.0, 100).unwrap(), 0);
    }

    #[test]
    fn resolve_count_concentration_150mm() {
        // 150 mM = 0.150 M with 10000 waters.
        // n = round(0.150 * 10000 / (55.5 + 0.300)) = round(1500 / 55.8) = round(26.88) = 27
        let n = resolve_count(IonCount::Concentration(0.150), 0.0, 1.0, 10_000).unwrap();
        assert_eq!(n, 27);
    }

    #[test]
    fn resolve_count_concentration_150mm_small_box() {
        // 150 mM with 3000 waters.
        // n = round(0.150 * 3000 / 55.8) = round(450 / 55.8) = round(8.06) = 8
        let n = resolve_count(IonCount::Concentration(0.150), 0.0, 1.0, 3_000).unwrap();
        assert_eq!(n, 8);
    }

    #[test]
    fn resolve_count_concentration_1m() {
        // 1.0 M with 10000 waters.
        // n = round(1.0 * 10000 / (55.5 + 2.0)) = round(10000 / 57.5) = round(173.91) = 174
        let n = resolve_count(IonCount::Concentration(1.0), 0.0, 1.0, 10_000).unwrap();
        assert_eq!(n, 174);
    }

    #[test]
    fn resolve_count_concentration_zero_waters() {
        let result = resolve_count(IonCount::Concentration(0.150), 0.0, 1.0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_count_concentration_negative() {
        let result = resolve_count(IonCount::Concentration(-0.1), 0.0, 1.0, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_count_concentration_high_but_valid() {
        // Very high concentration (10 M) with 10000 waters.
        // n = round(10.0 * 10000 / (55.5 + 20)) = round(100000 / 75.5) = round(1324.5) = 1325
        let n = resolve_count(IonCount::Concentration(10.0), 0.0, 1.0, 10_000).unwrap();
        assert_eq!(n, 1325);
        // Sanity: 2 * 1325 = 2650 < 10000, OK.
    }

    #[test]
    fn count_water_residues_empty() {
        let system = System::new();
        assert_eq!(count_water_residues(&system), 0);
    }
}
