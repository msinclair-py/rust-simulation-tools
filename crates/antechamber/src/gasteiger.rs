//! Gasteiger electronegativity equalization charges.
//!
//! Iterative charge equalization based on electronegativity parameters.

use crate::data;
use crate::molecule::AcMolecule;

/// Gasteiger parameters for one atom type.
#[derive(Debug, Clone)]
struct GasParam {
    /// Atom type pattern.
    type_name: String,
    /// Electronegativity coefficient a.
    a: f64,
    /// Electronegativity coefficient b.
    b: f64,
    /// Electronegativity coefficient c.
    c: f64,
    /// Total electronegativity d = a + b + c.
    d: f64,
    /// Formal charge for this type.
    formal_charge: f64,
}

/// Parse GASPARM.DAT into parameters.
fn parse_gasteiger_params() -> Vec<GasParam> {
    let mut params = Vec::new();

    for line in data::GASPARM.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 7 || parts[0] != "GASPARM" {
            continue;
        }

        let type_name = parts[1].to_string();
        let a: f64 = parts[2].parse().unwrap_or(0.0);
        let b: f64 = parts[3].parse().unwrap_or(0.0);
        let c: f64 = parts[4].parse().unwrap_or(0.0);
        let d: f64 = parts[5].parse().unwrap_or(0.0);
        let formal_charge: f64 = parts[6].parse().unwrap_or(0.0);

        params.push(GasParam {
            type_name,
            a,
            b,
            c,
            d,
            formal_charge,
        });
    }

    params
}

/// Map a GAFF2 type to Gasteiger parameter type.
fn gaff2_to_gasteiger(_gaff2_type: &str, atomic_number: u8, degree: usize) -> String {
    match atomic_number {
        1 => "h".to_string(),
        6 => {
            match degree {
                4 => "c3".to_string(),
                3 => "c2".to_string(),
                2 => "c1".to_string(),
                _ => "c3".to_string(),
            }
        }
        7 => {
            match degree {
                4 => "n4".to_string(),
                3 => "n3".to_string(),
                2 => "n2".to_string(),
                1 => "n1".to_string(),
                _ => "n3".to_string(),
            }
        }
        8 => {
            match degree {
                1 => "o2".to_string(),
                _ => "o3".to_string(),
            }
        }
        9 => "f".to_string(),
        15 => "p".to_string(),
        16 => {
            match degree {
                1 => "s2".to_string(),
                _ => "s3".to_string(),
            }
        }
        17 => "cl".to_string(),
        35 => "br".to_string(),
        53 => "i".to_string(),
        _ => "c3".to_string(),
    }
}

/// Find Gasteiger parameters for a given type name.
fn find_gas_params<'a>(type_name: &str, params: &'a [GasParam]) -> Option<&'a GasParam> {
    params.iter().find(|p| p.type_name == type_name)
}

/// Compute Gasteiger charges for all atoms.
///
/// Iterative electronegativity equalization:
/// 1. chi_i = a_i + b_i * q_i + c_i * q_i^2
/// 2. Transfer charge along bonds based on electronegativity differences
/// 3. Dampen by factor 0.5^iteration
pub fn compute_gasteiger_charges(mol: &mut AcMolecule) -> Result<(), String> {
    let params = parse_gasteiger_params();
    let n = mol.atoms.len();

    // Map atoms to Gasteiger parameter types
    let gas_types: Vec<String> = mol
        .atoms
        .iter()
        .map(|a| gaff2_to_gasteiger(&a.gaff2_type, a.atomic_number, a.degree))
        .collect();

    // Get parameters for each atom
    let atom_params: Vec<&GasParam> = gas_types
        .iter()
        .enumerate()
        .map(|(_i, t)| {
            find_gas_params(t, &params).unwrap_or_else(|| {
                // Fallback: use default hydrogen params
                find_gas_params("h", &params).unwrap_or(&params[0])
            })
        })
        .collect();

    // Initialize charges
    let mut charges = vec![0.0f64; n];
    for i in 0..n {
        charges[i] = atom_params[i].formal_charge;
    }

    // Iterate
    let max_iter = 6;
    for iter in 0..max_iter {
        let damping = 0.5_f64.powi(iter + 1);

        // Compute electronegativity for each atom
        let chi: Vec<f64> = (0..n)
            .map(|i| {
                let p = atom_params[i];
                let q = charges[i];
                p.a + p.b * q + p.c * q * q
            })
            .collect();

        // Transfer charge along each bond
        for bond in &mol.bonds {
            let a1 = bond.atom1;
            let a2 = bond.atom2;

            // Charge flows from lower electronegativity to higher
            let chi_diff = chi[a1] - chi[a2];

            // The denominator uses the maximum of the two total electronegativities
            let denom = if chi_diff > 0.0 {
                atom_params[a2].d
            } else {
                atom_params[a1].d
            };

            if denom.abs() < 1.0e-10 {
                continue;
            }

            let transfer = chi_diff / denom * damping;
            charges[a1] -= transfer;
            charges[a2] += transfer;
        }
    }

    // Store charges
    for i in 0..n {
        mol.atoms[i].charge = charges[i];
    }

    Ok(())
}
