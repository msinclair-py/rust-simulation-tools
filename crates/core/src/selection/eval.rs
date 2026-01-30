//! Evaluator for selection expressions against an AmberTopology.

use crate::amber::prmtop::AmberTopology;
use crate::selection::ast::*;
use crate::selection::error::SelectionError;
use crate::selection::keywords;

/// Context for evaluating selection expressions.
pub struct SelectionContext<'a> {
    pub topology: &'a AmberTopology,
    pub coordinates: Option<&'a [[f64; 3]]>,
}

impl<'a> SelectionContext<'a> {
    pub fn new(topology: &'a AmberTopology) -> Self {
        Self {
            topology,
            coordinates: None,
        }
    }

    pub fn with_coordinates(topology: &'a AmberTopology, coordinates: &'a [[f64; 3]]) -> Self {
        Self {
            topology,
            coordinates: Some(coordinates),
        }
    }

    /// Parse and evaluate a selection expression string, returning sorted atom indices.
    pub fn eval_str(&self, expr: &str) -> Result<Vec<usize>, SelectionError> {
        let ast = crate::selection::parser::parse_selection(expr)?;
        self.eval(&ast)
    }

    /// Evaluate an AST expression, returning sorted atom indices.
    pub fn eval(&self, expr: &Expr) -> Result<Vec<usize>, SelectionError> {
        let mask = self.eval_mask(expr)?;
        Ok(mask_to_indices(&mask))
    }

    fn eval_mask(&self, expr: &Expr) -> Result<Vec<bool>, SelectionError> {
        let n = self.topology.n_atoms;
        match expr {
            Expr::And(lhs, rhs) => {
                let a = self.eval_mask(lhs)?;
                let b = self.eval_mask(rhs)?;
                Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect())
            }
            Expr::Or(lhs, rhs) => {
                let a = self.eval_mask(lhs)?;
                let b = self.eval_mask(rhs)?;
                Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x || y).collect())
            }
            Expr::Not(inner) => {
                let a = self.eval_mask(inner)?;
                Ok(a.iter().map(|&x| !x).collect())
            }
            Expr::NameMatch { field, pattern } => {
                let mut mask = vec![false; n];
                let residue_indices = self.topology.atom_residue_indices();
                for i in 0..n {
                    let value = match field {
                        StringField::Name => self.topology.atom_names[i].trim(),
                        StringField::Resname => {
                            self.topology.residue_labels[residue_indices[i]].trim()
                        }
                        StringField::Type => self.topology.atom_names[i].trim(), // no separate type field
                    };
                    mask[i] = pattern.matches(value);
                }
                Ok(mask)
            }
            Expr::NumericCmp { field, op, value } => {
                let mut mask = vec![false; n];
                for i in 0..n {
                    let lhs = match field {
                        NumericField::Mass => self.topology.masses[i],
                        NumericField::Charge => self.topology.charges[i],
                        NumericField::Radius => self.topology.radii[i],
                        NumericField::Sigma => self.topology.atom_sigmas[i],
                        NumericField::Epsilon => self.topology.atom_epsilons[i],
                    };
                    mask[i] = op.compare(lhs, *value);
                }
                Ok(mask)
            }
            Expr::RangeSelect { field, ranges } => {
                let mut mask = vec![false; n];
                let residue_indices = self.topology.atom_residue_indices();
                for i in 0..n {
                    let val = match field {
                        RangeField::Resid => (residue_indices[i] as i64) + 1, // 1-based
                        RangeField::Index => i as i64,                        // 0-based
                    };
                    mask[i] = ranges.iter().any(|r| r.contains(val));
                }
                Ok(mask)
            }
            Expr::Within { distance, inner } => {
                let coords = self.coordinates.ok_or_else(|| {
                    SelectionError::new("'within' selection requires coordinates")
                })?;
                if coords.len() != n {
                    return Err(SelectionError::new(format!(
                        "Coordinate count ({}) doesn't match atom count ({})",
                        coords.len(),
                        n
                    )));
                }
                let inner_mask = self.eval_mask(inner)?;
                let inner_indices: Vec<usize> = mask_to_indices(&inner_mask);
                let cutoff_sq = distance * distance;
                let mask = within_cell_list(coords, &inner_indices, cutoff_sq, n);
                Ok(mask)
            }
            Expr::Convenience(kw) => self.eval_convenience(*kw),
        }
    }

    fn eval_convenience(&self, kw: ConvenienceKeyword) -> Result<Vec<bool>, SelectionError> {
        let n = self.topology.n_atoms;
        let residue_indices = self.topology.atom_residue_indices();
        match kw {
            ConvenienceKeyword::All => Ok(vec![true; n]),
            ConvenienceKeyword::None => Ok(vec![false; n]),
            ConvenienceKeyword::Protein => {
                let mut mask = vec![false; n];
                for i in 0..n {
                    let resname = &self.topology.residue_labels[residue_indices[i]];
                    mask[i] = keywords::is_protein_residue(resname);
                }
                Ok(mask)
            }
            ConvenienceKeyword::Water => {
                let mut mask = vec![false; n];
                for i in 0..n {
                    let resname = &self.topology.residue_labels[residue_indices[i]];
                    mask[i] = keywords::is_water_residue(resname);
                }
                Ok(mask)
            }
            ConvenienceKeyword::Backbone => {
                let mut mask = vec![false; n];
                for i in 0..n {
                    let resname = &self.topology.residue_labels[residue_indices[i]];
                    let atomname = &self.topology.atom_names[i];
                    mask[i] = keywords::is_protein_residue(resname)
                        && keywords::is_backbone_atom(atomname);
                }
                Ok(mask)
            }
            ConvenienceKeyword::Sidechain => {
                // protein and not backbone
                let protein = self.eval_convenience(ConvenienceKeyword::Protein)?;
                let backbone = self.eval_convenience(ConvenienceKeyword::Backbone)?;
                Ok(protein
                    .iter()
                    .zip(backbone.iter())
                    .map(|(&p, &b)| p && !b)
                    .collect())
            }
            ConvenienceKeyword::Hydrogen => {
                let mut mask = vec![false; n];
                for i in 0..n {
                    mask[i] = self.topology.masses[i] < 1.1;
                }
                Ok(mask)
            }
        }
    }
}

fn mask_to_indices(mask: &[bool]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect()
}

/// Cell-list based spatial query for "within" selections.
/// Returns a bitmask of atoms within `cutoff_sq` distance of any atom in `source_indices`.
fn within_cell_list(
    coords: &[[f64; 3]],
    source_indices: &[usize],
    cutoff_sq: f64,
    n_atoms: usize,
) -> Vec<bool> {
    let cutoff = cutoff_sq.sqrt();
    if source_indices.is_empty() {
        return vec![false; n_atoms];
    }

    // Find bounding box
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for coord in coords.iter() {
        for d in 0..3 {
            min[d] = min[d].min(coord[d]);
            max[d] = max[d].max(coord[d]);
        }
    }

    let cell_size = cutoff.max(0.1); // avoid zero cell size
    let dims = [
        ((max[0] - min[0]) / cell_size).ceil() as usize + 1,
        ((max[1] - min[1]) / cell_size).ceil() as usize + 1,
        ((max[2] - min[2]) / cell_size).ceil() as usize + 1,
    ];
    let total_cells = dims[0] * dims[1] * dims[2];

    // Build cell list for all atoms
    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); total_cells];
    for i in 0..n_atoms {
        let cx = ((coords[i][0] - min[0]) / cell_size) as usize;
        let cy = ((coords[i][1] - min[1]) / cell_size) as usize;
        let cz = ((coords[i][2] - min[2]) / cell_size) as usize;
        let idx = cx * dims[1] * dims[2] + cy * dims[2] + cz;
        cells[idx].push(i);
    }

    let mut mask = vec![false; n_atoms];

    // For each source atom, check 27 neighboring cells
    for &si in source_indices {
        let cx = ((coords[si][0] - min[0]) / cell_size) as usize;
        let cy = ((coords[si][1] - min[1]) / cell_size) as usize;
        let cz = ((coords[si][2] - min[2]) / cell_size) as usize;

        let x_lo = cx.saturating_sub(1);
        let x_hi = (cx + 1).min(dims[0] - 1);
        let y_lo = cy.saturating_sub(1);
        let y_hi = (cy + 1).min(dims[1] - 1);
        let z_lo = cz.saturating_sub(1);
        let z_hi = (cz + 1).min(dims[2] - 1);

        for ix in x_lo..=x_hi {
            for iy in y_lo..=y_hi {
                for iz in z_lo..=z_hi {
                    let cell_idx = ix * dims[1] * dims[2] + iy * dims[2] + iz;
                    for &j in &cells[cell_idx] {
                        if !mask[j] {
                            let dx = coords[si][0] - coords[j][0];
                            let dy = coords[si][1] - coords[j][1];
                            let dz = coords[si][2] - coords[j][2];
                            let dist_sq = dx * dx + dy * dy + dz * dz;
                            if dist_sq <= cutoff_sq {
                                mask[j] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amber::prmtop::AmberTopology;

    fn make_test_topology() -> AmberTopology {
        // Mini topology: 6 atoms, 2 residues (ALA with 3 atoms, WAT with 3 atoms)
        AmberTopology {
            n_atoms: 6,
            n_residues: 2,
            n_types: 2,
            atom_names: vec![
                "N".to_string(),
                "CA".to_string(),
                "C".to_string(),
                "O".to_string(),
                "H1".to_string(),
                "H2".to_string(),
            ],
            atom_type_indices: vec![0, 1, 0, 1, 0, 0],
            charges: vec![0.1, -0.2, 0.3, -0.4, 0.5, 0.5],
            charges_amber: vec![1.82, -3.64, 5.47, -7.29, 9.11, 9.11],
            residue_labels: vec!["ALA".to_string(), "WAT".to_string()],
            residue_pointers: vec![0, 3],
            lj_sigma: std::sync::Arc::new(vec![0.3, 0.25]),
            lj_epsilon: std::sync::Arc::new(vec![0.5, 0.3]),
            atom_sigmas: vec![0.3, 0.25, 0.3, 0.25, 0.3, 0.3],
            atom_epsilons: vec![0.5, 0.3, 0.5, 0.3, 0.5, 0.5],
            bonds: vec![],
            bond_types: vec![],
            masses: vec![14.0, 12.0, 12.0, 16.0, 1.008, 1.008],
            radii: vec![1.5, 1.7, 1.7, 1.5, 1.2, 1.2],
            screen: vec![0.0; 6],
            bond_force_constants: std::sync::Arc::new(vec![]),
            bond_equil_values: std::sync::Arc::new(vec![]),
            angle_force_constants: std::sync::Arc::new(vec![]),
            angle_equil_values: std::sync::Arc::new(vec![]),
            dihedral_force_constants: std::sync::Arc::new(vec![]),
            dihedral_periodicities: std::sync::Arc::new(vec![]),
            dihedral_phases: std::sync::Arc::new(vec![]),
            angles: vec![],
            dihedrals: vec![],
            num_excluded_atoms: vec![0; 6],
            excluded_atoms_list: vec![],
            scee_scale_factor: 1.2,
            scnb_scale_factor: 2.0,
            lj_acoef: std::sync::Arc::new(vec![]),
            lj_bcoef: std::sync::Arc::new(vec![]),
            nb_parm_index: std::sync::Arc::new(vec![]),
        }
    }

    #[test]
    fn test_select_name() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("name CA").unwrap();
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_select_name_glob() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("name H*").unwrap();
        assert_eq!(result, vec![4, 5]);
    }

    #[test]
    fn test_select_resname() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("resname ALA").unwrap();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_select_resid() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        // resid is 1-based, so resid 1 = residue index 0 = ALA
        let result = ctx.eval_str("resid 1").unwrap();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_select_resid_range() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("resid 1-2").unwrap();
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_select_index() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("index 0").unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_select_mass() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("mass > 13.0").unwrap();
        assert_eq!(result, vec![0, 3]); // N=14, O=16
    }

    #[test]
    fn test_select_and() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("resname ALA and name CA").unwrap();
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_select_or() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("name N or name O").unwrap();
        assert_eq!(result, vec![0, 3]);
    }

    #[test]
    fn test_select_not() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("not resname WAT").unwrap();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_select_protein() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("protein").unwrap();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_select_water() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("water").unwrap();
        assert_eq!(result, vec![3, 4, 5]);
    }

    #[test]
    fn test_select_hydrogen() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("hydrogen").unwrap();
        assert_eq!(result, vec![4, 5]);
    }

    #[test]
    fn test_select_backbone() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("backbone").unwrap();
        assert_eq!(result, vec![0, 1, 2]); // N, CA, C are all backbone in ALA
    }

    #[test]
    fn test_select_sidechain() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("sidechain").unwrap();
        // ALA atoms: N, CA, C — all backbone, so sidechain is empty for this mini topology
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_select_all() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("all").unwrap();
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_select_none() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("none").unwrap();
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_select_negative_charge() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("charge < -0.1").unwrap();
        assert_eq!(result, vec![1, 3]); // charges -0.2 and -0.4
    }

    #[test]
    fn test_select_within() {
        let top = make_test_topology();
        let coords: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ];
        let ctx = SelectionContext::with_coordinates(&top, &coords);
        // Select atoms within 1.5 of resname WAT (atoms 3,4,5 at x=10,10.5,11)
        let result = ctx.eval_str("within 1.5 of resname ALA").unwrap();
        // ALA atoms at 0,1,2. Within 1.5: atoms 0,1,2 (self) — all within 1.5 of each other
        assert_eq!(result, vec![0, 1, 2]);

        // Now test from WAT side
        let result = ctx.eval_str("within 1.5 of resname WAT").unwrap();
        assert_eq!(result, vec![3, 4, 5]);
    }

    #[test]
    fn test_within_requires_coordinates() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("within 5.0 of resname ALA");
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_expression() {
        let top = make_test_topology();
        let ctx = SelectionContext::new(&top);
        let result = ctx.eval_str("(protein or water) and not hydrogen").unwrap();
        assert_eq!(result, vec![0, 1, 2, 3]);
    }
}
