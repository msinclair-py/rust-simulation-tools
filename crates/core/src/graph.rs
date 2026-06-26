//! Molecular graph with ring perception.
//!
//! Provides a generic adjacency-list representation of molecular topology
//! with ring detection (SSSR + relevant rings) and planarity checking.

use nalgebra::DMatrix;

// ============================================================================
// Data Structures
// ============================================================================

/// A ring: ordered list of atom indices forming a cycle.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ring {
    /// Atom indices in traversal order.
    pub atoms: Vec<usize>,
}

/// Molecular graph with adjacency list representation.
#[derive(Debug, Clone)]
pub struct MolGraph {
    /// Number of atoms.
    pub n_atoms: usize,
    /// Adjacency list: for each atom, the list of bonded atom indices.
    pub neighbors: Vec<Vec<usize>>,
    /// Bond pairs (atom1, atom2) with atom1 < atom2.
    pub bonds: Vec<(usize, usize)>,
}

// ============================================================================
// Construction
// ============================================================================

impl MolGraph {
    /// Create a new molecular graph from atom count and bond pairs.
    ///
    /// Bond pairs should be 0-based atom indices. Duplicate/self bonds are ignored.
    pub fn new(n_atoms: usize, bonds: &[(usize, usize)]) -> Self {
        let mut neighbors = vec![Vec::new(); n_atoms];
        let mut bond_set = Vec::new();

        for &(a, b) in bonds {
            if a == b || a >= n_atoms || b >= n_atoms {
                continue;
            }
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if !neighbors[lo].contains(&hi) {
                neighbors[lo].push(hi);
                neighbors[hi].push(lo);
                bond_set.push((lo, hi));
            }
        }

        Self {
            n_atoms,
            neighbors,
            bonds: bond_set,
        }
    }

    /// Build from a mol2 molecule.
    pub fn from_mol2(mol: &crate::mol2::Mol2Molecule) -> Self {
        let bonds: Vec<(usize, usize)> = mol.bonds.iter().map(|b| (b.atom1, b.atom2)).collect();
        Self::new(mol.atoms.len(), &bonds)
    }

    /// Build from an SDF molecule.
    pub fn from_sdf(mol: &crate::sdf::SdfMolecule) -> Self {
        let bonds: Vec<(usize, usize)> = mol.bonds.iter().map(|b| (b.atom1, b.atom2)).collect();
        Self::new(mol.atoms.len(), &bonds)
    }

    /// Find all relevant rings up to `max_size` atoms.
    ///
    /// Uses a BFS-based approach: for each edge not in the spanning tree,
    /// find the fundamental cycle, then combines to find relevant rings.
    /// Returns rings sorted by size.
    pub fn find_rings(&self, max_size: usize) -> Vec<Ring> {
        if self.n_atoms == 0 || self.bonds.is_empty() {
            return Vec::new();
        }

        // Find all small rings using DFS from each atom
        let mut all_rings = Vec::new();
        let mut ring_set: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();

        for start in 0..self.n_atoms {
            self.find_rings_from(start, max_size, &mut all_rings, &mut ring_set);
        }

        // Sort by ring size
        all_rings.sort_by_key(|r| r.atoms.len());
        all_rings
    }

    /// Find rings starting from a given atom using BFS-based cycle detection.
    fn find_rings_from(
        &self,
        start: usize,
        max_size: usize,
        rings: &mut Vec<Ring>,
        ring_set: &mut std::collections::HashSet<Vec<usize>>,
    ) {
        // Use DFS to find cycles. max_depth = max_size - 1 because
        // a ring of size N needs N-1 edges from start before closing back.
        let max_depth = max_size;
        let mut path = vec![start];
        let mut visited = vec![false; self.n_atoms];
        visited[start] = true;

        self.dfs_ring_search(start, &mut path, &mut visited, max_size, max_depth, 0, rings, ring_set);
    }

    fn dfs_ring_search(
        &self,
        start: usize,
        path: &mut Vec<usize>,
        visited: &mut Vec<bool>,
        max_size: usize,
        max_depth: usize,
        depth: usize,
        rings: &mut Vec<Ring>,
        ring_set: &mut std::collections::HashSet<Vec<usize>>,
    ) {
        if depth >= max_depth {
            return;
        }

        let current = *path.last().unwrap();

        for &next in &self.neighbors[current] {
            // Found a ring back to start
            if next == start && path.len() >= 3 {
                let ring_atoms = path.clone();
                if ring_atoms.len() <= max_size {
                    let canonical = Self::canonicalize_ring(&ring_atoms);
                    if ring_set.insert(canonical) {
                        rings.push(Ring { atoms: ring_atoms });
                    }
                }
                continue;
            }

            // Skip if already in current path (except start) or visited
            if visited[next] {
                continue;
            }

            // Only visit atoms > start to avoid duplicate rings
            if next < start {
                continue;
            }

            visited[next] = true;
            path.push(next);
            self.dfs_ring_search(start, path, visited, max_size, max_depth, depth + 1, rings, ring_set);
            path.pop();
            visited[next] = false;
        }
    }

    /// Canonicalize a ring by rotating to start with the smallest index
    /// and choosing the lexicographically smallest direction.
    fn canonicalize_ring(atoms: &[usize]) -> Vec<usize> {
        let n = atoms.len();
        if n == 0 {
            return Vec::new();
        }

        // Find position of minimum element
        let min_pos = atoms
            .iter()
            .enumerate()
            .min_by_key(|(_, &v)| v)
            .unwrap()
            .0;

        // Try both directions from min_pos
        let forward: Vec<usize> = (0..n).map(|i| atoms[(min_pos + i) % n]).collect();
        let reverse: Vec<usize> = (0..n)
            .map(|i| atoms[(min_pos + n - i) % n])
            .collect();

        if forward <= reverse {
            forward
        } else {
            reverse
        }
    }
}

// ============================================================================
// Ring utilities
// ============================================================================

impl Ring {
    /// Check if this ring is approximately planar.
    ///
    /// Uses SVD to find the smallest singular value of the centered
    /// coordinate matrix. The ring is planar if the smallest singular
    /// value is below the tolerance (in Angstroms).
    pub fn is_planar(&self, coords: &[[f64; 3]], tolerance: f64) -> bool {
        let n = self.atoms.len();
        if n < 3 {
            return true;
        }

        // Compute centroid
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &idx in &self.atoms {
            cx += coords[idx][0];
            cy += coords[idx][1];
            cz += coords[idx][2];
        }
        cx /= n as f64;
        cy /= n as f64;
        cz /= n as f64;

        // Build centered coordinate matrix (n x 3)
        let mat = DMatrix::from_fn(n, 3, |i, j| {
            let idx = self.atoms[i];
            match j {
                0 => coords[idx][0] - cx,
                1 => coords[idx][1] - cy,
                2 => coords[idx][2] - cz,
                _ => unreachable!(),
            }
        });

        // SVD: the smallest singular value tells us the "thickness" of the point cloud
        let svd = mat.svd(false, false);
        let min_sv = svd.singular_values.iter().copied().fold(f64::INFINITY, f64::min);

        min_sv < tolerance
    }

    /// Check if this ring contains the given atom index.
    pub fn contains(&self, atom: usize) -> bool {
        self.atoms.contains(&atom)
    }

    /// Get the size of this ring.
    pub fn size(&self) -> usize {
        self.atoms.len()
    }

    /// Check whether two rings share at least one atom (fused rings).
    pub fn is_fused_with(&self, other: &Ring) -> bool {
        self.atoms.iter().any(|a| other.atoms.contains(a))
    }

    /// Get the shared atoms between two rings.
    pub fn shared_atoms(&self, other: &Ring) -> Vec<usize> {
        self.atoms
            .iter()
            .filter(|a| other.atoms.contains(a))
            .copied()
            .collect()
    }
}
