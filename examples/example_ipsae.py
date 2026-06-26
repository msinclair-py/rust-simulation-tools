"""ipSAE: Interface prediction Score from Aligned Errors.

Demonstrates:
- Computing pDockQ, pDockQ2, LIS, ipTM, and ipSAE from a structure file
- Computing scores from pre-parsed arrays
- Working with directed vs undirected (max) chain pair scores
"""

import numpy as np
from rust_simulation_tools import compute_ipsae, compute_ipsae_from_arrays

# =============================================================================
# Option 1: From a structure file (PDB or CIF)
# =============================================================================

structure_path = "model.pdb"  # or "model.cif"

# pLDDT: per-residue confidence (0-100 scale), length N
# If your pLDDT is 0-1 scale (e.g. from Boltz), multiply by 100 first.
data = np.load("confidences.npz")
plddt = data["plddt"] * 100.0  # shape (N,)

# PAE: predicted aligned error matrix, shape (N, N)
# Flatten to 1D row-major for the Rust interface.
pae = np.load("pae.npz")["pae"].flatten()  # shape (N*N,)

results = compute_ipsae(structure_path, plddt, pae)

# results is a dict with two keys:
#   "directed_pairs" - scores for all ordered (A->B, B->A) chain pairs
#   "max_pairs"      - one entry per unordered pair, keeping the direction
#                       with the highest ipSAE

print("=== Directed pairs ===")
for pair in results["directed_pairs"]:
    print(
        f"  {pair['chain1']} -> {pair['chain2']}: "
        f"pDockQ={pair['pDockQ']:.3f}  "
        f"pDockQ2={pair['pDockQ2']:.3f}  "
        f"LIS={pair['LIS']:.3f}  "
        f"ipTM={pair['ipTM']:.3f}  "
        f"ipSAE={pair['ipSAE']:.3f}"
    )

print("\n=== Max (undirected) pairs ===")
for pair in results["max_pairs"]:
    print(
        f"  {pair['chain1']} - {pair['chain2']}: "
        f"ipSAE={pair['ipSAE']:.3f}"
    )

# =============================================================================
# Option 2: From pre-parsed arrays (no file I/O)
# =============================================================================

# Useful when you already have coordinates and chain assignments in memory,
# e.g. from MDAnalysis, BioPython, or another parser.

coords = np.array([
    [0.0, 0.0, 0.0],
    [3.0, 0.0, 0.0],
    [6.0, 0.0, 0.0],
    [7.0, 0.0, 0.0],
    [8.0, 0.0, 0.0],
    [9.0, 0.0, 0.0],
])  # shape (N, 3), CA/C1' coordinates in Angstroms

chains = ["A", "A", "A", "B", "B", "B"]
chain_types = {"A": "protein", "B": "protein"}  # or "nucleic_acid"

plddt = np.array([90.0, 85.0, 80.0, 75.0, 70.0, 65.0])

n = len(plddt)
pae = np.full(n * n, 5.0)  # flat row-major PAE

results = compute_ipsae_from_arrays(
    coords, chains, chain_types, plddt, pae,
    pdockq_cutoff=8.0,  # distance cutoff for contacts (default 8 A)
    pae_cutoff=12.0,     # PAE cutoff for LIS/ipSAE (default 12 A)
)

for pair in results["max_pairs"]:
    print(
        f"{pair['chain1']}-{pair['chain2']}: "
        f"pDockQ={pair['pDockQ']:.3f}, ipSAE={pair['ipSAE']:.3f}"
    )
