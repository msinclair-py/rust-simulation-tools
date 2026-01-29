"""Accessing atom properties from selections."""

import numpy as np
from rust_simulation_tools import read_prmtop, read_inpcrd

# Load topology and coordinates
topo = read_prmtop("system.prmtop")
coords, box_dims = read_inpcrd("system.inpcrd")

# --- Topology-level arrays ---
# These return data for ALL atoms; use selection indices to slice.

all_names = topo.atom_names          # list[str], length n_atoms
all_charges = topo.charges()         # np.ndarray (float64), partial charges (e)
all_sigmas = topo.sigmas()           # np.ndarray (float64), LJ sigma (nm)
all_epsilons = topo.epsilons()       # np.ndarray (float64), LJ epsilon (kJ/mol)
all_res_idx = topo.atom_residue_indices()  # np.ndarray (int64), residue index per atom
res_labels = topo.residue_labels     # list[str], length n_residues

print(f"System: {topo.n_atoms} atoms, {topo.n_residues} residues")

# --- Slice properties by selection ---

sel = topo.select_atoms("protein and name CA")  # returns list[int]
indices = np.array(sel)

print(f"\n--- Protein CA atoms ({len(sel)} atoms) ---")
print(f"Indices (first 5):   {indices[:5]}")
print(f"Names (first 5):     {[all_names[i] for i in sel[:5]]}")
print(f"Charges (first 5):   {all_charges[indices[:5]]}")
print(f"Residue ids:         {all_res_idx[indices[:5]]}")
print(f"Residue names:       {[res_labels[r] for r in all_res_idx[indices[:5]]]}")

# --- Coordinates for a selection ---

ca_coords = coords[indices]  # shape (n_selected, 3)
print(f"\nCA coordinates shape: {ca_coords.shape}")
print(f"Center of mass (CA):  {ca_coords.mean(axis=0)}")

# --- Bond information ---

all_bonds = topo.bonds()  # list of (atom_i, atom_j) tuples
print(f"\nTotal bonds: {len(all_bonds)}")

# Bonds for a specific residue
res_bonds = topo.get_bonds_for_residue(0)
print(f"Bonds in residue 0:  {res_bonds}")

# Bonds for multiple residues
res_bonds_multi = topo.get_bonds_for_residues([0, 1, 2])
print(f"Bonds in residues 0-2: {len(res_bonds_multi)} bonds")

# --- Residue mapping ---

# Residue pointers: first atom index of each residue
res_ptrs = topo.residue_pointers()  # np.ndarray (int64), length n_residues
print(f"\nResidue pointers (first 5): {res_ptrs[:5]}")
print(f"Atoms in residue 0: indices {res_ptrs[0]} to {res_ptrs[1] - 1}")

# Build a detailed selection from residue indices
detail = topo.build_selection([0, 1, 2])
print(f"\nbuild_selection for residues 0-2:")
print(f"  atom_indices:    {detail['atom_indices'][:10]}...")
print(f"  residue_offsets: {detail['residue_offsets']}")
print(f"  residue_labels:  {detail['residue_labels']}")

# Get atom indices for specific residues
atom_idx = topo.get_atom_indices([0, 1])
print(f"\nAtom indices for residues 0,1: {atom_idx}")

# --- Combining properties for analysis ---

# Example: find the most negatively charged sidechain atom
sc = topo.select_atoms("protein and sidechain")
sc_idx = np.array(sc)
sc_charges = all_charges[sc_idx]
most_neg = sc_idx[np.argmin(sc_charges)]
print(f"\nMost negative sidechain atom:")
print(f"  Index:   {most_neg}")
print(f"  Name:    {all_names[most_neg]}")
print(f"  Charge:  {all_charges[most_neg]:.4f} e")
print(f"  Residue: {res_labels[all_res_idx[most_neg]]} {all_res_idx[most_neg]}")
