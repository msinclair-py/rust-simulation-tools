"""AMBER file readers example - read topology and coordinate files."""

import numpy as np
from rust_simulation_tools import read_prmtop, read_inpcrd

# Read AMBER topology file (prmtop)
topo = read_prmtop("system.prmtop")

print(f"System: {topo.n_atoms} atoms, {topo.n_residues} residues")
print(f"First 5 atoms: {topo.atom_names[:5]}")
print(f"First 5 residues: {topo.residue_labels[:5]}")

# Get force field parameters
charges = np.array(topo.charges())       # Elementary charge units
sigmas = np.array(topo.sigmas())         # nm
epsilons = np.array(topo.epsilons())     # kJ/mol

print(f"\nCharge range: {charges.min():.3f} to {charges.max():.3f} e")
print(f"Sigma range: {sigmas.min():.3f} to {sigmas.max():.3f} nm")

# Get atom-to-residue mapping
atom_res_idx = np.array(topo.atom_residue_indices())
print(f"\nAtom 0 belongs to residue {atom_res_idx[0]} ({topo.residue_labels[0]})")

# Read initial coordinates (inpcrd/rst7)
positions, box = read_inpcrd("system.inpcrd")

print(f"\nCoordinates shape: {positions.shape}")  # (n_atoms, 3) in nm
if box is not None:
    print(f"Box dimensions: {box} nm")
