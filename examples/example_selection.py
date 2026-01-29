"""Atom selection language example - VMD/MDAnalysis-style atom queries."""

import numpy as np
from rust_simulation_tools import read_prmtop, read_inpcrd

# Load topology
topo = read_prmtop("system.prmtop")
print(f"System: {topo.n_atoms} atoms, {topo.n_residues} residues")

# --- Basic selections (topology only) ---

# Select by atom name
ca_atoms = topo.select_atoms("name CA")
print(f"\nCA atoms: {len(ca_atoms)} atoms")

# Select by residue name
ala_atoms = topo.select_atoms("resname ALA")
print(f"ALA atoms: {len(ala_atoms)} atoms")

# Glob patterns (* and ? wildcards)
carbon_atoms = topo.select_atoms("name C*")
print(f"Atoms starting with C: {len(carbon_atoms)} atoms")

# --- Range selections ---

# resid is 1-based (AMBER/PDB convention)
first_10 = topo.select_atoms("resid 1-10")
print(f"\nResidues 1-10: {len(first_10)} atoms")

# Comma-separated list
specific = topo.select_atoms("resid 1,5,10")
print(f"Residues 1,5,10: {len(specific)} atoms")

# index is 0-based
first_atom = topo.select_atoms("index 0")
print(f"Index 0: {len(first_atom)} atom(s)")

# --- Numeric comparisons ---

charges = np.array(topo.charges())
negative = topo.select_atoms("charge < -0.5")
print(f"\nHighly negative atoms (charge < -0.5): {len(negative)}")

heavy = topo.select_atoms("mass > 32.0")
print(f"Heavy atoms (mass > 32): {len(heavy)}")

# --- Convenience keywords ---

protein = topo.select_atoms("protein")
water = topo.select_atoms("water")
backbone = topo.select_atoms("backbone")
sidechain = topo.select_atoms("sidechain")
hydrogens = topo.select_atoms("hydrogen")
everything = topo.select_atoms("all")
nothing = topo.select_atoms("none")

print(f"\nProtein: {len(protein)} atoms")
print(f"Water: {len(water)} atoms")
print(f"Backbone: {len(backbone)} atoms")
print(f"Sidechain: {len(sidechain)} atoms")
print(f"Hydrogen: {len(hydrogens)} atoms")
print(f"All: {len(everything)} atoms")
print(f"None: {len(nothing)} atoms")

# --- Boolean operators ---

# AND
protein_ca = topo.select_atoms("protein and name CA")
print(f"\nProtein CA atoms: {len(protein_ca)}")

# OR
n_or_o = topo.select_atoms("name N or name O")
print(f"N or O atoms: {len(n_or_o)}")

# NOT
not_water = topo.select_atoms("not water")
print(f"Non-water atoms: {len(not_water)}")

# Parentheses for grouping
charged = topo.select_atoms("(charge < -0.5 or charge > 0.5) and protein")
print(f"Charged protein atoms: {len(charged)}")

# Complex expressions
active_site = topo.select_atoms("resid 45-60 and sidechain")
print(f"Active site sidechain: {len(active_site)} atoms")

# --- Distance-based selection (requires coordinates) ---

coords, box_dims = read_inpcrd("system.inpcrd")
print(f"\nCoordinates shape: {coords.shape}")

# Select atoms within 5 Angstroms of a ligand
near_lig = topo.select_atoms("within 5.0 of resname LIG", coordinates=coords)
print(f"Atoms within 5A of LIG: {len(near_lig)}")

# Combine within with other selections
protein_near_lig = topo.select_atoms(
    "protein and within 5.0 of resname LIG",
    coordinates=coords,
)
print(f"Protein atoms within 5A of LIG: {len(protein_near_lig)}")
