"""Atom selection and property access using the object-oriented API.

The `select` method returns a Selection object with properties for the
selected atoms - no manual array slicing required.
"""

import numpy as np
from rust_simulation_tools import read_prmtop, read_inpcrd

# -----------------------------------------------------------------------------
# Load topology and coordinates
# -----------------------------------------------------------------------------

topo = read_prmtop("system.prmtop")
coords, box = read_inpcrd("system.inpcrd")

print(f"System: {topo.n_atoms} atoms, {topo.n_residues} residues")

# -----------------------------------------------------------------------------
# Selection objects
# -----------------------------------------------------------------------------

# select() returns a Selection object with properties for the selected atoms
protein = topo.select("protein")
print(f"\nProtein selection: {protein}")  # <Selection: 1234 atoms, 100 residues>
print(f"  Atoms: {protein.n_atoms}")
print(f"  Residues: {protein.n_residues}")

# Access properties directly on the selection - no slicing needed
ca = topo.select("protein and name CA")
print(f"\nCA atoms: {ca.n_atoms}")
print(f"  Charges: {ca.charges[:5]}")           # first 5 charges
print(f"  Masses: {ca.masses[:5]}")             # first 5 masses
print(f"  Atom names: {list(ca.atom_names)[:5]}")
print(f"  Residue names: {list(ca.residue_names)[:5]}")

# Aggregate properties
print(f"  Total mass: {ca.total_mass():.2f}")
print(f"  Total charge: {ca.total_charge():.4f}")

# -----------------------------------------------------------------------------
# Selections with coordinates
# -----------------------------------------------------------------------------

# Pass coordinates to get positions on the selection
backbone = topo.select("backbone", coordinates=coords)
print(f"\nBackbone with coordinates:")
print(f"  Has positions: {backbone.has_positions}")
print(f"  Positions shape: {backbone.positions.shape}")
print(f"  Center of geometry: {backbone.positions.mean(axis=0)}")

# Distance-based selections require coordinates
near_ligand = topo.select("protein and within 5.0 of resname LIG", coordinates=coords)
print(f"\nProtein atoms within 5A of ligand: {near_ligand.n_atoms}")
print(f"  Unique residues: {list(near_ligand.unique_residue_names)}")

# -----------------------------------------------------------------------------
# Set operations on selections
# -----------------------------------------------------------------------------

sidechain = topo.select("sidechain")
charged = topo.select("charge < -0.5 or charge > 0.5")

# Intersection: charged sidechain atoms
charged_sidechain = sidechain & charged
print(f"\nCharged sidechain atoms: {charged_sidechain.n_atoms}")

# Union: backbone or charged
backbone_or_charged = topo.select("backbone") | charged
print(f"Backbone or charged: {backbone_or_charged.n_atoms}")

# Difference: protein minus backbone = sidechain
protein_minus_backbone = topo.select("protein") - topo.select("backbone")
print(f"Protein - backbone: {protein_minus_backbone.n_atoms}")

# -----------------------------------------------------------------------------
# Common selection patterns
# -----------------------------------------------------------------------------

# Residue range
binding_site = topo.select("resid 45-60 and sidechain", coordinates=coords)
print(f"\nBinding site (resid 45-60 sidechain): {binding_site.n_atoms} atoms")
print(f"  Residues: {list(binding_site.unique_residue_names)}")

# Find most charged atom in binding site
charges = binding_site.charges
most_neg_idx = np.argmin(charges)
print(f"\n  Most negative atom:")
print(f"    Name: {list(binding_site.atom_names)[most_neg_idx]}")
print(f"    Residue: {list(binding_site.residue_names)[most_neg_idx]}")
print(f"    Charge: {charges[most_neg_idx]:.3f} e")
