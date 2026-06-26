"""
Example: Parameterize a small-molecule ligand using the built-in antechamber.

This example demonstrates two workflows:
1. Standalone antechamber: parameterize_ligand() to assign GAFF2 types + AM1-BCC charges
2. SystemBuilder integration: load_ligand() for a complete simulation-ready system
"""

import rust_simulation_tools as rst

# ===========================================================================
# Workflow 1: Standalone parameterization
# ===========================================================================
# Reads an SDF file, assigns GAFF2 atom types and AM1-BCC charges,
# writes a parameterized mol2 file.

print("=== Standalone Antechamber ===")
rst.parameterize_ligand(
    input_path="ligand.sdf",
    output_path="ligand_gaff2.mol2",
    net_charge=0,
    charge_method="am1bcc",  # or "gasteiger" for faster but less accurate
)
print("Wrote ligand_gaff2.mol2 with GAFF2 types and AM1-BCC charges")

# ===========================================================================
# Workflow 2: SystemBuilder with automatic ligand parameterization
# ===========================================================================
# The load_ligand() method runs antechamber internally, producing a System
# with GAFF2 types and AM1-BCC charges, ready for solvation and simulation.

print("\n=== SystemBuilder Integration ===")
builder = rst.SystemBuilder()
builder.load_protein_ff19sb()
builder.load_gaff2()
builder.load_water_opc()

# Load protein
protein = builder.load_pdb("protein.pdb")

# Load and parameterize ligand in one step
ligand = builder.load_ligand("ligand.sdf", net_charge=0)

# Combine, solvate, ionize
system = builder.combine([protein, ligand])
builder.solvate_box(system, buffer=12.0)
builder.add_ions(system, "Na+", count="neutralize")
builder.add_ions(system, "Cl-", count="neutralize")

# Write output
builder.write_amber(system, "complex.prmtop", "complex.inpcrd")
builder.write_pdb(system, "complex.pdb")
print("Wrote complex.prmtop, complex.inpcrd, complex.pdb")

# ===========================================================================
# Workflow 3: Compute AM1 charges directly
# ===========================================================================
# For advanced use, compute AM1 Mulliken charges directly.

import numpy as np

print("\n=== Direct AM1 Charges ===")
# Water molecule
atomic_numbers = np.array([8, 1, 1], dtype=np.int64)
coords = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.757, 0.587],
    [0.0, -0.757, 0.587],
])
charges = rst.compute_am1_charges(atomic_numbers, coords, charge=0)
print(f"Water AM1 charges: O={charges[0]:.4f}, H1={charges[1]:.4f}, H2={charges[2]:.4f}")
