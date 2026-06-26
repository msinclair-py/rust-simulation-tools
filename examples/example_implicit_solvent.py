"""Build a protein system in implicit solvent (no water, no ions).

Demonstrates the simplest system-building workflow:
1. Create a SystemBuilder and load the protein force field.
2. Load a PDB file into a System.
3. Write prmtop and inpcrd files suitable for implicit solvent (GB) simulations.

No solvation or ion placement is performed - the resulting topology is
appropriate for Generalized Born (GB) calculations or vacuum energy evaluations.
"""

from rust_simulation_tools import SystemBuilder

# -----------------------------------------------------------------------------
# Set up the builder and load force fields
# -----------------------------------------------------------------------------

builder = SystemBuilder()
builder.load_protein_ff19sb()

# -----------------------------------------------------------------------------
# Load the protein structure
# -----------------------------------------------------------------------------

system = builder.load_pdb("protein.pdb")

print(f"Loaded protein:")
print(f"  Atoms:    {system.n_atoms}")
print(f"  Residues: {system.n_residues}")
print(f"  Charge:   {system.total_charge:.2f} e")
print(f"  Box:      {system.box_dimensions}")  # None for implicit solvent

# -----------------------------------------------------------------------------
# Write output files
# -----------------------------------------------------------------------------

builder.write_prmtop(system, "protein_implicit.prmtop")
builder.write_inpcrd(system, "protein_implicit.inpcrd")
builder.write_pdb(system, "protein_implicit.pdb")

print("\nWrote: protein_implicit.prmtop, protein_implicit.inpcrd, protein_implicit.pdb")
