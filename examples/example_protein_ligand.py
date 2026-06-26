"""Build a protein-ligand complex in explicit solvent using a custom mol2 ligand.

Demonstrates loading a pre-parameterized ligand (from antechamber) alongside
a protein, combining them, solvating, and writing output files.

Prerequisites:
  The ligand mol2 file should already have GAFF2 atom types and AM1-BCC
  charges assigned by antechamber. If the ligand requires custom parameters
  (e.g. from parmchk2), load those via load_custom_frcmod().

  Example antechamber/parmchk2 commands:
    antechamber -i ligand.pdb -fi pdb -o ligand.mol2 -fo mol2 \
                -c bcc -at gaff2 -nc 0
    parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s gaff2
"""

from rust_simulation_tools import SystemBuilder

# -----------------------------------------------------------------------------
# Set up the builder and load force fields
# -----------------------------------------------------------------------------

builder = SystemBuilder()
builder.load_protein_ff19sb()
builder.load_gaff2()
builder.load_water_opc()

# Load custom ligand parameters (from parmchk2)
builder.load_custom_frcmod("ligand.frcmod")

# If the ligand uses a custom residue library, load it too:
# builder.load_custom_lib("ligand.lib")

# -----------------------------------------------------------------------------
# Load structures
# -----------------------------------------------------------------------------

protein = builder.load_pdb("protein.pdb")
print(f"Protein: {protein.n_atoms} atoms, {protein.n_residues} residues")

ligand = builder.load_mol2("ligand.mol2")
print(f"Ligand:  {ligand.n_atoms} atoms, {ligand.n_residues} residues")

# -----------------------------------------------------------------------------
# Combine protein and ligand into a single system
# -----------------------------------------------------------------------------

system = builder.combine([protein, ligand])
print(f"\nCombined: {system.n_atoms} atoms, {system.n_residues} residues")
print(f"  Charge: {system.total_charge:.2f} e")

# -----------------------------------------------------------------------------
# Solvate and add ions
# -----------------------------------------------------------------------------

builder.solvate_box(system, buffer=12.0)
print(f"\nAfter solvation: {system.n_atoms} atoms")

builder.add_ions(system, "Na+", count="neutralize")
builder.add_ions(system, "Na+", count=15)
builder.add_ions(system, "Cl-", count=15)

print(f"After ions:      {system.n_atoms} atoms, charge={system.total_charge:.2f} e")

# -----------------------------------------------------------------------------
# Write output files
# -----------------------------------------------------------------------------

builder.write_amber(system, "complex.prmtop", "complex.inpcrd")
builder.write_pdb(system, "complex.pdb")

print("\nWrote: complex.prmtop, complex.inpcrd, complex.pdb")
