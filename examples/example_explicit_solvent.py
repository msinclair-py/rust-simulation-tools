"""Build a protein system in explicit OPC water with counterions.

Demonstrates the full explicit-solvent system-building workflow:
1. Create a SystemBuilder and load force fields (protein + water).
2. Load a PDB file.
3. Solvate in a rectangular box of OPC water with a 12 A buffer.
4. Neutralize the system charge with Na+ or Cl- counterions.
5. Add excess salt — either by ion count or by molar concentration.
6. Write prmtop, inpcrd, and PDB output files.
"""

from rust_simulation_tools import SystemBuilder

# =============================================================================
# Set up the builder and load force fields
# =============================================================================

builder = SystemBuilder()
builder.load_protein_ff19sb()
builder.load_water_opc()

# =============================================================================
# Load the protein structure
# =============================================================================

system = builder.load_pdb("protein.pdb")

print(f"Loaded protein:")
print(f"  Atoms:    {system.n_atoms}")
print(f"  Residues: {system.n_residues}")
print(f"  Charge:   {system.total_charge:.2f} e")

# =============================================================================
# Solvate in a rectangular box of OPC water
# =============================================================================

builder.solvate_box(system, buffer=12.0, closeness=1.0)

print(f"\nAfter solvation:")
print(f"  Atoms:    {system.n_atoms}")
print(f"  Residues: {system.n_residues}")
print(f"  Box:      {system.box_dimensions}")

# =============================================================================
# Method A: Manual ion counts (neutralize + explicit excess)
# =============================================================================

print("\n--- Method A: Manual ion counts ---")

# Neutralize first — adds Na+ or Cl- depending on the net charge
n_neutralize = builder.add_ions(system, "Na+", count="neutralize")
print(f"Added {n_neutralize} Na+ to neutralize system charge")

# Add excess NaCl (e.g. 20 of each for a rough salt concentration)
n_na = builder.add_ions(system, "Na+", count=20)
n_cl = builder.add_ions(system, "Cl-", count=20)
print(f"Added {n_na} Na+ and {n_cl} Cl- for excess salt")

# =============================================================================
# Method B: Concentration-based salt addition (one-liner)
# =============================================================================

# Alternatively, use add_salt() which neutralizes the system AND adds NaCl
# at the requested molar concentration in a single call:
#
#   n_cation, n_anion = builder.add_salt(system, concentration=0.150)
#   print(f"Added {n_cation} Na+ and {n_anion} Cl- for 150 mM NaCl")
#
# You can also pass the concentration directly to add_ions() as a float:
#
#   builder.add_ions(system, "Na+", count="neutralize")
#   builder.add_ions(system, "Na+", count=0.150)  # 150 mM
#   builder.add_ions(system, "Cl-", count=0.150)   # 150 mM

# =============================================================================
# Final system summary
# =============================================================================

print(f"\nFinal system:")
print(f"  Atoms:    {system.n_atoms}")
print(f"  Residues: {system.n_residues}")
print(f"  Charge:   {system.total_charge:.2f} e")
print(f"  Box:      {system.box_dimensions}")

# =============================================================================
# Write output files
# =============================================================================

builder.write_amber(system, "protein_explicit.prmtop", "protein_explicit.inpcrd")
builder.write_pdb(system, "protein_explicit.pdb")

print("\nWrote: protein_explicit.prmtop, protein_explicit.inpcrd, protein_explicit.pdb")
