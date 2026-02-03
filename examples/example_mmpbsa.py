"""MM-PBSA/GBSA binding free energy calculations.

Demonstrates:
- Trajectory-averaged binding energy with GB solvation
- Trajectory-averaged binding energy with PB solvation
- Per-residue energy decomposition

NOTE: This example supports both solvated and dry (stripped) topologies. When
using solvated inputs, the receptor and ligand residue indices should correspond
to the ORIGINAL topology numbering - solvent atoms are automatically excluded
based on the receptor/ligand residue selection. Only atoms belonging to receptor
or ligand residues are included in the MM-PBSA calculation.
"""

import numpy as np
from rust_simulation_tools import (
    read_prmtop,
    read_inpcrd,
    GbModel,
    GbParams,
    PbParams,
    SaParams,
    compute_binding_energy,
    decompose_binding_energy,
)

# -----------------------------------------------------------------------------
# Load system
# -----------------------------------------------------------------------------

# Can be either a solvated or dry topology - solvent is automatically stripped
# based on the receptor/ligand residue selection. For solvated systems, the
# trajectory should match the topology (i.e., include solvent coordinates).
topo = read_prmtop("complex.prmtop")
coords, _ = read_inpcrd("complex.inpcrd")

print(f"System: {topo.n_atoms} atoms, {topo.n_residues} residues")

# Define receptor and ligand by residue indices (0-based, relative to topology)
# These indices refer to residue numbers in the FULL topology, including any
# solvent residues. Only atoms in these residues are used for MM-PBSA.
receptor_residues = list(range(0, 250))
ligand_residues = list(range(250, 251))

# Calculate the number of atoms that will be used in MM-PBSA
receptor_sel = topo.build_selection(receptor_residues)
ligand_sel = topo.build_selection(ligand_residues)
receptor_n_atoms = len(receptor_sel["atom_indices"])
ligand_n_atoms = len(ligand_sel["atom_indices"])
complex_atoms = receptor_n_atoms + ligand_n_atoms

print(f"Receptor: {len(receptor_residues)} residues ({receptor_n_atoms} atoms)")
print(f"Ligand: {len(ligand_residues)} residues ({ligand_n_atoms} atoms)")
if complex_atoms < topo.n_atoms:
    print(f"Complex: {complex_atoms} atoms ({topo.n_atoms - complex_atoms} solvent atoms excluded)")
else:
    print(f"Complex: {complex_atoms} atoms")

# -----------------------------------------------------------------------------
# MM-GBSA binding energy
# -----------------------------------------------------------------------------

print("\n=== MM-GBSA Binding Energy ===")

gb_result = compute_binding_energy(
    topo,
    trajectory_path="trajectory.dcd",
    receptor_residues=receptor_residues,
    ligand_residues=ligand_residues,
    gb_params=GbParams(model=GbModel.ObcII, salt_concentration=0.15),
    sa_params=SaParams(),
    trajectory_format="dcd",
)

print(f"\nFrames analyzed: {len(gb_result.frames)}")
print(f"\nMean Binding Energy (kcal/mol):")
print(f"  Delta MM:    {gb_result.mean_delta_mm:8.2f} +/- {gb_result.std_delta_mm:.2f}")
print(f"  Delta GB:    {gb_result.mean_delta_gb:8.2f} +/- {gb_result.std_delta_gb:.2f}")
print(f"  Delta SA:    {gb_result.mean_delta_sa:8.2f} +/- {gb_result.std_delta_sa:.2f}")
print(f"  Delta Total: {gb_result.mean_delta_total:8.2f} +/- {gb_result.std_delta_total:.2f}")
print(f"  SEM:         {gb_result.sem_delta_total:.2f}")

# -----------------------------------------------------------------------------
# MM-PBSA binding energy
# -----------------------------------------------------------------------------

print("\n=== MM-PBSA Binding Energy ===")

pb_params = PbParams(
    grid_spacing=0.5,
    solute_dielectric=1.0,
    solvent_dielectric=80.0,
    salt_concentration=0.15,
)

pb_result = compute_binding_energy(
    topo,
    trajectory_path="trajectory.dcd",
    receptor_residues=receptor_residues,
    ligand_residues=ligand_residues,
    pb_params=pb_params,
    sa_params=SaParams(),
    trajectory_format="dcd",
)

print(f"\nFrames analyzed: {len(pb_result.frames)}")
print(f"\nMean Binding Energy (kcal/mol):")
print(f"  Delta MM:    {pb_result.mean_delta_mm:8.2f} +/- {pb_result.std_delta_mm:.2f}")
print(f"  Delta PB:    {pb_result.mean_delta_polar:8.2f} +/- {pb_result.std_delta_polar:.2f}")
print(f"  Delta SA:    {pb_result.mean_delta_sa:8.2f} +/- {pb_result.std_delta_sa:.2f}")
print(f"  Delta Total: {pb_result.mean_delta_total:8.2f} +/- {pb_result.std_delta_total:.2f}")
print(f"  SEM:         {pb_result.sem_delta_total:.2f}")

# -----------------------------------------------------------------------------
# Per-residue decomposition
# -----------------------------------------------------------------------------

print("\n=== Per-Residue Decomposition ===")

decomp = decompose_binding_energy(
    topo, coords,
    receptor_residues=receptor_residues,
    ligand_residues=ligand_residues,
    gb_params=GbParams(model=GbModel.ObcII),
    sa_params=SaParams(),
)

# Top contributing receptor residues (most favorable)
receptor = sorted(decomp.receptor_residues, key=lambda r: r.total())
print(f"\nTop 10 receptor contributions (kcal/mol):")
print(f"  {'Residue':<12} {'vdW':>8} {'Elec':>8} {'GB':>8} {'SA':>8} {'Total':>8}")
for res in receptor[:10]:
    label = f"{res.residue_label}{res.residue_index}"
    print(f"  {label:<12} {res.vdw:8.2f} {res.elec:8.2f} {res.gb:8.2f} {res.sa:8.2f} {res.total():8.2f}")

# Ligand residue contributions
print(f"\nLigand contributions (kcal/mol):")
for res in decomp.ligand_residues:
    label = f"{res.residue_label}{res.residue_index}"
    print(f"  {label}: {res.total():.2f}")
