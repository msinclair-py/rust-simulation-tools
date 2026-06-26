"""Basic trajectory analysis: SASA, unwrapping, and alignment.

Demonstrates common trajectory operations using rust_simulation_tools only.
"""

import numpy as np
from rust_simulation_tools import (
    read_prmtop,
    DcdReader,
    compute_sasa_from_topology,
    compute_sasa_trajectory_from_topology,
    unwrap_dcd,
    kabsch_align,
)

# -----------------------------------------------------------------------------
# Load topology and trajectory
# -----------------------------------------------------------------------------

topo = read_prmtop("system.prmtop")
reader = DcdReader("trajectory.dcd")

print(f"System: {topo.n_atoms} atoms, {topo.n_residues} residues")
print(f"Trajectory: {reader.n_frames} frames")

# Read all frames into memory
trajectory, boxes = reader.read_all()
trajectory = trajectory.reshape(reader.n_frames, reader.n_atoms, 3)

print(f"Trajectory shape: {trajectory.shape}")
print(f"Box dimensions shape: {boxes.shape}")

# -----------------------------------------------------------------------------
# SASA calculation
# -----------------------------------------------------------------------------

# Single frame - topology provides radii and residue mapping
sasa = compute_sasa_from_topology(topo, trajectory[0])

print(f"\nSASA (frame 0):")
print(f"  Total: {sasa['total']:.1f} A^2")
print(f"  Per-atom shape: {sasa['per_atom'].shape}")

# Top 5 most exposed residues
# per_residue is a numpy array ordered by residue index
per_residue = sasa['per_residue']
top_idx = np.argsort(per_residue)[::-1][:5]
print(f"  Most exposed residues:")
for res_idx in top_idx:
    res_label = topo.residue_labels[res_idx]
    print(f"    {res_label} {res_idx + 1}: {per_residue[res_idx]:.1f} A^2")

# Trajectory SASA
traj_sasa = compute_sasa_trajectory_from_topology(topo, trajectory)
print(f"\nTrajectory SASA:")
print(f"  Mean: {np.mean(traj_sasa['total']):.1f} A^2")
print(f"  Std:  {np.std(traj_sasa['total']):.1f} A^2")

# -----------------------------------------------------------------------------
# Trajectory unwrapping
# -----------------------------------------------------------------------------

# unwrap_dcd reads and unwraps in one step
unwrapped, boxes = unwrap_dcd("trajectory.dcd")

print(f"\nUnwrapping:")
print(f"  Unwrapped shape: {unwrapped.shape}")

# Verify: check atom displacement continuity
atom_0_displacement = np.linalg.norm(unwrapped[-1, 0] - unwrapped[0, 0])
print(f"  Atom 0 total displacement: {atom_0_displacement:.1f} A")

# -----------------------------------------------------------------------------
# Kabsch alignment
# -----------------------------------------------------------------------------

# Align to first frame using backbone atoms
backbone = topo.select("backbone")
align_indices = np.array(backbone.indices, dtype=np.uintp)
reference = unwrapped[0].copy()

aligned = kabsch_align(unwrapped, reference, align_indices)

print(f"\nAlignment:")
print(f"  Aligned on: {len(align_indices)} backbone atoms")

# RMSD before/after alignment
ref_coords = reference[align_indices]
orig_rmsd = np.sqrt(np.mean((unwrapped[-1][align_indices] - ref_coords) ** 2))
aligned_rmsd = np.sqrt(np.mean((aligned[-1][align_indices] - ref_coords) ** 2))
print(f"  Last frame backbone RMSD: {orig_rmsd:.2f} -> {aligned_rmsd:.2f} A")

# -----------------------------------------------------------------------------
# Selection-based analysis
# -----------------------------------------------------------------------------

# SASA for protein only (using selection for atom indices)
protein = topo.select("protein", coordinates=trajectory[0])
print(f"\nProtein selection: {protein.n_atoms} atoms")
print(f"  Center of geometry: {protein.positions.mean(axis=0)}")

# Per-residue SASA for binding site
binding_site = topo.select("resid 45-60")
print(f"\nBinding site (resid 45-60): {binding_site.n_atoms} atoms")
print(f"  Residues: {list(binding_site.unique_residue_names)}")
