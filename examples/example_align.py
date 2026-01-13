"""Kabsch alignment example - align trajectory frames to a reference structure."""

import MDAnalysis as mda
import numpy as np
from rust_simulation_tools import kabsch_align

# Load trajectory
u = mda.Universe("topology.pdb", "trajectory.dcd")

# Select atoms for alignment (e.g., backbone)
align_selection = u.select_atoms("backbone")
align_indices = align_selection.indices.astype(np.uintp)

# Use first frame as reference
reference = u.atoms.positions.copy()

# Extract full trajectory (preserves native float32)
trajectory = np.array([u.atoms.positions.copy() for _ in u.trajectory])

# Align trajectory to reference
aligned = kabsch_align(trajectory, reference, align_indices)

# Write aligned trajectory
with mda.Writer("aligned_trajectory.dcd", len(u.atoms)) as W:
    for i, ts in enumerate(u.trajectory):
        u.atoms.positions = aligned[i]
        W.write(u.atoms)

print(f"Aligned {len(trajectory)} frames using {len(align_indices)} backbone atoms")

# Verify alignment improved RMSD
if len(trajectory) > 1:
    ref_coords = reference[align_indices]
    orig_rmsd = np.sqrt(np.mean((trajectory[1][align_indices] - ref_coords) ** 2))
    aligned_rmsd = np.sqrt(np.mean((aligned[1][align_indices] - ref_coords) ** 2))
    print(f"Frame 1 RMSD: {orig_rmsd:.3f} -> {aligned_rmsd:.3f} Å")
