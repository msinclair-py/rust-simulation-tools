"""Trajectory unwrapping example - remove periodic boundary artifacts."""

import MDAnalysis as mda
import numpy as np
from rust_simulation_tools import unwrap_system

# Load trajectory
u = mda.Universe("topology.pdb", "trajectory.dcd")

# Build fragment indices from MDAnalysis molecule detection
fragment_indices = np.zeros(len(u.atoms), dtype=np.int64)
for frag_id, fragment in enumerate(u.atoms.fragments):
    fragment_indices[fragment.indices] = frag_id

# Extract trajectory and box dimensions
trajectory = np.array([u.atoms.positions.copy() for _ in u.trajectory])
box_dims = np.array([ts.dimensions[:3] for ts in u.trajectory], dtype=trajectory.dtype)

# Unwrap trajectory
unwrapped = unwrap_system(trajectory, box_dims, fragment_indices)

# Write unwrapped trajectory
with mda.Writer("unwrapped_trajectory.dcd", len(u.atoms)) as W:
    for i, ts in enumerate(u.trajectory):
        u.atoms.positions = unwrapped[i]
        W.write(u.atoms)

print(f"Unwrapped {len(trajectory)} frames with {len(u.atoms.fragments)} fragments")

# Verify: check molecule size before/after (should be stable after unwrapping)
mol_indices = u.atoms.fragments[0].indices
for i in [0, len(trajectory) - 1]:
    wrapped_size = np.ptp(trajectory[i][mol_indices], axis=0).max()
    unwrapped_size = np.ptp(unwrapped[i][mol_indices], axis=0).max()
    print(f"Frame {i}: molecule extent {wrapped_size:.1f} -> {unwrapped_size:.1f} Å")
