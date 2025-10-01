import MDAnalysis as mda
import numpy as np
from rust_simulation_tools import unwrap_system

# Load trajectory
u = mda.Universe("topology.pdb", "trajectory.dcd")

# Get fragment (molecule) assignments - no dtype conversion needed!
fragment_indices = np.zeros(len(u.atoms), dtype=np.int64)
for frag_id, fragment in enumerate(u.atoms.fragments):
    fragment_indices[fragment.indices] = frag_id

print(f"Number of atoms: {len(u.atoms)}")
print(f"Number of fragments/molecules: {len(u.atoms.fragments)}")
print(f"Fragment indices shape: {fragment_indices.shape}")

# Extract trajectory coordinates
n_frames = len(u.trajectory)
n_atoms = len(u.atoms)

# Use MDAnalysis native dtype (float32) - no conversion needed!
trajectory = np.zeros((n_frames, n_atoms, 3), dtype=u.atoms.positions.dtype)
box_dimensions = np.zeros((n_frames, 3), dtype=u.atoms.positions.dtype)

for i, ts in enumerate(u.trajectory):
    trajectory[i] = u.atoms.positions
    # Get box dimensions [a, b, c] for orthogonal boxes
    box_dimensions[i] = ts.dimensions[:3]

print(f"Trajectory shape: {trajectory.shape}")
print(f"Box dimensions shape: {box_dimensions.shape}")

# Unwrap the trajectory
print("\nUnwrapping trajectory...")
unwrapped_trajectory = unwrap_system(trajectory, box_dimensions, fragment_indices)

print(f"Unwrapped trajectory shape: {unwrapped_trajectory.shape}")

# Write unwrapped trajectory
with mda.Writer("unwrapped_trajectory.dcd", n_atoms) as W:
    for i in range(n_frames):
        u.trajectory[i]  # Move to frame i
        u.atoms.positions = unwrapped_trajectory[i]  # Update positions
        W.write(u.atoms)

print(f"Wrote unwrapped trajectory to: unwrapped_trajectory.dcd")

# Verify unwrapping worked by checking a molecule's size doesn't change dramatically
# Pick first molecule
first_mol_indices = u.atoms.fragments[0].indices
print(f"\nChecking first molecule (fragment 0) with {len(first_mol_indices)} atoms:")

for frame_idx in [0, n_frames//2, n_frames-1]:
    # Original (wrapped) coordinates
    wrapped_coords = trajectory[frame_idx][first_mol_indices]
    wrapped_size = np.max(wrapped_coords, axis=0) - np.min(wrapped_coords, axis=0)
    
    # Unwrapped coordinates
    unwrapped_coords = unwrapped_trajectory[frame_idx][first_mol_indices]
    unwrapped_size = np.max(unwrapped_coords, axis=0) - np.min(unwrapped_coords, axis=0)
    
    print(f"Frame {frame_idx}:")
    print(f"  Wrapped size: {wrapped_size}")
    print(f"  Unwrapped size: {unwrapped_size}")
    print(f"  Improvement: {wrapped_size.max() / unwrapped_size.max():.2f}x smaller")
