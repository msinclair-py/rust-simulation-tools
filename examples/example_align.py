import MDAnalysis as mda
import numpy as np
from rust_simulation_tools import kabsch_align

# Load trajectory
u = mda.Universe("topology.pdb", "trajectory.dcd")

# Define selection for alignment (e.g., backbone atoms)
align_selection = u.select_atoms("backbone")
align_indices = align_selection.indices

print(f"Number of atoms: {len(u.atoms)}")
print(f"Number of alignment atoms: {len(align_indices)}")
print(f"Alignment indices (first 10): {align_indices[:10]}")

# Get reference structure (first frame or specific reference)
# No need to convert to float64! Works with MDAnalysis native float32
reference = u.atoms.positions.copy()
# Or use a specific reference structure
# ref_u = mda.Universe("reference.pdb")
# reference = ref_u.atoms.positions

# Extract trajectory coordinates
n_frames = len(u.trajectory)
n_atoms = len(u.atoms)

# Use MDAnalysis native dtype (float32) - no conversion needed!
trajectory = np.zeros((n_frames, n_atoms, 3), dtype=u.atoms.positions.dtype)

for i, ts in enumerate(u.trajectory):
    trajectory[i] = u.atoms.positions

# Convert align_indices to 1D array as expected by Rust function
align_indices_1d = align_indices.astype(np.uintp)

# Perform alignment using Rust function
aligned_trajectory = kabsch_align(trajectory, reference, align_indices_1d)

# Write aligned trajectory back to MDAnalysis universe
# You need to write to a NEW file, not modify the existing trajectory
with mda.Writer("aligned_trajectory.dcd", n_atoms) as W:
    for i in range(n_frames):
        u.trajectory[i]  # Move to frame i
        u.atoms.positions = aligned_trajectory[i]  # Update positions
        W.write(u.atoms)  # Write this frame

print(f"Aligned {n_frames} frames using {len(align_indices)} atoms for alignment")
print(f"Shape of aligned trajectory: {aligned_trajectory.shape}")
print(f"Wrote aligned trajectory to: aligned_trajectory.dcd")

# Example: Compare RMSD before and after alignment
# For a proper test, use a different frame than the reference
if n_frames > 1:
    test_frame_idx = 1  # Use second frame for testing
    
    # RMSD of original trajectory to reference (alignment atoms only)
    original_align_coords = trajectory[test_frame_idx][align_indices]
    reference_align_coords = reference[align_indices]
    
    # Compute RMSD manually
    diff = original_align_coords - reference_align_coords
    original_rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    print(f"\nOriginal RMSD (frame {test_frame_idx}, alignment atoms): {original_rmsd:.3f} Å")
    
    # RMSD of aligned trajectory to reference (alignment atoms only)
    aligned_align_coords = aligned_trajectory[test_frame_idx][align_indices]
    diff_aligned = aligned_align_coords - reference_align_coords
    aligned_rmsd = np.sqrt(np.mean(np.sum(diff_aligned**2, axis=1)))
    print(f"Aligned RMSD (frame {test_frame_idx}, alignment atoms): {aligned_rmsd:.3f} Å")
    
    # All-atom RMSD (should be larger if alignment selection was subset)
    diff_all = trajectory[test_frame_idx] - reference
    original_rmsd_all = np.sqrt(np.mean(np.sum(diff_all**2, axis=1)))
    print(f"\nOriginal RMSD (frame {test_frame_idx}, all atoms): {original_rmsd_all:.3f} Å")
    
    diff_all_aligned = aligned_trajectory[test_frame_idx] - reference
    aligned_rmsd_all = np.sqrt(np.mean(np.sum(diff_all_aligned**2, axis=1)))
    print(f"Aligned RMSD (frame {test_frame_idx}, all atoms): {aligned_rmsd_all:.3f} Å")
else:
    print("\nNeed at least 2 frames to test alignment properly")
