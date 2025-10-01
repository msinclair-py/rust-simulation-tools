import MDAnalysis as mda
import numpy as np
from rust_simulation_tools import kabsch_align

# Load trajectory
u = mda.Universe('../data/topology.pdb', '../data/trajectory.dcd')

# Define selection for alignment (e.g., backbone atoms)
align_selection = u.select_atoms("backbone")
align_indices = align_selection.indices

# Get reference structure (first frame or specific reference)
reference = u.atoms.positions.copy().astype(np.float64)
# Or use a specific reference structure
# ref_u = mda.Universe("reference.pdb")
# reference = ref_u.atoms.positions.astype(np.float64)

# Extract trajectory coordinates
n_frames = len(u.trajectory)
n_atoms = len(u.atoms)
trajectory = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)

for i, ts in enumerate(u.trajectory):
    trajectory[i] = u.atoms.positions.astype(np.float64)

# Convert align_indices to 1D array as expected by Rust function
align_indices_1d = align_indices.astype(np.uintp)

# Perform alignment using Rust function
aligned_trajectory = kabsch_align(trajectory, reference, align_indices_1d)

# Write aligned trajectory back to MDAnalysis universe
with mda.Writer('../data/aligned.dcd', n_atoms) as W:
    for i in range(n_frames):
        u.trajectory[i]
        u.atoms.positions = aligned_trajectory[i]
        W.write(u.atoms)

print(f"Aligned {n_frames} frames using {len(align_indices)} atoms for alignment")
print(f"Shape of aligned trajectory: {aligned_trajectory.shape}")

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
