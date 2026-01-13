"""SASA calculation example - compute solvent accessible surface area."""

import MDAnalysis as mda
import numpy as np
from rust_simulation_tools import (
    calculate_sasa,
    calculate_sasa_trajectory,
    get_radii_array,
)

# Load structure
u = mda.Universe("protein.pdb")
protein = u.select_atoms("protein")

# Get atomic radii from element names (uses built-in VDW radii)
elements = [atom.element for atom in protein]
radii = get_radii_array(elements)

# Get residue indices for per-residue SASA
residue_indices = np.array([atom.resindex for atom in protein], dtype=np.int64)

# Single frame SASA calculation
coords = protein.positions.astype(np.float64)
result = calculate_sasa(coords, radii, residue_indices, probe_radius=1.4)

print(f"Total SASA: {result['total']:.2f} Å²")
print(f"Per-atom SASA shape: {result['per_atom'].shape}")
print(f"Number of residues: {len(result['per_residue'])}")

# Show top 5 most exposed residues
per_res = result['per_residue']
sorted_res = sorted(per_res.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nMost exposed residues:")
for res_idx, sasa in sorted_res:
    res = u.residues[res_idx]
    print(f"  {res.resname}{res.resid}: {sasa:.1f} Å²")

# Trajectory SASA (if multiple frames available)
if len(u.trajectory) > 1:
    trajectory = np.array([protein.positions.copy() for _ in u.trajectory], dtype=np.float64)
    traj_result = calculate_sasa_trajectory(trajectory, radii, residue_indices)

    total_sasa = traj_result['total']
    print(f"\nTrajectory SASA: mean={np.mean(total_sasa):.1f}, std={np.std(total_sasa):.1f} Å²")
