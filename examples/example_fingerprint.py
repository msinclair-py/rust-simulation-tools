"""Interaction fingerprint example - compute per-residue LJ and electrostatic energies."""

import numpy as np
from rust_simulation_tools import compute_fingerprints, read_prmtop, DcdReader

# Load AMBER topology for force field parameters
topo = read_prmtop("system.prmtop")

# Get charges and LJ parameters
charges = np.array(topo.charges())
sigmas = np.array(topo.sigmas())
epsilons = np.array(topo.epsilons())

# Build residue map (atom indices grouped by residue)
resmap_indices, resmap_offsets = topo.build_resmap()
resmap_indices = np.array(resmap_indices, dtype=np.int64)
resmap_offsets = np.array(resmap_offsets, dtype=np.int64)

# Define target residues (e.g., first 10 residues of protein)
n_target = 10
target_resmap_indices = resmap_indices[:resmap_offsets[n_target]]
target_resmap_offsets = resmap_offsets[:n_target + 1]

# Define binder atoms (all atoms after target residues)
binder_indices = np.arange(resmap_offsets[n_target], topo.n_atoms, dtype=np.int64)

# Load trajectory
reader = DcdReader("trajectory.dcd")
print(f"Computing fingerprints for {n_target} residues over {reader.n_frames} frames")

# Compute fingerprints for each frame
lj_fingerprints = []
es_fingerprints = []

while True:
    frame = reader.read_frame()
    if frame is None:
        break
    positions, _ = frame

    lj_fp, es_fp = compute_fingerprints(
        positions, charges, sigmas, epsilons,
        target_resmap_indices, target_resmap_offsets, binder_indices
    )
    lj_fingerprints.append(lj_fp)
    es_fingerprints.append(es_fp)

lj_fingerprints = np.array(lj_fingerprints)
es_fingerprints = np.array(es_fingerprints)

# Summarize results
print(f"\nPer-residue mean LJ energies (kJ/mol):")
for i in range(n_target):
    resname = topo.residue_labels[i]
    print(f"  {resname}: {np.mean(lj_fingerprints[:, i]):.2f}")

print(f"\nPer-residue mean ES energies (kJ/mol):")
for i in range(n_target):
    resname = topo.residue_labels[i]
    print(f"  {resname}: {np.mean(es_fingerprints[:, i]):.2f}")
