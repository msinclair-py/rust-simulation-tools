"""Interaction fingerprints: per-residue LJ and electrostatic energies.

Computes pairwise interaction energies between target and binder residues
across a trajectory using the FingerprintSession API.
"""

import numpy as np
from rust_simulation_tools import FingerprintSession, FingerprintMode

# -----------------------------------------------------------------------------
# Setup session
# -----------------------------------------------------------------------------

session = FingerprintSession("system.prmtop", "trajectory.dcd")

# Define target (e.g., ligand) and binder (e.g., protein) residues
n_target = 10
session.set_target_residues(range(n_target))
session.set_binder_residues(range(n_target, session.n_residues))

print(f"System: {session.n_residues} residues, {session.n_frames} frames")
print(f"Target: residues 0-{n_target - 1}")
print(f"Binder: residues {n_target}-{session.n_residues - 1}")

# -----------------------------------------------------------------------------
# Compute target fingerprints
# -----------------------------------------------------------------------------

session.return_residue_names = True

lj_fingerprints = []
es_fingerprints = []
residue_names = None

for lj_fp, es_fp, names in session:
    lj_fingerprints.append(lj_fp)
    es_fingerprints.append(es_fp)
    if residue_names is None:
        residue_names = list(names)

lj_fingerprints = np.array(lj_fingerprints)
es_fingerprints = np.array(es_fingerprints)

print(f"\nFingerprint shape: {lj_fingerprints.shape} (frames x residues)")

# Per-residue mean energies
print(f"\nPer-residue mean energies (kJ/mol):")
print(f"  {'Residue':<8} {'LJ':>10} {'Elec':>10}")
for i, name in enumerate(residue_names):
    lj_mean = np.mean(lj_fingerprints[:, i])
    es_mean = np.mean(es_fingerprints[:, i])
    print(f"  {name:<8} {lj_mean:>10.2f} {es_mean:>10.2f}")

# Total interaction energy (last frame)
total_lj = np.sum(lj_fingerprints[-1])
total_es = np.sum(es_fingerprints[-1])
print(f"\nTotal interaction (last frame): LJ={total_lj:.2f}, ES={total_es:.2f} kJ/mol")

# -----------------------------------------------------------------------------
# Switch to binder fingerprints
# -----------------------------------------------------------------------------

session.set_fingerprint_mode(FingerprintMode.Binder)
session.seek(0)

binder_lj = []
binder_es = []

for lj_fp, es_fp, _ in session:
    binder_lj.append(lj_fp)
    binder_es.append(es_fp)

binder_lj = np.array(binder_lj)
binder_es = np.array(binder_es)

print(f"\nBinder fingerprints: {binder_lj.shape[1]} residues")

# Energy conservation check: sum(target) should equal sum(binder)
target_total = total_lj + total_es
binder_total = np.sum(binder_lj[-1]) + np.sum(binder_es[-1])
print(f"Energy symmetry check:")
print(f"  Target total: {target_total:.2f} kJ/mol")
print(f"  Binder total: {binder_total:.2f} kJ/mol")
print(f"  Difference: {abs(target_total - binder_total):.6f} kJ/mol")
