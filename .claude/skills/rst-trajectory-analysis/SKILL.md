---
name: rst-trajectory-analysis
description: Analyze MD trajectories and topologies with rust_simulation_tools — VMD-style atom selections with property access, SASA (per-atom/per-residue/trajectory), Kabsch alignment, PBC unwrapping, and per-residue LJ/electrostatic interaction fingerprints. Use for "select atoms", "atom selection", "SASA", "solvent accessible surface", "align trajectory", "RMSD fit", "unwrap", "PBC", or "interaction fingerprint".
---

# Trajectory & topology analysis

## VMD-style atom selections

`topo.select(expression, coordinates=None)` returns a `Selection` object with
direct property access (no manual array slicing).

```python
protein = topo.select("protein")
ca      = topo.select("protein and name CA")
ca.n_atoms, ca.n_residues
ca.indices          # numpy array of atom indices
ca.masses, ca.charges, ca.radii   # per-atom property arrays
ca.atom_names, ca.residue_names   # lists
ca.total_mass(), ca.total_charge()

# Distance selections require coordinates:
near = topo.select("protein and within 5.0 of resname LIG", coordinates=coords)
near.positions          # coords of selected atoms (needs coordinates=)
near.has_positions
near.unique_residue_names

# Set operations:
charged   = topo.select("charge < -0.5 or charge > 0.5")
sidechain = topo.select("sidechain")
both = sidechain & charged   # & intersection, | union, - difference
```

Supported keywords: `protein`, `backbone`, `sidechain`, `water`; `name CA`,
`resname ALA`, `resid 1-50`; numeric `charge`/`mass` comparisons;
`within R of <sel>`; booleans `and`/`or`/`not`. Full example:
`examples/example_selections_and_properties.py`.

## SASA (solvent accessible surface area)

```python
sasa = rst.compute_sasa_from_topology(topo, coords)   # topology supplies radii
sasa["total"]        # float, A^2
sasa["per_atom"]     # ndarray (n_atoms,)
sasa["per_residue"]  # ndarray (n_residues,), ordered by residue index

traj_sasa = rst.compute_sasa_trajectory_from_topology(topo, trajectory)
# traj_sasa["total"] -> ndarray (n_frames,); np.mean / np.std over it.
# traj_sasa["per_residue"] -> list of per-frame dicts (residue_idx -> area).
# Lower level: rst.calculate_sasa(coords, radii, residue_indices)
```

## Alignment (Kabsch / RMSD fit)

```python
backbone = topo.select("backbone")
aligned  = rst.kabsch_align(trajectory, trajectory[0], backbone.indices)
# Minimizes RMSD of each frame to the reference using only align_indices atoms.
```

## PBC unwrapping

```python
unwrapped, boxes = rst.unwrap_dcd("trajectory.dcd")         # from file
unwrapped        = rst.unwrap_system(trajectory, boxes)     # from arrays
```

## Interaction fingerprints (per-residue LJ + electrostatics)

`FingerprintSession` streams per-residue LJ and electrostatic interaction
energies (kJ/mol) between a target set and a binder set across a trajectory.

```python
from rust_simulation_tools import FingerprintSession, FingerprintMode

session = FingerprintSession("system.prmtop", "trajectory.dcd")
session.n_residues, session.n_frames
session.set_target_residues(range(10))
session.set_binder_residues(range(10, session.n_residues))

session.return_residue_names = True   # yields names as a 3rd value
for lj_fp, es_fp, names in session:   # without names flag: (lj_fp, es_fp)
    ...                               # lj_fp, es_fp are per-residue arrays

# Switch which side is fingerprinted, then rewind:
session.set_fingerprint_mode(FingerprintMode.Binder)
session.seek(0)
```

Stack frames with `np.array(list_of_fp)` → shape `(frames, residues)`. Full
example: `examples/example_fingerprint.py` and
`examples/example_trajectory_analysis.py`.
