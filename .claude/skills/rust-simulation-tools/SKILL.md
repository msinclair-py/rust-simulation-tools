---
name: rust-simulation-tools
description: Overview and entry point for the rust_simulation_tools Python package (high-performance MD analysis written in Rust/PyO3). Use when the user wants to install/build the package, load AMBER topologies/coordinates/trajectories, or needs to know which capability-specific skill (system building, ligand parameterization, trajectory analysis, MM-PBSA, minimization, ipSAE) to reach for.
---

# rust_simulation_tools

`rust_simulation_tools` (import alias `rst`) is a high-performance molecular
dynamics toolkit written in Rust and exposed to Python via PyO3. It parses AMBER
files, runs VMD-style selections, computes SASA/alignment, builds simulation
systems, parameterizes ligands, minimizes energy, and computes MM-PBSA/GBSA
binding free energies — all without OpenMM, AmberTools, or MDAnalysis at runtime.

## Installation

From PyPI:
```bash
uv pip install rust-simulation-tools
```

From source (this repo) — requires `maturin` and Python dev headers:
```bash
uv venv .venv && source .venv/bin/activate
uv pip install maturin numpy pytest
maturin develop --release          # builds crates/bindings and installs into the venv
```
After editing any Rust source, re-run `maturin develop --release` to rebuild.
Run the test suite with `pytest tests/ -v`.

## Importing

```python
import rust_simulation_tools as rst
# or pull specific names
from rust_simulation_tools import read_prmtop, read_inpcrd, DcdReader
```

## Loading a system (the common starting point)

```python
topo = rst.read_prmtop("system.prmtop")     # -> AmberTopology
coords, box = rst.read_inpcrd("system.inpcrd")  # coords: (n_atoms, 3) ndarray
print(topo.n_atoms, topo.n_residues)

dcd = rst.DcdReader("trajectory.dcd")
trajectory, boxes = dcd.read_all()           # read everything into memory
trajectory = trajectory.reshape(dcd.n_frames, dcd.n_atoms, 3)
# AMBER ASCII trajectories: rst.MdcrdReader(path, n_atoms, has_box)
```

`AmberTopology` exposes `.n_atoms`, `.n_residues`, `.atom_names`,
`.residue_labels`, `.charges()`, `.sigmas()`, `.epsilons()`, `.bonds()`,
`.select(expr, coordinates=None)`, and `.build_selection(residue_indices)`.

## Which skill to use

| Task | Skill |
|------|-------|
| Build a solvated/implicit system, add ions, write prmtop/inpcrd | `rst-build-system` |
| Assign GAFF2 types + AM1-BCC/Gasteiger charges; raw AM1 charges | `rst-parameterize-ligand` |
| Atom selections, SASA, alignment, unwrapping, interaction fingerprints | `rst-trajectory-analysis` |
| MM-PBSA / MM-GBSA binding free energy, per-residue decomposition, entropy | `rst-mmpbsa` |
| Energy minimization of a prmtop/inpcrd | `rst-minimize` |
| ipSAE / pDockQ / LIS interface confidence scoring | `rst-ipsae` |

Runnable example scripts for every workflow live in `examples/` at the repo root.
